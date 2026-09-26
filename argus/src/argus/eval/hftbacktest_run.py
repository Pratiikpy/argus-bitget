"""Run hftbacktest itself — the rival, unmodified — on the same real events ARGUS replays.

`eval/queueproof.py::reproduce` shows the ported probability *formulas* agree with the Rust to
1e-16.
That is a statement about five functions, not about an engine: how a trade is apportioned, when
``cum_trade_qty`` resets, when an order counts as filled, what the L3 truth does on a partial fill.
This module answers the engine question by running hftbacktest 2.4.4 from PyPI (the published
wheel, not a transcription) on hftbacktest's own conversion of the real CME file, and comparing
its fills order for order with ARGUS's:

* **L3 truth.** hftbacktest's ``L3FIFOQueueModel`` + ``NoPartialFillExchange`` against
  :class:`argus.eval.l3queue.TruthReplay` — both fed the identical event array.
* **L2 models.** hftbacktest's ``ProbQueueModel`` (each shipped function) on the L2 array its own
  Level-3 notebook derives (``convert_l3_to_l2``, reproduced below verbatim from
  ``examples/Level-3 Backtesting.ipynb`` cell 7, MIT) against ARGUS's port driven through the same
  view by :func:`argus.eval.realqueue.drive`.

hftbacktest is **not** an ARGUS dependency; it is imported from ``HFTBACKTEST_SITE`` (a directory
``pip install --no-deps --target`` put it in) so the evaluation never changes the environment the
paper desk runs in. The committed run (2026-09-26) used its own environment on hftbacktest's
pins where they resolve (numpy 2.2.6; numba 0.67.0, since 0.61 has no wheel for this Python) and
records the versions it actually used.

**What the run found** (``data/hftbacktest_run.json``). On the ESH4 file hftbacktest's L3 FIFO
engine and ARGUS's truth replay agree on every one of 1,438 orders, with the same fill timestamp on
all 1,130 fills. The L2 comparison first showed ARGUS's port of hftbacktest's queue models 8 to
29 points behind the engine; the cause was ARGUS's replay closing an order's L2 feed the moment
the truth filled it, so a model that fills a few events later, as the engine does, was scored as
a miss. With the feed kept open to the horizon (`eval/l3queue.py`) the port matches the engine
exactly on four models and within one point on the other four.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from argus.eval.hftbacktest_import import RivalUnavailable, import_hftbacktest
from argus.eval.hftbacktest_numba import build_l3_to_l2, build_schedule_runner
from argus.eval.l3queue import (
    ADD,
    CANCEL,
    CLEAR,
    FILL,
    MODIFY,
    TRADE,
    L3Event,
    RealEpisode,
    ReplayConfig,
    TruthReplay,
)

EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
_HBT_KIND = {10: ADD, 11: CANCEL, 12: MODIFY, 13: FILL, 2: TRADE, 3: CLEAR}


def events_from_hbt_array(data: np.ndarray, tick: float) -> list[L3Event]:
    """hftbacktest's own L3 event array as ARGUS events — literally the same input for both.

    Exchange-side events only, in array (exchange-timestamp) order. Prices become integer ticks.
    An exchange event ends where the next record's exchange timestamp differs: every record of
    one CME match shares its ``ts_event``.
    """
    out: list[L3Event] = []
    ev = data["ev"].astype(np.int64)
    exch = data["exch_ts"]
    n = len(data)
    for i in range(n):
        flags = int(ev[i])
        if not flags & EXCH_EVENT:
            continue
        kind = _HBT_KIND.get(flags & 0xFF)
        if kind is None:
            continue
        side = "B" if flags & BUY_EVENT else ("A" if flags & SELL_EVENT else "N")
        nxt = i + 1
        while nxt < n and not int(ev[nxt]) & EXCH_EVENT:
            nxt += 1
        batch_end = nxt >= n or int(exch[nxt]) != int(exch[i])
        out.append(L3Event(
            ts=int(exch[i]), kind=kind, side=side, px=round(float(data["px"][i]) / tick),
            size=float(data["qty"][i]), order_id=int(data["order_id"][i]), batch_end=batch_end,
        ))
    return out


@dataclass(frozen=True, slots=True)
class EngineFill:
    status: int
    exch_ts: int


def run_engine(
    data_path: Path, queue: tuple[str, float | None], *, tick: float, lot: float,
    times: np.ndarray, sides: np.ndarray, prices: np.ndarray, horizon: int,
) -> list[EngineFill]:  # pragma: no cover - requires hftbacktest
    """One hftbacktest run: the published engine, a named queue model, the given schedule."""
    hbt = import_hftbacktest()
    asset = (
        hbt.BacktestAsset()
        .data([str(data_path)])
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick)
        .lot_size(lot)
    )
    name, n = queue
    configured = getattr(asset, name)(n) if n is not None else getattr(asset, name)()
    backtest = hbt.HashMapMarketDepthBacktest([configured])
    runner = build_schedule_runner()
    out = runner(backtest, times, sides, prices, horizon)
    backtest.close()
    return [EngineFill(status=int(r[0]), exch_ts=int(r[1])) for r in out]


RIVAL_ENGINE_MODELS: tuple[tuple[str, str, float | None], ...] = (
    ("hftbacktest default: LogProbQueueFunc2", "log_prob_queue_model2", None),
    ("hftbacktest PowerProbQueueFunc n=1", "power_prob_queue_model", 1.0),
    ("hftbacktest PowerProbQueueFunc n=2", "power_prob_queue_model", 2.0),
    ("hftbacktest PowerProbQueueFunc n=3", "power_prob_queue_model", 3.0),
    ("hftbacktest PowerProbQueueFunc2 n=2", "power_prob_queue_model2", 2.0),
    ("hftbacktest PowerProbQueueFunc3 n=3", "power_prob_queue_model3", 3.0),
    ("hftbacktest LogProbQueueFunc", "log_prob_queue_model", None),
    ("hftbacktest RiskAdverseQueueModel", "risk_adverse_queue_model", None),
)


def schedule_from(episodes: Sequence[RealEpisode], tick_px: float) -> tuple[np.ndarray, ...]:
    """The exact orders ARGUS's replay placed, as the arrays the numba strategy consumes. Each is
    submitted 1ns after the event it joined behind, so every record of that event is already in
    both engines' books."""
    eps = sorted(episodes, key=lambda e: (e.t_join, e.side))
    times = np.array([e.t_join + 1 for e in eps], dtype=np.int64)
    sides = np.array([1 if e.side == "B" else -1 for e in eps], dtype=np.int64)
    prices = np.array([e.px * tick_px for e in eps], dtype=np.float64)
    return times, sides, prices


def argus_truth(
    events: Sequence[L3Event], dataset: str, config: ReplayConfig,
) -> list[RealEpisode]:
    replay = TruthReplay(dataset, config)
    return replay.run(events)


# --- the run: both engines on the same file, the same orders ------------------------------------

ARTEFACT = Path(__file__).resolve().parents[3] / "data" / "hftbacktest_run.json"
ES_TICK = 0.25
ES_LOT = 1.0


def _truth_filled(ep: RealEpisode) -> bool:
    return ep.end == "filled" and ep.fill_ts >= 0


def compare_l3(episodes: Sequence[RealEpisode], engine: Sequence[EngineFill],
               filled_status: int) -> dict[str, Any]:
    """hftbacktest's L3 FIFO engine against ARGUS's truth replay, order for order."""
    both = argus_only = engine_only = neither = same_ts = 0
    for ep, fill in zip(episodes, engine, strict=True):
        t, e = _truth_filled(ep), fill.status == filled_status
        both += int(t and e)
        argus_only += int(t and not e)
        engine_only += int(e and not t)
        neither += int(not t and not e)
        same_ts += int(t and e and fill.exch_ts == ep.fill_ts)
    n = len(episodes)
    return {
        "orders": n, "both_filled": both, "argus_only": argus_only,
        "engine_only": engine_only, "neither": neither,
        "fill_agreement": round((both + neither) / n, 5) if n else None,
        "same_fill_timestamp_of_both_filled": same_ts,
    }


def compare_l2(episodes: Sequence[RealEpisode], engine: Sequence[EngineFill],
               filled_status: int) -> dict[str, Any]:
    """One hftbacktest L2 model in its own engine, scored against the L3 truth."""
    false = missed = agree = 0
    for ep, fill in zip(episodes, engine, strict=True):
        t, e = _truth_filled(ep), fill.status == filled_status
        false += int(e and not t)
        missed += int(t and not e)
        agree += int(e == t)
    n = max(len(episodes), 1)
    return {"orders": len(episodes), "fill_agreement": round(agree / n, 5),
            "false_fill_rate": round(false / n, 5), "missed_fill_rate": round(missed / n, 5)}


def main(hbt_npz: Path, *, out: Path = ARTEFACT) -> dict[str, Any]:  # pragma: no cover
    """Convert once with hftbacktest's own tools, replay ARGUS's truth on the identical events,
    then run hftbacktest's engine with every queue model on the orders the truth placed."""
    import json
    import platform
    import tempfile
    from decimal import Decimal

    import numba

    from argus.eval.realqueue import rival_models, score_models

    hbt = import_hftbacktest()
    data = np.load(hbt_npz)["data"]
    events = events_from_hbt_array(data, ES_TICK)
    first_trade = next(e.ts for e in events if e.kind == TRADE)
    config = ReplayConfig(start_ns=first_trade)
    episodes = sorted(argus_truth(events, "ESH4 2023-12-25", config),
                      key=lambda e: (e.t_join, e.side))
    times, sides, prices = schedule_from(episodes, ES_TICK)
    filled = int(hbt.FILLED)

    l3 = run_engine(hbt_npz, ("l3_fifo_queue_model", None), tick=ES_TICK, lot=ES_LOT,
                    times=times, sides=sides, prices=prices, horizon=config.horizon_ns)
    with tempfile.TemporaryDirectory() as tmp:
        l2_path = Path(tmp) / "esh4_l2.npz"
        # np.load returns the records unaligned; hftbacktest's event_dtype is aligned, and numba
        # 0.67 will not cast one record type to the other inside the converter.
        aligned = data.astype(hbt.event_dtype)
        np.savez_compressed(l2_path, data=build_l3_to_l2()(aligned, ES_TICK))
        engine_l2 = {
            label: compare_l2(episodes, run_engine(
                l2_path, (method, n), tick=ES_TICK, lot=ES_LOT, times=times, sides=sides,
                prices=prices, horizon=config.horizon_ns), filled)
            for label, method, n in RIVAL_ENGINE_MODELS
        }
    ported = {r.model: r.as_dict() for r in score_models(
        episodes, rival_models(), lot=Decimal(repr(ES_LOT)), view="record")}
    report = {
        "input": {
            "file": "esh4-glbx-mdp3-20231225.mbo.dbn.zst (nautilus_trader test data)",
            "converted_by": "hftbacktest.data.utils.databento.convert(symbol='ESH4')",
            "records": len(data), "events": len(events), "orders": len(episodes),
            "replay": {"every_ns": config.every_ns, "horizon_ns": config.horizon_ns,
                       "start_ns": config.start_ns},
        },
        "versions": {"hftbacktest": getattr(hbt, "__version__", "2.4.4"),
                     "numpy": np.__version__, "numba": numba.__version__,
                     "python": platform.python_version()},
        "l3_truth_vs_hftbacktest_l3_fifo": compare_l3(episodes, l3, filled),
        "l2_models": {
            label: {"hftbacktest_engine": engine_l2[label], "argus_port": ported.get(label)}
            for label, _m, _n in RIVAL_ENGINE_MODELS
        },
    }
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover - CLI
    import json as _json

    print(_json.dumps(main(Path(sys.argv[1])), indent=1))


__all__ = [
    "RIVAL_ENGINE_MODELS",
    "EngineFill",
    "RivalUnavailable",
    "argus_truth",
    "build_l3_to_l2",
    "build_schedule_runner",
    "compare_l2",
    "compare_l3",
    "events_from_hbt_array",
    "import_hftbacktest",
    "main",
    "run_engine",
    "schedule_from",
]
