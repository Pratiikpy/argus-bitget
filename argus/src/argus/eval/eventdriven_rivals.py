"""Event significance when events cluster: ARGUS against the event-study specialists, same events.

`eval/eventdriven_comparison.py` measured ARGUS's `research/eventstudy.py` against one rival,
whale-signals' fixed-50% binomial test. That rival tests one thing badly, and beating it says
little about the field. This module runs the strongest implementations of the same job that could
be found, unmodified, on the same real events:

* **HKUDS/Vibe-Trading `quantlib/eventstudy.py`** (MIT, vendored byte-exact and hash-pinned in
  `eval/baselines/vibe_trading_eventstudy.py`). A MacKinlay market-model study with a plain t,
  Patell, Boehmer-Musumeci-Poulsen, Corrado rank and Cowan generalised-sign tests, and a CAR
  standard error that carries the covariance of the parameter-estimation error across window days
  (its issue #1466). It is the closest specialist to ARGUS's module: same five-ish tests, cleaner
  CAR variance. Its docstring names the one thing it does not do — "None of the three fixes
  cross-sectional correlation from events that share a date" — and reports
  ``shared_event_dates`` instead of correcting for them.
* **Ritapossible/Ballast `stats.t_stat`** (MIT, S2 entry, vendored in `ballast_stats.py`): the
  pooled one-sample t behind Ballast's published Gate 1 / 1b event verdicts. Their own research
  text concedes that stacking names that move together inflates it; it is not corrected.
* **whale-signals `compute_hit_rates`** (MIT, already vendored): the fixed-null binomial test.

**The inputs are real.** Hourly Bitget candles for nine single-name rTokens and QQQUSDT as the
market, from a snapshot fetched once (`data/eventdriven_rivals_hourly.json.gz`) so every number
here re-derives byte for byte. Every method gets the identical market series, estimation window
(480 bars), purge gap (24) and event window (24 bars) — only the statistics differ.

**The experiment is the one Kolari and Pynnonen (2010) ran to justify their correction**: place
placebo events on real history, where the true abnormal return is zero by construction, and count
how often each test rejects at 5%. A well-sized test rejects about 5% of the time. The events are
drawn in several shapes, because the shape is the whole question:

* ``independent`` — 108 events on distinct, spaced hours and random names. No clustering. Every
  test should be near 5% here; this is the control.
* ``clustered_3`` / ``clustered_9`` — the same 108 events on 36 or 12 hours, three or all nine
  names at once. This is a CPI print or an FOMC decision hitting a universe driven by one
  underlying market.
* ``us_open_clustered`` — twelve hours at the US cash open, all nine names: clustering plus the
  event-induced variance the open carries.
* ``crypto_pair`` — COIN and MSTR together on 30 hours; two names sharing a factor QQQ does not
  span.
* ``near_clustered`` — ``clustered_9`` with each name's event jittered 0-2 hours. Vibe-Trading's
  ``shared_event_dates`` flag matches identical timestamps only; this measures what a flag keyed
  on equality misses, and whether ARGUS's position-aligned correlation estimate misses it too.
* ``independent_short_estimation`` — ``independent`` on a 120-bar estimation window, ARGUS's own
  floor, where the covariance term Vibe-Trading carries in its CAR variance and ARGUS's summed
  standardised residuals do not is largest.

Power is measured by injecting a known abnormal return into the first event bar, then read two
ways: raw (at the nominal 5%), and **size-adjusted** (at each method's own empirical 5% critical
value under the matching null) — a test that rejects too often under the null also "detects"
more, and raw power rewards that.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from typing import Any

from argus.eval.artefact import write as write_artefact
from argus.eval.baselines.vibe_trading_eventstudy_loader import (
    BALLAST_COMMIT,
    VIBE_COMMIT,
    load_ballast_stats,
    load_vibe_eventstudy,
    vendored_body_sha256,
)
from argus.eval.baselines.whale_signals_event_study_loader import load_event_study_module
from argus.research.eventstudy import (
    EventStudyError,
    EventWindow,
    build_window,
    kolari_pynnonen,
    returns_from,
    study,
)

DATA = Path(__file__).resolve().parents[3] / "data"
SNAPSHOT_HOURLY = DATA / "eventdriven_rivals_hourly.json.gz"
REPORT_PATH = DATA / "eventdriven_rivals.json"

NAMES: tuple[str, ...] = (
    "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "MSFTUSDT", "METAUSDT", "GOOGLUSDT", "AMZNUSDT",
    "COINUSDT", "MSTRUSDT",
)
"""Nine single names. TQQQ/SQQQ are left out: they are QQQ at a multiple, so their abnormal return
against QQQ is noise around a constant and would pad every sample with non-events."""
MARKET = "QQQUSDT"
EVENT_BARS = 24
ESTIMATION_BARS = 480
GAP_BARS = 24
EVENTS_PER_DRAW = 108
ALPHA = 0.05

ARGUS_TESTS = ("patell", "bmp", "corrado_rank", "generalised_sign")
VIBE_TESTS = ("t", "patell", "bmp", "corrado_rank", "cowan_sign")


class RivalStudyError(RuntimeError):
    """The comparison could not be run as specified; raised rather than reported as a number."""


# --- the shared input ------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """Aligned real returns: ``returns[s][k]`` is realised over the bar stamped ``stamps[k]``."""

    stamps: tuple[datetime, ...]
    returns: dict[str, list[float]]
    market: list[float]
    fetched_at: str
    source_sha256: str

    @property
    def bars(self) -> int:
        return len(self.stamps)


def load_hourly_panel(path: Path = SNAPSHOT_HOURLY) -> Panel:
    """The snapshot, aligned on the timestamps every series shares.

    Inner-joined rather than forward-filled: a filled bar is a zero return nobody traded, and on
    placebo events that is a free non-rejection for every test.
    """
    import hashlib

    raw = path.read_bytes()
    blob = json.loads(gzip.decompress(raw))
    candles: dict[str, list[list[float]]] = blob["candles"]
    wanted = (*NAMES, MARKET)
    missing = [s for s in wanted if s not in candles]
    if missing:
        raise RivalStudyError(f"snapshot {path} lacks {missing}")
    closes = {s: {int(r[0]): float(r[4]) for r in candles[s]} for s in wanted}
    common = sorted(set.intersection(*(set(v) for v in closes.values())))
    if len(common) < ESTIMATION_BARS + GAP_BARS + EVENT_BARS + 200:
        raise RivalStudyError(f"only {len(common)} aligned bars in {path}")
    stamps = tuple(datetime.fromtimestamp(ms / 1000, tz=UTC) for ms in common[1:])
    rets = {s: returns_from([closes[s][ms] for ms in common]) for s in NAMES}
    market = returns_from([closes[MARKET][ms] for ms in common])
    return Panel(stamps=stamps, returns=rets, market=market,
                 fetched_at=str(blob.get("fetched_at", "")),
                 source_sha256=hashlib.sha256(raw).hexdigest())


# --- event draws -----------------------------------------------------------------------------

Event = tuple[str, int]
"""(symbol, index of the first event-window bar in the return series)."""


def eligible(panel: Panel, *, estimation_bars: int = ESTIMATION_BARS) -> tuple[int, int]:
    """First and one-past-last event index with a full estimation window, gap and event window."""
    low = estimation_bars + GAP_BARS + 1
    high = panel.bars - EVENT_BARS
    if high - low < 200:
        raise RivalStudyError("too few eligible bars for any draw")
    return low, high


def _spaced(rng: random.Random, pool: Sequence[int], count: int, spacing: int) -> list[int]:
    chosen: list[int] = []
    order = list(pool)
    rng.shuffle(order)
    for idx in order:
        if all(abs(idx - c) >= spacing for c in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                return sorted(chosen)
    raise RivalStudyError(f"could not place {count} events {spacing} bars apart in this range")


def draw_events(
    shape: str, rng: random.Random, panel: Panel, low: int, high: int,
) -> list[Event]:
    """One placebo draw of the named shape. Pure function of the RNG state and the range."""
    pool = range(low, high)
    if shape in ("independent", "independent_short_estimation"):
        return [(rng.choice(NAMES), i) for i in _spaced(rng, pool, EVENTS_PER_DRAW, EVENT_BARS)]
    if shape in ("clustered_3", "clustered_9", "near_clustered"):
        k = 3 if shape == "clustered_3" else 9
        dates = _spaced(rng, pool, EVENTS_PER_DRAW // k, 2 * EVENT_BARS + 4)
        out: list[Event] = []
        for d in dates:
            for name in rng.sample(NAMES, k):
                jitter = rng.randint(0, 2) if shape == "near_clustered" else 0
                out.append((name, d + jitter))
        return out
    if shape == "us_open_clustered":
        opens = [i for i in pool
                 if panel.stamps[i].weekday() < 5 and panel.stamps[i].hour in (13, 14)]
        return [(name, d) for d in _spaced(rng, opens, 12, 2 * EVENT_BARS)
                for name in NAMES]
    if shape == "crypto_pair":
        return [(name, d) for d in _spaced(rng, pool, 30, 2 * EVENT_BARS)
                for name in ("COINUSDT", "MSTRUSDT")]
    raise RivalStudyError(f"unknown draw shape {shape!r}")


# --- the four scorers, on identical input -----------------------------------------------------


def _injected(panel: Panel, events: Sequence[Event], shock: float) -> dict[str, list[float]]:
    """Returns with ``shock`` added to each event's first window bar. Copies only touched names."""
    if not shock:
        return panel.returns
    out = dict(panel.returns)
    for name in {e[0] for e in events}:
        out[name] = list(panel.returns[name])
    for name, idx in events:
        out[name][idx] += shock
    return out


def argus_windows(
    panel: Panel, events: Sequence[Event], returns: dict[str, list[float]],
    *, estimation_bars: int = ESTIMATION_BARS,
) -> list[EventWindow]:
    windows = []
    for name, idx in events:
        w = build_window(
            name, panel.stamps[idx - 1], panel.stamps, returns[name], panel.market,
            event_bars=EVENT_BARS, estimation_bars=estimation_bars, gap_bars=GAP_BARS,
        )
        if w is None:
            raise RivalStudyError(f"ARGUS could not build a window for {name}@{idx}")
        windows.append(w)
    return windows


def _p(z: float) -> float:
    from statistics import NormalDist

    return 2.0 * (1.0 - NormalDist().cdf(abs(z)))


def score_argus(windows: Sequence[EventWindow]) -> dict[str, float]:
    """ARGUS's p-values: each test adjusted and raw, the verdict statistic, and the ablations.

    ``verdict`` is the largest of the four adjusted p-values — ARGUS says EFFECT ESTABLISHED only
    when all four reject, so it rejects at 5% exactly when this is at or below 0.05.

    ``patell_summed`` is the ablation of the one place ARGUS's SCAR differs from Vibe-Trading's:
    the sum of per-bar standardised residuals over sqrt(L), which ignores the covariance the
    shared parameter estimates induce across window bars. It is computed here from the window's
    own per-bar standardised returns whatever `EventWindow.scar` currently does, so the arm stays
    measurable after the module changes.
    """
    result = study("rivals", windows, event_bars=EVENT_BARS)
    stats = result.statistics
    out: dict[str, float] = {}
    for name in ARGUS_TESTS:
        out[f"{name}_adjusted"] = stats[name]["adjusted_p_value"]
        out[f"{name}_raw"] = stats[name]["p_value"]
    out["verdict"] = max(stats[name]["adjusted_p_value"] for name in ARGUS_TESTS)
    out["verdict_raw"] = max(stats[name]["p_value"] for name in ARGUS_TESTS)
    out["any_adjusted"] = min(stats[name]["adjusted_p_value"] for name in ARGUS_TESTS)
    out["correlation"] = result.correlation
    summed = [sum(w.standardised) / sqrt(len(w.standardised)) for w in windows]
    variance = sum((w.model.observations - 2) / (w.model.observations - 4) for w in windows)
    z_summed = sum(summed) / sqrt(variance)
    out["patell_summed_adjusted"] = _p(kolari_pynnonen(
        z_summed, events=len(windows), correlation=result.correlation))
    out["patell_summed_raw"] = _p(z_summed)
    return out


class VibeFrame:
    """Vibe-Trading's `event_study` wants a DataFrame and a Series; built once per panel."""

    def __init__(self, panel: Panel) -> None:
        import pandas as pd

        index = pd.DatetimeIndex(list(panel.stamps))
        self.index = index
        self.frame = pd.DataFrame({s: panel.returns[s] for s in NAMES}, index=index)
        self.market = pd.Series(panel.market, index=index)

    def with_returns(self, returns: dict[str, list[float]], touched: set[str]) -> Any:
        if not touched:
            return self.frame
        frame = self.frame.copy()
        for name in touched:
            frame[name] = returns[name]
        return frame


def score_vibe(
    vibe: Any, frame: VibeFrame, panel: Panel, events: Sequence[Event],
    returns_frame: Any, *, estimation_bars: int = ESTIMATION_BARS,
) -> dict[str, float]:
    """Vibe-Trading's real `event_study`, called exactly as its own tests call it."""
    result = vibe.event_study(
        returns_frame, frame.market,
        [(name, frame.index[idx]) for name, idx in events],
        event_window=(0, EVENT_BARS - 1),
        estimation_window=estimation_bars,
        estimation_gap=GAP_BARS,
        model="market",
    )
    if result.dropped:
        raise RivalStudyError(f"Vibe-Trading dropped {len(result.dropped)} event(s): "
                              f"{result.dropped[:3]}")
    return {
        "t": float(result.t_p_value),
        "patell": float(result.patell_p_value),
        "bmp": float(result.bmp_p_value),
        "corrado_rank": float(result.corrado_rank_p_value),
        "cowan_sign": float(result.cowan_sign_p_value),
        "flagged_shared_dates": 1.0 if result.shared_event_dates else 0.0,
    }


def raw_window_returns(returns: dict[str, list[float]], events: Sequence[Event],
                       bars: int) -> list[float]:
    out = []
    for name, idx in events:
        level = 1.0
        for r in returns[name][idx: idx + bars]:
            level *= 1.0 + r
        out.append(level - 1.0)
    return out


def score_ballast(ballast: Any, raw: Sequence[float]) -> dict[str, float]:
    """Ballast's real pooled ``t_stat`` on the raw event-window returns, as its Gate 1 does."""
    _mean, t = ballast.t_stat(list(raw))
    return {"pooled_t": _p(t) if t == t else 1.0, "t": float(t)}


def score_whale(whale: Any, returns: dict[str, list[float]],
                events: Sequence[Event]) -> dict[str, float]:
    """whale-signals' real ``compute_hit_rates`` with the event window as its 24h horizon."""
    import pandas as pd

    rows = []
    for name, idx in events:
        fwd = {h: raw_window_returns(returns, [(name, idx)], h)[0] for h in (1, 6, 24)}
        rows.append({"tx_category": "exchange_withdrawal", "fwd_return_1h": fwd[1],
                     "fwd_return_6h": fwd[6], "fwd_return_24h": fwd[24]})
    # Their function prints "Skipping <category>: only 0 events" for the two categories this
    # sample does not use; the print is theirs and is kept out of our output, not edited out.
    with contextlib.redirect_stdout(io.StringIO()):
        hit_rates = whale.compute_hit_rates(pd.DataFrame(rows))
    got = hit_rates.get("exchange_withdrawal", {}).get(24)
    if not got:
        raise RivalStudyError("whale-signals skipped the sample (fewer than 30 events)")
    return {"fixed_null": float(got["pvalue"]), "hit_rate": float(got["hit_rate"])}


# --- one draw, every method --------------------------------------------------------------------


@dataclass
class Scorers:
    vibe: Any
    ballast: Any
    whale: Any
    frame: VibeFrame


def load_scorers(panel: Panel) -> Scorers:
    return Scorers(load_vibe_eventstudy(), load_ballast_stats(), load_event_study_module(),
                   VibeFrame(panel))


def score_draw(
    panel: Panel, scorers: Scorers, events: Sequence[Event], *, shock: float = 0.0,
    estimation_bars: int = ESTIMATION_BARS,
) -> dict[str, float]:
    """Every method's p-value on one set of events. Keys are ``<system>.<test>``."""
    returns = _injected(panel, events, shock)
    out: dict[str, float] = {}
    clock = time.perf_counter()
    windows = argus_windows(panel, events, returns, estimation_bars=estimation_bars)
    out.update({f"argus.{k}": v for k, v in score_argus(windows).items()})
    out["seconds.argus"] = time.perf_counter() - clock
    clock = time.perf_counter()
    touched = {e[0] for e in events} if shock else set()
    vibe_frame = scorers.frame.with_returns(returns, touched)
    out.update({f"vibe.{k}": v for k, v in score_vibe(
        scorers.vibe, scorers.frame, panel, events, vibe_frame,
        estimation_bars=estimation_bars).items()})
    out["seconds.vibe"] = time.perf_counter() - clock
    clock = time.perf_counter()
    raw = raw_window_returns(returns, events, EVENT_BARS)
    out.update({f"ballast.{k}": v for k, v in score_ballast(scorers.ballast, raw).items()})
    out["seconds.ballast"] = time.perf_counter() - clock
    clock = time.perf_counter()
    out.update({f"whale.{k}": v for k, v in score_whale(scorers.whale, returns, events).items()})
    out["seconds.whale"] = time.perf_counter() - clock
    return out


DECISION_KEYS: tuple[str, ...] = (
    "argus.verdict", "argus.bmp_adjusted", "argus.patell_adjusted",
    "argus.corrado_rank_adjusted", "argus.generalised_sign_adjusted", "argus.any_adjusted",
    "argus.verdict_raw", "argus.bmp_raw", "argus.patell_raw",
    "argus.patell_summed_adjusted", "argus.patell_summed_raw",
    "vibe.t", "vibe.patell", "vibe.bmp", "vibe.corrado_rank", "vibe.cowan_sign",
    "ballast.pooled_t", "whale.fixed_null",
)
"""The p-values compared. ``argus.verdict`` is the system's own decision (all four adjusted
tests reject); ``vibe.bmp`` is Vibe-Trading's own recommendation ("When the three disagree, BMP is
the one to report"); ``argus.*_raw`` and ``argus.patell_summed_*`` are ablation arms, not ARGUS's
answer."""


# --- aggregation ------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rejection_table(draws: Sequence[dict[str, float]]) -> dict[str, dict[str, Any]]:
    n = len(draws)
    table: dict[str, dict[str, Any]] = {}
    for key in DECISION_KEYS:
        k = sum(1 for d in draws if d[key] <= ALPHA)
        lo, hi = wilson(k, n)
        table[key] = {"rejections": k, "draws": n, "rate": round(k / n, 4),
                      "ci95": [round(lo, 4), round(hi, 4)],
                      "nominal_inside_ci95": lo <= ALPHA <= hi}
    return table


def critical_p(null_draws: Sequence[dict[str, float]], key: str) -> float:
    """The p-value below which exactly 5% of null draws fall, for size-adjusted power."""
    ps = sorted(d[key] for d in null_draws)
    return ps[max(0, int(ALPHA * len(ps)) - 1)]


@dataclass
class Run:
    shape: str
    draws: list[dict[str, float]]
    seconds: float
    extra: dict[str, float]


def run_shape(
    panel: Panel, scorers: Scorers, shape: str, *, n_draws: int, seed: int,
    shock: float = 0.0, window: tuple[int, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Run:
    estimation = 120 if shape == "independent_short_estimation" else ESTIMATION_BARS
    low, high = window or eligible(panel, estimation_bars=estimation)
    rng = random.Random(seed)
    draws: list[dict[str, float]] = []
    start = time.perf_counter()
    for i in range(n_draws):
        events = draw_events(shape, rng, panel, low, high)
        try:
            draws.append(score_draw(panel, scorers, events, shock=shock,
                                    estimation_bars=estimation))
        except EventStudyError as exc:
            raise RivalStudyError(f"{shape} draw {i}: {exc}") from exc
        if progress and (i + 1) % 25 == 0:
            progress(f"{shape} shock={shock} {i + 1}/{n_draws}")
    seconds = time.perf_counter() - start
    extra = {
        "mean_measured_correlation": sum(d["argus.correlation"] for d in draws) / len(draws),
        "vibe_flagged_shared_dates_rate": sum(
            d["vibe.flagged_shared_dates"] for d in draws) / len(draws),
    }
    return Run(shape, draws, seconds, extra)


NULL_SHAPES = (
    "independent", "clustered_3", "clustered_9", "us_open_clustered", "crypto_pair",
    "near_clustered", "independent_short_estimation",
)
POWER = (("independent", 0.005), ("clustered_9", 0.005), ("clustered_9", 0.01))
"""(shape, injected abnormal return at the first event bar). 50bps and 100bps."""
OOS_SHAPES = ("independent", "clustered_9", "us_open_clustered")


Job = tuple[str, str, int, int, float, "tuple[int, int] | None"]
"""(label, shape, n_draws, seed, shock, window). Plain data so a worker process can run it."""


def _run_job(job: Job) -> tuple[str, Run]:
    """One job in a worker. Loads its own panel from the snapshot: nothing live crosses a process
    boundary, and every job is a pure function of the snapshot bytes and its seed."""
    label, shape, n_draws, seed, shock, window = job
    panel = load_hourly_panel()
    return label, run_shape(panel, load_scorers(panel), shape, n_draws=n_draws, seed=seed,
                            shock=shock, window=window)


def study_jobs(panel: Panel, *, n_draws: int, oos_draws: int, seed: int) -> list[Job]:
    jobs: list[Job] = []
    for offset, shape in enumerate(NULL_SHAPES):
        jobs.append((f"null:{shape}", shape, n_draws, seed + offset, 0.0, None))
    for offset, (shape, shock) in enumerate(POWER):
        jobs.append((f"power:{shape}:{shock}", shape, n_draws, seed + 100 + offset, shock, None))
    low, high = eligible(panel)
    mid = (low + high) // 2
    for half, bounds in (("first_half", (low, mid)), ("second_half", (mid, high))):
        for offset, shape in enumerate(OOS_SHAPES):
            jobs.append((f"oos:{half}:{shape}", shape, oos_draws,
                         seed + 200 + offset + (50 if half == "second_half" else 0), 0.0, bounds))
    return jobs


def run_study(
    panel: Panel | None = None, *, n_draws: int = 200, oos_draws: int = 100,
    seed: int = 20260925, workers: int = 1,
) -> dict[str, Any]:
    """The whole comparison: size on every null shape, power, and the two calendar halves.

    ``workers > 1`` runs the jobs in separate processes. Each job carries its own seed, so the
    result does not depend on how many workers ran it or in what order they finished.
    """
    panel = panel or load_hourly_panel()
    jobs = study_jobs(panel, n_draws=n_draws, oos_draws=oos_draws, seed=seed)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = dict(pool.map(_run_job, jobs))
    else:
        scorers = load_scorers(panel)
        runs = {label: run_shape(panel, scorers, shape, n_draws=n, seed=sd, shock=shock,
                                 window=window)
                for label, shape, n, sd, shock, window in jobs}

    nulls = {shape: runs[f"null:{shape}"] for shape in NULL_SHAPES}
    power: list[dict[str, Any]] = []
    for shape, shock in POWER:
        run = runs[f"power:{shape}:{shock}"]
        null = nulls[shape].draws
        adjusted = {}
        for key in DECISION_KEYS:
            crit = critical_p(null, key)
            adjusted[key] = round(sum(1 for d in run.draws if d[key] <= crit) / len(run.draws), 4)
        power.append({
            "shape": shape, "injected_abnormal_bps": round(shock * 10_000, 1),
            "raw_power": {k: v["rate"] for k, v in rejection_table(run.draws).items()},
            "size_adjusted_power": adjusted,
            "seconds": round(run.seconds, 1),
        })
    low, high = eligible(panel)
    mid = (low + high) // 2
    oos: dict[str, dict[str, Any]] = {}
    for half, bounds in (("first_half", (low, mid)), ("second_half", (mid, high))):
        oos[half] = {
            "bars": [panel.stamps[bounds[0]].isoformat(), panel.stamps[bounds[1] - 1].isoformat()],
        }
        for shape in OOS_SHAPES:
            oos[half][shape] = rejection_table(runs[f"oos:{half}:{shape}"].draws)
    return {"panel": panel, "nulls": nulls, "power": power, "oos": oos, "jobs": len(jobs)}


def reproducibility(
    panel: Panel | None = None, *, n_draws: int = 12, seed: int = 7,
) -> dict[str, Any]:
    """The same shape, seed and snapshot scored twice in one process: the p-values must match
    exactly, not approximately. Timings are excluded — they are the one thing that should differ."""
    import hashlib

    panel = panel or load_hourly_panel()
    scorers = load_scorers(panel)

    def digest() -> str:
        run = run_shape(panel, scorers, "clustered_9", n_draws=n_draws, seed=seed)
        body = [{k: v for k, v in d.items() if not k.startswith("seconds.")} for d in run.draws]
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    first, second = digest(), digest()
    return {"draws": n_draws, "seed": seed, "first_digest": first, "second_digest": second,
            "identical": first == second, "snapshot_sha256": panel.source_sha256}


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Artefact form: rejection tables with CIs, the headline comparison, and what it cost."""
    panel: Panel = result["panel"]
    nulls: dict[str, Run] = result["nulls"]
    size: dict[str, dict[str, Any]] = {
        shape: {
            "rejection_rates": rejection_table(run.draws),
            "draws": len(run.draws),
            "seconds": round(run.seconds, 1),
            **{k: round(v, 5) for k, v in run.extra.items()},
        }
        for shape, run in nulls.items()
    }
    headline: dict[str, dict[str, float]] = {}
    for shape in NULL_SHAPES:
        rates: dict[str, dict[str, Any]] = size[shape]["rejection_rates"]
        headline[shape] = {
            "argus_verdict": rates["argus.verdict"]["rate"],
            "argus_bmp_adjusted": rates["argus.bmp_adjusted"]["rate"],
            "vibe_bmp": rates["vibe.bmp"]["rate"],
            "vibe_patell": rates["vibe.patell"]["rate"],
            "vibe_t": rates["vibe.t"]["rate"],
            "ballast_pooled_t": rates["ballast.pooled_t"]["rate"],
            "whale_fixed_null": rates["whale.fixed_null"]["rate"],
        }
    return {
        "generated_from": {
            "snapshot": str(SNAPSHOT_HOURLY.name),
            "snapshot_sha256": panel.source_sha256,
            "fetched_at": panel.fetched_at,
            "bars": panel.bars,
            "first_bar": panel.stamps[0].isoformat(),
            "last_bar": panel.stamps[-1].isoformat(),
            "names": list(NAMES),
            "market": MARKET,
            "event_bars": EVENT_BARS,
            "estimation_bars": ESTIMATION_BARS,
            "gap_bars": GAP_BARS,
            "events_per_draw": EVENTS_PER_DRAW,
        },
        "rivals": {
            "vibe_trading": {"file": "agent/src/quantlib/eventstudy.py", "commit": VIBE_COMMIT,
                             "vendored_sha256": vendored_body_sha256(
                                 "vibe_trading_eventstudy.py"), "licence": "MIT"},
            "ballast": {"file": "ballast/stats.py", "commit": BALLAST_COMMIT,
                        "vendored_sha256": vendored_body_sha256("ballast_stats.py"),
                        "licence": "MIT"},
            "whale_signals": {"file": "src/analysis/event_study.py",
                              "commit": "6be10a598c9319aa6b64bf517618eac2dd2ec2f5",
                              "licence": "MIT"},
        },
        "size_under_null": size,
        "headline_false_positive_rates": headline,
        "power": result["power"],
        "out_of_sample_halves": result["oos"],
        "costs": costs(nulls),
    }


def costs(nulls: dict[str, Run]) -> dict[str, Any]:
    """Wall-clock seconds per draw of 108 events, per system, averaged over every null draw."""
    draws = [d for run in nulls.values() for d in run.draws]
    out: dict[str, Any] = {"draws": len(draws)}
    for system in ("argus", "vibe", "ballast", "whale"):
        values = [d[f"seconds.{system}"] for d in draws]
        out[f"{system}_seconds_per_draw"] = round(sum(values) / len(values), 5)
    out["argus_over_vibe"] = round(out["argus_seconds_per_draw"]
                                   / max(out["vibe_seconds_per_draw"], 1e-9), 2)
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m argus.eval.eventdriven_rivals`` — the full study, written to
    ``data/eventdriven_rivals.json``. No model is called; about eight minutes in one process."""
    import argparse

    parser = argparse.ArgumentParser(description="event significance under clustering, vs rivals")
    parser.add_argument("--draws", type=int, default=200, help="placebo draws per null shape")
    parser.add_argument("--oos-draws", type=int, default=100, help="draws per calendar half")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    panel = load_hourly_panel()
    result = run_study(panel, n_draws=args.draws, oos_draws=args.oos_draws, seed=args.seed,
                       workers=args.workers)
    report = summarise(result)
    report["protocol"] = {"draws_per_null_shape": args.draws, "oos_draws": args.oos_draws,
                          "seed": args.seed, "alpha": ALPHA, "jobs": result["jobs"],
                          "qwen_calls": 0}
    report["reproducibility"] = reproducibility(panel)
    non_finite = write_artefact(REPORT_PATH, report)
    for shape, row in report["headline_false_positive_rates"].items():
        print(f"[eventdriven-rivals] {shape}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
    if non_finite:
        print(f"[eventdriven-rivals] non-finite values written as null at: {non_finite}")
    print(f"written to {REPORT_PATH}")
    return 0


__all__ = [
    "DECISION_KEYS",
    "EVENTS_PER_DRAW",
    "NAMES",
    "NULL_SHAPES",
    "Panel",
    "RivalStudyError",
    "argus_windows",
    "draw_events",
    "eligible",
    "load_hourly_panel",
    "load_scorers",
    "main",
    "rejection_table",
    "run_shape",
    "run_study",
    "score_argus",
    "score_draw",
    "summarise",
    "wilson",
]


if __name__ == "__main__":
    raise SystemExit(main())
