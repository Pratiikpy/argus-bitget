"""General-purpose rival: the data-honest breadth rotation against general data-validation tools.

**The function, stated without trading.** The capability "Data-honest cross-asset breadth rotation"
is a computation over a panel of time series that must *refuse* rather than emit a quietly wrong
answer when its inputs cannot support it. Stripped of the finance, that is **data-contract
validation**: checking a panel for completeness, validity, ordering, timeliness and sufficiency
before the computation, the call's own parameters before it runs, and an invariant on its output
after. The rivals it has to beat are therefore not trading repositories but the general tools built
for exactly that job.

**Candidates, and why these two.** Surveyed 2026-09-25 (GitHub API, pypistats.org last-month
downloads): Great Expectations (Apache-2.0, 11.8k stars, 19.3M downloads/month, maintained — repo
now `fivetran/great_expectations`), pandera (MIT, 4.5k stars, 8.2M/month, maintained, built for
validating dataframes *inside* the function call path via ``check_input``/``check_output``), PyDeequ
(Apache-2.0, 12.7M/month but Spark/JVM, no in-process pandas path), patito (MIT, 1.9M/month,
polars-only), soda-core (1.8M/month, NOASSERTION licence, SQL-first), dataframely (BSD-3, 0.6M/
month, polars-only), pointblank (MIT, 17k/month). pandera and Great Expectations are the two that
validate the pandas frames the reference already builds, in-process, and between them hold the
bulk of adoption; both were installed into an isolated venv (pandera 0.33.1, great_expectations
1.23.1, pandas 3.0.6, pydantic from their own dependency set) and run by
``scripts/general_rotation_rival.py`` — never imported by ARGUS. pydantic ``@validate_call``-style
parameter validation is added on top of pandera as a third configuration, because neither frame
validator can see a function's arguments and pydantic is the general-purpose tool that does.

**Same input.** Real Bitget ``1D`` history for nine instruments (four rTokens/crypto majors and
two gold-backed tokens with thirteen-plus months of listing, plus QQQUSDT, XAUUSDT and XAGUSDT,
which really are shorter today), fetched once and frozen in ``data/general_rotation_corpus.json``.
From it :func:`build_cases` derives 36 cases — each a single, named contract violation injected
into the real clean panel, a real condition present in the data as fetched, or a control — and
every contestant receives the byte-identical case (``case_digest`` is recomputed by the rival
runner from the file it read and checked here).

**Contestants.** ``argus_after`` (the current :mod:`argus.desk.rotation`), ``argus_before`` (the
byte-pinned pre-contract module, the ablation), ``pytaa_raw`` (the specialist reference with no
validation), ``pandera`` (pytaa wrapped in lazy pandera schemas on bars, panel, freshness, scores
and output), ``pandera_pydantic`` (the same plus a pydantic parameter model) and
``great_expectations`` (pytaa wrapped in native GX expectation suites on the same frames).

**Scoring.** Every case is one of three kinds. *must_refuse*: no honest allocation exists, so any
emitted book is a silent accept. *tolerable*: the violation does not touch a value the formula
reads (a gap between anchor months, reordered bars), so refusing is an over-refusal and accepting
is correct only if the book equals the same contestant's book on the clean panel. *control*: a
valid input, so refusing is a false positive and the book must satisfy the output invariants and
equal the reference arithmetic. Diagnostics count how many of a refused case's offending symbols
the refusal names. Costs are wall-clock per full call on the clean panel.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.desk import rotation as after
from argus.eval import artefact
from argus.eval.baselines import argus_rotation_pre_contract as before

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
CORPUS_PATH = DATA / "general_rotation_corpus.json"
RIVAL_RESULTS_PATH = DATA / "general_rotation_rival_results.json"
ARTEFACT_PATH = DATA / "general_rotation_comparison.json"
RIVAL_RUNNER = ROOT / "scripts" / "general_rotation_rival.py"

CORPUS_SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT", "AAPLUSDT", "QQQUSDT", "BTCUSDT", "ETHUSDT",
    "PAXGUSDT", "XAUTUSDT", "XAUUSDT", "XAGUSDT",
)
CORPUS_DAYS = 800

RISK: tuple[str, ...] = ("NVDAUSDT", "AAPLUSDT", "BTCUSDT", "ETHUSDT")
SAFE: tuple[str, ...] = ("PAXGUSDT", "XAUTUSDT")
"""The clean universe: every member has at least thirteen real monthly closes in the frozen corpus
(NVDAUSDT and AAPLUSDT 14, BTCUSDT, ETHUSDT and PAXGUSDT 27 — the whole 800-day window —
XAUTUSDT 18). PAXGUSDT and XAUTUSDT are real gold-backed tokens: their last corpus closes,
$4,284.38 and $4,287.72 on the bar closing 2026-09-25, sit within 0.25% of Bitget's own XAUUSDT
gold contract on the same bar ($4,294.67), which has only ten monthly closes — too few for the
twelve-month lookback, and the reason the CLI's default safe asset is a must-refuse case here."""
TOP_K = 2
STEP = 0.25

CLI_DEFAULT_RISK: tuple[str, ...] = ("NVDAUSDT", "AAPLUSDT", "QQQUSDT", "BTCUSDT", "ETHUSDT")
CLI_DEFAULT_SAFE: tuple[str, ...] = ("XAUUSDT",)
"""`argus.desk.rotation.main`'s own defaults, used verbatim for the real short-history case."""

CONTESTANTS: tuple[str, ...] = (
    "argus_after", "argus_before", "pytaa_raw", "pandera", "pandera_pydantic",
    "great_expectations",
)
RIVALS: tuple[str, ...] = ("pytaa_raw", "pandera", "pandera_pydantic", "great_expectations")
GENERAL_RIVALS: tuple[str, ...] = ("pandera", "pandera_pydantic", "great_expectations")

Bars = list[tuple[datetime, float]]


# --- corpus ------------------------------------------------------------------------------------


def freeze_corpus(path: Path = CORPUS_PATH) -> dict[str, Any]:  # pragma: no cover - network
    """Fetch the real history once and write it. Timestamps are stored as Bitget returns them
    (candle *open* times); :func:`load_corpus` converts to close times."""
    from argus.market.history import CandleType, fetch_range

    bars: dict[str, list[list[str]]] = {}
    for symbol in CORPUS_SYMBOLS:
        candles = fetch_range(symbol, days=CORPUS_DAYS, interval="1D",
                              candle_type=CandleType.MARKET)
        bars[symbol] = [[c.ts.isoformat(), str(c.close)] for c in candles]
    blob = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": "argus.market.history.fetch_range(symbol, days=800, interval='1D', "
                  "candle_type=MARKET) — Bitget public USDT-futures candles",
        "timestamp_convention": "open time of the 1D candle (16:00 UTC); the close prints 24h "
                                "later — verified 2026-09-25 against the hourly candle opened "
                                "23h after each daily open",
        "bars": bars,
    }
    artefact.write(path, blob)
    return blob


def load_corpus(path: Path = CORPUS_PATH) -> dict[str, Bars]:
    """The frozen real bars, as ``(close_time, close)`` in chronological order."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Bars] = {}
    for symbol, rows in blob["bars"].items():
        out[symbol] = [
            (datetime.fromisoformat(ts) + after.BAR_INTERVAL, float(close)) for ts, close in rows
        ]
    return out


# --- cases -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    case_id: str
    stage: str  # "bars" | "scores"
    kind: str  # "must_refuse" | "tolerable" | "control"
    clause: str
    description: str
    arises_in_live_path: bool
    risk: tuple[str, ...]
    safe: tuple[str, ...]
    top_k: Any
    step: Any
    offending: tuple[str, ...]
    reference: str | None
    bars: dict[str, Bars] | None = None
    scores: dict[str, float] | None = None

    @property
    def universe(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.risk, *self.safe)))

    def wire(self) -> dict[str, Any]:
        """The exact bytes both sides compute on: floats as ``repr`` strings (so NaN and inf
        survive strict JSON), timestamps as ISO-8601 with offset."""
        return {
            "case_id": self.case_id, "stage": self.stage, "kind": self.kind,
            "risk": list(self.risk), "safe": list(self.safe),
            "top_k": self.top_k, "step": repr(float(self.step)),
            "bars": None if self.bars is None else {
                s: [[ts.isoformat(), repr(float(c))] for ts, c in rows]
                for s, rows in self.bars.items()
            },
            "scores": None if self.scores is None else {
                s: repr(float(v)) for s, v in self.scores.items()
            },
        }

    def meta(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "stage": self.stage, "kind": self.kind,
            "clause": self.clause, "description": self.description,
            "arises_in_live_path": self.arises_in_live_path,
            "risk": list(self.risk), "safe": list(self.safe),
            "top_k": self.top_k, "step": repr(float(self.step)),
            "offending": list(self.offending), "reference": self.reference,
            "digest": case_digest(self),
        }


def case_digest(case: Case) -> str:
    return hashlib.sha256(
        json.dumps(case.wire(), sort_keys=True).encode("utf-8")).hexdigest()


def _month(ts: datetime) -> int:
    return ts.year * 12 + ts.month - 1


def _drop_month(rows: Bars, month: int) -> Bars:
    return [(ts, c) for ts, c in rows if _month(ts) != month]


def _set_month_close(rows: Bars, month: int, value: float) -> Bars:
    last = max(i for i, (ts, _) in enumerate(rows) if _month(ts) == month)
    out = list(rows)
    out[last] = (rows[last][0], value)
    return out


def _duplicate_month_close(rows: Bars, month: int, factor: float) -> Bars:
    last = max(i for i, (ts, _) in enumerate(rows) if _month(ts) == month)
    out = list(rows)
    out.insert(last + 1, (rows[last][0], rows[last][1] * factor))
    return out


def build_cases(corpus: Mapping[str, Bars]) -> list[Case]:
    """Every case the comparison scores, derived deterministically from the frozen real corpus."""
    clean = {s: list(corpus[s]) for s in (*RISK, *SAFE)}
    freshest = max(rows[-1][0] for rows in clean.values())
    evaluation = _month(freshest)

    def bars_case(case_id: str, kind: str, clause: str, description: str, live: bool,
                  bars: dict[str, Bars], offending: Sequence[str] = (),
                  reference: str | None = None, risk: Sequence[str] = RISK,
                  safe: Sequence[str] = SAFE) -> Case:
        return Case(case_id, "bars", kind, clause, description, live, tuple(risk), tuple(safe),
                    TOP_K, STEP, tuple(offending), reference, bars=bars)

    def mutate(symbol: str, fn: Callable[[Bars], Bars]) -> dict[str, Bars]:
        out = {s: list(rows) for s, rows in clean.items()}
        out[symbol] = fn(out[symbol])
        return out

    cases: list[Case] = [
        bars_case("clean", "control", "none",
                  "the real clean panel: four risk assets, two gold tokens, all >= 13 months",
                  True, clean),
        bars_case("gap_between_anchors", "tolerable", "sufficiency",
                  "ETHUSDT loses every bar of the month five back — a month the formula never "
                  "reads (anchors are 0/1/3/6/12 back)", True,
                  mutate("ETHUSDT", lambda r: _drop_month(r, evaluation - 5)), reference="clean"),
        bars_case("gap_outside_lookback", "tolerable", "sufficiency",
                  "BTCUSDT loses a month eighteen back, older than the 12-month lookback", True,
                  mutate("BTCUSDT", lambda r: _drop_month(r, evaluation - 18)),
                  reference="clean"),
        bars_case("gap_on_anchor", "must_refuse", "sufficiency",
                  "ETHUSDT loses every bar of the 3-month anchor month", True,
                  mutate("ETHUSDT", lambda r: _drop_month(r, evaluation - 3)),
                  offending=("ETHUSDT",)),
        bars_case("stale_across_month", "must_refuse", "timeliness",
                  "NVDAUSDT stopped updating 45 days before the rest (a halted listing)", True,
                  mutate("NVDAUSDT",
                         lambda r: [(t, c) for t, c in r if t <= freshest - timedelta(days=45)]),
                  offending=("NVDAUSDT",)),
        bars_case("stale_within_month", "must_refuse", "timeliness",
                  "NVDAUSDT stopped updating 12 days before the rest, inside the same month", True,
                  mutate("NVDAUSDT",
                         lambda r: [(t, c) for t, c in r if t <= freshest - timedelta(days=12)]),
                  offending=("NVDAUSDT",)),
        bars_case("missing_series", "must_refuse", "completeness",
                  "ETHUSDT's series is absent altogether (a failed fetch)", True,
                  {s: rows for s, rows in clean.items() if s != "ETHUSDT"},
                  offending=("ETHUSDT",)),
        bars_case("nan_close_on_anchor", "must_refuse", "validity",
                  "BTCUSDT's 3-month anchor close is NaN", False,
                  mutate("BTCUSDT", lambda r: _set_month_close(r, evaluation - 3, math.nan)),
                  offending=("BTCUSDT",)),
        bars_case("negative_close_on_anchor", "must_refuse", "validity",
                  "BTCUSDT's 3-month anchor close is negated", False,
                  mutate("BTCUSDT", lambda r: _set_month_close(
                      r, evaluation - 3, -[c for t, c in r if _month(t) == evaluation - 3][-1])),
                  offending=("BTCUSDT",)),
        bars_case("inf_close_on_anchor", "must_refuse", "validity",
                  "BTCUSDT's 3-month anchor close is +inf", False,
                  mutate("BTCUSDT", lambda r: _set_month_close(r, evaluation - 3, math.inf)),
                  offending=("BTCUSDT",)),
        bars_case("zero_close_on_anchor", "must_refuse", "validity",
                  "BTCUSDT's 3-month anchor close is zero", False,
                  mutate("BTCUSDT", lambda r: _set_month_close(r, evaluation - 3, 0.0)),
                  offending=("BTCUSDT",)),
        bars_case("unsorted_bars", "tolerable", "ordering",
                  "ETHUSDT's bars arrive newest-first", False,
                  mutate("ETHUSDT", lambda r: list(reversed(r))), reference="clean"),
        bars_case("conflicting_duplicate_bar", "must_refuse", "ordering",
                  "ETHUSDT's 3-month anchor bar appears twice with closes 10% apart", False,
                  mutate("ETHUSDT", lambda r: _duplicate_month_close(r, evaluation - 3, 1.1)),
                  offending=("ETHUSDT",)),
        bars_case("real_short_history_cli_default", "must_refuse", "sufficiency",
                  "the CLI's own default universe on today's real data: QQQUSDT has 12 monthly "
                  "closes and XAUUSDT 10", True,
                  {s: list(corpus[s]) for s in (*CLI_DEFAULT_RISK, *CLI_DEFAULT_SAFE)},
                  offending=("QQQUSDT", "XAUUSDT"),
                  risk=CLI_DEFAULT_RISK, safe=CLI_DEFAULT_SAFE),
        bars_case("real_short_history_three", "must_refuse", "sufficiency",
                  "the default universe plus silver: three real short-history symbols", True,
                  {s: list(corpus[s]) for s in (*CLI_DEFAULT_RISK, "XAUUSDT", "XAGUSDT")},
                  offending=("QQQUSDT", "XAUUSDT", "XAGUSDT"),
                  risk=CLI_DEFAULT_RISK, safe=("XAUUSDT", "XAGUSDT")),
    ]
    multi = {s: list(rows) for s, rows in clean.items()}
    multi["ETHUSDT"] = _drop_month(multi["ETHUSDT"], evaluation - 3)
    multi["NVDAUSDT"] = [(t, c) for t, c in multi["NVDAUSDT"] if t <= freshest - timedelta(days=45)]
    multi["XAUTUSDT"] = [(t, c) for t, c in multi["XAUTUSDT"] if _month(t) > evaluation - 11]
    cases.append(bars_case(
        "three_faults_at_once", "must_refuse", "several",
        "ETHUSDT anchor gap + NVDAUSDT stale 45 days + XAUTUSDT cut to 11 months, in one call",
        True, multi, offending=("ETHUSDT", "NVDAUSDT", "XAUTUSDT")))

    clean_scores = {
        s: after.momentum_score(after.monthly_closes_from_bars(rows))
        for s, rows in clean.items()
    }

    def scores_case(case_id: str, kind: str, clause: str, description: str, live: bool,
                    scores: Mapping[str, float], *, risk: Sequence[str] = RISK,
                    safe: Sequence[str] = SAFE, top_k: Any = TOP_K, step: Any = STEP,
                    offending: Sequence[str] = (), reference: str | None = None) -> Case:
        # Only the universe's own scores: the reference counts negatives across every entry of
        # the Series it is handed (`np.where(data < 0, ...)`), so a stray extra score would
        # change its breadth count and make the comparison about input shape, not validation.
        members = dict.fromkeys((*risk, *safe))
        return Case(case_id, "scores", kind, clause, description, live, tuple(risk),
                    tuple(safe), top_k, step, tuple(offending), reference,
                    scores={s: scores[s] for s in members if s in scores})

    def with_scores(**overrides: float) -> dict[str, float]:
        return {**clean_scores, **overrides}

    ranked_risk = sorted(RISK, key=lambda s: clean_scores[s])
    cases += [
        scores_case("scores_clean", "control", "none",
                    "the real clean scores, default parameters", True, clean_scores),
        scores_case("nan_one_of_two_safe", "must_refuse", "completeness",
                    "PAXGUSDT's score is NaN; XAUTUSDT is valid", True,
                    with_scores(PAXGUSDT=math.nan), offending=("PAXGUSDT",)),
        scores_case("nan_only_safe", "must_refuse", "completeness",
                    "the only safe asset's score is NaN (the reference's measured 75%-vanish "
                    "class)", True, with_scores(PAXGUSDT=math.nan), safe=("PAXGUSDT",),
                    offending=("PAXGUSDT",)),
        scores_case("all_nan", "must_refuse", "completeness",
                    "every score is NaN", True,
                    dict.fromkeys(clean_scores, math.nan),
                    offending=(*RISK, *SAFE)),
        scores_case("inf_score", "must_refuse", "validity",
                    "BTCUSDT's score is +inf", False, with_scores(BTCUSDT=math.inf),
                    offending=("BTCUSDT",)),
        scores_case("empty_safe", "must_refuse", "parameters",
                    "no safe asset named", True, clean_scores, safe=()),
        scores_case("empty_risk", "must_refuse", "parameters",
                    "no risk asset named", True, clean_scores, risk=()),
        scores_case("top_k_exceeds_risk", "must_refuse", "parameters",
                    "top_k=5 with four risk assets (a --top-k typo on the CLI)", True,
                    clean_scores, top_k=5),
        scores_case("top_k_zero", "must_refuse", "parameters", "top_k=0", True,
                    clean_scores, top_k=0),
        scores_case("top_k_negative", "must_refuse", "parameters", "top_k=-1", True,
                    clean_scores, top_k=-1),
        scores_case("step_negative", "must_refuse", "parameters", "step=-0.25", False,
                    clean_scores, step=-0.25),
        scores_case("step_nan", "must_refuse", "parameters", "step=NaN", False,
                    clean_scores, step=math.nan),
        scores_case("asset_in_both_sets", "must_refuse", "parameters",
                    "PAXGUSDT named as both risk and safe (the CLI accepts it)", True,
                    clean_scores, risk=(*RISK, "PAXGUSDT"), offending=("PAXGUSDT",)),
        scores_case("duplicate_risk_asset", "must_refuse", "parameters",
                    "BTCUSDT named twice among the risk assets (the CLI accepts it)", True,
                    clean_scores, risk=(*RISK, "BTCUSDT"), offending=("BTCUSDT",)),
        scores_case("step_one_control", "control", "none",
                    "step=1.0: any negative score sends the whole book to safety — a valid "
                    "published variant", True, clean_scores, step=1.0),
        scores_case("top_k_all_risk_control", "control", "none",
                    "top_k equal to the risk count", True, clean_scores, top_k=len(RISK)),
    ]
    for flipped in range(1, len(RISK) + 1):
        overrides = {s: -abs(clean_scores[s]) or -0.01 for s in ranked_risk[-flipped:]}
        cases.append(scores_case(
            f"breadth_{flipped}_negative_control", "control", "none",
            f"{flipped} of {len(RISK)} risk scores negative (real magnitudes, sign flipped)",
            True, with_scores(**overrides)))
    return cases


# --- ARGUS contestants -------------------------------------------------------------------------


def _result(outcome: str, weights: Mapping[str, float] | None, message: str | None,
            exception: str | None, seconds: float,
            violations: list[dict[str, Any]] | None = None,
            scores: Mapping[str, float] | None = None) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "weights": None if weights is None else {k: float(v) for k, v in weights.items()},
        "scores": None if scores is None else {k: float(v) for k, v in scores.items()},
        "message": None if message is None else message[:4000],
        "exception": exception,
        "violations": violations,
        "seconds": seconds,
    }


def run_argus_after(case: Case) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if case.stage == "bars":
            assert case.bars is not None
            plan = after.propose_rotation_from_bars(
                {}, case.bars, case.risk, case.safe, top_k=case.top_k, step=case.step,
                min_leg=0.0)
            weights = {t.symbol: t.weight_after for t in plan.trades}
            scores: dict[str, float] | None = dict(plan.scores)
        else:
            assert case.scores is not None
            weights = after.breadth_allocation(case.scores, case.risk, case.safe,
                                               top_k=case.top_k, step=case.step)
            scores = None
    except after.RotationError as exc:
        return _result("refused", None, str(exc), "RotationError", time.perf_counter() - start,
                       [v.as_dict() for v in exc.violations])
    except Exception as exc:
        return _result("crashed", None, str(exc), type(exc).__name__,
                       time.perf_counter() - start)
    return _result("accepted", weights, None, None, time.perf_counter() - start,
                   scores=scores)


def run_argus_before(case: Case) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if case.stage == "bars":
            assert case.bars is not None
            monthly = {s: before.monthly_closes_from_bars(rows) for s, rows in case.bars.items()}
            plan = before.propose_rotation({}, monthly, case.risk, case.safe, top_k=case.top_k,
                                           step=case.step, min_leg=0.0)
            weights = {t.symbol: t.weight_after for t in plan.trades}
            # The pre-contract plan carries no scores; recomputed with its own functions.
            scores: dict[str, float] | None = {
                s: before.momentum_score(monthly[s]) for s in case.universe}
        else:
            assert case.scores is not None
            weights = before.breadth_allocation(case.scores, case.risk, case.safe,
                                                top_k=case.top_k, step=case.step)
            scores = None
    except before.RotationError as exc:
        return _result("refused", None, str(exc), "RotationError", time.perf_counter() - start)
    except Exception as exc:
        return _result("crashed", None, str(exc), type(exc).__name__,
                       time.perf_counter() - start)
    return _result("accepted", weights, None, None, time.perf_counter() - start,
                   scores=scores)


ARGUS_RUNNERS: dict[str, Callable[[Case], dict[str, Any]]] = {
    "argus_after": run_argus_after,
    "argus_before": run_argus_before,
}


def measure_argus_costs(case: Case, repeats: int = 200) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, runner in ARGUS_RUNNERS.items():
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            runner(case)
            samples.append(time.perf_counter() - t0)
        out[name] = {"repeats": repeats, "median_seconds": statistics.median(samples),
                     "min_seconds": min(samples)}
    return out


# --- rivals (recorded; produced out of process) -------------------------------------------------


def write_wire(cases: Sequence[Case], path: Path) -> str:
    """The case file the rival runner reads. Returns its SHA-256."""
    text = json.dumps({"cases": [c.wire() for c in cases]}, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_rivals(python: Path, wire_path: Path, out: Path = RIVAL_RESULTS_PATH,
               repeats: int = 30) -> None:  # pragma: no cover - runs the rival venv
    import subprocess

    subprocess.run(
        [str(python), str(RIVAL_RUNNER), "--cases", str(wire_path), "--out", str(out),
         "--repeats", str(repeats)],
        check=True, env={**_clean_env(), "GX_ANALYTICS_ENABLED": "false", "PYTHONUTF8": "1"},
    )


def _clean_env() -> dict[str, str]:  # pragma: no cover - runs the rival venv
    import os

    return {k: v for k, v in os.environ.items() if "KEY" not in k.upper()}


def load_rival_results(path: Path = RIVAL_RESULTS_PATH) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


# --- scoring -----------------------------------------------------------------------------------


def _invariants_hold(weights: Mapping[str, float], universe: Sequence[str]) -> bool:
    if set(weights) != set(universe):
        return False
    return all(0.0 <= w <= 1.0 for w in weights.values()) and \
        abs(sum(weights.values()) - 1.0) <= after.WEIGHT_SUM_TOLERANCE


def _same_book(a: Mapping[str, float] | None, b: Mapping[str, float] | None,
               tolerance: float = 1e-12) -> bool:
    if a is None or b is None or set(a) != set(b):
        return False
    return all(abs(a[k] - b[k]) <= tolerance for k in a)


def _same_signal(result: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    """Where both sides report the momentum scores behind the book, they must match too: a
    corrupted score that happens not to change the ranking still mislabels the regime, and it is
    only luck that the book survived."""
    ours, theirs = result.get("scores"), reference.get("scores")
    if ours is None or theirs is None:
        return True
    return _same_book(ours, theirs, tolerance=1e-9)


def classify(case: Case, result: Mapping[str, Any],
             own_reference: Mapping[str, Any] | None,
             arithmetic_reference: Mapping[str, Any] | None) -> str:
    """One word for what the contestant did with the case — see the module docstring."""
    blocked = result["outcome"] in ("refused", "crashed")
    if case.kind == "must_refuse":
        if result["outcome"] == "refused":
            return "refused"
        return "crashed" if result["outcome"] == "crashed" else "silent_accept"
    if case.kind == "tolerable":
        if blocked:
            return "over_refusal" if result["outcome"] == "refused" else "crashed"
        ok = own_reference is not None and own_reference["outcome"] == "accepted" and \
            _same_book(result["weights"], own_reference["weights"]) and \
            _same_signal(result, own_reference)
        return "correct" if ok else "silent_corruption"
    if blocked:
        return "false_positive" if result["outcome"] == "refused" else "crashed"
    weights = result["weights"]
    if not _invariants_hold(weights, case.universe):
        return "wrong"
    if arithmetic_reference is not None and arithmetic_reference["outcome"] == "accepted" and \
            not _same_book(weights, arithmetic_reference["weights"]):
        return "disagrees_with_reference"
    return "correct"


def _named(case: Case, result: Mapping[str, Any]) -> list[str]:
    text = result.get("message") or ""
    return [s for s in case.offending if s in text]


def compare(cases: Sequence[Case], results: Mapping[str, Mapping[str, Mapping[str, Any]]],
            ) -> dict[str, Any]:
    """``results[contestant][case_id]`` -> scoreboard per contestant plus the per-case grid."""
    by_id = {c.case_id: c for c in cases}
    grid: dict[str, dict[str, str]] = {}
    board: dict[str, dict[str, Any]] = {}
    for name, per_case in results.items():
        verdicts: dict[str, str] = {}
        named_total = offending_total = full_named = multi = 0
        for case in cases:
            result = per_case[case.case_id]
            own_ref = per_case.get(case.reference) if case.reference else None
            arith = results["pytaa_raw"].get(case.case_id) if name != "pytaa_raw" and \
                "pytaa_raw" in results else None
            verdict = classify(case, result, own_ref, arith)
            verdicts[case.case_id] = verdict
            if case.kind == "must_refuse" and case.offending and result["outcome"] == "refused":
                names = _named(case, result)
                named_total += len(names)
                offending_total += len(case.offending)
                if len(case.offending) > 1:
                    multi += 1
                    full_named += len(names) == len(case.offending)
        grid[name] = verdicts

        def ids(kind: str, verdict: str | None = None, *, _v: dict[str, str] = verdicts,
                ) -> list[str]:
            return [cid for cid, v in _v.items() if by_id[cid].kind == kind
                    and (verdict is None or v == verdict)]

        must = ids("must_refuse")
        live_must = [cid for cid in must if by_id[cid].arises_in_live_path]
        board[name] = {
            "must_refuse_cases": len(must),
            "refused": len(ids("must_refuse", "refused")),
            "silent_accepts": ids("must_refuse", "silent_accept"),
            "crashed_instead_of_refusing": ids("must_refuse", "crashed"),
            "live_path_must_refuse": len(live_must),
            "live_path_refused": sum(1 for cid in live_must if verdicts[cid] == "refused"),
            "tolerable_cases": len(ids("tolerable")),
            "tolerable_correct": len(ids("tolerable", "correct")),
            "over_refusals": ids("tolerable", "over_refusal"),
            "silent_corruptions": ids("tolerable", "silent_corruption"),
            "control_cases": len(ids("control")),
            "controls_correct": len(ids("control", "correct")),
            "false_positives": ids("control", "false_positive"),
            "controls_wrong": [cid for cid in ids("control")
                               if verdicts[cid] not in ("correct", "false_positive")],
            "offending_symbols_named": named_total,
            "offending_symbols_total": offending_total,
            "multi_symbol_refusals_fully_named": full_named,
            "multi_symbol_refusals": multi,
        }
        good = board[name]["refused"] + board[name]["tolerable_correct"] + \
            board[name]["controls_correct"]
        board[name]["cases_handled_correctly"] = good
        board[name]["cases_total"] = len(cases)
    return {"scoreboard": board, "grid": grid}


DIVERGENCE_CUTOFFS: tuple[str, ...] = (
    "2025-10-10", "2025-12-15", "2026-02-10", "2026-03-10", "2026-06-10", "2026-09-23",
)
"""Six evaluation dates across the corpus's real BTCUSDT history, chosen before running to span
both weekday and weekend month-ends among the anchors."""


def run_business_month_divergence(corpus: Mapping[str, Bars],
                                  symbol: str = "BTCUSDT") -> dict[str, Any]:
    """Not a validation question but a convention one, measured because it bounds every other
    agreement claim about this rotation: pytaa bins monthly closes by *business* month-end
    (``resample("BME")``), ARGUS by calendar month. A bar on a weekend month-end therefore lands
    in the next month's bin on the pytaa side only. Run on the same real bars, cut at each of
    :data:`DIVERGENCE_CUTOFFS`: where every anchor month ends on a weekday the two scores are
    identical; where one does not, they differ."""
    import pandas as pd

    from argus.eval.baselines.pytaa_signal_loader import load_signal_module

    signal = load_signal_module().Signal
    rows = corpus[symbol]
    points = []
    for cutoff in DIVERGENCE_CUTOFFS:
        end = datetime.fromisoformat(cutoff).replace(tzinfo=UTC) + timedelta(days=1)
        cut = [(ts, c) for ts, c in rows if ts < end]
        frame = pd.DataFrame(
            {symbol: [c for _, c in cut]},
            index=pd.DatetimeIndex([ts.replace(tzinfo=None) for ts, _ in cut]))
        reference = float(signal(frame).momentum_score()[symbol].iloc[-1])
        ours = after.momentum_score(after.monthly_closes_from_bars(cut))
        evaluation = _month(cut[-1][0])
        weekend_ends = []
        for _, lag in after.MOMENTUM_HORIZONS:
            month = evaluation - lag
            year, mon = month // 12, month % 12 + 1
            last_day = (datetime(year + (mon == 12), mon % 12 + 1, 1) - timedelta(days=1))
            if last_day.weekday() >= 5:
                weekend_ends.append(f"{year:04d}-{mon:02d}")
        points.append({
            "cutoff": cutoff,
            "pytaa_business_month": reference,
            "argus_calendar_month": ours,
            "difference": ours - reference,
            "identical": abs(ours - reference) <= 1e-12,
            "anchor_months_ending_on_a_weekend": weekend_ends,
        })
    return {
        "symbol": symbol,
        "points": points,
        "identical_exactly_when_no_anchor_month_ends_on_a_weekend": all(
            p["identical"] == (not p["anchor_months_ending_on_a_weekend"]) for p in points),
    }


def verify_same_input(cases: Sequence[Case], rival: Mapping[str, Any]) -> dict[str, Any]:
    """The rival runner recomputed each case's digest from the file it read; every one must
    match ARGUS's own, or the comparison is not on the same input."""
    mismatched = [c.case_id for c in cases
                  if rival.get("digests", {}).get(c.case_id) != case_digest(c)]
    return {"cases": len(cases), "mismatched": mismatched, "identical": not mismatched}


def run(corpus_path: Path = CORPUS_PATH, rival_path: Path = RIVAL_RESULTS_PATH,
        cost_repeats: int = 200) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    cases = build_cases(corpus)
    results: dict[str, dict[str, dict[str, Any]]] = {
        name: {c.case_id: runner(c) for c in cases} for name, runner in ARGUS_RUNNERS.items()
    }
    rival = load_rival_results(rival_path)
    for name in RIVALS:
        results[name] = rival["results"][name]
    same = verify_same_input(cases, rival)
    scored = compare(cases, results)
    clean = next(c for c in cases if c.case_id == "clean")
    costs = {**measure_argus_costs(clean, cost_repeats), **rival["costs"]}
    return {
        "capability": "Data-honest cross-asset breadth rotation vs. a silently-dropping reference",
        "general_function": "data-contract validation of a time-series panel feeding a "
                            "computation: completeness, validity, ordering, timeliness and "
                            "sufficiency of the inputs, the call's parameters, and an output "
                            "invariant — refuse rather than emit a silently wrong result",
        "corpus": {
            # Repository-relative and forward-slashed, so the artefact carries no machine's path.
            "path": corpus_path.relative_to(ROOT).as_posix() if corpus_path.is_relative_to(ROOT)
            else corpus_path.name,
            "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "fetched_at": json.loads(corpus_path.read_text(encoding="utf-8"))["fetched_at"],
            "bars_per_symbol": {s: len(rows) for s, rows in corpus.items()},
        },
        "same_input": same,
        "rival_environment": rival.get("environment"),
        "rival_notes": rival.get("notes"),
        "cases": [c.meta() for c in cases],
        "results": results,
        **scored,
        "costs": costs,
        "verdict": verdict(scored["scoreboard"], costs),
        "business_month_divergence": run_business_month_divergence(corpus),
        "scope_statement": SCOPE_STATEMENT,
    }


def verdict(board: Mapping[str, Mapping[str, Any]], costs: Mapping[str, Any]) -> dict[str, Any]:
    """Who won, computed from the scoreboard rather than written by hand."""
    def score(name: str) -> tuple[int, int]:
        b = board[name]
        return (b["cases_handled_correctly"], b["offending_symbols_named"])

    best_general = max(GENERAL_RIVALS, key=score)
    before_vs = score("argus_before")
    after_vs = score("argus_after")
    rival_vs = score(best_general)

    def word(ours: tuple[int, int]) -> str:
        if ours[0] > rival_vs[0]:
            return "argus_wins"
        if ours[0] < rival_vs[0]:
            return "rival_wins"
        return "tie"

    def cost(name: str) -> float | None:
        c = costs.get(name)
        return None if c is None else float(c["median_seconds"])

    return {
        "best_general_rival": best_general,
        "pre_contract_argus_vs_best_general_rival": word(before_vs),
        "argus_after_vs_best_general_rival": word(after_vs),
        "cases_correct": {n: board[n]["cases_handled_correctly"] for n in board},
        "cases_total": next(iter(board.values()))["cases_total"],
        "median_seconds_per_clean_call": {n: cost(n) for n in CONTESTANTS},
    }


SCOPE_STATEMENT = (
    "Measured: how each system handles a fixed corpus of real Bitget history with single, named "
    "contract violations injected, real short-history conditions present in the data as "
    "fetched, and valid controls — refusal, silent acceptance, over-refusal, false positives, "
    "which offending symbols a refusal names, and wall-clock cost. The general-purpose rivals "
    "are configured by us (their schemas and expectation suites are in "
    "scripts/general_rotation_rival.py); each clause is written in the tool's own idiom and "
    "against the same contract ARGUS enforces, and where a tool cannot express a clause natively "
    "that is recorded, not worked around. NOT claimed: that ARGUS is a better general-purpose "
    "validator than pandera or Great Expectations — they validate any frame, profile data and "
    "render reports, none of which ARGUS does; only that on this contract and this corpus the "
    "measured outcomes are as reported. NOT claimed: any forecasting value for the rotation rule. "
    "The corpus is one real snapshot (fetched once and frozen); a different day's panel could "
    "change which real symbols are short."
)


def render(report: Mapping[str, Any]) -> str:
    lines = ["DATA-HONEST ROTATION vs general-purpose data validators (same real corpus)\n"]
    board = report["scoreboard"]
    for name in CONTESTANTS:
        if name not in board:
            continue
        b = board[name]
        cost = report["costs"].get(name, {}).get("median_seconds")
        lines.append(
            f"  {name:20} correct {b['cases_handled_correctly']:2}/{b['cases_total']}  "
            f"refused {b['refused']:2}/{b['must_refuse_cases']}  "
            f"silent {len(b['silent_accepts']):2}  over-refused {len(b['over_refusals'])}  "
            f"false+ {len(b['false_positives'])}  named {b['offending_symbols_named']}/"
            f"{b['offending_symbols_total']}  "
            f"{'' if cost is None else f'{cost * 1000:.3f}ms/call'}")
    v = report["verdict"]
    lines.append(f"\n  best general rival: {v['best_general_rival']}")
    lines.append(f"  pre-contract ARGUS vs it: {v['pre_contract_argus_vs_best_general_rival']}")
    lines.append(f"  ARGUS now vs it:          {v['argus_after_vs_best_general_rival']}")
    lines.append(f"  same input: {report['same_input']['identical']}")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--freeze", action="store_true", help="re-fetch and freeze the corpus")
    parser.add_argument("--wire", type=Path, help="write the rival case file here and stop")
    parser.add_argument("--rival-python", type=Path,
                        help="run the rival runner with this interpreter before scoring")
    args = parser.parse_args()
    if args.freeze:
        freeze_corpus()
    cases = build_cases(load_corpus())
    if args.wire:
        print(write_wire(cases, args.wire))
        return 0
    if args.rival_python:
        wire = RIVAL_RESULTS_PATH.with_suffix(".cases.json")
        write_wire(cases, wire)
        try:
            run_rivals(args.rival_python, wire)
        finally:
            wire.unlink(missing_ok=True)
    report = run()
    print(render(report))
    undefined = artefact.write(ARTEFACT_PATH, report)
    print(f"\nwritten to {ARTEFACT_PATH}" + (f" ({len(undefined)} undefined -> null)"
                                             if undefined else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ARTEFACT_PATH",
    "CONTESTANTS",
    "CORPUS_PATH",
    "DIVERGENCE_CUTOFFS",
    "GENERAL_RIVALS",
    "RISK",
    "RIVALS",
    "RIVAL_RESULTS_PATH",
    "SAFE",
    "SCOPE_STATEMENT",
    "Case",
    "build_cases",
    "case_digest",
    "classify",
    "compare",
    "load_corpus",
    "load_rival_results",
    "main",
    "measure_argus_costs",
    "render",
    "run",
    "run_argus_after",
    "run_argus_before",
    "run_business_month_divergence",
    "verdict",
    "verify_same_input",
    "write_wire",
]
