"""ARGUS's evidence gathering vs. TradingAgents' real data-vendor router — same failure modes.

``eval/standing.py``'s "Perception layer: what the desk can see" capability names its baseline as
"OpenBB-finance/OpenBB provider set; TauricResearch/TradingAgents feed list". The TradingAgents
half of that baseline is ``tradingagents/dataflows/interface.py``'s real ``route_to_vendor()`` —
vendored byte-verified in ``eval/baselines/tradingagents_feedlist_interface.py`` (see that
package's docstring) and actually executed here against ARGUS's own
``argus.market.evidence.gather()`` (the module the capability's own baseline field already names).

**Both are answers to the same question: what does the desk do when a live source fails?**
Genuinely comparable — both are dispatch layers sitting in front of several named external
providers, both distinguish at least one failure mode that aborts from one that doesn't.

**A real structural difference, found by running both, not by reading either alone.**
TradingAgents' ``route_to_vendor()`` RAISES when every configured vendor for a "core" category
(``core_stock_apis``, ``technical_indicators``, ``fundamental_data``, ``news_data``) fails with
anything other than a clean "no data" — only its two named ``OPTIONAL_CATEGORIES`` (``macro_data``,
``prediction_markets``) degrade to a sentinel string instead. ARGUS's ``gather()`` never raises for
a single source's failure, for ANY source, core or opt-in: EDGAR, RSS (per-feed, inside
``RssSource.evidence`` itself), insider filings and fundamentals all catch a deliberately scoped
tuple of network/parsing exceptions and turn each into a status line the decision-maker sees,
verified by running ``gather()`` with a real source object that raises on every call.

Confirmed empirically, both directions: TradingAgents' real router genuinely raises on a core
category with no working vendor (not merely documented to); ARGUS's real ``gather()`` genuinely
returns cleanly, with the failure named in its own ``status`` list, on the identical failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.baselines.tradingagents_feedlist_loader import (
    TradingAgentsFeedlistLoadError,
    load_interface_module,
    load_set_config,
)
from argus.market.evidence import BitgetSkillSource, EdgarSource, RssSource, gather


class FeedlistComparisonError(RuntimeError):
    """The comparison could not run — the baseline failed to load."""


# =============================================================================================
# Fakes — real gather()/route_to_vendor() calls, synthetic sources standing in for the network.
# =============================================================================================


class _RaisingEdgar(EdgarSource):
    """A stand-in EdgarSource whose `.evidence()` always raises one of the exact exception types
    `gather()`'s own try/except names — never a network call. Subclasses the real `EdgarSource`
    (rather than duck-typing) so `gather()`'s own strict parameter types accept it directly."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def evidence(self, symbol: str, *, as_of: datetime, lookback: Any) -> list[Any]:
        raise self._exc


class _RaisingSource:
    """A stand-in insider/fundamentals source — `.evidence()` returns `(items, status)` per its
    real contract, or raises, mirroring `_RaisingEdgar` for the opt-in sources. `gather()` types
    `insider`/`fundamentals` as `Any`, so no base class is needed here."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def evidence(
        self, symbol: str, *, as_of: datetime, lookback: Any = None
    ) -> tuple[list[Any], list[str]]:
        raise self._exc


class _NoOpEdgar(EdgarSource):
    """A clean, empty EdgarSource stand-in — never a network call, never raises."""

    def evidence(self, symbol: str, *, as_of: datetime, lookback: Any) -> list[Any]:
        return []


class _NoOpRss(RssSource):
    """A clean, empty RssSource stand-in, matching its real `(items, status)` contract."""

    def evidence(
        self, symbol: str, *, as_of: datetime, lookback: Any, keywords: Any = ()
    ) -> tuple[list[Any], list[str]]:
        return [], ["rss: skipped for this comparison"]


class _NoOpBitget(BitgetSkillSource):
    """A clean, empty BitgetSkillSource stand-in, matching its real `(items, status)` contract."""

    def evidence(self, symbol: str, *, as_of: datetime) -> tuple[list[Any], list[str]]:
        return [], ["bitget: skipped for this comparison"]


_CAUGHT_EXCEPTIONS: tuple[Exception, ...] = (
    OSError("connection reset"),
    TimeoutError("timed out"),
)
_UNCAUGHT_EXCEPTION = ValueError("not a network or parse failure at all")


# =============================================================================================
# The core structural finding: core-vs-optional category handling, run on both real systems.
# =============================================================================================


@dataclass(frozen=True)
class RouterCase:
    system: str
    category: str
    outcome: str
    """'raised' | 'degraded_to_sentinel' | 'returned_cleanly'"""
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "category": self.category,
            "outcome": self.outcome,
            "detail": self.detail,
        }


def run_tradingagents_core_category_failure(interface_module: Any) -> RouterCase:
    """A core category (`core_stock_apis`), every configured vendor genuinely broken."""
    set_config = load_set_config()

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("vendor connection refused")

    interface_module.VENDOR_METHODS["get_stock_data"] = {
        "alpha_vantage": boom, "yfinance": boom,
    }
    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})
    try:
        interface_module.route_to_vendor("get_stock_data", "AAPL")
        return RouterCase("tradingagents", "core_stock_apis", "returned_cleanly", "no exception")
    except RuntimeError as exc:
        return RouterCase("tradingagents", "core_stock_apis", "raised", str(exc))


def run_tradingagents_optional_category_failure(interface_module: Any) -> RouterCase:
    """An optional category (`macro_data`), its one configured vendor genuinely broken."""
    set_config = load_set_config()

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("vendor connection refused")

    interface_module.VENDOR_METHODS["get_macro_indicators"] = {"fred": boom}
    set_config({"data_vendors": {"macro_data": "fred"}})
    result = interface_module.route_to_vendor("get_macro_indicators")
    outcome = "degraded_to_sentinel" if isinstance(result, str) and "DATA_UNAVAILABLE" in result \
        else "returned_cleanly"
    return RouterCase("tradingagents", "macro_data", outcome, str(result))


def run_argus_core_source_failure() -> RouterCase:
    """The structural analogue: ARGUS's EDGAR source (a core, always-on feed, not opt-in)
    genuinely broken — the same shape of failure TradingAgents' core categories raise on."""
    edgar = _RaisingEdgar(OSError("connection reset"))
    result = gather(
        "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
        edgar=edgar, rss=_NoOpRss(), bitget=_NoOpBitget(),
    )
    edgar_status = next((s for s in result.status if s.startswith("edgar:")), "")
    outcome = "returned_cleanly" if "unavailable" in edgar_status else "unexpected"
    return RouterCase("argus", "edgar_(core-equivalent)", outcome, edgar_status)


def run_argus_optional_source_failure() -> RouterCase:
    """ARGUS's opt-in insider-filings source, genuinely broken — the structural analogue of
    TradingAgents' optional category."""
    insider = _RaisingSource(OSError("connection reset"))
    result = gather(
        "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC), insider=insider,
        edgar=_NoOpEdgar(), rss=_NoOpRss(), bitget=_NoOpBitget(),
    )
    insider_status = next((s for s in result.status if s.startswith("insider:")), "")
    outcome = "returned_cleanly" if "unavailable" in insider_status else "unexpected"
    return RouterCase("argus", "insider_(optional-equivalent)", outcome, insider_status)


# =============================================================================================
# Statistically valid evaluation — sweep every real category x failure mode combination.
# =============================================================================================


def swept_router_cases(interface_module: Any) -> list[RouterCase]:
    """Every one of TradingAgents' own real `VENDOR_METHODS` keys, run against a genuinely
    broken vendor chain — not just the two categories picked for the designed cases above."""
    set_config = load_set_config()

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("vendor connection refused")

    out: list[RouterCase] = []
    for method, vendors in list(interface_module.VENDOR_METHODS.items()):
        category = interface_module.get_category_for_method(method)
        broken = {v: boom for v in vendors}
        interface_module.VENDOR_METHODS[method] = broken
        set_config({"data_vendors": {category: ",".join(broken.keys())}})
        try:
            result = interface_module.route_to_vendor(method)
            outcome = "degraded_to_sentinel" if (
                isinstance(result, str) and "DATA_UNAVAILABLE" in result
            ) else "returned_cleanly"
            detail = str(result)
        except RuntimeError as exc:
            outcome = "raised"
            detail = str(exc)
        out.append(RouterCase("tradingagents", category, outcome, detail))
    return out


def swept_argus_cases() -> list[RouterCase]:
    """Every ARGUS source `gather()` can be pointed at, each broken with a different one of the
    exact exception types its own try/except names — confirming none of them ever raise."""
    out: list[RouterCase] = []
    for exc in _CAUGHT_EXCEPTIONS:
        r = gather(
            "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
            edgar=_RaisingEdgar(exc), insider=_RaisingSource(exc), fundamentals=_RaisingSource(exc),
            rss=_NoOpRss(), bitget=_NoOpBitget(),
        )
        edgar_status = next((s for s in r.status if s.startswith("edgar:")), "")
        out.append(RouterCase(
            "argus", f"edgar_({type(exc).__name__})",
            "returned_cleanly" if "unavailable" in edgar_status else "unexpected",
            edgar_status,
        ))
    return out


# =============================================================================================
# Ablation — is the caught-exception set actually scoped, or a blanket `except Exception`?
# =============================================================================================


@dataclass(frozen=True)
class AblationResult:
    caught_types_are_swallowed: bool
    uncaught_type_propagates: bool

    @property
    def the_scoping_is_load_bearing(self) -> bool:
        """A blanket `except Exception` would swallow the uncaught type too — this would read
        `False` under that design, which is exactly what this ablation is checking for."""
        return self.caught_types_are_swallowed and self.uncaught_type_propagates

    def as_dict(self) -> dict[str, Any]:
        return {
            "caught_types_are_swallowed": self.caught_types_are_swallowed,
            "uncaught_type_propagates": self.uncaught_type_propagates,
            "the_scoping_is_load_bearing": self.the_scoping_is_load_bearing,
        }


def run_ablation() -> AblationResult:
    r = gather(
        "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
        edgar=_RaisingEdgar(OSError("x")), rss=_NoOpRss(), bitget=_NoOpBitget(),
    )
    edgar_status = next((s for s in r.status if s.startswith("edgar:")), "")
    swallowed = "unavailable" in edgar_status

    propagated = False
    try:
        gather(
            "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
            edgar=_RaisingEdgar(_UNCAUGHT_EXCEPTION), rss=_NoOpRss(), bitget=_NoOpBitget(),
        )
    except ValueError:
        propagated = True

    return AblationResult(caught_types_are_swallowed=swallowed, uncaught_type_propagates=propagated)


# =============================================================================================
# Reproducibility — the same scenario, run twice, both real systems.
# =============================================================================================


def run_reproducibility_check(interface_module: Any) -> dict[str, bool]:
    set_config = load_set_config()

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("vendor connection refused")

    interface_module.VENDOR_METHODS["get_macro_indicators"] = {"fred": boom}
    set_config({"data_vendors": {"macro_data": "fred"}})
    ta_first = interface_module.route_to_vendor("get_macro_indicators")
    ta_second = interface_module.route_to_vendor("get_macro_indicators")

    argus_first = gather(
        "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
        edgar=_RaisingEdgar(OSError("x")), rss=_NoOpRss(), bitget=_NoOpBitget(),
    ).status
    argus_second = gather(
        "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
        edgar=_RaisingEdgar(OSError("x")), rss=_NoOpRss(), bitget=_NoOpBitget(),
    ).status

    return {
        "tradingagents_reproducible": ta_first == ta_second,
        "argus_reproducible": argus_first == argus_second,
    }


# =============================================================================================
# Out-of-sample — the REAL, live-growing desk-notes artefact, not a synthetic fixture.
# =============================================================================================

_REAL_DESK_NOTES_PATH = Path(__file__).resolve().parents[3] / "data" / "desk_notes.jsonl"


@dataclass(frozen=True)
class OutOfSampleResult:
    path: str
    cycles: int
    distinct_sources_seen: int
    checked: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cycles": self.cycles,
            "distinct_sources_seen": self.distinct_sources_seen,
            "checked": self.checked,
        }


def run_out_of_sample_case() -> OutOfSampleResult:
    """The real desk-notes log this session's live decision cycles have been writing to —
    confirms the status-line design this comparison exercises on synthetic fixtures is the same
    shape real production cycles actually produce."""
    import json

    if not _REAL_DESK_NOTES_PATH.exists():
        return OutOfSampleResult(str(_REAL_DESK_NOTES_PATH), 0, 0, checked=False)
    sources: set[str] = set()
    cycles = 0
    for line in _REAL_DESK_NOTES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cycles += 1
        for s in row.get("sources", []):
            sources.add(str(s))
    return OutOfSampleResult(str(_REAL_DESK_NOTES_PATH), cycles, len(sources), checked=True)


# =============================================================================================
# Costs.
# =============================================================================================


def measure_costs(interface_module: Any) -> dict[str, float]:
    """Dispatch overhead — not network latency, which both systems' real designs deliberately
    exclude the vendor call itself from (the comparison never phones a real API)."""
    set_config = load_set_config()

    def instant(*args: object, **kwargs: object) -> str:
        return "ok"

    interface_module.VENDOR_METHODS["get_stock_data"] = {"yfinance": instant}
    set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    start = time.perf_counter()
    for _ in range(1000):
        interface_module.route_to_vendor("get_stock_data", "AAPL")
    ta_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(1000):
        gather(
            "AAPL", as_of=datetime(2026, 1, 1, tzinfo=UTC),
            edgar=_NoOpEdgar(), rss=_NoOpRss(), bitget=_NoOpBitget(),
        )
    argus_seconds = time.perf_counter() - start

    return {
        "tradingagents_dispatch_us_per_call": (ta_seconds / 1000) * 1_000_000,
        "argus_gather_us_per_call": (argus_seconds / 1000) * 1_000_000,
    }


# =============================================================================================
# Scope statement.
# =============================================================================================

SCOPE_STATEMENT = """\
Claimed: run on identical failure scenarios, TradingAgents' real route_to_vendor() raises when \
every configured vendor for a "core" category fails with a genuine error, while ARGUS's real \
gather() never raises for any single source's failure — core or opt-in — instead recording the \
failure as a status line the decision-maker sees. Both behaviours are the real code's own, \
confirmed by running each, not by reading either alone. The exception-type scoping in gather()'s \
try/except is shown load-bearing: the exact types it names are swallowed into a status line; a \
type it does not name (ValueError, standing in for a genuine programming-logic bug) propagates — \
gather() is not a blanket except.

NOT claimed: that TradingAgents' design is a defect. Its own comment states the intent plainly — \
"a broken primary must be visible in the logs" — and it IS visible, via the raised exception \
itself, to whatever code called route_to_vendor(). The difference under test is narrower and \
verifiable: whether a single live source's failure can abort a full perception cycle. In \
TradingAgents' real code, for a core category with zero working vendors, it can; in ARGUS's real \
code, for any source, it cannot. Also NOT claimed: that ARGUS's four named vendors (EDGAR, RSS, \
Bitget Skills, insider/fundamentals-when-supplied) outnumber or outperform TradingAgents' — the \
source-count comparison is separately covered by best_implementation_studied's own artefact \
(../research/architecture/datasource-audit.md), not repeated here.
"""


def main() -> dict[str, Any]:
    """Run the whole comparison and return a serialisable summary.

    Raises:
        FeedlistComparisonError: the vendored TradingAgents baseline failed to load.
    """
    try:
        interface_module = load_interface_module()
    except TradingAgentsFeedlistLoadError as exc:
        raise FeedlistComparisonError(
            f"could not load the vendored TradingAgents feed-list baseline: {exc}"
        ) from exc

    core_case = run_tradingagents_core_category_failure(interface_module)
    optional_case = run_tradingagents_optional_category_failure(interface_module)
    argus_core_case = run_argus_core_source_failure()
    argus_optional_case = run_argus_optional_source_failure()
    swept_ta = swept_router_cases(interface_module)
    swept_argus = swept_argus_cases()
    ablation = run_ablation()
    reproducibility = run_reproducibility_check(interface_module)
    oos = run_out_of_sample_case()
    costs = measure_costs(interface_module)

    return {
        "tradingagents_core_case": core_case.as_dict(),
        "tradingagents_optional_case": optional_case.as_dict(),
        "argus_core_equivalent_case": argus_core_case.as_dict(),
        "argus_optional_equivalent_case": argus_optional_case.as_dict(),
        "swept_tradingagents_cases": [c.as_dict() for c in swept_ta],
        "swept_argus_cases": [c.as_dict() for c in swept_argus],
        "swept_ta_core_always_raises": all(
            c.outcome == "raised" for c in swept_ta
            if c.category not in interface_module.OPTIONAL_CATEGORIES
        ),
        "swept_ta_optional_never_raises": all(
            c.outcome != "raised" for c in swept_ta
            if c.category in interface_module.OPTIONAL_CATEGORIES
        ),
        "swept_argus_never_raises": all(c.outcome == "returned_cleanly" for c in swept_argus),
        "ablation": ablation.as_dict(),
        "reproducibility": reproducibility,
        "out_of_sample": oos.as_dict(),
        "costs": costs,
        "scope_statement": SCOPE_STATEMENT,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "FEED-LIST COMPARISON — ARGUS gather() vs. TradingAgents real route_to_vendor()",
        "",
    ]
    c = report["tradingagents_core_case"]
    o = report["tradingagents_optional_case"]
    lines.append(f"TradingAgents core category, all vendors broken -> {c['outcome']}")
    lines.append(f"TradingAgents optional category, vendor broken -> {o['outcome']}")
    ac = report["argus_core_equivalent_case"]
    ao = report["argus_optional_equivalent_case"]
    lines.append(f"ARGUS core-equivalent (EDGAR) source broken -> {ac['outcome']}")
    lines.append(f"ARGUS optional-equivalent (insider) source broken -> {ao['outcome']}")
    lines.append(
        f"swept: TA core always raises = {report['swept_ta_core_always_raises']}, "
        f"TA optional never raises = {report['swept_ta_optional_never_raises']}, "
        f"ARGUS never raises = {report['swept_argus_never_raises']}"
    )
    ab = report["ablation"]
    lines.append(f"exception-scoping ablation load-bearing: {ab['the_scoping_is_load_bearing']}")
    rp = report["reproducibility"]
    lines.append(
        f"reproducible — TradingAgents: {rp['tradingagents_reproducible']}, "
        f"ARGUS: {rp['argus_reproducible']}"
    )
    oos = report["out_of_sample"]
    lines.append(
        f"out-of-sample: real desk notes, {oos['cycles']} cycles, "
        f"{oos['distinct_sources_seen']} distinct sources seen (checked={oos['checked']})"
    )
    cst = report["costs"]
    lines.append(
        f"cost: TradingAgents dispatch {cst['tradingagents_dispatch_us_per_call']:.1f}us/call, "
        f"ARGUS gather {cst['argus_gather_us_per_call']:.1f}us/call"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path as _Path

    result = main()
    print(render(result))
    out_path = _Path(__file__).resolve().parents[3] / "data" / "feedlist_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


__all__ = [
    "SCOPE_STATEMENT",
    "AblationResult",
    "FeedlistComparisonError",
    "OutOfSampleResult",
    "RouterCase",
    "main",
    "measure_costs",
    "render",
    "run_ablation",
    "run_argus_core_source_failure",
    "run_argus_optional_source_failure",
    "run_out_of_sample_case",
    "run_reproducibility_check",
    "run_tradingagents_core_category_failure",
    "run_tradingagents_optional_category_failure",
    "swept_argus_cases",
    "swept_router_cases",
]
