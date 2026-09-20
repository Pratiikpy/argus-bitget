"""Population RCT (Track 2 §13.7a) — does each Constitution gate earn its place, measured.

The plan's own table entry (`Activity/08_TRADING_OS_PLAN.md`) names this: *"N gate-ablated ARGUS
variants trading the identical book in parallel"*, scored on risk-violation rate and
incremental-value-over-baseline, *as a controlled experiment*. `risk_proof.json`
(`eval/riskproof.py`) proves the eight Foundation 5 gates are internally consistent over a swept
state space; it does not compare against a variant missing one of them, which is what
"incremental value" actually means.

**Read before written, per standing rule #3.** `eval/ablation.py` already exists and already is
the right instrument: :func:`argus.eval.ablation.deterministic` replays a sequence of frames
through a ``with_component``/``without_component`` pair and reports an exact, statistics-free
count of decisions that moved — correct for the Constitution because it is a pure function of the
frame, unlike an LLM analyst, which needs :func:`argus.eval.ablation.paired`'s sign test instead.
`eval/ablations.py` already applies the same engine to the analyst-selection layer; this is the
first application of it to the risk-gate chain. Building a second comparison engine here would
have been the exact redundancy standing rule #3 exists to prevent.

**The data source is `agents.desk.TradingDesk.run`'s `ablated_constitutions` parameter** (added
alongside this module, same session): one real `attacked` intent, N gate-ablated
`ConstitutionPolicy` variants ruled against it for the cost of N cheap deterministic calls, never
N more calls against a hackathon key with a stated limited balance.

**Ablating one gate without disturbing the others is not always "raise one cap".** Eight of the
nine dimensions have an independent, already-``None``-or-huge-means-inert lever
(`max_position_notional`, `max_gross_exposure_notional`, `max_signed_exposure_notional`,
`max_margin_usage_ratio`, `factor_exposure_limits`, `max_scenario_loss_pct`,
`max_liquidation_cost_bps`, `min_symbol_realized_pnl`). ``hedge_integrity`` is the one structural
exception: it has no numeric cap to raise, because it fires on a *relationship* (a
`desk.book.HedgeLink`) rather than a threshold. Ablating it means a book identical in every
position and margin fact, with its hedge links removed — :func:`_ablate` builds that copy rather
than reaching for a field that does not exist.

**"Risk-violation rate" is computed from the sign of each change, not just its existence.** Every
gate here can only narrow a proposal, never widen it (the Constitution's own invariant, guarded
structurally in `decision.verdicts.apply_constraint`), so a variant with one gate removed can only
approve a size greater than or equal to the full policy's — never less. A
:class:`~argus.eval.ablation.Change` where the ablated size is strictly larger than the baseline's
is therefore not merely "a different answer", it is the concrete case the gate existed to prevent:
a decision this desk would have taken had that one protection not been there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import ConstitutionRuling, Intent
from argus.desk.book import Book
from argus.eval.ablation import Change, DeterministicResult, deterministic
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionState

DIMENSIONS: tuple[str, ...] = (
    "order_size",
    "gross_exposure",
    "signed_exposure",
    "hedge_integrity",
    "margin_usage",
    "factor_exposure",
    "scenario_loss",
    "liquidation_cost",
    "per_symbol_underperformance",
)
"""Foundation 5's eight safety dimensions, plus the ninth (`per_symbol_underperformance`, added
2026-09-15 alongside `desk.book.Book.symbol_realized_pnl_since`) — the ARGUS-native counterpart to
freqtrade's ``LowProfitPairs`` — in the order named in the product plan / added to the chain."""

_HUGE = Decimal("1e30")
"""Effectively-infinite ceiling for ablating a notional/ratio cap. Not ``Decimal("Infinity")``:
some downstream arithmetic (``f"{x:.1f}"`` formatting in the gate's own reason strings) raises on
a literal infinity, and a cap ten orders of magnitude above anything `QUANTITIES` sweeps is
functionally unreachable without that fragility."""


def _book_without_hedge_links(book: Any) -> Any:
    """A copy of ``book`` with every :class:`~argus.desk.book.HedgeLink` removed, everything else
    (positions, balances, reservations, the margin snapshot) identical.

    `Book` is not a frozen dataclass, so `dataclasses.replace` does not apply — its state is
    plain mutable attributes, copied here rather than constructed through fills a second time,
    which would risk the copy silently drifting from the original as `Position.apply_fill`'s
    logic evolves.
    """
    clone = Book()
    clone.positions = dict(book.positions)
    clone.balances = dict(book.balances)
    clone.reservations = dict(book.reservations)
    clone.margin = book.margin
    clone.hedge_links = []
    return clone


def _ablate(dimension: str, baseline: ConstitutionPolicy) -> ConstitutionPolicy:
    """One policy identical to ``baseline`` except that ``dimension`` cannot bind.

    Raises on an unknown dimension rather than silently returning ``baseline`` unchanged — a typo
    here would otherwise report a gate as "inert" because it was never actually ablated, the exact
    false negative this module exists to prevent.
    """
    if dimension == "order_size":
        return replace(baseline, max_position_notional=_HUGE)
    if dimension == "gross_exposure":
        return replace(baseline, max_gross_exposure_notional=_HUGE)
    if dimension == "signed_exposure":
        return replace(baseline, max_signed_exposure_notional=_HUGE)
    if dimension == "hedge_integrity":
        book = baseline.book
        return replace(baseline, book=None if book is None else _book_without_hedge_links(book))
    if dimension == "margin_usage":
        return replace(baseline, max_margin_usage_ratio=_HUGE)
    if dimension == "factor_exposure":
        return replace(baseline, factor_exposure_limits={})
    if dimension == "scenario_loss":
        return replace(baseline, max_scenario_loss_pct=None)
    if dimension == "liquidation_cost":
        return replace(baseline, max_liquidation_cost_bps=None)
    if dimension == "per_symbol_underperformance":
        return replace(baseline, min_symbol_realized_pnl=None)
    raise ValueError(f"unknown dimension {dimension!r}; known: {', '.join(DIMENSIONS)}")


def ablated_variants(
    baseline: ConstitutionPolicy, *, dimensions: Sequence[str] = DIMENSIONS,
) -> dict[str, ConstitutionPolicy]:
    """One gate-ablated variant per dimension, keyed by name — the ``ablated_constitutions``
    argument :meth:`agents.desk.TradingDesk.run` expects."""
    return {dimension: _ablate(dimension, baseline) for dimension in dimensions}


@dataclass(frozen=True, slots=True)
class GateValue(DeterministicResult):
    """A :class:`~argus.eval.ablation.DeterministicResult` plus the risk-relevant reading of it —
    "did the answer change" is not the same claim as "did removing the gate approve more risk",
    and a gate could in principle change an answer's *reason* without widening its size (a
    different ceiling could bind instead, at the same quantity)."""

    risk_violations: int = 0
    """Frames where the ablated variant approved a strictly larger quantity than the full policy
    did — the concrete case this gate existed to prevent, not merely "a different answer"."""

    @property
    def violation_rate(self) -> float:
        return self.risk_violations / self.frames if self.frames else 0.0

    def as_dict(self) -> dict[str, Any]:
        # Explicit two-argument `super()`, not the zero-arg form: `@dataclass(slots=True)`
        # rebuilds the class object it decorates, and the zero-arg form's implicit `__class__`
        # cell can end up pointing at the pre-rebuild class when both this class and its base use
        # `slots=True` — `super(type, obj): obj must be an instance or subtype of type` at
        # runtime, caught by this module's own tests before it shipped.
        out = super(GateValue, self).as_dict()
        out["risk_violations"] = self.risk_violations
        out["violation_rate"] = round(self.violation_rate, 6)
        return out


def population_rct(
    frames: Sequence[tuple[str, Intent]],
    *,
    session: SessionState,
    hedges: HedgeabilitySurface,
    baseline: ConstitutionPolicy | None = None,
    dimensions: Sequence[str] = DIMENSIONS,
) -> dict[str, GateValue]:
    """Rule every frame through the full policy and each single-gate-ablated variant, and report,
    per dimension, how often the answer changed and how often that change was strictly riskier.

    ``frames`` are ``(frame_id, intent)`` — real `attacked` intents from settled cycles, or
    synthetic ones for a targeted sweep; :func:`deterministic` does not care which, only that the
    same intent is ruled twice.
    """
    active = baseline or ConstitutionPolicy()
    variants = ablated_variants(active, dimensions=dimensions)

    results: dict[str, GateValue] = {}
    def _ruler(policy: ConstitutionPolicy) -> Callable[[Intent], ConstitutionRuling]:
        def _rule(intent: Intent) -> ConstitutionRuling:
            return policy.rule(intent, session=session, hedges=hedges)
        return _rule

    with_component = _ruler(active)
    for dimension in dimensions:
        det = deterministic(
            frames,
            component=dimension,
            with_component=with_component,
            without_component=_ruler(variants[dimension]),
        )
        violations = sum(
            1 for change in det.changes
            if _quantity_of(change.without_component) > _quantity_of(change.with_component)
        )
        results[dimension] = GateValue(
            component=det.component, frames=det.frames, changes=det.changes,
            risk_violations=violations,
        )
    return results


def _quantity_of(ruling: Any) -> Decimal:
    ruling_: ConstitutionRuling = ruling
    return ruling_.resulting_intent.quantity


REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "population_rct.json"
"""Where :func:`main` writes. Matches `eval/riskproof.py`'s `REPORT_PATH` convention."""


def report_from_records(
    records: Sequence[dict[str, Any]], *, dimensions: Sequence[str] = DIMENSIONS,
) -> dict[str, GateValue]:
    """Aggregate `paper.runner._write_risk_record`'s persisted ``ablated_rulings`` into the same
    per-dimension shape :func:`population_rct` computes live — no re-ruling, since the primary
    ruling's ``quantity_after`` and each variant's are already on the row.

    **Older records carry no ``ablated_rulings`` key at all, and that is read as absent, never as
    zero.** Population RCT started accumulating the day this module shipped (2026-09-15); a row
    from before that is silently missing the key, and counting it as "this gate did not fire on
    that row" would misstate the sample size the same way a dropped `chain_schema` once did
    (`eval/autopsy.py`'s schema-1/2 distinction). Rows without the key are excluded from every
    dimension's ``frames`` count, not folded in as a false negative.
    """
    results: dict[str, GateValue] = {}
    for dimension in dimensions:
        changes: list[Change] = []
        frames = 0
        violations = 0
        for row in records:
            ablated = row.get("ablated_rulings")
            if not isinstance(ablated, dict) or dimension not in ablated:
                continue
            frames += 1
            variant = ablated[dimension]
            baseline_fields = {
                "binding_constraint": row.get("binding_constraint"),
                "quantity_after": row.get("quantity_after"),
            }
            if variant != baseline_fields:
                changes.append(Change(
                    frame_id=str(row.get("seq", "?")),
                    with_component=baseline_fields, without_component=variant,
                ))
                try:
                    if Decimal(str(variant["quantity_after"])) > Decimal(
                        str(baseline_fields["quantity_after"])
                    ):
                        violations += 1
                except (KeyError, TypeError, InvalidOperation):
                    # A malformed row still counts as a frame (the `Change` above is real — the
                    # constraint names genuinely differed) but cannot honestly contribute a
                    # violation count with a quantity that does not parse. Caught by this
                    # module's own tests before it could raise on real, accumulated live data.
                    pass
        results[dimension] = GateValue(
            component=dimension, frames=frames, changes=tuple(changes),
            risk_violations=violations,
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    from argus.eval.riskaudit import read_records
    from argus.paper.runner import RISK_PATH

    parser = argparse.ArgumentParser(
        description="Population RCT: how often did each Foundation 5 gate change the decision, "
        "and how often was that change strictly riskier — aggregated from accumulated live cycles."
    )
    parser.add_argument("--records", type=Path, default=RISK_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = read_records(args.records)
    results = report_from_records(records)

    def _status(r: GateValue) -> str:
        # `.inert` alone does not distinguish "0 frames, never tested" from "many frames, always
        # clean" — that is `eval.ablation.DeterministicResult`'s own deliberate, tested contract
        # (`tests/test_ablations.py::...frames == 0 and ... inert`), and `frames` is preserved
        # alongside it for exactly this disambiguation. Spelled out explicitly here so a JSON
        # consumer never has to re-derive it, rather than reading a bare "inert": true the same
        # way whether the gate was tested and found nothing, or never tested at all.
        if r.frames == 0:
            return "UNTESTED"
        return "INERT" if r.inert else "ACTIVE"

    payload = {
        "generated_from": str(args.records),
        "rows_read": len(records),
        "dimensions": {
            d: {**r.as_dict(), "status": _status(r)} for d, r in results.items()
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"POPULATION RCT — {len(records)} row(s) read from {args.records}")
    for dimension, result in results.items():
        if result.frames == 0:
            print(f"  {dimension:18} UNTESTED — no row carries this dimension's ablation yet")
            continue
        tag = "inert" if result.inert else f"{result.violation_rate:.1%} violation rate"
        print(
            f"  {dimension:18} {result.frames:4} frame(s), "
            f"{len(result.changes):3} changed, {tag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIMENSIONS",
    "REPORT_PATH",
    "GateValue",
    "ablated_variants",
    "main",
    "population_rct",
    "report_from_records",
]
