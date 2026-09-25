"""Is the risk layer right? Certified gate by gate, then broken on purpose to prove the check works.

`risk/certify.py` is a second implementation of every risk gate, written from the gates' stated
specification and sharing no code with them (PlanBench's VAL pattern; MIT). This module is the
measurement that makes it evidence rather than a claim, in four parts:

1. **Agreement on every order stream.** The certifier replays every order `eval/risk_shadow.py`
   knows about — the two orders the live cycle actually sent to the venue gate, 6,156
   counterfactual sizes at real recorded decision points on real venue rows, the S13 self-check's
   190,944 swept orders and the edge probes — through the guard, and certifies each ruling, its
   whole :class:`~argus.execution.guard.PermissionCheckTrace` and its rate-window effect. Run on the
   current guard, on the guard as it stood this morning (before S20) and on the older copy under
   ``deploy/api`` — so the certifier's findings on the older versions are measured, not described.
2. **The recorded decision chain.** Every ledger decision is replayed through the live path's order
   of gates — the Constitution, the protocol, the venue gate, the ledger — as preconditions on the
   records each one left. Scored against the one ground truth the project has for this: the rows a
   human audit already found and voided (`paper/corrections.py`).
3. **The other gates.** The circuit breaker and its transitions, position sizing, the session
   throttle and the risk-mode stack, each over a grid built around its own thresholds.
4. **Mutation testing.** Each gate is broken, one at a time — a comparison flipped, a threshold
   moved, a branch disabled, a trace event mislabelled — by editing a copy of its source, loading
   the copy beside the real module, and running the same cases until the certifier objects. A
   mutant the certifier never flags is reported as **survived**, with the input space it survived.
   One mutant is included that is *equivalent by construction* (it cannot change any output); it is
   expected to survive, and it is checked that it does, so the harness is not simply flagging
   everything it touches.

**What this does not prove.** The certifier is only as good as its specification: a rule the
specification leaves out is a rule neither side checks. Sizing and the throttle are certified on
the figures they report (the calibration error, the path volatility), which are measurements
rather than gates and are not re-derived here. The Constitution's own rules are out of scope —
`eval/riskproof.py` sweeps those — and appear here only as the first step of the recorded chain.
Zero Qwen calls: nothing here reaches a model.

    python -m argus.eval.risk_certify
"""

from __future__ import annotations

import itertools
import sys
import types
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval import artefact, risk_shadow
from argus.eval.risk_shadow import (
    BASELINES,
    CURRENT_CIRCUIT,
    CURRENT_GUARD,
    NOW,
    Intent,
    Run,
    book_grid,
    load_engine,
    load_source,
    rows_for,
    run_guard,
)
from argus.risk import certify as c

ROOT = risk_shadow.ROOT
REPORT_PATH = ROOT / "data" / "risk_certify.json"
SRC = ROOT / "src" / "argus"
SOURCES = {
    "guard": CURRENT_GUARD,
    "circuit": CURRENT_CIRCUIT,
    "sizing": SRC / "risk" / "sizing.py",
    "session_risk": SRC / "risk" / "session_risk.py",
    "modes": SRC / "risk" / "modes.py",
}
EXAMPLES = 8
"""Certificates kept per stream and outcome. Counts are always complete; examples are a sample."""


def _dec(raw: str | None) -> Decimal | None:
    return None if raw is None else Decimal(raw)


# --- 1. the venue guard ---------------------------------------------------------------------------


def certify_run(run: Run, row: dict[str, Any] | None) -> c.Certificate:
    """Certify one guard ruling, its trace and its rate-window effect, from the venue's raw row."""
    intent = run.intent
    seen = c.raised(run.error) if run.error is not None else c.observe_venue(run.ruling)
    window = None if run.stamps_before is None else c.Window(
        run.stamps_before, run.max_orders, run.window_seconds, NOW)
    order = c.VenueOrder(
        quantity=Decimal(intent.quantity), price=_dec(intent.price),
        reference_price=_dec(intent.reference), side=intent.side,
        available_balance=_dec(intent.balance), symbol=intent.symbol or intent.instrument,
    )
    rules = None if row is None else c.VenueRules.from_row(row)
    return c.certify_venue(order, rules, seen, lookup=intent.lookup, window=window,
                           stamps_after=run.stamps_after)


def guard_agreement(
    module: types.ModuleType, intents: Iterable[Intent], specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Certify every intent's ruling under one version of the guard, stream by stream."""
    streams: dict[str, dict[str, Any]] = {}
    for intent in intents:
        row = rows_for(intent, specs)
        cert = certify_run(run_guard(module, intent, row), row)
        stream = streams.setdefault(intent.stream, {
            "orders": 0, "certified": 0, "violated": 0, "violations_by_gate": Counter(),
            "examples_violated": [], "examples_refused": [],
        })
        stream["orders"] += 1
        if cert.certified:
            stream["certified"] += 1
            if cert.first_unmet and len(stream["examples_refused"]) < 3:
                stream["examples_refused"].append({"intent": intent.as_dict(),
                                                   "first_unmet": cert.first_unmet})
            continue
        stream["violated"] += 1
        stream["violations_by_gate"][_gate_of(cert)] += 1
        if len(stream["examples_violated"]) < EXAMPLES:
            stream["examples_violated"].append({"intent": intent.as_dict(),
                                                "certificate": cert.as_dict() | {"steps": []}})
    for stream in streams.values():
        stream["violations_by_gate"] = dict(stream["violations_by_gate"].most_common())
        stream["agreement"] = round(stream["certified"] / stream["orders"], 6)
    return streams


def _gate_of(cert: c.Certificate) -> str:
    """The gate a violated certificate points at, for grouping."""
    for step in cert.steps:
        if not step.holds:
            return step.gate
    text = cert.violations[0] if cert.violations else ""
    for spec in c.VENUE_SPEC:
        if spec.gate in text:
            return spec.gate
    return "outcome" if text.startswith(("ruled", "the implementation raised")) else "goal"


# --- 2. the recorded decision chain ---------------------------------------------------------------


def recorded_chains() -> list[c.RecordedChain]:
    """Each ledger decision as the record of every gate it passed through."""
    records = {int(r["seq"]): r for r in risk_shadow.read_jsonl(risk_shadow.RISK_RECORDS)}
    notes = {int(r["seq"]): r.get("notes", []) for r in risk_shadow.read_jsonl(risk_shadow.NOTES)}
    out: list[c.RecordedChain] = []
    for row in risk_shadow.recorded_decisions():
        seq = int(row["seq"])
        record = records.get(seq)
        lines = notes.get(seq, [])
        protocol = next((n for n in lines if n.startswith("[protocol]")), None)
        rulings = [risk_shadow.ruling_line(n) for n in lines if n.startswith("[guard]")
                   and not n.startswith("[guard] venue rules")]
        venue_allowed: Decimal | None = None
        adjusted = False
        if rulings:
            last = rulings[-1]
            match = risk_shadow.ALLOWED_NOTE.match(last)
            venue_allowed = Decimal(match.group("q")) if match else Decimal(0)
            adjusted = bool(match and match.group("adj"))
        out.append(c.RecordedChain(
            seq=seq, symbol=str(row["symbol"]),
            proposed=None if record is None else Decimal(str(record["quantity_before"])),
            approved=None if record is None else Decimal(str(record["quantity_after"])),
            constitution_verdict=None if record is None else str(record["verdict"]),
            protocol_changed=None if protocol is None else "nothing changed" not in protocol,
            venue_consulted=bool(rulings), venue_allowed=venue_allowed, venue_adjusted=adjusted,
            ledger_verdict=str(row["verdict"]), ledger_quantity=Decimal(str(row["quantity"])),
        ))
    return out


def chain_report() -> dict[str, Any]:
    from argus.paper.corrections import VOIDED_SEQS

    certs = [(chain, c.certify_chain(chain)) for chain in recorded_chains()]
    tally = Counter(str(cert.status) for _, cert in certs)
    flagged = sorted(chain.seq for chain, cert in certs if cert.status is c.Status.VIOLATED)
    voided = sorted(VOIDED_SEQS)
    return {
        "decisions": len(certs),
        "certified": tally["certified"], "violated": tally["violated"],
        "uncertifiable": tally["uncertifiable"],
        "flagged_seqs": flagged,
        "voided_by_human_audit": voided,
        "caught_every_voided_row": set(voided) <= set(flagged),
        "flagged_outside_the_audit": sorted(set(flagged) - set(voided)),
        "violated_certificates": [cert.as_dict() for _, cert in certs
                                  if cert.status is c.Status.VIOLATED],
        "uncertifiable_reason": "no risk record: risk_records.jsonl begins at seq 83",
    }


# --- 3. the other gates ---------------------------------------------------------------------------

VERDICTS = ("trade", "reduce", "hedge", "delay", "no_trade", "human_review", "data_insufficient")


def circuit_cases(module: types.ModuleType) -> Iterator[tuple[str, c.Certificate]]:
    """Every book in the grid through ``apply``, then every start state through ``evaluate``."""
    for book, verdict in book_grid():
        label = f"book {({k: str(v) for k, v in book.items()})} verdict {verdict}"
        try:
            state = module.BookState(**book)
            ruling = module.apply(module.Verdict(verdict), state)
        except Exception as exc:  # a crash is a finding, reported like any other
            yield label, _crashed("circuit breaker", exc)
            continue
        yield label, c.certify_circuit(state, verdict, ruling)
    seen_books: set[str] = set()
    for book, _ in book_grid():
        key = repr(sorted(book.items()))
        if key in seen_books:
            continue
        seen_books.add(key)
        for start in ("active", "reduce_only", "halted"):
            label = f"breaker from {start} on book {({k: str(v) for k, v in book.items()})}"
            breaker = module.Breaker(activation=module.Activation(start))
            try:
                ruling = breaker.evaluate(module.Verdict("trade"), module.BookState(**book),
                                          now=datetime(2026, 9, 25, tzinfo=UTC))
            except Exception as exc:
                yield label, _crashed("breaker transition", exc)
                continue
            yield label, c.certify_transition(start, str(ruling.activation),
                                              str(breaker.activation))


def sizing_cases(module: types.ModuleType) -> Iterator[tuple[str, c.Certificate]]:
    """Graded-sample sizes and calibration errors either side of both gates, crossed with Kelly
    inputs and multipliers, including the ones the sizing layer must refuse."""
    from argus.eval.observatory import Prediction, expected_calibration_error

    for graded, accuracy in itertools.product((0, 19, 20, 21), ("0.6", "0.4", "0.3")):
        right = int(Decimal(graded) * Decimal(accuracy))
        predictions = [Prediction(confidence=0.6, correct=i < right) for i in range(graded)]
        ece = expected_calibration_error(predictions) if predictions else None
        for p, payoff, rm, sm in itertools.product(
            (0.3, 0.5, 0.55, 0.7, 0.9, 1.0, 1.2), ("0", "0.5", "1", "2", "3"),
            ("0", "0.5", "0.75", "1"), ("0.25", "0.5", "1", "1.5", "-0.1"),
        ):
            label = (f"graded {graded} accuracy {accuracy} p {p} payoff {payoff} "
                     f"multipliers {rm}x{sm}")
            try:
                sizing = module.size(
                    win_probability=p, payoff=Decimal(payoff), predictions=predictions,
                    risk_multiplier=Decimal(rm), session_multiplier=Decimal(sm),
                )
                refused = None
            except ValueError as exc:
                sizing, refused = None, str(exc)
            except Exception as exc:
                yield label, _crashed("position sizing", exc)
                continue
            yield label, c.certify_sizing(
                win_probability=p, payoff=Decimal(payoff), graded=graded, ece=ece,
                risk_multiplier=Decimal(rm), session_multiplier=Decimal(sm), sizing=sizing,
                refused=refused,
            )


def throttle_cases(module: types.ModuleType) -> Iterator[tuple[str, c.Certificate]]:
    """Measured and unmeasured profiles over horizons that do and do not cross a reopen."""
    from argus.truth.clocks import DualClock

    measured_at = datetime(2026, 9, 25, 12, tzinfo=UTC)
    profiles: dict[str, Any] = {"unmeasured": None}
    for name, rth, reopen in (("typical", 31.5, 52.1), ("violent reopen", 31.5, 180.0),
                              ("no reopen sample", 31.5, None), ("no baseline", 0.0, 52.1)):
        profiles[name] = module.SessionRisk(
            symbol="NVDAUSDT",
            phase_bps={"rth": rth, "extended": 14.1, "overnight": 12.8, "weekend": 5.2,
                       "holiday": 5.2},
            phase_counts={"rth": 400, "extended": 300, "overnight": 500, "weekend": 700,
                          "holiday": 40},
            reopen_bps=reopen, reopens=64, measured_at=measured_at, window_days=90,
        )
    starts = {
        # An hour before the open: the very next bar is a reopen, the loudest bar of the week,
        # which is where the throttle's floor binds.
        "Monday an hour before the open": datetime(2026, 9, 28, 13, tzinfo=UTC),
        "Tuesday an hour before the open": datetime(2026, 9, 22, 13, tzinfo=UTC),
        "Tuesday regular hours": datetime(2026, 9, 22, 15, tzinfo=UTC),
        "Friday after the close": datetime(2026, 9, 25, 21, tzinfo=UTC),
        "Saturday midday": datetime(2026, 9, 26, 12, tzinfo=UTC),
        "Sunday late": datetime(2026, 9, 27, 23, tzinfo=UTC),
    }
    clock = DualClock()
    for (pname, profile), (sname, start), horizon, floor in itertools.product(
        profiles.items(), starts.items(), (1, 8, 24, 72), ("0.25", "0.5"),
    ):
        label = f"{pname}, {sname}, {horizon} bar(s), floor {floor}"
        try:
            got = module.throttle(profile, start=start, horizon_bars=horizon,
                                  clock=clock, floor=Decimal(floor))
        except Exception as exc:
            yield label, _crashed("session throttle", exc)
            continue
        yield label, c.certify_throttle(
            measured=profile is not None, baseline_bps=got.baseline_bps,
            expected_bps=got.expected_bps, floor=Decimal(floor), multiplier=got.multiplier,
        )


OPERATOR_SETTINGS: tuple[str | None, ...] = (
    None, "normal", "paper", "reduce_only", "halted", " HALTED ", "hlated",
)
ACTIVATIONS = ("active", "reduce_only", "halted", "bogus")


def mode_cases(
    module: types.ModuleType, guard_module: types.ModuleType,
) -> Iterator[tuple[str, c.Certificate]]:
    """Every combination of layer inputs; every verdict, venue and open-position count under each
    resulting stack; and the composed trace of a real order through ``check_order``."""
    nvda = guard_module.Instrument.from_payload(risk_shadow.variant_row("nvda"))
    nvda_rules = c.VenueRules.from_row(risk_shadow.variant_row("nvda"))
    for operator, loaded, activation, live in itertools.product(
        OPERATOR_SETTINGS, (True, False), ACTIVATIONS, (False, True),
    ):
        facts = f"operator {operator!r}, rules {loaded}, breaker {activation}, live {live}"
        environ = {} if operator is None else {module.OPERATOR_ENV: operator}
        try:
            demands = (module.operator_demand(environ),
                       module.venue_demand(rules_loaded=loaded),
                       module.circuit_demand(activation),
                       module.posture_demand(live_enabled=live))
            stack = module.resolve(demands)
        except Exception as exc:
            yield facts, _crashed("risk mode", exc)
            continue
        observed = [(str(d.layer), None if d.mode is None else str(d.mode))
                    for d in stack.demands]
        yield facts, c.certify_layers(observed, operator=operator, rules_loaded=loaded,
                                      activation=activation, live_enabled=live)
        effective, binding = str(stack.effective), str(stack.binding.layer)
        for verdict, venue, open_positions in itertools.product(
            VERDICTS, ("paper", "live", "moon"), (0, 1),
        ):
            admission = module.admit(stack, verdict=verdict, venue=venue,
                                     open_positions=open_positions)
            yield (f"{facts}; {verdict} on {venue}, {open_positions} open",
                   c.certify_mode(observed, verdict=verdict, venue=venue,
                                  open_positions=open_positions, effective=effective,
                                  binding=binding, admitted=bool(admission.admitted)))
        for verdict in ("trade", "hedge", "reduce", "no_trade"):
            guard = guard_module.Guard(instruments={"NVDAUSDT": nvda})
            label = f"{facts}; check_order {verdict}"
            before = tuple(guard.limiter._stamps)
            ruling = module.check_order(
                guard, stack, symbol="NVDAUSDT", verdict=verdict, quantity=Decimal("1"),
                side="buy", reference_price=Decimal("100"), now=NOW,
            )
            yield label, _certify_composed(
                ruling, verdict, effective, before, tuple(guard.limiter._stamps), nvda_rules)


def _certify_composed(
    ruling: Any, verdict: str, effective: str, before: tuple[float, ...],
    after: tuple[float, ...], rules: c.VenueRules,
) -> c.Certificate:
    """A ``check_order`` ruling: the mode event first, then either the whole venue stack not
    reached (a refusal, spending nothing) or the venue gates, certified as a session guard call."""
    admitted, words = c.expect_admission(effective, verdict, "paper", 0)
    events = tuple((str(e.gate), str(e.verdict), str(e.reason)) for e in ruling.trace.events)
    violations: list[str] = []
    if not events or events[0][0] != "risk_mode":
        violations.append("the trace does not begin with the risk_mode gate")
        return c.Certificate("composed order", c.Status.VIOLATED, "risk_mode first",
                             f"{[e[0] for e in events][:2]}", None, tuple(violations))
    if events[0][1] != ("pass" if admitted else "deny"):
        violations.append(f"risk_mode records {events[0][1]}; the specification "
                          f"{'admits' if admitted else 'refuses'} it — {words}")
    if not admitted:
        rest = [(gate, state) for gate, state, _ in events[1:]]
        wanted = [(spec.gate, "not_reached") for spec in c.VENUE_SPEC]
        if rest != wanted:
            violations.append("a refusal by the mode must list every venue gate as not reached")
        if str(ruling.denial) != "risk_mode" or Decimal(str(ruling.quantity)) != 0:
            violations.append(f"refusal recorded as {ruling.denial} with quantity "
                              f"{ruling.quantity}")
        if after != before:
            violations.append("a refusal by the mode spent a rate-window slot")
        return c.Certificate("composed order", c.Status.VIOLATED if violations
                             else c.Status.CERTIFIED, f"refused: {words}",
                             f"{ruling.denial}", f"risk_mode: unmet precondition — {words}",
                             tuple(violations))
    seen = c.observe_venue(ruling)
    venue_part = c.ObservedVenue(seen.denial, seen.quantity, seen.price, seen.adjusted,
                                 events[1:])
    cert = c.certify_venue(
        c.VenueOrder(quantity=Decimal("1"), reference_price=Decimal("100"), side="buy",
                     symbol="NVDAUSDT"),
        rules, venue_part, lookup=True, window=c.Window(before, 10, 60.0, NOW),
        stamps_after=after, subject="composed order",
    )
    if violations:
        return c.Certificate(cert.subject, c.Status.VIOLATED, cert.expected, cert.observed,
                             cert.first_unmet, (*violations, *cert.violations), cert.steps)
    return cert


def _crashed(subject: str, exc: BaseException) -> c.Certificate:
    return c.Certificate(subject, c.Status.VIOLATED, "a ruling", f"raised {type(exc).__name__}",
                         None, (f"raised {type(exc).__name__}: {exc}",))


def tally(cases: Iterable[tuple[str, c.Certificate]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for label, cert in cases:
        counts[str(cert.status)] += 1
        if not cert.certified and len(examples) < EXAMPLES:
            examples.append({"case": label, "certificate": cert.as_dict()})
    total = sum(counts.values())
    return {"cases": total, **{s.value: counts[s.value] for s in c.Status},
            "agreement": round(counts["certified"] / total, 6) if total else None,
            "examples_violated": examples}


# --- 4. mutation testing --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mutant:
    """One deliberate defect: which module, which gate, what was changed, and exactly where."""

    ident: str
    target: str
    gate: str
    change: str
    find: str
    replace: str
    equivalent: bool = False
    """Cannot change any output by construction. Expected to survive; checked that it does."""


class StaleMutant(RuntimeError):
    """A mutant whose target text no longer occurs exactly once. Raised, never skipped: a mutant
    that silently stops applying reads as a certifier that silently stopped being tested."""


MUTANTS: tuple[Mutant, ...] = (
    Mutant("G01", "guard", "finite_inputs", "the finiteness check is removed",
           "if value is not None and not value.is_finite():", "if False:"),
    Mutant("G02", "guard", "side_known", "any side is accepted",
           "if normalised not in _SIDES:", "if False:"),
    Mutant("G03", "guard", "instrument_online", "offline instruments are accepted",
           "    if not instrument.is_online:\n", "    if False:\n"),
    Mutant("G04", "guard", "rate_window", "the rate cap is never enforced",
           "        if exceeded:\n", "        if False:\n"),
    Mutant("G05", "guard", "rate_window", "a stamp exactly one window old still counts",
           "while self._stamps and self._stamps[0] <= cutoff:",
           "while self._stamps and self._stamps[0] < cutoff:"),
    Mutant("G06", "guard", "price_positive", "a zero limit price passes this gate",
           "    elif price <= 0:\n", "    elif price < 0:\n"),
    Mutant("G07", "guard", "price_step", "a limit price stepped to zero is allowed",
           "        if limit <= 0:\n", "        if limit < 0:\n"),
    Mutant("G08", "guard", "price_step", "the limit price is not stepped to the tick",
           "        limit = quantise_down(price, tick)\n", "        limit = price\n"),
    Mutant("G09", "guard", "price_band", "the band's upper edge is doubled",
           "            if not lower <= limit <= upper:\n",
           "            if not lower <= limit <= upper * 2:\n"),
    Mutant("G10", "guard", "price_band", "sells are held to the buy band",
           'instrument.buy_limit_ratio if side == "buy" else instrument.sell_limit_ratio',
           "instrument.buy_limit_ratio"),
    Mutant("G11", "guard", "quantity_step", "the quantity is not stepped",
           "    stepped_qty = quantise_down(quantity, step)\n", "    stepped_qty = quantity\n"),
    Mutant("G12", "guard", "quantity_step", "stepping rounds half-up instead of down",
           ".to_integral_value(rounding=ROUND_DOWN)",
           '.to_integral_value(rounding="ROUND_HALF_UP")'),
    Mutant("G13", "guard", "quantity_ceiling", "the venue maximum is not enforced",
           "    if stepped_qty > instrument.max_order_qty:\n", "    if False:\n"),
    Mutant("G14", "guard", "quantity_ceiling", "a maximum below one step is not refused",
           "        if stepped_qty <= 0:\n", "        if False:\n"),
    Mutant("G15", "guard", "quantity_floor", "an order exactly at the minimum is refused",
           "    if stepped_qty < instrument.min_order_qty:\n",
           "    if stepped_qty <= instrument.min_order_qty:\n"),
    Mutant("G16", "guard", "notional_floor", "the minimum notional is halved",
           "        if notional < instrument.min_order_amount:\n",
           "        if notional < instrument.min_order_amount / 2:\n"),
    Mutant("G17", "guard", "balance", "the balance is doubled",
           "            if notional > available_balance:\n",
           "            if notional > available_balance * 2:\n"),
    Mutant("G18", "guard", "rate_window", "an allowed order never spends its slot",
           "    if limiter is not None and record:\n", "    if False:\n"),
    Mutant("G19", "guard", "rate_window", "every order past the rate gate spends a slot",
           "        in_window = limiter.count(now=moment)\n",
           "        in_window = limiter.count(now=moment)\n"
           "        if record:\n            limiter.record(now=moment)\n"),
    Mutant("G20", "guard", "instrument_known",
           "an unlisted symbol is ruled against another instrument's rules",
           "        instrument = self.instruments.get(symbol)\n",
           "        instrument = self.instruments.get(symbol) or "
           "next(iter(self.instruments.values()))\n"),
    Mutant("G21", "guard", "quantity_floor", "the trace calls a passed gate skipped",
           'trail.note("quantity_floor", GateVerdict.PASS,',
           'trail.note("quantity_floor", GateVerdict.SKIP,'),
    Mutant("G22", "guard", "quantity_step", "stepping runs at 28 digits again",
           "        ctx.prec = exact_precision(value, step)\n", "        ctx.prec = 28\n"),
    Mutant("G23", "guard", "notional_floor", "the value arithmetic runs at 28 digits again",
           "        ctx.prec = max(getcontext().prec, instrument.arithmetic_width",
           "        ctx.prec = 28 or max(getcontext().prec, instrument.arithmetic_width"),
    Mutant("G24", "guard", "price_band", "the trace names the band gate as the step gate",
           'trail.note("price_band", GateVerdict.PASS,',
           'trail.note("price_step", GateVerdict.PASS,'),
    Mutant("G25", "guard", "trace", "gates after a refusal are omitted from the trace",
           "        return PermissionCheckTrace((*self.events, *not_reached(rest, refused_by)))",
           "        return PermissionCheckTrace(tuple(self.events))"),
    Mutant("C01", "circuit", "total_drawdown", "a drawdown exactly at 10% no longer halts",
           "    if state.total_drawdown >= TOTAL_DRAWDOWN_HALT:\n",
           "    if state.total_drawdown > TOTAL_DRAWDOWN_HALT:\n"),
    Mutant("C02", "circuit", "session_drawdown", "the session drawdown rule is removed",
           "    if state.session_drawdown >= DAILY_DRAWDOWN_HALT:\n", "    if False:\n"),
    Mutant("C03", "circuit", "losing_streak", "four losses in a row no longer halt",
           "    if state.consecutive_losses >= CONSECUTIVE_LOSS_HALT:\n",
           "    if state.consecutive_losses > CONSECUTIVE_LOSS_HALT:\n"),
    Mutant("C04", "circuit", "shock", "the shock threshold moves from -3 to -4 sigma",
           "    if state.realised_move_sigma <= -SIGMA_SHOCK:\n",
           "    if state.realised_move_sigma <= -SIGMA_SHOCK - 1:\n"),
    Mutant("C05", "circuit", "stale_evidence", "evidence exactly six hours old is stale",
           "    if state.evidence_age > STALE_EVIDENCE:\n",
           "    if state.evidence_age >= STALE_EVIDENCE:\n"),
    Mutant("C06", "circuit", "drawdown_ladder", "the reduce-only ladder is removed",
           "    if REDUCE_ONLY_DRAWDOWN <= state.total_drawdown < TOTAL_DRAWDOWN_HALT:\n",
           "    if False:\n"),
    Mutant("C07", "circuit", "resolution", "one reduce-only rule outvotes a halting one",
           "    if any(t.demands is Activation.HALTED for t in trips):\n",
           "    if all(t.demands is Activation.HALTED for t in trips):\n"),
    Mutant("C08", "circuit", "narrowing", "a halted book may still hedge",
           "    if activation is Activation.HALTED and verdict.opens_exposure:\n",
           "    if activation is Activation.HALTED and verdict is Verdict.TRADE:\n"),
    Mutant("C09", "circuit", "narrowing", "reduce-only no longer narrows a trade",
           "    elif activation is Activation.REDUCE_ONLY and verdict is Verdict.TRADE:\n",
           "    elif False:\n"),
    Mutant("C10", "circuit", "multiplier", "the 6% rung scales to 0.75 instead of 0.5",
           '        return Decimal("0.5")\n', '        return Decimal("0.75")\n'),
    Mutant("C11", "circuit", "multiplier", "a halted book still sizes at half",
           '        return Decimal("0")\n    if drawdown >= Decimal("0.06"):\n',
           '        return Decimal("0.5")\n    if drawdown >= Decimal("0.06"):\n'),
    Mutant("C12", "circuit", "transition", "a halted breaker may jump straight to active",
           "    Activation.HALTED: {Activation.REDUCE_ONLY},\n",
           "    Activation.HALTED: {Activation.REDUCE_ONLY, Activation.ACTIVE},\n"),
    Mutant("S01", "sizing", "calibration_sample", "nineteen graded outcomes are enough",
           "    if len(predictions) < MIN_GRADED:\n",
           "    if len(predictions) < MIN_GRADED - 1:\n"),
    Mutant("S02", "sizing", "calibration_error", "the calibration ceiling is doubled",
           "    if ece > MAX_ECE:\n", "    if ece > MAX_ECE * 2:\n"),
    Mutant("S03", "sizing", "kelly_floor", "a negative Kelly stake is not floored at zero",
           '    return max(Decimal("0"), raw)\n', "    return raw\n"),
    Mutant("S04", "sizing", "kelly_share", "full Kelly instead of half",
           "    staked = raw * KELLY_FRACTION * applied\n", "    staked = raw * applied\n"),
    Mutant("S05", "sizing", "position_cap", "the 25% cap is removed from Kelly sizing",
           "        fraction=min(MAX_FRACTION, staked),\n", "        fraction=staked,\n"),
    Mutant("S06", "sizing", "reduce_only_multipliers", "a session multiplier of 1.5 is accepted",
           "    if session_multiplier > 1:\n", "    if session_multiplier > 2:\n"),
    Mutant("S07", "sizing", "reduce_only_multipliers", "the fixed fraction ignores multipliers",
           "            fraction=min(MAX_FRACTION, FIXED_FRACTION * applied),\n",
           "            fraction=min(MAX_FRACTION, FIXED_FRACTION),\n"),
    Mutant("T01", "session_risk", "throttle_floor", "the throttle floor is removed",
           '    bounded = max(floor, min(Decimal("1"), raw))\n',
           '    bounded = min(Decimal("1"), raw)\n'),
    Mutant("T02", "session_risk", "throttle_unmeasured", "an unmeasured symbol is halved",
           '            multiplier=Decimal("1"), reason="session volatility not measured for '
           'this instrument",\n',
           '            multiplier=Decimal("0.5"), reason="session volatility not measured for '
           'this instrument",\n'),
    Mutant("T03", "session_risk", "throttle_ratio", "the ratio is inverted",
           "    raw = Decimal(str(baseline / expected))\n",
           "    raw = Decimal(str(expected / baseline))\n"),
    Mutant("T04", "session_risk", "throttle_cap", "the cap at one is removed",
           '    bounded = max(floor, min(Decimal("1"), raw))\n',
           "    bounded = max(floor, raw)\n", equivalent=True),
    Mutant("D01", "modes", "resolution", "the least restrictive demand wins",
           "        return max((d.mode for d in self.demands if d.mode is not None), "
           "key=lambda m: m.rank)\n",
           "        return min((d.mode for d in self.demands if d.mode is not None), "
           "key=lambda m: m.rank)\n"),
    Mutant("D02", "modes", "resolution", "the last layer, not the first, is named binding",
           "        return next(d for d in self.demands if d.mode is mode)\n",
           "        return next(d for d in reversed(self.demands) if d.mode is mode)\n"),
    Mutant("D03", "modes", "risk_mode", "a halted desk may still hedge",
           "    if mode is RiskMode.HALTED:\n",
           '    if mode is RiskMode.HALTED and kind == "trade":\n'),
    Mutant("D04", "modes", "risk_mode", "paper and reduce-only desks may open on the live venue",
           '    if place == "live" and mode is not RiskMode.NORMAL:\n',
           '    if place == "live" and mode is RiskMode.HALTED:\n'),
    Mutant("D05", "modes", "operator", "an unreadable operator switch is ignored",
           "        return Demand(Layer.OPERATOR, RiskMode.HALTED,\n",
           "        return Demand(Layer.OPERATOR, None,\n"),
    Mutant("D06", "modes", "venue", "missing venue rules only reduce instead of halting",
           "    return Demand(Layer.VENUE, RiskMode.HALTED,\n",
           "    return Demand(Layer.VENUE, RiskMode.REDUCE_ONLY,\n"),
    Mutant("D07", "modes", "circuit", "an unrecognised breaker state is ignored",
           "        return Demand(Layer.CIRCUIT, RiskMode.HALTED,\n"
           '                      f"unrecognised breaker state',
           "        return Demand(Layer.CIRCUIT, None,\n"
           '                      f"unrecognised breaker state'),
    Mutant("D08", "modes", "risk_mode", "a mode refusal omits the venue gates from the trace",
           '            PermissionCheckTrace((event, *not_reached(GUARD_GATES, "risk_mode"))),\n',
           "            PermissionCheckTrace((event,)),\n"),
)


def source_of(target: str) -> str:
    """A module's current source with line endings normalised, so a mutant's text matches."""
    return SOURCES[target].read_text(encoding="utf-8").replace("\r\n", "\n")


def mutate(source: str, mutant: Mutant) -> str:
    count = source.count(mutant.find)
    if count != 1:
        raise StaleMutant(f"{mutant.ident}: target text occurs {count} times in "
                          f"{mutant.target}; the mutant no longer applies")
    return source.replace(mutant.find, mutant.replace)


@dataclass(frozen=True, slots=True)
class Hunt:
    """One mutant against the certifier: caught or not, and after how many cases."""

    mutant: Mutant
    caught: bool
    cases_run: int
    exposed_by: str | None
    certificate: c.Certificate | None

    def as_dict(self) -> dict[str, Any]:
        m = self.mutant
        return {
            "id": m.ident, "target": m.target, "gate": m.gate, "change": m.change,
            "equivalent": m.equivalent,
            "outcome": ("caught" if self.caught else
                        "survived (equivalent, as expected)" if m.equivalent else "SURVIVED"),
            "cases_run": self.cases_run, "exposed_by": self.exposed_by,
            "certifier_said": None if self.certificate is None else (
                self.certificate.violations[0] if self.certificate.violations else None),
        }


def _hunt(mutant: Mutant, cases: Iterable[tuple[str, c.Certificate]]) -> Hunt:
    run = 0
    for label, cert in cases:
        run += 1
        if not cert.certified:
            return Hunt(mutant, True, run, label, cert)
    return Hunt(mutant, False, run, None, None)


def guard_cases(
    module: types.ModuleType, intents: Sequence[Intent], specs: dict[str, dict[str, Any]],
) -> Iterator[tuple[str, c.Certificate]]:
    for intent in itertools.chain(intents, risk_shadow.swept_intents()):
        row = rows_for(intent, specs)
        label = f"{intent.stream} {intent.ident or ''} {intent.as_dict()}".strip()
        yield label, certify_run(run_guard(module, intent, row), row)


def cases_for(
    target: str, module: types.ModuleType, *, intents: Sequence[Intent],
    specs: dict[str, dict[str, Any]], guard_module: types.ModuleType,
) -> Iterable[tuple[str, c.Certificate]]:
    runners: dict[str, Callable[[], Iterable[tuple[str, c.Certificate]]]] = {
        "guard": lambda: guard_cases(module, intents, specs),
        "circuit": lambda: circuit_cases(module),
        "sizing": lambda: sizing_cases(module),
        "session_risk": lambda: throttle_cases(module),
        "modes": lambda: mode_cases(module, guard_module),
    }
    return runners[target]()


def mutation_report(
    intents: Sequence[Intent], specs: dict[str, dict[str, Any]],
    guard_module: types.ModuleType,
) -> dict[str, Any]:
    """Every mutant, hunted until the certifier objects or the case space runs out."""
    hunts: list[Hunt] = []
    for mutant in MUTANTS:
        module = load_source(mutate(source_of(mutant.target), mutant), f"mutant {mutant.ident}")
        hunts.append(_hunt(mutant, cases_for(mutant.target, module, intents=intents,
                                             specs=specs, guard_module=guard_module)))
    real = [h for h in hunts if not h.mutant.equivalent]
    caught = sum(h.caught for h in real)
    equivalent_survived = all(not h.caught for h in hunts if h.mutant.equivalent)
    by_target: dict[str, dict[str, int]] = {}
    for hunt in real:
        row = by_target.setdefault(hunt.mutant.target, {"mutants": 0, "caught": 0})
        row["mutants"] += 1
        row["caught"] += hunt.caught
    return {
        "mutants": len(real), "caught": caught, "survived": len(real) - caught,
        "kill_rate": round(caught / len(real), 6) if real else None,
        "equivalent_mutants": sum(h.mutant.equivalent for h in hunts),
        "equivalent_mutants_survived_as_expected": equivalent_survived,
        "by_target": by_target,
        "gates_mutated": sorted({h.mutant.gate for h in real}),
        "hunts": [h.as_dict() for h in hunts],
    }


# --- the run --------------------------------------------------------------------------------------


def run(*, path: Path = REPORT_PATH, refresh: bool = False) -> dict[str, Any]:
    intents, specs, described = risk_shadow.streams(refresh=refresh)
    current = load_engine(CURRENT_GUARD, "guard current")
    engines = {"current": current}
    for name, label in (("guard_2026-09-25_pre_s20.py.txt", "before S20 (2026-09-25 morning)"),
                        ("guard_2026-09-24_deployed.py.txt", "deploy/api copy (2026-09-24)")):
        if (BASELINES / name).exists():
            engines[label] = load_engine(BASELINES / name, label)

    agreement = {
        label: {"engine": engine.as_dict(),
                "streams": guard_agreement(engine.module,
                                           [*intents, *risk_shadow.swept_intents()], specs)}
        for label, engine in engines.items()
    }
    real = {name: load_engine(path_, f"{name} current") for name, path_ in SOURCES.items()
            if name != "guard"}
    report = {
        "what": ("the risk layer certified gate by gate against a written specification, and "
                 "every gate broken on purpose to test the certifier"),
        "sources": {
            "certifier": ("PlanBench (MIT): llm_planning_analysis/full_validator/__init__.py:6-63 "
                          "step replay; utils/task_utils.py:549-561, 589-706 localised "
                          "unmet-precondition messages; back_prompting.py:355-359 sound versus "
                          "unsound correction signals"),
            "trace_and_modes": ("letta-code (Apache-2.0): src/permissions/types.ts:32-52, "
                                "checker.ts:88-105, 202-229; src/permissions/mode.ts:3-18; "
                                "src/reminders/engine.ts:337-377"),
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "qwen_calls": 0,
        "specification": {
            "venue": [s.as_dict() for s in c.VENUE_SPEC], "venue_goals": list(c.VENUE_GOALS),
            "circuit": [s.as_dict() for s in c.CIRCUIT_SPEC],
            "sizing": [s.as_dict() for s in c.SIZING_SPEC],
            "modes": [s.as_dict() for s in c.MODE_SPEC],
            "chain": [s.as_dict() for s in c.CHAIN_SPEC],
        },
        "streams": described,
        "guard_agreement": agreement,
        "recorded_chain": chain_report(),
        "circuit": tally(circuit_cases(real["circuit"].module)),
        "sizing": tally(sizing_cases(real["sizing"].module)),
        "session_throttle": tally(throttle_cases(real["session_risk"].module)),
        "modes": tally(mode_cases(real["modes"].module, current.module)),
        "mutation": mutation_report(intents, specs, current.module),
        "not_covered": [
            "the Constitution's own rules (agents/desk.py) — swept by eval/riskproof.py; here "
            "only its recorded input and output are certified, as step 1 of the chain",
            "the measured inputs sizing and the throttle report (calibration error, path "
            "volatility) are taken as given, not re-derived",
            "Decimal exponents beyond the default context's Emax (|exponent| > 999999) are not "
            "probed",
        ],
    }
    artefact.write(path, report)
    return report


def main() -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = run()
    for label, block in report["guard_agreement"].items():
        print(f"guard {label}:")
        for name, stream in block["streams"].items():
            print(f"  {name:<15} {stream['orders']:>7} orders  {stream['certified']:>7} "
                  f"certified  {stream['violated']:>6} violated  {stream['violations_by_gate']}")
    chain = report["recorded_chain"]
    print(f"recorded chain: {chain['decisions']} decisions, {chain['certified']} certified, "
          f"{chain['violated']} violated {chain['flagged_seqs']}, {chain['uncertifiable']} "
          f"uncertifiable; human audit voided {chain['voided_by_human_audit']}")
    for name in ("circuit", "sizing", "session_throttle", "modes"):
        block = report[name]
        print(f"{name}: {block['cases']} cases, {block['certified']} certified, "
              f"{block['violated']} violated")
    mutation = report["mutation"]
    print(f"mutants: {mutation['caught']}/{mutation['mutants']} caught; equivalent survived as "
          f"expected: {mutation['equivalent_mutants_survived_as_expected']}")
    for hunt in mutation["hunts"]:
        print(f"  {hunt['id']} {hunt['gate']:<22} {hunt['outcome']:<34} after "
              f"{hunt['cases_run']} case(s)")
    return 0 if mutation["survived"] == 0 else 1


__all__ = [
    "MUTANTS",
    "REPORT_PATH",
    "Hunt",
    "Mutant",
    "StaleMutant",
    "certify_run",
    "chain_report",
    "circuit_cases",
    "guard_agreement",
    "main",
    "mode_cases",
    "mutate",
    "mutation_report",
    "recorded_chains",
    "run",
    "sizing_cases",
    "tally",
    "throttle_cases",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
