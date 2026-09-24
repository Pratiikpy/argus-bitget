"""Paper-trading runner — drives the desk against live Bitget data and writes the log.

One invocation is one decision cycle: read the real market, build the session state and hedge
surface, run the analyst panel, let the Meta-PM decide, apply the Constitution, and append the
result to a hash-chained ledger. Run it on a schedule and the log accumulates.

**Settlement is the part that makes it honest.** Open entries are settled against the price at a
*later* fetch, never against the price in the same cycle. A runner that decided and settled in one
pass would be scoring itself on information it already had.

    python -m argus.paper.runner --once
    python -m argus.paper.runner --report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from argus.agents.desk import ConstitutionPolicy, DeskRun, TradingDesk
from argus.cost.model import CostModel
from argus.decision.verdicts import Intent, Verdict
from argus.eval.autopsy import RECORD_SCHEMA
from argus.eval.gate_ablation import ablated_variants
from argus.execution.guard import Guard, GuardError
from argus.execution.guard import fetch_instruments as fetch_specs
from argus.llm.qwen import QwenClient, QwenError, Thinking, TokenBudget
from argus.market.bitget import RTOKEN_SYMBOLS, Ticker, fetch_rtokens
from argus.market.estimates import EstimatesSource
from argus.market.evidence import (
    BitgetSkillSource,
    RedditSource,
    TwitterSource,
    gather,
    underlying_ticker,
)
from argus.market.fundamentals import FundamentalsSource
from argus.market.insider import InsiderSource
from argus.market.macro import MacroError, fear_greed_evidence, fetch_fear_greed
from argus.market.macro import evidence as curve_evidence
from argus.market.macro import latest as latest_curve
from argus.market.macro import status as curve_status
from argus.market.microstructure import (
    Halt,
    MicrostructureError,
    ShortVolume,
    fetch_halts,
    fetch_short_volume,
)
from argus.market.microstructure import evidence as micro_evidence
from argus.market.microstructure import status as micro_status
from argus.market.skills import evidence as skill_evidence
from argus.market.skills import load as load_skill_health
from argus.market.skills import probe as probe_skills
from argus.market.skills import usable_probes
from argus.market.volatility import Reading, VolatilityError
from argus.market.volatility import evidence as vix_evidence
from argus.market.volatility import fetch as fetch_vix
from argus.market.volatility import status as vix_status
from argus.paper import marks
from argus.paper.ledger import PaperLedger
from argus.paper.protocol import enforce as enforce_protocol
from argus.paper.protocol import governing
from argus.paper.protocol import load as load_protocol
from argus.risk.effectiveness import lookup
from argus.risk.hedgeability import (
    HedgeabilitySurface,
    open_market_candidate,
    shut_market_candidate,
)
from argus.risk.session_risk import lookup as session_lookup
from argus.truth.clocks import DualClock
from argus.truth.evidence import Evidence

LEDGER_PATH = Path(__file__).resolve().parents[3] / "data" / "paper_ledger.jsonl"

SKILL_HEALTH_PATH = LEDGER_PATH.with_name("bitget_skills_health.json")
"""Last measured health of Bitget's official Skill tools.

The cycle calls only the tools that sweep found answering. A full sweep costs a timeout for every
dead upstream — thirteen of nineteen on 2026-09-13 — which is minutes per symbol, and no decision
cycle can pay that. Regenerate it with ``python -m argus.market.skills``."""

RISK_PATH = LEDGER_PATH.with_name("risk_records.jsonl")
"""What the risk layer did to each decision. Read by ``python -m argus.eval.riskaudit``."""

NOTES_PATH = LEDGER_PATH.with_name("desk_notes.jsonl")
"""Where the desk's per-decision checks are kept.

Beside the ledger rather than inside it. An :class:`~argus.paper.ledger.Entry` is a decision, its
shape is hashed, and appending free text to it would put a checker's verdict inside the thing the
checker is supposed to be independent of."""

# Phrases a check writes when it found something, as opposed to reporting that it looked. Matched on
# the rendered line because that is what the modules produce, and each entry is copied from the
# module that writes it rather than guessed — the first draft of this list invented wordings and
# missed a live grounding failure ("1 of 3 figure(s) do not resolve...") on the very next cycle.
# `tests/test_runner_notes.py` pins each one against the module's own output, so a reworded check
# fails a test instead of silently ceasing to be flagged.
_FLAG_MARKERS = (
    "is contradicted by",                # agents/claims.py Contradiction.render
    "defect in the reasoning",           # agents/claims.py ClaimReport.render
    "do not resolve to anything",        # agents/grounding.py GroundingReport.render
    "unsupported fact",                  # agents/grounding.py, second line of the same failure
    "[conflict:",                        # agents/conflict.py Conflict.render, a real disagreement
    "disagreement(s) unresolved",        # agents/conflict.py ConflictReport.render
    "contradicts itself",                # agents/desk.py, a self-contradictory earnings print
)

# How long a paper position is held before it is settled against a later fetch.
HOLD_HOURS = 24

STARTING_EQUITY = Decimal("100000")
"""Notional book the paper ledger's realised P&L accumulates against.

Only ever used as the *denominator* of a drawdown, so its absolute value changes nothing about a
decision — a 6% drawdown is 6% at any book size. It is stated here rather than threaded through
because inventing a different one per caller would make two drawdowns incomparable.
"""


def _median_funding(tickers: dict[str, Ticker]) -> Decimal:
    """The median live funding rate across the universe, as a fraction.

    Median rather than per-symbol because the ledger holds one cost model for the cycle, and median
    rather than mean because these rates are zero most of the time with occasional spikes — a mean
    would let one symbol's 5.8bps settlement price every other symbol's trade.
    """
    rates = sorted(t.funding_rate for t in tickers.values())
    if not rates:
        return Decimal("0")
    middle = len(rates) // 2
    if len(rates) % 2:
        return rates[middle]
    return (rates[middle - 1] + rates[middle]) / Decimal("2")


def _is_flag(note: str) -> bool:
    """Did a check find a problem, rather than merely report that it ran?"""
    low = note.lower()
    return any(marker in low for marker in _FLAG_MARKERS)


def _write_notes(
    seq: int, symbol: str, at: datetime, notes: list[str], sources: tuple[str, ...] = ()
) -> None:
    """Append one decision's check output. Best effort: a notes failure must not lose a decision.

    The ledger write has already happened by the time this runs, and a disk error here would
    otherwise abort the cycle after the record it was meant to annotate was committed.

    ``sources`` is every distinct evidence source that reached the decision. It is written here
    rather than into the ledger because the ledger's entry shape is hashed, and because this is an
    annotation about *how* the decision was reached rather than part of the decision itself. Rows
    written before 2026-09-15 carry no `sources` key at all, and `eval/sourceaudit.py` must read
    that absence as **unmeasured** rather than as "no sources reached this decision".
    """
    try:
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NOTES_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "seq": seq,
                "symbol": symbol,
                "at": at.isoformat(),
                "notes": notes,
                "flags": [n for n in notes if _is_flag(n)],
                "sources": list(sources),
            }) + "\n")
    except OSError as exc:  # pragma: no cover - disk failure
        print(f"could not write desk notes for seq {seq}: {exc}", file=sys.stderr)


def _book_state(ledger: PaperLedger) -> object:
    """The book as the risk layer judges it: the **realised** equity curve.

    **This returned ``None`` on an empty ledger for one revision, and that was over-cautious.** The
    reasoning was that zero settled trades means an unknown drawdown, and `risk/circuit.py` is right
    that a breaker whose unknown state is "carry on" protects nothing. But a paper book that opened
    at :data:`STARTING_EQUITY` and has closed nothing has realised equity of exactly that and a peak
    of exactly that; its drawdown is **0 as a measurement**, not as a guess. Returning ``None``
    disabled the position-size cap on precisely the book that has proven nothing — the opposite of
    what the caution intended.

    Equity is realised, not marked: an unsettled decision does not move it. Marking one would let an
    unrealised swing tighten or loosen the breaker before the outcome that justifies it exists. Open
    positions are still reported, because `circuit.assess` trips on concentration separately from
    drawdown.
    """
    from argus.risk.circuit import BookState

    equity = STARTING_EQUITY
    peak = equity
    consecutive = 0
    for entry in ledger.entries:
        if not entry.is_settled or entry.net_pnl is None:
            continue
        realised = Decimal(entry.net_pnl)
        equity += realised
        peak = max(peak, equity)
        consecutive = consecutive + 1 if realised < 0 else 0
    return BookState(
        equity=equity, peak_equity=peak, session_open_equity=equity,
        consecutive_losses=consecutive,
        open_positions=len([e for e in ledger.entries if not e.is_settled]),
    )


def _graded_predictions(ledger: PaperLedger) -> list[object]:
    """Every (stated confidence, was it right) pair the record can actually support.

    This is what decides whether confidence may set position size at all. It is deliberately hard
    to satisfy: `risk/sizing.py` wants **20** graded outcomes before it will let a stated confidence
    scale a position, and the record currently supplies **zero** — 183 decisions, 69 settled, none
    of them a settled trade and none of them a settled lean. So the calibration gate fails, sizing
    falls back to its fixed fraction, and it says so in the basis string rather than silently
    sizing on a confidence nobody has checked.

    A settled *trade* grades its stated confidence against `direction_correct`. A settled
    *abstention* grades its **lean** — the direction the desk would have taken — against the
    counterfactual move, which is the whole reason `eval/shadow.py` records a lean at all. The two
    are kept in one list because the calibration question is the same for both: when this desk says
    0.7, does it happen 70% of the time?
    """
    from argus.eval.observatory import Prediction

    out: list[object] = []
    for entry in ledger.entries:
        if not entry.is_settled:
            continue
        if not entry.is_abstention:
            if entry.direction_correct is not None:
                out.append(Prediction(
                    confidence=float(entry.stated_confidence),
                    correct=bool(entry.direction_correct),
                ))
            continue
        # An abstention's directional content is its lean, not its side — `side` was measured to
        # carry no information (BUY in all 53 settled abstentions, right 3.8% against a 3.8% base
        # rate), which is exactly why the lean field exists.
        lean = str(getattr(entry, "lean", "none")).lower()
        move = entry.counterfactual_move_bps
        if lean in ("none", "") or move is None:
            continue
        realised_up = Decimal(move) > 0
        out.append(Prediction(
            confidence=float(getattr(entry, "lean_confidence", 0.0)),
            correct=(lean == "up") == realised_up,
        ))
    return out


def _write_chain(seq: int, symbol: str, run: DeskRun) -> None:
    """Persist this decision's causal chain, if it made one.

    Failure here is reported and never raised: a chain that cannot be stored must not cost us the
    decision it describes. That is the same trade-off `_write_notes` makes, and for the same reason
    — these files sit *beside* the ledger because the ledger's entry shape is hashed.
    """
    if run.causal_chain is None:
        return
    try:
        from argus.paper import chains

        # `path` is passed explicitly and read off the module at call time. Relying on the
        # function's own default would bind `CHAINS_PATH` at import, which is invisible until
        # something tries to redirect it — and the first test written against this wiring wrote
        # into the real ledger directory while asserting against a temporary one.
        chains.write(run.causal_chain, seq=seq, symbol=symbol, path=chains.CHAINS_PATH)
    except OSError as exc:  # pragma: no cover - filesystem
        print(f"could not write causal chain for seq {seq}: {exc}", file=sys.stderr)


def _grade_chains(outcomes: dict[int, tuple[float, datetime]]) -> int:
    """Grade every stored chain whose decision has now settled. Returns how many were graded.

    The chains are graded here rather than at decision time for the obvious reason and one less
    obvious one: `grade_pending` refuses any pairing whose settlement instant does not strictly
    postdate the decision, so passing it the ledger's own settlement stamps is what keeps a chain
    from being scored against a move it could already see.
    """
    if not outcomes:
        return 0
    try:
        from argus.paper import chains

        path = chains.CHAINS_PATH
        stored = chains.load(path)
        before = len([r for r in stored if r.graded])
        after = chains.grade_pending(stored, outcomes, path=path)
        return len([r for r in after if r.graded]) - before
    except OSError as exc:  # pragma: no cover - filesystem
        print(f"could not grade causal chains: {exc}", file=sys.stderr)
        return 0


def _write_risk_record(seq: int, symbol: str, at: datetime, run: DeskRun) -> None:
    """Record what the risk layer did to this decision, so its effect can be counted.

    Track 2's judged half scores "risk control layer effectiveness", and until this was written
    nothing in the system could answer it: `ConstitutionRuling` carries a machine-readable
    `binding_constraint` precisely so "which constraint bound, how often" is countable, and nothing
    was counting. The audit is `argus.eval.riskaudit`.

    Kept beside the ledger rather than inside it, for the same reason as the desk notes: an
    :class:`~argus.paper.ledger.Entry` is the decision, and its shape is hashed.
    """
    ruling = run.ruling
    original = run.proof.llm_original_intent
    # `quantity_before` must be what the **Constitution** was handed, not the model's first draft.
    # See `DeskRun.ruled_intent` for the live rows where those two differed and made the record
    # self-contradictory. Falls back to the draft only when the desk recorded no ruled intent at
    # all, which is the pre-2026-09-20 record shape.
    ruled = run.ruled_intent or original
    final = run.proof.llm_revised_intent or (
        ruling.resulting_intent if ruling is not None else original
    )
    try:
        RISK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RISK_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "seq": seq,
                "symbol": symbol,
                "at": at.isoformat(),
                "verdict": str(final.verdict),
                "intervened": run.proof.constitution_intervened,
                # Stamped so a reader can tell schema 1's ambiguous "none" from schema 2's.
                # See `eval/autopsy.RECORD_SCHEMA` for why reinterpreting is not an option.
                "chain_schema": RECORD_SCHEMA,
                "binding_constraint": None if ruling is None else ruling.binding_constraint,
                "reason": None if ruling is None else ruling.reason,
                "quantity_before": str(ruled.quantity),
                # Kept alongside, not instead: the model's first draft is real and worth auditing,
                # it is simply not the number the risk layer acted on.
                "model_draft_quantity": str(original.quantity),
                "quantity_after": str(final.quantity),
                "model_changed_its_mind": run.proof.llm_changed_its_mind,
                "constitution_only_reduced": run.proof.constitution_only_reduced(),
                # Population RCT (Track 2 §13.7a) — additive, no schema bump: an old reader that
                # never looks at this key sees exactly the record it always did.
                "ablated_rulings": {
                    dimension: {
                        "binding_constraint": r.binding_constraint,
                        "quantity_after": str(r.resulting_intent.quantity),
                    }
                    for dimension, r in run.ablated_rulings.items()
                },
            }) + "\n")
    except OSError as exc:  # pragma: no cover - disk failure
        print(f"could not write risk record for seq {seq}: {exc}", file=sys.stderr)


def _executable_spread_bps(
    symbol: str, notional: Decimal, side: str, *, fallback: Decimal,
) -> Decimal:
    """What crossing the book costs for this size, or the quoted touch when the book cannot say.

    The quoted spread is the price of an infinitesimal trade. The desk's trades are not
    infinitesimal, and the difference is large: measured on 2026-09-14, SQQQUSDT quotes 4.96bps
    and costs 19.43bps to take $25,000 of, while QQQUSDT quotes 0.14bps and costs 0.82bps.
    Charging the quote for a real size understates the cost by a factor that varies by instrument,
    which is worse than understating it by a constant — it reorders which instruments look cheap.

    A zero or unpriceable size falls back to the quote rather than to zero, and a book that cannot
    absorb the size returns nothing at all (`market/depth.py:measured_spread_bps`), because a
    partial sweep prices the cheap half of a trade that could not be done.
    """
    if notional <= 0:
        return fallback
    try:
        from argus.market.depth import measured_spread_bps

        measured = measured_spread_bps(symbol, notional, direction=side)
    except Exception:
        measured = None
    return fallback if measured is None else measured


def _hedge_surface(session_asleep: bool, symbol: str) -> HedgeabilitySurface:
    """The anchor hedge, and whether it can actually be reached.

    Kept on the surface in both session states so the absence is visible rather than omitted — that
    distinction is the whole Sleeping-Anchor thesis.

    **The open-market branch used to return an empty surface**, which put a false sentence into the
    record of every regular-hours decision: "no hedge placeable". During regular hours the hedge is
    plainly placeable by somebody; what is true is narrower — ARGUS has no equity broker, so the
    venue is open, the liquidity is there, and we have no route to it. Those are different facts,
    and conflating them flatters us: "the market gave me no hedge" excuses an unhedged position,
    "I have not built the connection" does not. The menu stays empty either way, so no behaviour is
    invented; what changes is that the reason in the record is now the true one.

    **And the three statistical factors used to be constants.** ``risk_reduction=0.98``,
    ``correlation_confidence=0.98`` and ``basis_stability=0.95`` were typed here and in the
    constructor with nothing behind them. They now come from `risk/effectiveness.py`, which measures
    Ederington effectiveness, the minimum-variance ratio and a Fisher bound on the correlation from
    the venue's own market-versus-index series, split by session phase. When a fresh measurement
    exists the candidate is stamped MEASURED and carries its sample; when it does not, the old
    constants are used and it is stamped ASSUMED. Nothing silently improves — the record says which.
    """
    underlying = symbol.replace("USDT", "")
    phase = "rth" if not session_asleep else "weekend"
    measured = lookup(symbol, phase)
    if measured is None:
        # A stale or missing measurement is treated as no measurement. A correlation from three
        # weeks ago is not a weaker version of today's; it describes a different market.
        measured = lookup(symbol, "all")
    attribution = (
        f" Hedge factors measured: {measured.render()}" if measured is not None
        else " Hedge factors are ASSUMED constants; run `python -m argus.risk.effectiveness` to "
             "measure them"
    )
    if not session_asleep:
        return HedgeabilitySurface(
            (open_market_candidate(underlying, Decimal("0.98"), measured=measured),),
            session_note=(
                f"{underlying} is open and the hedge is reachable in the market; ARGUS has no "
                f"equity broker, so it is not reachable by us. The residual is a capability limit, "
                f"not a market state." + attribution
            ),
        )
    return HedgeabilitySurface(
        (shut_market_candidate(underlying, Decimal("0.95"), measured=measured),),
        session_note="anchor shut; no hedge placeable." + attribution,
    )


DEBATE_TOKENS = 12_000
"""What holding a debate adds to one symbol's cost. The single live measurement (`agents/desk.py`)
was 20,714 -> 27,691 tokens, about 7,000; 12,000 is headroom. Used to decide whether a split panel
can be argued without leaving a later symbol undecided."""

PER_SYMBOL_TOKENS = 25_000
"""The **marginal** cost of adding one symbol to a cycle. Measured at 20,838; 25,000 is headroom.

See :data:`CYCLE_OVERHEAD_TOKENS` for why a marginal figure alone was not enough.
"""

CYCLE_OVERHEAD_TOKENS = 30_000
"""What a cycle costs before it decides anything: panel setup, evidence gathering, the checks.

**The budget model had the wrong shape, and a single-symbol cycle could not complete because of
it.** ``PER_SYMBOL_TOKENS * len(symbols)`` assumes a cycle costs nothing until a symbol is added.
Two live cycles on identical code say otherwise:

    1 symbol  -> 42,478 tokens
    4 symbols -> 104,991 tokens

which solved to 20,838 marginal per symbol and 21,640 fixed per cycle, and predicted three
twelve-symbol cycles it was not fitted on to within 0.7% and 4.5%.

**That decomposition is now retracted, and the retraction is the finding.** Removing one
model call per unconstrained decision (`agents/desk.py`) and re-measuring gave 19,592 at one symbol
and 77,868 at four. Refitting the same two-point line puts almost the whole saving in the *fixed*
term (21,640 -> 167) while the marginal term barely moves — which cannot be true of a change that
removes a **per-decision** call. The line fits both pairs and attributes the saving differently, so
the decomposition was over-fitted to two noisy points.

Cycle cost is not linear in symbol count, because the expensive parts are conditional: the debate
runs only when the panel splits on direction, the adversary only when exposure is opened, and two
cycles hours apart see different evidence. Comparing totals across runs conflates the change with
the market.

What is measured rather than estimated is the **call count**, which is deterministic: the decision
is now put to the model once where it was put twice, pinned by a test. The constants below are
therefore left generous. Over-provisioning costs nothing — the guard simply never fires — while
under-provisioning kills a cycle, and that asymmetry is the whole reason this constant exists.

At twelve symbols the old formula was accidentally right: 300,000 covers a true 271,692, which is
why the scheduled cycles worked. At one symbol it budgeted 25,000 against a true 42,478 and the
guard fired **17,478 tokens short**, killing the cycle with nothing written — the same class of
failure recorded in :func:`run_once` below, and this time the shape was wrong rather than the value.

The overhead is set above the fitted 21,640 for the same reason the marginal figure is: this is a
hard ceiling on a key with a finite balance, and a ceiling that is occasionally too low costs a
whole cycle while one that is occasionally too high costs nothing. Both numbers are re-derivable —
run a one-symbol and a four-symbol cycle and solve the pair.
"""


def governed_intent(run: DeskRun, *, symbol: str = "") -> Intent:
    """The intent the ledger may record: what the **Constitution** decided, never the model's raw
    proposal.

    **Extracted from `run_once` on 2026-09-20 because being inline is why a critical bug survived.**
    The line read `proof.llm_revised_intent or proof.llm_original_intent`, whose fallback silently
    discarded the risk layer on every decision the model was not re-asked about — the common case,
    since the desk deliberately does not buy a second opinion when nothing bound
    (*"[constitution] nothing bound, so the decision was not re-put to the model"*).
    `agents/desk.py:657` computes the same value correctly as
    `proof.llm_revised_intent or ruling.resulting_intent`. The two disagreed; the ledger believed
    the wrong one.

    **It fabricated two trades on the live record, seq 264 and 265.** For both, `agents/desk.py`
    emitted *"no order: final verdict human_review with quantity 0"* and the risk record wrote
    `quantity_after: 0, binding_constraint: no_exposure`, while the ledger stored
    `verdict: trade, quantity: 1` and the settlement pass later booked **+9.6521** and **+5.3339**
    of P&L against positions the desk had refused to take. Those two rows were, until this fix,
    the entire basis of the project's win rate and net-P&L figures.

    No test caught it: every test fed this path an intent the Constitution had already allowed, so
    the divergent branch was never exercised. It was found by an adversarial audit reading
    `eval/decisioncard.py --seq 264`, which printed all four contradictory facts on one page.

    `DeskRun.ruling` is optional only because of a dataclass default; `desk.run()` always sets it.
    A genuinely absent ruling means the Constitution never ruled, and the fail-safe direction is to
    refuse — falling back to the unconstrained intent is exactly the bug.
    """
    if run.ruling is None:
        raise RuntimeError(
            f"{symbol or run.symbol}: the desk returned no Constitution ruling, so nothing may be "
            f"recorded; refusing rather than booking the model's unconstrained intent"
        )
    return run.proof.llm_revised_intent or run.ruling.resulting_intent


def run_once(
    *,
    symbols: tuple[str, ...] = ("NVDAUSDT",),
    ledger_path: Path = LEDGER_PATH,
    budget: int = 0,
) -> dict[str, object]:
    """One decision cycle across the given symbols.

    ``budget`` defaults to :data:`CYCLE_OVERHEAD_TOKENS` plus :data:`PER_SYMBOL_TOKENS` per symbol.

    It used to be a flat 150,000 regardless of the cycle, and that number was smaller than a
    twelve-symbol cycle needs: **every scheduled run from 08:52 on 2026-09-13 died partway through
    with "token budget exhausted" and wrote nothing at all.** The guard was doing exactly what it
    was told; it had been told a limit that did not fit the job.

    It was then made proportional — ``PER_SYMBOL_TOKENS * len(symbols)`` — and that **failed the
    same way at the other end of the range**, because a cycle costs about 21,640 tokens before it
    decides anything. A twelve-symbol cycle absorbed the error and a one-symbol cycle did not: it
    budgeted 25,000 against a measured 42,478 and died with nothing written. Adding the fixed term
    is the actual fix; raising the per-symbol constant would only have moved the cliff.

    The limit still exists and is still hard. What changed is that it is now derived from the work
    requested **and the work that happens regardless**, and that running out no longer costs the
    whole cycle — see the loop below.
    """
    if budget <= 0:
        budget = CYCLE_OVERHEAD_TOKENS + PER_SYMBOL_TOKENS * max(1, len(symbols))
    clock = DualClock()
    now = datetime.now(UTC)
    tickers = fetch_rtokens()
    # The ledger's cost model is built per cycle rather than once, so the venue's live funding rate
    # reaches it. `Ticker.funding_rate` has been fetched since the beginning and thrown away before
    # it touched a cost; a 24-hour hold crosses three settlements, which is ~0.7-1.7bps typically
    # and ~17bps in the tail against an 18.8bps hurdle. The median rToken rate is exactly zero, so
    # this changes nothing most days and stops the net PnL being overstated on the days it does not.
    funding = _median_funding(tickers)
    ledger = PaperLedger(path=ledger_path, cost=CostModel.bitget_perp(funding_rate=funding))

    settled = _settle_due(ledger, tickers, now)

    # Take the short-horizon mark before anything else touches the ledger. An abstention settles
    # once, at 24h, and 24h windows on decisions taken 2h apart overlap by ~92% — so the settled
    # record carries roughly three independent observations, not one per row, and cannot score a
    # refusal. The mark is the second, non-overlapping observation that can. See paper/marks.py.
    marked = _mark_due(ledger, tickers, now, governs_from_seq=_marking_governs_from())

    # Read once per cycle rather than per symbol: the file is append-only and a cycle must not
    # straddle two protocols.
    commitments = load_protocol()

    # The venue's own trading rules, read once per cycle rather than per symbol. A decision that
    # sizes a position the venue would reject is not a decision the ledger should record as taken,
    # and the rules are published — so they are fetched rather than assumed. A fetch failure leaves
    # the guard absent and is reported per decision, never silently treated as permission.
    # The rates curve, read once per cycle from the issuer. It is the same macro question Bitget's
    # `rates_yields` Skill is meant to answer and returns nothing for, so the desk gets it from the
    # US Treasury instead. A failure is carried as a status line, never as a silent absence.
    try:
        curve = latest_curve()
        macro_evidence = curve_evidence(curve, as_of=now)
        macro_status = curve_status(curve)
    except MacroError as exc:
        macro_evidence, macro_status = [], curve_status(None, str(exc))

    # Risk appetite, from the only free sentiment feed that answers this network. The three the
    # sentiment audit recommended were probed on 2026-09-13: StockTwits returns 403 to four
    # different User-Agents, CNN's Fear & Greed returns 418, and this one returns 200.
    #
    # It is carried at credibility 0.35 and labelled crypto-wide, because that is what it measures.
    # A venue-level risk reading is real information about the environment these tokens trade in
    # and is not a view on any single equity, and presenting it as one would inflate the evidence
    # count without adding information — which is the failure mode the sentiment demotion in
    # `argus.eval.standing` exists to record.
    try:
        fng = fetch_fear_greed()
        risk_evidence = fear_greed_evidence(fng, as_of=now)
        risk_status = f"fear-greed: {fng.value}/100 {fng.classification} (crypto-wide)"
    except MacroError as exc:
        risk_evidence, risk_status = [], f"fear-greed: unavailable ({exc})"

    # Implied volatility, from CBOE. Unlike the crypto risk-appetite reading above, this measures
    # the market these tokens are anchored to, so it is carried at credibility 0.9 rather than
    # 0.35. It is also the one feed that can say whether the tape is even moving far enough to pay
    # a round trip: at a normal VIX the implied one-day move is roughly 100bps against an 18.8bps
    # weekend hurdle, which is how the desk can distinguish "nothing is happening" from "something
    # is happening and I cannot call the direction".
    vix: Reading | None = None
    try:
        vix = fetch_vix()
        vol_evidence = vix_evidence(vix, as_of=now)
        vol_status = vix_status(vix)
    except VolatilityError as exc:
        vol_evidence, vol_status = [], vix_status(None, str(exc))

    # Short volume and halts on the UNDERLYING equities, from FINRA and Nasdaq. Fetched once per
    # cycle for the whole universe rather than per symbol: the FINRA file is one document covering
    # every US equity, and twelve requests for one file would be twelve times the courtesy cost for
    # the same bytes.
    #
    # A halt matters here more than anywhere else. When the underlying is halted the rToken keeps
    # trading, so the token's movement during that window is tracking nothing — and no other feed
    # in this system can see it.
    short_volume: dict[str, ShortVolume] = {}
    halts: tuple[Halt, ...] = ()
    try:
        underlyings = frozenset(underlying_ticker(s) for s in symbols)
        short_volume = fetch_short_volume(wanted=underlyings)
        halts = fetch_halts()
        micro_note = micro_status(short_volume, halts)
    except MicrostructureError as exc:
        micro_note = micro_status(None, None, str(exc))

    guard: Guard | None = None
    guard_note = ""
    try:
        loaded = Guard(instruments=fetch_specs())
    except GuardError as exc:
        guard_note = f"[guard] venue rules unavailable ({exc}); no order was validated"
    else:
        guard = loaded
        guard_note = f"[guard] venue rules loaded for {len(loaded.instruments)} instrument(s)"

    client = QwenClient(budget=TokenBudget(limit=budget))
    # Routine cycles at LOW: the priced hurdle drops from ~27bps to ~19bps off-hours, which
    # is the difference between a desk that can trade and one that abstains forever.
    desk = TradingDesk(client, pm_thinking=Thinking.LOW)
    written: list[dict[str, object]] = []

    stopped_early = ""
    for index, symbol in enumerate(symbols):
        if symbol not in tickers:
            continue
        # Stop cleanly with what has been written rather than dying with nothing. A cycle that
        # decided seven symbols and ran out is seven decisions; the same cycle raising is zero,
        # and zero is what the scheduled runs were producing. The reason is recorded so a short
        # cycle is never mistaken for a quiet market.
        if client.budget is not None and client.budget.remaining < PER_SYMBOL_TOKENS:
            stopped_early = (
                f"stopped after {len(written)} of {len(symbols)} symbol(s): "
                f"{client.budget.remaining} token(s) left of {client.budget.limit}, below the "
                f"{PER_SYMBOL_TOKENS} a symbol costs. The symbols not reached were not decided, "
                f"and no decision was recorded for them."
            )
            break
        ticker = tickers[symbol]
        session = clock.state(now, nav_age_seconds=(now - ticker.fetched_at).total_seconds())
        hedges = _hedge_surface(session.is_anchor_asleep, symbol)

        # Live facts only — never synthetic. The price fact stays; what changed is that the desk
        # now also receives real filings and headlines. Eighteen straight abstentions traced to
        # this list holding one price line and nothing an event analyst could act on.
        evidence = [
            Evidence(
                id=f"mkt-{symbol}",
                claim=(
                    f"{symbol} last {ticker.last}, 24h change {ticker.change_24h}, "
                    f"quoted spread {ticker.spread_bps:.2f}bps, "
                    f"24h base volume {ticker.base_volume}"
                ),
                source="news",
                available_at=ticker.fetched_at,
                credibility=1.0,
            ),
        ]
        # The insider source is supplied explicitly because `gather` will not reach SEC for Form 4
        # data unless asked: each filing costs two extra requests, so the caller that wants it pays
        # for it visibly. This is the live cycle, and it wants it. `twitter`/`reddit` are supplied
        # here for the same reason and because this call site is genuinely live (`as_of=now`) —
        # the one place both sources' own LIVE-ONLY constraint is satisfied by construction.
        gathered = gather(
            symbol, as_of=now,
            insider=InsiderSource(), fundamentals=FundamentalsSource(),
            twitter=TwitterSource(), reddit=RedditSource(),
        )
        evidence.extend(gathered.evidence)

        # What the market EXPECTS, which the desk itself asked for: ledger seq 41 ended
        # "fundamentals cannot be assessed against expectations without consensus estimates".
        # An index ETF has no earnings, so a 404 here is an absence rather than a failure and the
        # status line says which.
        consensus, consensus_status = EstimatesSource().evidence(
            underlying_ticker(symbol), as_of=now
        )
        evidence.extend(consensus)
        gathered.status.extend(consensus_status)

        # Macro is instrument-independent, so the same curve is given to every symbol's panel.
        evidence.extend(macro_evidence)
        gathered.status.append(macro_status)

        # Same reasoning as the curve: instrument-independent, so every panel sees the same
        # reading, and its scope travels with it in the claim text rather than in a comment.
        evidence.extend(risk_evidence)
        gathered.status.append(risk_status)

        evidence.extend(vol_evidence)
        gathered.status.append(vol_status)

        # Per-symbol, unlike the three macro feeds above: this is a fact about THIS underlying.
        # `underlying`, not `ticker`: `ticker` is already the live Ticker object in this loop and
        # rebinding it would have shadowed the price the decision is built from.
        underlying = underlying_ticker(symbol)
        evidence.extend(micro_evidence(
            ticker=underlying, as_of=now, short=short_volume.get(underlying), halts=halts,
        ))
        gathered.status.append(micro_note)

        # Bitget's own research Skills, as the handbook suggests for this track
        # (`BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md:425`). Only the calls the last health sweep saw
        # answer are made: a dead upstream costs a full timeout before it admits anything, and
        # paying that per symbol per cycle would make the cycle unrunnable. What is skipped is
        # skipped on measured evidence, and the measurement is on disk beside the ledger.
        live = usable_probes(load_skill_health(SKILL_HEALTH_PATH))
        if live:
            skill_report = probe_skills(
                BitgetSkillSource(), symbol=symbol, at=now, probes=live, timeout=10
            )
            evidence.extend(skill_evidence(skill_report, as_of=now))
            skill_notes = skill_report.render()
        else:
            skill_notes = [
                "[skills] no Bitget Skill tool is recorded as answering; none called this cycle"
            ]
        # Feed health is itself evidence: a dark source is a fact the decision-maker was told
        # about, not a silent gap. Rendered as a low-credibility note, not as a market claim.
        evidence.append(Evidence(
            id=f"feeds-{symbol}",
            claim="evidence feeds: " + "; ".join(gathered.status),
            source="news",
            available_at=now,
            credibility=0.1,
        ))

        # The measured session-volatility throttle, handed in rather than fetched inside the
        # rulebook: a deterministic rule that reaches for the network is a rule that can fail
        # open. An unmeasured symbol leaves the gate inert and the autopsy records it as
        # UNREACHED rather than as passed.
        active_policy = ConstitutionPolicy(
            session_risk=session_lookup(symbol),
            book_state=_book_state(ledger),
            graded_predictions=_graded_predictions(ledger),
        )
        # Enough left for this symbol's debate AND for every symbol still to come at its base cost.
        still_to_decide = len(symbols) - index
        debate_affordable = client.budget is None or (
            client.budget.remaining >= PER_SYMBOL_TOKENS * still_to_decide + DEBATE_TOKENS
        )
        run = desk.run(
            symbol=symbol,
            debate_affordable=debate_affordable,
            session=session,
            token_price=ticker.last,
            position=Decimal("0"),
            evidence=evidence,
            hedges=hedges,
            decision_id=f"paper-{symbol}-{int(now.timestamp())}",
            constitution=active_policy,
            # Population RCT (Track 2 §13.7a), accumulated as a free side effect of a cycle that
            # runs anyway rather than a separate spend against the hackathon key: the same
            # `attacked` intent this cycle decides on, re-ruled against one gate-ablated variant
            # per Foundation 5 dimension. `TradingDesk.run`'s own docstring/tests
            # (`test_ablated_rulings.py::test_ablated_variants_cost_no_extra_llm_calls`) establish
            # this costs zero extra model calls.
            ablated_constitutions=ablated_variants(active_policy),
            # The desk's own record, so it can see what it did here before. `recall` applies
            # point-in-time itself against `session.as_of`, so passing the whole ledger is safe:
            # an entry that settles after this instant contributes its decision and not its
            # outcome. Passing a pre-filtered slice would move that guarantee to the caller,
            # which is where it would eventually be got wrong.
            history=ledger.entries,
        )

        final = governed_intent(run, symbol=symbol)

        # The pre-registered protocol, applied last and able only to reduce. It is deliberately the
        # final gate: the Constitution reasons about this position's risk, and the protocol asks a
        # different question — is this decision inside what we publicly committed to before any
        # outcome existed? A commitment that is checked only after the fact is a description.
        governed_by = governing(commitments, len(ledger.entries) + 1)
        ruling = enforce_protocol(
            governed_by,
            verdict=str(final.verdict), quantity=final.quantity,
            session_phase=str(session.phase), symbol=symbol,
        )

        # The venue gate, applied after the Constitution and the protocol and, like both of them,
        # able only to shrink or refuse. Ordered last because it answers a different question from
        # either: not "should we" but "would this instruction be accepted at all".
        venue_note = guard_note
        venue_quantity = ruling.quantity
        venue_verdict = ruling.verdict
        if guard is not None and ruling.quantity > 0:
            checked = guard.check(
                symbol, quantity=ruling.quantity, reference_price=ticker.last,
                side=str(final.side).lower(),
            )
            venue_note = checked.render()
            if checked.allowed:
                venue_quantity = checked.quantity
            else:
                venue_quantity = Decimal("0")
                venue_verdict = str(Verdict.NO_TRADE)

        entry = ledger.record(
            symbol=symbol,
            verdict=venue_verdict,
            side=str(final.side).upper(),
            quantity=venue_quantity,
            entry_price=ticker.last,
            stated_confidence=final.stated_confidence,
            thesis=final.thesis,
            invalidation=final.invalidation,
            market_state_hash=run.proof.market_state_hash,
            approved_intent_hash=run.proof.approved_intent_hash,
            session_phase=str(session.phase),
            hours_to_discovery=session.hours_to_next_discovery,
            decided_at=now,
            spread_bps=_executable_spread_bps(
                symbol, venue_quantity * ticker.last, str(final.side).upper(),
                fallback=ticker.spread_bps,
            ),
            # The directional view, recorded whether or not the desk acts on it. Taken from the
            # model's ORIGINAL intent rather than the Constitution-narrowed one: the risk layer
            # resizes and refuses, it does not form opinions, and a lean read off its output would
            # be attributing the desk's view to the guard.
            lean=run.proof.llm_original_intent.lean,
            lean_confidence=run.proof.llm_original_intent.lean_confidence,
        )
        # The desk's own checks — numeric grounding, claim grounding, analyst conflict, the causal
        # chain — all write into `run.notes`, and until now the runner discarded every one of them.
        # The seq-41 hallucination was found by reading a thesis by hand; the checker that catches
        # it was running and its verdict went nowhere. Notes are written beside the ledger rather
        # than into it: the ledger is a record of decisions, and its entry shape is hashed.
        _write_notes(
            entry.seq, symbol, now,
            [*run.notes, *skill_notes, ruling.render(), venue_note],
            run.evidence_sources,
        )
        _write_risk_record(entry.seq, symbol, now, run)
        # The causal chain, persisted at decision time so it can be graded later. `paper/chains.py`
        # existed and was never called from here, which meant its own docstring — "the chain was
        # constructed, counted, described as checkable, and dropped on the floor" — was still an
        # accurate description of the live path rather than of the defect it closed. Every
        # event-driven decision carries three to five separately checkable claims; until this line
        # every one of them was discarded at the end of the cycle.
        _write_chain(entry.seq, symbol, run)
        flags = [n for n in run.notes if _is_flag(n)]
        written.append({
            "seq": entry.seq,
            "symbol": symbol,
            "verdict": entry.verdict,
            "quantity": entry.quantity,
            "confidence": entry.stated_confidence,
            "attests_llm_decided": run.proof.attests_llm_decided(),
            "notes_written": len(run.notes),
            "protocol": (
                None if governed_by is None else {
                    "digest": governed_by.protocol_digest,
                    "version": governed_by.protocol.version,
                    "reduced_this_decision": ruling.applied,
                }
            ),
            # Surfaced in the cycle summary, not merely filed. A check whose finding is only in a
            # sidecar nobody opens has the same effect as no check.
            "flags": flags,
        })

    return {
        "ran_at": now.isoformat(),
        "decisions_written": written,
        "settled_this_cycle": settled,
        # Reported, not merely written. The marks are the only observation at a horizon short
        # enough to be independent, so a cycle that took none is a cycle that added nothing to the
        # refusal record — and that has to be visible in the summary rather than inferred later
        # from the file's line count.
        "marked_this_cycle": marked,
        "tokens_spent": client.budget.spent if client.budget else 0,
        # Calls the venue never billed in writing. `tokens_spent` excludes them, so publishing the
        # count beside it is what stops the figure reading as complete. See `llm/qwen.TokenBudget`.
        "unreported_calls": client.budget.unreported_calls if client.budget else 0,
        "token_budget": client.budget.limit if client.budget else 0,
        "stopped_early": stopped_early,
        "chain": ledger.verify(),
        "protocol": (
            "no protocol committed" if not commitments
            else f"{commitments[-1].protocol.protocol_id} v{commitments[-1].protocol.version} "
                 f"({commitments[-1].protocol_digest})"
        ),
    }


def _marking_governs_from() -> int:
    """The first sequence number a protocol that declares marking may govern.

    Marks are an observation the protocol has to have declared *before* the decision was taken.
    Decisions recorded under an earlier version keep exactly the observations that version provided
    for; adding one now — with their outcomes already on the tape — would be choosing a measurement
    after seeing the result.
    """
    for commitment in load_protocol():
        if "refusal_marks.jsonl" in commitment.protocol.settlement_rule:
            return commitment.governs_from_seq
    # No committed protocol declares marking, so nothing may be marked. Returning 0 here would
    # mark the entire ledger under no declared rule at all.
    return sys.maxsize


def _mark_due(
    ledger: PaperLedger, tickers: dict[str, Ticker], now: datetime, *, governs_from_seq: int
) -> list[dict[str, object]]:
    """Record a short-horizon observation for every abstention that has come due for one.

    Observation only. Nothing here decides, scores, or writes to the ledger — the marks live in
    their own append-only file beside it, because the ledger is hash-chained and settles each entry
    exactly once. A symbol with no ticker this cycle is skipped and will be picked up by the next
    cycle while it remains inside the marking window; it is never marked at a fabricated price.
    """
    out: list[dict[str, object]] = []
    for entry in marks.due(
        list(ledger.entries), now=now, governs_from_seq=governs_from_seq
    ):
        ticker = tickers.get(entry.symbol)
        if ticker is None:
            continue
        try:
            row = marks.mark(
                seq=entry.seq,
                symbol=entry.symbol,
                decided_at=datetime.fromisoformat(entry.decided_at),
                marked_at=now,
                entry_price=Decimal(entry.entry_price),
                mark_price=ticker.last,
                lean=entry.lean or "none",
            )
        except marks.MarkError:
            # A decision with no usable entry price cannot yield a move. It is left unmarked
            # rather than marked with a number that would have to be invented.
            continue
        out.append({
            "seq": row.seq,
            "symbol": row.symbol,
            "horizon_hours": row.horizon_hours,
            "move_bps": row.move_bps,
            "lean": row.lean,
        })
    return out


def _settle_due(
    ledger: PaperLedger, tickers: dict[str, Ticker], now: datetime
) -> list[dict[str, object]]:
    """Settle open positions whose hold period has elapsed, at the current price.

    Only entries recorded in an *earlier* cycle are eligible, which is what keeps settlement from
    using information the decision already had.
    """
    out: list[dict[str, object]] = []
    # seq -> (realised move in bps, when it settled). Collected here and handed to the chain
    # grader after the loop, so a chain is scored against the same move the ledger recorded rather
    # than against a second, separately-derived number that could drift from it.
    outcomes: dict[int, tuple[float, datetime]] = {}
    for entry in ledger.entries:
        if entry.is_settled:
            continue
        decided = datetime.fromisoformat(entry.decided_at)
        if now - decided < timedelta(hours=HOLD_HOURS):
            continue
        ticker = tickers.get(entry.symbol)
        if ticker is None:
            continue

        if entry.is_abstention:
            # An abstention used to be skipped here, which made it **permanently ungradeable** —
            # a desk that abstained correctly looked identical to one paralysed by its own hurdle.
            # Recording the move that actually happened is what lets the Observatory score it.
            got = ledger.settle_abstention(entry.seq, price_now=ticker.last, settled_at=now)
            if got.counterfactual_move_bps is not None:
                outcomes[got.seq] = (float(got.counterfactual_move_bps), now)
            out.append({
                "seq": got.seq,
                "symbol": got.symbol,
                "abstention": True,
                "counterfactual_move_bps": got.counterfactual_move_bps,
            })
            continue

        # The exit is crossed now, at this size, so it is priced from the book now — not from
        # the 0.6bps literal that used to stand in for every exit ever.
        closing_side = "SELL" if str(entry.side).upper() == "BUY" else "BUY"
        got = ledger.settle(
            entry.seq, exit_price=ticker.last, settled_at=now,
            exit_spread_bps=_executable_spread_bps(
                entry.symbol, Decimal(entry.quantity) * ticker.last, closing_side,
                fallback=ticker.spread_bps,
            ),
        )
        # The chain predicted a *price move*, not a P&L, so it is graded against the move. Taking
        # net_pnl here would charge the forecast for the fee, and a chain that called the direction
        # and the magnitude correctly would be marked wrong on a trade the cost model killed.
        entry_px, exit_px = Decimal(got.entry_price), Decimal(got.exit_price or "0")
        if entry_px > 0 and exit_px > 0:
            outcomes[got.seq] = (
                float((exit_px - entry_px) / entry_px * Decimal("10000")), now,
            )
        out.append({
            "seq": got.seq,
            "symbol": got.symbol,
            "gross_pnl": got.gross_pnl,
            "net_pnl": got.net_pnl,
            "direction_correct": got.direction_correct,
        })

    graded = _grade_chains(outcomes)
    if graded:
        out.append({"causal_chains_graded": graded})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS paper-trading runner")
    parser.add_argument("--once", action="store_true", help="run one decision cycle")
    parser.add_argument("--report", action="store_true", help="print ledger performance")
    parser.add_argument("--verify", action="store_true", help="verify the hash chain only")
    parser.add_argument(
        "--symbols", default="NVDAUSDT",
        help=f"comma-separated; any of {','.join(RTOKEN_SYMBOLS)}",
    )
    args = parser.parse_args()

    if args.verify or args.report:
        ledger = PaperLedger(path=LEDGER_PATH)
        payload = ledger.verify() if args.verify else ledger.performance()
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.once:
        try:
            result = run_once(symbols=tuple(s.strip() for s in args.symbols.split(",")))
        except QwenError as exc:
            print(f"LLM unavailable: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
