"""Same evidence, said differently: does the decision survive? Scored on the worst case.

`eval/consistency.py` asks whether the desk returns the same action when the **identical** frame is
put to it again. That is the sampling baseline, and it is necessary. It is not sufficient: a
decision-maker can be perfectly reproducible on one byte sequence and still flip when the same facts
arrive in a different order, with ``12.4bps`` written as ``0.124%``, or with the evidence header
laid out another way. None of those changes alters what is known, so none of them should alter what
is done. This module measures whether they do.

**What is perturbed, and why each is meaning-preserving.** Four seeded perturbations, applied to the
evidence block the Meta-PM actually reads (:meth:`MarketFrame.to_prompt_block`):

* ``reorder`` — the evidence lines, shuffled. The desk promises nothing about evidence order, and
  its own ranking (`agents/mandate.order_evidence`) is a presentation choice, not a fact.
* ``unit_restate`` — every figure quoted in ``bps`` restated in percent and every percent restated
  in bps, exactly (``Decimal`` arithmetic, no rounding), and large integers given thousands
  separators. HELM's text perturbations include a paraphrase family; a paraphrase needs a model to
  write it, and a model-written paraphrase can silently change a number, which would turn a
  robustness test into a corrupted-feed test. Unit restatement is the paraphrase that can be proved
  exact, so it is the one used. The price of that choice is stated: wording is not varied.
* ``restyle`` — each item's header, ``[id] (source, credibility c, available t) claim``, rewritten
  as ``claim — source; credibility c; as of t; ref id``. Same fields, different layout.
* ``reorder+restate`` — the first two together, so the worst case covers a compound change.

**Taken from HELM** (``stanford-crfm/helm``, Apache-2.0; notice at
``argus/licenses/helm-APACHE-2.0.txt``):

* ``src/helm/benchmark/augmentations/perturbation.py:23-32`` — ``Perturbation.get_rng``: the
  generator is seeded from the instance id, or from ``str(seed) + instance.id`` when a seed is
  given, so every perturbation is reproducible per snapshot and differs across snapshots. Kept
  verbatim in :meth:`Perturbation.rng`.
* ``src/helm/benchmark/metrics/metric.py:253-315`` — ``compute_worst_case_metrics``: for each
  instance, the score on the original is merged into each perturbation's score and the **minimum**
  is kept, alongside a ``before`` figure computed on the original alone. Rebuilt in
  :func:`worst_case`: per snapshot, the unperturbed runs are the original, each perturbation is
  merged with them, the minimum is the worst case, and the aggregate is always published as a
  ``before``/``worst`` pair.

**What was changed, and why.**

1. *The score is agreement with the desk's own unperturbed decision, not accuracy.* HELM scores
   against gold references. A trading decision has no gold answer at decision time, so the
   reference is the modal action over the unperturbed runs, and a run scores 1 when it took that
   action. This makes the test a pure invariance test, which is what HELM's worst-case metric is
   for (its own docstring: *"reason about the invariances of a model"*).
2. *The original is sampled several times.* HELM runs the original once, which is fine for a
   greedy decoder it can treat as deterministic. `eval/consistency.py` measured that this endpoint
   is not deterministic at ``temperature=0``, so a single original run would let ordinary sampling
   noise read as perturbation sensitivity. The unperturbed runs are therefore the **resample
   baseline** and every figure below is reported next to it.
3. *A second comparison HELM does not have: pairwise disagreement.* The HELM worst case over
   ``n + k`` runs is harsher than over ``n`` runs for noise alone, simply because it has more draws.
   So the report also gives the rate at which two unperturbed runs disagree and the rate at which a
   perturbed run disagrees with an unperturbed one. Only the excess of the second over the first is
   attributable to the perturbation.
4. *Abstentions carry no side.* The decision contract makes ``side`` mandatory, and on a NO_TRADE
   the model fills it arbitrarily (``Intent.lean``'s docstring: BUY on 53 of 53 settled
   abstentions). `eval/consistency.py` keeps it in the action tuple; here a zero-quantity decision
   has side ``-``, so a coin-flip on a field that moves no money is not counted as a flip.

**What is under test.** The Meta-PM's first, unconstrained decision (`MetaPM.decide`), with the
production system prompt, the production thinking budget for routine cycles (LOW) and the
production token cap (900). It is the one call that decides, so it is where invariance matters. The
analyst panel, the debate and the Constitution are **not** in the loop: each would multiply the
model calls per decision by four to six against a fixed budget, and the panel lines they add are
themselves model output, which would perturb the input a second time. The snapshots are recorded
without them and say so (``Snapshot.scope``).

**Where the snapshots come from.** ARGUS never persisted the evidence behind a live decision
(`agents/desk.py`, ``DeskRun.evidence_sources`` docstring), so there was no recorded frame to
replay.
:func:`record` builds them from the same public, point-in-time feeds the live cycle reads
(`paper/runner.run_once`): the venue ticker, filings and headlines, consensus, the Treasury curve,
the crypto risk-appetite reading, the VIX, and FINRA short volume and halts. It calls no model. The
frames are written to ``data/robustness_snapshots.json`` and every eval here replays those exact
bytes, so a figure can be reproduced against the frame it was measured on.

**The call budget is shared and enforced in code.** S1, S2 (`eval/vocab_stress.py`) and S3
(`eval/feedbugged.py`) draw on one ceiling of :data:`QWEN_CAP` requests. :class:`CappedQwen` counts
every request it sends — retries included, because a retry spends the key exactly as a first attempt
does — and refuses the next one at the ceiling. What the ceiling cut short is recorded as not run,
never estimated.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Any

from argus.agents.meta_pm import MarketFrame, MetaPM, deliberation_cost_bps
from argus.eval.artefact import write
from argus.llm.base import ChatModel
from argus.llm.qwen import Completion, QwenClient, QwenError, Thinking, TokenBudget
from argus.proof.autonomy import AutonomyProof
from argus.truth.clocks import SessionPhase, SessionState

DATA = Path(__file__).resolve().parents[3] / "data"
SNAPSHOTS_PATH = DATA / "robustness_snapshots.json"
BASELINE_PATH = DATA / "robustness_baseline.json"
REPORT_PATH = DATA / "perturbation_robustness.json"

QWEN_CAP = 80
"""Model requests allowed across S1, S2 and S3 together. A ceiling on a finite key, not a target."""

BUDGET_ARTEFACTS: tuple[str, ...] = (
    "robustness_baseline.json", "perturbation_robustness.json", "vocab_stress.json",
    "feedbugged.json",
)
"""Every artefact that spends from :data:`QWEN_CAP`. Each records its own ``qwen_requests``."""

CLEAN_RUNS = 3
"""Unperturbed decisions per snapshot. The floor `eval/consistency.MIN_RUNS` sets for the same
reason: two runs report 0% or 100% agreement and nothing between."""

PM_MAX_TOKENS = 900
"""The desk's own cap for the decision call (`agents/desk.TradingDesk.__init__`)."""

PM_THINKING = Thinking.LOW
"""The budget routine paper cycles run at (`paper/runner.run_once`)."""

NO_SIDE = "-"

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "NVDAUSDT", "TSLAUSDT", "METAUSDT", "COINUSDT", "MSTRUSDT", "QQQUSDT",
)
"""The six frames the budget affords, chosen for spread rather than taken in list order: two
mega-cap single names, one with an earnings-heavy frame, two whose prices move with crypto, and the
index fund, whose frame carries no company filings at all."""

SNAPSHOT_SCOPE = (
    "Meta-PM frame from live public feeds; no analyst panel, debate, memory or mandate block"
)
"""What a recorded frame leaves out, carried on every snapshot so no figure outlives its caveat."""


# --- the decision, reduced to what agreement is defined over -------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    """One decision, as the fields robustness is scored on."""

    verdict: str
    side: str
    quantity: str
    lean: str
    confidence: float
    thesis: str
    reasoning: str
    invalidation: tuple[str, ...] = ()

    @property
    def action(self) -> tuple[str, str, str]:
        """Verdict, side and size — with no side on a decision that moves no money."""
        side = self.side if Decimal(self.quantity) > 0 else NO_SIDE
        return (self.verdict, side, self.quantity)

    @property
    def text(self) -> str:
        """Everything the model wrote, for checks that read its words."""
        return " ".join([self.thesis, self.reasoning, *self.invalidation])

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "side": self.side, "quantity": self.quantity,
            "action": list(self.action), "lean": self.lean,
            "confidence": round(self.confidence, 4), "thesis": self.thesis,
            "reasoning": self.reasoning, "invalidation": list(self.invalidation),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Outcome:
        return cls(
            verdict=str(row["verdict"]), side=str(row["side"]), quantity=str(row["quantity"]),
            lean=str(row.get("lean", "none")), confidence=float(row.get("confidence", 0.0)),
            thesis=str(row.get("thesis", "")), reasoning=str(row.get("reasoning", "")),
            invalidation=tuple(str(x) for x in row.get("invalidation", [])),
        )

    @classmethod
    def of(cls, proof: AutonomyProof) -> Outcome:
        """The model's first, unconstrained decision — the one :meth:`MetaPM.decide` returns."""
        intent = proof.llm_original_intent
        return cls(
            verdict=str(intent.verdict), side=str(intent.side), quantity=str(intent.quantity),
            lean=intent.lean, confidence=intent.stated_confidence, thesis=intent.thesis,
            reasoning=proof.llm_original_reasoning, invalidation=intent.invalidation,
        )


# --- the recorded frame --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One market frame, recorded once and replayed byte for byte.

    Every field is what `MarketFrame` needs, and nothing is recomputed on replay: the session state
    is stored rather than re-derived, because re-deriving it from ``as_of`` would depend on the
    holiday calendar in force on the day of the replay.
    """

    id: str
    symbol: str
    as_of: datetime
    phase: str
    hours_to_discovery: float
    nav_age_seconds: float | None
    token_price: Decimal
    position: Decimal
    round_trip_bps: Decimal
    deliberation_bps: Decimal
    hedge_menu: tuple[dict[str, str], ...]
    evidence: tuple[str, ...]
    scope: str = SNAPSHOT_SCOPE

    @property
    def session(self) -> SessionState:
        return SessionState(
            phase=SessionPhase(self.phase), as_of=self.as_of,
            hours_to_next_discovery=self.hours_to_discovery,
            nav_age_seconds=self.nav_age_seconds,
        )

    def frame(self) -> MarketFrame:
        return MarketFrame(
            symbol=self.symbol, as_of=self.as_of, session=self.session,
            token_price=self.token_price, position_quantity=self.position,
            round_trip_bps=self.round_trip_bps, hedge_menu=self.hedge_menu,
            evidence=self.evidence, deliberation_bps=self.deliberation_bps,
        )

    def with_evidence(self, evidence: Sequence[str], *, symbol: str | None = None) -> Snapshot:
        return replace(
            self, evidence=tuple(evidence), symbol=self.symbol if symbol is None else symbol,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "symbol": self.symbol, "as_of": self.as_of.isoformat(),
            "phase": self.phase, "hours_to_discovery": self.hours_to_discovery,
            "nav_age_seconds": self.nav_age_seconds, "token_price": str(self.token_price),
            "position": str(self.position), "round_trip_bps": str(self.round_trip_bps),
            "deliberation_bps": str(self.deliberation_bps),
            "hedge_menu": [dict(h) for h in self.hedge_menu],
            "evidence": list(self.evidence), "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Snapshot:
        nav = row.get("nav_age_seconds")
        return cls(
            id=str(row["id"]), symbol=str(row["symbol"]),
            as_of=datetime.fromisoformat(str(row["as_of"])), phase=str(row["phase"]),
            hours_to_discovery=float(row["hours_to_discovery"]),
            nav_age_seconds=None if nav is None else float(nav),
            token_price=Decimal(str(row["token_price"])), position=Decimal(str(row["position"])),
            round_trip_bps=Decimal(str(row["round_trip_bps"])),
            deliberation_bps=Decimal(str(row["deliberation_bps"])),
            hedge_menu=tuple({str(k): str(v) for k, v in h.items()} for h in row["hedge_menu"]),
            evidence=tuple(str(e) for e in row["evidence"]),
            scope=str(row.get("scope", SNAPSHOT_SCOPE)),
        )


def load_snapshots(path: Path = SNAPSHOTS_PATH) -> list[Snapshot]:
    import json

    blob = json.loads(path.read_text(encoding="utf-8"))
    return [Snapshot.from_dict(row) for row in blob["snapshots"]]


def record(
    symbols: Sequence[str], *, path: Path = SNAPSHOTS_PATH,
) -> list[Snapshot]:  # pragma: no cover - reads live public feeds
    """Build one frame per symbol from the live cycle's own public feeds. Calls no model.

    Mirrors the evidence assembly in `paper/runner.run_once` for every feed that needs no key and
    no model: the ticker line, `market.evidence.gather` (filings, headlines, fundamentals),
    consensus, the Treasury curve, the crypto risk-appetite reading, the VIX, and FINRA short volume
    and halts, plus the same feed-health line. Bitget's Skills are left out because a Skill probe
    that times out costs ten seconds per symbol and answers only for now; that absence is written
    into the frame's feed line, not hidden.
    """
    from argus.cost.model import CostModel
    from argus.market.bitget import fetch_rtokens
    from argus.market.estimates import EstimatesSource
    from argus.market.evidence import gather, underlying_ticker
    from argus.market.fundamentals import FundamentalsSource
    from argus.market.macro import MacroError, fear_greed_evidence, fetch_fear_greed
    from argus.market.macro import evidence as curve_evidence
    from argus.market.macro import latest as latest_curve
    from argus.market.microstructure import MicrostructureError, fetch_halts, fetch_short_volume
    from argus.market.microstructure import evidence as micro_evidence
    from argus.market.volatility import VolatilityError
    from argus.market.volatility import evidence as vix_evidence
    from argus.market.volatility import fetch as fetch_vix
    from argus.paper.runner import _hedge_surface
    from argus.truth.clocks import DualClock
    from argus.truth.evidence import Evidence

    now = datetime.now(UTC)
    clock = DualClock()
    cost = CostModel.bitget_perp()
    tickers = fetch_rtokens()
    shared: list[Evidence] = []
    status: list[str] = []
    try:
        shared.extend(curve_evidence(latest_curve(), as_of=now))
        status.append("treasury curve: ok")
    except MacroError as exc:
        status.append(f"treasury curve: unavailable ({str(exc)[:60]})")
    try:
        shared.extend(fear_greed_evidence(fetch_fear_greed(), as_of=now))
        status.append("fear-greed: ok (crypto-wide)")
    except MacroError as exc:
        status.append(f"fear-greed: unavailable ({str(exc)[:60]})")
    try:
        shared.extend(vix_evidence(fetch_vix(), as_of=now))
        status.append("vix: ok")
    except VolatilityError as exc:
        status.append(f"vix: unavailable ({str(exc)[:60]})")
    wanted = frozenset(underlying_ticker(s) for s in symbols)
    try:
        short = fetch_short_volume(wanted=wanted)
        halts = fetch_halts()
    except MicrostructureError as exc:
        short, halts = {}, ()
        status.append(f"finra/nasdaq: unavailable ({str(exc)[:60]})")

    out: list[Snapshot] = []
    for symbol in symbols:
        ticker = tickers.get(symbol)
        if ticker is None:
            continue
        session = clock.state(now, nav_age_seconds=(now - ticker.fetched_at).total_seconds())
        evidence: list[Evidence] = [Evidence(
            id=f"mkt-{symbol}",
            claim=(
                f"{symbol} last {ticker.last}, 24h change {ticker.change_24h}, "
                f"quoted spread {ticker.spread_bps:.2f}bps, 24h base volume {ticker.base_volume}"
            ),
            source="news", available_at=ticker.fetched_at, credibility=1.0,
        )]
        gathered = gather(symbol, as_of=now, fundamentals=FundamentalsSource())
        evidence.extend(gathered.evidence)
        underlying = underlying_ticker(symbol)
        consensus, consensus_status = EstimatesSource().evidence(underlying, as_of=now)
        evidence.extend(consensus)
        evidence.extend(shared)
        evidence.extend(micro_evidence(
            ticker=underlying, as_of=now, short=short.get(underlying), halts=halts,
        ))
        feeds = [*gathered.status, *consensus_status, *status,
                 "bitget skills: not called for this recording"]
        evidence.append(Evidence(
            id=f"feeds-{symbol}", claim="evidence feeds: " + "; ".join(feeds),
            source="news", available_at=now, credibility=0.1,
        ))
        hedges = _hedge_surface(session.is_anchor_asleep, symbol)
        menu = tuple(
            {
                "instrument": c.instrument,
                "risk_reduction": str(round(c.effective_risk_reduction, 4)),
                "cost_bps": str(c.all_in_cost_bps),
                "execution_probability": str(c.execution_probability),
                "efficiency": str(round(c.risk_neutralisation_efficiency, 4)),
            }
            for c in hedges.menu
        )
        deliberation = round(deliberation_cost_bps(
            session, thinking=PM_THINKING, annualised_vol=Decimal("0.45"),
        ), 2)
        out.append(Snapshot(
            id=f"{symbol}-{now.strftime('%Y%m%dT%H%M%SZ')}", symbol=symbol, as_of=now,
            phase=str(session.phase), hours_to_discovery=session.hours_to_next_discovery,
            nav_age_seconds=session.nav_age_seconds, token_price=ticker.last,
            position=Decimal("0"), round_trip_bps=cost.round_trip_bps(),
            deliberation_bps=deliberation, hedge_menu=menu,
            evidence=tuple(e.render() for e in evidence),
        ))
    write(path, {
        "recorded_at": now.isoformat(),
        "method": (
            "live public feeds read as in paper/runner.run_once, no model called; each frame is "
            "replayed byte for byte by eval/perturbations, eval/vocab_stress and eval/feedbugged"
        ),
        "snapshots": [s.as_dict() for s in out],
    })
    return out


# --- perturbations (HELM perturbation.py:23-32) --------------------------------------------------


class Perturbation(ABC):
    """A meaning-preserving rewrite of a snapshot's evidence."""

    name: str

    def rng(self, snapshot: Snapshot, seed: int | None = None) -> Random:
        """HELM ``Perturbation.get_rng``, verbatim in behaviour: seeded from the instance id, or
        from ``str(seed) + id`` when a seed is given (`perturbation.py:23-32`)."""
        return Random(snapshot.id if seed is None else str(seed) + snapshot.id)

    @abstractmethod
    def apply(self, snapshot: Snapshot, seed: int | None = None) -> Snapshot:
        """A new snapshot with the evidence rewritten; the original is never mutated."""


class Reorder(Perturbation):
    name = "reorder"

    def apply(self, snapshot: Snapshot, seed: int | None = None) -> Snapshot:
        lines = list(snapshot.evidence)
        rng = self.rng(snapshot, seed)
        # A shuffle that returns the original order is not a perturbation. Reshuffle a bounded
        # number of times; a one-line frame genuinely has only one order and is returned as is.
        for _ in range(16):
            rng.shuffle(lines)
            if tuple(lines) != snapshot.evidence or len(set(lines)) < 2:
                break
        return snapshot.with_evidence(lines)


_UNIT = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)\s?(bps|%)(?![\w])")
_BIG_INT = re.compile(r"(?<![\w.:/-])(\d{5,})(?![\w.:/-])")


def _plain(value: Decimal) -> str:
    """A decimal written without exponent or trailing zeros, sign kept."""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def restate_units(text: str) -> tuple[str, int]:
    """Every bps figure as a percent and every percent as bps, exactly; big integers grouped.

    Returns the rewritten text and the number of figures restated. ``12.4bps`` becomes ``0.124%``
    and ``+0.43%`` becomes ``+43bps``: the same number, times or divided by one hundred, in
    ``Decimal`` so no float rounding can creep in. Integers of five or more digits gain thousands
    separators. Dates, times, identifiers and URLs are excluded by the lookarounds.
    """
    edits = 0

    def unit(match: re.Match[str]) -> str:
        nonlocal edits
        raw, kind = match.group(1), match.group(2)
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return match.group(0)
        edits += 1
        sign = "+" if raw.startswith("+") else ""
        if kind == "bps":
            return f"{sign}{_plain(value / 100)}%"
        return f"{sign}{_plain(value * 100)}bps"

    def group(match: re.Match[str]) -> str:
        nonlocal edits
        edits += 1
        return f"{int(match.group(1)):,}"

    out = _UNIT.sub(unit, text)
    out = _BIG_INT.sub(group, out)
    return out, edits


class UnitRestate(Perturbation):
    name = "unit_restate"

    def apply(self, snapshot: Snapshot, seed: int | None = None) -> Snapshot:
        return snapshot.with_evidence([restate_units(line)[0] for line in snapshot.evidence])


_HEADER = re.compile(
    r"^\[(?P<id>[^\]]*)\] \((?P<source>[^,]*), credibility (?P<cred>[0-9.]+), "
    r"available (?P<at>[^)]*)\) (?P<claim>.*)$",
    re.DOTALL,
)


def split_evidence(line: str) -> tuple[int, str]:
    """(offset of the claim within the line, the claim) for a rendered `Evidence` line.

    A line without the standard header is all claim. Used wherever a rewrite must touch the claim
    and never the id, source, credibility or timestamp.
    """
    m = _HEADER.match(line)
    if m is None:
        return 0, line
    return m.start("claim"), m["claim"]


def restyle(line: str) -> str:
    """`Evidence.render`'s header, laid out differently with every field kept."""
    m = _HEADER.match(line)
    if m is None:
        return line
    return (
        f"{m['claim']} — source {m['source']}; credibility {m['cred']}; as of {m['at']}; "
        f"ref {m['id']}"
    )


class Restyle(Perturbation):
    name = "restyle"

    def apply(self, snapshot: Snapshot, seed: int | None = None) -> Snapshot:
        return snapshot.with_evidence([restyle(line) for line in snapshot.evidence])


class ReorderRestate(Perturbation):
    name = "reorder+restate"

    def apply(self, snapshot: Snapshot, seed: int | None = None) -> Snapshot:
        return UnitRestate().apply(Reorder().apply(snapshot, seed), seed)


PERTURBATIONS: tuple[Perturbation, ...] = (Reorder(), UnitRestate(), Restyle(), ReorderRestate())


# --- the seat under test -------------------------------------------------------------------------


class CallCapReached(QwenError):
    """The shared request ceiling is spent. Raised before the request, never after."""


class CappedQwen(QwenClient):
    """The production client with its cache off and a hard request ceiling.

    Counting happens in :meth:`complete`, which `complete_json` calls once per attempt, so a
    validation retry and a truncation retry are each counted — they spend the key like any call.
    The cache is off because a cached answer is not a decision (`eval/consistency.py`).
    """

    def __init__(self, *, cap: int, token_limit: int = 600_000) -> None:
        super().__init__(budget=TokenBudget(limit=token_limit), cache=False)
        self.cap = cap
        self.requests = 0

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        if self.requests >= self.cap:
            raise CallCapReached(f"request ceiling of {self.cap} reached")
        self.requests += 1
        return super().complete(messages, **kwargs)


@dataclass
class Seat:
    """The Meta-PM, asked once per frame, with every refusal recorded rather than retried."""

    model: ChatModel
    requests: Callable[[], int]
    failures: list[str] = field(default_factory=list)
    decisions: int = 0

    def decide(self, snapshot: Snapshot, decision_id: str) -> Outcome | None:
        """The decision, or ``None`` when the ceiling or the endpoint refused. Never a guess."""
        pm = MetaPM(self.model, max_tokens=PM_MAX_TOKENS, thinking=PM_THINKING)
        try:
            proof = pm.decide(snapshot.frame(), decision_id=decision_id)
        except CallCapReached:
            self.failures.append(f"{decision_id}: not run, request ceiling reached")
            return None
        except QwenError as exc:
            self.failures.append(f"{decision_id}: model error ({str(exc)[:120]})")
            return None
        self.decisions += 1
        return Outcome.of(proof)


def live_seat(cap: int) -> Seat:  # pragma: no cover - builds the live client
    client = CappedQwen(cap=cap)
    return Seat(model=client, requests=lambda: client.requests)


def spent(data: Path = DATA, *, excluding: str = "") -> int:
    """Requests already spent from :data:`QWEN_CAP` by the artefacts on disk."""
    import json

    total = 0
    for name in BUDGET_ARTEFACTS:
        if name == excluding:
            continue
        path = data / name
        if not path.exists():
            continue
        try:
            total += int(json.loads(path.read_text(encoding="utf-8")).get("qwen_requests", 0))
        except (ValueError, OSError):
            continue
    return total


# --- the resample baseline, shared by S1, S2 and S3 ----------------------------------------------


Key = Callable[[Outcome], Any]


def ACTION(o: Outcome) -> Any:
    """Verdict, side and size: the level that moves money."""
    return o.action


def LEAN(o: Outcome) -> Any:
    """The directional view the desk states even when it stands aside (`Intent.lean`).

    Scored beside the action because an abstaining desk's action is pinned by the hurdle and can
    look invariant while its view of the market flips; `eval/consistency.py` found exactly that
    split on its first live run, with the lean stable and the action not.
    """
    return o.lean


@dataclass(frozen=True, slots=True)
class Baseline:
    """The unperturbed decisions per snapshot: what every perturbation is scored against."""

    runs: dict[str, tuple[Outcome, ...]]

    def reference(self, snapshot_id: str, key: Key | None = None) -> Any:
        """The modal unperturbed value of ``key`` (the action by default); ties go to the
        earliest run. ``None`` when the snapshot has no unperturbed run."""
        read = key or ACTION
        outcomes = self.runs.get(snapshot_id, ())
        if not outcomes:
            return None
        counts = Counter(read(o) for o in outcomes)
        best = max(counts.values())
        return next(read(o) for o in outcomes if counts[read(o)] == best)

    def unanimous(self, snapshot_id: str, key: Key | None = None) -> bool:
        read = key or ACTION
        return len({read(o) for o in self.runs.get(snapshot_id, ())}) == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            sid: {
                "reference_action": None if self.reference(sid) is None
                else list(self.reference(sid) or ()),
                "unanimous": self.unanimous(sid),
                "runs": [o.as_dict() for o in outs],
            }
            for sid, outs in self.runs.items()
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> Baseline:
        return cls(runs={
            sid: tuple(Outcome.from_dict(r) for r in row["runs"])
            for sid, row in blob.items()
        })


def collect_baseline(
    snapshots: Sequence[Snapshot], seat: Seat, *, runs: int = CLEAN_RUNS,
) -> Baseline:
    """Each snapshot, unperturbed, ``runs`` times. Round-robin, so a ceiling hit mid-way costs
    every snapshot its last run rather than costing the last snapshot all of its runs."""
    got: dict[str, list[Outcome]] = {s.id: [] for s in snapshots}
    for attempt in range(runs):
        for snap in snapshots:
            outcome = seat.decide(snap, f"robust-clean-{snap.id}-{attempt}")
            if outcome is not None:
                got[snap.id].append(outcome)
    return Baseline(runs={sid: tuple(outs) for sid, outs in got.items() if outs})


def load_baseline(path: Path = BASELINE_PATH) -> Baseline:
    import json

    return Baseline.from_dict(json.loads(path.read_text(encoding="utf-8"))["baseline"])


# --- scoring (HELM metric.py:253-315) ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """One perturbed decision on one snapshot."""

    snapshot_id: str
    condition: str
    outcome: Outcome
    edits: int = 0
    """How many things the perturbation actually changed. Zero means the run was a resample."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id, "condition": self.condition, "edits": self.edits,
            "decision": self.outcome.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class WorstCase:
    """HELM's before/worst pair, plus the pairwise noise comparison HELM does not need."""

    snapshots: int
    before: float | None
    """Mean agreement of the unperturbed runs with their own reference (HELM ``computed_on=
    original``)."""
    resample_worst: float | None
    """Share of snapshots whose unperturbed runs never left the reference. Noise alone."""
    worst: float | None
    """Share of snapshots where no run — unperturbed or perturbed — left the reference (HELM
    ``computed_on=worst`` over the ``robustness`` merge)."""
    per_condition_worst: dict[str, float]
    clean_pair_disagreement: tuple[int, int]
    """(disagreeing, total) over pairs of unperturbed runs on the same snapshot."""
    cross_pair_disagreement: tuple[int, int]
    """(disagreeing, total) over (perturbed, unperturbed) pairs on the same snapshot."""
    flips_beyond_noise: tuple[str, ...]
    """Perturbed flips on snapshots whose unperturbed runs were unanimous — the only flips the
    resample baseline cannot explain."""
    flips_within_noise: tuple[str, ...]

    @property
    def excess_disagreement(self) -> float | None:
        c_bad, c_all = self.clean_pair_disagreement
        x_bad, x_all = self.cross_pair_disagreement
        if not c_all or not x_all:
            return None
        return x_bad / x_all - c_bad / c_all

    def as_dict(self) -> dict[str, Any]:
        def rate(pair: tuple[int, int]) -> float | None:
            return None if not pair[1] else round(pair[0] / pair[1], 4)

        def pct(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "snapshots": self.snapshots,
            "before": pct(self.before),
            "resample_worst": pct(self.resample_worst),
            "worst": pct(self.worst),
            "per_condition_worst": {k: round(v, 4) for k, v in self.per_condition_worst.items()},
            "clean_pair_disagreement": {
                "disagreeing": self.clean_pair_disagreement[0],
                "pairs": self.clean_pair_disagreement[1],
                "rate": rate(self.clean_pair_disagreement),
            },
            "cross_pair_disagreement": {
                "disagreeing": self.cross_pair_disagreement[0],
                "pairs": self.cross_pair_disagreement[1],
                "rate": rate(self.cross_pair_disagreement),
            },
            "excess_disagreement": pct(self.excess_disagreement),
            "flips_beyond_noise": list(self.flips_beyond_noise),
            "flips_within_noise": list(self.flips_within_noise),
        }


def worst_case(baseline: Baseline, cells: Sequence[Cell], key: Key | None = None) -> WorstCase:
    """Score perturbed decisions against the unperturbed ones, worst case first.

    Adapted from HELM ``compute_worst_case_metrics`` (`metric.py:253-315`): the original's scores
    are merged into each perturbation's, the per-instance **minimum** is kept, and a ``before``
    figure on the original alone is published beside it. A snapshot with no unperturbed run has no
    reference and is excluded, never scored against a guess. Cells whose perturbation changed
    nothing (``edits == 0`` on a rewrite that should edit) are still scored — they are honest
    resamples — and are listed so a reader can discount them.

    ``key`` picks the level scored: the action (default) or the lean (:func:`LEAN`).
    """
    read = key or ACTION
    ids = [sid for sid in baseline.runs if baseline.reference(sid, read) is not None]
    before_scores: list[float] = []
    clean_worst: list[int] = []
    overall_worst: list[int] = []
    by_condition: dict[str, list[int]] = {}
    clean_bad = clean_all = cross_bad = cross_all = 0
    beyond: list[str] = []
    within: list[str] = []

    for sid in ids:
        ref = baseline.reference(sid, read)
        clean = baseline.runs[sid]
        clean_scores = [int(read(o) == ref) for o in clean]
        before_scores.append(sum(clean_scores) / len(clean_scores))
        clean_worst.append(min(clean_scores))
        mine = [c for c in cells if c.snapshot_id == sid]
        perturbed_scores = [int(read(c.outcome) == ref) for c in mine]
        overall_worst.append(min(clean_scores + perturbed_scores))
        for cell, score in zip(mine, perturbed_scores, strict=True):
            by_condition.setdefault(cell.condition, []).append(min([*clean_scores, score]))
            if not score:
                label = f"{sid}/{cell.condition}: {_show(ref)} -> {_show(read(cell.outcome))}"
                (beyond if baseline.unanimous(sid, read) else within).append(label)
        for a, b in combinations(clean, 2):
            clean_all += 1
            clean_bad += int(read(a) != read(b))
        for cell in mine:
            for o in clean:
                cross_all += 1
                cross_bad += int(read(cell.outcome) != read(o))

    def share(values: list[int] | list[float]) -> float | None:
        return None if not values else sum(values) / len(values)

    return WorstCase(
        snapshots=len(ids),
        before=share(before_scores),
        resample_worst=share(clean_worst),
        worst=share(overall_worst),
        per_condition_worst={k: sum(v) / len(v) for k, v in sorted(by_condition.items())},
        clean_pair_disagreement=(clean_bad, clean_all),
        cross_pair_disagreement=(cross_bad, cross_all),
        flips_beyond_noise=tuple(beyond),
        flips_within_noise=tuple(within),
    )


def _show(value: Any) -> str:
    return " ".join(value) if isinstance(value, tuple) else str(value)


def verdict(score: WorstCase, *, what: str, cells: int) -> str:
    """The finding in words, with the noise baseline beside every number. Never a mean alone."""
    if score.snapshots == 0 or score.before is None:
        return f"UNDEFINED: no snapshot had an unperturbed decision to score {what} against"
    head = (
        f"{score.snapshots} snapshot(s), {cells} {what} decision(s). Unperturbed agreement "
        f"(before) {score.before:.0%}; resample worst case {score.resample_worst:.0%}; worst case "
        f"with {what} {score.worst:.0%}."
    )
    c_bad, c_all = score.clean_pair_disagreement
    x_bad, x_all = score.cross_pair_disagreement
    if c_all and x_all:
        head += (
            f" Pairwise disagreement: {c_bad}/{c_all} between unperturbed runs, {x_bad}/{x_all} "
            f"between a {what} run and an unperturbed one."
        )
    if score.flips_beyond_noise:
        return (
            f"{head} {len(score.flips_beyond_noise)} flip(s) on snapshots whose unperturbed runs "
            f"were unanimous, so noise does not explain them: "
            f"{'; '.join(score.flips_beyond_noise)}"
        )
    if score.flips_within_noise:
        return (
            f"{head} Every flip landed on a snapshot whose unperturbed runs already disagreed, so "
            f"none is attributable to {what} beyond sampling noise"
        )
    return (
        f"{head} No {what} run left its snapshot's reference action. With this few snapshots that "
        f"is an absence of observed flips, not a bound on the flip rate"
    )


# --- S1 ------------------------------------------------------------------------------------------


def perturbed_cells(
    snapshots: Sequence[Snapshot], seat: Seat,
    perturbations: Sequence[Perturbation] = PERTURBATIONS, *, seed: int = 0,
) -> list[Cell]:
    """Each snapshot under each perturbation, once. Round-robin by perturbation for the same
    reason as :func:`collect_baseline`."""
    cells: list[Cell] = []
    for p in perturbations:
        for snap in snapshots:
            changed = p.apply(snap, seed)
            edits = sum(a != b for a, b in zip(changed.evidence, snap.evidence, strict=True))
            if isinstance(p, Reorder):
                edits = int(changed.evidence != snap.evidence)
            outcome = seat.decide(changed, f"robust-{p.name}-{snap.id}")
            if outcome is not None:
                cells.append(Cell(snapshot_id=snap.id, condition=p.name, outcome=outcome,
                                  edits=edits))
    return cells


def report(baseline: Baseline, cells: Sequence[Cell], *, requests: int,
           failures: Sequence[str]) -> dict[str, Any]:
    score = worst_case(baseline, cells)
    lean = worst_case(baseline, cells, LEAN)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "item": "S1 worst-case decision consistency under seeded evidence perturbation",
        "source": (
            "HELM metric.py:253-315 (compute_worst_case_metrics), perturbation.py:23-32 "
            "(get_rng); Apache-2.0, notice at argus/licenses/helm-APACHE-2.0.txt"
        ),
        "seat": f"MetaPM.decide, production prompt, thinking {PM_THINKING}, max_tokens "
                f"{PM_MAX_TOKENS}, cache off",
        "perturbations": [p.name for p in PERTURBATIONS],
        "score": score.as_dict(),
        "verdict": verdict(score, what="perturbed", cells=len(cells)),
        "lean_score": lean.as_dict(),
        "lean_verdict": "Lean level. " + verdict(lean, what="perturbed", cells=len(cells)),
        "no_op_cells": [f"{c.snapshot_id}/{c.condition}" for c in cells if c.edits == 0],
        "not_run": list(failures),
        "qwen_requests": requests,
        "cells": [c.as_dict() for c in cells],
    }


def summary(data: Path = DATA) -> dict[str, Any]:
    """The three robustness results as console lines, read from the artefacts and never recomputed.

    The integration point for the research console (`lui/research.py` / `lui/server.py`, owned
    elsewhere): each line quotes a measured before/worst pair or says the eval has not run. An
    absent artefact is reported as absent, never as a pass.
    """
    import json

    def load(name: str) -> dict[str, Any] | None:
        path = data / name
        if not path.exists():
            return None
        try:
            blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        return blob

    lines: list[str] = []
    s1 = load("perturbation_robustness.json")
    lines.append(
        "S1 perturbation: not run" if s1 is None else f"S1 perturbation: {s1['verdict']}"
    )
    s2 = load("vocab_stress.json")
    if s2 is None:
        lines.append("S2 renamed vocabulary: not run")
    else:
        for layer, row in s2["layers"].items():
            lines.append(f"S2 renamed {layer} ({row['requirement']}): {row['verdict']}")
    s3 = load("feedbugged.json")
    if s3 is None:
        lines.append("S3 corrupted feed: not run")
    else:
        det = s3["detection"]
        rec = s3["error_recovery"]
        lines.append(
            f"S3 corrupted feed: {s3['conditions']['bugged']['moved']} of "
            f"{s3['conditions']['bugged']['decisions']} decision(s) moved; detection F1 "
            f"{det['f1']} (tp {det['tp']}, fp {det['fp']}, fn {det['fn']})"
        )
        lines.append(
            f"S3 injected reasoning: unchanged NR {rec['NR_unchanged_rate']}, CR "
            f"{rec['CR_unchanged_rate']}, IR {rec['IR_unchanged_rate']}; wrong direction "
            f"adopted {s3['conditions']['incorrect_reasoning'].get('adopted_wrong_direction')}"
        )
    requests = sum(
        int(b.get("qwen_requests", 0)) for b in (load(n) for n in BUDGET_ARTEFACTS) if b
    )
    return {"lines": lines, "qwen_requests": requests, "cap": QWEN_CAP}


def main() -> int:  # pragma: no cover - CLI, spends the model budget
    import argparse
    import json
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="S1: worst-case consistency under perturbation")
    parser.add_argument("--record", nargs="*", default=None,
                        help="record fresh snapshots for these symbols first (no model calls)")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    if args.record is not None:
        from argus.market.bitget import RTOKEN_SYMBOLS

        recorded = record(args.record or RTOKEN_SYMBOLS)
        print(f"recorded {len(recorded)} snapshot(s) to {SNAPSHOTS_PATH}")
    snaps = [s for s in load_snapshots() if s.symbol in set(args.symbols)]

    if not BASELINE_PATH.exists():
        left = QWEN_CAP - spent()
        seat = live_seat(left)
        base = collect_baseline(snaps, seat)
        write(BASELINE_PATH, {
            "generated_at": datetime.now(UTC).isoformat(),
            "runs_per_snapshot": CLEAN_RUNS,
            "snapshot_ids": [s.id for s in snaps],
            "baseline": base.as_dict(),
            "not_run": seat.failures,
            "qwen_requests": seat.requests(),
        })
        print(f"baseline: {seat.decisions} decision(s), {seat.requests()} request(s)")
    if args.baseline_only:
        return 0
    base = load_baseline()
    left = QWEN_CAP - spent(excluding=REPORT_PATH.name)
    seat = live_seat(left)
    cells = perturbed_cells(snaps, seat)
    blob = report(base, cells, requests=seat.requests(), failures=seat.failures)
    write(REPORT_PATH, blob)
    print(json.dumps(blob["score"], indent=2))
    print(blob["verdict"])
    print(f"requests this run: {seat.requests()}; total spent: {spent()} of {QWEN_CAP}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ACTION",
    "BASELINE_PATH",
    "CLEAN_RUNS",
    "DEFAULT_SYMBOLS",
    "LEAN",
    "PERTURBATIONS",
    "QWEN_CAP",
    "SNAPSHOTS_PATH",
    "Baseline",
    "CallCapReached",
    "CappedQwen",
    "Cell",
    "Outcome",
    "Perturbation",
    "Reorder",
    "ReorderRestate",
    "Restyle",
    "Seat",
    "Snapshot",
    "UnitRestate",
    "WorstCase",
    "collect_baseline",
    "load_baseline",
    "load_snapshots",
    "perturbed_cells",
    "record",
    "report",
    "restate_units",
    "restyle",
    "spent",
    "split_evidence",
    "summary",
    "verdict",
    "worst_case",
]
