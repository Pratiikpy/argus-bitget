"""Teacher forcing and a human baseline on the paper runner: proving the harness, not the model.

When a paper-trading number looks wrong, the instinct is to blame the model. It is the harness —
the runner, the Constitution, the protocol, the venue gate, the ledger, settlement, the scorer —
that turns a decision into that number, and nothing tested the harness on its own: every run went
through a model whose decisions nobody knew in advance, so a harness defect and a bad decision
looked the same. ``paper/runner.py``'s own history says what that costs: two trades fabricated on
the live record (seq 264, 265) survived because "every test fed this path an intent the
Constitution had already allowed".

**Teacher forcing.** The decision-maker's seat is given a scripted, known-correct sequence and the
*real* runner (`paper.runner.run_once`, unmodified) drives it end to end on a replayed market:
decide, then a mark cycle three hours later, then settlement twenty-five hours later. The script is
a hindsight oracle — each traded name is bought if it rose over the settlement window and sold if
it fell — so what a sound harness must report is known before the run: every trade right-way, each
gross and net P&L equal to arithmetic done here from the replayed prices and the published fee,
every abstention's counterfactual move equal to the price change, the chain intact. A harness that
reports anything else is broken, whatever the model would have done.

**A sabotage control, so the checks are not vacuous.** The same replay is run with one real defect
put into the harness — settlement handed the decision cycle's prices, the self-scoring bug the
runner's docstring exists to prevent — and the checks must fail. If they passed, they would not be
testing settlement at all.

**Human baseline.** The same replay, the same runner, the same scorer, with a person's decisions
(a JSONL file) in the decision seat and no model reasoning at all; two rule baselines a person
might follow are built in. This is the literal "baseline reproduced on the same harness" that the
OWNED condition in `eval/standing.py` asks for, rather than a separate script.

**What was taken, from where.**

* WebArena (``web-arena-x/webarena``, Apache-2.0; notice at
  ``argus/licenses/webarena-APACHE-2.0.txt``), ``agent/agent.py:45-97`` ``TeacherForcingAgent``:
  ``set_actions`` parses a reference action sequence, ``next_action`` returns the next scripted
  action instead of calling a model, and ``reset`` loads the reference from the task's config, so
  the environment and evaluator are validated decoupled from any LLM. Adapted, not copied:
  :class:`Seat` answers the desk's decision call for a symbol with that symbol's scripted
  decision, because the runner asks per symbol rather than per step, and it answers the analysts'
  calls with a fixed neutral view so the scripted decision is the only one that matters. An
  unparseable scripted action there becomes a no-op; here the validator's complaint is counted
  (ReAct's ``n_badcalls``) and returned, never silently repaired.
* OSWorld (``xlang-ai/OSWorld``, Apache-2.0; notice at ``argus/licenses/osworld-APACHE-2.0.txt``),
  ``lib_run_single.py:83-106`` ``run_single_example_human``: the identical environment reset and
  the identical ``env.evaluate()`` run against a human operator, with no agent loop. Adapted: the
  human's decision takes the decision seat inside the same runner, because in ARGUS the risk layer
  and ledger that a human's decision must pass through are part of the environment being scored.

**What was rejected.** WebArena's ``early_stop`` loop detector and OSWorld's VM snapshotting: the
runner is one bounded cycle per call, with nothing to loop on and no machine state to revert.

**What the replay is made of, stated rather than implied.** Prices are real: Bitget hourly closes
from ``data/risk_layer_candles_fixture.json`` (frozen 2026-09-23). The instant is chosen by a fixed
rule (:func:`replay_market`). Instrument rules are real: the venue's published specifications for
the replayed contracts, frozen by :func:`freeze_instruments` into
``data/replay_instruments_2026-09-25.json``, so the venue gate runs. Stated constants, because a
close is all the fixture holds: a 2bps quoted spread, a 24h volume of one million units, a zero
funding rate (the median rToken rate, per ``paper/runner.py:_median_funding``). Every other feed
the runner reads — filings, headlines, the rates curve, fear and greed, VIX, short volume, Skills
— is reported unavailable, which exercises the runner's own degrade paths. Every outbound
connection is refused, no model key is present, and every file the runner writes goes to a scratch
directory. Every file this process opens for writing under ``data/`` is noted, and a replay that
wrote one fails its checks; a before-and-after listing of ``data/`` is reported as well, but other
processes write there, so the listing is context and the per-process note is the test.

*Finding, 2026-09-26 (data/replay_harness.json).* On the replayed instant (decided 2026-09-17 15:00
UTC, settled 25 hours later) teacher forcing passes all 16 checks: the 12 scripted decisions are
booked once each; the 9 trades settle at the hand-computed gross and net P&L to 0.0001 (the scorer's
net 711.2375 against 711.2376 by hand, win rate 8 of 9 — the ninth is a right-way trade whose move
did not cover the 16bps round trip); the oversized NVDA order is cut by the risk layer
(``unhedgeable_gap``) from 2,281.64 to 91.26 and booked at the cut size; the 3 abstentions are
marked at three hours and graded; a one-cent forgery is caught. The planted settlement defect fails
four checks. The buy-every-name baseline surfaced one property of the harness itself: twelve
orders in one cycle meet the venue gate's ten-orders-a-minute cap, the two it denies are booked as
``no_trade``, and they are then marked and graded as abstentions the decision-maker did not choose.
The report says so under ``findings``; changing how the runner books a denied order is the runner's
decision, not this module's.

    python -m argus.eval.replay_harness
    python -m argus.eval.replay_harness --human decisions.jsonl
    python -m argus.eval.replay_harness --freeze-instruments
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
import types
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from functools import partial
from pathlib import Path
from typing import Any

from argus.eval.artefact import write
from argus.eval.trace_audit import offline

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "replay_harness.json"
CANDLES_PATH = DATA / "risk_layer_candles_fixture.json"
INSTRUMENTS_PATH = DATA / "replay_instruments_2026-09-25.json"

REPLAY_SPREAD_BPS = Decimal("2")
"""The quoted spread every replayed ticker carries. The fixture holds closes only; stated."""

REPLAY_BASE_VOLUME = Decimal("1000000")
TAKER_BPS = Decimal("6")
"""Bitget's published taker fee, 0.06% a side — the figure the gold arithmetic below charges. It is
written here, not read from `cost/model.py`, so that the check does not share the code it checks."""

NOTIONAL = Decimal("2000")
"""What each scripted trade asks for, in quote currency; the quantity is this over the price,
rounded down to the contract's published step."""

MARK_AFTER = timedelta(hours=3)
SETTLE_AFTER = timedelta(hours=25)
"""The runner settles after 24 hours (``HOLD_HOURS``); 25 leaves an hour of margin."""

TOLERANCE = Decimal("0.0001")
"""The ledger rounds P&L and moves to four decimal places."""


TEACHER_ABSTAINS = ("QQQUSDT", "TQQQUSDT", "SQQQUSDT")
TEACHER_OVERSIZES = ("NVDAUSDT",)


class ReplayError(RuntimeError):
    """The replay could not be run as specified: a call the script does not cover, or no instant
    in the fixture that satisfies the selection rule."""


# --- the decisions --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision in the decision seat, in the Meta-PM's own answer format."""

    symbol: str
    verdict: str
    side: str = "buy"
    quantity: Decimal = Decimal("0")
    confidence: float = 0.6
    thesis: str = ""
    invalidation: tuple[str, ...] = ()
    lean: str = "none"
    lean_confidence: float = 0.0

    def response(self) -> dict[str, Any]:
        return {"verdict": self.verdict.upper(), "side": self.side, "quantity": str(self.quantity),
                "confidence": self.confidence, "thesis": self.thesis,
                "invalidation": list(self.invalidation), "lean": self.lean.upper(),
                "lean_confidence": self.lean_confidence}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Decision:
        """A human's decision from one JSONL row: ``symbol``, ``verdict`` required; the rest as in
        the Meta-PM's answer format."""
        if "symbol" not in row or "verdict" not in row:
            raise ReplayError(f"a decision row needs symbol and verdict: {dict(row)}")
        return cls(symbol=str(row["symbol"]), verdict=str(row["verdict"]).lower(),
                   side=str(row.get("side", "buy")).lower(),
                   quantity=Decimal(str(row.get("quantity", "0"))),
                   confidence=float(row.get("confidence", 0.6)),
                   thesis=str(row.get("thesis", "a human's decision")),
                   invalidation=tuple(str(x) for x in row.get("invalidation", ()) or ()),
                   lean=str(row.get("lean", "none")).lower(),
                   lean_confidence=float(row.get("lean_confidence", 0.0)))


NEUTRAL_VIEW: dict[str, Any] = {
    "signal": "neutral", "confidence": 0.5, "magnitude_bps": 0,
    "reasoning": "replay seat: a fixed neutral view, so the scripted decision alone decides",
    "counter_case": "none stated in a replay", "chain": [],
}
CRITIC_VIEW: dict[str, Any] = {
    "counter_case": "none stated in a replay", "weakest_link": "", "severity": 0.0,
    "already_refuted": False, "refuted_condition": "",
}


class Seat:
    """The decision-maker's seat, answered from a script (teacher forcing or a human).

    Implements what the desk asks of a model (`llm/base.ChatModel`). The decision and its
    revision after a binding constraint are answered with the symbol's scripted decision; the
    revision is clamped by the Meta-PM to what the risk layer permitted (`agents/meta_pm.py:1017`),
    which is part of what the replay tests. Analysts and the adversary get fixed views. Any other
    call raises :class:`ReplayError`: a call the script does not cover must not be answered by
    invention."""

    def __init__(self, decisions: Mapping[str, Decision], *, name: str) -> None:
        from argus.llm.qwen import TokenBudget

        self.name = name
        self.decisions = dict(decisions)
        self.budget: Any = TokenBudget(limit=1)
        self.calls: Counter[str] = Counter()
        self.complaints: list[str] = []
        """What the desk's validator said about a scripted answer (ReAct's ``n_badcalls``)."""

    def _symbol(self, messages: Sequence[Mapping[str, Any]]) -> Decision:
        """The decision asked for, read from the frame's own ``SYMBOL:`` header
        (`agents/meta_pm.MarketFrame.to_prompt_block`) — the frame names other contracts too (a
        benchmark, a hedge), so a search for any scripted name would be ambiguous."""
        text = chr(10).join(str(m.get("content", "")) for m in messages)
        named = set(re.findall(r"^SYMBOL: (\S+)$", text, re.M))
        found = [self.decisions[s] for s in named if s in self.decisions]
        if len(found) != 1:
            raise ReplayError(f"{self.name}: the decision call's frame names {sorted(named)}; "
                              f"the seat has a scripted decision for {len(found)} of them")
        return found[0]

    def complete_json(self, messages: list[dict[str, Any]], *,
                      required_keys: tuple[str, ...] = (),
                      validate: Callable[[dict[str, Any]], str] | None = None,
                      max_tokens: int = 0, attempts: int = 0, thinking: Any = None,
                      ) -> dict[str, Any]:
        if "verdict" in required_keys:
            role = "revision" if any("constrained by the risk layer" in str(m.get("content", ""))
                                     for m in messages) else "decision"
            response = self._symbol(messages).response()
            self.calls[role] += 1
            complaint = validate(response) if validate is not None else ""
            if complaint:
                self.complaints.append(complaint)
            return response
        if "counter_case" in required_keys:
            self.calls["adversary"] += 1
            return dict(CRITIC_VIEW)
        if "signal" in required_keys:
            self.calls["analyst"] += 1
            return dict(NEUTRAL_VIEW)
        self.calls["unscripted"] += 1
        raise ReplayError(f"{self.name}: an unscripted call wanting {required_keys}")

    def complete(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        self.calls["unscripted"] += 1
        raise ReplayError(f"{self.name}: an unscripted free-text call")


# --- the market -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Market:
    """Real closes at the three replay instants, per symbol."""

    decided_at: datetime
    marked_at: datetime
    settled_at: datetime
    prices: dict[str, tuple[Decimal, Decimal, Decimal]]
    """Symbol → (close at the decision, at the mark, at settlement)."""

    day_before: dict[str, tuple[Decimal, Decimal, Decimal]]
    """Symbol → (24h high, 24h low, close 24h before the decision), for the ticker's day fields."""

    def at(self, instant: datetime) -> int:
        return (self.decided_at, self.marked_at, self.settled_at).index(instant)

    def tickers(self, instant: datetime) -> dict[str, Any]:
        from argus.market.bitget import Ticker

        which = self.at(instant)
        half = REPLAY_SPREAD_BPS / Decimal("20000")
        out = {}
        for symbol, closes in self.prices.items():
            last = closes[which]
            high, low, before = self.day_before[symbol]
            out[symbol] = Ticker(
                symbol=symbol, last=last, bid=last * (1 - half), ask=last * (1 + half),
                high_24h=max(high, last), low_24h=min(low, last),
                change_24h=(last / before - 1) if which == 0 else Decimal("0"),
                base_volume=REPLAY_BASE_VOLUME, funding_rate=Decimal("0"), fetched_at=instant)
        return out


def replay_market(path: Path = CANDLES_PATH) -> Market:
    """The latest decision instant in the frozen history that is a Tuesday, Wednesday or Thursday
    at 15:00 UTC (US regular hours, anchor awake, both later instants on weekdays) with a close for
    every contract at the decision, three hours later and twenty-five hours later."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    series = {s: {datetime.fromisoformat(t): Decimal(str(c)) for t, c in rows}
              for s, rows in fixture["candles"].items()}
    common = set.intersection(*(set(v) for v in series.values()))
    for t0 in sorted(common, reverse=True):
        if t0.weekday() not in (1, 2, 3) or t0.hour != 15:
            continue
        needed = (t0 + MARK_AFTER, t0 + SETTLE_AFTER)
        if not all(t in common for t in needed):
            continue
        window = [t0 - timedelta(hours=h) for h in range(1, 25)]
        if not all(t in common for t in window):
            continue
        prices = {s: (v[t0], v[needed[0]], v[needed[1]]) for s, v in series.items()}
        day = {s: (max(v[t] for t in window), min(v[t] for t in window), v[window[-1]])
               for s, v in series.items()}
        return Market(decided_at=t0, marked_at=needed[0], settled_at=needed[1], prices=prices,
                      day_before=day)
    raise ReplayError("no instant in the frozen history satisfies the replay's selection rule")


def _instruments(path: Path = INSTRUMENTS_PATH) -> dict[str, Any]:
    from argus.execution.guard import Instrument

    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {str(r["symbol"]): Instrument.from_payload(r) for r in rows}


def freeze_instruments(symbols: Sequence[str], path: Path = INSTRUMENTS_PATH) -> int:
    """Read the venue's published specifications for ``symbols`` once and freeze the raw rows."""
    from argus.execution.guard import INSTRUMENTS_URL, PRODUCT_TYPE

    url = f"{INSTRUMENTS_URL}?{urllib.parse.urlencode({'category': PRODUCT_TYPE})}"
    request = urllib.request.Request(url, headers={"User-Agent": "argus/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode())
    rows = [r for r in payload.get("data") or [] if r.get("symbol") in set(symbols)]
    write(path, {"source": url, "fetched_at": datetime.now(UTC).isoformat(),
                 "note": "Bitget's published instrument rules, frozen for the replay harness",
                 "rows": rows})
    return len(rows)


def _step_quantity(symbol: str, price: Decimal, instruments: Mapping[str, Any],
                   notional: Decimal = NOTIONAL) -> Decimal:
    spec = instruments.get(symbol)
    places = int(spec.quantity_precision) if spec is not None else 2
    return (notional / price).quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)


# --- the scripts ----------------------------------------------------------------------------------


OVERSIZED = Decimal("500000")
"""The notional the teacher asks for on its one oversized order: far past any position cap, so the
risk layer must bind, the Meta-PM must be re-asked, and the ledger must record the reduced size."""


def oracle(market: Market, instruments: Mapping[str, Any], *,
           abstain: Sequence[str] = (), oversize: Sequence[str] = ()) -> dict[str, Decision]:
    """The teacher: trade every name in the direction it actually went over the settlement window;
    abstain on ``abstain``, leaning the way it went; ask for :data:`OVERSIZED` on ``oversize``, and
    ask for it again when the risk layer re-puts the decision, as a model that ignores the cap
    would."""
    out = {}
    for symbol, (p0, _p1, p2) in sorted(market.prices.items()):
        up = p2 > p0
        if symbol in abstain:
            out[symbol] = Decision(symbol, "no_trade", side="buy", confidence=0.6,
                                   thesis="teacher: stand aside, leaning with the realised move",
                                   lean="up" if up else "down", lean_confidence=0.7)
            continue
        out[symbol] = Decision(
            symbol, "trade", side="buy" if up else "sell",
            quantity=_step_quantity(symbol, p0, instruments,
                                    OVERSIZED if symbol in oversize else NOTIONAL),
            confidence=0.8,
            thesis="teacher: the realised move over the settlement window, known in advance",
            invalidation=(f"{symbol} settles on the other side of {p0}",),
            lean="up" if up else "down", lean_confidence=0.8)
    return out


def always_abstain(market: Market, instruments: Mapping[str, Any]) -> dict[str, Decision]:
    """A rule a cautious person might follow: never trade, and state no lean."""
    return {s: Decision(s, "no_trade", thesis="baseline: always abstain") for s in market.prices}


def buy_everything(market: Market, instruments: Mapping[str, Any]) -> dict[str, Decision]:
    """A rule an eager person might follow: buy the same notional of every name."""
    return {s: Decision(s, "trade", side="buy", quantity=_step_quantity(s, p[0], instruments),
                        confidence=0.6, thesis="baseline: buy every name",
                        invalidation=(f"{s} closes below its entry",), lean="up",
                        lean_confidence=0.6)
            for s, p in market.prices.items()}


def human(path: Path) -> Callable[[Market, Mapping[str, Any]], dict[str, Decision]]:
    """A person's decisions from a JSONL file, one row per contract."""
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    decisions = {d.symbol: d for d in (Decision.from_row(r) for r in rows)}
    return lambda market, instruments: {s: d for s, d in decisions.items()
                                        if s in market.prices}


# --- driving the real runner ----------------------------------------------------------------------


def _clock(instant: datetime) -> type[datetime]:
    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> Clock:
            moment = instant if tz is None else instant.astimezone(tz)
            return cls.fromtimestamp(moment.timestamp(), tz=moment.tzinfo)

    return Clock


@contextmanager
def _patched(owner: Any, **values: Any) -> Iterator[None]:
    saved = {name: getattr(owner, name) for name in values}
    for name, value in values.items():
        setattr(owner, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(owner, name, value)


def _unavailable(error: type[Exception], what: str) -> Callable[..., Any]:
    def refuse(*_a: Any, **_k: Any) -> Any:
        raise error(f"replay: {what} is not read in a replay")
    return refuse


@contextmanager
def _replay_world(market: Market, instant: datetime, seat: Seat, work: Path,
                  instruments: Mapping[str, Any],
                  sabotage: Callable[..., Any] | None = None) -> Iterator[None]:
    """The runner's inputs for one cycle, replayed; its outputs, redirected into ``work``."""
    from argus.market.evidence import Gathered
    from argus.market.macro import MacroError
    from argus.market.microstructure import MicrostructureError
    from argus.market.volatility import VolatilityError
    from argus.paper import chains, marks, runner

    def budgeted(*, budget: Any) -> Seat:
        seat.budget = budget
        return seat

    class Estimates:
        def evidence(self, ticker: str, *, as_of: datetime) -> tuple[list[Any], list[str]]:
            return [], ["consensus estimates: not read in a replay"]

    marks_path = work / "refusal_marks.jsonl"
    replay_marks = types.SimpleNamespace(
        due=partial(marks.due, path=marks_path), mark=partial(marks.mark, path=marks_path),
        MarkError=marks.MarkError)
    values: dict[str, Any] = {
        "datetime": _clock(instant),
        "fetch_rtokens": lambda: market.tickers(instant),
        "QwenClient": budgeted,
        "gather": lambda symbol, **_: Gathered(evidence=[], status=[
            "filings, headlines and social feeds: not read in a replay"]),
        "EstimatesSource": Estimates,
        "latest_curve": _unavailable(MacroError, "the rates curve"),
        "fetch_fear_greed": _unavailable(MacroError, "fear and greed"),
        "fetch_vix": _unavailable(VolatilityError, "VIX"),
        "fetch_short_volume": _unavailable(MicrostructureError, "short volume"),
        "fetch_specs": lambda: dict(instruments),
        "usable_probes": lambda health: [],
        "mirror_missing": lambda ids, symbol: [],
        "NOTES_PATH": work / "desk_notes.jsonl",
        "RISK_PATH": work / "risk_records.jsonl",
        "marks": replay_marks,
        # The replay ledger starts at seq 1, so it declares marking from its first row; the live
        # record is governed by the committed protocol's own seq (`runner._marking_governs_from`).
        "_marking_governs_from": lambda: 1,
    }
    if sabotage is not None:
        values["_settle_due"] = sabotage
    with _patched(runner, **values), _patched(chains, CHAINS_PATH=work / "causal_chains.jsonl"):
        yield


def _snapshot() -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(DATA)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in DATA.rglob("*") if p.is_file()}


@dataclass
class Replay:
    """One replay: the cycles' summaries, and the record the runner wrote."""

    name: str
    decisions: dict[str, Decision]
    cycles: list[dict[str, Any]] = field(default_factory=list)
    entries: list[Any] = field(default_factory=list)
    risk: list[dict[str, Any]] = field(default_factory=list)
    marks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, Any]] = field(default_factory=list)
    """The desk's per-decision check output (`paper/runner._write_notes`), read back from the
    scratch directory: where a gate that turned a scripted trade into no trade says why."""

    performance: dict[str, Any] = field(default_factory=dict)
    chain: dict[str, Any] = field(default_factory=dict)
    tamper_detected: bool | None = None
    seat_calls: dict[str, int] = field(default_factory=dict)
    complaints: list[str] = field(default_factory=list)
    refused_connections: int = 0
    data_writes: list[str] = field(default_factory=list)
    """Files under ``data/`` this process opened for writing during the replay, as
    `eval/trace_audit.offline` notes them. Must be empty: the runner's files are redirected."""

    seconds: float = 0.0


def replay(name: str, decisions: Mapping[str, Decision], market: Market,
           instruments: Mapping[str, Any], *,
           sabotage: Callable[[Any], Callable[..., Any]] | None = None) -> Replay:
    """Run the real runner three times over the replayed market with ``decisions`` in the seat."""
    from argus.eval.performance import evaluate_ledger
    from argus.paper import runner
    from argus.paper.ledger import PaperLedger

    seat = Seat(decisions, name=name)
    result = Replay(name=name, decisions=dict(decisions))
    work = Path(tempfile.mkdtemp(prefix=f"argus-replay-{name}-"))
    ledger_path = work / "paper_ledger.jsonl"
    started = time.perf_counter()
    try:
        with offline() as blocked:
            broken = None if sabotage is None else sabotage(market)
            for instant, symbols in ((market.decided_at, tuple(sorted(decisions))),
                                     (market.marked_at, ()), (market.settled_at, ())):
                with _replay_world(market, instant, seat, work, instruments, broken):
                    summary = runner.run_once(symbols=symbols, ledger_path=ledger_path,
                                              budget=1_000_000)
                result.cycles.append({"at": instant.isoformat(), "symbols": list(symbols),
                                      "summary": json.loads(json.dumps(summary, default=str))})
            result.refused_connections = blocked.attempts
            result.data_writes = sorted(set(blocked.writes))
        ledger = PaperLedger(path=ledger_path)
        result.entries = list(ledger.entries)
        result.chain = {k: str(v) for k, v in ledger.verify().items()}
        scored = evaluate_ledger(ledger)
        result.performance = {**scored.as_dict(), "win_rate": scored.win_rate}
        for name_, target in (("risk_records.jsonl", result.risk),
                              ("refusal_marks.jsonl", result.marks),
                              ("desk_notes.jsonl", result.notes)):
            path = work / name_
            if path.exists():
                target.extend(json.loads(x) for x in path.read_text("utf-8").splitlines()
                              if x.strip())
        # A verifier that passes a tampered record verifies nothing: one byte of the first row's
        # entry price is changed in a copy, and the copy must fail.
        copy = work / "tampered.jsonl"
        text = ledger_path.read_text(encoding="utf-8")
        first = result.entries[0] if result.entries else None
        if first is not None and f'"entry_price": "{first.entry_price}"' in text:
            forged = str(Decimal(first.entry_price) + Decimal("0.01"))
            copy.write_text(text.replace(f'"entry_price": "{first.entry_price}"',
                                         f'"entry_price": "{forged}"', 1), encoding="utf-8")
            result.tamper_detected = not bool(PaperLedger(path=copy).verify()["chain_intact"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    result.seat_calls = dict(seat.calls)
    result.complaints = list(seat.complaints)
    result.seconds = round(time.perf_counter() - started, 2)
    return result


def settle_on_decision_prices(market: Market) -> Callable[..., Any]:
    """The sabotage: settlement is handed the decision cycle's prices — a runner scoring itself on
    information the decision already had, the defect ``paper/runner.py``'s docstring names."""
    from argus.paper import runner

    real = runner._settle_due
    stale = market.tickers(market.decided_at)

    def settle(ledger: Any, tickers: Any, now: datetime) -> list[dict[str, object]]:
        return real(ledger, stale, now)

    return settle


# --- the gold, computed here ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


def _gold_net(qty: Decimal, side: str, p0: Decimal, p2: Decimal) -> tuple[Decimal, Decimal]:
    """Gross and net P&L of one settled trade, from first principles: the price change times the
    size and direction, less a taker fee and the quoted spread on each side."""
    direction = Decimal("1") if side.upper() == "BUY" else Decimal("-1")
    gross = (p2 - p0) * qty * direction
    entry_bps = (TAKER_BPS + REPLAY_SPREAD_BPS).quantize(Decimal("0.001"))
    cost = qty * p0 * entry_bps / 10000 + qty * p2 * (TAKER_BPS + REPLAY_SPREAD_BPS) / 10000
    return gross, gross - cost


_GATE_NOTE = re.compile(r"^\[guard\] DENIED\b|^\[protocol\] (?!within the committed protocol)")
"""A desk note in which a gate stopped an order: the venue gate's refusal (`execution/guard.py`
renders ``[guard] DENIED <rule>: <why>``) or a protocol ruling that applied
(`paper/protocol.py:598`; one that changed nothing reads "within the committed protocol")."""


def gate_reasons(run: Replay) -> dict[str, list[str]]:
    """For each scripted trade the ledger recorded as no trade, what stopped it: the risk layer's
    binding constraint when it cut the size to zero, and every gate note the desk wrote."""
    by_seq = {int(n["seq"]): [str(x) for x in n.get("notes") or []] for n in run.notes}
    risk = {str(r["symbol"]): r for r in run.risk}
    out: dict[str, list[str]] = {}
    for entry in run.entries:
        scripted = run.decisions.get(entry.symbol)
        if scripted is None or scripted.verdict != "trade" or entry.verdict != "no_trade":
            continue
        why = [x for x in by_seq.get(entry.seq, []) if _GATE_NOTE.search(x)]
        bound = risk.get(entry.symbol, {})
        if bound.get("intervened") and Decimal(str(bound.get("quantity_after", "1"))) == 0:
            why.insert(0, f"[risk layer] {bound.get('binding_constraint')} cut the size to 0")
        out[entry.symbol] = why
    return out


def validate(run: Replay, market: Market, *, oracle_run: bool) -> list[Check]:
    """What a sound harness must have reported for this replay, checked against the record."""
    checks: list[Check] = []
    decisions = [e for e in run.entries if not e.is_void]
    by_symbol = {e.symbol: e for e in decisions}
    checks.append(Check(
        "every scripted decision reached the ledger once",
        sorted(by_symbol) == sorted(run.decisions) and len(decisions) == len(run.decisions),
        f"{len(decisions)} rows for {len(run.decisions)} scripted decisions"))
    abstained = [s for s, d in run.decisions.items() if d.verdict == "no_trade"]
    bad = [s for s in abstained if s in by_symbol and (
        by_symbol[s].verdict != "no_trade" or Decimal(by_symbol[s].quantity) != 0)]
    checks.append(Check("a scripted abstention is recorded as an abstention", not bad,
                        f"{len(abstained)} abstentions; wrong: {bad or 'none'}"))
    widened = [e.symbol for e in decisions
               if Decimal(e.quantity) > run.decisions[e.symbol].quantity]
    flipped = [e.symbol for e in decisions if Decimal(e.quantity) > 0
               and e.side.upper() != run.decisions[e.symbol].side.upper()]
    risk_widened = [r["symbol"] for r in run.risk
                    if Decimal(r["quantity_after"]) > Decimal(r["quantity_before"])]
    checks.append(Check(
        "the risk layer and venue gate only reduced",
        not (widened or flipped or risk_widened),
        f"widened past the script: {widened or 'none'}; side flipped: {flipped or 'none'}; "
        f"risk records enlarging: {risk_widened or 'none'}"))
    stopped = gate_reasons(run)
    silent = sorted(s for s, why in stopped.items() if not why)
    checks.append(Check(
        "a scripted trade the ledger records as no trade names the gate that stopped it",
        not silent,
        "; ".join(f"{s}: {why[0]}" for s, why in sorted(stopped.items()) if why)
        + (f"; stopped with no gate note: {silent}" if silent else "")
        if stopped else "no scripted trade was stopped"))
    hollow = [e.symbol for e in decisions
              if (e.verdict == "trade") != (Decimal(e.quantity) > 0)]
    checks.append(Check(
        "no row records a trade without a size, or a size without a trade",
        not hollow, f"rows where verdict and size disagree: {hollow or 'none'} (the seq 264 "
                    f"defect recorded a trade the risk layer had refused)"))
    oversized = [s for s, d in run.decisions.items() if d.verdict == "trade"
                 and d.quantity * market.prices[s][0] >= OVERSIZED / 2]
    risk_by_symbol = {str(r["symbol"]): r for r in run.risk}
    unbound = [s for s in oversized if s not in by_symbol
               or not risk_by_symbol.get(s, {}).get("intervened")
               or Decimal(by_symbol[s].quantity) >= run.decisions[s].quantity
               or Decimal(by_symbol[s].quantity)
               > Decimal(str(risk_by_symbol[s]["quantity_after"]))]
    checks.append(Check(
        "an oversized order is reduced by the risk layer and recorded at the reduced size",
        not unbound,
        "; ".join(f"{s}: scripted {run.decisions[s].quantity}, risk layer "
                  f"{risk_by_symbol.get(s, {}).get('binding_constraint')} -> "
                  f"{risk_by_symbol.get(s, {}).get('quantity_after')}, ledger "
                  f"{by_symbol[s].quantity if s in by_symbol else 'absent'}"
                  for s in oversized) or "no oversized order in this script"))
    traded = [e for e in decisions if Decimal(e.quantity) > 0]
    cost_wrong = [e.symbol for e in traded if Decimal(e.entry_cost_bps)
                  != (TAKER_BPS + REPLAY_SPREAD_BPS).quantize(Decimal("0.001"))]
    checks.append(Check("each entry is charged the taker fee plus the quoted spread",
                        not cost_wrong,
                        f"{len(traded)} trades; wrong entry cost: {cost_wrong or 'none'}"))
    pnl_wrong: list[str] = []
    wins = 0
    net_total = Decimal("0")
    for e in traded:
        p0, _p1, p2 = market.prices[e.symbol]
        gross, net = _gold_net(Decimal(e.quantity), e.side, p0, p2)
        wins += net > 0
        net_total += net
        if (e.gross_pnl is None or e.net_pnl is None
                or abs(Decimal(e.gross_pnl) - gross) > TOLERANCE
                or abs(Decimal(e.net_pnl) - net) > TOLERANCE
                or bool(e.direction_correct) != (gross > 0)):
            pnl_wrong.append(f"{e.symbol}: ledger gross {e.gross_pnl} net {e.net_pnl}, "
                             f"gold gross {gross:.4f} net {net:.4f}")
    checks.append(Check("each trade settles at the gold gross and net P&L, 25 hours later",
                        not pnl_wrong and all(e.is_settled for e in traded),
                        "; ".join(pnl_wrong) or f"{len(traded)} trades match to 0.0001"))
    moves_wrong: list[str] = []
    for e in decisions:
        if Decimal(e.quantity) != 0:
            continue
        p0, _p1, p2 = market.prices[e.symbol]
        gold = (p2 - p0) / p0 * 10000
        got = e.counterfactual_move_bps
        if got is None or abs(Decimal(got) - gold) > TOLERANCE:
            moves_wrong.append(f"{e.symbol}: ledger {got}, gold {gold:.4f}")
    checks.append(Check("each abstention settles at the gold counterfactual move",
                        not moves_wrong,
                        "; ".join(moves_wrong) or "all abstentions match to 0.0001bps"))
    mark_wrong: list[str] = []
    abstention_rows = {e.seq: e for e in decisions if Decimal(e.quantity) == 0}
    for row in run.marks:
        entry = abstention_rows.get(int(row["seq"]))
        if entry is None:
            mark_wrong.append(f"seq {row['seq']} marked but is not an abstention")
            continue
        p0, p1, _p2 = market.prices[entry.symbol]
        if abs(Decimal(row["move_bps"]) - (p1 - p0) / p0 * 10000) > TOLERANCE:
            mark_wrong.append(f"{entry.symbol}: mark {row['move_bps']}")
    checks.append(Check(
        "each abstention is marked once at the three-hour price",
        not mark_wrong and len(run.marks) == len(abstention_rows),
        "; ".join(mark_wrong) or f"{len(run.marks)} marks for {len(abstention_rows)} "
                                 f"abstentions"))
    perf = run.performance
    gold_rate = (wins / len(traded)) if traded else None
    rate = perf.get("win_rate")
    checks.append(Check(
        "the scorer reports the gold trade count, win rate and net P&L",
        perf.get("trades") == len(traded)
        and (rate is None if gold_rate is None
             else rate is not None and abs(float(rate) - gold_rate) < 1e-9)
        and abs(Decimal(str(perf.get("net_pnl", "0"))) - net_total) <= TOLERANCE * len(traded),
        f"scorer: {perf.get('trades')} trades, win rate {rate}, net {perf.get('net_pnl')}; gold: "
        f"{len(traded)} trades, win rate {gold_rate}, net {net_total:.4f}"))
    checks.append(Check(
        "a two-day record reports no Sharpe rather than a number",
        perf.get("sharpe") is None and "sharpe" in (perf.get("undefined") or {}),
        f"sharpe {perf.get('sharpe')}; stated reason: "
        f"{(perf.get('undefined') or {}).get('sharpe', 'none')}"))
    checks.append(Check("the hash chain verifies, and a one-cent forgery does not",
                        run.chain.get("chain_intact") == "True" and run.tamper_detected is True,
                        f"chain intact: {run.chain.get('chain_intact')}; forgery caught: "
                        f"{run.tamper_detected}"))
    checks.append(Check(
        "the replay wrote nothing under data/", not run.data_writes,
        f"files this process opened for writing there: {run.data_writes or 'none'}"))
    checks.append(Check(
        "the seat answered every call; nothing reached a model or the network",
        run.seat_calls.get("unscripted", 0) == 0 and int(run.cycles[-1]["summary"].get(
            "tokens_spent", 0) or 0) == 0,
        f"seat calls {run.seat_calls}; outbound connections refused: {run.refused_connections}"))
    if oracle_run:
        wrong_way = [e.symbol for e in traded if e.direction_correct is not True]
        checks.append(Check("every oracle trade is scored the right way", not wrong_way,
                            f"scored wrong-way: {wrong_way or 'none'}"))
    return checks


def _summary(run: Replay, checks: Sequence[Check]) -> dict[str, Any]:
    return {
        "name": run.name,
        "passed": all(c.passed for c in checks),
        "checks": [c.as_dict() for c in checks],
        "decisions": {s: {"verdict": d.verdict, "side": d.side, "quantity": str(d.quantity)}
                      for s, d in sorted(run.decisions.items())},
        "recorded": [{"seq": e.seq, "symbol": e.symbol, "verdict": e.verdict, "side": e.side,
                      "quantity": e.quantity, "entry_price": e.entry_price,
                      "exit_price": e.exit_price, "net_pnl": e.net_pnl,
                      "counterfactual_move_bps": e.counterfactual_move_bps}
                     for e in run.entries],
        "risk_bindings": dict(Counter(str(r.get("binding_constraint")) for r in run.risk)),
        "stopped_by_a_gate": gate_reasons(run),
        "performance": {k: run.performance.get(k) for k in
                        ("trades", "win_rate_pct", "net_pnl", "sharpe", "max_drawdown_pct",
                         "abstentions")},
        "seat_calls": run.seat_calls,
        "validator_complaints": run.complaints,
        "outbound_connections_refused": run.refused_connections,
        "data_files_this_replay_wrote": run.data_writes,
        "seconds": run.seconds,
    }


def _gate_findings(label: str, run: Replay) -> list[str]:
    """What the replay shows about a gate that stopped a scripted trade, stated from the record:
    the ledger books such an order as ``no_trade``, and the harness then marks and grades it as an
    abstention, which the decision-maker did not choose."""
    stopped = gate_reasons(run)
    if not stopped:
        return []
    graded = [e.symbol for e in run.entries if e.symbol in stopped
              and e.counterfactual_move_bps is not None]
    marked = {int(m["seq"]) for m in run.marks}
    marked_names = [e.symbol for e in run.entries if e.symbol in stopped and e.seq in marked]
    reasons = "; ".join(f"{s}: {' / '.join(why) or 'no gate note'}"
                        for s, why in sorted(stopped.items()))
    return [f"{label}: {len(stopped)} scripted trade(s) were booked as no_trade by a gate "
            f"({reasons}). The ledger records them as abstentions: {len(marked_names)} were marked "
            f"at three hours and {len(graded)} graded with a counterfactual move, so they count "
            f"in abstention scoring although the decision-maker chose to trade."]


def run_all(human_path: Path | None = None) -> dict[str, Any]:
    """Teacher forcing, the sabotage control, and the human baselines, on one replayed market."""
    market = replay_market()
    instruments = _instruments()
    before = _snapshot()
    teacher = oracle(market, instruments, abstain=TEACHER_ABSTAINS, oversize=TEACHER_OVERSIZES)
    teacher_run = replay("teacher", teacher, market, instruments)
    control = replay("teacher-sabotaged", teacher, market, instruments,
                     sabotage=settle_on_decision_prices)
    control_checks = validate(control, market, oracle_run=True)
    baselines: list[dict[str, Any]] = []
    policies: list[tuple[str, Callable[[Market, Mapping[str, Any]], dict[str, Decision]]]] = [
        ("human-rule: always abstain", always_abstain),
        ("human-rule: buy every name", buy_everything)]
    if human_path is not None:
        policies.append((f"human: {human_path.name}", human(human_path)))
    findings: list[str] = []
    for label, policy in policies:
        run = replay(label.replace(":", "").replace(" ", "-"), policy(market, instruments),
                     market, instruments)
        baselines.append({"label": label,
                          **_summary(run, validate(run, market, oracle_run=False))})
        findings.extend(_gate_findings(label, run))
    written = sorted(k for k, v in _snapshot().items()
                     if before.get(k) != v and not k.startswith(REPORT_PATH.name))
    teacher_checks = validate(teacher_run, market, oracle_run=True)
    findings[:0] = _gate_findings("teacher", teacher_run)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "market": {"source": CANDLES_PATH.name, "decided_at": market.decided_at.isoformat(),
                   "marked_at": market.marked_at.isoformat(),
                   "settled_at": market.settled_at.isoformat(),
                   "closes": {s: [str(x) for x in p] for s, p in sorted(market.prices.items())},
                   "stated_constants": {"quoted_spread_bps": str(REPLAY_SPREAD_BPS),
                                        "base_volume": str(REPLAY_BASE_VOLUME),
                                        "funding_rate": "0", "taker_bps": str(TAKER_BPS)},
                   "instruments": INSTRUMENTS_PATH.name},
        "teacher_forcing": _summary(teacher_run, teacher_checks),
        "sabotage_control": {
            "defect": "settlement handed the decision cycle's prices",
            "caught": not all(c.passed for c in control_checks),
            "failed_checks": [c.name for c in control_checks if not c.passed],
            **_summary(control, control_checks)},
        "human_baselines": baselines,
        "harness_verdict": (
            "sound on this replay" if all(c.passed for c in teacher_checks)
            and not all(c.passed for c in control_checks)
            else "NOT sound on this replay: see the failed checks"),
        "findings": findings,
        "files_under_data_changed_by_any_process": written,
        "limitations": [
            "one replayed market instant; the selection rule is fixed but a second instant would "
            "test more of the Constitution's session-dependent gates",
            "every non-price feed is reported unavailable, so the analysts' evidence paths are "
            "not exercised; the replay tests the decision-to-score path, not evidence gathering",
            "the committed protocol is read as it is; it governs the live record from its own "
            "sequence number, so it does not govern a replay ledger that starts at 1",
        ],
    }


def main() -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--human", type=Path, default=None,
                        help="a JSONL file of a person's decisions, one row per contract")
    parser.add_argument("--freeze-instruments", action="store_true",
                        help="read the venue's instrument rules once and freeze them")
    args = parser.parse_args()
    if args.freeze_instruments:
        market = replay_market()
        print(f"froze {freeze_instruments(sorted(market.prices))} instrument rows to "
              f"{INSTRUMENTS_PATH}")
        return 0
    report = run_all(args.human)
    write(REPORT_PATH, report)
    teacher = report["teacher_forcing"]
    print(f"teacher forcing: {'PASS' if teacher['passed'] else 'FAIL'}")
    for check in teacher["checks"]:
        print(f"  {'ok ' if check['passed'] else 'BAD'} {check['check']}: {check['detail']}")
    control = report["sabotage_control"]
    print(f"sabotage control caught: {control['caught']} ({control['failed_checks']})")
    for base in report["human_baselines"]:
        print(f"{base['label']}: {base['performance']}")
    for finding in report["findings"]:
        print(f"finding: {finding}")
    print(report["harness_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
