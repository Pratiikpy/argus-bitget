"""Execution truth, measured: injected venue faults must be caught, recorded history must not be.

Four execution-safety mechanisms landed on 2026-09-25 (`execution/confirm.py`, `execution/batch.py`,
`execution/grounded_ids.py`, `execution/consent.py`, wired through `paper/venue.py`,
`paper/ledger.py` and `paper/runner.py`). A safety check is worth exactly what it catches and what
it does not falsely catch, so this module measures both, and writes ``data/execution_truth.json``.

**Part A — fault injection against a scripted venue.** A :class:`ScriptedVenue` answers the order
status and the position from time-scripted tables on a simulated clock, so a late settle, a status
that never agrees with the position, or a read that fails for a second and a half are reproducible
to the millisecond. Each scenario has a known right answer, and each mechanism is run beside the
baselines it replaces:

* **Fill confirmation** against (1) *one immediate status read* — what `BitgetTradingClient.
  reconcile` did on the line after placement in `execution/preflight.py` and `demo/flow.py:322-323`;
  (2) *a fixed sleep, then one status read* — OSWorld's ``step`` pattern
  (`osworld/desktop_env/desktop_env.py:416,453`, ``pause=2``); (3) *the position alone*. A method
  is **unsafe** on a scenario when it books a final outcome (filled / partial / unfilled) that is
  wrong; answering "not final yet" when the truth is final is wrong but safe.
* **Batch writes** against (1) *retry the whole batch once on an exception*; (2) *trust the batch
  response*, reporting everything written when the call returns and everything failed when it
  raises; (3) *mem0's literal fallback* — per-item insert of every record after a failed batch, no
  probe (`mem0/memory/main.py:1064-1070`). Scored on duplicate placements at the venue and on
  items whose reported outcome contradicts what the venue holds.
* **Shown ids** against (1) *real ids shown, any id in the book accepted*; (2) *WebArena's literal
  lookup* — short ids through ``table[token]`` item by item (`webarena/browser_env/
  processors.py:641-642`). Scored on actions that reach an order the model was not shown, act
  twice, or act partially.
* **The consent gate**, run against the real `BitgetTradingClient` with its transport replaced by
  a recorder, so the count of requests that would have left the machine is exact.

**Part B — replay of what was actually recorded.** Nothing synthetic:

* every decision in every parseable cycle log (``data/paper_runs``), its logged row checked as the
  order record against the ledger file's position as the book, and against the Constitution's
  ruling in ``data/risk_records.jsonl``;
* every settlement the cycle logs reported, against the ledger's settled fields;
* every recorded settlement, replayed through ``PaperLedger.settle_batch`` one cycle-batch at a
  time on a reconstruction of the ledger, compared field by field with what was recorded, and the
  same settlements replayed one at a time on a second reconstruction — the two files must be
  byte-identical;
* a positive control on real incident data: the ledger file preserved from the 2026-09-12
  concurrent-writer incident, where seven sequence numbers were written twice.

**What this does not measure.** Nothing here touches a real venue: the Bitget client's transport
is replaced, and no Bitget position read exists to be measured (see `execution/confirm.py`). The
paper desk has recorded two non-zero positions in its whole history (seq 264 and 265, both voided),
so the replay exercises the no-exposure path far more than the fill path; the fill path's evidence
is Part A. No model is called anywhere in this module.

    python -m argus.eval.execution_truth
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.decision.verdicts import ConstitutionVerdict, Intent, Side, Verdict, apply_constraint
from argus.eval.artefact import write
from argus.execution.batch import BatchResponse, ItemStatus, write_batch
from argus.execution.bitget_client import (
    LIVE_PRODUCT_TYPE,
    BitgetOrderError,
    BitgetTradingClient,
    LiveOrderRefused,
)
from argus.execution.confirm import FillVerdict, StatusReading, VenueReadError, confirm_fill
from argus.execution.consent import (
    ConsentRefused,
    LiveOrderConsent,
    consent_phrase,
    grant_live_consent,
)
from argus.execution.grounded_ids import ShownList, UnshownId, show_orders
from argus.execution.orders import Order, OrderBook, OrderState
from argus.paper.ledger import GENESIS, Entry, PaperLedger, SettleRequest
from argus.paper.venue import LedgerVenue, confirm_recorded, paper_order_id, read_rows

ARGUS = Path(__file__).resolve().parents[3]
DATA = ARGUS / "data"
REPORT_PATH = DATA / "execution_truth.json"
LEDGER_PATH = DATA / "paper_ledger.jsonl"
RISK_PATH = DATA / "risk_records.jsonl"
RUNS_DIR = DATA / "paper_runs"
INCIDENT_PATH = DATA / "paper_ledger.corrupted-20260912T180308Z.jsonl"

Q = Decimal("2")
"""The order quantity every fill scenario uses."""

FINAL = ("filled", "partial", "unfilled")


# --- the scripted venue --------------------------------------------------------------------------


class SimClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


Scripted = StatusReading | Decimal | str
"""A script entry: a reading, or a string naming a read failure."""


def _at(script: Sequence[tuple[float, Scripted]], t: float) -> Scripted:
    """The entry in force at ``t``: the last whose start time is at or before it."""
    current = script[0][1]
    for start, value in script:
        if start <= t:
            current = value
    return current


@dataclass
class ScriptedVenue:
    """Order status and position, each a function of simulated time. Counts every read."""

    clock: SimClock
    status_script: tuple[tuple[float, Scripted], ...]
    position_script: tuple[tuple[float, Scripted], ...]
    status_reads: int = 0
    position_reads: int = 0

    def order_status(self, client_order_id: str, *, symbol: str) -> StatusReading:
        self.status_reads += 1
        value = _at(self.status_script, self.clock.now())
        if isinstance(value, str):
            raise VenueReadError(value)
        if not isinstance(value, StatusReading):  # pragma: no cover - script shape
            raise TypeError("a status script holds readings or failures")
        return value

    def position(self, symbol: str) -> Decimal:
        self.position_reads += 1
        value = _at(self.position_script, self.clock.now())
        if isinstance(value, str):
            raise VenueReadError(value)
        if not isinstance(value, Decimal):  # pragma: no cover - script shape
            raise TypeError("a position script holds quantities or failures")
        return value


def _st(state: OrderState | None, filled: str = "0") -> StatusReading:
    return StatusReading(
        state=state, filled=Decimal(filled), raw="" if state is None else state.value,
    )


# --- A1. fill confirmation -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FillScenario:
    key: str
    story: str
    status: tuple[tuple[float, Scripted], ...]
    position: tuple[tuple[float, Scripted], ...]
    truth: str
    """The right class: filled, partial, unfilled, disputed, or not_final."""


FILL_SCENARIOS: tuple[FillScenario, ...] = (
    FillScenario(
        "instant_fill", "status and position both show the fill on the first read",
        ((0.0, _st(OrderState.FILLED, "2")),), ((0.0, Decimal("2")),), "filled",
    ),
    FillScenario(
        "late_settle", "accepted, filled at 1.5s; the position books it only at 3.0s",
        ((0.0, _st(OrderState.ACCEPTED)), (1.5, _st(OrderState.FILLED, "2"))),
        ((0.0, Decimal("0")), (3.0, Decimal("2"))), "filled",
    ),
    FillScenario(
        "late_index", "no order record for 6s (still being indexed), then filled",
        ((0.0, _st(None)), (6.0, _st(OrderState.FILLED, "2"))),
        ((0.0, Decimal("0")), (7.0, Decimal("2"))), "filled",
    ),
    FillScenario(
        "still_working", "a resting order that never fills inside the deadline",
        ((0.0, _st(OrderState.ACCEPTED)),), ((0.0, Decimal("0")),), "not_final",
    ),
    FillScenario(
        "phantom_fill", "the status says filled at 0.3s; the position never moves",
        ((0.0, _st(OrderState.ACCEPTED)), (0.3, _st(OrderState.FILLED, "2"))),
        ((0.0, Decimal("0")),), "disputed",
    ),
    FillScenario(
        "rejected_but_exposed", "the status says rejected; the position moved by the full size",
        ((0.0, _st(None)), (0.2, _st(OrderState.REJECTED))),
        ((0.0, Decimal("0")), (0.5, Decimal("2"))), "disputed",
    ),
    FillScenario(
        "status_position_mismatch", "the status says 2 filled; the position shows 1",
        ((0.0, _st(OrderState.ACCEPTED)), (0.5, _st(OrderState.FILLED, "2"))),
        ((0.0, Decimal("0")), (0.5, Decimal("1"))), "disputed",
    ),
    FillScenario(
        "empty_record", "the venue never returns a record and the position never moves",
        ((0.0, _st(None)),), ((0.0, Decimal("0")),), "not_final",
    ),
    FillScenario(
        "genuine_rejection", "rejected, and the position agrees nothing happened",
        ((0.0, _st(OrderState.REJECTED)),), ((0.0, Decimal("0")),), "unfilled",
    ),
    FillScenario(
        "partial_then_cancel", "half fills, the rest is cancelled; the position shows the half",
        ((0.0, _st(OrderState.PARTIALLY_FILLED, "1")), (1.0, _st(OrderState.CANCELLED, "1"))),
        ((0.0, Decimal("0")), (0.5, Decimal("1"))), "partial",
    ),
    FillScenario(
        "flaky_status_reads", "status reads fail for 1.2s, then show the fill",
        ((0.0, "HTTP 503 from the order endpoint"), (1.2, _st(OrderState.FILLED, "2"))),
        ((0.0, Decimal("2")),), "filled",
    ),
    FillScenario(
        "overfill", "both records agree on 4 filled against an order of 2",
        ((0.0, _st(OrderState.FILLED, "4")),), ((0.0, Decimal("4")),), "disputed",
    ),
)

_OURS = {
    FillVerdict.CONFIRMED: "filled",
    FillVerdict.CONFIRMED_PARTIAL: "partial",
    FillVerdict.CONFIRMED_UNFILLED: "unfilled",
    FillVerdict.DISAGREED: "disputed",
    FillVerdict.OVERFILLED: "disputed",
    FillVerdict.UNSETTLED: "not_final",
    FillVerdict.UNCORROBORATED: "not_final",
    FillVerdict.NO_EVIDENCE: "not_final",
}


def _status_class(reading: Scripted, quantity: Decimal) -> str:
    """How a caller that trusts one status read books it — `reconcile`'s mapping, applied."""
    if not isinstance(reading, StatusReading) or reading.state is None:
        return "not_final"
    if reading.state is OrderState.FILLED:
        return "filled"
    if reading.state in (OrderState.CANCELLED, OrderState.EXPIRED, OrderState.REJECTED):
        if reading.filled == 0:
            return "unfilled"
        return "partial" if reading.filled < quantity else "filled"
    return "not_final"


def _position_class(delta: Scripted, quantity: Decimal) -> str:
    if not isinstance(delta, Decimal):
        return "not_final"
    if delta == quantity:
        return "filled"
    if delta == 0:
        return "unfilled"
    if 0 < delta < quantity:
        return "partial"
    return "disputed"


def _score(answer: str, truth: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "correct": answer == truth,
        "unsafe": answer in FINAL and answer != truth,
    }


def run_fill_scenarios(*, deadline_s: float = 30.0) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    for scenario in FILL_SCENARIOS:
        clock = SimClock()
        venue = ScriptedVenue(clock, scenario.status, scenario.position)
        got = confirm_fill(
            status=venue, position=venue, client_order_id=f"sim-{scenario.key}", symbol="SIMUSDT",
            side="BUY", quantity=Q, position_before=Decimal("0"), deadline_s=deadline_s,
            clock=clock.now, sleep=clock.sleep,
        )
        answers = {
            "argus_confirm_fill": _score(_OURS[got.verdict], scenario.truth),
            "single_immediate_status_read": _score(
                _status_class(_at(scenario.status, 0.0), Q), scenario.truth,
            ),
            "fixed_2s_sleep_then_status_read": _score(
                _status_class(_at(scenario.status, 2.0), Q), scenario.truth,
            ),
            "position_only_at_deadline": _score(
                _position_class(_at(scenario.position, deadline_s), Q), scenario.truth,
            ),
        }
        for method, score in answers.items():
            bucket = totals.setdefault(method, {"correct": 0, "unsafe": 0})
            bucket["correct"] += int(score["correct"])
            bucket["unsafe"] += int(score["unsafe"])
        rows.append({
            "scenario": scenario.key,
            "story": scenario.story,
            "truth": scenario.truth,
            "argus_verdict": got.verdict.value,
            "argus_reads": len(got.observations),
            "argus_simulated_seconds": round(clock.now(), 3),
            "answers": answers,
        })
    return {"scenarios": len(FILL_SCENARIOS), "deadline_s": deadline_s, "by_method": totals,
            "rows": rows}


# --- A2. batch writes ----------------------------------------------------------------------------


@dataclass
class BatchVenue:
    """A venue with a batch endpoint whose failure modes are scripted per scenario."""

    fail_in_response: Mapping[str, str] = field(default_factory=dict)
    silent_placed: frozenset[str] = frozenset()
    silent_dropped: frozenset[str] = frozenset()
    raise_after: int | None = None
    single_refuses: Mapping[str, str] = field(default_factory=dict)
    probe_answers: bool = True
    placements: dict[str, int] = field(default_factory=dict)

    def _place(self, oid: str) -> None:
        self.placements[oid] = self.placements.get(oid, 0) + 1

    def place_many(self, ids: Sequence[str]) -> BatchResponse:
        succeeded: set[str] = set()
        for index, oid in enumerate(ids):
            if self.raise_after is not None and index >= self.raise_after:
                raise TimeoutError(f"batch call timed out after landing {self.raise_after}")
            if oid in self.fail_in_response or oid in self.silent_dropped:
                continue
            self._place(oid)
            if oid not in self.silent_placed:
                succeeded.add(oid)
        return BatchResponse(
            succeeded=frozenset(succeeded),
            failed={k: v for k, v in self.fail_in_response.items() if k in ids},
        )

    def place_one(self, oid: str) -> None:
        if oid in self.single_refuses:
            raise BitgetOrderError(self.single_refuses[oid])
        self._place(oid)

    def holds(self, oid: str) -> bool | None:
        if not self.probe_answers:
            return None
        return self.placements.get(oid, 0) > 0


@dataclass(frozen=True, slots=True)
class BatchScenario:
    key: str
    story: str
    ids: tuple[str, ...]
    venue: Callable[[], BatchVenue]


BATCH_SCENARIOS: tuple[BatchScenario, ...] = (
    BatchScenario("clean", "every order lands and the response names each",
                  ("o1", "o2", "o3", "o4", "o5"), BatchVenue),
    BatchScenario(
        "failure_list", "the call succeeds but lists o2 and o4 as refused for margin",
        ("o1", "o2", "o3", "o4", "o5"),
        lambda: BatchVenue(fail_in_response={"o2": "40762 insufficient margin",
                                             "o4": "40762 insufficient margin"}),
    ),
    BatchScenario(
        "silent_items", "o4 lands but is not named; o5 is dropped and not named",
        ("o1", "o2", "o3", "o4", "o5"),
        lambda: BatchVenue(silent_placed=frozenset({"o4"}), silent_dropped=frozenset({"o5"})),
    ),
    BatchScenario(
        "raises_after_partial", "the call lands three orders then times out; o5 is refused singly",
        ("o1", "o2", "o3", "o4", "o5", "o6"),
        lambda: BatchVenue(raise_after=3, single_refuses={"o5": "40762 insufficient margin"}),
    ),
    BatchScenario(
        "raises_no_probe", "the call lands two orders then times out; the venue cannot be asked",
        ("o1", "o2", "o3", "o4"), lambda: BatchVenue(raise_after=2, probe_answers=False),
    ),
    BatchScenario(
        "duplicate_input", "the same order is named twice in one batch",
        ("o1", "o2", "o1"), BatchVenue,
    ),
)


def _batch_outcome_errors(reported: Mapping[str, str], venue: BatchVenue) -> int:
    """Items whose report contradicts the venue. ``unknown`` is honest uncertainty, not an error."""
    errors = 0
    for oid, status in reported.items():
        held = venue.placements.get(oid, 0) > 0
        if status == "written" and not held:
            errors += 1
        if status == "failed" and held:
            errors += 1
    return errors


def _ours_batch(ids: Sequence[str], venue: BatchVenue) -> dict[str, str]:
    report = write_batch(
        ids, key=lambda oid: oid, batch=venue.place_many, single=venue.place_one,
        applied=venue.holds,
    )
    out: dict[str, str] = {}
    for outcome in report.outcomes:
        if outcome.key in out:  # the refused duplicate; the first occurrence speaks for the key
            continue
        out[outcome.key] = (
            "written" if outcome.written
            else "unknown" if outcome.status is ItemStatus.UNKNOWN
            else "failed"
        )
    return out


def _retry_whole_batch(ids: Sequence[str], venue: BatchVenue) -> dict[str, str]:
    try:
        venue.place_many(ids)
    except Exception:
        try:
            venue.raise_after = None  # the retry goes through
            venue.place_many(ids)
        except Exception:
            return dict.fromkeys(ids, "failed")
    return dict.fromkeys(ids, "written")


def _trust_response(ids: Sequence[str], venue: BatchVenue) -> dict[str, str]:
    try:
        venue.place_many(ids)
    except Exception:
        return dict.fromkeys(ids, "failed")
    return dict.fromkeys(ids, "written")


def _mem0_literal(ids: Sequence[str], venue: BatchVenue) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        venue.place_many(ids)
    except Exception:
        for oid in ids:
            try:
                venue.place_one(oid)
                out[oid] = "written"
            except Exception:
                out[oid] = "failed"
        return out
    return dict.fromkeys(ids, "written")


def run_batch_scenarios() -> dict[str, Any]:
    methods: dict[str, Callable[[Sequence[str], BatchVenue], dict[str, str]]] = {
        "argus_write_batch": _ours_batch,
        "retry_whole_batch": _retry_whole_batch,
        "trust_the_response": _trust_response,
        "mem0_literal_fallback": _mem0_literal,
    }
    rows: list[dict[str, Any]] = []
    totals = {m: {"duplicates": 0, "misreported": 0, "unknown": 0} for m in methods}
    for scenario in BATCH_SCENARIOS:
        row: dict[str, Any] = {"scenario": scenario.key, "story": scenario.story, "methods": {}}
        for name, method in methods.items():
            venue = scenario.venue()
            reported = method(scenario.ids, venue)
            duplicates = sum(n - 1 for n in venue.placements.values() if n > 1)
            misreported = _batch_outcome_errors(reported, venue)
            unknown = sum(1 for v in reported.values() if v == "unknown")
            totals[name]["duplicates"] += duplicates
            totals[name]["misreported"] += misreported
            totals[name]["unknown"] += unknown
            row["methods"][name] = {
                "reported": reported, "venue_holds": dict(sorted(venue.placements.items())),
                "duplicates": duplicates, "misreported": misreported, "unknown": unknown,
            }
        rows.append(row)
    return {"scenarios": len(BATCH_SCENARIOS), "by_method": totals, "rows": rows}


# --- A3. shown ids -------------------------------------------------------------------------------


def _authorised(order: Order) -> Any:
    """A genuine Constitution authorisation for a fixture order, through the real path."""
    return apply_constraint(
        Intent(
            symbol=order.symbol, side=Side(order.side.lower()), quantity=order.quantity,
            verdict=Verdict.TRADE, stated_confidence=0.7, thesis="execution-truth fixture",
            invalidation=("the measurement ends",),
        ),
        verdict=ConstitutionVerdict.ALLOW, binding_constraint="none",
        reason="fixture order for the execution-truth measurement; no economic claim",
    ).authorise(order)


def fixture_book(at: datetime) -> tuple[OrderBook, ShownList]:
    """Five live orders on three symbols; the model is shown the three on NVDA and TSLA."""
    book = OrderBook()
    specs = (
        ("argus-7f3a01d2c4e5f6a7b8c9d0e1", "NVDAUSDT", "BUY", "2"),
        ("argus-1b2c3d4e5f60718293a4b5c6", "TSLAUSDT", "SELL", "1"),
        ("argus-9e8d7c6b5a4f3e2d1c0b9a88", "NVDAUSDT", "SELL", "3"),
        ("argus-0a1b2c3d4e5f6a7b8c9d0e1f", "COINUSDT", "BUY", "5"),
        ("argus-5f4e3d2c1b0a9f8e7d6c5b4a", "MSTRUSDT", "SELL", "4"),
    )
    for oid, symbol, side, qty in specs:
        book.submit(_authorised(Order(oid, symbol, side, Decimal(qty), "fixture-hash")), at=at)
    shown = show_orders(o for o in book.live_orders() if o.symbol in ("NVDAUSDT", "TSLAUSDT"))
    return book, shown


@dataclass(frozen=True, slots=True)
class IdScenario:
    """What the model does, written in each display's own terms.

    ``short`` is the answer under a short-id display (ours, and WebArena's); ``real`` is the same
    intent under a display that shows real ids, so each method is scored on the answer a model
    looking at *that* method's prompt would give.
    """

    key: str
    truth: str
    short: tuple[object, ...]
    real: tuple[object, ...]


def _id_scenarios(shown: ShownList, book: OrderBook) -> tuple[IdScenario, ...]:
    hidden = next(o.client_order_id for o in book.live_orders() if o.symbol == "COINUSDT")
    real = [r.target for r in shown.rows]
    invented = "argus-000000000000000000000000"
    return (
        IdScenario("picks_two_shown_orders", "act", ("1", "[3]"), (real[0], real[2])),
        IdScenario("invents_an_id", "refuse", ("7",), (invented,)),
        IdScenario("names_an_existing_order_it_was_not_shown", "refuse", (hidden,), (hidden,)),
        IdScenario("one_real_one_invented", "refuse", ("1", "9"), (real[0], invented)),
        IdScenario("same_order_twice", "refuse", ("2", "[2]"), (real[1], real[1])),
        IdScenario("boolean_token", "refuse", (True,), (True,)),
        IdScenario("malformed_token", "refuse", ("1a",), ("1a",)),
        IdScenario("empty_token", "refuse", ("",), ("",)),
        IdScenario("float_token", "refuse", (1.5,), (1.5,)),
        IdScenario("hash_prefixed_token", "refuse", ("#2",), ("#2",)),
    )


def _baseline_real_ids(tokens: Sequence[object], book: OrderBook) -> tuple[list[str], bool]:
    """Real ids shown; each token looked up in the whole book, acting as it goes."""
    acted: list[str] = []
    for token in tokens:
        try:
            acted.append(book.get(str(token)).client_order_id)
        except KeyError:
            return acted, True
    return acted, False


def _baseline_webarena(tokens: Sequence[object], shown: ShownList) -> tuple[list[str], bool]:
    """Short ids through a plain dict lookup, item by item, acting as it goes."""
    table = {r.short_id: r.target for r in shown.rows}
    acted: list[str] = []
    for token in tokens:
        try:
            acted.append(table[token])  # type: ignore[index]
        except (KeyError, TypeError):
            return acted, True
    return acted, False


def run_id_scenarios() -> dict[str, Any]:
    at = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
    book, shown = fixture_book(at)
    shown_targets = {r.target for r in shown.rows}
    rows: list[dict[str, Any]] = []
    totals = {m: {"correct": 0, "unsafe": 0} for m in (
        "argus_resolve_all", "real_ids_any_in_book", "webarena_literal_lookup",
    )}
    for scenario in _id_scenarios(shown, book):
        truth = scenario.truth
        try:
            ours: list[str] = list(shown.resolve_all(scenario.short))
            ours_refused = False
        except UnshownId:
            ours, ours_refused = [], True
        results = {
            "argus_resolve_all": (ours, ours_refused),
            "real_ids_any_in_book": _baseline_real_ids(scenario.real, book),
            "webarena_literal_lookup": _baseline_webarena(scenario.short, shown),
        }
        row: dict[str, Any] = {
            "scenario": scenario.key, "truth": truth,
            "short_tokens": [repr(t) for t in scenario.short],
            "real_tokens": [repr(t) for t in scenario.real], "methods": {},
        }
        for method, (acted, raised) in results.items():
            refused_cleanly = raised and not acted
            # "act" scenarios name exactly two shown orders; acting on both is the right answer.
            correct = (not raised and len(acted) == 2) if truth == "act" else refused_cleanly
            unsafe = bool(
                any(a not in shown_targets for a in acted)
                or len(acted) != len(set(acted))
                or (truth == "refuse" and acted)
            )
            totals[method]["correct"] += int(correct)
            totals[method]["unsafe"] += int(unsafe)
            row["methods"][method] = {
                "acted_on": len(acted), "refused": raised, "correct": correct, "unsafe": unsafe,
            }
        rows.append(row)
    return {
        "scenarios": len(rows),
        "shown": shown.as_dict(),
        "by_method": totals,
        "rows": rows,
    }


# --- A4. the consent gate ------------------------------------------------------------------------

_CREDENTIALS = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")


@contextmanager
def _fixture_credentials() -> Iterator[None]:
    """Placeholder credentials for a client whose transport never leaves the process."""
    saved = {k: os.environ.get(k) for k in _CREDENTIALS}
    for k in _CREDENTIALS:
        os.environ[k] = f"execution-truth-{k.lower()}"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RecordingClient(BitgetTradingClient):
    """The real client with its transport replaced: every request is counted, none is sent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sent: list[str] = []

    def _request(
        self, method: str, path: str, *, params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None, private: bool = True,
    ) -> Any:
        self.sent.append(f"{method} {path}")
        if path.endswith("/account/account"):
            return {"posMode": "one_way_mode"}
        return {"orderId": f"recorded-{len(self.sent)}"}


def run_consent_scenarios(spent_path: Path) -> dict[str, Any]:
    at = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
    order_counter = iter(range(1, 100))

    def attempt(client: RecordingClient) -> tuple[bool, str]:
        order = Order(f"argus-consent-{next(order_counter)}", "SBTCSUSDT", "BUY", Decimal("0.01"),
                      "fixture-hash")
        try:
            client.place_order(_authorised(order))
        except LiveOrderRefused as exc:
            return False, str(exc)[:160]
        return True, "sent"

    rows: list[dict[str, Any]] = []

    def record(key: str, expected_sent: bool, sent: bool, detail: str, requests: int) -> None:
        rows.append({"scenario": key, "expected_sent": expected_sent, "sent": sent,
                     "correct": sent == expected_sent, "venue_requests": requests,
                     "detail": detail})

    with _fixture_credentials():
        paper = RecordingClient(paper_trading=True)
        sent, detail = attempt(paper)
        record("paper_needs_no_consent", True, sent, detail, len(paper.sent))

        live = RecordingClient(paper_trading=False)
        sent, detail = attempt(live)
        record("live_without_consent", False, sent, detail, len(live.sent))

        try:
            grant_live_consent("run-B", token=consent_phrase("run-A"), now=at,
                               spent_path=spent_path)
            record("token_for_another_run", False, True, "granted", 0)
        except ConsentRefused as exc:
            record("token_for_another_run", False, False, str(exc)[:160], 0)

        stale = grant_live_consent("run-stale", token=consent_phrase("run-stale"),
                                   now=datetime.now(UTC) - timedelta(hours=1),
                                   spent_path=spent_path)
        expired = RecordingClient(paper_trading=False, live_consent=stale)
        sent, detail = attempt(expired)
        record("expired_consent", False, sent, detail, len(expired.sent))

        grant_live_consent("run-once", token=consent_phrase("run-once"), spent_path=spent_path)
        try:
            grant_live_consent("run-once", token=consent_phrase("run-once"),
                               spent_path=spent_path)
            record("token_reused", False, True, "granted twice", 0)
        except ConsentRefused as exc:
            record("token_reused", False, False, str(exc)[:160], 0)

        try:
            LiveOrderConsent("run-forged", at, at + timedelta(hours=1), _token=object())
            record("forged_consent_object", False, True, "constructed", 0)
        except ConsentRefused as exc:
            record("forged_consent_object", False, False, str(exc)[:160], 0)

        valid = grant_live_consent("run-valid", token=consent_phrase("run-valid"),
                                   spent_path=spent_path)
        consented = RecordingClient(paper_trading=False, live_consent=valid)
        sent, detail = attempt(consented)
        record("valid_consent", True, sent, detail, len(consented.sent))

        try:
            RecordingClient(paper_trading=True, product_type=LIVE_PRODUCT_TYPE)
            record("paper_flag_on_live_product", False, True, "constructed", 0)
        except ValueError as exc:
            record("paper_flag_on_live_product", False, False, str(exc)[:160], 0)

    return {
        "scenarios": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "real_money_orders_sent_without_valid_consent": sum(
            1 for r in rows if r["sent"] and not r["expected_sent"]
        ),
        "rows": rows,
        "baseline": (
            "not re-executed: before 2026-09-25 `BitgetTradingClient.place_order` had no consent "
            "check (read in source before the change), so every live-client scenario above would "
            "have been sent, and paper_trading=True with a live productType was constructible and "
            "reported trades_real_money=False"
        ),
    }


# --- B. replay of the recorded paper runs --------------------------------------------------------

_DECISIONS = re.compile(
    r'"decisions_written":\s*(\[.*?\])\s*,\s*"settled_this_cycle":\s*(\[.*?\])\s*,\s*"',
    re.DOTALL,
)
"""The cycle summary's two lists. The key after ``settled_this_cycle`` changed over the log's life
(``marked_this_cycle`` arrived later), so only the fact that another key follows is matched."""
_RAN_AT = re.compile(r'"ran_at":\s*"([^"]+)"')


def _read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


@dataclass(frozen=True, slots=True)
class LoggedCycle:
    log: str
    ran_at: datetime
    decisions: tuple[dict[str, Any], ...]
    settlements: tuple[dict[str, Any], ...]


def parse_cycle_logs(directory: Path = RUNS_DIR) -> tuple[list[LoggedCycle], list[str]]:
    """Every cycle log whose summary parses, and the names of those that do not."""
    cycles: list[LoggedCycle] = []
    unparsed: list[str] = []
    for path in sorted(directory.glob("cycle_*.log")):
        try:
            text = _read_log(path)
        except OSError:
            # The cycle writing this log right now holds it open; it is not history yet.
            unparsed.append(path.name)
            continue
        match = _DECISIONS.search(text)
        ran = _RAN_AT.search(text)
        if match is None or ran is None:
            unparsed.append(path.name)
            continue
        try:
            decisions = json.loads(match.group(1))
            settlements = json.loads(match.group(2))
        except json.JSONDecodeError:
            unparsed.append(path.name)
            continue
        cycles.append(LoggedCycle(
            log=path.name, ran_at=datetime.fromisoformat(ran.group(1)),
            decisions=tuple(decisions), settlements=tuple(settlements),
        ))
    return cycles, unparsed


class LoggedStatus:
    """The order record as the cycle itself reported it: its own log line, not the file."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row

    def order_status(self, client_order_id: str, *, symbol: str) -> StatusReading:
        if client_order_id != paper_order_id(int(self._row["seq"])):
            return StatusReading(state=None, filled=Decimal("0"), raw="another seq")
        if str(self._row["symbol"]) != symbol:
            return StatusReading(state=None, filled=Decimal("0"), raw="another symbol")
        quantity = Decimal(str(self._row["quantity"]))
        state = OrderState.FILLED if quantity > 0 else OrderState.DENIED
        return StatusReading(state=state, filled=quantity, raw=str(self._row.get("verdict", "")))


def _risk_ceilings(path: Path = RISK_PATH) -> dict[int, Decimal]:
    out: dict[int, Decimal] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            out[int(record["seq"])] = Decimal(str(record["quantity_after"]))
        except (json.JSONDecodeError, KeyError, ValueError, ArithmeticError):
            continue
    return out


def replay_logged_decisions(
    cycles: Sequence[LoggedCycle], rows: Sequence[Mapping[str, Any]],
    ceilings: Mapping[int, Decimal],
) -> dict[str, Any]:
    """Every logged decision: the log as the order record, the file as the book, the risk record
    as the ceiling. Single read each — recorded state does not move."""
    by_seq: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_seq.setdefault(int(row["seq"]), []).append(row)
    alarms: list[dict[str, Any]] = []
    verdicts: dict[str, int] = {}
    unevaluable_ceiling = 0
    checked = 0
    for cycle in cycles:
        for logged in cycle.decisions:
            seq = int(logged["seq"])
            symbol = str(logged["symbol"])
            on_disk = by_seq.get(seq, [])
            # The side is read from the file because the cycle log does not record it; for every
            # zero-quantity row it cannot affect the result, and the log-versus-file check below
            # is on the quantity, which the log does record.
            side = str(on_disk[0]["side"]) if on_disk else "BUY"
            before = LedgerVenue.frozen(rows, as_of=cycle.ran_at, inclusive=False).position(symbol)
            after_venue = LedgerVenue.frozen(rows, as_of=cycle.ran_at, inclusive=True)
            check = confirm_recorded(
                seq=seq, symbol=symbol, side=side,
                quantity=Decimal(str(logged["quantity"])),
                status=LoggedStatus(logged), position=after_venue, position_before=before,
                authorised_quantity=ceilings.get(seq), authorised_side=side,
                deadline_s=0.0,
            )
            checked += 1
            verdicts[check.confirmation.verdict.value] = (
                verdicts.get(check.confirmation.verdict.value, 0) + 1
            )
            if check.authorised_quantity is None:
                unevaluable_ceiling += 1
            if check.alarm:
                alarms.append({"log": cycle.log, **check.as_dict(), "note": check.render()})
    return {
        "decisions_checked": checked,
        "verdicts": verdicts,
        "no_risk_record_to_check_against": unevaluable_ceiling,
        "alarms": alarms,
        "alarm_seqs": sorted({int(a["seq"]) for a in alarms}),
    }


def replay_logged_settlements(
    cycles: Sequence[LoggedCycle], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Every settlement a cycle reported, against the ledger's settled fields."""
    by_seq = {int(r["seq"]): r for r in rows}
    compared = 0
    mismatches: list[dict[str, Any]] = []
    for cycle in cycles:
        for reported in cycle.settlements:
            if "seq" not in reported:
                continue
            row = by_seq.get(int(reported["seq"]))
            compared += 1
            if row is None:
                mismatches.append({"log": cycle.log, "seq": reported["seq"], "why": "no row"})
                continue
            for key in ("counterfactual_move_bps", "gross_pnl", "net_pnl", "direction_correct"):
                if key in reported and reported[key] != row.get(key):
                    mismatches.append({"log": cycle.log, "seq": reported["seq"], "field": key,
                                       "logged": reported[key], "ledger": row.get(key)})
    return {"settlements_compared": compared, "mismatches": mismatches}


_SETTLEMENT_FIELDS = (
    "settled_at", "exit_price", "gross_pnl", "net_pnl", "direction_correct",
    "counterfactual_move_bps",
)


def _pre_settlement(raw: Sequence[Mapping[str, Any]]) -> list[Entry]:
    """Every decision, outcome fields cleared, re-chained with no seals: the ledger before any
    settlement happened, as far as the recorded rows allow it to be rebuilt."""
    out: list[Entry] = []
    head = GENESIS
    for row in raw:
        if row.get("kind", "decision") != "decision":
            continue
        fields = {**row, "invalidation": tuple(row.get("invalidation", ())), "prev_hash": head}
        for key in _SETTLEMENT_FIELDS:
            fields[key] = None
        entry = Entry(**fields)
        out.append(entry)
        head = entry.content_hash
    return out


def _write_rows(path: Path, entries: Sequence[Entry]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(asdict(e), default=str) + "\n")


def replay_settlement_batches(raw: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay every recorded settlement through ``settle_batch``, and again one at a time."""
    recorded = [
        r for r in raw if r.get("kind", "decision") == "decision" and r.get("settled_at")
    ]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in recorded:
        groups.setdefault(str(row["settled_at"]), []).append(row)
    ordered = sorted(groups.items(), key=lambda kv: datetime.fromisoformat(kv[0]))
    pre = _pre_settlement(raw)

    with tempfile.TemporaryDirectory(prefix="execution-truth-") as tmp:
        batch_path = Path(tmp) / "batch.jsonl"
        single_path = Path(tmp) / "single.jsonl"
        _write_rows(batch_path, pre)
        _write_rows(single_path, pre)
        batched = PaperLedger(path=batch_path)
        singly = PaperLedger(path=single_path)

        failed: list[dict[str, Any]] = []
        started = time.perf_counter()
        for settled_at, members in ordered:
            requests = [
                SettleRequest(
                    seq=int(r["seq"]), price=Decimal(str(r["exit_price"])),
                    settled_at=datetime.fromisoformat(settled_at),
                )
                for r in members
            ]
            result = batched.settle_batch(requests)
            failed.extend({"settled_at": settled_at, **o.as_dict()} for o in result.report.failed)
        batch_seconds = time.perf_counter() - started

        started = time.perf_counter()
        for settled_at, members in ordered:
            at = datetime.fromisoformat(settled_at)
            for r in members:
                seq = int(r["seq"])
                entry = singly.entries[[e.seq for e in singly.entries].index(seq)]
                if entry.is_abstention:
                    singly.settle_abstention(seq, price_now=Decimal(str(r["exit_price"])),
                                             settled_at=at)
                else:
                    singly.settle(seq, exit_price=Decimal(str(r["exit_price"])), settled_at=at)
        single_seconds = time.perf_counter() - started

        identical = batch_path.read_bytes() == single_path.read_bytes()
        replayed = {e.seq: e for e in PaperLedger(path=batch_path).entries}
        field_mismatch: list[dict[str, Any]] = []
        not_reproducible: list[dict[str, Any]] = []
        for row in recorded:
            got = replayed[int(row["seq"])]
            for key in _SETTLEMENT_FIELDS:
                if getattr(got, key) == row.get(key):
                    continue
                if key == "net_pnl" and not got.is_abstention:
                    # The exit spread and the cycle's funding rate priced this net figure and
                    # neither was recorded, so it cannot be recomputed from the row. Said, not
                    # smoothed over; every other field of the same row is compared exactly.
                    not_reproducible.append({"seq": got.seq, "field": key,
                                             "recorded": row.get(key), "replayed": got.net_pnl})
                    continue
                field_mismatch.append({"seq": got.seq, "field": key,
                                       "recorded": row.get(key), "replayed": getattr(got, key)})
        verify = PaperLedger(path=batch_path).verify()

    return {
        "settlements_replayed": len(recorded),
        "batches": len(ordered),
        "failed_items": failed,
        "field_mismatches": field_mismatch,
        "not_reproducible_from_the_record": not_reproducible,
        "batch_file_identical_to_one_at_a_time": identical,
        "replayed_chain": {
            "chain_intact": verify["chain_intact"],
            "tampered_settlements": len(verify["tampered_settlements"]),
            "unsealed_settlements": len(verify["unsealed_settlements"]),
        },
        "seconds": {"batch": round(batch_seconds, 2), "one_at_a_time": round(single_seconds, 2)},
        "file_rewrites": {"batch": len(ordered), "one_at_a_time": len(recorded)},
    }


def positive_control_incident(path: Path = INCIDENT_PATH) -> dict[str, Any]:
    """The 2026-09-12 file: the confirmation must flag exactly the seq numbers written twice."""
    if not path.exists():
        return {"available": False}
    rows = read_rows(path)
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["seq"])] = counts.get(int(row["seq"]), 0) + 1
    duplicated = sorted(s for s, n in counts.items() if n > 1)
    flagged: set[int] = set()
    for row in rows:
        at = datetime.fromisoformat(str(row["decided_at"]))
        symbol = str(row["symbol"])
        before = LedgerVenue.frozen(rows, as_of=at, inclusive=False).position(symbol)
        venue = LedgerVenue.frozen(rows, as_of=at, inclusive=True)
        check = confirm_recorded(
            seq=int(row["seq"]), symbol=symbol, side=str(row["side"]),
            quantity=Decimal(str(row["quantity"])), status=venue, position=venue,
            position_before=before, authorised_quantity=None, authorised_side=None,
            deadline_s=0.0,
        )
        if check.alarm:
            flagged.add(int(row["seq"]))
    return {
        "available": True,
        "rows": len(rows),
        "duplicated_seqs": duplicated,
        "flagged_seqs": sorted(flagged),
        "exactly_the_duplicates": sorted(flagged) == duplicated,
    }


# --- the whole measurement -----------------------------------------------------------------------


def run(*, ledger_path: Path = LEDGER_PATH, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    raw = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in raw if r.get("kind", "decision") == "decision"]
    cycles, unparsed = parse_cycle_logs(runs_dir)
    with tempfile.TemporaryDirectory(prefix="execution-truth-consent-") as tmp:
        consent = run_consent_scenarios(Path(tmp) / "spent.jsonl")
    decisions = replay_logged_decisions(cycles, rows, _risk_ceilings())
    voided = {264, 265}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": "python -m argus.eval.execution_truth",
        "qwen_calls": 0,
        "fault_injection": {
            "fill_confirmation": run_fill_scenarios(),
            "batch_writes": run_batch_scenarios(),
            "shown_ids": run_id_scenarios(),
            "consent_gate": consent,
        },
        "replay": {
            "ledger_rows": len(raw),
            "decisions_on_ledger": len(rows),
            "cycle_logs_parsed": len(cycles),
            "cycle_logs_unparsed": unparsed,
            "logged_decisions": decisions,
            "false_alarms": [s for s in decisions["alarm_seqs"] if s not in voided],
            "true_alarms_on_known_voided_rows": [s for s in decisions["alarm_seqs"]
                                                 if s in voided],
            "logged_settlements": replay_logged_settlements(cycles, rows),
            "settlement_batches": replay_settlement_batches(raw),
            "incident_positive_control": positive_control_incident(),
        },
    }


def main() -> int:  # pragma: no cover - CLI
    report = run()
    write(REPORT_PATH, report)
    fill = report["fault_injection"]["fill_confirmation"]["by_method"]
    batch = report["fault_injection"]["batch_writes"]["by_method"]
    ids = report["fault_injection"]["shown_ids"]["by_method"]
    replay = report["replay"]
    print(json.dumps({
        "fill_confirmation": fill,
        "batch_writes": batch,
        "shown_ids": ids,
        "consent": {k: report["fault_injection"]["consent_gate"][k] for k in (
            "scenarios", "correct", "real_money_orders_sent_without_valid_consent")},
        "replay_false_alarms": replay["false_alarms"],
        "replay_true_alarms": replay["true_alarms_on_known_voided_rows"],
        "settlement_replay": {k: replay["settlement_batches"][k] for k in (
            "settlements_replayed", "batches", "batch_file_identical_to_one_at_a_time",
            "replayed_chain", "file_rewrites")},
        "incident": replay["incident_positive_control"],
    }, indent=2, default=str))
    return 0


__all__ = [
    "BATCH_SCENARIOS",
    "FILL_SCENARIOS",
    "REPORT_PATH",
    "BatchVenue",
    "LoggedStatus",
    "RecordingClient",
    "ScriptedVenue",
    "SimClock",
    "fixture_book",
    "main",
    "parse_cycle_logs",
    "positive_control_incident",
    "replay_logged_decisions",
    "replay_settlement_batches",
    "run",
    "run_batch_scenarios",
    "run_consent_scenarios",
    "run_fill_scenarios",
    "run_id_scenarios",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
