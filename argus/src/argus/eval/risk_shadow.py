"""Shadow evaluation: before a risk rule changes, replay the recorded orders through old and new.

A risk-layer change is the one kind of edit whose mistakes are invisible until they cost money: a
guard that is quietly looser still lets every test order through, because the tests were written
for the orders it already allowed. Standing Rule #2 — never guess — therefore needs a concrete form
for risk code, and this is it. Before a change to `execution/guard.py` or `risk/circuit.py` is
allowed to reach the live cycle, the **old and the new logic rule on the same order stream, side by
side**, and every ruling that would change is listed, with both rulings.

**Taken from letta-code** (Apache-2.0, ``src/permissions/checker.ts:152-197``;
``licenses/letta_code-APACHE-2.0.txt``). With ``LETTA_PERMISSIONS_DUAL_EVAL`` set, its permission
checker evaluates every call with both the primary and the shadow engine, logs a mismatch when the
decision or the matched rule differs, and acts only on the primary. **Changed:** letta-code shadows
live traffic as it arrives; a trading desk cannot wait for the order stream a new rule would matter
on, so this replays the orders already recorded, plus the order space around them, offline and
before the cutover. A mismatch here is not a log line; it is a list a person reads before deciding.

**The streams, and what each one is — never mixed in one number:**

* ``recorded`` — orders the live paper cycle actually sent to the venue gate, reconstructed from
  ``data/paper_ledger.jsonl`` and ``data/desk_notes.jsonl``. The runner records the gate's
  *output*, not its input, so an intent is replayable only when the gate allowed it unadjusted
  (input = output). Every decision that reached the gate and could not be replayed is counted and
  said so. On 2026-09-25 this stream is **two orders** — seq 264 and 265, both later voided
  (`paper/corrections.py`) — because the desk has abstained on nearly everything.
* ``counterfactual`` — every recorded decision point (real symbol, real side, real price at the
  moment of decision) at a ladder of sizes around each instrument's real venue limits. Synthetic
  sizes on real specifications; labelled as such.
* ``swept`` — the S13 self-check's order space (`eval/guard_selfcheck.py`, 190,944 orders), the
  cross product of every axis the venue rules branch on.
* ``probe`` — hand-built orders, one at each gate's own edge, and the edges the S20 certifier
  found: sides that are neither buy nor sell, sub-tick limit prices, and numbers wider than 28
  significant digits.

Instrument specifications for the ``recorded`` and ``counterfactual`` streams are the venue's own
rows from the public ``GET /api/v3/market/instruments`` (keyless), fetched once and frozen in
``data/risk_shadow_instruments.json`` so a rerun judges the same rules. They are **today's** rows,
not the rows in force at decision time, which the runner never recorded — a limit of the replay,
stated in the artefact.

**What counts as a changed ruling:** a different outcome (allowed, refused under another name, or
an exception where there was a ruling), a different allowed quantity or limit price, or a different
``adjusted`` flag. Numbers compare by value, so ``1`` and ``1.00`` are the same size. A ruling
whose only difference is its reason text is counted separately, as ``reason_only``.

    python -m argus.eval.risk_shadow                      # the standing pairs; writes the artefact
    python -m argus.eval.risk_shadow --old A.py --new B.py  # any two versions of the guard
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import re
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval import artefact, guard_selfcheck

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
REPORT_PATH = DATA / "risk_shadow.json"
SPECS_PATH = DATA / "risk_shadow_instruments.json"
BASELINES = DATA / "risk_baselines"
CURRENT_GUARD = ROOT / "src" / "argus" / "execution" / "guard.py"
CURRENT_CIRCUIT = ROOT / "src" / "argus" / "risk" / "circuit.py"
LEDGER = DATA / "paper_ledger.jsonl"
NOTES = DATA / "desk_notes.jsonl"
RISK_RECORDS = DATA / "risk_records.jsonl"

INSTRUMENTS_URL = "https://api.bitget.com/api/v3/market/instruments"

NOW = guard_selfcheck.NOW
"""The fixed clock the rate-window states are defined against."""

SWEPT_EXAMPLES = 25
"""Changes listed per transition for the swept stream. Every other stream lists every change; the
swept stream is regenerated identically on every run, so a group's remainder is reproducible."""


# --- loading a version of the logic from a file ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Engine:
    """One version of a risk module, loaded side by side with the others."""

    label: str
    path: str
    sha256: str
    module: types.ModuleType

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "path": self.path, "sha256": self.sha256}


def _register(name: str, module: types.ModuleType) -> None:
    # Dataclass creation looks its module up in sys.modules, so the module is registered, under a
    # name no real import will ever ask for, before its body runs.
    sys.modules[name] = module


def load_engine(path: Path, label: str) -> Engine:
    """Load a module's source from ``path`` (``.py`` or a frozen ``.py.txt``) as its own module."""
    text = path.read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    name = f"argus_shadow_{re.sub(r'[^0-9A-Za-z]+', '_', label)}_{digest[:12]}"
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    _register(name, module)
    loader.exec_module(module)
    shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return Engine(label, shown, digest, module)


def load_source(source: str, label: str) -> types.ModuleType:
    """Load a module from source text — how a mutant or a candidate rule is shadowed."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    name = f"argus_shadow_{re.sub(r'[^0-9A-Za-z]+', '_', label)}_{digest[:12]}"
    module = types.ModuleType(name)
    module.__file__ = f"<{label}>"
    _register(name, module)
    exec(compile(source, f"<{label}>", "exec"), module.__dict__)
    return module


# --- instrument specifications --------------------------------------------------------------------

VARIANT_ROWS: dict[str, dict[str, str]] = {
    # The four instruments of the S13 sweep (`eval/guard_selfcheck.py` ``instruments``), rebuilt
    # from raw rows so any version of the guard can parse them with its own ``Instrument``.
    "nvda": {},
    "offline": {"status": "limit_open"},
    "coarse": {"quantityMultiplier": "1", "minOrderQty": "1", "maxOrderQty": "5",
               "minOrderAmount": "50", "buyLimitPriceRatio": "0.05",
               "sellLimitPriceRatio": "0.01"},
    "sub_step_max": {"maxOrderQty": "0.005"},
    # Two more, for the probe stream: a whole-unit tick, where a sub-tick limit price steps to
    # zero, and steps that are not powers of ten, where a quotient does not terminate.
    "coarse_tick": {"priceMultiplier": "1"},
    "odd_step": {"quantityMultiplier": "0.03", "minOrderQty": "0.03", "priceMultiplier": "0.05"},
}


def variant_row(name: str) -> dict[str, Any]:
    row = dict(guard_selfcheck.NVDA_ROW)
    row.update(VARIANT_ROWS[name])
    return row


def fetch_rows(symbols: Iterable[str], *, timeout: int = 20) -> dict[str, dict[str, Any]]:
    """The venue's published rows for these symbols, from the public, keyless endpoint."""
    wanted = set(symbols)
    url = f"{INSTRUMENTS_URL}?{urllib.parse.urlencode({'category': 'USDT-FUTURES'})}"
    request = urllib.request.Request(url, headers={"User-Agent": "argus/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    if str(payload.get("code")) != "00000":
        raise RuntimeError(f"venue refused the instrument query: {payload.get('msg', payload)}")
    return {row["symbol"]: row for row in payload.get("data") or [] if row.get("symbol") in wanted}


def instrument_specs(symbols: Iterable[str], *, refresh: bool = False,
                     path: Path = SPECS_PATH) -> dict[str, Any]:
    """Frozen venue rows for ``symbols``: read from ``path``, fetched and frozen when absent."""
    wanted = sorted(set(symbols))
    if not refresh and path.exists():
        blob: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if set(wanted) <= set(blob.get("rows", {})) | set(blob.get("missing", [])):
            return blob
    rows = fetch_rows(wanted)
    blob = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": f"{INSTRUMENTS_URL}?category=USDT-FUTURES",
        "note": "today's rows; the rows in force at each decision were never recorded",
        "rows": {symbol: rows[symbol] for symbol in sorted(rows)},
        "missing": [symbol for symbol in wanted if symbol not in rows],
    }
    artefact.write(path, blob)
    return blob


# --- the order streams ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Intent:
    """One order to rule on, and where it came from."""

    stream: str
    ident: str
    instrument: str
    """A key of the stream's row table: a symbol, or a :data:`VARIANT_ROWS` name."""

    quantity: str
    price: str | None = None
    reference: str | None = None
    side: str = "buy"
    balance: str | None = None
    window: str = "none"
    """A :func:`window_states` name. Session-guard intents need a window; ``none`` is read as an
    empty one for them."""

    lookup: bool = False
    """Through :meth:`Guard.check` (with the instrument lookup) rather than ``validate``."""

    symbol: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"stream": self.stream, "id": self.ident, "instrument": self.instrument,
                "symbol": self.symbol or self.instrument, "quantity": self.quantity,
                "price": self.price, "reference": self.reference, "side": self.side,
                "balance": self.balance, "window": self.window, "lookup": self.lookup}


@functools.cache
def window_states() -> dict[str, tuple[float, ...] | None]:
    """The S13 sweep's rate-window states, as bare stamps any guard version can load."""
    return {
        name: None if limiter is None else tuple(limiter._stamps)
        for name, limiter in guard_selfcheck.limiter_states().items()
    }


ALLOWED_NOTE = re.compile(r"^\[guard\] allowed: (?P<q>\S+)(?: @ (?P<p>\S+))?(?P<adj> — .*)?$")
"""The runner's rendering of an allowed venue ruling (`execution/guard.py` ``Ruling.render``)."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def recorded_decisions(ledger: Path = LEDGER) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(ledger) if row.get("kind", "decision") == "decision"]


def ruling_line(note: str) -> str:
    """The one-line ruling at the head of a venue-gate note.

    The runner writes ``Ruling.render()`` today, one line. A note written with ``Ruling.explain()``
    carries the whole :class:`~argus.execution.guard.PermissionCheckTrace` beneath that line, and
    :data:`ALLOWED_NOTE` anchors at the end of the text, so the trace would make every allowed
    order unreplayable. Reading the head line keeps the replay working whichever of the two the
    runner records.
    """
    return note.split("\n", 1)[0]


def guard_notes(notes: Path = NOTES) -> dict[int, list[str]]:
    """The venue-gate lines the runner wrote beside each decision, by seq, head line only."""
    out: dict[int, list[str]] = {}
    for row in read_jsonl(notes):
        lines = [ruling_line(n) for n in row.get("notes", []) if n.startswith("[guard]")
                 and not n.startswith("[guard] venue rules")]
        if lines:
            out.setdefault(int(row["seq"]), []).extend(lines)
    return out


def recorded_intents(
    ledger: Path = LEDGER, notes: Path = NOTES,
) -> tuple[list[Intent], dict[str, Any]]:
    """The orders the live cycle actually sent to the venue gate, where they can be recovered."""
    decisions = {int(row["seq"]): row for row in recorded_decisions(ledger)}
    intents: list[Intent] = []
    unreplayable: list[dict[str, Any]] = []
    rulings = guard_notes(notes)
    for seq in sorted(rulings):
        row = decisions.get(seq)
        note = rulings[seq][-1]
        match = ALLOWED_NOTE.match(note)
        if row is None or match is None or match.group("adj"):
            unreplayable.append({
                "seq": seq, "note": note,
                "why": ("no ledger row" if row is None else
                        "the gate adjusted or refused the order, and the runner records the "
                        "gate's output, not its input"),
            })
            continue
        intents.append(Intent(
            "recorded", f"seq {seq}", row["symbol"], match.group("q"),
            price=match.group("p"), reference=str(row["entry_price"]),
            side=str(row["side"]).lower(), window="empty", lookup=True, symbol=row["symbol"],
        ))
    return intents, {
        "decisions_read": len(decisions),
        "reached_venue_gate": len(rulings),
        "replayable": len(intents),
        "not_replayable": unreplayable,
        "window": "replayed against an empty rate window; the live window was not recorded",
    }


NOTIONAL_LADDER: tuple[str, ...] = ("4.99", "5", "5.01", "100", "1000", "10000", "1000000")
"""Target notionals, in USDT, around the 5 USDT floor and up past every recorded instrument's
maximum. Divided by the recorded price without rounding, so the step is exercised too."""


def counterfactual_intents(
    decisions: Sequence[dict[str, Any]], rows: dict[str, dict[str, Any]],
) -> list[Intent]:
    """Every recorded decision point at a ladder of sizes around its instrument's real limits."""
    out: list[Intent] = []
    for row in decisions:
        symbol, price = str(row["symbol"]), Decimal(str(row["entry_price"]))
        if price <= 0:
            continue
        spec = rows.get(symbol)
        sizes = [str((Decimal(n) / price).quantize(Decimal("1e-9"))) for n in NOTIONAL_LADDER]
        if spec is not None:
            sizes += [str(spec["minOrderQty"]), str(Decimal(str(spec["maxOrderQty"])) * 2)]
        for index, size in enumerate(sizes):
            out.append(Intent(
                "counterfactual", f"seq {row['seq']} size {index}", symbol, size,
                reference=str(price), side=str(row["side"]).lower(), window="empty",
                lookup=True, symbol=symbol,
            ))
    return out


def swept_intents() -> Iterator[Intent]:
    for order in guard_selfcheck.order_space():
        yield Intent("swept", "", order.instrument, order.quantity, order.price, order.reference,
                     order.side, order.balance, order.window)


PROBES: tuple[tuple[str, dict[str, Any]], ...] = (
    # One order at each gate's own edge, so a defect in any gate shows within the first few
    # dozen orders rather than somewhere in the swept space.
    ("quantity not a number", {"quantity": "NaN"}),
    ("offline instrument", {"instrument": "offline"}),
    ("rate window at the cap", {"window": "at_cap"}),
    ("rate window at the cap, every stamp exactly one window old",
     {"window": "at_cap_on_the_boundary"}),
    ("limit price zero", {"price": "0"}),
    ("sell inside the buy band but outside the sell band",
     {"instrument": "coarse", "side": "sell", "price": "104"}),
    ("quantity exactly the venue minimum", {"instrument": "coarse"}),
    ("venue maximum below one step", {"instrument": "sub_step_max"}),
    ("balance just covers the order", {"balance": "100"}),
    # The edges the S20 certifier found.
    ("side neither buy nor sell", {"side": "hold"}),
    ("empty side", {"side": ""}),
    ("side with trailing space", {"side": "buy "}),
    ("side long", {"side": "long"}),
    ("sub-tick limit price, no reference", {"price": "0.005", "reference": None}),
    ("sub-tick limit price, reference", {"price": "0.005", "reference": "0.005"}),
    ("sub-tick limit price on a whole-unit tick",
     {"instrument": "coarse_tick", "price": "0.5", "reference": None}),
    ("sub-tick price, whole-unit tick, reference", {"instrument": "coarse_tick", "price": "0.5",
                                                   "reference": "0.5"}),
    ("31-digit quantity just under one step", {"quantity": "0.0099999999999999999999999999999"}),
    ("31-digit quantity just under two steps",
     {"quantity": "0.019999999999999999999999999999999", "reference": "1000"}),
    ("31-digit quantity just under the maximum",
     {"quantity": "51999.999999999999999999999999999", "reference": "100"}),
    ("notional a hair under the floor",
     {"quantity": "0.05", "reference": "99.99999999999999999999999999999"}),
    ("notional a hair over the balance",
     {"quantity": "1", "reference": "100.00000000000000000000000000001", "balance": "100"}),
    ("limit a hair above the band, stepped back onto the tick",
     {"price": "102.00000000000000000000000000001", "reference": "100"}),
    ("limit on the tick, band edge from a 31-digit reference",
     {"price": "102", "reference": "99.99999999999999999999999999999"}),
    ("limit exactly on the band edge", {"price": "102", "reference": "100"}),
    ("non-terminating step quotient", {"instrument": "odd_step", "quantity": "0.1"}),
    ("non-terminating price quotient", {"instrument": "odd_step", "price": "100.12",
                                        "reference": "100"}),
    ("exact odd step", {"instrument": "odd_step", "quantity": "0.09", "reference": "100"}),
    ("unknown symbol through the session guard", {"lookup": True, "symbol": "ZZZZUSDT",
                                                  "instrument": "unlisted"}),
)
"""Each probe starts from one clean order — 1 contract of the NVDA variant at a 100 reference —
and changes only what its label says."""


def probe_intents() -> list[Intent]:
    out: list[Intent] = []
    for label, change in PROBES:
        base: dict[str, Any] = {"instrument": "nvda", "quantity": "1", "price": None,
                                "reference": "100", "side": "buy", "balance": None,
                                "window": "empty", "lookup": False, "symbol": ""}
        base.update(change)
        if base["lookup"] and not base["symbol"]:
            base["symbol"] = "NVDAUSDT"
        out.append(Intent(
            "probe", label, base["instrument"], base["quantity"], base["price"],
            base["reference"], base["side"], base["balance"], base["window"], base["lookup"],
            base["symbol"],
        ))
    return out


# --- ruling on one intent with one version of the guard -------------------------------------------


def _dec(raw: str | None) -> Decimal | None:
    return None if raw is None else Decimal(raw)


@dataclass(frozen=True, slots=True)
class Run:
    """One version of the guard ruling on one intent, with what the certifier needs to replay it."""

    intent: Intent
    ruling: Any | None
    error: BaseException | None
    instrument: Any | None
    stamps_before: tuple[float, ...] | None
    stamps_after: tuple[float, ...] | None
    max_orders: int
    window_seconds: float


def run_guard(module: types.ModuleType, intent: Intent, row: dict[str, Any] | None) -> Run:
    """Rule on ``intent`` with ``module``'s own classes. ``row`` ``None``: an unlisted symbol."""
    instrument = None if row is None else module.Instrument.from_payload(row)
    stamps = window_states()[intent.window]
    if intent.lookup and stamps is None:
        stamps = ()
    limiter = None
    if stamps is not None:
        limiter = module.RateLimiter(max_orders=10, window_seconds=60.0)
        limiter._stamps.extend(stamps)
    kwargs: dict[str, Any] = {
        "quantity": Decimal(intent.quantity), "price": _dec(intent.price),
        "reference_price": _dec(intent.reference), "side": intent.side,
        "available_balance": _dec(intent.balance), "now": NOW,
    }
    ruling: Any | None = None
    error: BaseException | None = None
    try:
        if intent.lookup:
            symbol = intent.symbol or intent.instrument
            # An unlisted symbol is asked about beside one instrument that *is* published, so a
            # guard that fell back to some other instrument's rules would be caught doing it.
            table = ({symbol: instrument} if instrument is not None else
                     {"NVDAUSDT": module.Instrument.from_payload(variant_row("nvda"))})
            ruling = module.Guard(instruments=table, limiter=limiter).check(symbol, **kwargs)
        else:
            ruling = module.validate(instrument, limiter=limiter, **kwargs)
    except Exception as exc:  # the shadow reports an exception as an outcome; it never hides one
        error = exc
    return Run(intent, ruling, error, instrument, stamps,
               None if limiter is None else tuple(limiter._stamps), 10, 60.0)


@dataclass(frozen=True, slots=True)
class Outcome:
    """A ruling reduced to what the shadow compares."""

    denial: str | None
    quantity: Decimal | None
    price: Decimal | None
    adjusted: bool | None
    reason: str
    raised: str | None

    @classmethod
    def of(cls, run: Run) -> Outcome:
        if run.error is not None or run.ruling is None:
            name = type(run.error).__name__ if run.error is not None else "no ruling"
            return cls(None, None, None, None, str(run.error or ""), name)
        ruling = run.ruling
        return cls(str(ruling.denial), Decimal(str(ruling.quantity)),
                   None if ruling.price is None else Decimal(str(ruling.price)),
                   bool(ruling.adjusted), str(ruling.reason), None)

    def label(self) -> str:
        if self.raised is not None:
            return f"raised {self.raised}"
        if self.denial == "none":
            return "allowed"
        return f"denied {self.denial}"

    def same_ruling(self, other: Outcome) -> bool:
        return (self.raised, self.denial, self.quantity, self.price, self.adjusted) == (
            other.raised, other.denial, other.quantity, other.price, other.adjusted)

    def as_dict(self) -> dict[str, Any]:
        return {"outcome": self.label(),
                "quantity": None if self.quantity is None else str(self.quantity),
                "price": None if self.price is None else str(self.price),
                "adjusted": self.adjusted, "reason": self.reason}


def rows_for(intent: Intent, specs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The venue row an intent is ruled against; ``None`` means the venue published none."""
    if intent.stream in ("swept", "probe"):
        return variant_row(intent.instrument) if intent.instrument in VARIANT_ROWS else None
    return specs.get(intent.instrument)


# --- the comparison -------------------------------------------------------------------------------


def shadow_guard(
    old: types.ModuleType, new: types.ModuleType, intents: Iterable[Intent],
    specs: dict[str, dict[str, Any]], *, list_all: bool = False,
) -> dict[str, Any]:
    """Rule on every intent with both versions and report every ruling that differs."""
    by_stream: dict[str, dict[str, Any]] = {}
    for intent in intents:
        row = rows_for(intent, specs)
        before, after = Outcome.of(run_guard(old, intent, row)), Outcome.of(
            run_guard(new, intent, row))
        stream = by_stream.setdefault(intent.stream, {
            "intents": 0, "changed": 0, "reason_only": 0, "unchanged": 0,
            "transitions": Counter(), "changes": [], "listed_per_transition": Counter(),
        })
        stream["intents"] += 1
        if before.same_ruling(after):
            stream["reason_only" if before.reason != after.reason else "unchanged"] += 1
            continue
        stream["changed"] += 1
        move = f"{before.label()} -> {after.label()}"
        stream["transitions"][move] += 1
        if (list_all or intent.stream != "swept"
                or stream["listed_per_transition"][move] < SWEPT_EXAMPLES):
            stream["listed_per_transition"][move] += 1
            stream["changes"].append({"intent": intent.as_dict(), "old": before.as_dict(),
                                      "new": after.as_dict()})
    for stream in by_stream.values():
        stream["transitions"] = dict(stream["transitions"].most_common())
        listed = sum(stream.pop("listed_per_transition").values())
        stream["changes_listed"] = listed
        stream["changes_complete"] = listed == stream["changed"]
    return by_stream


def book_grid() -> Iterator[tuple[dict[str, Any], str]]:
    """Books around every circuit-breaker threshold, crossed with every verdict.

    Every value is exact in decimal, including the session drawdown (a session opened at 93.75
    against equity 90 is exactly 4%), so a threshold is met exactly rather than approximately.
    """
    books = [(equity, equity) for equity in ("100", "98.01", "98", "95", "94", "90.01", "90",
                                             "85")]
    # Session drawdown straddling its 4% halt: equity 90 against an opening of 93.74 (3.99%) and
    # of 93.75 (exactly 4%).
    books += [("90", "93.74"), ("90", "93.75")]
    for (equity, opened), losses, sigma, age, open_positions, verdict in itertools.product(
        books, (3, 4), ("3", "-2.99", "-3", "-3.01"),
        (timedelta(hours=6), timedelta(hours=6, seconds=1)), (0, 1),
        ("trade", "reduce", "hedge", "delay", "no_trade", "human_review", "data_insufficient"),
    ):
        yield ({"equity": Decimal(equity), "peak_equity": Decimal("100"),
                "session_open_equity": Decimal(opened), "consecutive_losses": losses,
                "open_positions": open_positions, "evidence_age": age,
                "realised_move_sigma": Decimal(sigma)}, verdict)


def circuit_outcome(module: types.ModuleType, book: dict[str, Any], verdict: str) -> str:
    """A circuit ruling as one comparable line: activation, rules fired, verdict, multiplier."""
    try:
        state = module.BookState(**book)
        ruling = module.apply(module.Verdict(verdict), state)
    except Exception as exc:  # reported as an outcome, as for the guard
        return f"raised {type(exc).__name__}"
    trips = ",".join(sorted(f"{t.rule}>{t.demands}" for t in ruling.trips))
    return (f"{ruling.activation} [{trips}] {verdict}->{ruling.verdict} "
            f"x{Decimal(str(ruling.risk_multiplier)).normalize()}")


def shadow_circuit(old: types.ModuleType, new: types.ModuleType) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    total = 0
    for book, verdict in book_grid():
        total += 1
        before, after = circuit_outcome(old, book, verdict), circuit_outcome(new, book, verdict)
        if before != after:
            changes.append({"book": {k: str(v) for k, v in book.items()}, "verdict": verdict,
                            "old": before, "new": after})
    return {"books_x_verdicts": total, "changed": len(changes), "changes": changes}


# --- the report -----------------------------------------------------------------------------------


def streams(
    *, refresh: bool = False,
) -> tuple[list[Intent], dict[str, dict[str, Any]], dict[str, Any]]:
    """Every stream's intents, the venue rows they need, and what each stream is."""
    recorded, recorded_stats = recorded_intents()
    decisions = recorded_decisions()
    symbols = {str(row["symbol"]) for row in decisions} | {i.instrument for i in recorded}
    specs_blob = instrument_specs(symbols, refresh=refresh)
    rows: dict[str, dict[str, Any]] = specs_blob["rows"]
    counterfactual = counterfactual_intents(decisions, rows)
    probes = probe_intents()
    described = {
        "recorded": recorded_stats,
        "counterfactual": {
            "decision_points": len(decisions), "intents": len(counterfactual),
            "sizes": f"notionals {list(NOTIONAL_LADDER)} USDT at the recorded price, plus each "
                     f"instrument's minimum and twice its maximum",
            "what": "synthetic sizes on real symbols, real sides and real recorded prices",
        },
        "swept": {"orders": 190_944, "source": "eval/guard_selfcheck.py order_space()"},
        "probe": {"orders": len(probes), "labels": [label for label, _ in PROBES]},
        "instrument_specs": {k: v for k, v in specs_blob.items() if k != "rows"}
        | {"symbols": sorted(rows)},
    }
    return [*recorded, *counterfactual, *probes], rows, described


def guard_pairs() -> list[tuple[Engine, Engine]]:
    current = load_engine(CURRENT_GUARD, "guard current")
    pairs = []
    for name, label in (("guard_2026-09-25_pre_s20.py.txt", "guard before S20"),
                        ("guard_2026-09-24_deployed.py.txt", "guard in deploy/api")):
        path = BASELINES / name
        if path.exists():
            pairs.append((load_engine(path, label), current))
    return pairs


def run(
    *, refresh: bool = False, list_all: bool = False, path: Path = REPORT_PATH,
) -> dict[str, Any]:
    intents, rows, described = streams(refresh=refresh)
    pairs_out = []
    for old, new in guard_pairs():
        by_stream = shadow_guard(old.module, new.module, [*intents, *swept_intents()], rows,
                                 list_all=list_all)
        realistic = sum(by_stream.get(s, {}).get("changed", 0)
                        for s in ("recorded", "counterfactual"))
        pairs_out.append({
            "old": old.as_dict(), "new": new.as_dict(), "by_stream": by_stream,
            "changed_on_realistic_orders": realistic,
            "changed_total": sum(s["changed"] for s in by_stream.values()),
        })
    circuit_pairs = []
    baseline = BASELINES / "circuit_2026-09-25.py.txt"
    if baseline.exists():
        old_c = load_engine(baseline, "circuit baseline")
        new_c = load_engine(CURRENT_CIRCUIT, "circuit current")
        circuit_pairs.append({"old": old_c.as_dict(), "new": new_c.as_dict(),
                              **shadow_circuit(old_c.module, new_c.module)})
    report = {
        "what": "old versus new risk logic on the same order stream; every ruling that differs",
        "source": "letta-code src/permissions/checker.ts:152-197 (Apache-2.0), replayed offline",
        "generated_at": datetime.now(UTC).isoformat(),
        "streams": described,
        "guard_pairs": pairs_out,
        "circuit_pairs": circuit_pairs,
        "limits": [
            "recorded intents are replayable only where the gate allowed them unadjusted, because "
            "the runner records the gate's output and not its input",
            "the recorded rate window was not recorded; recorded and counterfactual intents are "
            "replayed against an empty window",
            "instrument rows are today's, not the rows in force at each decision",
        ],
    }
    artefact.write(path, report)
    return report


def _print_pair(pair: dict[str, Any]) -> None:  # pragma: no cover - CLI
    print(f"\n{pair['old']['label']} ({pair['old']['sha256'][:12]}) -> "
          f"{pair['new']['label']} ({pair['new']['sha256'][:12]})")
    for name, stream in pair["by_stream"].items():
        print(f"  {name:<15} {stream['intents']:>7} intents  {stream['changed']:>6} changed  "
              f"{stream['reason_only']:>6} reason-only")
        for move, count in stream["transitions"].items():
            print(f"      {count:>6}  {move}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="replay orders through old and new risk logic")
    parser.add_argument("--old", type=Path, help="a version of execution/guard.py to shadow")
    parser.add_argument("--new", type=Path, default=CURRENT_GUARD)
    parser.add_argument("--refresh", action="store_true", help="re-fetch the venue rows")
    parser.add_argument("--all", action="store_true", help="list every swept change, not 25")
    args = parser.parse_args(argv)
    if args.old is not None:
        intents, rows, _ = streams(refresh=args.refresh)
        old, new = load_engine(args.old, "old"), load_engine(args.new, "new")
        by_stream = shadow_guard(old.module, new.module, [*intents, *swept_intents()], rows,
                                 list_all=args.all)
        pair = {"old": old.as_dict(), "new": new.as_dict(), "by_stream": by_stream}
        _print_pair(pair)
        for name in ("recorded", "counterfactual", "probe"):
            for change in by_stream.get(name, {}).get("changes", []):
                print(f"  [{name}] {change['intent']['id']}: {change['old']['outcome']} -> "
                      f"{change['new']['outcome']}")
        realistic = sum(by_stream.get(s, {}).get("changed", 0)
                        for s in ("recorded", "counterfactual"))
        return 1 if realistic else 0
    report = run(refresh=args.refresh, list_all=args.all)
    for pair in report["guard_pairs"]:
        _print_pair(pair)
    for pair in report["circuit_pairs"]:
        print(f"\ncircuit {pair['old']['sha256'][:12]} -> {pair['new']['sha256'][:12]}: "
              f"{pair['changed']} of {pair['books_x_verdicts']} rulings changed")
    return 0


__all__ = [
    "ALLOWED_NOTE",
    "PROBES",
    "REPORT_PATH",
    "VARIANT_ROWS",
    "Engine",
    "Intent",
    "Outcome",
    "Run",
    "book_grid",
    "circuit_outcome",
    "counterfactual_intents",
    "guard_notes",
    "instrument_specs",
    "load_engine",
    "load_source",
    "main",
    "probe_intents",
    "read_jsonl",
    "recorded_decisions",
    "recorded_intents",
    "rows_for",
    "ruling_line",
    "run",
    "run_guard",
    "shadow_circuit",
    "shadow_guard",
    "streams",
    "swept_intents",
    "variant_row",
    "window_states",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
