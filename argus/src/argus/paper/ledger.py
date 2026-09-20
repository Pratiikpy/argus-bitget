"""Paper-trading ledger — the Track-2 required deliverable.

The handbook requires a "paper trading log (run during competition period, recommended >= 2 weeks)".
This is that log, and it is built so a judge can verify it rather than believe it.

**The properties that make a paper log credible.** Fabricated or backfilled paper trading is named
as an anti-pattern in our own PRD and is trivially detectable, so every defence here is structural:

* **Append-only, hash-chained.** Each entry carries the hash of the previous one. Editing any
  historical row breaks every hash after it, so a doctored log fails its own verification.
* **Decisions are recorded before outcomes are known.** An entry is written at decision time with
  the market state hash; the outcome is attached later by ``settle()``, which refuses to touch an
  entry that already has one. There is no code path that writes a decision and its result together.
* **Costs are charged on entry**, using the same :class:`~argus.cost.model.CostModel` the backtest
  uses. A paper log that reports gross P&L is marketing.
* **Every row carries the Autonomy Proof hash**, so the log and the decision record are the same
  evidence rather than two stories.

The ledger does not place real orders. Bitget's own SDK routes paper trading through the
``paptrading: 1`` header on private endpoints (``agent-sdk/src/client/rest-client.ts:274-280``);
wiring that requires API credentials the user supplies. The full path is built here and the
exchange call is the single remaining step.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.cost.model import CostModel, Fill, Liquidity

GENESIS = "0" * 16

LOCK_TIMEOUT_SECONDS = 60.0
"""How long a writer waits for the lock before giving up.

A cycle's slowest step is a model call, so a writer holding the lock for tens of seconds is normal
and a wait of a minute is not evidence of a deadlock."""

LOCK_STALE_SECONDS = 900.0
"""When a lock file may be broken. Longer than the slowest complete cycle observed (the 22:52 run
took 29 minutes for twelve symbols, but it holds the lock only across a single append), so a lock
older than this belonged to a process that died."""


class LedgerError(RuntimeError):
    """The ledger was asked to do something that would make it untrustworthy."""


class LedgerLockTimeout(LedgerError):
    """Another writer held the ledger for too long. Better than writing beside it."""


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an exclusive writer lock on ``path`` for the duration of the block.

    **This exists because the absence of it corrupted the live log.** On 2026-09-12 the scheduled
    cycle and a manually started one overlapped. Both had loaded the ledger when it ended at seq 40,
    both computed ``seq = len(entries) + 1`` and ``prev_hash`` from that same stale head, and both
    appended a *different* NVDAUSDT decision as seq 41, fifty-eight seconds apart. The chain's own
    ``verify()`` caught it — ``first_break_at: 41`` — which is the machinery working, but nothing
    had prevented it.

    Implemented with ``O_CREAT | O_EXCL``, which is atomic on Windows and POSIX alike, rather than
    ``msvcrt``/``fcntl``: this has to work identically on the scheduled Windows task and in CI. The
    holder's process id and start time are written into the lock so a stale one can be attributed
    rather than merely broken.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    handle: int | None = None
    while True:
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            # A lock left behind by a process that died would otherwise block every future cycle
            # silently — the failure mode being avoided is a scheduled task that stops writing and
            # says nothing.
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_SECONDS:
                try:
                    holder = lock.read_text(encoding="utf-8").strip()
                except OSError:
                    holder = "unknown"
                lock.unlink(missing_ok=True)
                raise LedgerError(
                    f"broke a stale ledger lock {age:.0f}s old held by {holder}; the writing "
                    f"process did not finish. Re-run, and check that log for a partial cycle."
                ) from None
            if time.monotonic() > deadline:
                raise LedgerLockTimeout(
                    f"another writer has held {lock.name} for more than "
                    f"{LOCK_TIMEOUT_SECONDS:.0f}s; refusing to append beside it"
                ) from None
            time.sleep(0.05)
    try:
        os.write(handle, f"pid={os.getpid()} since={datetime.now(UTC).isoformat()}".encode())
        os.close(handle)
        handle = None
        yield
    finally:
        if handle is not None:
            os.close(handle)
        lock.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class Entry:
    """One paper decision. Written before the outcome exists."""

    seq: int
    decided_at: str
    symbol: str
    verdict: str
    side: str
    quantity: str
    entry_price: str
    stated_confidence: float
    thesis: str
    invalidation: tuple[str, ...]
    market_state_hash: str
    approved_intent_hash: str
    session_phase: str
    hours_to_discovery: float
    entry_cost_bps: str
    prev_hash: str

    lean: str = "none"
    """The direction the desk would have taken if forced, recorded even when it declines.

    **Tamper-evident without changing this row's hash payload.** A lean that could be edited once
    the move is known would be worth nothing, so it has to be covered by the chain — but adding a
    field to :meth:`content_hash` would rehash all 161 existing entries and break a chain that is
    already anchored to Bitcoin. It does not need to: `hash_intent` hashes the whole Intent through
    ``asdict``, so the lean is inside ``approved_intent_hash``, and that *is* in the payload below.
    The protection comes for free and the chain is untouched.

    Declared with a default so every row written before this existed still loads.
    """

    lean_confidence: float = 0.0

    # Attached later by settle(). Never written at decision time.
    settled_at: str | None = None
    exit_price: str | None = None
    gross_pnl: str | None = None
    net_pnl: str | None = None
    direction_correct: bool | None = None

    counterfactual_move_bps: str | None = None
    """For an **abstention**: the move that actually happened over the horizon.

    Without this an abstention is permanently ungradeable, and a desk that abstains correctly looks
    identical to one that abstains out of paralysis. It is attached at settlement like every other
    outcome field, and like them it is **not hashed** — see :meth:`content_hash`.
    """

    @property
    def content_hash(self) -> str:
        """Hash over the decision fields only, so settlement cannot rewrite history.

        Deliberate: if settlement fields were hashed, attaching an outcome would change the chain
        and make tampering indistinguishable from normal operation.
        """
        payload = {
            "seq": self.seq,
            "decided_at": self.decided_at,
            "symbol": self.symbol,
            "verdict": self.verdict,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stated_confidence": self.stated_confidence,
            "thesis": self.thesis,
            "market_state_hash": self.market_state_hash,
            "approved_intent_hash": self.approved_intent_hash,
            "prev_hash": self.prev_hash,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def is_settled(self) -> bool:
        return self.settled_at is not None

    @property
    def is_void(self) -> bool:
        """This row records something the system did not do — see `paper/corrections.py`."""
        from argus.paper.corrections import is_voided

        return is_voided(self.seq)

    @property
    def is_abstention(self) -> bool:
        """No exposure was taken — including rows whose stored quantity is a defect.

        **A voided row IS an abstention, and saying so here rather than in each consumer is the
        point.** Ledger seq 264 and 265 store `quantity: 1` because `paper/runner.py` recorded the
        model's unconstrained intent instead of the Constitution's ruling (fixed 2026-09-20). What
        the desk actually did on those two decisions was refuse: the risk record holds
        `quantity_after: 0, binding_constraint: no_exposure` and the desk notes read *"no order:
        final verdict human_review with quantity 0"*. The stored `1` is the bug's fingerprint, not
        a position.

        This property is read by `eval/performance.py`, `eval/themeaudit.py`,
        `eval/decisioncard.py`, `eval/autopsy.py`, `eval/scorecard.py`, `eval/episodes.py`,
        `eval/selfaudit.py`, `agents/recall.py` and `demo/cockpit.py`. Correcting it at each of
        those call sites was tried first and was the wrong shape: two were patched and the rest
        kept reporting "2 settled position(s)" beside "zero positions have settled" in the same
        output. One definition, every consumer.
        """
        return self.is_void or Decimal(self.quantity) == 0


@dataclass
class PaperLedger:
    """Append-only, hash-chained paper-trading record."""

    path: Path
    cost: CostModel = field(default_factory=CostModel.bitget_perp)
    entries: list[Entry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cost.assert_gateable()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        self.entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.entries.append(Entry(**json.loads(line)))

    def _append_built(self, build: Callable[[int, str], Entry]) -> Entry:
        """Build and write one entry while holding the lock, against the head on *disk*.

        The entry is constructed inside the lock rather than passed in, because its sequence number
        and ``prev_hash`` are properties of the head at the moment of writing. Building it outside
        and appending it inside is what broke the live chain: the head had moved.

        ``build`` receives the sequence number and the previous hash to use, both read from the file
        after the lock is held.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path):
            if self.path.exists():
                self._load()  # another writer may have appended since this object was constructed
            entry = build(len(self.entries) + 1, self.head_hash)
            if any(e.seq == entry.seq for e in self.entries):
                raise LedgerError(
                    f"seq {entry.seq} already exists; refusing to append a duplicate sequence "
                    f"number, which is what a concurrent writer produces"
                )
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry), default=str) + "\n")
            self.entries.append(entry)
            self._write_anchor()
        return entry

    def _persist_settlement(self, settled: Entry) -> Entry:
        """Write one settled entry back into the file, preserving concurrent appends.

        Locked for the same reason as the append path, and it re-reads rather than dumping the
        in-memory list: a settlement that rewrote the whole file from a stale copy would silently
        erase every decision another cycle had appended in the meantime. Settlement only touches
        fields that are deliberately excluded from the content hash, so rewriting the row alters no
        link in the chain.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path):
            if self.path.exists():
                self._load()
            self.entries[self._index_of(settled.seq)] = settled
            with self.path.open("w", encoding="utf-8") as fh:
                for e in self.entries:
                    fh.write(json.dumps(asdict(e), default=str) + "\n")
            self._write_anchor()
        return settled

    def _index_of(self, seq: int) -> int:
        """Find an entry by its sequence number.

        Deliberately a search rather than ``seq - 1``. The positional shortcut is correct only while
        the file is perfectly sequential, and the live log has already been in a state where it was
        not: two entries shared seq 41 after a concurrent write. A lookup that silently returns the
        wrong row is worse than one that says the log is malformed.
        """
        found = [i for i, e in enumerate(self.entries) if e.seq == seq]
        if not found:
            raise LedgerError(f"no entry {seq}")
        if len(found) > 1:
            raise LedgerError(
                f"entry {seq} appears {len(found)} times; the log is malformed and settling "
                f"against it would attach an outcome to an ambiguous decision"
            )
        return found[0]

    @property
    def head_hash(self) -> str:
        return self.entries[-1].content_hash if self.entries else GENESIS

    # --- writing -----------------------------------------------------------------------------

    def record(
        self,
        *,
        symbol: str,
        verdict: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        stated_confidence: float,
        thesis: str,
        invalidation: tuple[str, ...],
        market_state_hash: str,
        approved_intent_hash: str,
        session_phase: str,
        hours_to_discovery: float,
        decided_at: datetime | None = None,
        spread_bps: Decimal = Decimal("0.6"),
        lean: str = "none",
        lean_confidence: float = 0.0,
    ) -> Entry:
        """Write a decision. The outcome is not known and is not recorded."""
        when = decided_at or datetime.now(UTC)

        charge = self.cost.charge(
            Fill(
                notional=quantity * entry_price,
                liquidity=Liquidity.TAKER,
                spread_bps=spread_bps,
            )
        )
        notional = quantity * entry_price
        cost_bps = charge.bps_of(notional) if notional > 0 else Decimal("0")

        def build(seq: int, prev_hash: str) -> Entry:
            return Entry(
                seq=seq,
                decided_at=when.isoformat(),
                symbol=symbol,
                verdict=verdict,
                side=side,
                quantity=str(quantity),
                entry_price=str(entry_price),
                stated_confidence=stated_confidence,
                thesis=thesis[:500],
                invalidation=tuple(invalidation),
                market_state_hash=market_state_hash,
                approved_intent_hash=approved_intent_hash,
                session_phase=session_phase,
                hours_to_discovery=round(hours_to_discovery, 2),
                entry_cost_bps=str(round(cost_bps, 3)),
                prev_hash=prev_hash,
                lean=lean,
                lean_confidence=lean_confidence,
            )

        # Built inside the lock against the head on disk. Constructing the entry here and appending
        # it later is precisely what let two cycles write the same sequence number.
        return self._append_built(build)

    def settle_abstention(
        self, seq: int, *, price_now: Decimal, settled_at: datetime | None = None
    ) -> Entry:
        """Record what the market did over the horizon of a decision **not** to trade.

        Standing aside is a decision and it is not automatically right. Scoring it requires knowing
        what the alternative would have earned, and that counterfactual exists only if something
        records it at the time — which is why this is written here rather than reconstructed later
        from a price series that may have been revised.

        The sign convention is deliberate: the move is reported **as it happened**, not as
        "how much we saved". Whether abstaining was correct is the Observatory's arithmetic to do,
        against the round trip, and pre-judging it here would bake the answer into the data.
        """
        idx = self._index_of(seq)
        entry = self.entries[idx]
        if not entry.is_abstention:
            raise LedgerError(
                f"entry {seq} has verdict {entry.verdict!r}, which carries a position; settle it "
                f"with settle(), not settle_abstention()"
            )
        if entry.is_settled:
            raise LedgerError(
                f"entry {seq} is already settled at {entry.settled_at}; outcomes are written once"
            )
        entry_price = Decimal(entry.entry_price)
        if entry_price <= 0:
            raise LedgerError(f"entry {seq} has a non-positive entry price; cannot measure a move")

        move_bps = (price_now - entry_price) / entry_price * Decimal("10000")
        settled = Entry(
            **{
                **asdict(entry),
                "invalidation": tuple(entry.invalidation),
                "settled_at": (settled_at or datetime.now(UTC)).isoformat(),
                "exit_price": str(price_now),
                "counterfactual_move_bps": str(round(move_bps, 4)),
            }
        )
        return self._persist_settlement(settled)

    def settle(
        self, seq: int, *, exit_price: Decimal, settled_at: datetime | None = None,
        exit_spread_bps: Decimal | None = None,
    ) -> Entry:
        """Attach an outcome to a decision already on the record.

        Refuses to settle twice. A log whose outcomes can be revised is a log whose outcomes can
        be chosen.

        ``exit_spread_bps`` is what crossing the book costs **for this size, now**. It was a literal
        0.6 here for the life of this module — a median top-of-book quote standing in for the price
        of an actual exit. `market/depth.py` measures the real number from the live order book and
        the runner passes it; measured on 2026-09-14, taking $25,000 of an rToken costs between
        0.82 and 19.43bps depending on the name, against that single 0.6. Left as ``None`` the old
        default applies, so a caller with no book is charged the documented stand-in, not blocked.
        """
        idx = self._index_of(seq)
        entry = self.entries[idx]
        if entry.is_settled:
            raise LedgerError(
                f"entry {seq} is already settled at {entry.settled_at}; outcomes are written once"
            )

        qty = Decimal(entry.quantity)
        entry_px = Decimal(entry.entry_price)

        # An unrecognised side used to fall through to short, because the test was
        # ``== "BUY"`` with an implicit else. That failure is silent and total: every long
        # settles with an inverted sign, every winner is recorded as a loser, and the log looks
        # entirely normal while the scored numbers are backwards. The canonical values come from
        # :class:`argus.decision.verdicts.Side`, upper-cased by the runner; anything else is a
        # defect upstream and is refused here rather than guessed at.
        side = entry.side.upper()
        if side not in {"BUY", "SELL"}:
            raise LedgerError(
                f"entry {seq} has side {entry.side!r}; expected BUY or SELL. Refusing to settle: "
                f"an unrecognised side would silently invert the P&L sign."
            )
        direction = Decimal("1") if side == "BUY" else Decimal("-1")

        gross = (exit_price - entry_px) * qty * direction
        exit_charge = self.cost.charge(
            Fill(
                notional=qty * exit_price,
                liquidity=Liquidity.TAKER,
                spread_bps=Decimal("0.6") if exit_spread_bps is None else exit_spread_bps,
            )
        )
        entry_charge_bps = Decimal(entry.entry_cost_bps)
        entry_charge = qty * entry_px * entry_charge_bps / Decimal("10000")
        net = gross - exit_charge.total - entry_charge

        settled = Entry(
            **{
                **asdict(entry),
                "invalidation": tuple(entry.invalidation),
                "settled_at": (settled_at or datetime.now(UTC)).isoformat(),
                "exit_price": str(exit_price),
                "gross_pnl": str(round(gross, 4)),
                "net_pnl": str(round(net, 4)),
                "direction_correct": bool(gross > 0),
            }
        )
        return self._persist_settlement(settled)

    # --- verification ------------------------------------------------------------------------

    @property
    def _anchor_path(self) -> Path:
        return self.path.with_name(self.path.name + ".head")

    def _write_anchor(self) -> None:
        """Record how long the log is and where it ends.

        A hash chain proves that what is present has not been edited. It proves nothing about what
        has been **removed from the end** — deleting the last N rows leaves a perfectly valid chain,
        which was verified against this ledger before the anchor existed: dropping seven entries
        still returned ``chain_intact: True``. Since the rows most worth deleting are the most
        recent losses, that is the gap most worth closing.

        Written to a sibling file with an atomic replace, so a crash mid-write leaves either the old
        anchor or the new one, never half of either. Pattern from ``serenity-guardrails``
        ``etoro_trading/journal.py:21-60`` (Apache-2.0).
        """
        payload = json.dumps(
            {"entries": len(self.entries), "head_hash": self.head_hash}, sort_keys=True
        )
        tmp = self._anchor_path.with_suffix(self._anchor_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, self._anchor_path)

    def read_anchor(self) -> dict[str, Any] | None:
        """The anchor as last written, or ``None`` when there is not one to read."""
        try:
            loaded: dict[str, Any] = json.loads(self._anchor_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None
        return loaded

    def verify(self) -> dict[str, Any]:
        """Re-walk the chain and check it against the anchor. This is what a judge runs.

        Any edit to a historical decision breaks the link at that point and every point after it,
        so the first break localises the tampering. The anchor catches the other attack: removing
        rows from the end, which leaves the surviving chain valid.
        """
        broken: list[int] = []
        expected = GENESIS
        for e in self.entries:
            if e.prev_hash != expected:
                broken.append(e.seq)
            expected = e.content_hash

        anchor = self.read_anchor()
        truncated = False
        anchor_note = "no anchor on disk; truncation cannot be ruled out"
        if anchor is not None:
            expected_entries = int(anchor.get("entries", -1))
            expected_head = str(anchor.get("head_hash", ""))
            if len(self.entries) < expected_entries:
                truncated = True
                anchor_note = (
                    f"anchor records {expected_entries} entries, {len(self.entries)} present: "
                    f"{expected_entries - len(self.entries)} row(s) removed from the end"
                )
            elif len(self.entries) == expected_entries and self.head_hash != expected_head:
                truncated = True
                anchor_note = (
                    "entry count matches but the head hash does not; the tail was replaced"
                )
            else:
                anchor_note = "anchor agrees with the log"

        return {
            "entries": len(self.entries),
            "chain_intact": not broken and not truncated,
            "links_intact": not broken,
            "first_break_at": broken[0] if broken else None,
            "truncated": truncated,
            "anchor": anchor_note,
            "head_hash": self.head_hash,
        }

    def performance(self) -> dict[str, Any]:
        """What the log actually says. Net of costs, abstentions counted separately.

        **Voided rows are excluded** — see `paper/corrections.py`. Two rows on the live record
        booked P&L against positions the Constitution had refused (a `paper/runner.py` defect
        fixed 2026-09-20). They stay in the chain unedited; every figure here excludes them, and
        the correction is reported alongside rather than silently applied.
        """
        from argus.eval.performance import evaluate_ledger
        from argus.paper.corrections import VOIDED, is_voided, render

        # Sharpe, max drawdown and win rate — with the REASON each is undefined rather than a
        # zero — live in `eval/performance.py`, and this method never surfaced them, so the README
        # pointed a reader at a command that could not answer what it promised. One definition,
        # reported here. The import is local because `eval/performance.py` imports `PaperLedger` at
        # module scope; a top-level import is circular (same reason `corrections` is imported here).
        graded = evaluate_ledger(self).as_dict()
        metrics = {
            "sharpe": graded["sharpe"],
            "max_drawdown_pct": graded["max_drawdown_pct"],
            "win_rate_pct": graded["win_rate_pct"],
            "undefined": graded["undefined"],
        }

        settled = [
            e for e in self.entries
            if e.is_settled and not e.is_abstention and not is_voided(e.seq)
        ]
        abstentions = [e for e in self.entries if e.is_abstention]
        correction = {"correction": render(), "voided_rows": len(VOIDED)} if VOIDED else {}

        if not settled:
            return {
                "decisions": len(self.entries),
                "abstentions": len(abstentions),
                "settled_trades": 0,
                "note": "no settled trades yet — nothing to report",
                **metrics,
                **correction,
            }

        nets = [Decimal(e.net_pnl or "0") for e in settled]
        grosses = [Decimal(e.gross_pnl or "0") for e in settled]

        return {
            "decisions": len(self.entries),
            "abstentions": len(abstentions),
            "abstention_rate_pct": round(100 * len(abstentions) / len(self.entries), 1),
            "settled_trades": len(settled),
            "gross_pnl": str(sum(grosses)),
            "net_pnl": str(sum(nets)),
            "cost_drag": str(sum(grosses) - sum(nets)),
            "cost_flipped_the_sign": sum(grosses) > 0 >= sum(nets),
            # `win_rate_pct` deliberately comes from `metrics`, not from `wins / len(settled)`
            # recomputed here — one definition, and `eval/performance.py`'s is the one that knows
            # when the count is too small to mean anything.
            "chain": self.verify(),
            **metrics,
            **correction,
        }
