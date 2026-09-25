"""Real market-by-order feeds, normalised into :class:`argus.eval.l3queue.L3Event`.

Three venues, three different ways of saying the same five things, each read off the venue's own
documentation or its own records rather than assumed:

**CME Globex via DataBento MBO** (``GLBX.MDP3``, ``.dbn.zst``). Record semantics read from a real
file (see :mod:`argus.eval.l3queue`): a match is one ``T`` (aggressor side) then one ``F`` per
resting order hit, then the book change as ``C`` or ``M``. ``flags & 128`` (``F_LAST``) closes one
exchange event. hftbacktest's own converter
(``py-hftbacktest/hftbacktest/data/utils/databento.py``) maps
the same six actions one-for-one; so does this.

**NASDAQ via LOBSTER** (sample files, 2012-06-21). Semantics from LOBSTER's own
``LOBSTER_SampleFiles_ReadMe.txt`` (version 01 Sept 2013): type 1 submission, 2 partial
cancellation (size = shares removed, priority kept), 3 deletion, 4 execution of a visible order
(direction = the *resting* order's side, so the aggressor is the other one), 5 execution of a hidden
order, 7 halt. The orderbook file carries the top-N level totals after every message; the replay
adopts them (:meth:`TruthReplay.sync_level`) and censors any synthetic order whose queue stops
reconciling with them.

A Bitstamp-via-Tardis adapter was planned for a third venue and is not built: its feed has no
exchange-event boundary and would need its own reconciliation. Nothing here reads it.
"""

from __future__ import annotations

import gzip
import io
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval.l3queue import (
    ADD,
    CANCEL,
    CLEAR,
    EXEC,
    FILL,
    HALT,
    MODIFY,
    REDUCE,
    TRADE,
    L3Event,
    RealEpisode,
    ReplayConfig,
    TruthReplay,
)

F_LAST = 128
"""``databento_dbn`` ``F_LAST``: the last record of one exchange event for an instrument."""

_DBN_KIND = {"A": ADD, "C": CANCEL, "M": MODIFY, "F": FILL, "T": TRADE, "R": CLEAR}


class FeedError(RuntimeError):
    """A feed could not be read honestly. Raised rather than replaying a partial book."""


# --- CME / DataBento ---------------------------------------------------------------------------


def databento_events(path: Path, *, instrument_id: int | None = None) -> list[L3Event]:
    """Every record of one instrument, in file (``ts_recv``) order, normalised.

    ``N`` (none) actions carry no book information and are dropped; a ``T``/``F`` without a side
    is kept with side ``N`` so the replay can ignore it exactly where hftbacktest does.
    """
    if not path.exists():
        raise FeedError(f"{path} does not exist")
    try:
        import databento  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - environment
        raise FeedError("`pip install databento` is needed to read a .dbn file") from exc
    store = databento.DBNStore.from_file(path)
    # The SDK types a record as a union of every schema; an MBO file holds MBO records only, and
    # each field read below is an ``MBOMsg`` field, so the union is not narrowed record by record.
    records: list[Any] = list(store)
    if instrument_id is None:
        counts: dict[int, int] = {}
        for r in records:
            counts[int(r.instrument_id)] = counts.get(int(r.instrument_id), 0) + 1
        if not counts:
            raise FeedError(f"{path} holds no records")
        instrument_id = max(counts, key=lambda k: counts[k])
    out: list[L3Event] = []
    for r in records:
        if int(r.instrument_id) != instrument_id:
            continue
        kind = _DBN_KIND.get(str(r.action))
        if kind is None:
            continue
        out.append(L3Event(
            ts=int(r.ts_recv), kind=kind, side=str(r.side), px=int(r.price),
            size=float(r.size), order_id=int(r.order_id),
            batch_end=bool(int(r.flags) & F_LAST),
        ))
    return out


def replay_databento(
    path: Path, dataset: str, config: ReplayConfig,
) -> tuple[list[RealEpisode], dict[str, Any]]:
    events = databento_events(path)
    replay = TruthReplay(dataset, config)
    violations = 0

    def check(v: Any) -> str | None:
        nonlocal violations
        if not replay.self_consistent(v):
            violations += 1
            return "inconsistent"
        return None

    episodes = replay.run(events, check)
    meta = {
        "venue": "CME Globex (DataBento GLBX.MDP3 MBO)",
        "records": len(events), **replay.stats.as_dict(),
        "self_consistency_violations": violations,
    }
    return episodes, meta


# --- NASDAQ / LOBSTER --------------------------------------------------------------------------


LOBSTER_EMPTY_BID = -9999999999
LOBSTER_EMPTY_ASK = 9999999999


@dataclass(frozen=True, slots=True)
class LobsterSource:
    """Where one LOBSTER sample file pair lives and what it is.

    ``sha256`` values are the Git-LFS object ids of the mirrors, recorded so a reader can check
    that the bytes scored are the bytes named; two independent mirrors carrying the AMZN message
    file at the identical size (11,248,750 bytes) is the cross-check that the sample is LOBSTER's
    own and not a derivative.
    """

    ticker: str
    date: str
    levels: int
    message_url: str
    orderbook_url: str
    message_sha256: str
    orderbook_sha256: str


_LFS = "https://media.githubusercontent.com/media"
LOBSTER_SOURCES: dict[str, LobsterSource] = {
    "AMZN": LobsterSource(
        "AMZN", "2012-06-21", 10,
        f"{_LFS}/kpetridis24/lobsim/HEAD/sample_data/AMZN_2012-06-21_34200000_57600000_message_10.csv",
        f"{_LFS}/kpetridis24/lobsim/HEAD/sample_data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
        "c85a75b51f3616a683f825c2e03f984535e9329a8be5ab12325130d32223a2c5",
        "decfe952b3fa06c9922dc8fba763a37a0fb032d4231767997e4abf2aa0d29b50",
    ),
    "MSFT": LobsterSource(
        "MSFT", "2012-06-21", 10,
        f"{_LFS}/Clydexy/Distributed-DeepLOB/HEAD/datasets/MSFT_2012-06-21_34200000_57600000_message_10.csv",
        f"{_LFS}/Clydexy/Distributed-DeepLOB/HEAD/datasets/MSFT_2012-06-21_34200000_57600000_orderbook_10.csv",
        "669e5c35b1eab0f7336fa9ed919002e65b08d2aabd96bb4ae88d647184e7a18d",
        "ec2f53467cf7bfb00b5702381bea7b4a061bb4c1730454df3679b272b3561c0e",
    ),
    "GOOG": LobsterSource(
        "GOOG", "2012-06-21", 5,
        f"{_LFS}/asarfa/MMviaRL/HEAD/data/GOOG_2012-06-21_34200000_57600000_message_5.csv",
        f"{_LFS}/asarfa/MMviaRL/HEAD/data/GOOG_2012-06-21_34200000_57600000_orderbook_5.csv",
        "95a70c44ed1ae0effc56f28e79d12a8f105b29bd21482ed78deff20167a5e8e0",
        "1a4691864742cc8cd033f22892d4b3e953c620c4b4ffc9487a7edc4d0073d945",
    ),
}


def fetch_cached(url: str, cache: Path, sha256: str) -> Path:
    """Download once into ``cache`` (gzip), verify the bytes against the recorded object id, and
    return the cached path. A hash mismatch is a hard error: scoring different bytes under the
    same name is exactly the defect this project's artefact discipline exists to prevent."""
    import hashlib

    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (Path(url).name + ".gz")
    if target.exists():
        return target
    digest = hashlib.sha256()
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as resp, gzip.open(tmp, "wb", 6) as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise FeedError(f"{url}: sha256 {digest.hexdigest()} != recorded {sha256}")
    tmp.replace(target)
    return target


def _lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="ascii") as fh:
        yield from fh


def _ns(date: str, seconds: str) -> int:
    """LOBSTER time (seconds after midnight, up to nanosecond decimals) as epoch nanoseconds."""
    base = int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()) * 1_000_000_000
    whole, _, frac = seconds.partition(".")
    return base + int(whole) * 1_000_000_000 + int((frac + "000000000")[:9])


def lobster_message_events(fields: list[str], ts: int) -> list[L3Event]:
    """One LOBSTER message as normalised events (without ``batch_end``, set by the caller)."""
    kind, oid, size, px, direction = (
        int(fields[1]), int(fields[2]), float(fields[3]), int(fields[4]), int(fields[5]),
    )
    side = "B" if direction == 1 else "A"
    aggressor = "A" if side == "B" else "B"
    if kind == 1:
        return [L3Event(ts, ADD, side, px, size, oid, False)]
    if kind == 2:
        return [L3Event(ts, REDUCE, side, px, size, oid, False)]
    if kind == 3:
        return [L3Event(ts, CANCEL, side, px, size, oid, False)]
    if kind == 4:
        return [
            L3Event(ts, TRADE, aggressor, px, size, 0, False),
            L3Event(ts, EXEC, side, px, size, oid, False),
        ]
    if kind == 5:
        # A hidden order executed: the tape prints it, the displayed book does not change.
        return [L3Event(ts, TRADE, aggressor, px, size, 0, False)]
    if kind == 7:
        return [L3Event(ts, HALT, "N", px, 0.0, 0, False)]
    return []   # 6: cross trade (auction) — no displayed-book effect


def _book_row(
    fields: list[str], levels: int,
) -> tuple[dict[int, float], dict[int, float], int, int]:
    """(asks, bids, deepest visible ask, deepest visible bid) from one orderbook row."""
    asks: dict[int, float] = {}
    bids: dict[int, float] = {}
    deep_ask, deep_bid = LOBSTER_EMPTY_ASK, LOBSTER_EMPTY_BID
    full_ask = full_bid = True
    for lvl in range(levels):
        a_px, a_sz, b_px, b_sz = (int(fields[4 * lvl + j]) for j in range(4))
        if a_px == LOBSTER_EMPTY_ASK:
            full_ask = False
        else:
            asks[a_px] = float(a_sz)
            deep_ask = a_px
        if b_px == LOBSTER_EMPTY_BID:
            full_bid = False
        else:
            bids[b_px] = float(b_sz)
            deep_bid = b_px
    # A side with fewer than `levels` occupied prices is fully visible.
    return asks, bids, (deep_ask if full_ask else LOBSTER_EMPTY_ASK), (
        deep_bid if full_bid else LOBSTER_EMPTY_BID)


def replay_lobster(
    message_path: Path, orderbook_path: Path, source: LobsterSource, config: ReplayConfig,
    *, dataset: str | None = None,
) -> tuple[list[RealEpisode], dict[str, Any]]:
    """Replay one LOBSTER day with the venue's own level totals as the reconciliation reference.

    Before each message the replay adopts the previous row's totals (the residual from orders that
    predate the file is re-derived); after it, every open synthetic order must reconcile —
    ``ahead + known-behind`` equal to the venue's new total at its price — or it is censored.
    A price that has slid below the deepest visible level is censored as ``window``: the file no
    longer says what is resting there.
    """
    replay = TruthReplay(dataset or f"LOBSTER {source.ticker} {source.date}", config)
    msgs = _lines(message_path)
    rows = _lines(orderbook_path)
    counts = {"messages": 0, "hidden_prints": 0, "halts": 0}
    state: dict[str, Any] = {"asks": {}, "bids": {}, "deep_ask": LOBSTER_EMPTY_ASK,
                             "deep_bid": LOBSTER_EMPTY_BID}

    def check(v: Any) -> str | None:
        side, px = v.ep.side, v.ep.px
        if side == "B":
            if px < state["deep_bid"]:
                return "window"
            auth = state["bids"].get(px, 0.0)
        else:
            if px > state["deep_ask"]:
                return "window"
            auth = state["asks"].get(px, 0.0)
        if abs(auth - (v.ahead + v.behind)) > 1e-6:
            return "inconsistent"
        return None

    pending: tuple[list[str], list[str]] | None = None
    for raw_msg, raw_row in zip(msgs, rows, strict=False):
        m = raw_msg.strip().split(",")
        r = raw_row.strip().split(",")
        if pending is None:
            pending = (m, r)
            continue
        _lobster_step(replay, pending, m[0], source, state, counts, check)
        pending = (m, r)
    if pending is not None:
        _lobster_step(replay, pending, None, source, state, counts, check)
    episodes = replay.finish()
    meta = {
        "venue": f"NASDAQ via LOBSTER ({source.ticker}, {source.date}, {source.levels} levels)",
        "message_sha256": source.message_sha256, "orderbook_sha256": source.orderbook_sha256,
        **counts, **replay.stats.as_dict(),
    }
    return episodes, meta


def _lobster_step(
    replay: TruthReplay, pending: tuple[list[str], list[str]], next_time: str | None,
    source: LobsterSource, state: dict[str, Any], counts: dict[str, int], check: Any,
) -> None:
    m, r = pending
    counts["messages"] += 1
    ts = _ns(source.date, m[0])
    events = lobster_message_events(m, ts)
    for ev in events:
        if ev.kind == TRADE and len(events) == 1:
            counts["hidden_prints"] += 1
        if ev.kind == HALT:
            counts["halts"] += 1
        replay.apply(ev)
    asks, bids, deep_ask, deep_bid = _book_row(r, source.levels)
    state.update(asks=asks, bids=bids, deep_ask=deep_ask, deep_bid=deep_bid)
    best_bid = max(bids) if bids else None
    best_ask = min(asks) if asks else None
    replay.best_override = (best_bid, best_ask)
    # Sync the displayed levels the message just produced, so the next join and the next depth
    # step start from the venue's numbers.
    for px, total in asks.items():
        replay.sync_level("A", px, total)
    for px, total in bids.items():
        replay.sync_level("B", px, total)
    if next_time is None or next_time != m[0]:
        replay.end_batch(ts, check)


def iter_text(url: str, timeout: int = 120) -> Iterable[str]:  # pragma: no cover - network
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        yield from io.TextIOWrapper(resp, encoding="utf-8")


__all__ = [
    "F_LAST",
    "LOBSTER_SOURCES",
    "FeedError",
    "LobsterSource",
    "databento_events",
    "fetch_cached",
    "lobster_message_events",
    "replay_databento",
    "replay_lobster",
]
