"""The queue model's thirteenth condition, against a real market-by-order feed.

`eval/standing.py`'s own audit is exact about what is missing here: both this project and
`nkaz001/hftbacktest` score a probability model against a ground truth they hold, and neither
validates against live fills. Theirs is real CME market-by-order data (DataBento's
``glbx-mdp3-*.mbo.dbn.zst``); ours, until this module, was a simulator whose attribution rule we
stated and swept ourselves.

**No CME data has been bought.** DataBento (the vendor `eval/standing.py` already names) offers
$125 in free historical-data credit to new signups, but *creating the account is the one step this
project will not take on the owner's behalf* (this codebase's own standing rule: account creation is
his to do, not ours, free or not). What changed this module from untested to tested: DataBento's
own public GitHub SDK repo ships tiny CI fixtures for every dataset it supports, and
`nautechsystems/nautilus_trader` (LGPL-3.0, its own DataBento adapter's public test fixtures)
ships a real, substantial one — a full 2023-12-25 GLBX.MDP3 MBO session for ESH4 (E-mini S&P 500,
Mar-2024 contract), 68,756 real events, no account needed. Read via its public raw GitHub URL, run
locally, never committed into this repository (data, not code — cited by source and commit rather
than vendored, matching this project's own licence discipline for anything not MIT/BSD/Apache).

**What running it actually found, honestly, including where it disagrees with itself.** On the
Dec-25 session (1,366 real episodes, `join_every=20`): `LogProbability2` wins, queue_error 0.047
against the shipped default `PowerProbability n=1`'s 0.062 — a paired 95% CI entirely above zero
(beats the shipped default significantly) and also beats the best naive ablation significantly.
Real signal: on a real, if thin (Christmas Day), session, hftbacktest's probability-weighted
approach genuinely outperforms "ignore the queue and guess". **A second real file complicates
that story rather than confirming it.** A separate nautilus_trader fixture — a 2-second GLBX.MDP3
burst on 2024-05-08, NOT a holiday, 13,795 events but only 10 non-Add events total — produces the
OPPOSITE ranking: the naive "every cancellation is behind you" ablation wins outright, beating
every real probability function. The likely read: a 2-second window has almost no genuine queue
churn to model, so the simplest assumption wins by having nothing to be wrong about — but that is
a read, not confirmed, because a 2-second file is too data-poor to support much beyond "not the
same answer as the full day". **Neither file alone, nor both together, is the deliberately-chosen,
statistically adequate real-trading-day sample this capability's own bar calls for** — one is a
holiday session, the other a two-second burst nobody chose for representativeness. `standing.py`
reports both findings and does not promote the capability on the strength of either.

A third file from the same source (`esh4-glbx-mdp3-20231224.mbo.dbn.zst`) turned out to be a
single-instant full-book snapshot (8,725 Adds at one identical microsecond, no subsequent
activity) rather than a real session slice — checked, found degenerate, and excluded rather than
scored, because scoring it would have reported "every model ties at zero error" as if it were a
finding instead of an artefact of a snapshot with nothing to disagree about.

**The adapter, precisely.** `execution/queue.py::L3FIFOQueue` is already an order-by-order FIFO
book — `add`/`cancel`/`on_trade`/`reduce` — built to be the ground truth `eval/queueproof.py`'s
synthetic `simulate()` also produces, just estimated rather than stated. A real MBO feed replayed
through the same class produces the same ground truth from real exchange events instead of a
generative rule: :func:`episodes_from_mbo` walks the feed, periodically joins a synthetic order at
the back of whatever is resting, and records what real cancels and real trades did to it —
`Observation`/`Episode` objects in the exact shape `eval/queueproof.py::score` already consumes,
so nothing downstream of episode construction needs to change to score real ground truth. Every
file is filtered to its own :func:`dominant_instrument` first — a `.dbn` file can carry more than
one instrument even when its filename names only one, and replaying two books as if they were one
would corrupt every `ahead_of` query with a stranger's resting quantity.

**Two things stated as open rather than resolved.** (1) A real MBO Modify (Change) message with a
*smaller* size is modelled as priority-preserving (`L3FIFOQueue.reduce`) — the standard convention
across L3 feed protocols generally, but not independently confirmed against CME's own primary MDP3
specification, which sits behind a client-site login this project could not reach. A modify that
increases size or changes price is treated as cancel-then-add, losing priority, which every source
read agrees on. (2) The Clear (``R``) action resets the book and closes every open episode as
unfilled at that point — read from the schema's own `Action` enum (`CLEAR` exists as a value) but
neither real file read so far ever contained one, so this path is exercised only by the hand-built
tests, not by real data yet.

    python -m argus.eval.mbo_queueproof path/to/file.dbn.zst
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.queueproof import OUR_QTY, Episode, Observation, score
from argus.execution.queue import L3FIFOQueue, L3Order

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "mbo_queue_proof.json"

PRICE_SCALE = Decimal("1e-9")
"""DBN's fixed-price convention: `price` is an integer in units of 1e-9. Confirmed directly
against the real, installed `databento_dbn.MBOMsg` (constructing `price=100_000_000_000` reports
`pretty_price=100`), not assumed from memory of the format."""

JOIN_EVERY = 50
"""How often a new synthetic order joins the back of whatever price an Add event lands at.
50 real adds per synthetic episode start keeps episode count and book-realism in the same rough
range `eval/queueproof.py::simulate` already uses (a few thousand episodes from a session's worth
of adds), without so many concurrent open episodes that the O(active) scan per event dominates."""

MIN_OBSERVATIONS = 3
MAX_OBSERVATIONS = 40
"""Matches `eval/queueproof.py::simulate`'s own `rng.randint(8, 40)` upper bound and roughly its
lower one, so real and synthetic episodes are comparable in length rather than one dwarfing the
other in the scored mean."""


class MboQueueProofError(RuntimeError):
    """The real-data path cannot proceed honestly. Raised rather than reporting a fabricated 0."""


@dataclass(frozen=True, slots=True)
class MboEvent:
    """One market-by-order record, reduced to what the replay needs.

    Deliberately decoupled from `databento_dbn.MBOMsg`'s own object shape: the SDK's compiled
    type is what `read_dbn` reads, converted here immediately, so nothing below this line depends
    on which vendor's SDK produced the feed — the same reason `stumpy_run` and `riskfolio_weights`
    elsewhere in this project convert to plain Python at the boundary rather than threading a
    vendor type through the whole pipeline.
    """

    ts_recv: int
    action: str
    """One of "A" (add), "C" (cancel), "M" (modify), "T" (trade), "F" (fill), "R" (clear) —
    `databento_dbn.Action`'s own single-letter encoding, read directly off the real enum rather
    than guessed."""
    side: str
    """"B" (bid), "A" (ask), or "N" (none) — `databento_dbn.Side`'s own encoding."""
    order_id: str
    instrument_id: int
    """DataBento's per-dataset numeric instrument key. A single `.dbn` file CAN carry more than
    one instrument (a definition or options-chain pull, for instance) even when its filename
    names only one — replaying two instruments' order flow through one `L3FIFOQueue` would treat
    unrelated price levels as if they belonged to the same book. `episodes_from_mbo` filters to
    one instrument rather than trusting the filename."""
    price: Decimal
    size: Decimal


def _from_mbo_msg(msg: Any) -> MboEvent:
    return MboEvent(
        ts_recv=int(msg.ts_recv), action=str(msg.action), side=str(msg.side),
        order_id=str(msg.order_id), instrument_id=int(msg.instrument_id),
        price=Decimal(int(msg.price)) * PRICE_SCALE, size=Decimal(int(msg.size)),
    )


def read_dbn(path: Path) -> list[MboEvent]:
    """Real GLBX.MDP3 MBO records for one file, read via the real `databento` SDK and reduced to
    :class:`MboEvent`. Sorted by `ts_recv` because DBN files are ordered by capture time, which
    for a single-venue file is receipt order — the order the book actually changed in.

    The missing-file check comes first: a path that does not exist is the more basic fault, and
    reporting a missing optional package instead sends the caller to install something that would
    not have helped."""
    if not path.exists():
        raise MboQueueProofError(f"{path} does not exist")
    try:
        # Optional by design (see the error below); not installed where the console runs.
        import databento  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise MboQueueProofError(
            "the `databento` package is needed to read a .dbn/.dbn.zst file "
            "(`pip install databento`); it is a dev-only dependency, not part of the deployed "
            "console, matching how this project already keeps scikit-learn and riskfolio out of "
            "the shipped bundle"
        ) from exc
    store = databento.DBNStore.from_file(path)
    events = [
        _from_mbo_msg(record) for record in store
        if str(getattr(record, "side", "N")) in ("A", "B")
    ]
    events.sort(key=lambda e: e.ts_recv)
    return events


def dominant_instrument(events: Sequence[MboEvent]) -> int:
    """The instrument_id with the most events -- the one a single-instrument file actually is,
    confirmed by counting rather than assumed from a filename."""
    counts: dict[int, int] = {}
    for event in events:
        counts[event.instrument_id] = counts.get(event.instrument_id, 0) + 1
    if not counts:
        raise MboQueueProofError("no events to determine an instrument from")
    return max(counts, key=lambda iid: counts[iid])


@dataclass(slots=True)
class _OpenEpisode:
    entry_level: float
    observations: list[Observation] = field(default_factory=list)
    filled: float = 0.0


def episodes_from_mbo(
    events: Sequence[MboEvent], *,
    instrument_id: int | None = None,
    join_every: int = JOIN_EVERY, min_observations: int = MIN_OBSERVATIONS,
    max_observations: int = MAX_OBSERVATIONS,
) -> list[Episode]:
    """Real order-book events, replayed through `L3FIFOQueue`, into `Episode` objects whose
    `true_filled`/`true_ahead` come from real cancels and real trades rather than a stated rule.

    One synthetic order joins the back of the book every `join_every`-th real Add. From that
    point every event at that price produces an `Observation` for every synthetic order still
    resting there, with `true_ahead` read from `L3FIFOQueue.ahead_of` — genuinely known, not
    estimated, the same property that makes the synthetic simulator's ground truth trustworthy,
    here backed by an exchange's real order flow instead of a chosen generative rule.

    An episode closes when its order fills, when it reaches `max_observations`, or when a Clear
    event resets the book; episodes still open at the end of the file are kept only if they
    reached `min_observations` — long enough to be informative rather than one lonely event.

    `instrument_id` defaults to :func:`dominant_instrument` — the file's own busiest instrument,
    confirmed by counting, not assumed from a filename that could be wrong or the file could
    contain more than the one instrument it is named for.
    """
    if instrument_id is None:
        instrument_id = dominant_instrument(events)
    events = [e for e in events if e.instrument_id == instrument_id]
    book = L3FIFOQueue()
    active: dict[str, _OpenEpisode] = {}
    resting_at: dict[str, Decimal] = {}
    finished: list[Episode] = []
    add_count = 0
    next_id = 0

    def _close(order_id: str) -> None:
        state = active.pop(order_id)
        resting_at.pop(order_id, None)
        finished.append(Episode(
            entry_level=state.entry_level, observations=tuple(state.observations),
            true_filled=state.filled,
        ))

    def _observe(
        price: Decimal, traded: float, *, filled_now: frozenset[str] = frozenset(),
    ) -> None:
        """Append one `Observation` to every synthetic order resting at `price`.

        `filled_now` names orders `on_trade` has ALREADY removed from the book this event because
        they were consumed entirely -- `ahead_of` cannot find them any more (there is nothing
        left to find), so `true_ahead=0.0` is supplied directly rather than queried. Without this
        a full fill would leave its episode silently open with no closing observation, because
        the generic lookup below has nothing left in the book to look up.
        """
        for order_id, level_price in list(resting_at.items()):
            if level_price != price:
                continue
            state = active[order_id]
            if order_id in filled_now:
                ahead = Decimal(0)
                visible_level = book.depth_at(price)
            else:
                queried = book.ahead_of(price, order_id)
                if queried is None:
                    continue
                ahead = queried
                our_share = Decimal(0) if state.filled >= float(OUR_QTY) else OUR_QTY
                visible_level = max(book.depth_at(price) - our_share, Decimal(0))
            state.observations.append(Observation(
                traded=traded, new_level=float(visible_level), true_ahead=float(ahead),
            ))
            if state.filled >= float(OUR_QTY) or len(state.observations) >= max_observations:
                _close(order_id)

    for event in events:
        price = event.price
        if event.action == "A":
            book.add(price, L3Order(order_id=event.order_id, quantity=event.size))
            add_count += 1
            if add_count % join_every == 0:
                order_id = f"OURS-{next_id}"
                next_id += 1
                book.add(price, L3Order(order_id=order_id, quantity=OUR_QTY, is_ours=True))
                resting_at[order_id] = price
                # `ahead_of` after joining, not `depth_at` -- the quantity resting AHEAD of us,
                # excluding our own order, matching `eval/queueproof.py::simulate`'s own
                # `entry_level = sum(ahead)` convention exactly.
                ahead = book.ahead_of(price, order_id) or Decimal(0)
                active[order_id] = _OpenEpisode(entry_level=float(ahead))
            _observe(price, traded=0.0)
        elif event.action == "C":
            book.cancel(price, event.order_id)
            _observe(price, traded=0.0)
        elif event.action == "M":
            # A pure decrease preserves priority (`reduce`); anything else — a size increase, or
            # a modify this file encodes with an unchanged/larger size — is a priority-losing
            # change on a real venue and is modelled as cancel-then-add, same as a new order.
            if not book.reduce(price, event.order_id, event.size):
                book.cancel(price, event.order_id)
                book.add(price, L3Order(order_id=event.order_id, quantity=event.size))
            _observe(price, traded=0.0)
        elif event.action in ("T", "F"):
            fills = book.on_trade(price, event.size)
            fully_filled: set[str] = set()
            for order_id, qty in fills:
                if order_id in active:
                    active[order_id].filled += float(qty)
                    if active[order_id].filled >= float(OUR_QTY):
                        fully_filled.add(order_id)
            _observe(price, traded=float(event.size), filled_now=frozenset(fully_filled))
        elif event.action == "R":
            for order_id in list(active):
                _close(order_id)
            book = L3FIFOQueue()

    for state in active.values():
        if len(state.observations) >= min_observations:
            finished.append(Episode(
                entry_level=state.entry_level, observations=tuple(state.observations),
                true_filled=state.filled,
            ))
    return finished


def run(path: Path, **episode_kwargs: Any) -> dict[str, Any]:
    """Real MBO ground truth, scored the same way `eval/queueproof.py::run` scores the synthetic
    kind — the same `score()` function, so a reader comparing the two reports is comparing the
    scoring, not two different measurements of the same word."""
    events = read_dbn(path)
    if not events:
        raise MboQueueProofError(f"{path} contains no bid/ask MBO records")
    instrument_id = episode_kwargs.pop("instrument_id", None) or dominant_instrument(events)
    same_instrument = [e for e in events if e.instrument_id == instrument_id]
    episodes = episodes_from_mbo(events, instrument_id=instrument_id, **episode_kwargs)
    if not episodes:
        raise MboQueueProofError(
            f"{path} produced zero scoreable episodes for instrument {instrument_id} — "
            f"join_every={episode_kwargs.get('join_every', JOIN_EVERY)} may be too sparse for "
            "this file's depth, or the file is shorter than min_observations needs"
        )
    scored = score(episodes)
    from argus.eval.queueproof import paired_interval

    by_name = {s.model: s for s in scored}
    best = scored[0]
    significance: dict[str, Any] = {}
    # paired_interval returns (mean, lo, hi) -- its own real signature, unpacked in that order
    # rather than the more readable (lo, mean, hi) a first draft assumed and got wrong, caught by
    # `TestTheRealArtefact`'s own ci_lo <= mean <= ci_hi sanity check failing on real data.
    shipped = by_name.get("PowerProbability n=1")
    if shipped is not None and shipped.model != best.model:
        mean, lo, hi = paired_interval(best, shipped)
        significance["best_vs_shipped_default"] = {
            "best": best.model, "shipped_default": shipped.model,
            "ci_lo": round(lo, 5), "mean": round(mean, 5), "ci_hi": round(hi, 5),
            "significant": bool(hi < 0 or lo > 0),
        }
    best_ablation = min(
        (s for s in scored if s.is_ablation), key=lambda s: s.queue_error, default=None,
    )
    if best_ablation is not None and best_ablation.model != best.model:
        mean, lo, hi = paired_interval(best, best_ablation)
        significance["best_vs_best_ablation"] = {
            "best": best.model, "best_ablation": best_ablation.model,
            "ci_lo": round(lo, 5), "mean": round(mean, 5), "ci_hi": round(hi, 5),
            "significant": bool(hi < 0 or lo > 0),
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file": str(path),
        "instrument_id": instrument_id,
        "raw_events": len(events),
        "events_for_instrument": len(same_instrument),
        "episodes": len(episodes),
        "results": [s.as_dict() for s in scored],
        "significance": significance,
        "scope_statement": (
            "Real GLBX.MDP3 market-by-order events, replayed through the SAME L3FIFOQueue class "
            "this project's own synthetic ground truth (eval/queueproof.py) is built on, scored "
            "by the SAME score() function. NOT CLAIMED: that Modify handling matches CME's own "
            "primary spec exactly (see this module's docstring) — the size-decrease-preserves-"
            "priority rule is the standard L3 convention, not independently confirmed against "
            "CME's own documentation. NOT CLAIMED: that this file's session is representative of "
            "every session — one file is one sample, and different real samples read this "
            "session has disagreed on which specific model wins (see this module's own docstring "
            "for the cross-file finding)."
        ),
    }


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="score the queue model against a real MBO file")
    parser.add_argument("path", type=Path)
    parser.add_argument("--join-every", type=int, default=JOIN_EVERY)
    args = parser.parse_args()

    report = run(args.path, join_every=args.join_every)
    print(f"{report['raw_events']} events -> {report['episodes']} episodes")
    for row in report["results"]:
        print(f"  {row['model']:38} queue_error={row['queue_error']:.5f}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "JOIN_EVERY",
    "MAX_OBSERVATIONS",
    "MIN_OBSERVATIONS",
    "PRICE_SCALE",
    "REPORT_PATH",
    "MboEvent",
    "MboQueueProofError",
    "dominant_instrument",
    "episodes_from_mbo",
    "main",
    "read_dbn",
    "run",
]
