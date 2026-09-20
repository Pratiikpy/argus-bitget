"""The auto-resolver — mechanical, scheduled, and structurally unable to answer early.

A claim is only worth what its resolution is worth. If resolution can be run early, run twice, or
run by a human who has already seen the outcome, the register is a diary with hashes on it.

Three properties, each enforced rather than intended:

1. **It cannot resolve early.** A claim whose ``resolves_at`` has not passed returns PENDING without
   looking at a price at all. The check happens before the fetch, so there is no code path in which
   the resolver has seen the answer and then decided not to use it.
2. **It cannot see past the horizon.** The bar used is the last one at or before ``resolves_at``,
   never the newest available. A resolver that graded on today's price would be answering a
   different question than the one that was committed.
3. **It cannot silently fail.** A source that will not answer produces UNRESOLVABLE with the reason,
   never FALSE. Recording an ungradable claim as a miss would improve the record's apparent
   difficulty while corrupting every calibration figure computed from it.

Resolution is written as a **new line**, never an edit: the register is append-only, so a resolution
is a second entry that names the claim it resolves. Nothing in the file is ever rewritten, which is
what lets a stranger verify the chain from byte zero without trusting the process that wrote it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from argus.register.claims import (
    REGISTER_PATH,
    Claim,
    Predicate,
    Status,
    _read,
)

RESOLUTION_PATH = REGISTER_PATH.parent / "register_resolutions.jsonl"
"""Resolutions live beside the register, one line each, naming the claim hash they resolve."""


class ResolveError(RuntimeError):
    """Raised only for a broken call, never for a claim that cannot be graded."""


def _series(claim: Claim, *, now: datetime) -> list[tuple[datetime, Decimal]]:
    """The point-in-time series this claim named, truncated at its own horizon.

    Truncation happens here rather than at the comparison, so no later code can accidentally reach
    a bar the claim was not about. The source string is parsed rather than assumed: a claim that
    named ``bitget:1H:market`` is resolved against exactly that, and an unknown binding raises
    instead of falling back to a default that would silently grade the wrong series.
    """
    venue, interval, kind = claim.source.split(":")
    if venue != "bitget":
        raise ResolveError(f"no resolver for venue {venue!r}")
    from argus.market.history import CandleType, fetch_range

    try:
        candle_type = CandleType(kind)
    except ValueError as exc:
        raise ResolveError(f"unknown candle type {kind!r}") from exc

    registered = datetime.fromisoformat(claim.registered_at)
    span_days = max(2, int((now - registered).total_seconds() // 86400) + 2)
    bars = fetch_range(claim.subject, days=span_days, interval=interval, candle_type=candle_type)
    horizon = datetime.fromisoformat(claim.resolves_at)
    return [(b.ts, Decimal(str(b.close))) for b in bars if b.ts <= horizon]


def _price_at(series: Sequence[tuple[datetime, Decimal]], when: datetime) -> Decimal | None:
    """The last close at or before ``when``. ``None`` when the series does not reach it."""
    found: Decimal | None = None
    for stamp, close in series:
        if stamp <= when:
            found = close
        else:
            break
    return found


def resolve_one(claim: Claim, *, now: datetime | None = None) -> Claim:
    """Grade one claim, or return it unchanged as PENDING.

    Returns a **new** Claim; the input is never mutated, because the committed object is the thing
    the hash covers and an in-place update would break the chain it belongs to.
    """
    if claim.status is not Status.PENDING:
        return claim
    moment = now or datetime.now(UTC)
    horizon = datetime.fromisoformat(claim.resolves_at)

    # Before the fetch, deliberately: there is no path where the resolver has seen the outcome and
    # then declined to use it.
    if moment < horizon:
        return claim

    try:
        series = _series(claim, now=moment)
    except Exception as exc:
        return replace(
            claim, status=Status.UNRESOLVABLE, resolved_at=moment.isoformat(),
            observed=f"source unavailable: {type(exc).__name__}",
        )

    registered = datetime.fromisoformat(claim.registered_at)
    at_horizon = _price_at(series, horizon)
    at_registration = _price_at(series, registered)
    if at_horizon is None:
        return replace(
            claim, status=Status.UNRESOLVABLE, resolved_at=moment.isoformat(),
            observed="no bar at or before the resolution time",
        )

    threshold = Decimal(claim.threshold)
    if claim.predicate is Predicate.CLOSE_ABOVE:
        outcome, observed = at_horizon > threshold, at_horizon
    elif claim.predicate is Predicate.CLOSE_BELOW:
        outcome, observed = at_horizon < threshold, at_horizon
    else:
        if at_registration is None or at_registration <= 0:
            return replace(
                claim, status=Status.UNRESOLVABLE, resolved_at=moment.isoformat(),
                observed="no bar at or before registration to measure the move from",
            )
        move_bps = (at_horizon - at_registration) / at_registration * Decimal("10000")
        observed = move_bps
        if claim.predicate is Predicate.RETURN_OVER_ABOVE_BPS:
            outcome = move_bps > threshold
        elif claim.predicate is Predicate.RETURN_OVER_BELOW_BPS:
            outcome = move_bps < threshold
        elif claim.predicate is Predicate.ABS_MOVE_ABOVE_BPS:
            outcome = abs(move_bps) > threshold
        else:  # pragma: no cover - the enum is exhaustive
            raise ResolveError(f"no rule for predicate {claim.predicate}")

    return replace(
        claim,
        status=Status.TRUE if outcome else Status.FALSE,
        resolved_at=moment.isoformat(),
        observed=str(observed),
    )


def resolve_all(
    *, register_path: Path = REGISTER_PATH, resolution_path: Path = RESOLUTION_PATH,
    now: datetime | None = None,
) -> list[Claim]:
    """Resolve every claim whose horizon has passed and that is not already resolved.

    Idempotent: a claim already carrying a resolution is skipped, and re-running writes nothing.
    That matters because this runs on a schedule, and a resolver that appended a second verdict on
    every run would let the record say two different things about one claim.
    """
    moment = now or datetime.now(UTC)
    claims = _read(register_path)
    already = _resolved_hashes(resolution_path)

    fresh: list[Claim] = []
    for claim in claims:
        if claim.entry_hash in already:
            continue
        graded = resolve_one(claim, now=moment)
        if graded.status is not Status.PENDING:
            fresh.append(graded)

    if fresh:
        resolution_path.parent.mkdir(parents=True, exist_ok=True)
        with resolution_path.open("a", encoding="utf-8") as handle:
            for claim in fresh:
                blob = asdict(claim)
                blob["predicate"] = str(claim.predicate)
                blob["status"] = str(claim.status)
                handle.write(json.dumps(blob, sort_keys=True) + "\n")
    return fresh


def _resolved_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(str(json.loads(line).get("entry_hash", "")))
    return out


def resolutions(path: Path = RESOLUTION_PATH) -> list[Claim]:
    """Every resolution written so far, oldest first."""
    if not path.exists():
        return []
    out: list[Claim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        blob = json.loads(line)
        blob["predicate"] = Predicate(blob["predicate"])
        blob["status"] = Status(blob["status"])
        out.append(Claim(**blob))
    return out


def scoreboard(
    *, register_path: Path = REGISTER_PATH, resolution_path: Path = RESOLUTION_PATH,
) -> dict[str, Any]:
    """The public record: counts, Brier score, and every miss named.

    Brier rather than hit rate, for the reason the whole design turns on: hit rate is what every
    leaderboard shows and what every claimant games, and it is improved simply by only making easy
    claims. Brier cannot be improved that way — it charges for confidence, so the same accuracy
    stated loudly scores **worse** than stated honestly, and a claimant who inflates confidence to
    look decisive pays for it immediately.

    It is not a calibration-only measure, and an early version of this docstring said it was: Brier
    also rewards *resolution*, so being right more often always helps. A coin that knows it is a
    coin scores honestly and still loses to someone who actually knows something. Both halves are
    pinned by test, because a future reader "fixing" this into rewarding timidity would quietly
    invert the incentive the register exists to create.
    """
    graded = [c for c in resolutions(resolution_path) if c.status in (Status.TRUE, Status.FALSE)]
    pending = [
        c for c in _read(register_path)
        if c.entry_hash not in _resolved_hashes(resolution_path)
    ]
    by_claimant: dict[str, list[Claim]] = {}
    for claim in graded:
        by_claimant.setdefault(claim.claimant, []).append(claim)

    rows: list[dict[str, Any]] = []
    for claimant, claims in sorted(by_claimant.items()):
        brier = sum(
            (c.confidence - (1.0 if c.status is Status.TRUE else 0.0)) ** 2 for c in claims
        ) / len(claims)
        hits = sum(1 for c in claims if c.status is Status.TRUE)
        rows.append({
            "claimant": claimant,
            "resolved": len(claims),
            "correct": hits,
            "hit_rate": round(hits / len(claims), 4),
            "brier": round(brier, 5),
            "mean_confidence": round(sum(c.confidence for c in claims) / len(claims), 4),
        })
    rows.sort(key=lambda r: float(str(r["brier"])))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "registered": len(_read(register_path)),
        "resolved": len(graded),
        "pending": len(pending),
        "unresolvable": sum(
            1 for c in resolutions(resolution_path) if c.status is Status.UNRESOLVABLE
        ),
        "claimants": rows,
        # The Wall: every claim we got wrong, worst first. Published because a record that only
        # shows its hits is an advertisement.
        "wall": [
            {
                "claim": c.render(), "claimant": c.claimant, "confidence": c.confidence,
                # Both: `observed` is the exact graded value (the audit trail), `observed_shown`
                # is what a reader can check against the threshold. See `shown()`.
                "observed": c.observed, "observed_shown": shown(c), "resolved_at": c.resolved_at,
                "hash": c.entry_hash,
            }
            for c in sorted(
                (c for c in graded if c.status is Status.FALSE),
                key=lambda c: -c.confidence,
            )
        ],
    }


def shown(claim: Claim) -> str:
    """How an observed value is *printed*. `claim.observed` keeps the exact graded value.

    The Wall is read by strangers, and it was publishing two things a stranger cannot check.
    Every row printed an unrounded ``str(Decimal)`` — up to 28 decimal places of a basis-point
    figure sitting beside a threshold quoted to two. And under ``ABS_MOVE_ABOVE_BPS`` the stored
    value is *signed* (line 130) while the verdict is graded on ``abs(move_bps)`` (line 136), so
    seven of fourteen such rows published a negative observed against an ``|x|`` test: a reader
    comparing ``-5.58`` to a threshold of ``50.23`` cannot reproduce the FALSE without knowing to
    take the absolute value first.

    Fixed at display time, not at write time, deliberately: the stored string is the audit trail —
    it is the number the verdict was actually computed from — and rounding it at the source would
    silently rewrite claims already resolved and anchored.
    """
    if claim.observed is None:
        return "—"  # PENDING: the horizon has not passed, so nothing has been observed
    try:
        value = Decimal(claim.observed)
    except (InvalidOperation, ValueError):
        return claim.observed  # an UNRESOLVABLE reason, not a number
    if claim.predicate is Predicate.ABS_MOVE_ABOVE_BPS:
        return f"|{value:.2f}| = {abs(value):.2f}"
    return f"{value:.2f}"


def main() -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="resolve every claim whose horizon has passed")
    parser.add_argument("--scoreboard", action="store_true", help="print the record and exit")
    args = parser.parse_args()

    if not args.scoreboard:
        fresh = resolve_all()
        print(f"{len(fresh)} claim(s) resolved this run")
        for claim in fresh:
            print("  " + claim.render() + f"  observed {shown(claim)}")

    board = scoreboard()
    print()
    print(
        f"REGISTER — {board['registered']} claim(s), {board['resolved']} resolved, "
        f"{board['pending']} pending, {board['unresolvable']} unresolvable"
    )
    for row in board["claimants"]:
        print(
            f"  {row['claimant']:20} {row['resolved']:4d} resolved  "
            f"Brier {row['brier']:.4f}  hit rate {row['hit_rate']:.0%}"
        )
    if board["wall"]:
        print(f"\n  THE WALL — {len(board['wall'])} claim(s) we got wrong, most confident first:")
        for miss in board["wall"][:10]:
            print(f"    {miss['claim']}  observed {miss['observed_shown']}")

    out = REGISTER_PATH.parent / "register_scoreboard.json"
    out.write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "RESOLUTION_PATH",
    "ResolveError",
    "resolutions",
    "resolve_all",
    "resolve_one",
    "scoreboard",
]
