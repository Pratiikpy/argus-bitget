"""What the crowd on X and Reddit is saying about each name — collected here, served anywhere.

**Why a snapshot.** The console's sentiment answer read funding, the fear & greed index and news
headlines, and nothing a trader actually posts. X and Reddit are reachable from this machine through
the `agent-reach` router's command-line backends (`twitter-cli`, `rdt-cli`; `agent-reach doctor`
reports both ``ok``) — but those ride a logged-in session, so the hosted console can never run
them. So they run here, on a schedule, and the result is published as a dated artefact the console
reads, the same way it reads every other slow sweep. The answer states the snapshot's time; it is
never passed off as live.

**What is measured, not what is said.** Posts are grouped into stories by near-duplicate text with
the desk's own detector (`agents/novelty.cluster`), with each post's *author* as its source, so five
accounts posting one line are one story from five sources — and three or more distinct accounts
inside two hours is flagged as coordinated, as it is everywhere else in ARGUS. The pulse reports
volume, distinct stories, how many look coordinated, and the most-carried stories with their
authors. It never scores the text as bullish or bearish: the desk measured that a sentiment model
grows louder with every repeat of the same line (`data/sentiment_comparison.json`), which is
exactly what a coordinated campaign exploits.

**Taken from `agent-reach` (MIT, Panniantong/agent-reach).** Its routing choice — which backend
answers for which platform, and that Reddit has no working anonymous path (its
`channels/reddit.py` docstring, live-verified) — is followed; the CLIs are called through ARGUS's
existing `market.evidence.TwitterSource` / `RedditSource`, which already fix the Windows
encoding bugs those CLIs have.

    python -m argus.market.social_pulse            # all traded names, writes data/social_pulse.json
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.agents.novelty import COORDINATION_WINDOW, MIN_COORDINATED_SOURCES, cluster

DATA = Path(__file__).resolve().parents[3] / "data"
PULSE_PATH = DATA / "social_pulse.json"
PER_CHANNEL = 30
"""Posts asked of each platform per name. Enough for a story to repeat, few enough that a sweep
of twelve names finishes in minutes."""
LOOKBACK = timedelta(hours=48)
STALE_AFTER = timedelta(hours=12)
"""An older snapshot is still shown, with its age, but called out as stale."""


@dataclass(frozen=True, slots=True)
class Post:
    """One post in the shape `agents.novelty.cluster` reads: ``source`` is the author."""

    id: str
    claim: str
    source: str
    available_at: datetime
    channel: str


Fetch = Callable[[str, datetime], tuple[list[Post], list[str]]]


def _from_evidence(items: Sequence[Any], channel: str) -> list[Post]:
    posts = []
    for item in items:
        attributes = getattr(item, "attributes", {}) or {}
        author = str(attributes.get("author") or attributes.get("subreddit") or "").strip()
        if not author:
            continue
        prefix = "@" if channel == "x" else "u/"
        posts.append(Post(id=str(item.id), claim=str(item.claim), source=f"{prefix}{author}",
                          available_at=item.available_at, channel=channel))
    return posts


def fetch_live(symbol: str, now: datetime) -> tuple[list[Post], list[str]]:
    """X and Reddit posts naming ``symbol`` from the last :data:`LOOKBACK`, with a status per
    platform. A platform that cannot answer is reported, never silently empty."""
    from argus.market.evidence import RedditSource, TwitterSource

    tweets, x_status = TwitterSource(timeout=45).evidence(symbol, as_of=now, max_tweets=PER_CHANNEL)
    threads, r_status = RedditSource(timeout=45).evidence(symbol, as_of=now,
                                                         max_posts=PER_CHANNEL, newest=True)
    since = now - LOOKBACK
    posts = [p for p in (*_from_evidence(tweets, "x"), *_from_evidence(threads, "reddit"))
             if p.available_at >= since]
    return posts, [*x_status, *r_status]


def pulse(symbol: str, posts: Sequence[Post], status: Sequence[str],
          now: datetime) -> dict[str, Any]:
    """The crowd read for one name."""
    report = cluster(posts)
    stories = []
    for story in report.clusters[:5]:
        stories.append({
            "text": story.representative[:220],
            "posts": story.size,
            "accounts": story.distinct_sources,
            "coordinated": story.coordinated,
            "authors": sorted(set(story.sources))[:6],
            "first_seen": story.first_seen.isoformat(),
        })
    by_channel: dict[str, int] = {}
    for post in posts:
        by_channel[post.channel] = by_channel.get(post.channel, 0) + 1
    return {
        "symbol": symbol,
        "posts": len(posts),
        "accounts": len({p.source for p in posts}),
        "by_channel": by_channel,
        "distinct_stories": report.distinct_stories,
        "coordinated_stories": len(report.coordinated),
        "coordinated_posts": sum(c.size for c in report.coordinated),
        "top_stories": stories,
        "status": list(status),
        "checked_at": now.isoformat(),
    }


def sweep(symbols: Sequence[str], *, fetch: Fetch = fetch_live,
          now: datetime | None = None) -> dict[str, Any]:
    stamp = now or datetime.now(UTC)
    rows = []
    for symbol in symbols:
        posts, status = fetch(symbol, stamp)
        rows.append(pulse(symbol, posts, status, stamp))
    return {
        "generated_at": stamp.isoformat(),
        "lookback_hours": LOOKBACK.total_seconds() / 3600,
        "coordination_rule": (f"{MIN_COORDINATED_SOURCES} or more distinct accounts inside "
                              f"{COORDINATION_WINDOW.total_seconds() / 3600:g}h"),
        "sources": "X via twitter-cli, Reddit via rdt-cli (the agent-reach router's backends)",
        "symbols": rows,
    }


def load(path: Path = PULSE_PATH) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def lines_for(symbol: str, snapshot: dict[str, Any] | None,
              now: datetime | None = None) -> list[str]:
    """The console's sentences for one name, stating the snapshot's age. Empty when the name was
    not swept, so the answer omits the crowd rather than inventing a quiet one."""
    if not snapshot:
        return []
    row = next((r for r in snapshot.get("symbols", []) if r.get("symbol") == symbol), None)
    if row is None:
        return []
    ticker = symbol.removesuffix("USDT")
    try:
        taken = datetime.fromisoformat(str(snapshot.get("generated_at")))
    except ValueError:
        return []
    age = (now or datetime.now(UTC)) - taken
    hours = age.total_seconds() / 3600
    when = f"{taken:%d %b %H:%M} UTC, {hours:.0f}h ago" + (
        " — stale" if age > STALE_AFTER else "")
    if not row.get("posts"):
        failed = [s for s in row.get("status", []) if "unavailable" in s or "ok=false" in s]
        if failed:
            return [f"Crowd on X and Reddit ({when}): not collected — {'; '.join(failed)}."]
        return [f"Crowd on X and Reddit ({when}): nothing named {ticker} in the last 48h."]
    channels = ", ".join(f"{n} on {'X' if c == 'x' else 'Reddit'}"
                         for c, n in sorted(row.get("by_channel", {}).items()))
    out = [f"Crowd on X and Reddit ({when}): {row['posts']} posts from {row['accounts']} accounts "
           f"({channels}) carry {row['distinct_stories']} distinct stories."]
    if row.get("coordinated_stories"):
        top = next((s for s in row.get("top_stories", []) if s.get("coordinated")), None)
        example = (f" — e.g. \"{top['text'][:110]}\" from {top['accounts']} accounts"
                   if top else "")
        out.append(f"{row['coordinated_stories']} of them look coordinated ("
                   f"{row['coordinated_posts']} posts; {snapshot.get('coordination_rule', '')})"
                   f"{example}: repetition, not new information.")
    else:
        out.append("None of them looks coordinated, so the volume is independent voices, not "
                   "one line repeated.")
    # "Most carried" means carried by several people: one account posting a recap four times is
    # one voice, and was reported as the lead story on the first run (2026-09-25).
    lead = next((s for s in row.get("top_stories", [])
                 if not s.get("coordinated") and s.get("accounts", 0) >= 2), None)
    if lead:
        out.append(f"Most-shared story: \"{lead['text'][:140]}\" ({lead['posts']} posts from "
                   f"{lead['accounts']} accounts).")
    else:
        out.append("No story is carried by more than one account — the talk is scattered, not "
                   "converging on a single narrative.")
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    import argparse

    from argus.lui.question import TRADED_SYMBOLS

    parser = argparse.ArgumentParser(description="Sweep X and Reddit for the traded names.")
    parser.add_argument("--out", type=Path, default=PULSE_PATH)
    args = parser.parse_args(argv)
    snapshot = sweep(TRADED_SYMBOLS)
    args.out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in snapshot["symbols"]:
        print(f"{row['symbol']:10} posts {row['posts']:3} accounts {row['accounts']:3} stories "
              f"{row['distinct_stories']:3} coordinated {row['coordinated_stories']}  "
              f"{'; '.join(row['status'])}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PULSE_PATH", "Post", "fetch_live", "lines_for", "load", "pulse", "sweep"]
