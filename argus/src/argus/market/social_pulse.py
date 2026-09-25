"""What the crowd on X and Reddit is saying about each name — collected here, served anywhere.

**Why a snapshot.** The console's sentiment answer read funding, the fear & greed index and news
headlines, and nothing a trader actually posts. X and Reddit are reachable from this machine through
the `agent-reach` router's command-line backends (`twitter-cli`, `rdt-cli`; `agent-reach doctor`
reports both ``ok``) — but those ride a logged-in session, so the hosted console can never run
them. So they run here, on a schedule, and the result is published as a dated artefact the console
reads, the same way it reads every other slow sweep. The answer states the snapshot's time; it is
never passed off as live.

**What is measured, not what is said.** Posts are grouped into stories by near-duplicate text with
the desk's own detector (`truth/novelty.cluster`), with each post's *author* as its source, so five
accounts posting one line are one story from five sources — and three or more distinct accounts
inside two hours is flagged as coordinated, as it is everywhere else in ARGUS. The pulse reports
volume, distinct stories, how many look coordinated, and the most-carried stories with their
authors.

**Tone, counted by voices rather than by posts (2026-09-26).** Until this date the pulse assigned
no polarity at all, because the desk measured that a sentiment model grows louder with every
repeat of the same line (`data/sentiment_comparison.json`) — exactly what a coordinated campaign
exploits. That left it scoring the trivial-classifier floor on TweetEval (macro-recall 1/3,
``data/sentiment_tweeteval.json``) while VADER alone scored 0.570. So every post that survives
the abuse screen now carries a VADER compound score (Hutto & Gilbert, ICWSM 2014, run unmodified
from the MIT copy vendored at `argus/vendor/vader` through `eval/baselines/vader_loader.py`;
notice in ``licenses/vaderSentiment-MIT.txt``), classed with the authors' own thresholds —
positive at ``compound >= 0.05``, negative at ``<= -0.05``, neutral between (paper §4, p. 9:
"classification thresholds set at -0.05 and +0.05"; the inclusive boundaries are
``README.rst:241-245``). The
per-name summary is where the old objection is answered rather than ignored: shares are weighted
by the same clusters the coordination count uses, so **a coordinated story counts once however
many accounts pushed it**, and one account repeating a story counts once, as the most-shared rule
below already treats it. Independent accounts saying similar things each still count — that is
what a crowd is. The raw per-post counts are kept beside the weighted shares so the collapse is
visible, and posts the abuse screen withholds are not scored (their share is reported; the
screen withholds negative posts about seven times as often as positive ones, and
`eval/sentiment_tweeteval.py` measures what that does to the read).

**Abusive text is withheld, and a campaign is never quoted (2026-09-25).** Run over TweetEval's
hate and offensive test sets through `pulse()` and `lines_for()` (`eval/sentiment_tweeteval.py`,
``data/sentiment_tweeteval.json``), the unscreened pulse counted every abusive post as crowd
volume and, when three accounts posted one hateful line, quoted it back as the example of a
coordinated story. Every post now passes `market/abuse.py` first; a flagged post adds nothing to
any count and its text is never stored. That screen reads words, so it misses 39% of hateful and
60% of offensive test posts, and the misses still formed coordinated stories that were quoted in
every simulated sweep — so a coordinated story's text is no longer quoted at all, only its size.
What still gets through, measured, not assumed: missed abusive posts are counted as volume, and
an abusive line two accounts carry independently can still be quoted as the most-shared story.

**Taken from `agent-reach` (MIT, Panniantong/agent-reach).** Its routing choice — which backend
answers for which platform, and that Reddit has no working anonymous path (its
`channels/reddit.py` docstring, live-verified) — is followed; the CLIs are called through ARGUS's
existing `market.evidence.TwitterSource` / `RedditSource`, which already fix the Windows
encoding bugs those CLIs have.

    python -m argus.market.social_pulse            # all traded names, writes data/social_pulse.json
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from argus.eval.baselines import vader_loader
from argus.market.abuse import DEFAULT_SCREEN, REASON_TERM, Verdict, handle_is_abusive
from argus.market.abuse import screen as abuse_screen
from argus.truth.novelty import (
    COORDINATION_WINDOW,
    MIN_COORDINATED_SOURCES,
    NoveltyReport,
    cluster,
)

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
    """One post in the shape `truth.novelty.cluster` reads: ``source`` is the author."""

    id: str
    claim: str
    source: str
    available_at: datetime
    channel: str


Fetch = Callable[[str, datetime], tuple[list[Post], list[str]]]
Screen = Callable[[str], Verdict]
Tone = Callable[[str], float]
"""A polarity scorer: text to a compound score in [-1, 1]. VADER's in production."""

TONES = ("positive", "neutral", "negative")
TONE_SCORER = (f"VADER compound (Hutto & Gilbert, ICWSM 2014), >= {vader_loader.POSITIVE_AT} "
               f"positive, <= {vader_loader.NEGATIVE_AT} negative (paper §4; README.rst:241-245)")
TONE_WEIGHTING = ("a coordinated story counts once however many accounts pushed it; one account "
                  "repeating a story counts once; every other account counts once")

_VADER: Tone | None = None


def vader_tone() -> Tone:
    """VADER's compound score, from the vendored copy, loaded once per process.

    Raises `vader_loader.VaderLoadError` when it cannot be loaded; :func:`pulse` reports that
    as an unscored tone rather than inventing one."""
    global _VADER
    if _VADER is None:
        analyzer = vader_loader.load_analyzer()

        def compound(text: str) -> float:
            return float(analyzer.polarity_scores(text)["compound"])

        _VADER = compound
    return _VADER


def voice_weights(report: NoveltyReport) -> dict[str, float]:
    """Each post's share of one voice, from the clusters the coordination count already uses.

    A coordinated story (`truth.novelty.Cluster.coordinated`: three or more distinct accounts
    inside two hours) splits one vote across all its posts. In any other story, an account's
    posts split one vote between them, so a bot repeating itself is one voice and two people who
    independently posted the same headline are two. Keyed by post id."""
    weights: dict[str, float] = {}
    for story in report.clusters:
        if story.coordinated:
            for item in story.item_ids:
                weights[item] = 1.0 / story.size
            continue
        per_account = Counter(story.sources)
        for item, source in zip(story.item_ids, story.sources, strict=True):
            weights[item] = 1.0 / per_account[source]
    return weights


def percentages(shares: dict[str, float]) -> dict[str, int]:
    """Whole percentages that add to exactly 100 (largest remainder), so a reader never sees a
    tone line summing to 99 or 101."""
    raw = {name: shares.get(name, 0.0) * 100 for name in TONES}
    whole = {name: int(value) for name, value in raw.items()}
    short = 100 - sum(whole.values())
    for name in sorted(TONES, key=lambda n: raw[n] - whole[n], reverse=True)[:max(short, 0)]:
        whole[name] += 1
    return whole


def _tone(kept: Sequence[Post], report: NoveltyReport,
          scorer: Tone | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The per-name tone summary and the per-post scores behind it (ids, never text)."""
    if scorer is None:
        try:
            scorer = vader_tone()
        except vader_loader.VaderLoadError as exc:
            return {"status": f"not scored: {exc}", "scorer": TONE_SCORER}, []
    weights = voice_weights(report)
    counts = dict.fromkeys(TONES, 0)
    weighted = dict.fromkeys(TONES, 0.0)
    compound_sum = 0.0
    per_post = []
    for post in kept:
        compound = scorer(post.claim)
        label = vader_loader.label(compound)
        weight = weights.get(post.id, 1.0)
        counts[label] += 1
        weighted[label] += weight
        compound_sum += weight * compound
        per_post.append({"id": post.id, "compound": round(compound, 4), "label": label,
                         "weight": round(weight, 4)})
    voices = sum(weighted.values())
    coordinated = [c for c in report.clusters if c.coordinated]
    summary: dict[str, Any] = {
        "status": "ok",
        "scorer": TONE_SCORER,
        "weighting": TONE_WEIGHTING,
        "posts_scored": len(per_post),
        "voices": round(voices, 4),
        "counts": counts,
        "share": {k: v / voices for k, v in weighted.items()} if voices else None,
        "share_unweighted": ({k: v / len(per_post) for k, v in counts.items()}
                             if per_post else None),
        "mean_compound": compound_sum / voices if voices else None,
        "coordinated_stories_counted_once": len(coordinated),
        "posts_in_coordinated_stories": sum(c.size for c in coordinated),
    }
    return summary, per_post

WITHHELD_HANDLE = "[withheld handle]"
"""Shown in place of an account name that reads as abusive. The account still counts as a
source — coordination is about who posted, not what they are called — but its name is not
republished."""


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
          now: datetime, *, screen: Screen | None = abuse_screen,
          tone: Tone | None = None) -> dict[str, Any]:
    """The crowd read for one name.

    Posts the abuse screen flags are withheld before anything is counted: they add nothing to
    ``posts``, ``accounts``, the stories or the coordination count, their text is never stored,
    and the row says how many there were and why. ``screen=None`` reproduces the unscreened
    behaviour this module had before 2026-09-25 — `eval/sentiment_tweeteval.py` runs both to
    measure what the screen changes; nothing else passes it.

    Every post that is kept is scored for tone (``tone``, VADER when not given): ``post_tone``
    carries each post's id, compound score, class and voice weight, and ``tone`` the per-name
    summary. Withheld posts are not scored. The fields that existed before are unchanged.
    """
    kept: list[Post] = []
    withheld: dict[str, int] = {}
    withheld_accounts: set[str] = set()
    for post in posts:
        verdict = screen(post.claim) if screen is not None else None
        if verdict is not None and verdict.abusive:
            reason = verdict.reason or REASON_TERM
            withheld[reason] = withheld.get(reason, 0) + 1
            withheld_accounts.add(post.source)
        else:
            kept.append(post)

    def shown(author: str) -> str:
        if screen is not None and handle_is_abusive(author, screen):
            return WITHHELD_HANDLE
        return author

    report = cluster(kept)
    stories = []
    for story in report.clusters[:5]:
        stories.append({
            "text": story.representative[:220],
            "posts": story.size,
            "accounts": story.distinct_sources,
            "coordinated": story.coordinated,
            "authors": [shown(author) for author in sorted(set(story.sources))[:6]],
            "first_seen": story.first_seen.isoformat(),
        })
    by_channel: dict[str, int] = {}
    for post in kept:
        by_channel[post.channel] = by_channel.get(post.channel, 0) + 1
    tone_summary, post_tone = _tone(kept, report, tone)
    tone_summary["withheld_not_scored"] = sum(withheld.values())
    return {
        "symbol": symbol,
        "posts": len(kept),
        "accounts": len({p.source for p in kept}),
        "by_channel": by_channel,
        "distinct_stories": report.distinct_stories,
        "coordinated_stories": len(report.coordinated),
        "coordinated_posts": sum(c.size for c in report.coordinated),
        "withheld_abusive": sum(withheld.values()),
        "withheld_by_reason": dict(sorted(withheld.items())),
        "withheld_accounts": len(withheld_accounts),
        "top_stories": stories,
        "tone": tone_summary,
        "post_tone": post_tone,
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
        "abuse_screen": (f"{DEFAULT_SCREEN.name} (market/abuse.py): abusive posts are withheld, "
                         "not counted, not quoted"),
        "tone_scorer": TONE_SCORER,
        "tone_weighting": TONE_WEIGHTING,
        "symbols": rows,
    }


def load(path: Path = PULSE_PATH) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def lines_for(symbol: str, snapshot: dict[str, Any] | None,
              now: datetime | None = None, *,
              screen: Screen | None = abuse_screen) -> list[str]:
    """The console's sentences for one name, stating the snapshot's age. Empty when the name was
    not swept, so the answer omits the crowd rather than inventing a quiet one.

    A story's text is screened again before it is quoted, so a snapshot written before the screen
    existed — or by a future bug — still cannot put abusive text in front of a reader.
    ``screen=None`` reproduces the unscreened wording, for the same measurement as :func:`pulse`.
    """

    def quotable(text: object) -> bool:
        return isinstance(text, str) and (screen is None or not screen(text).abusive)

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
    withheld = int(row.get("withheld_abusive") or 0)
    if not row.get("posts"):
        if withheld:
            what = "the one post" if withheld == 1 else f"all {withheld} posts"
            verb = "was" if withheld == 1 else "were"
            return [f"Crowd on X and Reddit ({when}): {what} naming {ticker} in the last 48h "
                    f"{verb} withheld as abusive (slurs, profanity or threats), so there is "
                    "nothing to count."]
        failed = [s for s in row.get("status", []) if "unavailable" in s or "ok=false" in s]
        if failed:
            return [f"Crowd on X and Reddit ({when}): not collected — {'; '.join(failed)}."]
        return [f"Crowd on X and Reddit ({when}): nothing named {ticker} in the last 48h."]
    channels = ", ".join(f"{n} on {'X' if c == 'x' else 'Reddit'}"
                         for c, n in sorted(row.get("by_channel", {}).items()))
    out = [f"Crowd on X and Reddit ({when}): {row['posts']} posts from {row['accounts']} accounts "
           f"({channels}) carry {row['distinct_stories']} distinct stories."]
    if withheld:
        out.append(f"{withheld} further post{'s were' if withheld != 1 else ' was'} withheld as "
                   "abusive (slurs, profanity or threats): not counted above, not quoted.")
    if row.get("coordinated_stories"):
        top = next((s for s in row.get("top_stories", []) if s.get("coordinated")), None)
        # A campaign is described, never quoted (2026-09-25). Run as a brigade over TweetEval's
        # hate test set, the screen withheld 61% of the abusive posts, yet the console still
        # quoted a coordinated abusive line in 63 of 63 sweeps: the lines it misses still form
        # coordinated stories. Republishing a campaign's words is amplification whatever they
        # say, so its size is reported and its text is not. ``screen=None`` keeps the old quote
        # so the eval can measure what this changed.
        if screen is None:
            example = (f" — e.g. \"{top['text'][:110]}\" from {top['accounts']} accounts"
                       if top and quotable(top["text"]) else "")
        else:
            example = (f" — the largest is {top['posts']} posts from {top['accounts']} accounts, "
                       "not quoted" if top else "")
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
    if lead and not quotable(lead["text"]):
        out.append(f"The most-shared story ({lead['posts']} posts from {lead['accounts']} "
                   "accounts) is not quoted: it carries abusive language.")
    elif lead:
        out.append(f"Most-shared story: \"{lead['text'][:140]}\" ({lead['posts']} posts from "
                   f"{lead['accounts']} accounts).")
    else:
        out.append("No story is carried by more than one account — the talk is scattered, not "
                   "converging on a single narrative.")
    tone = tone_line(ticker, row)
    if tone:
        out.append(tone)
    return out


def tone_line(ticker: str, row: dict[str, Any]) -> str | None:
    """One plain sentence on the crowd's tone, or ``None`` for a row written before tone was
    scored (an old snapshot adds no line rather than a made-up one)."""
    tone = row.get("tone")
    if not isinstance(tone, dict):
        return None
    if tone.get("status") != "ok":
        return (f"Crowd tone on {ticker}: not scored in this snapshot "
                f"({tone.get('status') or 'no status'}).")
    share = tone.get("share")
    if not share:
        return None
    whole = percentages(share)
    coordinated = int(tone.get("coordinated_stories_counted_once") or 0)
    counted = ("" if not coordinated else
               "; the coordinated story counts once" if coordinated == 1 else
               f"; each of the {coordinated} coordinated stories counts once")
    line = (f"Crowd tone on {ticker}: {whole['positive']}% positive, {whole['neutral']}% "
            f"neutral, {whole['negative']}% negative across {tone.get('posts_scored')} posts "
            f"from {row.get('accounts')} accounts (VADER{counted}).")
    withheld = int(tone.get("withheld_not_scored") or 0)
    if withheld:
        line += (f" The {withheld} withheld abusive post{'s are' if withheld != 1 else ' is'} "
                 "not scored, so this reads less negative than the whole crowd.")
    # VADER's known lean, measured on TweetEval's 12,284 labelled tweets
    # (`eval/sentiment_tweeteval.py`): its positive share runs about 20 points above the human
    # labels. Said on the line, so "58% positive" reads as the tool's reading, not the crowd's.
    line += (" VADER reads tweets about 20 points more positive than human labellers do "
             "(TweetEval), so compare tone across names and days rather than read it as a level.")
    return line


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI, live network
    import argparse

    from argus.lui.question import TRADED_SYMBOLS

    parser = argparse.ArgumentParser(description="Sweep X and Reddit for the traded names.")
    parser.add_argument("--out", type=Path, default=PULSE_PATH)
    args = parser.parse_args(argv)
    snapshot = sweep(TRADED_SYMBOLS)
    args.out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in snapshot["symbols"]:
        share = row["tone"].get("share") or {}
        tone = " ".join(f"{k[:3]} {share.get(k, 0):.0%}" for k in TONES) if share else "-"
        print(f"{row['symbol']:10} posts {row['posts']:3} accounts {row['accounts']:3} stories "
              f"{row['distinct_stories']:3} coordinated {row['coordinated_stories']} "
              f"withheld {row['withheld_abusive']}  tone {tone}  {'; '.join(row['status'])}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["PULSE_PATH", "TONES", "TONE_SCORER", "TONE_WEIGHTING", "WITHHELD_HANDLE", "Post",
           "Screen", "Tone", "fetch_live", "lines_for", "load", "percentages", "pulse", "sweep",
           "tone_line", "vader_tone", "voice_weights"]
