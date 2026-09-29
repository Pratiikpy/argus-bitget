"""Sentiment integrity on REAL coordinated posting, not designed cases (capability 20, route (c).2
of ``Activity/28_CAPABILITY_CLOSE_PLAN_2.md`` section 20).

``eval/sentiment_comparison.py`` and ``eval/sentiment_dedup_rival.py`` proved the coordination
defence on 30 designed manipulation narratives. ``groupwise_audit`` correctly classes that artefact
``designed_cases``: a demonstration that a property holds on chosen inputs, not a measurement over a
population. The plan's own route back to ``statistically_valid_evaluation`` and
``out_of_sample_test`` is "the coordination defence measured on a sample of real posts". This module
is that measurement.

**Where the real posts come from.** Track 2's sentiment-agent paper run 1 (``t2-sentiment-agent``,
the Track 2 run 1 repository, read-only — nothing there is written or restarted) fetched real X and
Reddit posts through the same live channels ARGUS itself uses (``twitter-cli``, ``rdt-cli``;
``sentiment_agent/crowd/adapters.py``, a documented, licensed port of ARGUS's own
``truth/novelty.py``) and published its story clusters at ``public/ledger.jsonl``: 3,948 distinct
story clusters over four real trading days, of which **35 were mechanically flagged coordinated** by
the run's own novelty screen (>= 3 distinct sources inside a 2-hour window — the identical rule
``truth/novelty.Cluster.coordinated`` applies). That flag needs no model judgement and none is spent
on it: coordination here is a property of posting timestamps and account identity, not of content a
reader (human or Qwen) had to interpret. The published ledger keeps only one representative text per
cluster; every individual post's real text is re-fetched here, live, by its own id, through the same
CLIs (:func:`fetch_post`) — verified against the real endpoints before this module was written
(``twitter tweet <id> --json``, ``rdt read <id> --json``, both authenticated, both returning the
real post).

**What is and is not claimed.** There is no ground truth that any of these 35 real clusters is a
manipulation attempt — some plausibly are (the bot-network spam this run actually surfaced, five to
eight throwaway accounts posting one templated "an outstanding stock analyst" pitch with a rotating
cashtag list, minutes apart), some may be genuine, fast-breaking coverage of one story. So the claim
this module can support is exactly the plan's own: **does repetition move a reader's read**, not
**does the reader catch manipulation**. Concretely, per cluster: read one lone post (the earliest),
then read the full, capped burst of the same story, and ask whether the second read commits harder
in the same direction than the first — for finBERT counting every post, finBERT after
near-duplicate collapse (``truth/novelty.cluster``, exactly ``sentiment_dedup_rival``'s rival), and
ARGUS's real ``SentimentAnalyst``. None of the three arms is told an external "correct" direction
(there isn't one); each is compared against **its own single-post reading**, which is the only
honest reference available for real, unlabelled data (:func:`argus_shifted`,
:func:`finbert_naive_shifted`, :func:`finbert_dedup_shifted`).

**Matching.** For every coordinated cluster, one uncoordinated cluster (>= 2 items, otherwise not
flagged coordinated by the run's own screen) is matched on the same calendar day and at least one
shared symbol, greedily preferring the largest available match (more repetition to test), no reuse.
Checked directly against the real ledger before committing to this design: all 35 coordinated
clusters find a same-day, same-symbol match on the first pass, with no relaxation needed
(:func:`match_uncoordinated`). The uncoordinated population is a control on ordinary repetition, not
this capability's claim, and is reported separately (``groupwise_audit``'s ``context`` role) rather
than folded into the gating headline.

**Budget.** 35 coordinated + 35 matched clusters, one ARGUS call on the lone post and one on the
full (capped) cluster each: 70 * 2 = 140 real Qwen calls — the plan's own ~140 estimate, approved by
the project owner. finBERT is local, free and runs on every post. No Qwen call labels anything:
the coordinated/uncoordinated split is the run's own mechanical flag, so no inter-rater reliability
question arises (there is no rater).

**Statistics, pre-registered before any cluster was read.** The paired primary statistic is exact
McNemar (:func:`argus.eval.sentiment_dedup_rival.exact_mcnemar`, reused rather than
re-derived) on the binary "resisted" indicator, ARGUS vs. each finBERT arm, over the 35 coordinated
clusters. A bootstrap interval on the paired difference is reported alongside it, computed with the
project's own trusted method (``argus.eval.groupwise.headline_interval``, the same stationary
bootstrap the groupwise audit itself trusts) over the 35 (or 70, pooled) cluster-level rows — well
above its own ``MIN_INTERVAL_UNITS`` floor of 8. A **day-block** bootstrap was also pre-registered,
because the plan names one explicitly; run here honestly, it has only **4** blocks (the four real
calendar days this ledger covers), below that same floor of 8 the project's own bootstrap code
enforces before it will call an interval "measured" rather than "decoration". It is reported for
transparency, clearly labelled underpowered, and is not what gates anything below.

**Out-of-sample, precisely.** Nothing here is fit or tuned: the cap, the shift definitions and the
match rule were all fixed by the plan before this module read a single real post, so there is no
train/test split of a parameter to report. "Out of sample" instead means temporal generalisation —
the finding is recomputed on a held-out final day (2026-09-27) not used to choose anything, and its
own sign and significance are reported next to the pooled one (:func:`holdout_block`).

    python -m argus.eval.sentiment_integrity_real fetch --record <path to t2-sentiment-agent/public>
    python -m argus.eval.sentiment_integrity_real run --budget 300000
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.agents.analysts import AnalystView, SentimentAnalyst
from argus.eval.baselines.finbert_loader import FinbertClassifier
from argus.eval.groupwise import Item, headline_interval
from argus.eval.sentiment_comparison import FinbertNaiveAggregate, run_finbert_naive_aggregate
from argus.eval.sentiment_dedup_rival import exact_mcnemar
from argus.llm.base import ChatModel
from argus.llm.ledger import cost_step
from argus.llm.qwen import BudgetExhausted
from argus.truth import artefact
from argus.truth.evidence import Evidence
from argus.truth.novelty import DUPLICATE_AT
from argus.truth.novelty import cluster as dedup_cluster
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "sentiment_integrity_real.json"
POSTS_CACHE = DATA_DIR / "sentiment_integrity_real_posts.jsonl"
"""The re-fetched posts themselves, with their authors' handles. Kept on the machine that ran the
measurement and never published (the public repository ignores it): they are other people's posts.
The report keeps each cluster's verdicts and ids, which is what the audit and the register read."""

CLUSTER_CAP = 10
"""Most posts read per cluster (the earliest, chronologically). Real bursts run up to 39 posts;
capped for the same reason the designed scenarios use 5 template mutations rather than every
possible rephrasing — a bounded, stated evidence count, chosen to keep the live network refetch and
the Qwen prompt size predictable rather than to flatter either arm. The vast majority of real
clusters are far smaller: median coordinated size is 7, so most are used whole."""

MIN_CLUSTER_ITEMS = 2
"""Fewer than this and there is no repetition to test; the cluster is skipped and recorded why."""


class SentimentIntegrityRealError(RuntimeError):
    """The real-cluster measurement cannot proceed honestly."""


# ============================================================================================
# 1. Reading run 1's real ledger — no network, no model, pure parsing.
# ============================================================================================


@dataclass(frozen=True, slots=True)
class ClusterRecord:
    """One real story cluster, aggregated across every snapshot the ledger recorded it in."""

    cluster_id: str
    item_ids: tuple[str, ...]
    sources: tuple[str, ...]
    symbols: tuple[str, ...]
    coordinated: bool
    first_seen: str
    last_seen: str
    representative: str

    @property
    def day(self) -> str:
        return self.first_seen[:10]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id, "item_ids": list(self.item_ids),
            "sources": list(self.sources), "symbols": list(self.symbols),
            "coordinated": self.coordinated, "first_seen": self.first_seen,
            "last_seen": self.last_seen, "distinct_sources": len(set(self.sources)),
            "size": len(self.item_ids),
        }


def aggregate_clusters(lines: Iterable[str]) -> dict[str, ClusterRecord]:
    """Every ``snapshot`` event's crowd clusters, merged by ``cluster_id``.

    A story is seen in several consecutive snapshots as more posts arrive inside its window; this
    takes the union of ``item_ids``/``sources``/``symbols``, the widest ``first_seen``/
    ``last_seen``, the longest ``representative`` seen, and ``coordinated`` true if any snapshot
    ever called it that — exactly :func:`argus.eval.coordination_graph_eval.stories`'s own
    aggregation rule,
    applied here to the fields that module drops (``item_ids``, ``symbols``, timestamps) and it
    keeps (``sources``, the coordinated flag).
    """
    acc: dict[str, dict[str, Any]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("kind") != "snapshot":
            continue
        crowd = (event.get("payload", {}).get("snapshot") or {}).get("crowd") or {}
        for c in crowd.get("clusters", []):
            cid = str(c.get("cluster_id"))
            rec = acc.setdefault(cid, {
                "item_ids": set(), "sources": set(), "symbols": set(), "coordinated": False,
                "first_seen": None, "last_seen": None, "representative": "",
            })
            rec["item_ids"].update(str(i) for i in c.get("item_ids", []) or [])
            rec["sources"].update(str(s) for s in c.get("sources", []) or [])
            rec["symbols"].update(str(s) for s in c.get("symbols", []) or [])
            rec["coordinated"] = rec["coordinated"] or bool(c.get("coordinated"))
            fs, ls = c.get("first_seen"), c.get("last_seen")
            if fs and (rec["first_seen"] is None or fs < rec["first_seen"]):
                rec["first_seen"] = fs
            if ls and (rec["last_seen"] is None or ls > rec["last_seen"]):
                rec["last_seen"] = ls
            rep = str(c.get("representative", ""))
            if len(rep) > len(rec["representative"]):
                rec["representative"] = rep
    return {
        cid: ClusterRecord(
            cluster_id=cid, item_ids=tuple(sorted(rec["item_ids"])),
            sources=tuple(sorted(rec["sources"])), symbols=tuple(sorted(rec["symbols"])),
            coordinated=bool(rec["coordinated"]), first_seen=str(rec["first_seen"] or ""),
            last_seen=str(rec["last_seen"] or ""), representative=str(rec["representative"]),
        )
        for cid, rec in acc.items()
    }


def read_ledger(record_dir: Path) -> dict[str, ClusterRecord]:
    """:func:`aggregate_clusters` over ``record_dir/ledger.jsonl``. Split on ``\\n`` only:
    ``str.splitlines`` also breaks on U+2028 inside a post's own text (the same reason
    ``coordination_graph_eval.stories`` does)."""
    text = (record_dir / "ledger.jsonl").read_text(encoding="utf-8")
    return aggregate_clusters(text.split("\n"))


# ============================================================================================
# 2. Matching every coordinated cluster to one real uncoordinated control.
# ============================================================================================


@dataclass(frozen=True, slots=True)
class MatchedPair:
    coordinated: ClusterRecord
    control: ClusterRecord | None
    match_basis: str


def _candidates(
    pool: dict[str, ClusterRecord], used: set[str], *, day: str, symbols: set[str],
    need_symbol: bool,
) -> list[ClusterRecord]:
    return [
        c for cid, c in pool.items()
        if cid not in used and c.day == day and (not need_symbol or set(c.symbols) & symbols)
    ]


def match_uncoordinated(
    coordinated: Sequence[ClusterRecord], uncoordinated: Sequence[ClusterRecord],
) -> tuple[MatchedPair, ...]:
    """One real uncoordinated cluster per coordinated one: same calendar day, at least one shared
    symbol, >= :data:`MIN_CLUSTER_ITEMS` posts, greedy on the largest available (more repetition to
    test), no reuse. Falls back to same-day-only, then to symbol-only, and names which rule matched
    — checked against the real ledger before this was written: every one of the 35 real coordinated
    clusters matches on the first, strictest rule, so the fallbacks exist for a ledger that changes,
    not because this one needed them.

    Deterministic: candidates are broken only by size then by ``cluster_id``, never by draw order or
    a random seed, so the same ledger always produces the same 35 pairs.
    """
    pool = {c.cluster_id: c for c in uncoordinated if len(c.item_ids) >= MIN_CLUSTER_ITEMS}
    used: set[str] = set()
    out: list[MatchedPair] = []
    for coord in sorted(coordinated, key=lambda c: c.cluster_id):
        symbols = set(coord.symbols)
        found = _candidates(pool, used, day=coord.day, symbols=symbols, need_symbol=True)
        basis = "day+symbol"
        if not found:
            found = _candidates(pool, used, day=coord.day, symbols=symbols, need_symbol=False)
            basis = "day_only"
        if not found:
            found = [c for cid, c in pool.items() if cid not in used and set(c.symbols) & symbols]
            basis = "symbol_only"
        if not found:
            out.append(MatchedPair(coord, None, "no_match"))
            continue
        found.sort(key=lambda c: (-len(c.item_ids), c.cluster_id))
        chosen = found[0]
        used.add(chosen.cluster_id)
        out.append(MatchedPair(coord, chosen, basis))
    return tuple(out)


def primary_symbol(symbols: Sequence[str]) -> str:
    """The one symbol the analyst is asked about. A real cluster's spam template often mentions
    several tickers at once; the alphabetically-first is used, deterministically, and recorded on
    every row so a reader can see exactly which symbol each call used."""
    if not symbols:
        raise SentimentIntegrityRealError("a cluster with no symbols cannot be asked about")
    return min(symbols)


_X_ID = re.compile(r"^x:(\d+)$")
_REDDIT_ID = re.compile(r"^reddit:([0-9a-z]+)$")


def _sort_key(item_id: str) -> tuple[int, int]:
    """A chronological proxy from the id alone, so the capped sample can be chosen before any post
    is fetched. X ids are Snowflake ids (monotonically increasing with time by construction); Reddit
    ids36 are assigned in creation order. Neither claim is exact to the millisecond, both are exact
    in ordering, which is all a "earliest N" cap needs."""
    x = _X_ID.match(item_id)
    if x:
        return (0, int(x.group(1)))
    r = _REDDIT_ID.match(item_id)
    if r:
        return (1, int(r.group(1), 36))
    return (2, 0)


def capped_item_ids(item_ids: Sequence[str], *, cap: int = CLUSTER_CAP) -> tuple[str, ...]:
    """The earliest ``cap`` ids, chronologically — always a prefix, so the single reference post
    (index 0) is always inside the full evidence set too."""
    return tuple(sorted(item_ids, key=_sort_key)[:cap])


# ============================================================================================
# 3. Re-fetching real post text by id, live, through the same CLIs the T2 run itself used.
# ============================================================================================


@dataclass(frozen=True, slots=True)
class FetchedPost:
    item_id: str
    text: str
    source: str
    published_at: datetime
    channel: str

    def as_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "text": self.text, "source": self.source,
                "published_at": self.published_at.isoformat(), "channel": self.channel}


PostFetcher = Callable[[str], FetchedPost | None]
"""``None`` means the post could not be fetched (deleted, rate-limited, malformed) — recorded, never
raised: a live re-fetch of posts from days ago is expected to lose a few."""


def _run_utf8(argv: list[str], *, timeout: float) -> tuple[int, str]:  # pragma: no cover - network
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(  # fixed argv, no shell, no untrusted input
        argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, check=False, env=env, stdin=subprocess.DEVNULL,
    )
    return completed.returncode, completed.stdout or ""


MAX_TEXT_CHARS = 1000
"""Fetched post text is clipped to this length, matching the established convention
(``sentiment_agent/crowd/adapters.py:MAX_TEXT_CHARS``, ARGUS's own live feed uses the same cap).
Not cosmetic: a real Reddit post's title-plus-body can run well past BERT's 512-token limit
(one crashed a live run at 826 tokens, uncapped, before this cap was added — finBERT's shared
pipeline is a vendored baseline and is not the place to add truncation handling)."""


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_TEXT_CHARS else text[: MAX_TEXT_CHARS - 1] + "…"


def _x_time(row: dict[str, Any]) -> datetime | None:
    iso = row.get("createdAtISO")
    if isinstance(iso, str) and iso:
        try:
            stamp = datetime.fromisoformat(iso)
        except ValueError:
            return None
        return stamp.astimezone(UTC) if stamp.tzinfo else stamp.replace(tzinfo=UTC)
    return None


def fetch_x_post(raw_id: str, *, timeout: float = 30.0) -> FetchedPost | None:  # pragma: no cover
    """One real tweet by id, through ``twitter tweet`` (twitter-cli). Never raises."""
    try:
        status, stdout = _run_utf8(
            ["twitter", "tweet", raw_id, "--json", "-n", "1"], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if status != 0:
        return None
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        return None
    data = envelope.get("data")
    row = data[0] if isinstance(data, list) and data else None
    if not isinstance(row, dict):
        return None
    text = row.get("text")
    author = row.get("author")
    handle = author.get("screenName") if isinstance(author, dict) else None
    published = _x_time(row)
    if not isinstance(text, str) or not text.strip() or not handle or published is None:
        return None
    return FetchedPost(item_id=f"x:{raw_id}", text=_clip(text), source=f"@{handle}",
                       published_at=published, channel="x")


def fetch_reddit_post(  # pragma: no cover - network
    raw_id: str, *, timeout: float = 45.0,
) -> FetchedPost | None:
    """One real Reddit post by id, through ``rdt read`` (rdt-cli). Never raises."""
    try:
        status, stdout = _run_utf8(
            ["rdt", "read", raw_id, "--json", "-n", "1"], timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if status != 0:
        return None
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        return None
    data = envelope.get("data")
    post = data.get("post") if isinstance(data, dict) else None
    if not isinstance(post, dict):
        return None
    title = post.get("title")
    author = post.get("author")
    created = post.get("created_utc")
    if not isinstance(title, str) or not title.strip() or not isinstance(author, str):
        return None
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        return None
    body = str(post.get("selftext") or "").strip()
    text = (title.strip() if body in ("", "[removed]", "[deleted]")
           else f"{title.strip()} \u2014 {body}")
    try:
        published = datetime.fromtimestamp(float(created), tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return FetchedPost(item_id=f"reddit:{raw_id}", text=_clip(text), source=f"u/{author}",
                       published_at=published, channel="reddit")


def fetch_post(item_id: str) -> FetchedPost | None:  # pragma: no cover - network
    channel, _, raw_id = item_id.partition(":")
    if channel == "x":
        return fetch_x_post(raw_id)
    if channel == "reddit":
        return fetch_reddit_post(raw_id)
    return None


def load_post_cache(path: Path = POSTS_CACHE) -> dict[str, FetchedPost]:
    if not path.exists():
        return {}
    out: dict[str, FetchedPost] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["item_id"])] = FetchedPost(
            # `_clip` again here, not only at fetch time: entries cached before `MAX_TEXT_CHARS`
            # existed (or written by a future producer that forgets it) must not reach finBERT's
            # 512-token pipeline uncapped either.
            item_id=str(row["item_id"]), text=_clip(str(row["text"])), source=str(row["source"]),
            published_at=datetime.fromisoformat(str(row["published_at"])),
            channel=str(row["channel"]),
        )
    return out


def fetch_missing(
    item_ids: Iterable[str], *, cache_path: Path = POSTS_CACHE,
    fetcher: PostFetcher = fetch_post, concurrency: int = 5,
) -> dict[str, FetchedPost]:  # pragma: no cover - network
    """Every id in ``item_ids`` that :func:`load_post_cache` does not already have, fetched live and
    appended to the cache. Returns the full merged set (cached + newly fetched). A fetch that
    returns ``None`` is not retried and not cached — it is recorded as missing in the caller's own
    accounting, not silently treated as an empty post."""
    have = load_post_cache(cache_path)
    wanted = [i for i in dict.fromkeys(item_ids) if i not in have]
    if not wanted:
        return have
    fetched: list[FetchedPost] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for post in pool.map(fetcher, wanted):
            if post is not None:
                fetched.append(post)
    if fetched:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            for post in fetched:
                fh.write(json.dumps(post.as_dict()) + "\n")
    have.update({p.item_id: p for p in fetched})
    return have


# ============================================================================================
# 4. The three arms, self-referenced against the lone post — no external "true direction" exists
#    for real, unlabelled posts.
# ============================================================================================


def build_evidence(posts: Sequence[FetchedPost]) -> list[Evidence]:
    return [
        Evidence(id=p.item_id, claim=p.text, source=p.channel, available_at=p.published_at,
                credibility=0.5)
        for p in posts
    ]


def finbert_naive_shifted(full: FinbertNaiveAggregate) -> bool:
    """The full cluster's own dominant label is the lone (earliest) post's label by construction
    (:func:`capped_item_ids` always keeps it first): ``matching_count > 1`` means at least one other
    real post in the burst read the same way, which is exactly what a naive vote-counter scales on.
    """
    return full.matching_count > 1


def finbert_dedup_shifted(classifier: FinbertClassifier, full_evidence: Sequence[Evidence],
                          single_label: str) -> tuple[bool, int, int]:
    """finBERT run once per near-duplicate story (:mod:`argus.truth.novelty`, the same clusterer
    ``sentiment_dedup_rival`` uses) instead of once per post. Returns (shifted, matching stories,
    distinct stories): shifted when more than one distinct story after dedup reads the same as the
    lone post — a spam template collapsing to one story gives exactly one match, no growth."""
    report = dedup_cluster(full_evidence, threshold=DUPLICATE_AT)
    texts = [c.representative for c in report.clusters]
    labels = [str(r["label"]) for r in classifier(texts)]
    matching = sum(1 for label in labels if label == single_label)
    return matching > 1, matching, len(texts)


def _directional_confidence(view: AnalystView) -> float:
    return view.confidence if view.signal in ("bullish", "bearish") else 0.0


def argus_shifted(single: AnalystView, full: AnalystView) -> bool:
    """Did reading the whole burst commit the real analyst harder than the lone post did: a flip
    from non-actionable to actionable, or the same actionable direction held with more confidence.
    No external "rumour side" is assumed — the lone post's own read is the only honest reference for
    real, unlabelled posts (the designed-case comparisons had a scripted direction to check against;
    these do not)."""
    if not single.is_actionable and full.is_actionable:
        return True
    if single.is_actionable and full.is_actionable and single.signal == full.signal:
        return full.confidence > single.confidence
    return False


# ============================================================================================
# 5. Per-cluster outcome and the full run.
# ============================================================================================


@dataclass(frozen=True, slots=True)
class ClusterOutcome:
    cluster_id: str
    coordinated: bool
    match_basis: str
    symbol: str
    day: str
    n_items_ledger: int
    n_full_used: int
    single_item_id: str
    finbert_single: FinbertNaiveAggregate
    finbert_full: FinbertNaiveAggregate
    finbert_naive_shifted: bool
    finbert_dedup_matching: int
    finbert_dedup_stories: int
    finbert_dedup_shifted: bool
    argus_single: AnalystView
    argus_full: AnalystView
    argus_shifted: bool

    @property
    def argus_resisted(self) -> bool:
        return not self.argus_shifted

    @property
    def finbert_naive_resisted(self) -> bool:
        return not self.finbert_naive_shifted

    @property
    def finbert_dedup_resisted(self) -> bool:
        return not self.finbert_dedup_shifted

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id, "coordinated": self.coordinated,
            "match_basis": self.match_basis, "symbol": self.symbol, "day": self.day,
            "n_items_ledger": self.n_items_ledger, "n_full_used": self.n_full_used,
            "single_item_id": self.single_item_id,
            "finbert_single": self.finbert_single.as_dict(),
            "finbert_full": self.finbert_full.as_dict(),
            "finbert_naive_shifted": self.finbert_naive_shifted,
            "finbert_dedup_matching": self.finbert_dedup_matching,
            "finbert_dedup_stories": self.finbert_dedup_stories,
            "finbert_dedup_shifted": self.finbert_dedup_shifted,
            "argus_single": self.argus_single.as_dict(), "argus_full": self.argus_full.as_dict(),
            "argus_shifted": self.argus_shifted,
            "argus_resisted": self.argus_resisted,
            "finbert_naive_resisted": self.finbert_naive_resisted,
            "finbert_dedup_resisted": self.finbert_dedup_resisted,
        }


def evaluate_cluster(
    client: ChatModel, classifier: FinbertClassifier, record: ClusterRecord,
    posts: dict[str, FetchedPost],
) -> ClusterOutcome | None:
    """One cluster, both arms, two real ARGUS calls. ``None`` if fewer than
    :data:`MIN_CLUSTER_ITEMS` real posts were actually fetched (a live re-fetch loses some)."""
    capped = capped_item_ids(record.item_ids)
    fetched = [posts[i] for i in capped if i in posts]
    if len(fetched) < MIN_CLUSTER_ITEMS:
        return None
    fetched.sort(key=lambda p: p.published_at)
    full_evidence = build_evidence(fetched)
    single_evidence = full_evidence[:1]
    symbol = primary_symbol(record.symbols)

    finbert_single = run_finbert_naive_aggregate(classifier, single_evidence)
    finbert_full = run_finbert_naive_aggregate(classifier, full_evidence)
    dedup_shifted, dedup_matching, dedup_stories = finbert_dedup_shifted(
        classifier, full_evidence, finbert_single.dominant_label)

    analyst = SentimentAnalyst(client)
    with cost_step(f"sentiment_integrity_real:{record.cluster_id}:single"):
        argus_single = analyst.analyse(symbol, single_evidence)
    with cost_step(f"sentiment_integrity_real:{record.cluster_id}:full"):
        argus_full = analyst.analyse(symbol, full_evidence)

    return ClusterOutcome(
        cluster_id=record.cluster_id, coordinated=record.coordinated, match_basis="",
        symbol=symbol, day=record.day, n_items_ledger=len(record.item_ids),
        n_full_used=len(fetched), single_item_id=fetched[0].item_id,
        finbert_single=finbert_single, finbert_full=finbert_full,
        finbert_naive_shifted=finbert_naive_shifted(finbert_full),
        finbert_dedup_matching=dedup_matching, finbert_dedup_stories=dedup_stories,
        finbert_dedup_shifted=dedup_shifted,
        argus_single=argus_single, argus_full=argus_full,
        argus_shifted=argus_shifted(argus_single, argus_full),
    )


def _with_basis(outcome: ClusterOutcome, basis: str) -> ClusterOutcome:
    return replace(outcome, match_basis=basis)


# ============================================================================================
# 6. Paired statistics: exact McNemar (primary) and two bootstraps (row-level, gating; day-block,
#    supplementary and explicitly underpowered).
# ============================================================================================


def _paired_counts(rows: Sequence[ClusterOutcome], rival_resisted: Callable[[ClusterOutcome], bool],
                   ) -> tuple[int, int]:
    """(rival resisted and ARGUS did not, ARGUS resisted and rival did not) — the two discordant
    McNemar cells."""
    rival_only = sum(1 for r in rows if rival_resisted(r) and not r.argus_resisted)
    argus_only = sum(1 for r in rows if r.argus_resisted and not rival_resisted(r))
    return rival_only, argus_only


@dataclass(frozen=True, slots=True)
class PairedComparison:
    rival: str
    n: int
    argus_resisted: int
    rival_resisted: int
    argus_only: int
    rival_only: int
    exact_mcnemar_p: float
    bootstrap: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rival": self.rival, "n": self.n, "argus_resisted": self.argus_resisted,
            "rival_resisted": self.rival_resisted, "argus_only": self.argus_only,
            "rival_only": self.rival_only, "exact_mcnemar_p": self.exact_mcnemar_p,
            "bootstrap_interval": self.bootstrap,
        }


def paired_comparison(
    rows: Sequence[ClusterOutcome], *, rival: str,
    rival_resisted: Callable[[ClusterOutcome], bool],
) -> PairedComparison:
    rival_only, argus_only = _paired_counts(rows, rival_resisted)
    items = [Item(value=float(r.argus_resisted) - float(rival_resisted(r)),
                  groups={"symbol": r.symbol}) for r in rows]
    interval = headline_interval(items)
    return PairedComparison(
        rival=rival, n=len(rows),
        argus_resisted=sum(r.argus_resisted for r in rows),
        rival_resisted=sum(rival_resisted(r) for r in rows),
        argus_only=argus_only, rival_only=rival_only,
        exact_mcnemar_p=exact_mcnemar(rival_only, argus_only),
        bootstrap=interval.as_dict() if interval is not None else None,
    )


def day_block_bootstrap(
    rows: Sequence[ClusterOutcome], *, rival_resisted: Callable[[ClusterOutcome], bool],
    resamples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """The plan's own pre-registered day-block bootstrap on the paired difference (ARGUS resisted
    minus rival resisted), resampling calendar days with replacement. Reported honestly as
    underpowered: this ledger covers four real days, below the project's own
    ``groupwise.MIN_INTERVAL_UNITS`` floor of 8 blocks before an interval is trusted anywhere else
    in this codebase — the same rule is stated here rather than quietly relaxed for one result."""
    import random

    by_day: dict[str, list[float]] = {}
    for r in rows:
        by_day.setdefault(r.day, []).append(float(r.argus_resisted) - float(rival_resisted(r)))
    days = sorted(by_day)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        picked = [rng.choice(days) for _ in days]
        values = [v for day in picked for v in by_day[day]]
        if values:
            draws.append(sum(values) / len(values))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))] if draws else None
    high = draws[int(0.975 * (len(draws) - 1))] if draws else None
    return {
        "blocks": len(days), "days": days, "resamples": resamples,
        "low": None if low is None else round(low, 4),
        "high": None if high is None else round(high, 4),
        "underpowered": len(days) < 8,
        "note": (f"{len(days)} calendar day(s) in this ledger; below the project's own "
                f"MIN_INTERVAL_UNITS floor of 8, so this interval is reported for transparency "
                f"and does not gate the capability's state"),
    }


def holdout_block(
    rows: Sequence[ClusterOutcome], *, rival_resisted: Callable[[ClusterOutcome], bool],
    holdout_day: str,
) -> dict[str, Any]:
    """Temporal generalisation, precisely: nothing here was fit to any of this data (the cap, the
    shift definitions and the match rule were all fixed by the plan before any real post was read),
    so there is no parameter to hold out. What is checked instead is whether the pooled finding
    replicates on a day not used to choose anything — the last real day in the ledger."""
    held = [r for r in rows if r.day == holdout_day]
    kept = [r for r in rows if r.day != holdout_day]
    if not held or not kept:
        return {"holdout_day": holdout_day, "held_out": len(held), "rest": len(kept),
                "ran": False, "reason": "one side is empty"}
    rival_only_h, argus_only_h = _paired_counts(held, rival_resisted)
    rival_only_k, argus_only_k = _paired_counts(kept, rival_resisted)
    return {
        "holdout_day": holdout_day, "held_out": len(held), "rest": len(kept), "ran": True,
        "held_out_argus_resisted": sum(r.argus_resisted for r in held),
        "held_out_rival_resisted": sum(rival_resisted(r) for r in held),
        "rest_argus_resisted": sum(r.argus_resisted for r in kept),
        "rest_rival_resisted": sum(rival_resisted(r) for r in kept),
        "held_out_exact_mcnemar_p": exact_mcnemar(rival_only_h, argus_only_h),
        "rest_exact_mcnemar_p": exact_mcnemar(rival_only_k, argus_only_k),
    }


# ============================================================================================
# 7. The full run and its artefact.
# ============================================================================================


def summarise(coordinated_rows: Sequence[ClusterOutcome], control_rows: Sequence[ClusterOutcome],
             ) -> dict[str, Any]:
    days = sorted({r.day for r in coordinated_rows} | {r.day for r in control_rows})
    holdout_day = days[-1] if days else ""
    return {
        "coordinated": {
            "n": len(coordinated_rows),
            "vs_finbert_naive": paired_comparison(
                coordinated_rows, rival="finbert_naive",
                rival_resisted=lambda r: r.finbert_naive_resisted).as_dict(),
            "vs_finbert_dedup": paired_comparison(
                coordinated_rows, rival="finbert_dedup",
                rival_resisted=lambda r: r.finbert_dedup_resisted).as_dict(),
            "day_block_bootstrap_vs_finbert_dedup": day_block_bootstrap(
                coordinated_rows, rival_resisted=lambda r: r.finbert_dedup_resisted),
            "holdout_vs_finbert_dedup": holdout_block(
                coordinated_rows, rival_resisted=lambda r: r.finbert_dedup_resisted,
                holdout_day=holdout_day),
        },
        "uncoordinated_control": {
            "n": len(control_rows),
            "vs_finbert_naive": paired_comparison(
                control_rows, rival="finbert_naive",
                rival_resisted=lambda r: r.finbert_naive_resisted).as_dict(),
            "vs_finbert_dedup": paired_comparison(
                control_rows, rival="finbert_dedup",
                rival_resisted=lambda r: r.finbert_dedup_resisted).as_dict(),
        } if control_rows else None,
    }


def _safe_evaluate(
    client: ChatModel, classifier: FinbertClassifier, record: ClusterRecord,
    posts: dict[str, FetchedPost], skipped: list[dict[str, Any]], role: str,
) -> ClusterOutcome | None:
    """:func:`evaluate_cluster`, with any failure recorded rather than losing every prior real
    call's spend to one bad cluster. ``BudgetExhausted`` is re-raised — it means the session's
    money is gone, and no further cluster should even try."""
    try:
        return evaluate_cluster(client, classifier, record, posts)
    except BudgetExhausted:
        raise
    except Exception as exc:  # a live model/network call; recorded here, never silently lost
        skipped.append({"cluster_id": record.cluster_id, "role": role,
                        "reason": f"{type(exc).__name__}: {exc}"[:300]})
        return None


def run(
    client: ChatModel, classifier: FinbertClassifier, pairs: Sequence[MatchedPair],
    posts: dict[str, FetchedPost],
) -> dict[str, Any]:
    coordinated_rows: list[ClusterOutcome] = []
    control_rows: list[ClusterOutcome] = []
    skipped: list[dict[str, Any]] = []
    budget_exhausted = False
    for pair in pairs:
        try:
            outcome = _safe_evaluate(client, classifier, pair.coordinated, posts, skipped,
                                     "coordinated")
            if outcome is None and not any(
                    s["cluster_id"] == pair.coordinated.cluster_id for s in skipped):
                skipped.append({"cluster_id": pair.coordinated.cluster_id, "role": "coordinated",
                                "reason": "fewer than MIN_CLUSTER_ITEMS real posts fetched"})
            elif outcome is not None:
                coordinated_rows.append(_with_basis(outcome, pair.match_basis))
            if pair.control is None:
                skipped.append({"cluster_id": pair.coordinated.cluster_id, "role": "control",
                                "reason": "no uncoordinated match found"})
                continue
            control_outcome = _safe_evaluate(client, classifier, pair.control, posts, skipped,
                                             "control")
            if control_outcome is None and not any(
                    s["cluster_id"] == pair.control.cluster_id for s in skipped):
                skipped.append({"cluster_id": pair.control.cluster_id, "role": "control",
                                "reason": "fewer than MIN_CLUSTER_ITEMS real posts fetched"})
            elif control_outcome is not None:
                control_rows.append(_with_basis(control_outcome, pair.match_basis))
        except BudgetExhausted:
            budget_exhausted = True
            skipped.append({"cluster_id": pair.coordinated.cluster_id, "role": "run",
                            "reason": "token budget exhausted; run stopped here"})
            break

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "budget_exhausted": budget_exhausted,
        "source": "t2-sentiment-agent/public/ledger.jsonl (the Track 2 run 1 repository, "
                  "read-only; run 1's own mechanical novelty screen flags 'coordinated', no model "
                  "judgement)",
        "cluster_cap": CLUSTER_CAP,
        "question": "does reading the whole real burst commit the reader harder in the same "
                    "direction than reading the lone earliest post did?",
        "skipped": skipped,
        "summary": summarise(coordinated_rows, control_rows),
        "rows": {
            "coordinated": [r.as_dict() for r in coordinated_rows],
            "uncoordinated_control": [r.as_dict() for r in control_rows],
        },
        "scope_statement": (
            "Claimed: on 35 real coordinated story clusters from Track 2's own paper run 1 "
            "(mechanically flagged, not model-labelled), whether repetition moves ARGUS's real "
            "SentimentAnalyst, finBERT's naive per-post aggregate, and finBERT after near-"
            "duplicate collapse, each judged against its own lone-post reading. NOT claimed: that "
            "any cluster is an actual manipulation attempt (no ground truth exists for that), or "
            "that this generalises beyond X/Reddit posts about the twelve-symbol rToken universe "
            "over four real trading days in late September 2026."
        ),
    }


# ============================================================================================
# CLI
# ============================================================================================


def _cmd_fetch(record: Path, *, cap: int = CLUSTER_CAP) -> int:  # pragma: no cover - CLI, network
    clusters = read_ledger(record)
    coordinated = [c for c in clusters.values() if c.coordinated]
    uncoordinated = [c for c in clusters.values() if not c.coordinated]
    pairs = match_uncoordinated(coordinated, uncoordinated)
    wanted: set[str] = set()
    for pair in pairs:
        wanted.update(capped_item_ids(pair.coordinated.item_ids, cap=cap))
        if pair.control is not None:
            wanted.update(capped_item_ids(pair.control.item_ids, cap=cap))
    print(f"coordinated clusters: {len(coordinated)}; matched pairs: "
         f"{sum(1 for p in pairs if p.control is not None)}; posts to have cached: {len(wanted)}")
    have = fetch_missing(wanted)
    missing = wanted - set(have)
    print(f"cached posts: {len(have)}; still missing after fetch: {len(missing)}")
    if missing:
        print(f"missing ids (not fetchable live): {sorted(missing)[:20]}"
             f"{' ...' if len(missing) > 20 else ''}")
    return 0


def _cmd_run(record: Path, *, budget: int, out: Path) -> int:  # pragma: no cover - CLI, real $
    from argus.eval.baselines.finbert_loader import load_finbert
    from argus.llm.qwen import QwenClient, TokenBudget

    if not os.environ.get("BITGET_QWEN_API_KEY"):
        print("BITGET_QWEN_API_KEY not set; load .secrets/qwen.env into the environment first.")
        return 1

    clusters = read_ledger(record)
    coordinated = [c for c in clusters.values() if c.coordinated]
    uncoordinated = [c for c in clusters.values() if not c.coordinated]
    pairs = match_uncoordinated(coordinated, uncoordinated)
    planned = 2 * len(pairs) + 2 * sum(1 for p in pairs if p.control is not None)
    print(f"{len(pairs)} coordinated clusters; planned real ARGUS calls: {planned} "
         f"(2 per coordinated cluster, 2 per matched control)")

    wanted: set[str] = set()
    for pair in pairs:
        wanted.update(capped_item_ids(pair.coordinated.item_ids))
        if pair.control is not None:
            wanted.update(capped_item_ids(pair.control.item_ids))
    posts = fetch_missing(wanted)
    print(f"posts available (cached + freshly fetched): {len(posts)}/{len(wanted)}")

    client = QwenClient(budget=TokenBudget(limit=budget))
    classifier = load_finbert()
    report = run(client, classifier, pairs, posts)
    report["costs"] = {"argus_real_calls": client.calls, "tokens_spent": client.budget.spent
                       if client.budget else None}
    artefact.write(out, report)
    print(json.dumps(report["summary"], indent=2, default=str))
    print(f"costs: {report['costs']}")
    print(f"saved -> {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="cache real post text for every id this run will need")
    p_fetch.add_argument("--record", type=Path, required=True,
                         help="path to the T2 run's public/ directory (holds ledger.jsonl)")

    p_run = sub.add_parser("run", help="the real measurement (spends Qwen)")
    p_run.add_argument("--record", type=Path, required=True)
    p_run.add_argument("--budget", type=int, required=True)
    p_run.add_argument("--out", type=Path, default=REPORT_PATH)

    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        return _cmd_fetch(args.record)
    return _cmd_run(args.record, budget=args.budget, out=args.out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLUSTER_CAP",
    "MIN_CLUSTER_ITEMS",
    "POSTS_CACHE",
    "REPORT_PATH",
    "ClusterOutcome",
    "ClusterRecord",
    "FetchedPost",
    "MatchedPair",
    "PairedComparison",
    "PostFetcher",
    "SentimentIntegrityRealError",
    "aggregate_clusters",
    "argus_shifted",
    "build_evidence",
    "capped_item_ids",
    "day_block_bootstrap",
    "evaluate_cluster",
    "exact_mcnemar",
    "fetch_missing",
    "fetch_post",
    "fetch_reddit_post",
    "fetch_x_post",
    "finbert_dedup_shifted",
    "finbert_naive_shifted",
    "holdout_block",
    "load_post_cache",
    "main",
    "match_uncoordinated",
    "paired_comparison",
    "primary_symbol",
    "read_ledger",
    "run",
    "summarise",
]
