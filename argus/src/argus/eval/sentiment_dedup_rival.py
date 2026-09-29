"""The fair general rival for sentiment integrity: collapse near-duplicates, then run finBERT.

`eval/sentiment_comparison.py` showed ProsusAI/finBERT's naive aggregate growing louder with every
repeat of one rumour while ARGUS's sentiment analyst did not. That comparison has two gaps an
honest reviewer names at once (Activity/28_CAPABILITY_CLOSE_PLAN_2.md §20):

* **the rival was the weakest version of the general tool.** No careful engineer feeds five
  copies of one post to a classifier and counts the votes; they deduplicate first. The category's
  own best sources are coordination and near-duplicate collapse (CooRTweet, MinHash), and ARGUS
  already carries an exact near-duplicate clusterer (`truth/novelty.py`, Jaccard over word
  shingles, threshold 0.55). So the rival run here is *dedup, then finBERT*: one finBERT reading
  per story, not per post;
* **the two arms were scored on different questions.** ARGUS counted as resisting if repetition
  did not flip a non-actionable read to an actionable one; finBERT counted as pushed if its
  aggregate moved toward the rumour at all. Here every arm is scored on the same question:
  *did repetition move the reader toward the rumour's side?* For a vote counter that is more
  rumour-side votes after repetition than before; for ARGUS's analyst, a rumour-side signal after
  repetition that was not there before, or a higher confidence in it.

The ARGUS arm is read from the recorded answers in `data/sentiment_comparison.json` (the analyst
is a model call; nothing is re-asked here, so this module spends no Qwen). The posts are rebuilt
deterministically from `eval/sentiment_cases.py` exactly as that harness built them, and finBERT
runs locally.

NOT claimed: anything about real posts. The cases are designed; the route to a real population is
run 1's coordinated story clusters, which needs model calls for the ARGUS arm.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from argus.eval.sentiment_cases import MANIPULATION
from argus.truth import artefact
from argus.truth.novelty import DUPLICATE_AT, cluster
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "sentiment_dedup_rival.json"
SOURCE_PATH = DATA_DIR / "sentiment_comparison.json"

_SIDE = {"bullish": "positive", "bearish": "negative"}


class DedupRivalError(RuntimeError):
    """The recorded comparison cannot be matched to the rebuilt cases."""


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: the binomial test of ``b`` against ``b + c`` at one half."""
    from fractions import Fraction
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(Fraction(comb(n, k), 2 ** n) for k in range(0, min(b, c) + 1))
    return float(min(Fraction(1), 2 * tail))


def rumour_votes(labels: Sequence[str], direction: str) -> int:
    """How many readings land on the rumour's side (positive for a bullish rumour)."""
    wanted = _SIDE.get(direction)
    return sum(1 for label in labels if label == wanted)


def dedup_labels(classifier: Any, evidence: Sequence[Any]) -> tuple[list[str], int]:
    """finBERT once per near-duplicate cluster: the representative's label, and the cluster
    count. Single-link clustering at ``truth.novelty.DUPLICATE_AT``."""
    report = cluster(evidence, threshold=DUPLICATE_AT)
    texts = [c.representative for c in report.clusters]
    return [str(r["label"]) for r in classifier(texts)], len(texts)


def argus_moved(single: dict[str, Any], repeated: dict[str, Any], direction: str) -> bool:
    """The analyst moved toward the rumour: a rumour-side signal after repetition that was not
    there on the single post, or the same side held with more confidence."""
    before = single.get("signal") == direction
    after = repeated.get("signal") == direction
    if after and not before:
        return True
    return bool(after and before
                and float(repeated.get("confidence", 0)) > float(single.get("confidence", 0)))


def run(classifier: Any, recorded: dict[str, Any]) -> dict[str, Any]:
    from argus.eval.sentiment_comparison import (
        coordinated_scenario,
        diverse_scenario,
        single_source_scenario,
    )

    by_claim = {(n["symbol"], n["narrative"]): n for n in recorded["narratives"]}
    rows: list[dict[str, Any]] = []
    for case in MANIPULATION:
        rec = by_claim.get((case.symbol, case.claim))
        if rec is None:
            raise DedupRivalError(f"no recorded answer for {case.symbol}: {case.claim[:40]}")
        direction = str(rec["direction"])
        single = single_source_scenario(case.claim)
        base = rumour_votes([str(r["label"]) for r in classifier([e.claim for e in single])],
                            direction)
        row: dict[str, Any] = {"symbol": case.symbol, "direction": direction,
                               "claim": case.claim, "single_rumour_votes": base}
        for kind, evidence in (("template", coordinated_scenario(case.claim)),
                               ("diverse", diverse_scenario(case))):
            naive = rumour_votes([str(r["label"]) for r in classifier([e.claim for e in evidence])],
                                 direction)
            labels, stories = dedup_labels(classifier, evidence)
            deduped = rumour_votes(labels, direction)
            argus_key = "argus_coordinated" if kind == "template" else "argus_diverse"
            row[kind] = {
                "posts": len(evidence), "stories_after_dedup": stories,
                "finbert_naive_votes": naive, "finbert_dedup_votes": deduped,
                "finbert_naive_moved": naive > base, "finbert_dedup_moved": deduped > base,
                "argus_moved": argus_moved(rec["argus_single"], rec[argus_key], direction),
            }
        rows.append(row)

    def count(kind: str, arm: str) -> int:
        return sum(1 for r in rows if r[kind][arm])

    summary: dict[str, dict[str, Any]] = {kind: {arm: count(kind, arm) for arm in
                      ("finbert_naive_moved", "finbert_dedup_moved", "argus_moved")}
               for kind in ("template", "diverse")}
    for kind in ("template", "diverse"):
        summary[kind]["cases"] = len(rows)
        # Exact McNemar on the discordant cases: rival moved and ARGUS held, against the reverse.
        rival_only = sum(1 for r in rows if r[kind]["finbert_dedup_moved"]
                         and not r[kind]["argus_moved"])
        argus_only = sum(1 for r in rows if r[kind]["argus_moved"]
                         and not r[kind]["finbert_dedup_moved"])
        summary[kind]["discordant"] = {"dedup_moved_argus_held": rival_only,
                                       "argus_moved_dedup_held": argus_only,
                                       "exact_mcnemar_p": exact_mcnemar(rival_only, argus_only)}
        summary[kind]["mean_stories_after_dedup"] = round(
            sum(r[kind]["stories_after_dedup"] for r in rows) / len(rows), 2)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": "did repetition move the reader toward the rumour's side?",
        "dedup": f"truth.novelty.cluster, word-shingle Jaccard >= {DUPLICATE_AT}, single link",
        "argus_answers_from": "data/sentiment_comparison.json (recorded; no model call here)",
        "summary": summary,
        "rows": rows,
        "scope_statement": (
            "Claimed: on the 30 designed manipulation narratives, how often repetition moved each "
            "reader toward the rumour, scored on the same question for finBERT counting every "
            "post, finBERT after near-duplicate collapse, and ARGUS's recorded analyst. NOT "
            "claimed: anything about real posts, or about catching manipulation; the cases are "
            "designed."),
    }


def main() -> int:  # pragma: no cover - CLI
    from argus.eval.baselines.finbert_loader import load_finbert

    recorded = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    report = run(load_finbert(), recorded)
    artefact.write(REPORT_PATH, report)
    for kind, block in report["summary"].items():
        print(f"{kind}: {block}")
    print(f"saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["DedupRivalError", "argus_moved", "dedup_labels", "rumour_votes", "run"]
