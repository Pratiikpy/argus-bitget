"""Novelty tests — syndication must be caught and independent reporting must not be.

Both errors are possible and only one of them flatters. Calling copies copies understates the
evidence base, which is the safe direction; calling independent stories copies would let a thin
feed look like a careful one. So every clustering test is paired: a wire story rewritten three ways
must collapse to one, and three genuinely different stories about the same company must stay three.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.agents.novelty import (
    COORDINATION_WINDOW,
    DUPLICATE_AT,
    MIN_COORDINATED_SOURCES,
    SHINGLE,
    Cluster,
    NoveltyError,
    cluster,
    jaccard,
    shingles,
)

T = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
WIRE = "NVIDIA reports record quarterly revenue of 35 billion dollars beating analyst estimates"


@dataclass(frozen=True)
class Item:
    id: str
    claim: str
    source: str
    available_at: datetime


def _items(*rows: tuple[str, str, str, int]) -> list[Item]:
    return [
        Item(id=i, claim=c, source=s, available_at=T + timedelta(minutes=m)) for i, c, s, m in rows
    ]


class TestShingling:
    def test_it_ignores_case_and_punctuation(self) -> None:
        assert shingles("The Cat, Sat! On A Mat") == shingles("the cat sat on a mat")

    def test_a_short_text_becomes_one_shingle_rather_than_none(self) -> None:
        """A one-line headline must still be comparable. Returning an empty set would make it
        dissimilar to everything, including its own copy."""
        got = shingles("two words")
        assert got == frozenset({"two words"})

    def test_empty_text_has_no_shingles(self) -> None:
        assert shingles("") == frozenset()
        assert shingles("!!! ???") == frozenset()

    def test_the_window_is_the_stated_size(self) -> None:
        got = shingles("a b c d e", size=4)
        assert got == frozenset({"a b c d", "b c d e"})
        assert SHINGLE == 4

    def test_identical_text_has_identical_shingles(self) -> None:
        assert shingles(WIRE) == shingles(WIRE)


class TestJaccard:
    def test_identical_sets_score_one(self) -> None:
        assert jaccard(shingles(WIRE), shingles(WIRE)) == pytest.approx(1.0)

    def test_disjoint_sets_score_zero(self) -> None:
        assert jaccard(frozenset({"a b c d"}), frozenset({"w x y z"})) == 0.0

    def test_an_empty_set_is_uncomparable_not_similar(self) -> None:
        """Two empty texts are not the same story; there is nothing to compare."""
        assert jaccard(frozenset(), frozenset()) == 0.0
        assert jaccard(shingles(WIRE), frozenset()) == 0.0

    def test_it_is_symmetric(self) -> None:
        left, right = shingles(WIRE), shingles(WIRE + " today")
        assert jaccard(left, right) == jaccard(right, left)

    def test_a_rewrite_scores_high_and_a_different_story_scores_low(self) -> None:
        rewrite = shingles(WIRE + " in its latest quarter")
        other = shingles("Commerce Department tightens semiconductor export controls on chips")
        assert jaccard(shingles(WIRE), rewrite) > DUPLICATE_AT
        assert jaccard(shingles(WIRE), other) < DUPLICATE_AT


class TestClusteringCatchesSyndication:
    def test_three_rewrites_of_one_wire_story_collapse_to_one(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "bloomberg", 8),
            ("c", WIRE.replace("dollars beating", "dollars, beating"), "cnbc", 15),
        ))
        assert got.distinct_stories == 1
        assert got.items == 3

    def test_three_different_stories_stay_three(self) -> None:
        """The error that would flatter a thin feed: collapsing independent reporting."""
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", "Commerce Department tightens export controls on advanced chips", "reuters", 60),
            ("c", "Officer files an insider sale under a pre-arranged 10b5-1 plan", "sec", 120),
        ))
        assert got.distinct_stories == 3

    def test_the_effective_count_is_the_story_count(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "bloomberg", 5),
            ("c", "An entirely unrelated filing about executive compensation", "sec", 10),
        ))
        assert got.effective_items == 2
        assert got.duplication_ratio == pytest.approx(1.5)

    def test_the_earliest_item_represents_the_cluster(self) -> None:
        """The first to publish is the one the others may have copied. Picking the longest would
        usually pick a later aggregation."""
        got = cluster(_items(
            ("late", WIRE + " according to a company statement released this afternoon", "x", 30),
            ("first", WIRE, "reuters", 0),
        ))
        assert got.clusters[0].representative == WIRE

    def test_clustering_is_single_link_so_rewrites_chain(self) -> None:
        """A copy of a copy can drift past the threshold from the original while still matching its
        parent. Comparing only to a representative would split one story in two, which understates
        duplication — the direction that flatters."""
        a = "alpha beta gamma delta epsilon zeta eta theta"
        b = "beta gamma delta epsilon zeta eta theta iota"
        c = "gamma delta epsilon zeta eta theta iota kappa"
        got = cluster(_items(("a", a, "s1", 0), ("b", b, "s2", 5), ("c", c, "s3", 10)),
                      threshold=0.4)
        assert got.distinct_stories == 1

    def test_an_empty_evidence_set_is_not_an_error(self) -> None:
        got = cluster([])
        assert got.items == 0
        assert got.distinct_stories == 0
        assert "no evidence" in got.verdict

    def test_an_impossible_threshold_raises(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(NoveltyError, match="not in"):
                cluster(_items(("a", WIRE, "s", 0)), threshold=bad)


class TestCoordination:
    def test_several_distinct_sources_in_a_short_window_reads_as_coordinated(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "bloomberg", 8),
            ("c", WIRE + " in the quarter", "cnbc", 15),
        ))
        assert got.clusters[0].coordinated
        assert len(got.coordinated) == 1

    def test_one_source_repeating_itself_is_not_coordination(self) -> None:
        """A busy newsroom is not a campaign. The threshold is on distinct sources, not items."""
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "reuters", 8),
            ("c", WIRE + " in the quarter", "reuters", 15),
        ))
        assert not got.clusters[0].coordinated

    def test_the_same_story_spread_over_a_day_is_syndication_not_coordination(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "bloomberg", 400),
            ("c", WIRE + " in the quarter", "cnbc", 800),
        ))
        assert not got.clusters[0].coordinated
        assert "syndication" in got.clusters[0].verdict

    def test_two_sources_are_not_enough(self) -> None:
        got = cluster(_items(("a", WIRE, "reuters", 0), ("b", WIRE + " today", "bloomberg", 5)))
        assert not got.clusters[0].coordinated
        assert MIN_COORDINATED_SOURCES == 3

    def test_the_window_is_the_stated_one(self) -> None:
        assert timedelta(hours=2) == COORDINATION_WINDOW

    def test_coordination_is_not_reported_as_falsehood(self) -> None:
        """A promoted narrative can be true. The verdict must say what it licenses and no more."""
        got = cluster(_items(
            ("a", WIRE, "reuters", 0),
            ("b", WIRE + " today", "bloomberg", 8),
            ("c", WIRE + " in the quarter", "cnbc", 15),
        ))
        assert "does not make it false" in got.clusters[0].verdict


class TestVelocity:
    def test_a_single_item_has_no_rate(self) -> None:
        got = cluster(_items(("a", WIRE, "reuters", 0)))
        assert got.clusters[0].velocity is None

    def test_copies_arriving_faster_give_a_higher_rate(self) -> None:
        fast = cluster(_items(
            ("a", WIRE, "r", 0), ("b", WIRE + " today", "b", 6), ("c", WIRE + " now", "c", 12),
        )).clusters[0]
        slow = cluster(_items(
            ("a", WIRE, "r", 0), ("b", WIRE + " today", "b", 300), ("c", WIRE + " now", "c", 600),
        )).clusters[0]
        assert fast.velocity is not None and slow.velocity is not None
        assert fast.velocity > slow.velocity

    def test_simultaneous_copies_do_not_divide_by_zero(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "r", 0), ("b", WIRE + " today", "b", 0), ("c", WIRE + " now", "c", 0),
        )).clusters[0]
        assert got.velocity == pytest.approx(3.0)


class TestTheReport:
    def test_no_duplication_says_the_item_count_is_the_evidence_count(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "r", 0),
            ("b", "Commerce Department tightens export controls on advanced chips", "r", 60),
        ))
        assert "the item count is the evidence count" in got.verdict

    def test_duplication_names_the_factor_a_confidence_is_overstated_by(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "r", 0), ("b", WIRE + " today", "b", 5), ("c", WIRE + " now", "c", 9),
        ))
        assert "overstated by that factor" in got.verdict

    def test_clusters_are_ordered_largest_first(self) -> None:
        got = cluster(_items(
            ("solo", "An unrelated regulatory filing on executive compensation", "sec", 0),
            ("a", WIRE, "r", 10), ("b", WIRE + " today", "b", 12), ("c", WIRE + " now", "c", 14),
        ))
        assert got.clusters[0].size == 3

    def test_the_dict_carries_every_cluster_and_its_sources(self) -> None:
        got = cluster(_items(
            ("a", WIRE, "reuters", 0), ("b", WIRE + " today", "bloomberg", 5),
        )).as_dict()
        assert got["distinct_stories"] == 1
        assert got["clusters"][0]["distinct_sources"] == 2
        assert "bloomberg" in got["clusters"][0]["sources"]

    def test_the_rendered_report_names_each_story(self) -> None:
        text = cluster(_items(("a", WIRE, "reuters", 0))).render()
        assert "NOVELTY" in text
        assert "no duplicate found" in text

    def test_a_cluster_reports_its_own_span(self) -> None:
        got = Cluster(
            representative="x", item_ids=("a", "b"), sources=("r", "b"),
            first_seen=T, last_seen=T + timedelta(hours=3),
        )
        assert got.span == timedelta(hours=3)
        assert not got.coordinated


class TestAgainstRealEvidence:
    def test_it_accepts_the_evidence_type_the_desk_already_carries(self) -> None:
        """Nothing upstream has to change to be measured: the fields are the ones
        `agents.analysts.Evidence` already has."""
        from argus.truth.evidence import Evidence

        got = cluster([
            Evidence(id="a", claim=WIRE, source="reuters", available_at=T),
            Evidence(id="b", claim=WIRE + " today", source="bloomberg", available_at=T),
        ])
        assert got.distinct_stories == 1
