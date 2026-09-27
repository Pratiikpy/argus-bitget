"""A headline must not travel without its breakdown, and the breakdown must be read.

The two failures this module exists for are reproduced here as fixtures, with their shapes taken
from the project's own record: a universe whose whole return came from one instrument, and a
momentum result that was good on the full sample and negative in both halves. Each must raise the
named flag; the honest versions of the same numbers must not.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.eval import groupwise_audit
from argus.eval.groupwise import (
    FLAGS,
    HISTOGRAM_BUCKETS,
    MIN_HALF_ITEMS,
    GroupwiseError,
    Item,
    audit,
    breakdown,
    halves_from,
    split_half,
)


def _items(rows: list[tuple[float, str, str]]) -> list[Item]:
    """``(value, symbol, date)`` rows, ordered by date."""
    return [Item(value=v, groups={"symbol": s, "date": d}, order=d) for v, s, d in rows]


class TestOneGroupCarryingTheResult:
    def test_one_instrument_producing_the_whole_return_is_flagged(self) -> None:
        """Single-name trend: four big wins on one symbol, small losses on every other."""
        rows = [(40.0, "NVDA", f"2026-0{m}-01") for m in range(1, 5)]
        rows += [(-3.0, s, "2026-05-01") for s in ("AAPL", "MSFT", "TSLA", "AMZN", "META")]
        table = breakdown(_items(rows), "symbol")
        assert table.direction == 1
        assert table.carried and table.largest == "NVDA"
        assert table.total_without_largest == pytest.approx(-15.0)
        assert "NVDA carries the headline" in table.reading

    def test_a_result_every_group_shares_is_not_carried_even_when_one_group_is_most_of_it(
            self) -> None:
        """The rejected share threshold: the largest mover supplies most of the total, and
        nothing about the result depends on it."""
        rows = [(30.0, "NVDA", "2026-01-01"), (1.0, "AAPL", "2026-01-01"),
                (1.0, "MSFT", "2026-01-01")]
        table = breakdown(_items(rows), "symbol")
        assert not table.carried
        assert table.largest == "NVDA" and table.largest_share == pytest.approx(30 / 32)

    def test_a_single_group_is_flagged_as_one(self) -> None:
        report = audit("one", _items([(1.0, "NVDA", "2026-01-01"), (2.0, "NVDA", "2026-01-02")]),
                       ["symbol"], headline="h", orientation="o")
        assert report.flags == ("single_group:symbol=NVDA",)
        assert not report.broken_down

    def test_macro_average_disagreeing_with_the_item_average_is_flagged(self) -> None:
        """Many small positive items in one group outvote few large negative ones by count."""
        rows = [(1.0, "A", f"2026-01-{d:02d}") for d in range(1, 21)]
        rows += [(-5.0, "B", "2026-02-01"), (-5.0, "C", "2026-02-02")]
        table = breakdown(_items(rows), "symbol")
        assert table.micro_mean > 0 > table.macro_mean
        assert table.macro_disagrees

    def test_the_histogram_uses_mind2webs_buckets_and_sums_to_one(self) -> None:
        rows = [(1.0, "A", "d1"), (-1.0, "B", "d1"), (1.0, "B", "d2")]
        rows += [(-1.0, "C", f"d{i}") for i in range(5)] + [(9.0, "C", "d9")]
        table = breakdown(_items(rows), "symbol")
        assert tuple(table.loss_histogram) == HISTOGRAM_BUCKETS
        assert sum(table.loss_histogram.values()) == pytest.approx(1.0)
        assert table.loss_histogram[">3"] == pytest.approx(1 / 3)

    def test_orientation_changes_words_never_the_verdict(self) -> None:
        rows = [(40.0, "NVDA", "2026-01-01"), (-3.0, "AAPL", "2026-01-02"),
                (-3.0, "MSFT", "2026-01-03")]
        flipped = [(-v, s, d) for v, s, d in rows]
        a = audit("a", _items(rows), ["symbol"], headline="h", orientation="o")
        b = audit("b", _items(flipped), ["symbol"], headline="h", orientation="o")
        assert a.flags == b.flags


class TestTheHalves:
    def test_momentum_good_in_full_sample_and_negative_in_both_halves_is_caught(self) -> None:
        """The recorded failure: two halves the same sign as each other, the full sample the
        other. Only a ratio-like statistic can do it, so it arrives as a recorded split."""
        split = halves_from(-0.2, -0.1, full_mean=0.4, first_items=500, second_items=500,
                            label="IS vs OOS Sharpe")
        assert split.contradicts_full and not split.flips and split.flagged
        assert "both halves point the other way" in split.reading

    def test_halves_that_disagree_are_flagged(self) -> None:
        rows = [(2.0, "A", f"2026-01-{d:02d}") for d in range(1, 6)]
        rows += [(-1.0, "A", f"2026-02-{d:02d}") for d in range(1, 6)]
        split, why = split_half(_items(rows))
        assert split is not None and why == ""
        assert split.flips and split.first_mean > 0 > split.second_mean

    def test_halves_agreeing_with_the_headline_are_not_flagged(self) -> None:
        rows = [(1.0 + d / 10, "A", f"2026-01-{d:02d}") for d in range(1, 11)]
        split, _ = split_half(_items(rows))
        assert split is not None and not split.flagged

    def test_a_date_shared_by_every_symbol_never_straddles_the_cut(self) -> None:
        rows = [(1.0, s, d) for d in ("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04")
                for s in ("A", "B", "C")]
        split, _ = split_half(_items(rows))
        assert split is not None
        assert split.first_items % 3 == 0 and split.second_items % 3 == 0

    def test_too_few_rows_for_a_sign_to_mean_anything_is_reported_not_flipped(self) -> None:
        rows = [(1.0, "A", "d1"), (-1.0, "A", "d2")]
        split, why = split_half(_items(rows))
        assert split is None and str(MIN_HALF_ITEMS) in why

    def test_rows_without_an_order_are_not_silently_split(self) -> None:
        items = [Item(1.0, {"symbol": "A"}, "d1"), Item(1.0, {"symbol": "A"})]
        split, why = split_half(items)
        assert split is None and "carry no order" in why


class TestRefusals:
    def test_an_empty_comparison_is_refused(self) -> None:
        with pytest.raises(GroupwiseError, match="no items"):
            audit("x", [], ["symbol"], headline="h", orientation="o")

    def test_a_row_missing_a_declared_key_is_refused_rather_than_dropped(self) -> None:
        items = [Item(1.0, {"symbol": "A"}), Item(1.0, {"date": "d"})]
        with pytest.raises(GroupwiseError, match="silently"):
            breakdown(items, "symbol")

    def test_no_key_is_refused(self) -> None:
        with pytest.raises(GroupwiseError, match="no grouping key"):
            audit("x", [Item(1.0, {})], [], headline="h", orientation="o")

    def test_a_non_finite_value_is_refused(self) -> None:
        with pytest.raises(GroupwiseError, match="non-finite"):
            breakdown([Item(float("nan"), {"symbol": "A"})], "symbol")

    def test_every_flag_is_one_of_the_four(self) -> None:
        rows = [(40.0, "NVDA", "d1"), (-3.0, "AAPL", "d2"), (-3.0, "MSFT", "d3")]
        report = audit("x", _items(rows), ["symbol", "date"], headline="h", orientation="o")
        assert report.flags
        assert all(flag.split(":", 1)[0] in FLAGS for flag in report.flags)
        blob = report.as_dict()
        assert blob["flagged"] and json.dumps(blob)


class TestTheAuditArtefact:
    """``data/groupwise_audit.json`` is what the standing register's gate reads."""

    def test_every_register_artefact_ends_in_one_of_four_states_with_a_reason(self) -> None:
        blob = json.loads(groupwise_audit.REPORT_PATH.read_text(encoding="utf-8"))
        entries = blob["artefacts"]
        for ref in blob["register_artefacts"]:
            assert ref in entries, ref
        for ref, entry in entries.items():
            assert entry["status"] in groupwise_audit.STATUSES, ref
            if entry["status"] == groupwise_audit.CHECKED:
                assert entry["headlines"] and entry["sha256"], ref
                assert all(h["role"] in groupwise_audit.ROLES for h in entry["headlines"]), ref
            else:
                assert entry.get("reason"), ref
        assert sum(blob["counts"].values()) == len(entries)
        assert blob["qwen_calls"] == 0

    def test_the_flagged_list_is_exactly_the_flagged_headlines(self) -> None:
        blob = json.loads(groupwise_audit.REPORT_PATH.read_text(encoding="utf-8"))
        from_entries = sorted(
            (ref, h["name"]) for ref, e in blob["artefacts"].items()
            for h in e.get("headlines", []) if h.get("flags"))
        listed = sorted((f["artefact"], f["headline"]) for f in blob["flagged"])
        assert listed == from_entries


class TestTheHeadlineInterval:
    """A 95% interval on the headline that keeps each date's cross-section whole and respects
    serial dependence (research/harvest/21-fairlearn.md)."""

    @staticmethod
    def _rows(n_dates: int, per_date: int, mean: float, noise: float, seed: int = 1) -> list[Any]:
        import random

        from argus.eval.groupwise import Item

        rng = random.Random(seed)
        return [Item(value=mean + rng.gauss(0, noise), groups={"symbol": f"S{k}"},
                     order=f"2026-{d // 28 + 1:02d}-{d % 28 + 1:02d}")
                for d in range(n_dates) for k in range(per_date)]

    def test_a_clear_effect_excludes_zero_and_a_null_does_not(self) -> None:
        from argus.eval.groupwise import headline_interval

        clear = headline_interval(self._rows(40, 4, mean=1.0, noise=0.5))
        null = headline_interval(self._rows(40, 4, mean=0.0, noise=0.5))
        assert clear is not None and not clear.spans_zero and clear.low < 1.0 < clear.high
        assert null is not None and null.spans_zero
        assert clear.method.startswith("stationary") and clear.units == 40

    def test_dates_are_the_unit_not_rows(self) -> None:
        """Forty rows on one date are one draw: the interval is refused, not narrowed by rows."""
        from argus.eval.groupwise import headline_interval

        assert headline_interval(self._rows(1, 40, mean=1.0, noise=0.5)) is None

    def test_rows_without_an_order_are_resampled_one_by_one(self) -> None:
        from argus.eval.groupwise import Item, headline_interval

        rows = [Item(value=float(v), groups={"k": "a"}) for v in range(20)]
        interval = headline_interval(rows)
        assert interval is not None and interval.method.startswith("bootstrap over rows")
        assert interval.block_length == 1.0 and interval.low < 9.5 < interval.high

    def test_too_few_units_give_no_interval(self) -> None:
        from argus.eval.groupwise import headline_interval

        assert headline_interval(self._rows(5, 3, mean=1.0, noise=0.1)) is None

    def test_the_report_carries_it_and_the_group_rows_carry_an_error(self) -> None:
        from argus.eval.groupwise import audit

        report = audit("t", self._rows(40, 4, mean=1.0, noise=0.5), ["symbol"],
                       headline="h", orientation="higher is better")
        blob = report.as_dict()
        assert blob["interval"]["spans_zero"] is False and blob["interval"]["resamples"] == 2000
        assert all(row["mean_se"] is not None for row in blob["tables"][0]["per_group"])
        assert not any("interval" in flag for flag in report.flags)  # reported, not a new flag

    def test_it_is_reproducible(self) -> None:
        from argus.eval.groupwise import headline_interval

        rows = self._rows(40, 4, mean=0.3, noise=1.0)
        assert headline_interval(rows) == headline_interval(rows)
