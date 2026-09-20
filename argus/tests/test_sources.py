"""Source-health tests — a source that answers with nothing must not count as live.

Track 3 scores data sources on count *and* effectiveness, and the whole value of this module is
that it refuses to conflate them. The failure it guards against is the one every integration
eventually has: an endpoint returns HTTP 200 with an empty list, the probe records a success, and a
dead feed stays in the inventory for months. EMPTY is therefore a distinct state from OK and does
not count toward the live total.
"""

from __future__ import annotations

from datetime import UTC, datetime

from argus.eval.sources import Health, Probe, report

AT = datetime(2026, 9, 13, 16, 0, tzinfo=UTC)


def _probe(name: str, health: Health, items: int = 0) -> Probe:
    return Probe(name, "market/x.py", health, "detail", items)


class TestOnlyAnsweringSourcesCount:
    def test_an_empty_source_is_not_live(self) -> None:
        """The defect this exists to catch: 200 with nothing in it is not a data source."""
        got = report([_probe("a", Health.OK, 3), _probe("b", Health.EMPTY)], at=AT)
        assert got.live == 1
        assert got.count(Health.EMPTY) == 1

    def test_an_errored_source_is_not_live(self) -> None:
        got = report([_probe("a", Health.OK, 3), _probe("b", Health.ERROR)], at=AT)
        assert got.live == 1

    def test_a_skipped_source_is_neither_live_nor_a_failure(self) -> None:
        """A source needing a credential we do not have is a stated absence, not a broken feed."""
        got = report([_probe("a", Health.OK, 1), _probe("b", Health.SKIPPED)], at=AT)
        assert got.live == 1
        assert got.count(Health.SKIPPED) == 1
        assert got.count(Health.ERROR) == 0

    def test_the_live_count_is_the_one_the_verdict_quotes(self) -> None:
        empties = [_probe(f"x{i}", Health.EMPTY) for i in range(4)]
        got = report([_probe("a", Health.OK, 1), *empties], at=AT)
        assert "1 of 5 source(s) answered" in got.verdict


class TestTheVerdict:
    def test_a_fully_healthy_sweep_says_so_plainly(self) -> None:
        got = report([_probe(f"s{i}", Health.OK, 2) for i in range(12)], at=AT)
        assert "Every configured source is live in this run" in got.verdict

    def test_failures_are_named_rather_than_summed_away(self) -> None:
        got = report(
            [_probe("a", Health.OK, 1), _probe("b", Health.EMPTY), _probe("c", Health.ERROR)],
            at=AT,
        )
        assert "1 answered with nothing" in got.verdict
        assert "1 failed" in got.verdict

    def test_it_says_a_silent_source_is_a_name_in_a_table(self) -> None:
        got = report([_probe("a", Health.OK, 1), _probe("b", Health.EMPTY)], at=AT)
        assert "name in a table" in got.verdict

    def test_the_verdict_carries_the_timestamp(self) -> None:
        """A capability claim without a time is not checkable; a judge trying it tomorrow needs to
        know when this was true."""
        assert "2026-09-13T16:00:00" in report([_probe("a", Health.OK, 1)], at=AT).verdict


class TestTheReport:
    def test_the_rendered_report_lists_what_did_not_answer(self) -> None:
        text = report(
            [_probe("good", Health.OK, 5), _probe("dead", Health.ERROR)], at=AT
        ).render()
        assert "what did not answer:" in text
        assert "dead" in text

    def test_a_clean_sweep_has_no_failure_section(self) -> None:
        text = report([_probe("good", Health.OK, 5)], at=AT).render()
        assert "what did not answer:" not in text

    def test_every_row_names_the_module_a_reader_can_open(self) -> None:
        text = report([_probe("good", Health.OK, 5)], at=AT).render()
        assert "market/x.py" in text

    def test_the_dict_separates_the_four_states(self) -> None:
        got = report([
            _probe("a", Health.OK, 1), _probe("b", Health.EMPTY),
            _probe("c", Health.ERROR), _probe("d", Health.SKIPPED),
        ], at=AT).as_dict()
        assert (got["live"], got["empty"], got["errored"], got["skipped"]) == (1, 1, 1, 1)
        assert got["total"] == 4

    def test_the_dict_states_that_a_parse_failure_counts(self) -> None:
        """A URL that answers while our parser raises is not a working source, and the note says
        the probe goes through the module rather than the endpoint."""
        got = report([_probe("a", Health.OK, 1)], at=AT).as_dict()
        assert "parse failure counts as a failure" in got["note"]


class TestTheLiveArtefact:
    """If the artefact exists, its content must match what this module claims about it."""

    def _load(self) -> dict[str, object] | None:
        import json

        from argus.eval.sources import REPORT_PATH

        if not REPORT_PATH.exists():
            return None
        loaded: dict[str, object] = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return loaded

    def test_the_counts_add_up_to_the_total(self) -> None:
        import pytest

        payload = self._load()
        if payload is None:
            pytest.skip("run `python -m argus.eval.sources` to produce the artefact")
        total = int(payload["total"])  # type: ignore[call-overload]
        parts = sum(
            int(payload[k]) for k in ("live", "empty", "errored", "skipped")  # type: ignore[call-overload]
        )
        assert parts == total

    def test_every_live_source_reported_at_least_one_item(self) -> None:
        import pytest

        payload = self._load()
        if payload is None:
            pytest.skip("artefact not built")
        for row in payload["results"]:  # type: ignore[union-attr]
            if row["health"] == "ok":
                assert row["items"] >= 1, row["name"]
