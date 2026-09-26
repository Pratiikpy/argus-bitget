"""Tests for the ARGUS-vs-OpenBB research-workbench comparison.

Live-network tests: real SEC XBRL data is the whole point of the point-in-time comparison.
Reading OpenBB's real, locally-cloned source for the point-in-time grep and the provider count
is fast and offline. Every expensive function is wrapped in a module-scoped fixture and called
once per test session. See `eval/workbench_comparison.py`'s module docstring for the findings
these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.workbench_comparison import (
    SCOPE_STATEMENT,
    measure_costs,
    render,
    run_failure_cases,
    run_openbb_source_read,
    run_point_in_time_comparison,
    run_reproducibility_check,
    run_source_breadth,
)


@pytest.fixture(scope="module")
def breadth() -> dict:
    try:
        return run_source_breadth()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def openbb_read() -> dict:
    try:
        return run_openbb_source_read()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def pit() -> dict:
    return run_point_in_time_comparison()


@pytest.fixture(scope="module")
def failures() -> dict:
    return run_failure_cases()


@pytest.fixture(scope="module")
def costs() -> dict:
    return measure_costs(repeats=2)


@pytest.fixture(scope="module")
def reproducibility() -> dict:
    return run_reproducibility_check()


class TestSourceBreadth:
    def test_openbb_has_a_real_positive_provider_count(self, breadth: dict) -> None:
        assert breadth["openbb_total_providers"] > 0
        assert breadth["openbb_keyless_providers"] > 0
        assert breadth["openbb_keyless_providers"] <= breadth["openbb_total_providers"]

    def test_openbb_wins_breadth_and_it_is_reported_honestly(self, breadth: dict) -> None:
        """This capability's whole point is that a real loss is stated, not hidden."""
        assert breadth["openbb_wins_breadth"]
        assert breadth["openbb_total_providers"] > breadth["argus_live_sources"]

    def test_argus_live_source_count_is_real_and_positive(self, breadth: dict) -> None:
        assert breadth["argus_live_sources"] > 0
        assert breadth["argus_sources_answering"] > 0


class TestOpenbbSourceRead:
    def test_every_configured_term_was_searched(self, openbb_read: dict) -> None:
        assert len(openbb_read["terms_searched"]) == 6

    def test_zero_point_in_time_representation_found(self, openbb_read: dict) -> None:
        assert openbb_read["total_hits"] == 0
        assert openbb_read["zero_point_in_time_representation"]
        assert openbb_read["hits"] == {}


class TestPointInTimeComparison:
    def test_the_far_past_cutoff_sees_materially_fewer_facts(self, pit: dict) -> None:
        assert pit["far_past_sees_materially_fewer_facts_than_now"]
        assert pit["n_facts_visible_far_past"] < pit["n_facts_visible_now"]

    def test_every_far_past_fact_was_actually_filed_before_the_cutoff(self, pit: dict) -> None:
        assert pit["all_far_past_facts_filed_before_cutoff"]


class TestFailureCases:
    def test_the_boundary_is_the_acceptance_second(self, failures: dict) -> None:
        assert failures["withheld_an_hour_before_acceptance"]
        assert failures["visible_an_hour_after_acceptance"]
        assert failures["boundary_is_sharp"]

    def test_midnight_of_the_filed_day_does_not_see_an_evening_filing(self, failures: dict
                                                                        ) -> None:
        from datetime import datetime

        accepted = datetime.fromisoformat(failures["accepted_at"])
        midnight = datetime.fromisoformat(failures["cutoff_midnight_of_filed_day"])
        assert accepted > midnight
        assert failures["withheld_at_midnight_of_filed_day"]

    def test_the_cutoffs_are_two_hours_apart(self, failures: dict) -> None:
        from datetime import datetime

        after = datetime.fromisoformat(failures["cutoff_hour_after_acceptance"])
        before = datetime.fromisoformat(failures["cutoff_hour_before_acceptance"])
        assert (after - before).total_seconds() == 7200


class TestCosts:
    def test_the_real_fetch_is_measured_and_positive(self, costs: dict) -> None:
        assert costs["argus_as_of_gated_fetch_seconds_per_call"] > 0

    def test_openbb_agent_cost_is_stated_not_measured(self, costs: dict) -> None:
        assert costs["openbb_agent_real_llm_calls_per_query"] == 3
        assert not costs["openbb_agent_keyless"]


class TestReproducibility:
    def test_repeated_fetches_agree(self, reproducibility: dict) -> None:
        assert reproducibility["identical"]


class TestMain:
    def test_render_produces_readable_text(
        self, breadth: dict, openbb_read: dict, pit: dict, failures: dict,
        reproducibility: dict,
    ) -> None:
        report = {
            "source_breadth": breadth,
            "openbb_source_read": openbb_read,
            "point_in_time_comparison": pit,
            "failure_cases": failures,
            "reproducibility": reproducibility,
        }
        text = render(report)
        assert "PERSONALIZED RESEARCH WORKBENCH" in text
        assert "OpenBB" in text


def test_scope_statement_is_honest_about_what_is_not_claimed() -> None:
    assert "NOT claimed" in SCOPE_STATEMENT
    assert SCOPE_STATEMENT.count("NOT claimed") >= 2


def test_scope_statement_discloses_the_real_loss_on_breadth() -> None:
    assert "breadth is competitive with OpenBB's -- it is not" in SCOPE_STATEMENT
