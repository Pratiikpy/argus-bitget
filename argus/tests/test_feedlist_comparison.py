"""Tests for the ARGUS-vs-TradingAgents perception-layer failure-handling comparison.

Every case here runs BOTH systems' real code — the vendored TradingAgents
``route_to_vendor()`` and ARGUS's real ``argus.market.evidence.gather()`` — nothing is mocked
at the logic level; only the network-facing leaf calls are replaced with controlled stand-ins
(subclasses of ARGUS's real source classes, or direct entries in TradingAgents' real
``VENDOR_METHODS`` dict) so the comparison never touches a real network. See
``eval/feedlist_comparison.py``'s module docstring for the structural finding these tests pin.
"""

from __future__ import annotations

import pytest

from argus.eval.baselines.tradingagents_feedlist_loader import load_interface_module
from argus.eval.feedlist_comparison import (
    SCOPE_STATEMENT,
    main,
    measure_costs,
    render,
    run_ablation,
    run_argus_core_source_failure,
    run_argus_optional_source_failure,
    run_out_of_sample_case,
    run_reproducibility_check,
    run_tradingagents_core_category_failure,
    run_tradingagents_optional_category_failure,
    swept_argus_cases,
    swept_router_cases,
)


@pytest.fixture
def interface_module():
    return load_interface_module()


class TestCoreVsOptionalCategoryHandling:
    def test_tradingagents_raises_on_a_broken_core_category(self, interface_module) -> None:
        case = run_tradingagents_core_category_failure(interface_module)
        assert case.outcome == "raised"

    def test_tradingagents_degrades_on_a_broken_optional_category(self, interface_module) -> None:
        case = run_tradingagents_optional_category_failure(interface_module)
        assert case.outcome == "degraded_to_sentinel"
        assert "DATA_UNAVAILABLE" in case.detail

    def test_argus_returns_cleanly_on_a_broken_core_equivalent_source(self) -> None:
        case = run_argus_core_source_failure()
        assert case.outcome == "returned_cleanly"
        assert "unavailable" in case.detail

    def test_argus_returns_cleanly_on_a_broken_optional_equivalent_source(self) -> None:
        case = run_argus_optional_source_failure()
        assert case.outcome == "returned_cleanly"
        assert "unavailable" in case.detail


class TestSweptCases:
    def test_every_tradingagents_core_category_raises_when_fully_broken(
        self, interface_module
    ) -> None:
        results = swept_router_cases(interface_module)
        core = [r for r in results if r.category not in interface_module.OPTIONAL_CATEGORIES]
        assert core, "expected at least one core category in the real TOOLS_CATEGORIES"
        assert all(r.outcome == "raised" for r in core)

    def test_every_tradingagents_optional_category_never_raises(self, interface_module) -> None:
        results = swept_router_cases(interface_module)
        optional = [r for r in results if r.category in interface_module.OPTIONAL_CATEGORIES]
        assert optional, "expected at least one optional category in the real TOOLS_CATEGORIES"
        assert all(r.outcome != "raised" for r in optional)

    def test_argus_never_raises_across_every_caught_exception_type(self) -> None:
        results = swept_argus_cases()
        assert len(results) >= 2
        assert all(r.outcome == "returned_cleanly" for r in results)


class TestAblation:
    def test_the_exception_scoping_is_load_bearing(self) -> None:
        result = run_ablation()
        assert result.caught_types_are_swallowed is True
        assert result.uncaught_type_propagates is True
        assert result.the_scoping_is_load_bearing is True


class TestReproducibility:
    def test_both_systems_are_deterministic_on_the_same_scenario(self, interface_module) -> None:
        result = run_reproducibility_check(interface_module)
        assert result["tradingagents_reproducible"] is True
        assert result["argus_reproducible"] is True


class TestOutOfSample:
    def test_the_real_desk_notes_log_has_multiple_distinct_sources(self) -> None:
        result = run_out_of_sample_case()
        if result.checked:
            assert result.cycles > 0
            assert result.distinct_sources_seen > 0
        else:
            pytest.skip("real desk-notes artefact not present in this environment")


class TestCosts:
    def test_costs_are_measured_and_positive(self, interface_module) -> None:
        costs = measure_costs(interface_module)
        assert costs["tradingagents_dispatch_us_per_call"] > 0
        assert costs["argus_gather_us_per_call"] > 0


class TestMainAndRender:
    def test_main_runs_the_whole_comparison(self) -> None:
        report = main()
        assert report["tradingagents_core_case"]["outcome"] == "raised"
        assert report["tradingagents_optional_case"]["outcome"] == "degraded_to_sentinel"
        assert report["argus_core_equivalent_case"]["outcome"] == "returned_cleanly"
        assert report["argus_optional_equivalent_case"]["outcome"] == "returned_cleanly"
        assert report["swept_ta_core_always_raises"] is True
        assert report["swept_ta_optional_never_raises"] is True
        assert report["swept_argus_never_raises"] is True
        assert report["ablation"]["the_scoping_is_load_bearing"] is True

    def test_render_names_both_systems_and_the_disagreement(self) -> None:
        report = main()
        text = render(report)
        assert "ARGUS gather()" in text
        assert "TradingAgents" in text
        assert "raised" in text
        assert "returned_cleanly" in text

    def test_scope_statement_names_what_is_not_claimed(self) -> None:
        assert "NOT claimed" in SCOPE_STATEMENT
        assert "design is a defect" in SCOPE_STATEMENT
