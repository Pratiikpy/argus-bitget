"""Bake-off tests — the comparison must be controlled or it measures plumbing, not intelligence."""

from __future__ import annotations

from decimal import Decimal

from argus.eval.bakeoff import ModelRun


class TestReplayScoring:
    def test_a_model_that_never_flips_is_stable(self) -> None:
        r = ModelRun("qwen", verdicts=["no_trade"] * 3, quantities=[Decimal("0")] * 3)
        got = r.as_dict("h")
        assert got["verdict_consistency"] == 1.0
        assert got["stable"] is True

    def test_a_flip_at_temperature_zero_is_reported_unstable(self) -> None:
        """A model that flips on identical input has sampled a decision, not made one."""
        r = ModelRun(
            "x", verdicts=["trade", "no_trade", "trade"],
            quantities=[Decimal("50"), Decimal("0"), Decimal("50")],
        )
        got = r.as_dict("h")
        assert got["verdict_consistency"] < 0.8
        assert got["stable"] is False

    def test_same_side_wildly_different_size_is_unstable(self) -> None:
        r = ModelRun(
            "x", verdicts=["trade"] * 3,
            quantities=[Decimal("10"), Decimal("190"), Decimal("40")],
        )
        got = r.as_dict("h")
        assert got["verdict_consistency"] == 1.0
        assert got["size_dispersion"] > 0.25
        assert got["stable"] is False

    def test_a_failed_model_reports_errors_rather_than_a_score(self) -> None:
        """An unreachable provider must not silently score as consistent."""
        r = ModelRun("down", errors=["replay 0: timeout"])
        got = r.as_dict("h")
        assert got["decisions"] == 0
        assert got["verdict_consistency"] is None
        assert got["stable"] is None
        assert got["errors"]

    def test_mean_confidence_is_reported(self) -> None:
        r = ModelRun(
            "x", verdicts=["no_trade"] * 2, quantities=[Decimal("0")] * 2,
            confidences=[0.9, 0.7],
        )
        assert r.as_dict("h")["mean_confidence"] == 0.8


class TestAgreementIsNotInferredFromSilence:
    """An earlier version counted a failed provider as consent: it filtered out the None and
    found one distinct verdict left, then reported the models as agreeing."""

    def test_one_reporting_model_is_not_agreement(self) -> None:
        from argus.eval.bakeoff import ModelRun

        working = ModelRun("qwen", verdicts=["reduce"] * 3,
                           quantities=[Decimal("100")] * 3)
        broken = ModelRun("deepseek_nim", errors=["replay 0: timeout"])

        modal = {
            r.provider: max(set(r.verdicts), key=r.verdicts.count) if r.verdicts else None
            for r in (working, broken)
        }
        reporting = {k: v for k, v in modal.items() if v}
        assert len(reporting) == 1
        agree = len(reporting) >= 2 and len(set(reporting.values())) == 1
        assert agree is False


class TestTruncationRetry:
    """Reasoning can exhaust the token budget before any answer is emitted."""

    def test_empty_content_with_reasoning_is_a_truncation_not_a_parse_failure(self) -> None:
        from argus.llm.qwen import Completion, Usage

        truncated = Completion(
            content="", reasoning="thinking at length about the decision" * 40,
            usage=Usage(100, 900, 900, 1000), finish_reason="length",
        )
        # The signature of the case: no answer, but real reasoning spent.
        assert not truncated.content.strip()
        assert truncated.reasoning.strip()
        assert truncated.usage.reasoning_share == 1.0
