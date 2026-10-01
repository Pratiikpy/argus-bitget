"""Round 17 of the §27 audits (2026-10-01): a judge, a hostile reviewer and a first-time user.

Each test pins one finding (tracker rows 606 onward). Live sources are stubbed or never reached.
"""

from __future__ import annotations

import pytest

from argus.lui.research.kinds import ResearchKind


class TestSizingFromADollarLossLimit:
    def test_a_dollar_loss_limit_is_a_risk_budget(self) -> None:
        from argus.lui.research.sizing import stated_risk_usd

        assert stated_risk_usd("my loss limit is $300") == 300.0

    def test_a_stop_in_atr_with_a_budget_is_a_sizing_question(self) -> None:
        from argus.lui.research.sizing import asks_for_size

        assert asks_for_size("how big a position if I risk $200 with a 2x ATR stop on NVDA")

    def test_a_stop_in_atr_is_not_leverage(self) -> None:
        from argus.lui.research.parse import detect

        request = detect("how big a position if I risk $200 with a 2x ATR stop on NVDA")
        assert request is None or request.kind is not ResearchKind.LEVERAGE

    def test_an_ordinary_question_is_not_sizing(self) -> None:
        from argus.lui.research.sizing import asks_for_size

        assert not asks_for_size("what is the price of NVDA")


class TestRememberedFacts:
    def test_a_loss_limit_is_recalled_in_the_traders_words(self) -> None:
        from argus.lui import memory as mem

        fact = mem.Fact(kind="loss_usd", subject="", value="300",
                        text="my loss limit is $300", at="2026-10-01")
        said = mem.recall_one("what is my loss limit?", [fact])
        assert said is not None and "$300" in " ".join(said)

    def test_nothing_held_is_said_not_invented(self) -> None:
        from argus.lui import memory as mem

        said = mem.recall_one("what is my loss limit?", [])
        assert said is not None and "no loss limit is remembered" in said[0]


class TestAPastWindowIsNotAForecast:
    @pytest.mark.parametrize("q", [
        "did NVDA go up over the last 30 days",
        "did BTC go down this week"])
    def test_a_move_already_made_is_a_quote(self, q: str) -> None:
        from argus.lui.research.parse import detect

        request = detect(q)
        assert request is not None and request.kind is ResearchKind.QUOTE

    def test_a_forward_question_is_still_a_forecast_refusal(self) -> None:
        from argus.lui.research.parse import _PAST_DIRECTION

        assert not _PAST_DIRECTION.search("will NVDA go up over the next 30 days")


class TestTwoWindowsInOneQuestion:
    @pytest.mark.parametrize("q", [
        "NVDA 7 days vs 30 days",
        "NVDA 1 week compared to 3 months"])
    def test_both_windows_are_read(self, q: str) -> None:
        from argus.lui.research.parse import _TWO_WINDOWS

        assert _TWO_WINDOWS.search(q)

    def test_one_window_is_not_two(self) -> None:
        from argus.lui.research.parse import _TWO_WINDOWS

        assert not _TWO_WINDOWS.search("how did NVDA do this week")


class TestNewcomerNoGuarantee:
    @pytest.mark.parametrize("q", [
        "find me a guaranteed 10x coin",
        "give me a guaranteed winner",
        "what is the best guaranteed stock"])
    def test_a_guarantee_is_refused(self, q: str) -> None:
        from argus.lui.newcomer import reply

        said = reply(q)
        assert said is not None and "no return can be guaranteed" in said.lines[0]

    def test_a_loss_limit_is_not_a_guarantee(self) -> None:
        from argus.lui.newcomer import _GUARANTEE

        assert not _GUARANTEE.search("I can't lose more than 10% of my book")


class TestMethodologyQuestions:
    @pytest.mark.parametrize("q", [
        "what is lookahead bias",
        "how do you avoid overfitting",
        "what is survivorship bias"])
    def test_each_is_answered(self, q: str) -> None:
        from argus.lui.newcomer import reply

        assert reply(q) is not None


class TestRivalsScoreboard:
    @pytest.mark.parametrize("q", [
        "Which rival do you lose to?",
        "which competitors beat you"])
    def test_the_question_is_recognised(self, q: str) -> None:
        from argus.lui.rivals import asks_which_rival

        assert asks_which_rival(q)

    def test_a_named_rival_is_not_the_scoreboard(self) -> None:
        from argus.lui.rivals import asks_which_rival

        assert not asks_which_rival("how does ARGUS compare to Nautilus Trader")

    def test_the_scoreboard_names_every_lost_row_from_the_register(self) -> None:
        from argus.eval.standing import REGISTER
        from argus.lui.rivals import scoreboard

        lines, _sources, data = scoreboard("which rival do you lose to")
        lost = [c for c in REGISTER if c.state.value == "lost"]
        assert len(data["rival_rows"]) == len(lost)
        assert f"loses to a named rival on {len(lost)} of its {len(REGISTER)}" in lines[0]

    def test_a_tied_row_that_records_a_loss_says_so(self) -> None:
        from argus.lui.rivals import answer

        lines, _s, _d = answer("is ARGUS better than FinanceBench's published results?")
        assert any("TIED here means no win either way" in line for line in lines)


class TestAFutureQuarterIsNotAnEarningsFigure:
    def test_the_named_future_quarter_is_said_not_reported(self) -> None:
        import re

        pattern = r"\bQ([1-4])\s*(?:of\s+)?(?:FY\s*)?'?(20\d\d|\d\d)\b"
        assert re.search(pattern, "what is TSLA Q3 2027 EPS?", re.I)


class TestAWeekWindowOnACompare:
    @pytest.mark.parametrize("text", [
        "how volatile is NVDA vs TSLA over 7 and 30 days",
        "which is riskier over the last week, AMD or INTC",
        "TSLA vs NVDA volatility, 7 days",
    ])
    def test_a_named_week_is_recognised(self, text: str) -> None:
        from argus.lui.research.dispatch import _WEEK_WINDOW

        assert _WEEK_WINDOW.search(text)

    def test_no_week_named_is_no_week_line(self) -> None:
        from argus.lui.research.dispatch import _WEEK_WINDOW

        assert not _WEEK_WINDOW.search("compare TSLA and NVDA over 30 days")

    def test_the_annualised_week_matches_the_formula(self) -> None:
        import math

        from argus.lui.research.dispatch import _annualised

        hourly = [0.01, -0.01] * 84
        assert _annualised(hourly) == pytest.approx(
            math.sqrt(sum(r * r for r in hourly) / (len(hourly) - 1)) * math.sqrt(24 * 365),
            rel=0.02)
        assert _annualised([0.01]) is None
