"""Round 16 of the §27 audits (2026-10-01): a hostile reviewer's pass on routing, named crash
episodes, multi-name rate questions and questions about the project itself.

Each test pins one finding (tracker rows 599 onward). Live sources are stubbed or never reached.
"""

from __future__ import annotations

from datetime import date

import pytest

from argus.lui.research.kinds import ResearchKind


class TestTheTestSuiteIsNotTheHashChain:
    @pytest.mark.parametrize("q", [
        "how many tests does ARGUS have and are they all passing?",
        "are all the tests passing",
        "what is your test coverage"])
    def test_the_question_is_recognised(self, q: str) -> None:
        from argus.lui.intro import TESTS_Q

        assert TESTS_Q.search(q)

    def test_an_unrelated_question_is_not(self) -> None:
        from argus.lui.intro import TESTS_Q

        assert not TESTS_Q.search("what is the price of BTC")

    def test_the_answer_says_it_did_not_run_them(self) -> None:
        from argus.lui.intro import tests_answer

        lines, _sources, _data = tests_answer()
        assert any("does not run the tests" in line for line in lines)


class TestAMacroQuestionNamingTwoAssets:
    def test_both_names_are_kept(self) -> None:
        from argus.lui.research.parse import detect

        got = detect("If the Fed hikes 100bp tomorrow, what happens to QQQ and BTC?")
        assert got is not None and got.kind is ResearchKind.MACRO
        assert set(got.symbols) >= {"QQQUSDT", "BTCUSDT"}

    def test_the_sensitivity_line_names_its_asset(self) -> None:
        from argus.lui.research.macro import _sensitivity_text

        text = _sensitivity_text("BTCUSDT", {
            "days": 61, "corr_10y": -0.3, "pct_per_10bp": -1.5, "corr_dollar": -0.39})
        assert "BTC" in text and "Correlation, not a cause" in text


class TestANamedCrashIsMeasured:
    @staticmethod
    def _days(prices: list[float]) -> list[object]:
        from argus.market.equity_history import Day

        return [Day(day=date(2020, 2, 18 + i) if i < 11 else date(2020, 3, i - 10), open=p, close=p)
                for i, p in enumerate(prices)]

    def test_the_episode_is_found_by_name(self) -> None:
        from argus.lui.research import episodes

        assert episodes.named("stress a book for a March-2020 style crash").name \
            == "the March 2020 crash"
        assert episodes.named("what about 2008") is not None
        assert episodes.named("a 10% drop") is None

    def test_the_fall_is_peak_to_trough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import episodes

        prices = [100.0, 110.0, 105.0, 90.0, 99.0, 88.0, 95.0]
        monkeypatch.setattr(episodes.equity_history, "daily", lambda _t: self._days(prices))
        episode = episodes.Episode("test", date(2020, 2, 1), date(2020, 3, 31),
                                   episodes.EPISODES[0].pattern)
        got = episodes.nasdaq_fall(episode)
        assert got is not None and round(got[0], 1) == -20.0

    def test_no_history_means_no_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui.research import episodes

        def boom(_t: str) -> list[object]:
            raise RuntimeError("down")

        monkeypatch.setattr(episodes.equity_history, "daily", boom)
        assert episodes.nasdaq_fall(episodes.EPISODES[0]) is None


class TestAnAssumedShockIsCalledAssumed:
    def test_the_word_follows_the_flag(self) -> None:
        from argus.desk.stress_tree import StressInputs

        for flag, word in ((False, "stated"), (True, "assumed")):
            inputs = StressInputs.__new__(StressInputs)
            object.__setattr__(inputs, "assumed", flag)
            assert inputs.word == word


class TestEmojiCarryNoQuestion:
    def test_pictographs_are_stripped_and_the_name_kept(self) -> None:
        from argus.lui.server import PICTOGRAPHS

        assert " ".join(PICTOGRAPHS.sub(" ", "🚀🚀 DOGE 🌕 ??").split())             == "DOGE ??"

    def test_ordinary_text_is_untouched(self) -> None:
        from argus.lui.server import PICTOGRAPHS

        assert PICTOGRAPHS.sub(" ", "what is the price of BTC?") == "what is the price of BTC?"
