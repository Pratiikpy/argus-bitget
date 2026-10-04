"""The backtest critic (`research/backtest_critic.py`, build-list 3.2): every planted bug named,
clean code left alone, and the false alarms found on real scripts kept fixed."""

from __future__ import annotations

import pytest

from argus.eval import backtest_critic_comparison as comparison
from argus.lui import server
from argus.research import backtest_critic as critic
from argus.research.backtest_critic import ABSENT, PRESENT, UNCLEAR, review


@pytest.mark.parametrize("fixture", [f for f in comparison.FIXTURES if f.kind],
                         ids=lambda f: f.name)
def test_each_planted_bug_is_named(fixture: comparison.Fixture) -> None:
    report = review(fixture.code)
    check = report.checks[str(fixture.kind)]
    assert check.state == PRESENT, (fixture.name, check)
    assert check.findings[0].why


@pytest.mark.parametrize("fixture", [f for f in comparison.FIXTURES if not f.kind],
                         ids=lambda f: f.name)
def test_clean_code_raises_nothing(fixture: comparison.Fixture) -> None:
    assert review(fixture.code).present == []


class TestRealCodeFalseAlarms:
    """Each found on marketcalls' vectorbt scripts (2026-10-04) and fixed."""

    def test_a_count_is_not_a_level(self) -> None:
        code = ("import pandas as pd\n\ndef f(close, lookback):\n"
                "    mask = close.index >= lookback\n    if mask.sum() == 0:\n        return None\n"
                "    total = mask.sum()\n    if total != 0:\n        return total\n")
        assert review(code).checks["look-ahead"].state == ABSENT

    def test_a_resample_for_a_chart_is_not_a_signal(self) -> None:
        code = ("import pandas as pd\n\ndef plot(pf, equity):\n"
                "    cash = pf.cash().resample('D').last()\n"
                "    invested = equity - cash.reindex(equity.index).ffill()\n"
                "    return invested\n")
        assert review(code).checks["repainting"].state == ABSENT

    def test_an_end_date_of_today_is_read(self) -> None:
        code = ("import pandas as pd\nfrom datetime import datetime\n\n"
                "start = '2019-06-01'\nend = datetime.now().date()\n")
        assert review(code).checks["regime coverage"].state == ABSENT

    def test_a_python_file_using_a_ta_library_is_python(self) -> None:
        code = "import pandas_ta as ta\n\ndef f(df):\n    return ta.rsi(df['close'])\n"
        assert critic.language_of(code) == "python"

    def test_a_backfilled_benchmark_is_named(self) -> None:
        code = ("import pandas as pd\n\ndef bench(df, idx):\n"
                "    return df['close'].reindex(idx).ffill().bfill()\n")
        assert review(code).checks["look-ahead"].state == PRESENT


def test_code_that_does_not_parse_is_unclear_not_clean() -> None:
    report = review("import pandas as pd\ndef f(:\n")
    assert all(c.state == UNCLEAR for c in report.checks.values())


def test_the_traders_words_around_the_code_are_not_code() -> None:
    text = ("can you review my backtest?\nimport pandas as pd\n\ndef f(df):\n"
            "    return df['close'].shift(-1)\nthanks, is it ok?")
    code = critic.extract_code(text)
    assert code.startswith("import pandas") and code.rstrip().endswith("shift(-1)")


def test_the_console_reads_a_pasted_backtest_first() -> None:
    code = "\n".join(f.code for f in comparison.FIXTURES if f.name == "negative shift")
    got = server.handle_ask("can you review my backtest for bias?\n" + code, [])
    assert got["lines"][0].startswith("Bottom line: 1 of 8 checks found a problem")
    assert any(line.startswith("Look-ahead — PRESENT: line") for line in got["lines"])
    # a question about bias with no code is not a code review
    assert critic.lines("what is lookahead bias in a backtest?") is None


def test_against_backtest_truth_on_the_same_fixtures() -> None:
    result = comparison.run()
    assert result["argus_caught"] == result["planted"]
    assert result["argus_false_alarm_fixtures"] == 0
