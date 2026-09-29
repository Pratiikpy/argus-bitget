"""The fundamentals answer leads with what was asked (`lui/research/fundamentals.py`; stranger QA
of 2026-09-29: sector comparison, analyst counts and yes-or-no dividend questions)."""

from __future__ import annotations

from argus.lui.research.fundamentals import _fundamentals_focus, _lead_with_what_was_asked

TARGETS = ("NVDA reports in 49 days.",
           "Analyst price targets, last 90 days (21 firms, each firm's latest): median 330, range "
           "300 to 515, +44% from the stock's 228.97; 21 buy, 0 hold, 0 sell; 12 raised and 0 cut "
           "their target.")


def test_an_analyst_count_question_leads_with_the_count() -> None:
    lines = _lead_with_what_was_asked(list(TARGETS), "how many analysts rate NVDA a buy")
    assert lines[0] == ("Bottom line: 21 of the 21 firms with a price target in the last 90 "
                        "days rate it a buy, 0 hold and 0 sell.")
    assert lines[2].startswith("Analyst price targets")


def test_a_target_question_still_leads_with_the_targets() -> None:
    lines = _lead_with_what_was_asked(list(TARGETS), "what are analysts' price targets for NVDA")
    assert lines[0].startswith("Bottom line: analyst price targets")


def test_a_yes_or_no_dividend_question_says_yes() -> None:
    lines = ["KO reports in 21 days.", "KO corporate actions: last cash dividend $0.53 a share."]
    assert _fundamentals_focus(lines, "does KO pay a dividend", "KO")[0] == (
        "Bottom line: yes — KO corporate actions: last cash dividend $0.53 a share.")


def test_a_split_alone_is_not_a_dividend() -> None:
    lines = ["TSLA reports in 20 days.", "TSLA corporate actions: last split 3-for-1."]
    lead = _fundamentals_focus(lines, "does TSLA pay a dividend", "TSLA")[0]
    assert lead.startswith("Bottom line: no dividend is on record for TSLA")


def test_a_sector_question_leads_with_the_sector_line() -> None:
    lines = ["Valuation on 2026-09-28: P/E 38.6.", "Against its sector (Technology, XLK): x."]
    lead = _fundamentals_focus(lines, "AAPL's P/E compared to its sector", "AAPL")[0]
    assert lead.startswith("Bottom line: Against its sector")
