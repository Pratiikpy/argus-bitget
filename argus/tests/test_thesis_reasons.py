"""Each reason in a thesis, traced to the typed evidence behind its figures (build-list 5.2)."""

from __future__ import annotations

from argus.eval import thesis_reasons as tr

EVIDENCE = [
    "[xbrl-NVDA-revenue-2026-07-26] (filing, credibility 1.00, available 2026-08-26) revenue "
    "96.22B USD for the quarter",
    "[twitter-1] (social, credibility 0.40, available 2026-09-26) NVDA up 3.5% says everyone",
]


def test_each_sentence_gets_its_own_state_and_kind() -> None:
    thesis = ("Revenue of 96.22B beat the quarter. Social chatter cites a 3.5% move. "
              "The anchor is asleep for 17.8 hours. A 42% jump is coming.")
    got = tr.reason_states(thesis, {"hours_to_discovery": 17.8}, EVIDENCE)
    assert [r["state"] for r in got] == ["traced", "traced", "traced", "untraced"]
    assert got[0]["kinds"] == ["fundamental"]
    assert got[1]["kinds"] == ["social"]
    assert got[2]["kinds"] == ["frame"]


def test_a_reason_without_a_figure_is_counted_not_guessed() -> None:
    got = tr.reason_states("No hedge is available.", {}, EVIDENCE)
    assert got == [{"state": "no figure", "figures": 0, "resolved": 0, "kinds": []}]


def test_sentences_split_on_a_full_stop_before_a_capital() -> None:
    assert tr.sentences("Price is 1.5x book. Volume fell.") == ["Price is 1.5x book.",
                                                               "Volume fell."]
