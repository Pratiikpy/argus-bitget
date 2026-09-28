"""A changed mind is shown, not silently applied (research/harvest/04-mem0.md)."""

from __future__ import annotations

from typing import Any


class TestAChangedMindIsShownNotSilentlyApplied:
    """A replaced fact keeps what it replaced (research/harvest/04-mem0.md, decision 1)."""

    @staticmethod
    def _fact(text: str, value: str, at: str) -> Any:
        from argus.lui.memory import Fact

        return Fact(kind="style", subject="", value=value, text=text, at=at)

    def test_a_replacement_records_the_earlier_words_and_date(self) -> None:
        from argus.lui.memory import merge, remembered_line

        old = [self._fact("I'm conservative", "conservative", "2026-09-10")]
        [now] = merge(old, [self._fact("I'm aggressive", "aggressive", "2026-09-24")])
        assert now.replaces == "“I'm conservative” (2026-09-10)"
        line = remembered_line(now, "sizing uses it")
        assert "replacing “I'm conservative” (2026-09-10)" in line

    def test_repeating_the_same_statement_keeps_the_record(self) -> None:
        from argus.lui.memory import merge

        first = merge([self._fact("I'm conservative", "conservative", "2026-09-10")],
                      [self._fact("I'm aggressive", "aggressive", "2026-09-24")])
        again = merge(first, [self._fact("I'm aggressive", "aggressive", "2026-09-25")])
        assert again[0].replaces == "“I'm conservative” (2026-09-10)"

    def test_a_first_statement_replaces_nothing(self) -> None:
        from argus.lui.memory import merge

        [fact] = merge([], [self._fact("I'm aggressive", "aggressive", "2026-09-24")])
        assert fact.replaces == ""

    def test_it_survives_the_browser_round_trip_and_is_bounded(self) -> None:
        from argus.lui.memory import MAX_TEXT, dumps, merge, parse

        merged = merge([self._fact("I'm conservative", "conservative", "2026-09-10")],
                       [self._fact("I'm aggressive", "aggressive", "2026-09-24")])
        assert parse(dumps(merged))[0].replaces == merged[0].replaces
        forged = '[{"kind": "style", "value": "x", "text": "t", "at": "2026-09-24", ' \
                 '"replaces": "' + "a" * 5000 + '"}]'
        assert len(parse(forged)[0].replaces) == MAX_TEXT + 20

    def test_the_acknowledgement_says_what_changed(self) -> None:
        from argus.lui.memory import acknowledgement, merge

        merged = merge([self._fact("I'm conservative", "conservative", "2026-09-10")],
                       [self._fact("I'm aggressive", "aggressive", "2026-09-24")])
        assert "replacing “I'm conservative” (2026-09-10)" in acknowledgement(merged)[0]


def test_two_views_in_one_message_are_two_theses() -> None:
    """The multi-topic case from research/harvest/04-mem0.md: until 2026-09-28 the first view's
    claim ran to the end of the message and the second was never kept."""
    from argus.lui.memory import extract

    facts = extract("I think NVDA will rise and I think BTC will crash, I never trade MSTR, "
                    "I don't touch COIN.")
    got = {(f.kind, f.subject, f.value) for f in facts}
    assert got == {("thesis", "NVDAUSDT", "bull"), ("thesis", "BTCUSDT", "bear"),
                   ("avoid", "MSTRUSDT", "avoid"), ("avoid", "COINUSDT", "avoid")}
