"""A question for one side of the news is answered with that side (`lui/research/news.py`;
stranger QA of 2026-09-29: "any negative news on NVDA" got the same list as any news question)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from argus.lui.research import news


@pytest.mark.parametrize(("text", "tone"), [
    ("any negative news on NVDA this week", "negative"),
    ("is there bad news on TSLA", "negative"),
    ("any good news for AAPL", "positive"),
    ("bullish headlines on COIN?", "positive"),
    ("what is the news on NVDA", None),
    ("why is NVDA down", None),
])
def test_the_asked_side_is_read_from_the_question(text: str, tone: str | None) -> None:
    assert news.tone_asked(text) == tone


@dataclass
class _Headline:
    title: str
    feed: str
    link: str
    published: datetime


def test_the_asked_side_leads_and_the_lexicons_limit_is_said() -> None:
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)
    kept = [_Headline("NVDA shares crash after terrible guidance", "cnbc", "u1",
                      now - timedelta(hours=2)),
            _Headline("NVDA wins huge new contract, great results", "cnbc", "u2",
                      now - timedelta(hours=3)),
            _Headline("NVDA holds its annual meeting", "cnbc", "u3", now - timedelta(hours=4))]
    found = news._tone_lines(kept, "NVDA", "negative", now)
    assert found is not None
    lines, payload = found
    assert lines[0].startswith("Bottom line: of 3 headline(s) naming NVDA")
    assert "1 read negative" in lines[0] and "scores wording" in lines[0]
    assert lines[1].startswith("Negative (") and "crash" in lines[1]
    assert payload == {"asked": "negative", "scorer": "VADER compound", "on_side": 1,
                       "scored": 3}


def test_no_headline_on_the_asked_side_is_said_plainly() -> None:
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)
    kept = [_Headline("NVDA holds its annual meeting", "cnbc", "u3", now)]
    found = news._tone_lines(kept, "NVDA", "negative", now)
    assert found is not None
    assert "0 read negative" in found[0][0] and "none do" in found[0][0]

