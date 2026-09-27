"""The CPI and news answers ask bitget-signal first and say which source spoke (audit 109)."""

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import research, skillroute
from argus.market.skills import Health


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skillroute, "_DOWN", {})


def _route_with(reply: Any) -> Any:
    """``research.skill_route`` with the Skill replying ``reply`` (an exception to raise, or a
    payload) and the answer's own mirror."""
    def route(tool: str, action: str, args: Any = None, **kwargs: Any) -> Any:
        def skill(tool: str, call_args: dict[str, Any], wait: float) -> tuple[Any, str]:
            if isinstance(reply, Exception):
                raise reply
            return reply, "ok"

        kwargs.pop("skill_call", None)
        return skillroute.route(tool, action, args, skill_call=skill, **kwargs)

    return route


def _fred(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = {f"{y}-{m:02d}-01": 300.0 + (y - 2025) * 9 + m * 0.7
            for y in (2025, 2026) for m in range(1, 13) if (y, m) <= (2026, 8)}
    monkeypatch.setattr(research.macro, "_fred", lambda series, days=45: sorted(rows.items()))


def test_cpi_names_fred_as_the_skills_source_when_the_skill_is_hollow(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _fred(monkeypatch)
    for _module in (research.macro, research.news, research.sentiment):
        monkeypatch.setattr(_module, "skill_route", _route_with({"error": ""}))
    lines, sources = research.macro._cpi_lines()
    assert lines[0].startswith("Bottom line: US CPI for Aug 2026:")
    assert "read because the Skill answered with no data" in sources[0].detail
    assert "macro_indicators names" in sources[0].detail
    assert len(sources) == 1


def test_cpi_credits_the_skill_when_it_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    _fred(monkeypatch)
    for _module in (research.macro, research.news, research.sentiment):
        monkeypatch.setattr(_module, "skill_route",
                            _route_with({"indicator": "cpi", "value": 322.1, "period": "2026-08"}))
    lines, sources = research.macro._cpi_lines()
    assert any("macro-analyst Skill answered the same release" in line for line in lines)
    assert sources[-1].ref == "bitget-signal macro_indicators.latest_release"
    assert "read because" not in sources[0].detail


def test_skill_headlines_show_only_titled_items_not_already_shown() -> None:
    payload = [
        {"feed": "cnbc", "error": "", "items": [
            {"title": "Nvidia beats", "link": "https://x/1"},
            {"summary": "no title here"},
            {"title": "Already shown"}]},
        {"feed": "fed", "error": "", "items": []},
        "not a feed",
    ]
    lines = research.news._skill_headlines(payload, {"Already shown"})
    assert lines == ["bitget-signal news_feed (cnbc): Nvidia beats https://x/1"]


def test_skill_headlines_stop_at_three() -> None:
    payload = [{"feed": "f", "items": [{"title": f"t{i}"} for i in range(6)]}]
    assert len(research.news._skill_headlines(payload, set())) == 3


def test_the_news_mirror_says_the_answer_read_its_own_feeds() -> None:
    health, _detail, payload, upstream, same = research.news._own_feeds_are_the_mirror("x", {})
    assert health is Health.OK and payload is None and same
    assert upstream == "publishers' RSS feeds"
