"""The Skill-first route (`lui/skillroute.py`): which of bitget-signal or its mirror answered, and
that every answer says so. Fakes stand in for both, so nothing here touches the network."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from argus.lui import skillroute
from argus.lui.research import _skill_btc_line, _skill_fear_greed
from argus.market.skills import Health


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    skillroute.reset()
    yield
    skillroute.reset()


def _skill(payload: Any, status: str = "bitget:x: ok") -> Any:
    calls: list[str] = []

    def call(tool: str, args: dict[str, Any], wait: float) -> tuple[Any, str]:
        calls.append(tool)
        return payload, status

    call.calls = calls  # type: ignore[attr-defined]
    return call


def _mirror(payload: Any, health: Health = Health.OK) -> Any:
    def call(ident: str, args: dict[str, Any]) -> tuple[Health, str, Any, str, bool]:
        return health, "ok", payload, "alternative.me Fear & Greed", True

    return call


def test_a_skill_that_answers_is_credited_to_the_skill() -> None:
    routed = skillroute.route("sentiment_index", "current",
                              skill_call=_skill({"value": 61, "classification": "Greed"}),
                              mirror_call=_mirror([{"value": "50"}]))
    assert routed.via == "skill"
    assert routed.skill == "sentiment-analyst"
    assert routed.source().ref == "bitget-signal sentiment_index.current"
    assert "sentiment-analyst Skill" in routed.source().detail


def test_an_error_envelope_is_not_an_answer_and_the_mirror_is_credited_as_itself() -> None:
    routed = skillroute.route("sentiment_index", "current",
                              skill_call=_skill({"alt_me_error": ""}),
                              mirror_call=_mirror([{"value": "50"}]))
    assert routed.via == "mirror"
    assert routed.skill_health is Health.EMPTY
    source = routed.source()
    assert source.ref == "alternative.me Fear & Greed"
    assert "the source bitget-signal's sentiment_index.current names" in source.detail
    assert "the Skill answered with no data" in source.detail


def test_a_dark_skill_is_not_asked_again_while_it_cools_down() -> None:
    clock = [1000.0]
    first = _skill(None, "bitget:sentiment_index: unavailable (TimeoutError)")
    routed = skillroute.route("sentiment_index", "current", skill_call=first,
                              mirror_call=_mirror([{"value": "50"}]), now=lambda: clock[0])
    assert routed.skill_health is Health.TIMEOUT
    clock[0] += 60
    again = _skill({"value": 61})
    reused = skillroute.route("sentiment_index", "current", skill_call=again,
                              mirror_call=_mirror([{"value": "50"}]), now=lambda: clock[0])
    assert again.calls == []
    assert reused.reused
    assert reused.via == "mirror"
    assert "when last asked at" in reused.source().detail
    clock[0] += skillroute.COOLDOWN_S
    asked = skillroute.route("sentiment_index", "current", skill_call=again,
                             mirror_call=_mirror([{"value": "50"}]), now=lambda: clock[0])
    assert again.calls == ["sentiment_index"]
    assert asked.via == "skill"


def test_when_neither_answers_nothing_is_presented() -> None:
    routed = skillroute.route("sentiment_index", "current",
                              skill_call=_skill(None, "bitget:x: tool reported an error (boom)"),
                              mirror_call=_mirror(None, Health.UNAVAILABLE))
    assert routed.via == "none"
    assert routed.payload is None
    assert not routed.answered


def test_a_tool_no_skill_names_is_labelled_as_such() -> None:
    routed = skillroute.route("crypto_derivatives", "ticker_24h",
                              skill_call=_skill({"last": 84000.0}), mirror_call=_mirror(None))
    assert routed.skill == "server"
    assert "no SKILL.md names" in routed.source().detail


@pytest.mark.parametrize(
    ("payload", "reading"),
    [
        ({"value": "61", "value_classification": "Greed"},
         {"value": 61, "value_classification": "Greed"}),
        ({"data": [{"value": 40, "classification": "Fear"}]},
         {"value": 40, "value_classification": "Fear"}),
        ({"current": {"value": 12.0}}, {"value": 12, "value_classification": ""}),
        ({"alt_me_error": ""}, None),
        ("text", None),
    ],
)
def test_the_skills_fear_and_greed_is_read_or_refused(payload: Any, reading: Any) -> None:
    assert _skill_fear_greed(payload) == reading


class _Done:
    def __init__(self, routed: Any) -> None:
        self._routed = routed

    def result(self, timeout: float | None = None) -> Any:
        return self._routed


class _Ticker:
    def __init__(self, last: float) -> None:
        self.last = last


def test_the_btc_cross_check_agrees_or_says_it_does_not() -> None:
    routed = skillroute.route("crypto_derivatives", "ticker_24h",
                              skill_call=_skill({"last": 84000.0, "change_pct": 0.2}),
                              mirror_call=_mirror(None))
    line, source = _skill_btc_line(_Done(routed), _Ticker(84050.0)) or ("", None)
    assert "+6.0bp basis — the Skill and Bitget's own ticker agree" in line
    assert source is not None
    assert source.ref == "bitget-signal crypto_derivatives.ticker_24h"
    far, _ = _skill_btc_line(_Done(routed), _Ticker(86000.0)) or ("", None)
    assert "one of the two readings is stale" in far
    dark = skillroute.Routed("crypto_derivatives", "ticker_24h", "server", Health.TIMEOUT, "",
                             "none", None)
    assert _skill_btc_line(_Done(dark), _Ticker(84000.0)) is None
