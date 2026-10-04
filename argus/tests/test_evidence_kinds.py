"""Typed evidence (build-list 5.2): every live producer declares what its item is, routing reads
the kind, and the record's count of what channel routing handed each analyst is reproducible."""

from __future__ import annotations

import ast
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from argus.agents.selection import DIRECT, KINDS, reads
from argus.eval import evidence_routing as er
from argus.market.skills import PROBES, TOOL_KINDS, Probe, kind_for
from argus.truth.evidence import Evidence, Kind, kind_of

AT = datetime(2026, 10, 5, tzinfo=UTC)
SRC = Path(__file__).resolve().parents[1] / "src" / "argus"


def test_every_skill_tool_has_a_kind_and_an_unknown_one_is_refused() -> None:
    assert {p.tool for p in PROBES} <= set(TOOL_KINDS)
    assert kind_for(next(p for p in PROBES if p.tool == "technical_analysis")) is Kind.TECHNICAL
    with pytest.raises(KeyError, match="no evidence kind"):
        kind_for(Probe("market-intel", "new_tool", "x", {}, "y"))


def test_every_live_producer_declares_a_kind() -> None:
    """A producer that forgets its kind falls back to its channel — the defect this fixes."""
    missing = []
    for folder in ("market", "paper"):
        for path in (SRC / folder).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\bEvidence\(", text):
                call = text[match.end():match.end() + 900]
                if "source=" in call.split("Evidence(")[0] and "kind=" not in call.split(
                        "Evidence(")[0]:
                    missing.append(f"{path.name}:{text[:match.start()].count(chr(10)) + 1}")
    assert not missing, missing


def test_an_untyped_item_routes_by_its_channel_as_before() -> None:
    old = Evidence(id="e", claim="c", source="social", available_at=AT)
    assert kind_of(old) is Kind.SOCIAL and reads("sentiment", old)
    assert kind_of(Evidence(id="e", claim="c", source="elsewhere", available_at=AT)) is None


def test_routing_reads_the_kind_not_the_channel() -> None:
    news_skill = Evidence(id="skill-news_feed-latest-1", claim="headline", source="social",
                          available_at=AT, kind=Kind.NEWS)
    assert reads("event", news_skill) and not reads("sentiment", news_skill)
    assert all(not reads(a, Evidence(id="mkt-X", claim="q", source="news", available_at=AT,
                                     kind=Kind.QUOTE)) for a in KINDS)
    assert Kind.QUOTE in DIRECT


@pytest.mark.parametrize(("item_id", "source", "kind"), [
    ("skill-technical_analysis-rsi-1790431938", "social", Kind.TECHNICAL),
    ("mirror-derivatives_sentiment-long_short-1", "social", Kind.POSITIONING),
    ("mirror-macro_indicators-latest_release-1", "macro", Kind.MACRO),
    ("edgar-0001193125-26-401636", "sec-edgar", Kind.FILING),
    ("edgar-0001193125-26-401637", "filing", Kind.REPORT),
    ("mkt-NVDAUSDT", "news", Kind.QUOTE),
    ("finra-short-NVDA-2026-09-25", "macro", Kind.POSITIONING),
])
def test_a_kept_line_is_typed_by_the_producer_its_id_names(item_id: str, source: str,
                                                            kind: Kind) -> None:
    assert er.kind_from_line(item_id, source) is kind
    assert er.kind_from_line("nobody-1", "news") is None


def test_the_count_over_a_small_record(tmp_path: Path) -> None:
    lines = ["[skill-technical_analysis-rsi-1] (social, credibility 0.75, available x) RSI 70",
             "[twitter-9] (social, credibility 0.40, available x) moon",
             "[mkt-NVDAUSDT] (news, credibility 1.00, available x) last 180"]
    path = tmp_path / "notes.jsonl"
    path.write_text(json.dumps({"seq": "1", "evidence": str(lines)}) + "\n"
                    + json.dumps({"seq": "2", "notes": "[]"}) + "\n", encoding="utf-8")
    assert ast.literal_eval(str(lines)) == lines
    got = er.count(path)
    assert got["decisions"] == 1
    sentiment = got["analysts"]["sentiment"]
    assert sentiment["handed_before"] == 2 and sentiment["handed_now"] == 1
    assert sentiment["no_longer_handed"] == {"technical": 1}
    assert got["analysts"]["event"]["no_longer_handed"] == {"quote": 1}
