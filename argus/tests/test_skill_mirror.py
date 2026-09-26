"""The Skill mirror: dark Bitget Skills answered from their own upstreams, never passed off as the
Skill, and never inventing a reading when the upstream fails too."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from argus.lui import status_page
from argus.market import skill_mirror as sm
from argus.market.skills import PROBES, Health, SkillReport, ToolHealth

AT = datetime(2026, 9, 25, 12, tzinfo=UTC)
BY_IDENT = {p.ident: p for p in PROBES}


def _report(answered: set[str]) -> SkillReport:
    rows = tuple(ToolHealth(p, Health.OK if p.ident in answered else Health.EMPTY, "x",
                            {"rsi": 40.0} if p.ident in answered else None, AT.isoformat())
                 for p in PROBES)
    return SkillReport("NVDAUSDT", AT.isoformat(), rows)


@pytest.fixture(autouse=True)
def _no_cache() -> None:
    sm._CACHE.clear()


def _stub_all(monkeypatch: pytest.MonkeyPatch, fetch: Any) -> None:
    table = {k: sm.Mirror(v.upstream, v.same_upstream, fetch) for k, v in sm.MIRRORS.items()}
    monkeypatch.setattr(sm, "MIRRORS", table)


def test_every_non_technical_probe_has_a_mirror_and_technicals_have_none() -> None:
    for probe in PROBES:
        assert (probe.ident in sm.MIRRORS) == (probe.tool != "technical_analysis"), probe.ident
    assert set(sm.MIRRORS) <= set(BY_IDENT)


def test_substitutes_are_the_two_the_skill_cannot_be_matched_on() -> None:
    subs = {k for k, m in sm.MIRRORS.items() if not m.same_upstream}
    assert subs == {"tradfi_news.earnings", "network_status.eth_gas"}


def test_fill_keeps_bitget_rows_and_mirrors_only_the_dark_ones(monkeypatch: pytest.MonkeyPatch
                                                               ) -> None:
    calls: list[str] = []

    def fetch(args: Any) -> dict[str, Any]:
        calls.append(str(args.get("action")))
        return {"value": 71}

    _stub_all(monkeypatch, fetch)
    tech = {p.ident for p in PROBES if p.tool == "technical_analysis"}
    rows = sm.fill(_report(tech))
    assert len(rows) == len(PROBES)
    for row in rows:
        if row.probe.ident in tech:
            assert row.bitget is Health.OK and row.mirror is None
            assert row.upstream == "Bitget Skill"
        else:
            assert row.bitget is Health.EMPTY and row.mirror is Health.OK
    assert len(calls) == len(PROBES) - len(tech)
    s = sm.summary(rows)
    assert (s["answered_by_bitget"], s["answered_by_mirror"], s["answered_total"]) == (6, 13, 19)
    assert s["answered_by_mirror_same_upstream"] == 11
    assert s["skills_reached_through_bitget"] == ["technical-analysis"]
    assert len(s["skills_answered_in_total"]) == 5


def test_a_failing_upstream_is_unavailable_not_a_neutral_reading(monkeypatch: pytest.MonkeyPatch
                                                                  ) -> None:
    def fetch(_: Any) -> dict[str, Any]:
        raise sm.MirrorError("api.alternative.me answered HTTP 503")

    _stub_all(monkeypatch, fetch)
    health, detail, payload, _ = sm.mirror_one(BY_IDENT["sentiment_index.current"], {})
    assert health is Health.UNAVAILABLE and payload is None and "503" in detail


def test_a_hollow_upstream_answer_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_all(monkeypatch, lambda _: {"t10y": {"error": ""}, "spread": 0.0})
    health, _, payload, _ = sm.mirror_one(BY_IDENT["rates_yields.yield_curve"], {})
    assert health is Health.EMPTY and payload is None


def test_only_successes_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter([sm.MirrorError("down"), {"value": 1}, {"value": 2}])

    def fetch(_: Any) -> dict[str, Any]:
        got = next(outcomes)
        if isinstance(got, Exception):
            raise got
        return got

    _stub_all(monkeypatch, fetch)
    probe = BY_IDENT["sentiment_index.current"]
    assert sm.mirror_one(probe, {})[0] is Health.UNAVAILABLE
    assert sm.mirror_one(probe, {})[2] == {"value": 1}
    assert sm.mirror_one(probe, {})[2] == {"value": 1}  # cached, third fetch never made


def test_evidence_names_the_source_and_skips_bitget_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_all(monkeypatch, lambda _: {"value": 71})
    tech = {p.ident for p in PROBES if p.tool == "technical_analysis"}
    out = sm.evidence(sm.fill(_report(tech)), as_of=AT)
    assert len(out) == 13
    fng = next(e for e in out if "sentiment_index" in e.id)
    assert "Bitget server dark" in fng.claim and "alternative.me" in fng.claim
    assert "the Skill's own upstream" in fng.claim
    gas = next(e for e in out if "network_status" in e.id)
    assert "substitute" in gas.claim and gas.credibility < fng.credibility


def test_fill_missing_skips_what_the_sweep_measured_as_answering(monkeypatch: pytest.MonkeyPatch
                                                                 ) -> None:
    _stub_all(monkeypatch, lambda _: {"value": 1})
    rows = sm.fill_missing({"sentiment_index.current", "technical_analysis.rsi"},
                           symbol="NVDAUSDT")
    assert "sentiment_index.current" not in {r.probe.ident for r in rows}
    assert len(rows) == 12


def test_pearson_and_anchor() -> None:
    assert sm.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert sm.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert sm.pearson([1, 1, 1], [1, 2, 3]) is None
    assert sm.anchor_symbol("NVDAUSDT") == "NVDA" and sm.anchor_symbol("spy") == "SPY"


def test_fred_csv_parsing_drops_missing_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"observation_date,DGS10\n2026-09-18,4.9\n2026-09-19,.\n2026-09-22,4.96\n"
    monkeypatch.setattr(sm, "_get", lambda *a, **k: body)
    assert sm.fred_series("DGS10") == [(date(2026, 9, 18), 4.9), (date(2026, 9, 22), 4.96)]


def test_fred_with_no_rows_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sm, "_get", lambda *a, **k: b"observation_date,DGS10\n")
    with pytest.raises(sm.MirrorError):
        sm.fred_series("DGS10")


def test_rss_items_are_read_and_bad_xml_raises() -> None:
    rss = (b"<rss><channel><item><title>BTC ETF flows</title><link>https://x/a</link>"
           b"<pubDate>Thu</pubDate></item><item><title></title></item></channel></rss>")
    assert sm._items("coindesk", rss, 5) == [
        {"feed": "coindesk", "title": "BTC ETF flows", "link": "https://x/a", "published": "Thu"}]
    with pytest.raises(sm.MirrorError):
        sm._items("coindesk", b"<rss><oops", 5)


def test_status_page_states_both_counts(tmp_path: Any) -> None:
    (tmp_path / "skill_mirror.json").write_text(
        '{"checked_at": "2026-09-25T00:00:00+00:00", "tools_probed": 19, '
        '"answered_by_bitget": 6, "answered_by_mirror": 13, '
        '"answered_by_mirror_same_upstream": 11, "answered_total": 19, '
        '"skills_answered_in_total": ["a", "b", "c", "d", "e"]}', encoding="utf-8")
    lines = dict(status_page.sweep_lines(tmp_path))
    text = lines["bitget-signal, covered"]
    assert "of the desk's 19 calls, Bitget's server answered 6" in text
    assert "11 the same source, 2 a named substitute" in text
    # Bitget's answers and ARGUS's own reads are never summed into one Skill count.
    assert "labelled as that source, never as the Skill" in text
    assert "of 5 Skills" not in text
