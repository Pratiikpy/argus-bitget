"""Engines that were OWNED but unreachable from a question, now answering in the console.

An audit (2026-09-24) found five measured capabilities the console never called: the event study,
the path-shape matcher, the Almgren-Chriss schedule, the decision-latency price and the sentiment
integrity layer. These tests pin each to the question that now reaches it, and pin what the answer
may and may not claim. None needs the network.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from argus.lui import research
from argus.lui.research import ResearchKind, detect
from argus.market.depth import Level, OrderBook
from argus.research.event_reactions import aligned, parse_cpi_archive, reaction

# --- the event study ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "how does NVDA react to CPI", "what happens to TSLA after Fed decisions",
    "how did COIN trade after earnings", "does MSTR move on FOMC days",
    "how does QQQ respond to inflation data"])
def test_event_reaction_questions_reach_the_event_study(question: str) -> None:
    request = detect(question)
    assert request is not None and request.kind is ResearchKind.EVENT
    assert research.pattern_reading_wins(request, question)


@pytest.mark.parametrize(("question", "kind"), [
    ("hedge my book before CPI", ResearchKind.HEDGE),
    ("when does NVDA report earnings", ResearchKind.FUNDAMENTALS),
    ("what does a Fed cut do to my book? I hold 50% NVDA, 50% AAPL", ResearchKind.MACRO)])
def test_neighbouring_questions_keep_their_engines(question: str, kind: ResearchKind) -> None:
    request = detect(question)
    assert request is not None and request.kind is kind


def test_the_bls_archive_gives_the_day_each_report_was_actually_published() -> None:
    text = ("[September 2025 Consumer Price Index](https://www.bls.gov/news.release/archives/"
            "cpi_10242025.htm) ... October 2025 Consumer Price Index - Not published because of "
            "2025 lapse ... [August 2025](https://www.bls.gov/news.release/archives/cpi_09112025.htm)")
    days = parse_cpi_archive(text)
    assert [d.isoformat() for d in days] == ["2025-10-24", "2025-09-11"]


def test_the_proxy_uses_whichever_other_names_traded_that_hour() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    hours = [t0 + timedelta(hours=i) for i in range(4)]
    closes = {
        "NVDAUSDT": {h: 100.0 + i for i, h in enumerate(hours)},
        "AAPLUSDT": {h: 50.0 for h in hours},
        "LATEUSDT": {h: 10.0 * (1.1 ** i) for i, h in enumerate(hours) if i >= 2},
    }
    stamps, asset, proxy = aligned("NVDAUSDT", closes)
    assert stamps == hours[1:]
    assert asset[0] == pytest.approx(0.01)
    assert proxy[0] == 0.0                       # only AAPL existed
    assert proxy[2] == pytest.approx((0.0 + 0.1) / 2)   # AAPL and the late listing


def test_too_few_events_is_a_refusal_not_a_figure() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    stamps = [t0 + timedelta(hours=i) for i in range(900)]
    asset = [0.0001 * math.sin(i) for i in range(900)]
    market = [0.0001 * math.cos(i * 1.7) for i in range(900)]
    row = reaction("NVDAUSDT", "FOMC", [t0 + timedelta(hours=700)], stamps, asset, market)
    assert row.average_car_bps is None and row.size_ratio is None
    assert "the study needs 5" in row.verdict


def test_a_macro_day_on_which_the_name_reported_is_left_out_and_named() -> None:
    from argus.research.event_reactions import unconfounded

    fed = [datetime(2025, 10, 29, 18, tzinfo=UTC), datetime(2025, 12, 10, 19, tzinfo=UTC)]
    earnings = [datetime(2025, 10, 29, 20, 5, tzinfo=UTC)]
    clear, close = unconfounded(fed, earnings)
    assert clear == [fed[1]] and close == [fed[0]]


def _report(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    (tmp_path / "event_reactions.json").write_text(json.dumps({
        "generated_at": "2026-09-24T12:00:00+00:00", "reactions": rows}), encoding="utf-8")


def test_a_bigger_move_without_a_direction_says_size_not_direction(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _report(tmp_path, [{
        "symbol": "NVDAUSDT", "kind": "CPI", "events": 9, "average_car_bps": 31.0,
        "size_ratio": 1.6, "verdict": "NO EFFECT ESTABLISHED across 9 event(s).",
        "tests": {"patell": {"adjusted_p_value": 0.41}}, "dates": ["2025-11-13", "2025-12-18"]}])
    monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(research, "_event_lines", lambda *a, **k: ([], None))
    lines, sources = research._event_reaction("NVDAUSDT", "how does NVDA react to CPI")
    assert lines[0].startswith("Actionable: expect a bigger move than usual, not a direction")
    assert "1.6x its ordinary 24 hours" in lines[0]
    assert "no reliable direction" in lines[1]
    assert sources and sources[0].ref == "argus.research.event_reactions"


def test_a_fund_has_no_earnings_to_react_to(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _report(tmp_path, [{"symbol": "QQQUSDT", "kind": "CPI", "events": 5, "average_car_bps": 1.0,
                        "size_ratio": 0.4, "verdict": "NO EFFECT", "tests": {}, "dates": []}])
    monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
    lines, _ = research._event_reaction("QQQUSDT", "how does QQQ trade after earnings")
    assert lines == ["QQQ files no earnings of its own, so there is no earnings reaction to "
                     "measure; ask how it reacts to CPI or Fed decisions."]


# --- the schedule and the cost of waiting --------------------------------------------------------


def _book(mid: float = 100.0, spread: float = 0.02, levels: int = 50) -> OrderBook:
    bids = tuple(Level(Decimal(str(round(mid - spread / 2 - 0.01 * i, 4))), Decimal(40))
                 for i in range(levels))
    asks = tuple(Level(Decimal(str(round(mid + spread / 2 + 0.01 * i, 4))), Decimal(40))
                 for i in range(levels))
    return OrderBook(symbol="NVDAUSDT", fetched_at=datetime.now(UTC), bids=bids, asks=asks)


def _candles(monkeypatch: pytest.MonkeyPatch, step: float = 0.003) -> None:
    import argus.market.history as history

    bars = [SimpleNamespace(close=Decimal(str(100 * (1 + step * math.sin(i))))) for i in range(49)]
    monkeypatch.setattr(history, "fetch", lambda *a, **k: bars)


def test_the_schedule_is_the_almgren_chriss_optimum_priced_against_an_even_split(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _candles(monkeypatch)
    monkeypatch.setattr(research, "_session_change", lambda start, hours: None)
    lines = research._optimal_schedule("NVDAUSDT", Decimal(200_000), Decimal(20_000_000),
                                       _book(), 6.0, 3.0, False)
    assert lines[0].startswith("Schedule (Almgren-Chriss optimum, inventory decaying e-fold")
    assert "for an even split" in lines[0] and "live book" in lines[0]
    shares = [int(p.rstrip("%")) for p in
              lines[0].split("trade ", 1)[1].split(" of $", 1)[0].split(", ")]
    assert shares == sorted(shares, reverse=True) and shares[0] > shares[-1]
    assert lines[1].startswith("Waiting costs too:")


def test_urgency_front_loads_harder(monkeypatch: pytest.MonkeyPatch) -> None:
    _candles(monkeypatch)
    monkeypatch.setattr(research, "_session_change", lambda start, hours: None)

    def first(urgent: bool) -> int:
        line = research._optimal_schedule("NVDAUSDT", Decimal(200_000), Decimal(20_000_000),
                                          _book(), 6.0, 3.0, urgent)[0]
        return int(line.split("trade ", 1)[1].split("%", 1)[0])

    assert first(True) > first(False)


def test_the_schedule_stops_at_the_session_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    _candles(monkeypatch)
    monkeypatch.setattr(research, "_session_change", lambda start, hours: (2, "opens"))
    lines = research._optimal_schedule("NVDAUSDT", Decimal(1_000_000), Decimal(20_000_000),
                                       _book(), 6.0, 12.0, False)
    session = next(line for line in lines if line.startswith("Session:"))
    assert "opens in about 2 hour(s)" in session
    assert "plan the remaining" in session
    assert "over 2 hour(s)" in lines[0]


def test_waiting_cost_is_volatility_over_the_square_root_of_time() -> None:
    line = research._waiting_cost([0.0025, -0.0025] * 24)
    one = float(line.split("expected ", 1)[1].split("bps", 1)[0])
    ten = float(line.split(", ", 1)[1].split("bps", 1)[0])
    assert one == pytest.approx(25 / math.sqrt(60), abs=0.1)
    assert ten == pytest.approx(one * math.sqrt(10), abs=0.15)


# --- the path-shape matcher ----------------------------------------------------------------------


def test_the_analogue_answer_carries_the_path_match_and_its_null() -> None:
    fixture = Path(__file__).resolve().parents[1] / "data" / "risk_layer_candles_fixture.json"
    rows = json.loads(fixture.read_text(encoding="utf-8"))["candles"]["COINUSDT"]
    closes = [(datetime.fromisoformat(t), float(c)) for t, c in rows]
    found = research._shape_line(closes, "COINUSDT", 90)
    assert found is not None
    text, payload = found
    assert text.startswith("Path match")
    assert payload["null_trials"] == 50
    if not payload["has_precedent"]:
        assert "not distinguishable from chance" in text or "too far" in text


@pytest.mark.parametrize("question", ["when did TSLA last look like this",
                                      "has NVDA ever looked like this",
                                      "same chart shape on COIN before?"])
def test_path_questions_reach_the_analogue_engine(question: str) -> None:
    request = detect(question)
    assert request is not None and request.kind is ResearchKind.ANALOGUE


# --- the personal mandate ------------------------------------------------------------------------


def test_no_mandate_is_applied_when_none_is_stated() -> None:
    assert research.stated_profile("should I buy 10% MSTR") is None


def test_a_preset_named_in_words_and_limits_stated_in_numbers() -> None:
    conservative = research.stated_profile("I'm a conservative investor, should I add NVDA")
    assert conservative is not None and conservative.max_position_pct == Decimal(5)
    own = research.stated_profile("no more than 10% in any name and I can't lose more than 6%")
    assert own is not None and own.name == "your stated mandate"
    assert own.max_position_pct == Decimal(10) and own.loss_tolerance_pct == Decimal(6)
    mixed = research.stated_profile("aggressive trader, my horizon is 2 weeks")
    assert mixed is not None and mixed.holding_horizon_hours == 336
    assert mixed.name.startswith("aggressive event trader")


def _falling(symbol: str, drop: float) -> dict[str, dict[datetime, float]]:
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    returns = [0.001] * 100 + [drop / 24] * 24 + [0.001] * 50
    return {symbol: {t0 + timedelta(hours=i): r for i, r in enumerate(returns)}}


def test_two_mandates_disagree_on_the_identical_trade() -> None:
    raw = _falling("COINUSDT", -0.10)
    lines = research._mandate_lines("COINUSDT", 0.10, raw, "I'm an aggressive trader")
    assert lines[0].startswith("Actionable: for your mandate (aggressive event trader")
    assert "yes to a 10% ($10,000) COIN position" in lines[0]
    assert lines[1].startswith("The same trade under a conservative income mandate: no")
    assert "the mandate, not the model, decides" in lines[1]


def test_the_bad_case_is_the_names_own_worst_day() -> None:
    worst = research._worst_day_pct("COINUSDT", _falling("COINUSDT", -0.12))
    assert Decimal("11") < worst < Decimal("12")


def test_a_resize_prints_money_not_bare_decimals() -> None:
    raw = _falling("COINUSDT", -0.05)
    lines = research._mandate_lines("COINUSDT", 0.30, raw, "aggressive trader")
    assert "but at $25,000 rather than the 30% ($30,000) COIN position asked" in lines[0]
    assert "30000" not in lines[0] and "$100,000 of capital" in lines[0]


# --- how a decision was reached ------------------------------------------------------------------


def test_a_decision_is_explained_from_the_notes_the_desk_wrote(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.answer import _how_it_was_reached

    notes = ["[quarantine] 26 evidence item(s) screened, none withheld",
             "[panel] 1 of 3 analysts run: sentiment",
             "[panel] 1.514bps of deliberation not spent, against a 12bps round trip",
             "[panel] cross_asset also ran, outside the evidence-cost selection above",
             "panel: 2 analysts, 2 distinct sources, independence 1.00 -> bearish at 0.25 "
             "after provenance discount",
             "[memory] 8 prior decision(s) on SQQQUSDT shown to the PM, 8 of them graded",
             "[grounding] 1 of 9 figure(s) do not resolve to anything the desk was given: 8",
             "[grounding] an unattributable number in a thesis is the easiest place",
             "ENTITY GATE — 2 instrument(s) named, 0 unsupported",
             "[protocol] within the committed protocol; nothing changed"]
    (tmp_path / "desk_notes.jsonl").write_text(
        json.dumps({"seq": 627, "symbol": "SQQQUSDT", "notes": notes}), encoding="utf-8")
    monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
    lines = _how_it_was_reached(627)
    assert lines[0] == "How the desk reached decision 627, from the notes it wrote at the time:"
    text = "\n".join(lines)
    assert "Panel: 1.514bps of deliberation not spent" in text
    assert "cross_asset" not in text
    assert "Agreement, discounted for shared sources: 2 analysts, 2 distinct sources" in text
    assert text.count("Grounding check:") == 1
    assert "Entity check: 2 instrument(s) named, 0 unsupported." in text
    assert _how_it_was_reached(1) == []


def test_the_stress_band_uses_the_frozen_head_to_head_scale() -> None:
    import json as _json

    report = _json.loads((Path(__file__).resolve().parents[1] / "data"
                          / "analogstress_comparison.json").read_text(encoding="utf-8"))
    chosen = report["after_absorbing"]["selection"]["chosen"]
    frozen = report["predictors"][chosen]["frozen_scale"]
    assert abs(research.STRESS_BLEND_SCALE - frozen) < 1e-4
