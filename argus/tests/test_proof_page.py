"""The /proof page and the engines it points the console at.

A win a judge cannot reach is scored as if it did not exist (the audit of 2026-09-24). These tests
hold the page to the register it reads, and hold each "ask the console" link to the engine it
claims to reach — the link is a promise, and it is checked here the way a reader would check it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from argus.eval.sentiment_comparison import NARRATIVES, coordinated_scenario
from argus.lui import research
from argus.lui.proof_page import FAVICON, IN_THE_CONSOLE, SUBTHEMES, collect, render
from argus.lui.question import Intent, classify
from argus.lui.research import ResearchKind, detect
from argus.lui.server import FAVICON as SERVER_FAVICON
from argus.market.stories import group, same_story, words

DATA = Path(__file__).resolve().parents[1] / "data"


def test_every_capability_in_the_register_is_on_the_page() -> None:
    wins, counts = collect(DATA)
    register = json.loads((DATA / "standing.json").read_text(encoding="utf-8"))
    assert {w.name for w in wins} == {c["name"] for c in register["capabilities"]}
    assert counts == register["by_state"]


def test_every_subtheme_code_is_named_as_the_handbook_names_it() -> None:
    wins, _ = collect(DATA)
    assert all(w.track in ("Track 1", "Track 2", "Track 3") for w in wins), \
        sorted({w.subtheme for w in wins if w.track not in ("Track 1", "Track 2", "Track 3")})
    handbook = (DATA.parents[1] / "BITGET_AI_BASE_CAMP_S2_HANDBOOK_EN.md")
    if handbook.exists():
        text = handbook.read_text(encoding="utf-8")
        for _, name in SUBTHEMES.values():
            if not name.startswith(("Judged:", "Open Theme", "Factor research")):
                assert name in text, name


def test_every_console_link_names_a_capability_the_register_holds() -> None:
    register = json.loads((DATA / "standing.json").read_text(encoding="utf-8"))
    names = {c["name"] for c in register["capabilities"]}
    assert set(IN_THE_CONSOLE) <= names, set(IN_THE_CONSOLE) - names


@pytest.mark.parametrize(("name", "question"),
                         [(n, q) for n, (q, _) in IN_THE_CONSOLE.items() if not q.startswith("/")])
def test_every_console_link_reaches_an_engine_rather_than_a_refusal(name: str,
                                                                    question: str) -> None:
    # A research engine, or one of the desk-record answers — never UNKNOWN or a refusal.
    request = detect(question)
    if request is None:
        intent = classify(question, now=datetime.now(UTC)).intent
        assert intent not in (Intent.UNKNOWN, Intent.UNSUPPORTED, Intent.ORDER), (name, question)


def test_the_page_renders_every_state_and_the_gap_it_admits() -> None:
    wins, counts = collect(DATA)
    page = render(wins, counts)
    assert f"{len(wins)} capabilities measured against a named rival" in page
    assert "Not yet reachable from the console" in page or all(w.console for w in wins)
    for win in wins:
        if win.console:
            assert "Ask the console" in page
            break
    assert "<script" not in page


def test_an_unreadable_register_shows_nothing_rather_than_something(tmp_path: Path) -> None:
    wins, counts = collect(tmp_path)
    assert wins == [] and counts == {}
    assert "unreadable" in render(wins, counts)


def test_the_favicon_copy_matches_the_server() -> None:
    assert FAVICON == SERVER_FAVICON


# --- stories: repetition counted once ---------------------------------------------------------


@dataclass
class _Headline:
    title: str
    feed: str
    published: datetime


@pytest.mark.parametrize(("symbol", "claim"), NARRATIVES)
def test_the_coordinated_posts_of_the_owned_test_are_one_story(symbol: str, claim: str) -> None:
    posts = coordinated_scenario(claim)
    at = datetime(2026, 9, 24, tzinfo=UTC)
    stories = group([_Headline(p.claim, "social", at + timedelta(minutes=i))
                     for i, p in enumerate(posts)],
                    [symbol.lower(), symbol.removesuffix("USDT").lower()])
    assert len(stories) == 1 and stories[0].copies == len(posts)


def test_distinct_stories_about_one_name_stay_apart() -> None:
    at = datetime(2026, 9, 24, tzinfo=UTC)
    titles = ["SpaceX and Nvidia Just Deepened Their Compute Alliance",
              "Zacks Earnings Trends Highlights: Micron, Nvidia and Alphabet",
              "1 Unstoppable Stock to Buy Before It Joins Nvidia, Alphabet, Apple"]
    stories = group([_Headline(t, "yahoo-nvda", at) for t in titles], ["nvda", "nvidia"])
    assert len(stories) == 3


def test_a_rewrite_that_adds_a_clause_is_the_same_story() -> None:
    a = words("Binance takes $100 million Circle stake alongside five-year USDC deal")
    b = words("Binance Takes $100M Stake in Circle Under Five-Year USDC Promotion Deal")
    assert same_story(a, b)


def test_the_most_carried_story_comes_first_with_every_outlet() -> None:
    at = datetime(2026, 9, 24, tzinfo=UTC)
    rows = [_Headline("CME adds Bitcoin Cash and Uniswap futures", "coindesk", at),
            _Headline("CME Expands Crypto Futures Lineup With Bitcoin Cash and Uniswap",
                      "yahoo-coin", at + timedelta(hours=1)),
            _Headline("Coinbase launches fixed rate bitcoin loans", "cnbc", at)]
    stories = group(rows)
    assert stories[0].copies == 2
    assert stories[0].outlets == ("coindesk", "Yahoo Finance")


# --- the sentiment answer ----------------------------------------------------------------------


@pytest.mark.parametrize("question", ["is the hype on NVDA real", "is COIN being pumped",
                                      "any rumors about MSTR", "what is twitter saying about TSLA"])
def test_questions_about_whether_talk_is_real_reach_the_sentiment_engine(question: str) -> None:
    request = detect(question)
    assert request is not None and request.kind is ResearchKind.SENTIMENT
    assert research.pattern_reading_wins(request, question) is (
        "hype" in question or "pumped" in question or "rumors" in question)


def test_the_desk_integrity_read_is_taken_from_its_own_notes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notes = [
        {"seq": 7, "symbol": "NVDAUSDT", "at": "2026-09-23T19:41:00+00:00",
         "notes": ["panel: 3 analysts, 5 distinct sources, independence 1.67 -> bullish at 0.40 "
                   "after provenance discount"]},
        {"seq": 8, "symbol": "COINUSDT", "at": "2026-09-23T19:42:00+00:00", "notes": []},
    ]
    (tmp_path / "desk_notes.jsonl").write_text(
        "\n".join(json.dumps(n) for n in notes), encoding="utf-8")
    monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
    read = research._desk_integrity_read("NVDAUSDT")
    assert read == {"seq": 7, "at": "2026-09-23T19:41:00+00:00", "analysts": 3, "sources": 5,
                    "signal": "bullish", "confidence": 0.40}
    assert research._desk_integrity_read("COINUSDT") is None
    assert research._desk_integrity_read("TSLAUSDT") is None


def test_the_coordination_test_is_quoted_from_its_artefact(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_DATA_DIR", str(DATA))
    found = research._coordination_test("NVDAUSDT")
    assert found is not None
    text, counts = found
    report = json.loads((DATA / "sentiment_comparison.json").read_text(encoding="utf-8"))
    assert counts["narratives"] == len(report["narratives"])
    assert f"on {counts['finbert_louder']} of {counts['narratives']}" in text
    assert "On the NVDA narrative" in text


def test_a_page_link_names_a_route_the_server_serves() -> None:
    import inspect

    from argus.lui import server

    source = inspect.getsource(server.Handler.do_GET)
    for question, _ in IN_THE_CONSOLE.values():
        if question.startswith("/"):
            assert f'path == "{question}"' in source, question


@pytest.mark.parametrize("question", [
    "what was NVDA's revenue last quarter", "what was NVDA's net income as of 1 March 2026",
    "what is TSLA's EPS", "AAPL gross margin"])
def test_line_item_questions_reach_the_filings(question: str) -> None:
    request = detect(question)
    assert request is not None and request.kind is ResearchKind.FUNDAMENTALS
    assert research.pattern_reading_wins(request, question)


def test_an_as_of_date_is_read_in_the_ways_people_write_it() -> None:
    for text in ("as of 1 March 2026", "as of March 1, 2026", "as of 2026-03-01",
                 "as known on 1st March 2026"):
        when = research._as_of(f"NVDA revenue {text}")
        assert when is not None and (when.year, when.month, when.day) == (2026, 3, 1), text
    assert research._as_of("NVDA revenue last quarter") is None
