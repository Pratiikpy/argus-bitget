"""Round 15 of the §27 audits (2026-10-01): a judge's pass on a trader who states a book and a
thesis in words, asks what would prove it wrong, and recalls it all later.

Each test pins one finding (tracker rows 586-600). Live sources are stubbed or never reached.
"""

from __future__ import annotations

import pytest

from argus.lui.research.kinds import ResearchKind


class TestHoldingsStatedInWords:
    @pytest.mark.parametrize(("text", "expected"), [
        ("im long half NVDA half ETH", {"NVDAUSDT": 0.5, "ETHUSDT": 0.5}),
        ("a third in BTC and two thirds in SOL", {"BTCUSDT": 1 / 3, "SOLUSDT": 2 / 3}),
        ("a quarter in ETH, three quarters in BTC", {"ETHUSDT": 0.25, "BTCUSDT": 0.75})])
    def test_word_weights_become_a_book(self, text: str, expected: dict[str, float]) -> None:
        from argus.lui.research.parse import parse_book

        got = parse_book(text)
        assert {k: round(v, 4) for k, v in got.items()} == {
            k: round(v, 4) for k, v in expected.items()}


class TestWhatWouldProveMeWrong:
    def test_the_question_is_recognised(self) -> None:
        from argus.lui.thesis_answer import FALSIFY

        assert FALSIFY.search("What would prove this thesis wrong?")
        assert FALSIFY.search("what would change my mind")
        assert not FALSIFY.search("what is the price of ETH")

    def test_without_a_thesis_it_says_so(self) -> None:
        from argus.lui.thesis_answer import falsify

        got = falsify("what would prove my thesis wrong", [])
        assert got is not None
        assert "no thesis earlier" in got[0][0]

    def test_other_questions_are_left_alone(self) -> None:
        from argus.lui.thesis_answer import falsify

        assert falsify("what is funding on BTC", []) is None


class TestCryptoNewsIsNotADeskQuestion:
    @pytest.mark.parametrize("q", [
        "What news is moving crypto today?",
        "any news on crypto today",
        "news on the crypto market",
        "What news is moving crypto today and what should I do about it?"])
    def test_crypto_news_reads_bitcoin(self, q: str) -> None:
        from argus.lui.research.parse import detect

        got = detect(q)
        assert got is not None
        assert got.kind is ResearchKind.NEWS and got.symbols == ("BTCUSDT",)

    def test_stock_market_news_still_reads_the_nasdaq(self) -> None:
        from argus.lui.research.parse import detect

        got = detect("what news is moving the market today")
        assert got is not None and got.symbols == ("QQQUSDT",)

    def test_a_request_that_stands_alone_is_not_the_previous_question(self) -> None:
        from argus.lui.research.parse import follow_up

        prior = ["whats the sentiment on doge rn, are ppl too long?"]
        got = follow_up("What news is moving crypto today and what should I do about it?",
                        prior, "")
        assert got is None or got.symbols != ("DOGEUSDT",)


class TestSimplerWithAnIntroduction:
    @pytest.mark.parametrize("q", [
        "explain that in simple words, I'm new to trading",
        "explain that simply, i am a beginner",
        "explain that in simple words"])
    def test_asks_the_last_answer_again_simply(self, q: str) -> None:
        from argus.lui.server import _SIMPLER

        assert _SIMPLER.match(q)

    def test_a_different_question_is_not_matched(self) -> None:
        from argus.lui.server import _SIMPLER

        assert not _SIMPLER.match("explain that in simple words and then buy 5 ETH")


class TestRatioAtAnExtreme:
    def test_a_ratio_question_is_placed_in_its_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.lui import thesis

        def closes(symbol: str) -> list[tuple[float, float, float, float]]:
            base = 2000.0 if symbol == "ETHUSDT" else 80000.0
            drift = 1.0 if symbol == "ETHUSDT" else 1.3
            return [(0.0, 0.0, base * (1 + 0.002 * drift * i), 0.0) for i in range(90)]

        monkeypatch.setattr(thesis, "_closes", closes)
        got = thesis.ratio_extreme("is the eth/btc ratio near its 90 day low?")
        assert got is not None and got.result is thesis.Result.SUPPORTED
        assert "of the way up" in got.line

    def test_other_text_asks_no_such_thing(self) -> None:
        from argus.lui import thesis

        assert thesis.ratio_extreme("what is the eth/btc ratio") is None


class TestSavedBookReachesTheCalendar:
    BOOK = "60% NVDA, 25% ETH, 15% BTC"

    def test_asking_about_my_book_uses_the_saved_book(self) -> None:
        from argus.lui.watchlist import resolve_book

        book, _cash, notes = resolve_book("Given my ETH thesis and my book, what should I watch "
                                          "this week?", self.BOOK)
        assert set(book) == {"NVDAUSDT", "ETHUSDT", "BTCUSDT"}
        assert any("saved book" in n for n in notes)

    def test_a_name_with_its_own_weights_wins(self) -> None:
        from argus.lui.watchlist import resolve_book

        book, _cash, _notes = resolve_book("I hold 50% SOL and 50% ETH, what should I watch",
                                           self.BOOK)
        assert set(book) == {"SOLUSDT", "ETHUSDT"}

    def test_an_unrelated_name_is_read_on_its_own(self) -> None:
        from argus.lui.watchlist import resolve_book

        book, _cash, _notes = resolve_book("what should I watch for SOL", self.BOOK)
        assert set(book) == {"SOLUSDT"}


class TestMemoryOfWhatWasSaid:
    def test_an_idea_with_a_reason_is_kept_as_a_thesis(self) -> None:
        from argus.lui import memory

        facts = memory.extract("I want to short TSLA because funding is crowded long",
                               price_of=lambda _s: 250.0)
        theses = [f for f in facts if f.kind == "thesis"]
        assert len(theses) == 1 and theses[0].subject == "TSLAUSDT"
        assert "short" in theses[0].value.lower() or "bear" in theses[0].value.lower()

    @pytest.mark.parametrize("q", [
        "remind me what I told you",
        "tell me what I told you",
        "what was it I told you"])
    def test_recall_phrasings(self, q: str) -> None:
        from argus.lui import memory

        assert memory._RECALL.search(q)

    def test_a_stress_answer_is_set_against_the_stated_limit(self) -> None:
        from argus.lui import memory

        facts = memory.extract("my loss limit is 3%")
        lines = memory.after(
            ["Bottom line: If BTC moves -10%: your book moves about -7.5%, about $1,500."],
            None, facts)
        assert any("limit" in line.lower() for line in lines[1:])


class TestABookWrittenAsBareNumbers:
    """A test of the research bench reached the live ticker feed because "META 30, GOOGL 30,
    AMZN 40" was read as 30 GOOGL and 30 AMZN contracts (the count pattern took the comma as part
    of the number), and then as an equal-weight book. Numbers that add to 100 are percent."""

    def test_numbers_adding_to_a_hundred_are_weights(self) -> None:
        from argus.lui.research.parse import holding_pairs

        got = {sym: round(w, 2) for _, sym, w in holding_pairs(
            "Stress test my book (META 30, GOOGL 30, AMZN 40) for a 2022-style tech drawdown.")}
        assert got == {"METAUSDT": 0.3, "GOOGLUSDT": 0.3, "AMZNUSDT": 0.4}

    @pytest.mark.parametrize("text", ["NVDA 50", "I hold 50 NVDA and 50 AAPL shares",
                                      "BTC 70 ETH 40"])
    def test_anything_that_does_not_add_to_a_hundred_is_left_alone(self, text: str) -> None:
        from argus.lui.research.parse import holding_pairs

        assert holding_pairs(text) == []

    def test_a_comma_after_a_number_is_not_part_of_it(self) -> None:
        from argus.lui.research.parse import _BOOK_COUNT

        assert not list(_BOOK_COUNT.finditer("META 30, GOOGL 30, AMZN 40"))
        assert [m.group(1) for m in _BOOK_COUNT.finditer("I hold 1,200 NVDA")] == ["1,200"]
