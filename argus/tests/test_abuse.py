"""The crowd read's abuse screen: what it withholds, what it must not, and that the copied lexicon
is the one that was measured."""

from __future__ import annotations

import pytest

from argus.market import abuse
from argus.market.abuse import (
    CUSS_RATED_2,
    CUSS_RATED_2_SHA256,
    DEFAULT_SCREEN,
    OBSCENITY,
    REASON_TERM,
    REASON_THREAT,
    AbuseScreen,
    blacklist_form,
    handle_is_abusive,
    handle_words,
    lexicon_digest,
    screen,
)


class TestLexiconProvenance:
    def test_the_copied_cuss_list_is_the_pinned_one(self) -> None:
        assert len(CUSS_RATED_2) == 1255
        assert list(CUSS_RATED_2) == sorted(CUSS_RATED_2)
        assert lexicon_digest(CUSS_RATED_2) == CUSS_RATED_2_SHA256

    def test_every_obscenity_phrase_compiles_and_has_a_pattern(self) -> None:
        assert len(OBSCENITY) == 69
        for word, patterns, _ in OBSCENITY:
            assert patterns, word
        assert AbuseScreen()._lexicon.obscenity


class TestTransforms:
    def test_leetspeak_case_and_repeats_are_folded_like_obscenity(self) -> None:
        assert blacklist_form("SH1T")[0] == "shit"
        assert blacklist_form("fuuuuck")[0] == "fuck"
        # b, e, o, l, s and g keep two copies, so these words survive the collapse.
        assert blacklist_form("boob")[0] == "boob"
        assert blacklist_form("class")[0] == "class"

    def test_full_width_letters_are_folded(self) -> None:
        full_width = "".join(chr(0xFF00 + ord(c) - 0x20) for c in "fuck")
        assert screen(full_width).abusive

    def test_the_index_map_points_back_into_the_original(self) -> None:
        form, where = blacklist_form("a  SHIIIT b")
        start = form.index("shit")
        assert where[start] == 3


class TestWithholds:
    @pytest.mark.parametrize("text", [
        "you are a fucking idiot",
        "this is bullshit",
        "sh1t project, devs are clowns",
        "f*ck this chart",
        "total a**hole move",
        "f***",
    ])
    def test_profanity_including_leet_and_masks(self, text: str) -> None:
        verdict = screen(text)
        assert verdict.abusive and verdict.reason == REASON_TERM

    @pytest.mark.parametrize("text", [
        "kys",
        "kill yourself",
        "the devs should be hanged",
        "they are vermin",
        "go back to your country",
        "stfu",
    ])
    def test_threats_and_harassment(self, text: str) -> None:
        verdict = screen(text)
        assert verdict.abusive and verdict.reason == REASON_THREAT

    def test_a_verdict_names_entries_never_the_post(self) -> None:
        text = "honestly you are a fucking idiot for buying this"
        verdict = screen(text)
        assert all(term not in ("honestly", text) for term in verdict.terms)
        assert set(verdict.terms) <= set(CUSS_RATED_2) | {w for w, _, _ in OBSCENITY} | {
            blacklist_form(w)[0] for w in CUSS_RATED_2}


class TestDoesNotWithhold:
    @pytest.mark.parametrize("text", [
        "NVDA guidance looks conservative given hyperscaler capex plans for next year",
        "assessment of the assets under management",
        "Scunthorpe United", "cockpit view of the order book", "analyst upgrade on MSFT",
        "cumulative returns since listing", "kung fu panda", "a 45s record",
        "ho ho ho, santa rally", "ok k", "spicy pricing from amazon", "classic bass pro",
        "Hanging man candle on the daily", "the order should be executed at the open",
        "shorts are going to get killed", "estimates should be beaten",
        "*very* bad print", "**BTC** breaking out",
    ])
    def test_ordinary_market_and_english_text(self, text: str) -> None:
        assert not screen(text).abusive, screen(text)

    @pytest.mark.parametrize("text", [
        "Top ETF gainers & losers for today's session",
        "dumb money is buying the top", "let's dumb it down for beginners",
        "classic sucker's rally", "another sucker" + chr(0x2019) + "s rally",
        "shitcoin season is back", "shitcoins are pumping", "rapeseed futures fell",
        "Dick's Sporting Goods beat",
    ])
    def test_market_jargon_is_allowlisted(self, text: str) -> None:
        assert not screen(text).abusive
        assert AbuseScreen(allowlist=False)(text).abusive

    def test_the_allowlist_excuses_only_the_phrase(self) -> None:
        assert screen("this shitcoin is shit").abusive


class TestSwitches:
    def test_each_matcher_can_be_measured_alone(self) -> None:
        only_threats = AbuseScreen(obscenity=False, cuss=False)
        assert only_threats("kys").abusive and not only_threats("fucking idiot").abusive
        only_cuss = AbuseScreen(obscenity=False, threats=False)
        assert only_cuss("idiot").abusive and not only_cuss("kys").abusive
        assert DEFAULT_SCREEN.name == "obscenity+cuss2+threats"
        assert AbuseScreen(allowlist=False).name.endswith("(no allowlist)")


class TestHandles:
    def test_run_together_handles_are_split_into_words(self) -> None:
        assert handle_words("@FuckTheFed") == "Fuck The Fed"
        assert handle_words("u/fuck_the_fed99") == "fuck the fed"

    def test_abusive_handles_are_caught_and_ordinary_ones_are_not(self) -> None:
        assert handle_is_abusive("@FuckTheFed")
        assert not handle_is_abusive("@ScunthorpeFan")
        assert not handle_is_abusive("@shitcoin_king")
        assert not handle_is_abusive("u/value_investor")

    def test_the_check_can_be_swapped(self) -> None:
        assert not handle_is_abusive("@FuckTheFed", abuse.AbuseScreen(obscenity=False,
                                                                        cuss=False))
