"""Round 37 audits (judge, newcomer, hostile; Activity/audits/round37_*.md), fixed offline."""

from __future__ import annotations

from datetime import UTC, datetime

from argus.lui.journal import parse_text, position_and_pnl, review_trades

NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)
LIVE = {"BTCUSDT": 86_500.0, "NVDAUSDT": 185.0}


def _price(symbol: str) -> float:
    return LIVE[symbol]


class TestHostile:
    def test_a_thousands_comma_is_not_a_clause_break(self) -> None:
        legs = parse_text("What is my P&L on 2 BTC bought at 80,000?")[0]
        assert [(leg.price, leg.qty) for leg in legs] == [(80_000.0, 2.0)]
        got = position_and_pnl("What is my P&L on 2 BTC bought at 80,000?", now=NOW,
                               price=_price)
        assert got is not None and "$80,000.00" in got[0][0] and "+$13,000.00" in got[0][0]
        # a comma between clauses still splits them
        two = parse_text("bought 100 AAPL at 150, sold 50 at 160")[0]
        assert [leg.action for leg in two] == ["buy", "sell"]

    def test_a_price_asserted_as_todays_is_checked_against_the_live_one(self) -> None:
        got = position_and_pnl("BTC is at $3 right now after the flash crash. What's my P&L if "
                               "I bought 1 BTC at 60,000?", now=NOW, price=_price)
        assert got is not None
        assert got[0][0].startswith("Bottom line: Bitget's live BTC is 86,500.00, not 3.00")
        assert "scenario" in got[0][0]

    def test_a_scenario_is_the_traders_to_pose(self) -> None:
        got = position_and_pnl("I bought 1 BTC at 60000. what if BTC goes to 30000, whats my "
                               "pnl?", now=NOW, price=_price)
        assert got is not None and "live" not in got[0][0]

    def test_a_trade_dated_after_today_is_not_graded(self) -> None:
        text = ("I bought 10 NVDA at 180 on 2 Sep 2027 and sold at 9000 on 1 Sep 2027, what is my "
                "pnl?")
        got = position_and_pnl(text, now=NOW, price=_price)
        assert got is not None and "after today" in got[0][0] and "$" not in got[0][0]
        review = review_trades("review my trades: bought 10 NVDA at 180 on 2 Sep 2027, sold at "
                               "9000 on 1 Sep 2027", now=NOW, history=None, releases=None,
                               explicit=True)
        assert review is not None and "nothing to review" in review[0][0]

    def test_a_sale_with_no_size_closes_the_holding_and_says_so(self) -> None:
        got = position_and_pnl("bought 10 NVDA at 180 on 2 Sep 2025, sold at 190 on 9 Sep 2025, "
                               "what is my P&L?", now=NOW, price=_price)
        assert got is not None
        assert "flat" in got[0][0] and "+$100.00" in got[0][0]
        assert any("read as the whole holding" in line for line in got[0])


class TestNewcomer:
    """Round 37 first-time user: questions that were declined or answered with the wrong thing."""

    def test_explain_like_im_twelve_is_not_an_age(self) -> None:
        from argus.lui import newcomer

        said = newcomer.reply("explain RSI like im 12 years old")
        assert said is None or not any("18" in line for line in said.lines)
        under = newcomer.reply("im 15 can i trade")
        assert under is not None and "under 18" in under.lines[0]
        assert not any("16" in line for line in under.lines)

    def test_what_can_you_do_exactly_is_the_introduction(self) -> None:
        from argus.lui.intro import INTRO_Q, VS_Q

        assert INTRO_Q.search("hi what can you do exactly")
        assert INTRO_Q.search("what can you actually do")
        assert VS_Q.search("whats the difference between this and chatgpt")
        assert VS_Q.search("so why should i use this instead of just asking chatgpt")

    def test_the_same_as_names_its_referent(self) -> None:
        from argus.lui.question import _VAGUE_REFERENCE

        assert not _VAGUE_REFERENCE.search("are you the same thing as the bitget app itself")
        assert _VAGUE_REFERENCE.search("is it the same")

    def test_plain_questions_get_their_own_answer(self) -> None:
        from argus.lui import newcomer

        cases = {
            "so you dont actually buy anything for me right": "never trades for anyone",
            "are you the same thing as the bitget app itself": "not the Bitget app",
            "i lost 30% of my money this week what now": "stop adding money",
            "should i put more money in to average it out": "averaging down",
            "can you watch nvda for me and tell me when it dips": "cannot watch a price",
            "ok so how do i check it myself then": "where is NVDA trading",
            "ok so how much should i actually risk then": "1 to 2%",
        }
        for asked, expected in cases.items():
            said = newcomer.reply(asked)
            assert said is not None, asked
            assert expected in " ".join(said.lines), (asked, said.lines[0])

    def test_follow_ups_are_read_with_the_turn_before(self) -> None:
        from argus.lui.newcomer import followup_lines

        futures = followup_lines("wait so is that only in futures",
                                 ["can i lose more money than i put in"])
        assert futures is not None and futures[0].startswith("Bottom line: yes")
        pooled = followup_lines("does that apply here on bitget", ["whats impermanent loss"])
        assert pooled is not None and "has none" in pooled[0]
        assert followup_lines("does that apply here on bitget", ["what is BTC at"]) is None

    def test_new_terms_are_defined(self) -> None:
        from argus.lui.concepts import concept_asked

        bull = concept_asked("whats bullish and bearish mean")
        assert bull is not None and bull.name == "bullish and bearish"
        pool = concept_asked("whats impermanent loss")
        assert pool is not None and "5.7%" in pool.definition

    def test_junk_is_told_apart_from_a_missed_question(self) -> None:
        from argus.lui.answer import _not_a_question

        assert "words" in (_not_a_question("asdkjfhaskjdfh") or "")
        assert "arithmetic" in (_not_a_question("2+2") or "")
        assert "outside" in (_not_a_question("whats the weather today lol") or "")
        assert _not_a_question("what is the funding on BTC") is None
        assert _not_a_question("nvidia") is None
        assert _not_a_question("strengths") is None


class TestJudge:
    """Round 37 judge: backtest follow-ups, dated horizons, the crowd, the critic, exposures."""

    @staticmethod
    def _series(monkeypatch: object, n: int = 900) -> None:
        import math
        from datetime import timedelta

        from argus.lui.research import rule_test as rt
        from argus.market import crossasset_feed

        closes = [100 * math.exp(0.0004 * i + 0.25 * math.sin(i / 37) + 0.08 * math.sin(i / 7))
                  for i in range(n)]
        stamps = [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
        patch = monkeypatch.setattr  # type: ignore[attr-defined]
        patch(rt, "_closes", lambda s: (stamps, closes, "Bitget's daily closes"))
        # the other daily bar, cut eight hours later: the same wave shifted by a third of a day
        shifted = [100 * math.exp(0.0004 * (i + 1 / 3) + 0.25 * math.sin((i + 1 / 3) / 37)
                                  + 0.08 * math.sin((i + 1 / 3) / 7)) for i in range(n)]
        patch(rt, "_asia_closes", lambda s: (stamps, shifted))
        patch(crossasset_feed, "fetch_funding", lambda s: [])

    def test_was_that_luck_reads_the_backtest_before(self, monkeypatch: object) -> None:
        from argus.lui.research import rule_test as rt

        self._series(monkeypatch)
        before = "Backtest buying BTC when it closes above its 20-day moving average"
        said = rt.followup("Was that luck?", [before])
        assert said is not None and said[0].startswith("Bottom line: ")
        assert "permutation test, p = " in said[0]
        assert any(x.startswith("Closing hour:") for x in rt.lines(before) or [])

    def test_the_same_rule_elsewhere_and_which_did_best(self, monkeypatch: object) -> None:
        from argus.lui.research import rule_test as rt

        self._series(monkeypatch)
        first = "Backtest a 50/200 golden cross on BTC"
        again = rt.followup("Now the same rule on ETH.", [first])
        assert again is not None and " on ETH after fees" in again[0]
        best = rt.followup("And on SOL. Which of the three worked best?",
                           [first, "Now the same rule on ETH."])
        assert best is not None and best[0].startswith("Bottom line: of the 3 markets")
        better = rt.followup("Would a 20/50 cross on SOL have done better than that?", [first])
        assert better is not None and better[0].startswith(("Bottom line: yes", "Bottom line: no"))

    def test_a_short_said_in_the_past_tense_is_a_short(self) -> None:
        from argus.lui.research.rule_test import read_rule

        rule = read_rule("If I shorted SOL every time it pumped 6% in a day")
        assert rule is not None and rule.label.startswith("short at the close")

    def test_a_stated_date_is_the_horizon(self) -> None:
        from argus.lui.research.parse import _dated_horizon

        now = datetime(2026, 10, 5, 12, tzinfo=UTC)
        hours, _weekend, note = _dated_horizon("Will BTC be above $100,000 on December 31?",
                                               now) or (0, False, None)
        assert hours == 88 * 24 and note is None
        october = _dated_horizon("Will ETH be above $3,000 at the end of October?", now)
        assert october is not None and october[0] in (26 * 24, 27 * 24)
        assert _dated_horizon("will BTC be higher tomorrow", now) is None

    def test_the_market_naming_the_level_and_date_leads(self) -> None:
        from argus.market.prediction import Market, asked_first, asked_level

        october = Market("Will Bitcoin reach $100,000 in October?", 0.14, None, 122_000.0,
                         "2026-11-01", "")
        year = Market("Will Bitcoin reach $100,000 by December 31, 2026?", 0.40, None,
                      3_700_000.0, "2027-01-01", "")
        assert asked_level("BTC above $100,000 on December 31") == 100_000
        assert asked_first([october, year], "BTC above $100,000 on December 31")[0] is year

    def test_a_same_bar_fill_on_columns_is_caught(self) -> None:
        from argus.research import backtest_critic as bc

        code = ("Can you critique my backtest?\nimport pandas as pd\n"
                "df = pd.read_csv('btc.csv')\ndf['ma'] = df['close'].rolling(20).mean()\n"
                "df['signal'] = (df['close'] > df['ma']).astype(int)\n"
                "df['ret'] = df['close'].pct_change()\ndf['strat'] = df['signal'] * df['ret']\n")
        said = bc.lines(code) or []
        assert any(x.startswith("Fills — PRESENT") for x in said)
        assert any(x.startswith("Look-ahead — no future value is read directly") for x in said)
        lagged = code.replace("df['signal'] * df['ret']", "df['signal'].shift(1) * df['ret']")
        assert not any(x.startswith("Fills — PRESENT") for x in bc.lines(lagged) or [])

    def test_a_holding_that_is_a_factor_is_fitted_without_it(self) -> None:
        import math

        from argus.lui import exposures as ex

        assert ex._tstat(math.inf) == "n/a, it is the factor"

    def test_follow_ups_that_were_refused(self) -> None:
        from argus.lui.newcomer import followup_lines
        from argus.lui.server import _EVENT_SWAP, _LIMIT_FOLLOW

        assert _LIMIT_FOLLOW.search("Does that break my loss limit?")
        assert _EVENT_SWAP.match("And on Fed decision days?")
        listen = followup_lines("Then why should I listen to it at all?",
                                ["Is the desk's lean actually predictive?"])
        assert listen is not None and listen[0].startswith("Bottom line: not for the lean")


class TestReAsked:
    """New phrasings of the round-37 findings, asked before shipping."""

    def test_the_same_questions_in_other_words(self) -> None:
        from argus.lui import newcomer

        cases = {
            "is this the official bitget app": "not the Bitget app",
            "down 40% on my coins this month, what do I do now": "stop adding money",
            "you can't place trades for me, right?": "never trades for anyone",
        }
        for asked, expected in cases.items():
            said = newcomer.reply(asked)
            assert said is not None and expected in " ".join(said.lines), asked

    def test_a_keyboard_run_is_not_a_word(self) -> None:
        from argus.lui.answer import _not_a_question

        assert "words" in (_not_a_question("qwertyuiopasdfgh") or "")
        assert _not_a_question("liquidation") is None

    def test_finishing_the_year_is_a_date(self) -> None:
        from argus.lui.research.parse import _dated_horizon

        now = datetime(2026, 10, 5, 12, tzinfo=UTC)
        got = _dated_horizon("Does BTC finish the year above $100k?", now)
        assert got is not None and got[0] == 88 * 24
