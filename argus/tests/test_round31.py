"""Round 31's three audits (judge, first-time user, hostile reviewer), run offline: each fix is
checked on the phrasing that exposed it and on the near-miss beside it, with every live figure
stubbed by a hand-made input whose answer can be read off it."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from argus.lui import (
    arbiter,
    goal_target,
    honesty,
    mandate,
    mcp_server,
    multistep,
    newcomer,
    position_math,
    premise_facts,
    server,
)
from argus.lui.memory_model import _valid
from argus.lui.question import Intent, classify
from argus.lui.research import carry, events_stress
from argus.lui.research.statement import is_statement


def _gated(text: str, intent: Intent, prior: list[str] | None = None) -> str:
    question = replace(classify(text, now=datetime(2026, 10, 4, tzinfo=UTC)), intent=intent)
    _q, by = arbiter.gate_ledger_reading(question, "ngram", text, prior=prior or [], audit={},
                                         desk_first=False)
    return by


class TestTheRecordAnswersOnlyTheDesk:
    @pytest.mark.parametrize("said", [
        "i heard u can just copy what a pro trader does on bitget and make money, is that real",
        "omg a stablecoin lost its peg, i have usdt should i panic sell right now",
        "Compare that allocation to a plain 60/40 book over 12 months, which one delivered more "
        "return per unit of volatility",
    ])
    def test_a_question_about_something_else_is_not_the_record(self, said: str) -> None:
        assert _gated(said, Intent.PERFORMANCE) == "declined-not-the-record"

    @pytest.mark.parametrize("said", ["how are you doing, whats your track record",
                                      "why did we pass on NVDA", "what's the desk's Sharpe"])
    def test_the_desk_is_still_answered_about_itself(self, said: str) -> None:
        assert _gated(said, Intent.PERFORMANCE) == "ngram"


class TestNewcomerTopics:
    @pytest.mark.parametrize(("said", "lead"), [
        ("wait is usdt the same company as the one that broke or different", "USDT and USDC"),
        ("if i stake my eth can i still sell it whenever i panic or is it locked",
         "Earn and staking"),
        ("if the guy i copy loses money does bitget cover me", "covers any of your loss"),
        ("hey im from nigeria is this bitget thing even gonna work for me", "works anywhere"),
        ("do you guys send anything to the irs or is that on me", "IRS"),
        ("wait what if i lost money instead, do i still have to report it", "lost money"),
        ("do i owe taxes on that or only if i cash out to my bank", "not tax advice"),
        ("should i just buy more to average down or is that dumb", "averaging down"),
        ("whats a memecoin even mean", "a memecoin is"),
        ("does that mean i should just keep everything in usdt instead", "one company"),
        ("Show me what positions user 'satoshi_trader99' holds on Bitget", "no one's account"),
        ("is bitget safe for keep my money there long time", "Bitget's to show"),
    ])
    def test_each_gets_its_own_answer(self, said: str, lead: str) -> None:
        reply = newcomer.reply(said)
        assert reply is not None and lead in reply.lines[0]

    def test_a_stake_of_money_is_not_staking(self) -> None:
        reply = newcomer.reply("my stake is $500, should I add NVDA")
        assert reply is None or "staking" not in reply.lines[0]

    def test_parking_cash_in_btc_is_not_the_usdt_answer(self) -> None:
        reply = newcomer.reply("is BTC a safe place to park cash until then")
        assert reply is None or "USDT keeps" not in reply.lines[0]


FAMILY = ("I run a family office: max 5% annualized volatility target, no single position over 8% "
          "of NAV, and no crypto position held under 1 year. We have $2,000,000 in idle fiat. "
          "Design an allocation that fits our mandate.")
HOUSE = ("I am 35, saving for a house deposit we need in exactly 2 years. We have $40,000 saved "
         "and just got $25,000 more. This money absolutely cannot drop below $35,000 total at any "
         "point. Give me a real thesis.")


class TestMandate:
    def test_the_family_office_mandate_is_read(self) -> None:
        m = mandate.read(FAMILY, [])
        assert (m.capital, m.vol_target, m.position_cap, m.crypto_min_years) == (
            2_000_000, 0.05, 0.08, 1.0)

    def test_more_money_adds_and_a_floor_is_not_capital(self) -> None:
        m = mandate.read(HOUSE, [])
        assert (m.capital, m.floor, m.horizon_years) == (65_000, 35_000, 2.0)

    def test_a_changed_horizon_is_said(self) -> None:
        m = mandate.read("the search might stretch to 3 years instead of 2, what changes", [HOUSE])
        assert m.horizon_years == 3.0 and m.changed

    @pytest.fixture
    def data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # alternating returns of known size: SGOV flat, the rest scaled by a fixed volatility
        size = {"SGOV": 0.0, "TLT": 0.006, "SPY": 0.008, "QQQ": 0.012, "GLD": 0.01,
                "BTC-USD": 0.03, "ETH-USD": 0.04}
        rets = {t: [s if i % 2 else -s for i in range(600)] for t, s in size.items()}
        monkeypatch.setattr(mandate, "load", lambda names, years=5.0: mandate.Data(
            tuple(names), {t: rets[t] for t in names}, 250))

    @pytest.mark.usefixtures("data")
    def test_no_position_exceeds_the_cap_and_crypto_is_left_out(self) -> None:
        plan = mandate.build(mandate.read(FAMILY, []), mandate.load([t for t, *_ in mandate.MENU]))
        risky = {t: w for t, w in plan.weights.items() if t != mandate.CASH}
        assert all(w <= 0.08 + 1e-9 for w in risky.values())
        assert "BTC-USD" not in risky and "ETH-USD" not in risky
        assert plan.vol <= 0.05 + 1e-6
        assert abs(sum(plan.weights.values()) - 1) < 1e-9

    @pytest.mark.usefixtures("data")
    def test_the_floor_binds_when_it_is_tight(self) -> None:
        m = mandate.read("$100,000, it cannot drop below $99,800 at any point, build a plan", [])
        plan = mandate.build(m, mandate.load([t for t, *_ in mandate.MENU]))
        assert plan.binding == "floor"
        assert 100_000 * (1 - plan.deepest) >= 99_800 - 1

    @pytest.mark.usefixtures("data")
    def test_the_follow_ups(self) -> None:
        cut = mandate.cut_first_lines("which single line item would you cut first if vol "
                                      "breached 5%", [FAMILY])
        assert cut is not None and cut[0].startswith("Bottom line: cut ")
        versus = mandate.compare_lines("compare it to a 60/40 book, which one wins", [FAMILY])
        assert versus is not None and "wins on return per unit of volatility" in versus[0]
        legs = mandate.ticket_legs("give me the exact execution plan to get to that allocation",
                                   [FAMILY])
        assert legs is not None
        held, plan = legs
        assert held.capital is not None
        assert all(w * held.capital <= 160_000 + 1 for t, w in plan.weights.items()
                   if t != mandate.CASH)

    def test_single_name_var_sizing_shows_the_math(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rets = [0.01] * 230 + [-0.04] * 20  # 8% of days at -4%: the 5th-percentile day is -4%
        monkeypatch.setattr(mandate, "load", lambda names, years=1.0: mandate.Data(
            tuple(names), {names[0]: rets}, 250))
        said = mandate.var_size_lines("Size a $30,000 NVDA position so the 1-day 95% VaR does "
                                      "not exceed $1,200. Show the math.")
        assert said is not None
        assert "$30,000" in said[0] and "$1,200 / 0.0400 = $30,000" in said[1]


class TestPositionMath:
    @pytest.fixture(autouse=True)
    def _price(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(position_math, "_last", lambda symbol: 85_000.0)

    def test_holding_cost_with_funding(self) -> None:
        said = position_math.holding_cost_lines(
            "I'm long 1 BTC at 10x leverage, round-trip taker fee 0.06%, and funding runs -0.01% "
            "against me every 8 hours. If I hold for exactly 5 days, total cost?")
        assert said is not None
        assert "$51.00 in fees" in said[0] and "$127.50 in funding" in said[0]

    def test_ten_x_is_not_an_entry_price(self) -> None:
        assert not is_statement("I'm long 1 BTC at 10x leverage on Bitget, round-trip taker fee "
                                "0.06%, hold for 5 days")

    def test_average_entry(self) -> None:
        said = position_math.average_entry_lines("I bought 1 BTC at 90000, then 2 BTC at 80000, "
                                                 "then 1 BTC at 70000 -- what's my average entry?")
        assert said is not None and "80,000.00 across 4 BTC" in said[0]

    def test_return_on_margin(self) -> None:
        said = position_math.margin_return_lines("I shorted 1 ETH at 3000 and covered at 2700, "
                                                 "using 10x leverage. Percentage return on my "
                                                 "margin?")
        assert said is not None and said[0].startswith("Bottom line: +100% on your margin")

    def test_a_corrected_entry_gives_the_pnl(self) -> None:
        said = position_math.unrealised_lines(
            "Actually, my BTC entry was 95000, not 85000 -- what is my unrealized P&L?",
            ["My BTC entry price is 85000, I'm long 2 BTC at 10x. Remember that."])
        assert said is not None and "-$20,000.00 unrealised" in said[0]


class TestPremisesAndDates:
    def test_a_written_future_date(self) -> None:
        assert honesty.future_date_asked_as_past(
            "What was BTC's closing price on January 1, 2030?",
            date(2026, 10, 4)) == date(2030, 1, 1)

    def test_an_index_claim_is_said_unchecked(self) -> None:
        said = premise_facts.index_membership("Since Coinbase was removed from the S&P 500 this "
                                              "week, outlook?", "COINUSDT")
        assert said is not None and "neither confirmed nor assumed" in said

    def test_a_fee_promotion_against_the_contract_list(self,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda path, params=None: [
            {"makerFeeRate": "0.0002", "takerFeeRate": "0.0006", "maxOrderQty": "1200"}])
        said = premise_facts.fee_promotion("Bitget's new 0% maker fee promotion on RWA tokens",
                                           "NVDAUSDT")
        assert said is not None and "0.02% maker" in said

    def test_an_order_past_bitgets_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "public_get", lambda path, params=None: [
            {"maxOrderQty": "1200"}])
        sized = server._SIZE_STATED.search("I want to long 50000000000 BTC at 125x")
        assert sized is not None
        said = server._size_sanity(sized, "BTCUSDT")
        assert said is not None and "1,200 BTC" in said and "21 million" in said

    def test_a_long_stop_above_an_at_entry(self) -> None:
        said = server._stop_side_lines("I'm long ETH at 3000 and placed my stop-loss at 3100.")
        assert said is not None and said[1]

    def test_a_var_window_is_not_a_horizon(self) -> None:
        assert _valid("horizon", "24", "", "1-day") is None
        assert _valid("horizon", "24", "", "1-day 95% VaR") is None


class TestStructure:
    def test_a_lettered_checklist(self) -> None:
        assert multistep._numbered("(a) what is BTC's price (b) is it a good time to buy (c) what "
                                   "leverage should I use") == [
            "what is BTC's price", "is it a good time to buy", "what leverage should I use"]

    def test_a_decimal_is_not_a_list(self) -> None:
        assert multistep._numbered("BTC fell 3.5% and ETH 2.1% today") is None

    def test_an_unknown_mcp_argument_is_named(self) -> None:
        schema = {"type": "object", "properties": {"book": {"type": "object"},
                                                   "shocked": {"type": "string"},
                                                   "shock_percent": {"type": "number"}}}
        errors = mcp_server.schema_errors(schema, {"book": {}, "shock_symbol": "BTCUSDT"})
        assert any("shock_symbol" in e and "shocked" in e for e in errors)

    def test_balance_sheet_leverage_is_not_trading_leverage(self) -> None:
        from argus.lui.research.parse import _LEVERAGE

        said = "research case for MSTR including BTC balance sheet leverage"
        assert _LEVERAGE.search(said) is None
        assert _LEVERAGE.search("10x leverage on BTC") is not None


class TestEventsAndCarry:
    def test_a_named_crash_is_measured_on_its_own_dates(self,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import equity_history

        start = date(2022, 5, 1)
        closes = [SimpleNamespace(day=start + timedelta(days=i), close=100.0 - (8 * (i - 4)
                                                                                if i > 4 else 0))
                  for i in range(14)]
        monkeypatch.setattr(equity_history, "daily", lambda ticker: closes)
        said = events_stress.lines("stress test a long SOL position against the Terra/Luna week",
                                   ("SOLUSDT",))
        assert said is not None and "Terra/Luna" in said[0]
        assert "-56.0% by the end" in said[1]

    def test_the_carry_plan_has_two_legs_and_no_stop(self,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        from argus.market import bitget

        monkeypatch.setattr(bitget, "fetch_tickers", lambda: {
            "ETHUSDT": SimpleNamespace(last="2500", funding_rate="0.0001")})
        monkeypatch.setattr(carry, "_per_year", lambda s: 3 * 365)
        said = carry.execution_lines("ETHUSDT", 50_000)
        assert said is not None and "short 20.0000 ETHUSDT" in said[0]
        assert said[2].startswith("No stop and no take-profit on either leg")


class TestGoalTarget:
    TODAY = date(2026, 10, 4)

    def test_a_multiple_is_a_target(self) -> None:
        target = goal_target.stated("my goal is to double 1000 dollars in like 6 months, should i "
                                    "just buy and hold or use leverage", today=self.TODAY)
        assert target is not None and (target.start, target.goal) == (1000.0, 2000.0)
        assert target.days == 180

    def test_a_moved_date_carries_to_later_turns(self) -> None:
        prior = ["i only got 50 bucks, i need to turn it into like 300 for a concert in 3 weeks, "
                 "is that even possible", "actually wait the concert is in 5 weeks not 3"]
        target = goal_target.remembered("if it doesnt work by then should i borrow money",
                                        prior, today=self.TODAY)
        assert target is not None and target.days == 35

    def test_borrowing_leads_when_asked(self) -> None:
        said = goal_target.missed_lines(None, "should i just borrow money instead")
        assert said[0].startswith("Bottom line: no — borrowing")
        assert goal_target.missed_lines(None, "what if it doesnt work")[0].startswith(
            "Bottom line: decide it now")


@pytest.mark.parametrize("said", [
    "I'm long BTC, entry 85000, and I set my stop-loss order at 86000. Does that stop-loss "
    "actually protect me against a drop, or is something backwards about it?",
    "I'm short BTC, entry 85000, and I set my stop-loss at 84000. Does that actually protect me "
    "against BTC rising, or is something backwards about it?",
])
def test_a_backwards_stop_is_not_answered_with_a_fresh_plan(
        said: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # "entry" and "stop" in one question looked like an execution-plan request and got a new plan
    # instead of the check of the stop the user set (round-30 hostile set, re-run in round 31)
    monkeypatch.setattr(server, "_execution_levels", lambda *a, **k: pytest.fail("planned"))
    got = server._round27_follow_up(said, [], now=None, visitor="local", book="")
    assert got is not None and "backwards" in got["lines"][0]


def test_two_newcomer_questions_get_both_answers() -> None:
    said = newcomer.reply("yo i saw this coin called PEPE on my tiktok fyp everyone was saying its "
                          "gonna blow up, and also is bitget even legit/safe to put money on",
                          named=True)
    assert said is not None and "being sold to you" in said.lines[0]
    assert any(x.startswith("Also asked — that is Bitget's to show") for x in said.lines)


def test_the_research_case_leads_with_the_evidence_weighed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import task as task_module

    step = SimpleNamespace(applicable=True, refused=False, title="Price", engine="quote",
                           lines=("Bottom line: SOL last 120.",), data={})
    fake = SimpleNamespace(steps=[step], verdict=SimpleNamespace(
        call="Size it by its worst day", lines=("No book was given.",)), conclusion=[])
    monkeypatch.setattr(task_module, "read_question",
                        lambda *a, **k: SimpleNamespace(summary="should I go long SOL"))
    monkeypatch.setattr(task_module, "research_task", lambda **k: fake)
    monkeypatch.setattr(task_module, "weighing", lambda t: SimpleNamespace(
        call="No measured edge", reason="Five episodes cannot tell 80% from a coin flip."))
    said = server._research_case_lines("research case for going long SOL", "SOLUSDT", "")
    assert said is not None
    assert said[0].endswith("— no measured edge. Five episodes cannot tell 80% from a coin flip.")
    assert said[1] == "Sizing: Size it by its worst day."


@pytest.mark.parametrize(("said", "spot"), [
    ("how much would it cost me to exit a $500,000 long in SOL right now", False),
    ("what's the slippage to sell 50 ETH at market on bitget", False),
    ("what does it cost to buy $500 of ETH", True),
])
def test_the_spot_fee_line_is_for_a_first_buy_only(
        said: str, spot: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui.research import ResearchKind
    from argus.market import bitget

    monkeypatch.setattr(bitget, "spot_taker_fee", lambda symbol: 0.001)
    request = SimpleNamespace(kind=ResearchKind.EXECUTION, symbols=("ETHUSDT",), notional=500.0)
    assert (server._spot_cost_line(said, request) is not None) is spot  # type: ignore[arg-type]


def test_a_stated_day_drop_is_held_against_the_day() -> None:
    lines = ["Bottom line: AAPL moved +0.8% yesterday (02 Oct, UTC), open to close, and travelled "
             "1.3% from its low to its high."]
    said = server._move_premise("Apple reported yesterday and the stock dropped 8% — should I buy "
                                "the dip?", lines)
    assert said is not None and "not -8%" in said and "no dip of that size" in said
    assert server._move_premise("AAPL dropped 1% yesterday, why",
                                ["Bottom line: AAPL moved -1.1% yesterday (02 Oct, UTC)."]) is None


class TestLiveReAskRound31:
    TODAY = date(2026, 10, 4)

    def test_a_needed_sum_with_a_date_is_a_target_and_answered(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        said = "i need 400 dollars for a festival in 4 weeks and i have 80"
        target = goal_target.stated(said, today=self.TODAY)
        assert target is not None and (target.start, target.goal, target.days) == (80, 400, 28)
        monkeypatch.setattr(goal_target, "realism_lines",
                            lambda t: [f"Bottom line: turning ${t.start:,.0f} into ${t.goal:,.0f}"])
        got = goal_target.lines(said, [], now=datetime(2026, 10, 4, tzinfo=UTC))
        assert got is not None and "turning $80 into $400" in got[0]

    def test_a_need_without_a_date_is_not_a_target(self) -> None:
        assert goal_target.stated("i need 10 bucks for fees and i have 3", today=self.TODAY) is None

    @pytest.mark.parametrize(("said", "prior"), [
        ("i made 500 on eth this year, do i pay taxes on that", []),
        ("what if i lost instead, does that count", ["i made 500 on eth, do i pay taxes on that"]),
    ])
    def test_tax_is_answered_before_the_research_readers(self, said: str,
                                                         prior: list[str]) -> None:
        got = server._round27_follow_up(said, prior, now=None, visitor="local", book="")
        assert got is not None and got["classified_by"] == "newcomer"
        assert "tax" in " ".join(got["lines"]).lower()

    def test_a_loss_follow_up_leads_with_the_loss_line(self) -> None:
        got = server._round27_follow_up("what if i lost instead, does that count",
                                        ["i made 500 on eth, do i pay taxes on that"], now=None,
                                        visitor="local", book="")
        assert got is not None and got["lines"][0].startswith("Bottom line: If you lost money")

    def test_trusting_bitget_is_the_bitget_answer(self) -> None:
        said = newcomer.reply("saw a coin on twitter everyone says will 100x, and also can i "
                              "trust bitget with my money")
        assert said is not None and any("Bitget's to show" in x for x in said.lines)

    @pytest.mark.parametrize(("said", "average"), [
        ("I bought ETH at 2400 and again at 2000, same size each — where's my average?", 2200.0),
        ("bought BTC at 60k and 40k, same dollars each, my average?", 48000.0),
    ])
    def test_equal_buys_average(self, said: str, average: float,
                                monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(position_math, "_last", lambda symbol: None)
        got = position_math.average_entry_lines(said)
        assert got is not None and f"{average:,.2f}" in got[0]

    def test_which_goes_first_is_the_cut_first_question(self) -> None:
        assert mandate.CUT_FIRST.search("which goes first if markets drop 20%?")

    def test_stems(self) -> None:
        assert [newcomer._stem(w) for w in ("counts", "countries", "losses", "loss")] == [
            "count", "countr", "loss", "loss"]

    def test_a_close_out_cost_is_an_execution_question(self) -> None:
        from argus.lui.research import parse

        assert parse._EXECUTION.search("what does it cost to close out a $300,000 long in ETH")


@pytest.mark.parametrize(("said", "concept"), [
    ("is BTC a good buy right now", False),
    ("is ETH a good investment", False),
    ("is 10x leverage ok for a small account", True),
])
def test_a_buy_question_is_not_a_definition(said: str, concept: bool) -> None:
    from argus.lui.concepts import concept_asked

    assert (concept_asked(said) is not None) is concept
