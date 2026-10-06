"""The trader's own arithmetic (round 45): every audit input, the careless phrasings, and the
questions the reader must not claim. Fully offline: Bitget's price, tier table, fees and candles
are stubbed, so each expected figure below is worked by hand from the stubbed inputs."""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from argus.lui.research import trade_calc
from argus.lui.research.trade_calc import lines

PRICES = {"BTCUSDT": 86_000.0, "ETHUSDT": 2_700.0, "SOLUSDT": 121.145, "DOGEUSDT": 0.095,
          "EURUSDUSDT": 1.1228, "USDJPYUSDT": 157.85, "GBPUSDUSDT": 1.3228, "XAUUSDT": 4_152.0,
          "XAGUSDT": 50.0}
MMR = {"ETHUSDT": 0.004, "BTCUSDT": 0.004, "DOGEUSDT": 0.0066}
MAX_LEV = {"ETHUSDT": 150.0, "BTCUSDT": 150.0, "DOGEUSDT": 75.0}


def _candles(symbol: str, days: int) -> list[tuple[float, float, float, float]]:
    """Thirty daily candles: every third one falls 2% below its open, the rest 0.5%."""
    base = PRICES.get(symbol, 100.0)
    return [(base, base * 1.01, base * (0.98 if i % 3 == 0 else 0.995), base) for i in range(30)]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trade_calc, "_live", lambda symbol: PRICES.get(symbol))
    monkeypatch.setattr(trade_calc, "_mmr", lambda symbol, notional: MMR.get(symbol))
    monkeypatch.setattr(trade_calc, "_max_lev", lambda symbol, notional: MAX_LEV.get(symbol))
    monkeypatch.setattr(trade_calc, "_perp_fees", lambda symbol: (0.0002, 0.0006))
    monkeypatch.setattr(trade_calc, "_candles", _candles)


def _say(text: str, prior: tuple[str, ...] = ()) -> str:
    got = lines(text, prior)
    assert got is not None, f"expected an answer for: {text}"
    assert got[0].startswith("Bottom line: ")
    assert got[-1].startswith("Data: ")
    joined = "\n".join(got)
    assert not re.search(r"\d[eE][+-]\d", joined), "scientific notation"
    return joined


# --- 1. risk-reward on a stated entry, stop and target -----------------------------------------


def test_risk_reward_with_units() -> None:
    out = _say("Long ETH 2 units entry 2500, stop 2400, target 3000. What's the risk-reward?")
    assert "5.0 to 1" in out
    assert "$200 at risk" in out and "$1,000 to gain" in out
    assert "2,500" in out and "16.7%" in out


def test_risk_reward_forms_the_older_reader_misses() -> None:
    for text in (
        "Long ETH entry 2500 stop 2400 target 3000, risk reward?",
        "bought ETH at 2,500 with a stop-loss 2400 and take profit 3000, what's the R:R",
        "ETH long from 2500, stop 2400, tp 3000 - reward to risk ratio?",
    ):
        assert "5.0 to 1" in _say(text), text


def test_risk_reward_short() -> None:
    out = _say("Short BTC 0.5 BTC entry 90000, stop 92000, target 84000. Risk reward?")
    assert "3.0 to 1" in out
    assert "$1,000 at risk" in out and "$3,000 to gain" in out


def test_target_on_the_losing_side_follow_up() -> None:
    first = "Long ETH 2 units entry 2500, stop 2400, target 3000. What's the risk-reward?"
    out = _say("Now what if the target is 2000 instead?", (first,))
    assert "losing side" in out
    assert "loses 500 a unit ($1,000 on 2 units)" in out
    assert "-5.0 to 1" in out


def test_target_on_the_losing_side_in_one_question() -> None:
    out = _say("Long ETH entry 2500, stop 2400, target 2000, what is the risk reward?")
    assert "losing side" in out and "loses 500 a unit" in out


def test_stop_on_the_wrong_side() -> None:
    first = "Long ETH 2 units entry 2500, stop 2400, target 3000. What's the risk-reward?"
    out = _say("Move the stop to 2600", (first,))
    assert "wrong side" in out
    assert "gain of 100 a unit ($200 on 2 units)" in out
    assert "does not cap the loss" in out


def test_follow_up_changes_only_the_stop() -> None:
    first = ("My position is short 1 BTC from 90000 with a stop at 80000. What is my max loss in "
             "dollars?")
    assert "wrong side" in _say(first)
    out = _say("What if my stop was at 95000 instead?", (first,))
    assert "$5,000 on 1 unit" in out and "5,000 a unit" in out


def test_max_loss_at_the_stop() -> None:
    out = _say("I'm long 3 ETH from 2,500 with a stop at 2,450. What's my max loss?")
    assert "$150" in out


def test_entry_of_zero_is_called_impossible() -> None:
    out = _say("Long ETH entry 0, stop 2400, target 3000, what's the risk-reward?")
    assert "is not a price" in out and "2,700" in out


def test_percentage_levels_are_left_to_the_older_reader() -> None:
    assert lines("Set my stop on SOL at 5% and take profit 10% away, what is the reward to risk?") \
        is None


def test_leveraged_stop_inside_liquidation() -> None:
    out = _say("Long ETH 10x entry 2500, stop 2200, target 3000. Risk reward?")
    assert "liquidation comes first" in out


# --- 2. position sizing on a stated entry and risk percent ------------------------------------


def test_sizing_audit_input() -> None:
    out = _say("I'm long 2 ETH at 2,900 entry. Where should my stop be if I risk 1.5% of a "
               "$50,000 account, and what's the position size?")
    assert "1.5% of $50,000 is $750" in out
    assert "375 (12.9%) below the 2,900 entry, at 2,525" in out
    assert "$5,800, 11.6% of the account" in out
    assert "risking 1.5% of" in out and "1.0%" not in out
    assert "-$400" in out  # 2 ETH from 2,900 with ETH at 2,700


def test_sizing_with_a_stated_stop() -> None:
    out = _say("Long ETH at 2,900, stop 2,800. I risk 2% of a $20,000 account - what size?")
    assert "$400" in out and "4 ETH" in out and "$11,600" in out


def test_sizing_uses_the_stated_percent_not_one_percent() -> None:
    one = _say("long ETH at 2900 stop 2800, risk 1% of a $20,000 account, position size?")
    three = _say("long ETH at 2900 stop 2800, risk 3% of a $20,000 account, position size?")
    assert "2 ETH" in one and "6 ETH" in three


def test_sizing_short_mirrors_the_stop() -> None:
    out = _say("I'm short 2 ETH at 2,900 entry; risk 1.5% of a $50,000 account, where is my stop "
               "and size?")
    assert "above the 2,900 entry, at 3,275" in out


def test_sizing_when_the_stated_size_risks_more() -> None:
    out = _say("Long 10 ETH at 2900 with a stop at 2800, risking 1% of a $50,000 account, what "
               "size should I hold?")
    assert "5 ETH" in out
    assert "would risk $1,000" in out and "2%" in out


def test_sizing_with_no_stop_and_no_size_uses_the_daily_range() -> None:
    out = _say("I risk 1% of a $10,000 account on ETH at 2,700 - what's my stop and position "
               "size?")
    assert "$100" in out and "average daily range" in out


def test_sizing_needs_an_account() -> None:
    assert lines("I risk 1% per trade on ETH, how big should the position be?") is None


# --- 3. a stated fee over a stated pace and horizon -------------------------------------------


def test_fee_audit_input() -> None:
    out = _say("If fees are 0.1 percent per side and I trade 40 times a month, what do I lose in "
               "a year on a 10,000 USDT account?")
    assert "$800 a month" in out and "$9,600" in out and "96.0%" in out
    assert "0.2% of $10,000 is $20.00" in out
    assert "0.06%" in out and "$480 a month" in out  # Bitget's own, as a comparison
    assert "0.12%" not in out


def test_fee_shorthand() -> None:
    out = _say("0.1 percent per side, 40 times a month, a year, 10,000 USDT account - total fees?")
    assert "$9,600" in out and "$800 a month" in out


def test_fee_round_trip_and_daily_pace() -> None:
    out = _say("My fee is 0.05% per round trip, 2 trades a day, on a $20,000 account. What does "
               "it cost me over 30 days?")
    assert "$10.00" in out  # 0.05% of 20,000, once per trade
    assert "$600" in out  # x 2 x 30


def test_fee_without_an_account_gives_percent_only() -> None:
    out = _say("With a 0.1% fee per side and 10 trades a week, how much do I lose in a year?")
    assert "no dollar figure" in out and "104.0%" in out


def test_fee_in_basis_points_per_side() -> None:
    out = _say("a fee of 6 bps per side, 20 times a month, 5,000 USDT trade size, over 6 months")
    # 12 bps round trip = $6.00 a trade, 20 a month = $120, 6 months = $720
    assert "$120 a month" in out and "$720" in out


# --- 4. a fee in basis points on a stated notional --------------------------------------------


def test_bps_fee_on_a_stated_notional() -> None:
    out = _say("I paid a 5 bps fee on a 2,000 USDT trade; how many dollars is that?")
    assert "$1.00" in out and "0.05%" in out


def test_percent_fee_on_a_stated_notional() -> None:
    out = _say("What is a 0.1% fee on a $2,500 trade in dollars?")
    assert "$2.50" in out


def test_bps_and_percent_together_left_alone() -> None:
    assert lines("How big is 0.1 bps versus 0.1% fee on 1 million?") is None


# --- 5. conversions at the live price ---------------------------------------------------------


def test_convert_with_a_decimal_comma() -> None:
    out = _say("Convert 2,5 BTC to USDT")
    assert "215,000" in out and "2,5 read as 2.5" in out


def test_convert_german_format_buys_eth() -> None:
    out = _say("How much would 3.000 USDT buy me in ETH? (German format)")
    # 3,000 / 2,700 = 1.1111
    assert "1.1111 ETH" in out and "3.000 read as 3,000" in out


def test_convert_without_a_format_hint_reads_a_dot_as_decimal() -> None:
    out = _say("Convert 3.5 BTC to USDT")
    assert "301,000" in out


def test_satoshis_per_dollar() -> None:
    out = _say("What is the price of BTC in satoshis per dollar?")
    assert "1,163 satoshis" in out  # 1e8 / 86,000 = 1,162.79


def test_price_in_cents() -> None:
    out = _say("What is SOL's price in cents right now?")
    assert "12,114.5 cents" in out


def test_price_in_cents_and_a_stated_yen_rate() -> None:
    out = _say("What is the price of BTC in Japanese yen at 150 yen per dollar, and ETH in cents?")
    assert "12,900,000 JPY at your rate of 150 per dollar" in out
    assert "157.85" in out and "13,575,100 JPY" in out  # Bitget's own, beside the stated rate
    assert "270,000 cents" in out


def test_euros_yen_and_gold_ounces() -> None:
    out = _say("Show me the BTC price in euros, and in yen, and in gold ounces.")
    assert "76,594 EUR" in out  # 86,000 / 1.1228
    assert "13,575,100 JPY" in out and "20.71 ounces of gold" in out


def test_stated_euro_rate_is_used() -> None:
    out = _say("What is BTC in euros at 1.05 dollars per euro?")
    assert "81,905 EUR at your rate of 1.05" in out and "1.1228" in out


def test_convert_euros_to_a_coin() -> None:
    out = _say("How much ETH can I buy with 5,000 euros?")
    # 5,000 x 1.1228 = 5,614 USD / 2,700
    assert "2.0793 ETH" in out and "$5,614" in out


def test_plain_price_question_is_not_claimed() -> None:
    assert lines("What is the price of BTC?") is None
    assert lines("BTC in USDT") is None


# --- 6. an either-or price --------------------------------------------------------------------


def test_either_or_price() -> None:
    out = _say("Is ETH at 2,700 or 27,000 right now? A friend swears 27,000.")
    assert "2,700, not 27,000" in out and "10.0 times" in out


def test_either_or_neither() -> None:
    out = _say("Is ETH trading at 1,000 or 5,000 right now?")
    assert "neither" in out


def test_either_or_without_a_price_cue_or_with_percent_is_left_alone() -> None:
    assert lines("Should I put 5 or 10 percent in ETH?") is None
    assert lines("ETH 3% or 5%?") is None


# --- 7. return paths --------------------------------------------------------------------------


def test_round_trip_and_loss_then_gain() -> None:
    out = _say("A coin went from 100 to 150 and then back to 100 - what is the total return "
               "percent? And if I lose 50% then gain 50% where do I end?")
    assert "100 to 150 to 100 is 0.0% in total" in out
    assert "takes 100 to 75, -25.0%" in out
    assert "+50.0%, then -33.3%" in out and "100.0% gain" in out


def test_loss_then_gain_on_a_stated_balance() -> None:
    out = _say("I start with $2,000, lose 50% then gain 50%. Where do I end up?")
    assert "$1,500" in out and "-25.0%" in out


def test_gain_then_loss_signed_forms() -> None:
    out = _say("What is my net result after +20% then -20%?")
    assert "96" in out and "-4.0%" in out


def test_single_move_is_not_a_path() -> None:
    assert lines("The coin lost 50%. Where do I end?") is None


# --- 8. leverage with a decimal comma ---------------------------------------------------------


def test_liquidation_with_leverage_written_with_a_comma() -> None:
    out = _say("I short 1000 USDT of ETH at 1,5 leverage with entry 2.700,5 - liquidation?")
    # 1/1.5 - 0.004 = 0.66267; 2,700.5 x 1.66267 = 4,490.03
    assert "1.5x short" in out and "4,490.03" in out and "66.27%" in out
    assert "Margin $666.67, position $1,000" in out
    assert "read as 1.5x" in out and "not as a price" in out
    assert "10x" not in out


def test_liquidation_long_with_a_stated_stop() -> None:
    out = _say("I'm 20x long ETH from 2,700 with a stop at 2,600, where do I get liquidated?")
    # 1/20 - 0.004 = 0.046 -> 2,575.8
    assert "2,575.8" in out
    assert "beyond the liquidation price" not in out and "inside the liquidation price" in out


def test_liquidation_leverage_below_one_long_cannot_be_liquidated_by_a_fall() -> None:
    out = _say("What is the liquidation price of a 0.5x long on ETH from 2,700?")
    assert "cannot be liquidated" in out


def test_liquidation_without_a_side_gives_both() -> None:
    out = _say("ETH 10x from 2,700, where is the liquidation price?")
    assert "Long in ETH" in out or "long in ETH" in out
    assert "short in ETH" in out.replace("Short", "short")


def test_liquidation_left_alone_when_a_move_is_stated() -> None:
    assert lines("10x long BTC, margin 1,000 USDT, BTC drops 5% - am I liquidated?") is None


# --- 9. impossible premises -------------------------------------------------------------------


def test_entry_of_zero_on_a_liquidation_question() -> None:
    out = _say("What is the liquidation price of a 20x long on BTC entered at 0 dollars?")
    assert "An entry of 0 is not a price" in out and "86,000" in out
    # 86,000 x (1 - 0.046) = 82,044
    assert "82,044" in out


def test_negative_price_premise() -> None:
    out = _say("BTC is trading at -5000 dollars right now, so what is my upside if it goes back "
               "to 90000?")
    assert out.splitlines()[0].startswith("Bottom line: No - BTC is not trading at -5,000")
    assert "86,000" in out and "+4.7%" in out


def test_negative_price_without_a_target() -> None:
    out = _say("ETH is priced at minus 300 today")
    assert "cannot be negative" in out and "2,700" in out


# --- 10. 50x on DOGE --------------------------------------------------------------------------


def test_fifty_x_doge() -> None:
    out = _say("50x long on DOGE, margin 10 USDT, liquidation price? safe?")
    # 1/50 - 0.0066 = 0.0134; 0.095 x 0.9866 = 0.093727
    assert "0.09373" in out and "1.34% fall" in out
    assert "Margin $10.00, position $500" in out
    assert "Not safe" in out and "10 of the last 30" in out
    assert "No entry was given" in out


def test_leverage_beyond_the_contract_limit_is_said() -> None:
    out = _say("100x long on DOGE with 10 USDT margin, liquidation price?")
    assert "at most 75x" in out


# --- what must not be claimed ----------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "", "   ", "What is the weather today?", "What is the price of BTC?",
    "Should I buy ETH?", "How is NVDA doing this quarter?", "BTC or ETH, which is stronger?",
    "What is the funding rate on BTCUSDT?", "Explain what a stop loss is.",
    "What is the maximum leverage on SOL?", "Compare BTC and ETH volatility.",
    "Tell me about fees on Bitget.", "What's 15% of 80?", "How much is 2,5 plus 3,5?",
    "Explain risk reward in trading.",
])
def test_not_claimed(text: str) -> None:
    assert lines(text) is None


def test_follow_up_without_a_remembered_trade_is_not_claimed() -> None:
    assert lines("Now what if the target is 2000 instead?") is None
    assert lines("What if my stop was at 95000 instead?", ("What is BTC doing?",)) is None


# --- reading numbers --------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "european", "value"), [
    ("2.700,5", False, 2700.5), ("2,5", False, 2.5), ("2,700", False, 2700.0),
    ("27,000", False, 27000.0), ("3.000", True, 3000.0), ("3.000", False, 3.0),
    ("1.234.567", False, 1234567.0), ("1,234,567.5", False, 1234567.5), ("0,125", False, 0.125),
    ("10k", False, 10000.0), ("1,5m", False, 1500000.0), ("2500", False, 2500.0),
])
def test_number_reading(raw: str, european: bool, value: float) -> None:
    assert trade_calc._val(raw, european)[0] == pytest.approx(value)


def test_formatting_helpers() -> None:
    assert trade_calc._p(2500.0) == "2,500"
    assert trade_calc._p(2700.5) == "2,700.5"
    assert trade_calc._p(0.093727) == "0.09373"
    assert trade_calc._usd(1.0) == "$1.00" and trade_calc._usd(1000.0) == "$1,000"
    assert trade_calc._pcts(0.0006) == "0.06%" and trade_calc._pcts(0.001) == "0.1%"


# --- every answer is well formed -------------------------------------------------------------

ALL_INPUTS = [
    "Long ETH 2 units entry 2500, stop 2400, target 3000. What's the risk-reward?",
    "I'm long 2 ETH at 2,900 entry. Where should my stop be if I risk 1.5% of a $50,000 "
    "account, and what's the position size?",
    "If fees are 0.1 percent per side and I trade 40 times a month, what do I lose in a year on a "
    "10,000 USDT account?",
    "I paid a 5 bps fee on a 2,000 USDT trade; how many dollars is that?",
    "Convert 2,5 BTC to USDT", "How much would 3.000 USDT buy me in ETH? (German format)",
    "What is the price of BTC in satoshis per dollar?", "What is SOL's price in cents right now?",
    "Show me the BTC price in euros, and in yen, and in gold ounces.",
    "What is the price of BTC in Japanese yen at 150 yen per dollar, and ETH in cents?",
    "Is ETH at 2,700 or 27,000 right now? A friend swears 27,000.",
    "A coin went from 100 to 150 and then back to 100 - what is the total return percent? And if "
    "I lose 50% then gain 50% where do I end?",
    "I short 1000 USDT of ETH at 1,5 leverage with entry 2.700,5 - liquidation?",
    "What is the liquidation price of a 20x long on BTC entered at 0 dollars?",
    "BTC is trading at -5000 dollars right now, so what is my upside if it goes back to 90000?",
    "50x long on DOGE, margin 10 USDT, liquidation price? safe?",
]


@pytest.mark.parametrize("text", ALL_INPUTS)
def test_every_audit_input_has_a_bottom_line_and_a_data_line(text: str) -> None:
    out = lines(text)
    assert out is not None
    assert out[0].startswith("Bottom line: ") and out[-1].startswith("Data: ")
    assert all(isinstance(x, str) and x.strip() for x in out)
    assert not any(re.search(r"\d[eE][+-]\d|\bnan\b|\binf\b", x) for x in out)


# --- the data being unavailable is said, not hidden ---------------------------------------------


def test_a_missing_price_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trade_calc, "_live", lambda symbol: None)
    out = _say("Convert 2,5 BTC to USDT") if lines("Convert 2,5 BTC to USDT") else ""
    assert "did not answer" in out
    either = lines("Is ETH at 2,700 or 27,000 right now?")
    assert either is not None and "did not answer" in either[0]
    zero = lines("What is the liquidation price of a 20x long on BTC entered at 0 dollars?")
    assert zero is not None and "did not answer" in zero[0]


def test_a_missing_tier_table_is_said(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trade_calc, "_mmr", lambda symbol, notional: None)
    out = _say("50x long on DOGE, margin 10 USDT, liquidation price? safe?")
    assert "did not answer, so none is deducted" in out


def test_missing_candles_skip_only_the_history_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trade_calc, "_candles", lambda symbol, days: [])
    out = _say("50x long on DOGE, margin 10 USDT, liquidation price? safe?")
    assert "Not safe" not in out and "0.09373" in out


def test_entry_points_take_a_callable_prior() -> None:
    fn: Callable[..., object] = lines
    assert fn("Convert 2,5 BTC to USDT", []) is not None


# --- found by sweeping the console's own question corpus through the reader --------------------


def test_figures_that_do_not_agree_are_named() -> None:
    out = _say("I put $1,000 margin on 5x leverage to hold $20,000 of ETH. What is my "
               "liquidation distance?")
    assert "do not agree" in out and "$5,000 position, not $20,000" in out


def test_two_different_sizes_for_one_position_are_not_guessed() -> None:
    assert lines("I have 5 BTC long at 10x leverage, which is also 8 BTC long at 10x leverage, "
                 "what's my liquidation price?") is None


def test_two_different_leverages_are_not_guessed() -> None:
    assert lines("My ETH long has 3x leverage and 2x leverage at once, what is the liquidation "
                 "price?") is None


def test_leverage_with_a_thousands_comma_is_not_ten() -> None:
    out = _say("What would a 10,000x leveraged long on BTCUSDT liquidate at?")
    assert "10,000x" in out and "the moment it opens" in out and "at most 150x" in out


def test_negative_leverage_has_no_sign() -> None:
    out = _say("What's my liquidation price for a BTC long with -10x leverage?")
    assert "Leverage has no sign" in out and "10x long" in out


def test_leverage_at_or_below_one_cannot_be_liquidated_on_a_long() -> None:
    out = _say("What is the liquidation price of a 0.5x long on ETH from 2,700?")
    assert "cannot be liquidated by the price falling" in out and ".." not in out


def test_percentage_stop_against_liquidation() -> None:
    out = _say("BTC at 86000, I want to long with 20x leverage and put my stop at 1% below, will "
               "I be liquidated first?")
    assert "inside the 4.60% liquidation distance: it triggers first" in out


def test_sizing_from_a_percentage_stop() -> None:
    out = _say("position size for ETH at 2,700 with a 3% stop, risk 1% of a 50k account")
    # $500 risk / 3% = $16,666.67; the stop is 81 below 2,700
    assert "$16,666.67" in out and "2,619" in out


def test_a_bought_price_below_zero() -> None:
    out = _say("What's my P&L if I bought 1 BTC at -50000 and it's now at the current price?")
    assert "nothing can be bought at -50,000" in out and "86,000" in out


def test_a_move_the_reader_does_not_know_is_not_skipped() -> None:
    out = _say("If ETH falls 5%, then rebounds 3%, then falls 2%, where does my money end?")
    # 100 -> 95 -> 97.85 -> 95.893
    assert "95.89" in out and "-4.1%" in out
    assert lines("If ETH falls 5%, then wobbles 3%, then falls 2%, where does my money end?") \
        is None


def test_a_stated_balance_in_the_path_is_used() -> None:
    out = _say("account was $40k, lost 30% then gained 30%, where am I now?")
    assert "$36,400" in out and "-9.0%" in out


def test_a_second_amount_of_money_blocks_a_conversion() -> None:
    assert lines("BTC is trading at 92,000 EUR right now. I have $10,000 USD -- how many BTC can "
                 "I buy?") is None
    assert lines("NVDA price in euros 2000.50 EUR per share, 100 shares worth how many USD?") \
        is None


@pytest.mark.parametrize("text", [
    "put $500 into BTC?", "should I DCA into BTC with $1,000?",
    "Back when BTC was trading at $120,000 last month, was that a local top?",
    "is NVDA liquid enough for a $250k buy?", "I have R$100,000. How many BTC can I buy?",
    "If I put $100 a week into bitcoin for a year what would I have?",
    "Solana daily fees +32.7% (75% percentile)",
    "What's the maximum I could lose on a $50,000 account?",
])
def test_ordinary_questions_with_money_are_not_claimed(text: str) -> None:
    assert lines(text) is None


def test_fee_needs_a_trading_context() -> None:
    assert lines("Solana daily fees rose 32.7% last month, per side of the chain") is None


def test_bps_display() -> None:
    out = _say("What is a 0.1% fee on a $2,500 trade in dollars?")
    assert "a 0.1% fee on $2,500 is $2.50" in out
    assert "a 5 bps fee" in _say("I paid a 5 bps fee on a 2,000 USDT trade; how many dollars?")
