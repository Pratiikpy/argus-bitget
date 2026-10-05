"""The pair-trade reader: statistics checked on series whose answer is known, sizing arithmetic
checked by hand, follow-ups resolved from earlier turns, and every door a question can reach it
by. Prices are synthetic and stubbed in; no test touches the network."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from argus.lui.research import pair_trade, rule_test
from argus.research.cointegration import ols

DAYS = 700
START = datetime(2024, 9, 1, tzinfo=UTC)
Q1 = "I want to run a pair trade long NVDA short AMD. Is the spread mean-reverting?"
Q2 = "Given that, how would I size it on a 100k account?"
Q3 = "What would prove this pair trade wrong?"
NVDA_AMD = pair_trade.Pair("NVDAUSDT", "AMDUSDT", True)


def _walk(seed: int, n: int = DAYS, vol: float = 0.02) -> list[float]:
    rng = random.Random(seed)
    out, level = [], math.log(100.0)
    for _ in range(n):
        level += rng.gauss(0.0, vol)
        out.append(level)
    return out


def _ou(seed: int, n: int = DAYS, phi: float = 0.9, vol: float = 0.02) -> list[float]:
    rng = random.Random(seed)
    out, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, vol)
        out.append(x)
    return out


def _series(logs: list[float]) -> tuple[list[datetime], list[float], str]:
    days = [START + timedelta(days=i) for i in range(len(logs))]
    return days, [math.exp(v) for v in logs], "synthetic daily closes"


def _install(monkeypatch: pytest.MonkeyPatch, y: list[float], x: list[float]) -> None:
    table = {"NVDAUSDT": _series(y), "AMDUSDT": _series(x), "ETHUSDT": _series(y),
             "SOLUSDT": _series(x)}
    monkeypatch.setattr(rule_test, "daily_closes", lambda symbol: table[symbol])


def _cointegrated(beta: float = 1.2) -> tuple[list[float], list[float]]:
    x = _walk(1)
    noise = _ou(2)
    return [beta * a + b + 1.0 for a, b in zip(x, noise, strict=True)], x


def test_cointegrated_pair_is_called_mean_reverting_and_the_ratio_is_recovered(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    out = pair_trade.lines(Q1)
    assert out is not None
    assert out[0].startswith("Bottom line: cointegrated at 5%")
    assert "the relation held out of sample" in out[0]
    assert out[-1].startswith("Data: ") and out[-1].endswith("Not advice.")
    study = pair_trade.study(NVDA_AMD)
    assert study.beta == pytest.approx(1.2, abs=0.08)
    assert study.train.cointegrated_at_5pct
    assert study.life_train is not None and 3 < study.life_train < 12  # ln 2 / 0.1 = 6.6 days


def test_independent_walks_are_two_directional_bets(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _walk(11), _walk(12))
    out = pair_trade.lines(Q1)
    assert out is not None
    assert "not mean-reverting" in out[0] and "two directional bets" in out[0]


def test_a_relation_that_breaks_after_the_fit_is_called_broken(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    split = int(DAYS * 0.7)
    drift = _walk(5, DAYS - split, 0.05)
    y = y[:split] + [y[split - 1] + (v - drift[0]) for v in drift]
    _install(monkeypatch, y, x)
    out = pair_trade.lines(Q1)
    assert out is not None
    assert "broke out of sample" in out[0]


def test_hedge_ratio_stability_flags_a_drifting_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    x = _walk(21)
    noise = _ou(22)
    half = DAYS // 2
    y = [(0.6 if i < half else 1.4) * a + b for i, (a, b) in enumerate(zip(x, noise, strict=True))]
    _install(monkeypatch, y, x)
    ok, gap = pair_trade.stable(pair_trade.study(NVDA_AMD))
    assert not ok and gap > 0.25
    out = pair_trade.lines(Q1)
    assert out is not None and any("not stable" in line for line in out)


def test_sizing_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    study = pair_trade.study(NVDA_AMD)
    long_leg, short_leg, each, net = pair_trade._legs(study, 100_000.0)
    assert long_leg + short_leg == pytest.approx(100_000.0)
    assert short_leg / long_leg == pytest.approx(study.beta)
    assert each == pytest.approx(50_000.0)
    assert net == pytest.approx((study.beta - 1) * 50_000.0)
    out = pair_trade.lines(Q2, [Q1])
    assert out is not None
    assert out[0].startswith("Bottom line: on $100,000 gross, hedge-ratio neutral is")
    assert "dollar neutral is $50,000 each side" in out[0]
    assert any(line.startswith("Stop: exit if the spread z-score reaches") for line in out)
    assert any(line.startswith("Costs: 4 legs") and "$120" in line for line in out)


def test_stop_loss_is_the_stated_distance_times_the_long_leg(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    study = pair_trade.study(NVDA_AMD)
    _, entry, stop = pair_trade._levels(study)
    assert abs(stop) >= pair_trade.STOP_FLOOR_Z and abs(stop) >= abs(entry)
    long_leg, _, _, _ = pair_trade._legs(study, 100_000.0)
    line = pair_trade._stop_lines(study, long_leg, 100_000.0)[0]
    loss = long_leg * study.spread_sd * (abs(stop) - abs(entry))
    assert f"${loss:,.0f}" in line


def test_size_without_an_account_says_what_it_assumed(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    out = pair_trade.lines("how would I size it?", [Q1])
    assert out is not None
    assert "$10,000 gross" in out[0]
    assert any("No account size was stated" in line for line in out)


def test_falsifiers_name_each_condition_with_todays_reading(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    out = pair_trade.lines(Q3, [Q1, Q2])
    assert out is not None
    text = " ".join(out)
    assert out[0].startswith("Bottom line: the NVDA/AMD pair trade is wrong if")
    for needle in ("z-score goes beyond", "hedge ratio drifts outside", "rolling 252-day window",
                   "regime break", "0.10"):
        assert needle in text


def test_falsifier_reports_a_condition_already_tripped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _walk(31), _walk(32))
    out = pair_trade.lines(Q3, [Q1])
    assert out is not None
    assert any(line.startswith("Already tripped today:") for line in out)


def test_pair_is_found_from_an_earlier_turn_and_a_new_pair_overrides(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    assert pair_trade._pair_from(Q3, [Q1]) == NVDA_AMD
    assert pair_trade._pair_from("pair trade ETH vs SOL", [Q1]) == pair_trade.Pair(
        "ETHUSDT", "SOLUSDT", False)
    out = pair_trade.lines("pair trade ETH vs SOL")
    assert out is not None and "ETH/SOL" in out[0]


def test_direction_words_set_the_long_leg() -> None:
    pair = pair_trade.read_pair("pair trade short NVDA long AMD, is the spread mean-reverting?")
    assert pair == pair_trade.Pair("AMDUSDT", "NVDAUSDT", True)


@pytest.mark.parametrize("question", [
    "what is the price of NVDA?",
    "how would I size a BTC trade on a 100k account?",
    "what would prove my thesis wrong?",
    "long NVDA short AMD",
    "pair trade",
    "hello",
])
def test_other_questions_are_left_alone(question: str) -> None:
    assert pair_trade.lines(question) is None


def test_a_follow_up_with_no_pair_in_the_history_is_left_alone() -> None:
    assert pair_trade.lines(Q2, ["what is the price of NVDA?"]) is None
    assert pair_trade.lines(Q3, []) is None


def test_a_follow_up_that_names_another_instrument_is_left_alone() -> None:
    assert pair_trade.lines("how would I size BTC on a 100k account?", [Q1]) is None


def test_too_little_shared_history_is_said_not_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _walk(41, 90), _walk(42, 90))
    out = pair_trade.lines(Q1)
    assert out is not None
    assert out[0].startswith("Bottom line: the NVDA/AMD pair could not be studied just now")
    assert "90 daily closes" in out[0]
    assert out[-1].endswith("Not advice.")


def test_a_failed_source_is_one_honest_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(symbol: str) -> tuple[list[datetime], list[float], str]:
        raise OSError("source down")

    monkeypatch.setattr(rule_test, "daily_closes", broken)
    out = pair_trade.lines(Q1)
    assert out is not None
    assert out[0].startswith("Bottom line: the daily history for NVDA and AMD could not be read")
    assert out[-1].endswith("Not advice.")


def test_a_negative_hedge_ratio_is_not_sized_as_a_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    x = _walk(51)
    y = [-0.8 * a + b for a, b in zip(x, _ou(52), strict=True)]
    _install(monkeypatch, y, x)
    out = pair_trade.lines(Q2, [Q1])
    assert out is not None
    assert any("does not hedge" in line for line in out)


def test_the_window_is_the_last_two_years_of_shared_days(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    study = pair_trade.study(NVDA_AMD)
    assert (study.days[-1] - study.days[0]).days <= pair_trade.DAYS
    assert study.split == int(len(study.days) * 0.7)


def test_hedge_ratio_matches_plain_ols_on_the_training_slice(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _cointegrated(1.2)
    _install(monkeypatch, y, x)
    study = pair_trade.study(NVDA_AMD)
    fit = ols(study.ly[:study.split], [study.lx[:study.split], [1.0] * study.split])
    assert study.train.hedge_ratio == pytest.approx(fit.params[0])
    assert study.train_se == pytest.approx(fit.stderr[0])
