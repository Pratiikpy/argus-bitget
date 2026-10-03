"""The research task says whether to enter, where its engines disagree, what would change the call
and what only the trader can answer (lui/weigh.py; research/h2h/RESULTS.md). Figures below are
the ones the live task produced on 2026-09-28 for "Should I enter TSLA?", then varied."""

from __future__ import annotations

from typing import Any

from argus.lui.task import STEPS, Step, Task, as_dict
from argus.lui.task_page import render_task
from argus.lui.weigh import weigh


def engines(*, hit: float = 0.74, median: float = 34.88, episodes: int = 13,
            p25: float = -0.19, hist: float = -0.265, days: int | None = 23,
            target: float | None = None, price: float = 371.06,
            support: float | None = None, long_run: list[dict[str, Any]] | None = None,
            ) -> dict[str, dict[str, Any]]:
    tech: dict[str, Any] = {"macd_histogram": hist, "timeframe": "4h", "price": price}
    if support is not None:
        tech["support"] = support
    fund: dict[str, Any] = {}
    if days is not None:
        fund.update(earnings_days=days, earnings_date="2026-10-21")
    if target is not None:
        fund.update(target_mean=target, price=price, analysts=40)
    analogue: dict[str, Any] = {"analogue": {"usable": True, "distribution": {
        "hit_rate": hit, "median_bps": median, "effective_n": episodes, "p25_bps": p25,
        "p75_bps": 166.7}}}
    if long_run is not None:
        analogue["long_run"] = {"horizons": long_run}
    quote = {"round_trip_bps": 12.27, "quotes": {"TSLAUSDT": {
        "last": str(price), "low_24h": "368.5", "high_24h": "375.2"}}}
    return {"quote": quote, "technicals": {"technicals": tech},
            "fundamentals": {"fundamentals": fund}, "analogue": analogue,
            "impact": {"sizing": {"worst_24h_pct": -8.0}}}


def test_a_base_rate_inside_chance_is_no_edge_and_says_why() -> None:
    w = weigh(engines(), name="TSLA", budget=0.25)
    assert w is not None
    assert w.call == "No measured edge: enter only on your own view"
    assert "13 independent episodes cannot tell 74% from an even chance" in w.reason


def test_the_tape_and_the_base_rate_are_set_against_each_other() -> None:
    w = weigh(engines(), name="TSLA")
    assert w is not None and len(w.disagreements) == 1
    assert "momentum is turning down" in w.disagreements[0]
    assert "rose" in w.disagreements[0] and "Which to weigh: neither —" in w.disagreements[0]
    agreeing = weigh(engines(hist=0.2), name="TSLA")
    assert agreeing is not None and agreeing.disagreements == ()


def test_a_significant_base_rate_that_clears_the_cost_makes_the_call() -> None:
    up = weigh(engines(hit=0.8, episodes=30, median=40), name="TSLA")
    assert up is not None and up.call == "The base rate supports it: enter long"
    against = weigh(engines(hit=0.2, episodes=30, median=-40), name="TSLA")
    assert against is not None and against.call == "The base rate is against it: do not enter long"
    short = weigh(engines(hit=0.2, episodes=30, median=-40), name="TSLA", side="short")
    assert short is not None and short.call == "The base rate supports it: enter short"
    # significant, but the median does not pay the round trip
    thin = weigh(engines(hit=0.8, episodes=30, median=5), name="TSLA")
    assert thin is not None and thin.call.startswith("No measured edge")
    assert "does not clear the round trip" in thin.reason


def test_earnings_inside_a_week_means_wait() -> None:
    w = weigh(engines(days=3, hit=0.8, episodes=30, median=40), name="TSLA")
    assert w is not None and w.call == "Wait: TSLA reports in 3 days"


def test_a_long_run_signal_decides_when_the_hourly_one_cannot() -> None:
    rows = [{"days": 20, "up": 0.80, "base_up": 0.60, "z": 2.4, "episodes": 24,
             "median_bps": 500.0}]
    w = weigh(engines(long_run=rows), name="TSLA")
    assert w is not None and w.call == "The long-run base rate supports it over 20 days"


def test_analysts_against_the_tape_is_a_disagreement_only_past_ten_percent() -> None:
    far = weigh(engines(target=450.0, hist=-0.3, hit=0.3, median=-20), name="TSLA")
    assert far is not None and any("Analysts' mean target is 21% above" in d
                                   for d in far.disagreements)
    near = weigh(engines(target=380.0, hist=-0.3, hit=0.3, median=-20), name="TSLA")
    assert near is not None and not any("Analysts" in d for d in near.disagreements)


def test_triggers_come_from_the_levels_the_quartile_and_the_report() -> None:
    w = weigh(engines(support=360.0, p25=-120.0), name="TSLA")
    assert w is not None
    assert any(t.startswith("Below 360 (3.0% under the price)") for t in w.triggers)
    assert any("fall of more than 1.20% within 24 hours" in t for t in w.triggers)
    assert any("The report on 2026-10-21 (23 days)" in t for t in w.triggers)
    # a quartile that rounds to nothing is not a trigger
    flat = weigh(engines(p25=-0.19), name="TSLA")
    assert flat is not None and not any("within 24 hours is worse" in t for t in flat.triggers)
    # nor is one under a quarter of a percent, which read as a units bug (round 27)
    noise = weigh(engines(p25=-5.0), name="TSLA")
    assert noise is not None and not any("within 24 hours is worse" in t for t in noise.triggers)


def test_the_questions_are_the_inputs_the_task_did_not_have() -> None:
    w = weigh(engines(support=360.0), name="TSLA", budget=0.25)
    assert w is not None
    assert [q.split("?")[0] for q in w.questions] == [
        "How long will you hold it", "What else do you hold",
        "What loss on this position would you accept", "Where would you be wrong"]
    stated = weigh(engines(), name="TSLA", horizon_hours=24, book_given=True, budget_stated=True)
    assert stated is not None and [q.split("?")[0] for q in stated.questions] == [
        "Where would you be wrong"]


def test_nothing_to_weigh_is_none() -> None:
    data = engines(days=None)
    data["analogue"] = {}
    assert weigh(data, name="TSLA") is None


def test_the_page_and_the_json_carry_it() -> None:
    data = engines(support=360.0)
    del data["impact"]  # no sizing, so the page carries the weighing alone
    by_kind = {kind.value: title for title, kind, _ in STEPS if kind is not None}
    steps = [Step(title=t, engine=e, lines=["Bottom line: x"],
                  data=data.get(k.value, {}) if k is not None else {})
             for t, k, e in STEPS]
    assert by_kind["analogue"]
    task = Task(question="Should I enter TSLA?", name="TSLA", size_pct=20.0, book={},
                steps=steps, seconds=1.0, asked="Should I enter TSLA?")
    page = render_task(task, "")
    assert "No measured edge: enter only on your own view" in page
    assert page.index("Where the engines disagree") < page.index("What would change the call") \
        < page.index("Questions only you can answer")
    blob = as_dict(task)
    assert blob["weighing"]["call"].startswith("No measured edge")
    assert blob["weighing"]["figures"]["base_rate"]["episodes"] == 13


def test_without_a_level_in_range_the_24_hour_range_is_the_trigger() -> None:
    w = weigh(engines(), name="TSLA")
    assert w is not None
    assert any(t.startswith("Below 368.5, the last 24 hours' low (0.7% under the price)")
               for t in w.triggers)
    assert any(t.startswith("Above 375.2, the last 24 hours' high (1.1% over the price)")
               for t in w.triggers)
    levels = weigh(engines(support=360.0), name="TSLA")
    assert levels is not None
    assert not any("24 hours' low" in t for t in levels.triggers)
    assert any("24 hours' high" in t for t in levels.triggers)


def test_a_price_at_the_edge_of_its_range_says_so() -> None:
    w = weigh(engines(price=368.6), name="TSLA")
    assert w is not None
    assert any(t.startswith("TSLA is at its 24-hour low (368.5): any further fall breaks the "
                            "day's range") for t in w.triggers)
