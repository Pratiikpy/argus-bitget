"""Whether to enter, where the engines disagree, what would change the call, and what only the
trader can answer — the research task's evidence set against itself.

The research task (`lui/task.py`) runs eight engines and its verdict sizes the position: how much,
given the book. Asked "Should I enter TSLA?" it answered "Size it by its worst day" and never said
whether; its technicals read "momentum turning down" while its own base rate said TSLA rose 74% of
the time from states like today's, and nothing set the two against each other. Two Season-2 entrants
run on the same question did what it did not (research/h2h/RESULTS.md, 2026-09-28):

* **clinch** (Techkeyy/clinch) answers the question asked — "Not enough evidence yet" — names the
  one unresolved point, and gives the price levels that would change the read;
* **Precedent** (oladipsinigami/Precedent) lists where its evidence disagrees, with why each
  conflict matters, and closes with the questions only the trader can answer.

Both are taken, as behaviour (neither licence was needed: nothing is copied). What is ours: every
sentence here is computed from the engines' figures (`Step.data`), never from their prose, and the
call is held to the same standard as the rest of ARGUS — an edge is claimed only when the base
rate differs from chance by two standard errors on independent episodes and its median clears the
round-trip cost. Technical readings never make the call on their own: the systematic signals this
desk tested on these names did not clear costs (research/harvest/quant/), and the disagreement
lines say so where it decides which side to weigh.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from argus.lui.answer import plural

EARNINGS_WAIT_DAYS = 7
"""A report inside this many days makes the call "wait": the position would carry the gap
(the same window the fundamentals engine warns on, `research.fundamentals.EARNINGS_NEAR_DAYS`)."""

MIN_EPISODES = 8
"""The fewest independent episodes a base rate may rest on (`research.analogue.MIN_INDEPENDENT_
EPISODES`)."""

QUARTILE_FLOOR_BPS = 25
"""The smallest 24-hour quartile move worth naming as a trigger: below a quarter of a percent it is
noise, and "a fall of more than 5 bps" read as a units bug (round 27)."""

AT_EDGE_PCT = 0.3
"""A price this close to the day's low or high is at it: a trigger "0.0% under the price" says
nothing a trader can act on."""

TARGET_GAP = 0.10
"""Analysts' mean target this far from the last close counts as a direction worth setting against
the tape; closer than that it is noise around a price."""


@dataclass(frozen=True)
class Weighing:
    """The evidence weighed: one call and the reason for it, then the three lists."""

    call: str
    reason: str
    disagreements: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    figures: dict[str, Any] = field(default_factory=dict)
    """The numbers the call rests on, for the JSON view and for tests."""

    def as_dict(self) -> dict[str, Any]:
        return {"call": self.call, "reason": self.reason,
                "disagreements": list(self.disagreements), "triggers": list(self.triggers),
                "questions": list(self.questions), "figures": self.figures}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _edge(analog: Mapping[str, Any], cost_bps: float | None) -> dict[str, Any] | None:
    """The hourly base rate as a test: its up-rate against an even chance, in standard errors of a
    rate measured on its independent episodes. The engine has no every-hour base rate beside it,
    so an even chance is the null, and the reason says so."""
    dist = analog.get("distribution") or {}
    hit, median, episodes = (_num(dist.get("hit_rate")), _num(dist.get("median_bps")),
                             _num(dist.get("effective_n")))
    if hit is None or median is None or episodes is None or episodes < 1:
        return None
    z = (hit - 0.5) / math.sqrt(0.25 / episodes)
    return {"up": hit, "median_bps": median, "episodes": int(episodes), "z": z,
            "p25_bps": _num(dist.get("p25_bps")), "p75_bps": _num(dist.get("p75_bps")),
            "clears_cost": cost_bps is not None and abs(median) > cost_bps,
            "significant": abs(z) >= 2 and episodes >= MIN_EPISODES}


def _long_run_signal(long_run: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = [r for r in long_run.get("horizons") or []
            if _num(r.get("z")) is not None and int(r.get("episodes") or 0) >= MIN_EPISODES]
    telling = [r for r in rows if abs(float(r["z"])) >= 2]
    return max(telling, key=lambda r: abs(float(r["z"]))) if telling else None


def weigh(data: Mapping[str, Mapping[str, Any]], *, name: str, side: str = "long",
          horizon_hours: float | None = None, book_given: bool = False,
          budget_stated: bool = False, budget: float | None = None) -> Weighing | None:
    """``data`` maps each step's kind (``quote``, ``technicals``, ``fundamentals``, ``analogue``,
    ``impact``) to that step's ``Step.data``. None when neither the base rate nor the earnings
    calendar answered: without them there is nothing to decide whether with."""
    quote, tech = data.get("quote") or {}, (data.get("technicals") or {}).get("technicals") or {}
    fund = (data.get("fundamentals") or {}).get("fundamentals") or {}
    analogue = data.get("analogue") or {}
    sizing = (data.get("impact") or {}).get("sizing") or {}
    cost = _num(quote.get("round_trip_bps"))
    edge = _edge(analogue.get("analogue") or {}, cost) if (analogue.get("analogue") or {}).get(
        "usable", True) else None
    long_signal = _long_run_signal(analogue.get("long_run") or {})
    days = fund.get("earnings_days")
    if edge is None and days is None:
        return None
    wants_up = side != "short"
    figures: dict[str, Any] = {"round_trip_bps": cost, "earnings_days": days}
    if edge is not None:
        figures["base_rate"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in edge.items()}

    # --- the call ----------------------------------------------------------------------------
    if days is not None and int(days) <= EARNINGS_WAIT_DAYS:
        call = f"Wait: {name} reports in {plural(int(days), 'day')}"
        reason = (f"A position opened now carries the earnings gap on {fund.get('earnings_date')}; "
                  f"nothing below prices that gap, so the honest read is after the report.")
    elif edge is not None and edge["significant"] and edge["clears_cost"]:
        leans_up = edge["median_bps"] > 0
        with_you = leans_up == wants_up
        call = (f"The base rate supports it: enter {side}" if with_you else
                f"The base rate is against it: do not enter {side}")
        reason = (f"From states like today's, {name} rose {edge['up']:.0%} of the time over 24 "
                  f"hours, median {edge['median_bps']:+.0f} bps, on {edge['episodes']} independent "
                  f"episodes — {abs(edge['z']):.1f} standard errors from an even chance, and the "
                  f"median clears the {cost:.1f} bps round trip.")
    elif long_signal is not None:
        leans_up = float(long_signal["up"]) > float(long_signal["base_up"])
        with_you = leans_up == wants_up
        call = (f"The long-run base rate supports it over {long_signal['days']} days" if with_you
                else f"The long-run base rate is against it over {long_signal['days']} days")
        reason = (f"Over years of daily closes, states like today's rose {long_signal['up']:.0%} "
                  f"of the time at {long_signal['days']} days against {long_signal['base_up']:.0%}"
                  f" for any day, {abs(float(long_signal['z'])):.1f} standard errors apart; the "
                  f"24-hour base rate is not a measured edge.")
    else:
        call = "No measured edge: enter only on your own view"
        parts = []
        if edge is not None:
            why = ("its median does not clear the round trip" if edge["significant"] else
                   f"{edge['episodes']} independent episodes cannot tell {edge['up']:.0%} from an "
                   f"even chance (±{2 * math.sqrt(0.25 / edge['episodes']):.0%} at two standard "
                   f"errors)")
            parts.append(f"from states like today's {name} rose {edge['up']:.0%} of the time over "
                         f"24 hours, median {edge['median_bps']:+.0f} bps, but {why}")
        parts.append("the long-run daily base rate does not move either" if (
            analogue.get("long_run") or {}).get("horizons") else
            "no long-run base rate answered")
        reason = ("Nothing the engines measured favours entering over waiting: "
                  + "; ".join(parts) + ". Entering is a view you hold, not one the data gives.")

    # --- where the engines disagree ----------------------------------------------------------
    disagreements: list[str] = []
    hist = _num(tech.get("macd_histogram"))
    frame = str(tech.get("timeframe") or "4h")
    if hist is not None and edge is not None and abs(edge["median_bps"]) >= 1 and (
            (hist > 0) != (edge["median_bps"] > 0)):
        tape = "turning up" if hist > 0 else "turning down"
        past = "rose" if edge["median_bps"] > 0 else "fell"
        weight = ("the base rate, the measured one" if edge["significant"] else
                  "neither — the technical signals tested on these names did not clear costs, "
                  "and the base rate is within chance")
        disagreements.append(
            f"The tape says momentum is {tape} (MACD histogram {hist:+.3f}, {frame}); the past "
            f"says states like this {past} — median {edge['median_bps']:+.0f} bps over 24 hours, "
            f"up {edge['up']:.0%} of the time. Which to weigh: {weight}.")
    target, price = _num(fund.get("target_mean")), _num(fund.get("price"))
    if target and price and hist is not None:
        gap = target / price - 1
        if abs(gap) >= TARGET_GAP and (gap > 0) != (hist > 0):
            disagreements.append(
                f"Analysts' mean target is {abs(gap):.0%} {'above' if gap > 0 else 'below'} the "
                f"last close (${target:,.2f} against ${price:,.2f}, {fund.get('analysts')} "
                f"analysts); the tape is turning {'down' if hist < 0 else 'up'}. A target is an "
                f"opinion with no stated horizon; the tape is the last few hours. They answer "
                f"different questions, and neither is a timing signal.")
    if edge is not None and (analogue.get("long_run") or {}).get("horizons"):
        rows = {int(r["days"]): r for r in analogue["long_run"]["horizons"] if "days" in r}
        month = rows.get(20)
        if month is not None and abs(edge["median_bps"]) >= 1:
            later_up = float(month["median_bps"]) > 0
            if later_up != (edge["median_bps"] > 0):
                disagreements.append(
                    f"The horizons disagree: over 24 hours states like this "
                    f"{'rose' if edge['median_bps'] > 0 else 'fell'} (median "
                    f"{edge['median_bps']:+.0f} bps); over 20 trading days they "
                    f"{'rose' if later_up else 'fell'} (median {float(month['median_bps']):+.0f} "
                    f"bps). How long you hold decides which one is yours.")

    # --- what would change the call ----------------------------------------------------------
    triggers: list[str] = []
    spot = _num(tech.get("price"))
    support, resistance = _num(tech.get("support")), _num(tech.get("resistance"))
    if spot and support:
        triggers.append(f"Below {support:g} ({(1 - support / spot) * 100:.1f}% under the price) "
                        f"the nearest support is broken — a {'long' if wants_up else 'short'} "
                        f"case {'loses' if wants_up else 'gains'} its floor.")
    if spot and resistance:
        triggers.append(f"Above {resistance:g} ({(resistance / spot - 1) * 100:.1f}% over the "
                        f"price) the nearest resistance is cleared.")
    if not (support and resistance):
        # bitget-signal gives only levels near the price; with none in range, the last 24 hours'
        # low and high are the structure a trader sees (clinch reads its triggers off the recent
        # range, research/h2h/RESULTS.md), and the line names them as the range, not as levels.
        ticker: Mapping[str, Any] = next(iter((quote.get("quotes") or {}).values()), {})
        low, high, last = (_num(ticker.get("low_24h")), _num(ticker.get("high_24h")),
                           _num(ticker.get("last")))
        if last and low and not support and low <= last:
            gap = (1 - low / last) * 100
            triggers.append(
                (f"{name} is at its 24-hour low ({low:g}): any further fall breaks the day's range"
                 if gap < AT_EDGE_PCT else
                 f"Below {low:g}, the last 24 hours' low ({gap:.1f}% under the price): the range "
                 f"breaks down")
                + " — no support level from bitget-signal is in range, so the range is the "
                  "nearest structure.")
        if last and high and not resistance and high >= last:
            gap = (high / last - 1) * 100
            triggers.append(f"{name} is at its 24-hour high ({high:g}): any further rise breaks "
                            f"the day's range." if gap < AT_EDGE_PCT else
                            f"Above {high:g}, the last 24 hours' high ({gap:.1f}% over the "
                            f"price): the range breaks up.")
    # The quartile against the position: a long is hurt by the bottom quarter, a short by the top.
    against = (edge or {}).get("p25_bps" if wants_up else "p75_bps")
    # "a fall of more than 5 bps" read as a units bug (a first-time user, round 27): a quartile
    # that small is noise, not a trigger, and the move is said in percent
    if against is not None and (against <= -QUARTILE_FLOOR_BPS if wants_up
                                else against >= QUARTILE_FLOOR_BPS):
        triggers.append(f"A {'fall' if wants_up else 'rise'} of more than "
                        f"{abs(against) / 100:.2f}% within 24 hours is worse for a {side} than "
                        f"three in four comparable past states — the point a base-rate case "
                        f"stops describing what is happening.")
    if days is not None and EARNINGS_WAIT_DAYS < int(days) <= 45:
        triggers.append(f"The report on {fund.get('earnings_date')} ({int(days)} days): every "
                        f"figure above is from before it and must be read again after.")
    if not tech and not (quote.get("quotes") or {}):
        triggers.append("No technical levels or live range answered this time, so no price "
                        "trigger is given.")

    # --- questions only the trader can answer ------------------------------------------------
    questions: list[str] = []
    if horizon_hours is None:
        month = next((r for r in (analogue.get("long_run") or {}).get("horizons") or []
                      if int(r.get("days") or 0) == 20), None)
        questions.append(
            "How long will you hold it? The call above reads the next 24 hours"
            + (f"; over 20 trading days states like this rose {float(month['up']):.0%} of the "
               f"time, against {float(month['base_up']):.0%} for any day" if month else "")
            + " — the horizon changes the answer.")
    if not book_given:
        questions.append("What else do you hold? Without a book it is sized by its own worst day; "
                         "with one, by its share of your risk.")
    if not budget_stated:
        questions.append("What loss on this position would you accept? "
                         + (f"The sizing assumed the default {budget:.0%} risk budget."
                            if budget is not None else "No budget was stated."))
    worst = _num(sizing.get("worst_24h_pct"))
    if support or worst is not None:
        questions.append(
            "Where would you be wrong? "
            + (f"The nearest support is {support:g}" if support else "")
            # With a book, the figure is the book's worst window with the name in it, not the
            # name's own: "its worst 24 hours was -2.0%" sat beside TSLA's own -7.83% on the same
            # page (a judge, round 21)
            + (("; the book's" if support else "The book's") if worst is not None and book_given
               else "; its" if support and worst is not None else "Its" if worst is not None
               else "")
            + (f" worst 24 hours in the data{' with it added' if book_given else ''} was "
               f"{worst:+.1f}%" if worst is not None else "")
            + " — decide the exit now, while no move is pushing you either way.")
    return Weighing(call=call, reason=reason, disagreements=tuple(disagreements),
                    triggers=tuple(triggers), questions=tuple(questions), figures=figures)
