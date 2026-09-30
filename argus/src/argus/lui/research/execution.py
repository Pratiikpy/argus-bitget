"""How to execute a size: the cost of each slice, the schedule, and what the live book absorbs."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.lui.question import (
    TRADED_SYMBOLS,
)
from argus.lui.research.data import (
    _FETCH_SLOTS,
)
from argus.lui.research.parse import (
    is_us_equity,
)
from argus.lui.research.session import (
    anchor_is_open,
)
from argus.lui.trace import trace_module

MAX_HOURLY_PARTICIPATION = Decimal("0.10")
"""A slice is spaced so the order never takes more than this share of an hour's volume."""


_SELL_WORDS = re.compile(r"\b(?:sell\w*|liquidat\w*|exit\w*|dump\w*|unload\w*|trim\w*|"
                         r"reduce\w*|close\s+(?:out|my))\b", re.I)


SCHEDULE_DECAYS = {False: 1.0, True: 3.0}
"""How far the optimal schedule front-loads, as kappa times the horizon: inventory decays e-fold
over the run by default, e-cubed when the order is urgent. Almgren and Chriss leave the risk
aversion to the trader; stating it as a decay over the horizon makes the choice readable and
independent of the order's size, where a raw lambda is neither."""


MAX_SCHEDULE_HOURS = 24


def _session_change(start: datetime, hours: int) -> tuple[int, str] | None:
    """The first whole hour within ``hours`` at which the stock's own market opens or closes,
    from the desk's session clock (`truth/clocks.py`, holidays included)."""
    is_open = anchor_is_open()
    state = is_open(start)
    for h in range(1, hours + 1):
        if is_open(start + timedelta(hours=h)) != state:
            return h, ("closes" if state else "opens")
    return None


def _waiting_cost(hourly_moves: Sequence[float]) -> str:
    """What deciding slowly costs: the expected drift of the price between making up your mind
    and the first child order landing, from `execution/latency.py` — OWNED against hftbacktest,
    whose latency model has no input for a delay measured in minutes of deliberation.

    The module's depth multiplier is left at 1 here on purpose. It exists to scale *slippage* for
    a thinner book; the drift of the price over a delay is volatility's, and the 48 hourly moves it
    is measured on already include the hours the stock's market is shut. Tripling it off-hours
    printed 9.4bps a minute for NVDA (2026-09-24), which double-counts the session."""
    import statistics

    from argus.execution.latency import NANOS_PER_SECOND, latency_slippage_bps

    annual = Decimal(str(statistics.pstdev(hourly_moves) * math.sqrt(24 * 365.25)))

    def after(seconds: int) -> float:
        return float(latency_slippage_bps(latency_ns=seconds * NANOS_PER_SECOND,
                                          annualised_vol=annual))

    return (f"Waiting costs too: at this volatility the price drifts an expected "
            f"{after(60):.1f}bps in the minute between deciding and the first order landing, "
            f"{after(600):.1f}bps in ten — a cost of deliberating, separate from the impact "
            f"above.")


def _daily_volatility_bps(symbol: str, days: int = 30) -> Decimal | None:
    """The name's daily return volatility over ``days`` daily bars, in bps — the sigma the
    square-root impact law scales by (`cost/model.py`)."""
    import statistics

    from argus.market.history import CandleType, fetch

    try:
        with _FETCH_SLOTS:
            bars = fetch(symbol, interval="1D", candle_type=CandleType.MARKET, recent=True,
                         limit=days + 1)
    except Exception:
        return None
    closes = [float(b.close) for b in bars]
    moves = [b / a - 1.0 for a, b in itertools.pairwise(closes) if a > 0]
    if len(moves) < 10:
        return None
    return Decimal(str(round(statistics.pstdev(moves) * 10_000, 1)))


def _optimal_schedule(symbol: str, notional: Decimal, adv: Decimal, book: Any, fee_bps: float,
                      total_hours: float, urgent: bool, side: str = "BUY") -> list[str]:
    """The Almgren-Chriss (2000) optimal trajectory for this order, priced against an even split
    under the same impact model, and stopped at the next session boundary.

    `execution/schedule.py` is OWNED against TWAP — the even split every execution tool ships by
    default (Nautilus's `twap.rs` read in full) — and the console used to print exactly that even
    split ("child orders of about $3,712 every 5 minutes"). Its impact is calibrated on the live
    book, not on a rule of thumb: temporary impact is what sweeping one hour's share of the order
    costs across the 50 visible levels right now, beyond half the spread (no replenishment within
    the hour, so it is the conservative reading); permanent impact is a tenth of it, the ratio in
    the paper's own worked example (one spread per 1% of daily volume temporary, per 10%
    permanent); the fixed cost is half the spread plus the fee. The paper's spread-per-volume rule
    was tried first and priced a $134k NVDA run at 0.4bps of impact beside a book that measured
    11bps for the same size (2026-09-24) — two numbers in one answer that could not both be true.
    Volatility is the last 48 hourly bars. And a schedule that runs through the stock's
    own open or close is solving a problem whose liquidity changes halfway: the schedule is
    computed only up to the boundary, and says so, rather than assuming today's book survives it.
    """
    import statistics

    from argus.execution.schedule import ImpactParameters, ScheduleError, trajectory
    from argus.market.history import CandleType, fetch

    try:
        price = book.mid
        spread = book.asks[0].price - book.bids[0].price
        with _FETCH_SLOTS:
            bars = fetch(symbol, interval="1H", candle_type=CandleType.MARKET, recent=True,
                         limit=49)
    except Exception:
        return []
    closes = [float(b.close) for b in bars]
    moves = [b / a - 1.0 for a, b in itertools.pairwise(closes) if a > 0]
    if price <= 0 or spread <= 0 or len(moves) < 24:
        return []
    now = datetime.now(UTC)
    hours = min(MAX_SCHEDULE_HOURS, max(2, math.ceil(total_hours)))
    # Only a contract with a US-stock anchor has a session to plan around: "the stock's own
    # market opens in about 3 hours" was said of DOGE, SOL and AVAX orders and cut their schedules
    # short (2026-09-25 audit, round 2).
    anchored = symbol in TRADED_SYMBOLS or is_us_equity(symbol)
    change = _session_change(now, hours) if anchored else None
    share = Decimal(1)
    if change is not None and change[0] < hours:
        hours = max(1, change[0])
        share = min(Decimal(1), (adv / 24 * MAX_HOURLY_PARTICIPATION * hours) / notional)
    part = notional * share
    shares = part / price
    sigma = Decimal(str(statistics.pstdev(moves))) * price
    hour_notional = min(part, adv / 24 * MAX_HOURLY_PARTICIPATION)
    try:
        swept = book.sweep(hour_notional, direction=side)
    except Exception:
        return []
    walk = swept.slippage_bps / 10_000 * price - spread / 2
    eta = max(walk, spread / 2) / (hour_notional / price)
    impact = ImpactParameters(sigma=sigma, gamma=eta / 10, eta=eta,
                              epsilon=spread / 2 + price * Decimal(str(fee_bps)) / 10_000)
    kappa = SCHEDULE_DECAYS[urgent] / hours
    eta_tilde = impact.eta - impact.gamma / 2
    kappa_tilde_sq = 2 * (math.cosh(kappa) - 1.0)
    try:
        risk_aversion = Decimal(str(kappa_tilde_sq)) * eta_tilde / (sigma * sigma)
        best = trajectory(quantity=shares, horizon=Decimal(hours), intervals=hours,
                          impact=impact, risk_aversion=risk_aversion)
        even = trajectory(quantity=shares, horizon=Decimal(hours), intervals=hours,
                          impact=impact, risk_aversion=Decimal(0))
    except (ScheduleError, ArithmeticError, ValueError):
        return []

    def bps(amount: Decimal) -> float:
        return float(amount / part * 10_000)

    shape = ", ".join(f"{float(sl.quantity / shares):.0%}" for sl in best.slices[:8])
    more = "" if len(best.slices) <= 8 else f", then the last {len(best.slices) - 8} hours"
    lines = [
        f"Schedule (Almgren-Chriss optimum, inventory decaying "
        f"{'e-cubed' if urgent else 'e-fold'} over {hours} hour(s)): trade {shape}{more} of "
        f"${float(part):,.0f} hour by hour — expected cost {bps(best.expected_cost):.1f}bps with "
        f"a one-standard-deviation risk of ±{bps(best.variance.sqrt()):.1f}bps, against "
        f"{bps(even.expected_cost):.1f}bps ± {bps(even.variance.sqrt()):.1f}bps for an even "
        f"split, with the slice styles above applied inside each hour. Impact is priced from "
        f"sweeping one hour's share of the order on the live book"
        + ("" if swept.complete else " (deeper than the visible 50 levels, so at least this)")
        + "."]
    worst = _worst_case_line(shares, hours, impact, bps)
    if worst:
        lines.append(worst)
    cadence = _cadence_line(float(part))
    if cadence:
        lines.append(cadence)
    lines.append(_waiting_cost(moves))
    if change is not None and share < 1:
        lines.append(
            f"Session: the stock's own market {change[1]} in about {change[0]} hour(s), and depth "
            f"on the token changes with it, so the schedule covers only the "
            f"${float(part):,.0f} that fits before then at {MAX_HOURLY_PARTICIPATION:.0%} of an "
            f"hour's volume — plan the remaining ${float(notional - part):,.0f} after it "
            f"{change[1]} rather than assume this book survives it.")
    elif change is not None:
        lines.append(f"Session: the stock's own market {change[1]} in about {change[0]} "
                     f"hour(s); the whole order fits before then.")
    return lines


WORST_CASE_CONFIDENCE = 0.95


def _worst_case_line(shares: Decimal, hours: int, impact: Any,
                     bps: Any) -> str | None:
    """The schedule that keeps the 95% worst-case execution cost lowest, read off the swept
    cost-risk frontier (`execution/frontier.py`, Almgren and Chriss 2000 eq. 22), beside an even
    split's: a trader can set execution risk as a loss they will tolerate, not as a decay rate."""
    from argus.execution.frontier import choose_by_confidence, efficient_frontier
    from argus.execution.schedule import ScheduleError

    try:
        frontier = efficient_frontier(quantity=shares, horizon=Decimal(hours), intervals=hours,
                                      impact=impact)
        choice = choose_by_confidence(frontier, WORST_CASE_CONFIDENCE)
    except (ScheduleError, ArithmeticError, ValueError):
        return None
    return (f"Worst case: at {WORST_CASE_CONFIDENCE:.0%} confidence, the schedule that keeps the "
            f"cost lowest does {float(choice.point.front_loading):.0%} of the order in the first "
            f"half, and its {WORST_CASE_CONFIDENCE:.0%} worst case is "
            f"{bps(choice.value_at_risk):.1f}bps against {bps(choice.twap_value_at_risk):.1f}bps "
            f"for an even split — read off {len(frontier)} optimal schedules, cost against risk.")


def _cadence_line(notional: float) -> str | None:
    """How often to send the children inside each hour, from the full-depth replay in
    `eval/execution_arena.py`: the hour's amount in one child paid about twice what one-minute
    children paid on the same book, the same finding that makes Bitget's own 60-second TWAP the
    incumbent to match."""
    from argus.lui.answer import desk_notes_path

    try:
        report = json.loads((desk_notes_path().parent / "execution_arena.json")
                            .read_text(encoding="utf-8"))
        sizes = report["by_size"]
    except (OSError, ValueError, KeyError):
        return None
    key = min(sizes, key=lambda k: abs(float(k.replace(",", "")) - notional))
    cost = sizes[key]["cost_bps"]
    return (f"Cadence: send each hour's share as one-minute children, as Bitget's own TWAP does at "
            f"a 60-second interval. Replaying a full day of Bitget's NVDA order book, ${key} "
            f"orders cost {cost['argus_ac_60s']:.1f}bps with fees in one-minute children against "
            f"{cost['argus_ac']:.1f}bps in hourly ones (Bitget's TWAP: "
            f"{cost['bitget_twap_60s']:.1f}bps; all at once: {cost['immediate']:.1f}bps).")


def _depth_lines(symbol: str, notional: Decimal, adv: Decimal, plan: Any,
                 raw_text: str, *, urgent: bool = False,
                 modelled: float | None = None,
                 measured: dict[str, float] | None = None) -> list[str]:
    """The live order book behind the plan: what taking the whole order at once costs, what each
    slice costs when swept alone, and how far apart to space the slices. The plan above prices a
    slice by its style alone, which is why every slice of a $50k NVDA order read "6bps" — the book
    says what the size itself does (a judge's probe, 2026-09-24).

    ``measured``, when given, receives the two all-in figures the lines state — ``single_order_bps``
    (fee plus the whole order swept) and ``sliced_bps`` (each slice's fee plus its own sweep,
    weighted) — so a verdict can quote the page's figure rather than a third estimate. Left empty
    when the book did not answer or ran out before the order did."""
    from argus.market.depth import fetch_orderbook

    side = "SELL" if _SELL_WORDS.search(raw_text) else "BUY"
    try:
        with _FETCH_SLOTS:
            book = fetch_orderbook(symbol, limit=50)
        whole = book.sweep(notional, direction=side)
    except Exception:
        return [f"Slice {s.index}: {s.fraction:.0%} as {s.style}, ~{s.expected_cost_bps:.1f}bps "
                f"(the live order book did not answer, so size impact is not included)"
                for s in plan.slices]
    lines: list[str] = []
    fee = float(plan.slices[0].expected_cost_bps) if plan.slices else 6.0
    reach = ("" if whole.complete else
             f" — the visible 50 levels hold only ${float(whole.filled_notional):,.0f}, so the "
             f"rest would fill beyond what can be seen")
    lines.append(
        f"Order book ({side.lower()} side, live): taking all ${float(notional):,.0f} at once "
        f"walks {whole.levels_consumed} level(s) for {float(whole.slippage_bps):.1f}bps of "
        f"slippage on top of the fee{reach}.")
    hourly = adv / Decimal(24)
    sliced: float | None = 0.0 if whole.complete else None
    slice_lines: list[str] = []
    for part in plan.slices:
        size = notional * part.fraction
        try:
            swept = book.sweep(size, direction=side)
            impact = (f"{float(swept.slippage_bps):.1f}bps book impact" if swept.complete else
                      f"at least {float(swept.slippage_bps):.1f}bps book impact — larger than the "
                      f"visible book")
            if sliced is not None and swept.complete:
                sliced += float(part.fraction) * (float(part.expected_cost_bps)
                                                  + float(swept.slippage_bps))
            else:
                sliced = None
        except Exception:
            impact = "book impact unreadable"
            sliced = None
        slice_lines.append(f"Slice {part.index}: {part.fraction:.0%} (${float(size):,.0f}) as "
                           f"{part.style} — about {float(part.expected_cost_bps):.1f}bps fee "
                           f"plus {impact}.")
    total_hours = (float(notional / (hourly * MAX_HOURLY_PARTICIPATION)) if hourly > 0
                   else 0.0)
    optimal = (_optimal_schedule(symbol, notional, adv, book, fee, total_hours, urgent, side)
               if total_hours > 1 else [])
    if optimal and len(plan.slices) > 1:
        # Worked over hours, the slices are an order-type mix inside each hour's share, not three
        # blocks sent now; quoting each block's own sweep read as a second, rival plan beside the
        # schedule (stranger QA, 2026-09-29).
        mix = ", ".join(f"{part.fraction:.0%} as {part.style}" for part in plan.slices)
        lines.append(f"Order types inside each hour's share: {mix}. The timing is the schedule "
                     f"below; the one plan is the two together.")
    else:
        lines.extend(slice_lines)
    if optimal:
        lines.extend(optimal)
    elif total_hours > 1:
        # Bigger than an hour of fair participation: the slices are a style mix worked across a
        # schedule of small child orders, not three big blocks — each block alone would sweep
        # past the visible book.
        child = hourly * MAX_HOURLY_PARTICIPATION / 12
        lines.append(
            f"Schedule: at no more than {MAX_HOURLY_PARTICIPATION:.0%} of an hour's volume the "
            f"order takes about "
            + ("an hour" if total_hours < 1.5 else f"{total_hours:.0f} hours" if total_hours < 48
               else f"{total_hours / 24:.1f} days")
            + f" — work it as child orders of about ${float(child):,.0f} every 5 minutes, "
              f"applying the style mix above across the run.")
    elif len(plan.slices) > 1:
        gap = total_hours * 60 / len(plan.slices)
        lines.append(f"Schedule: space the {len(plan.slices)} slices about {max(gap, 1):.0f} "
                     f"minute(s) apart, so the order never takes more than "
                     f"{MAX_HOURLY_PARTICIPATION:.0%} of an hour's volume.")
    one_shot = fee + float(whole.slippage_bps)
    if measured is not None and sliced is not None:
        measured.update(single_order_bps=one_shot, sliced_bps=sliced)
    at_least = "" if whole.complete else "at least "
    # The verdict follows the plan, not a threshold of its own. It used to say "splitting on the
    # schedule below is worth doing" whenever the sweep cost 2bps or more — beside a plan that
    # had just said one order is cheapest (a $15k TSLA order, 2026-09-24).
    if not whole.complete:
        # The visible book ran out before the order did, so the sweep is a floor, not a cost, and
        # "the book absorbs it" was printed beside a $17m BTC order the book held a tenth of
        # (2026-09-25 audit). The impact law's figure is the better estimate of the rest.
        law = (f"; the impact law puts it nearer {fee + modelled:.1f}bps all in"
               if modelled is not None and fee + modelled > one_shot else "")
        verdict = (f" — more than the visible book holds (${float(whole.filled_notional):,.0f} of "
                   f"${float(notional):,.0f}){law}, so work it on the schedule below rather than "
                   f"at once.")
    elif len(plan.slices) <= 1:
        verdict = (" — one order is the plan; the book impact is the price of filling now, and a "
                   "limit at the touch saves it if the fill can wait."
                   if float(whole.slippage_bps) >= 2 else
                   " — the book absorbs it; one order is the plan.")
    else:
        verdict = (" — the book absorbs it, so splitting saves little beyond the passive fills."
                   if float(whole.slippage_bps) < 2 else
                   " — enough book impact that splitting on the schedule below is worth doing.")
    # The cost in dollars beside the basis points: "what does it cost to buy $500 of BTC" was
    # answered "6.0bps", a unit a newcomer asking it does not read (2026-09-30).
    dollars = float(notional) * one_shot / 10_000
    lines.insert(0, (
        f"Bottom line: in one market order this costs {at_least or 'about '}{one_shot:.1f}bps "
        f"({fee:.1f}bps fee + {at_least}{float(whole.slippage_bps):.1f}bps book impact), "
        f"{at_least or 'about '}${dollars:,.2f} on ${float(notional):,.0f}"
        + verdict))
    return lines


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
