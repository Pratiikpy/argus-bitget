"""Answering a request: :func:`run` reads the question, routes it to its engine and adds the lines
every answer carries."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from argus.desk.portfolio import (
    PortfolioError,
    Shock,
    align,
    beta,
    copilot,
    correlation,
    stress_by_beta,
    variance,
)
from argus.lui.answer import LEAD, Answer, Source, unlead
from argus.lui.question import (
    TRADED_SYMBOLS,
    Question,
)
from argus.lui.research.analogue import (
    _NO_CANDLES,
    _analogue,
    _analogue_data,
    _long_run,
    _no_candles,
    _odds_lines,
    _scope_lead,
)
from argus.lui.research.anchor import (
    _premium_line,
)
from argus.lui.research.book import (
    _BOOK_HISTORY_Q,
    _WORTH_Q,
    _book_history_lines,
    _book_report,
    _equal_risk_weights,
    _event_reaction,
    _exposure_future,
    _exposure_lines,
    _hedge_line,
    _hedge_plan,
    _impact_lines,
    _within_limits,
    impact_sizing,
)
from argus.lui.research.claims import (
    SESSION_CLAIM,
    _claim_check,
)
from argus.lui.research.data import (
    load,
)
from argus.lui.research.evidence import (
    _WITH_ITS_PERP,
    _beta_track_record,
    _crowd_lines,
    _flow_lines,
    _same_name_perp_hedge,
    _spot_hedge_answer,
)
from argus.lui.research.execution import (
    _SELL_WORDS,
    _daily_volatility_bps,
    _depth_lines,
)
from argus.lui.research.fundamentals import (
    _VALUATION,
    _as_of,
    _fundamentals,
    _lead_with_what_was_asked,
    _valuation_compare,
)
from argus.lui.research.kinds import (
    BENCHMARK,
    CRYPTO_ANCHOR,
    DEFAULT_SIZE,
    ResearchKind,
    ResearchRequest,
    _t,
)
from argus.lui.research.macro import (
    _macro,
    fed_premise_line,
)
from argus.lui.research.news import (
    _news,
    tone_asked,
)
from argus.lui.research.parse import (
    _LOSS_OVER_PERIOD,
    _ROUND_TRIP,
    _SINGLE_NAME,
    _STOP_QUESTION,
    _TAKE_PROFIT_Q,
    _VAR,
    _WEEKEND_GAP_Q,
    CRYPTO_ETF_QUESTION,
    DARK_POOL_Q,
    HEDGE_BOOK_VALUE,
    IMPLIED_OPEN_QUESTION,
    LONG_SHORT_QUESTION,
    OPEN_INTEREST_QUESTION,
    OPTIONS_POSITIONING_Q,
    PRICE_AT,
    SHORT_FLOW_Q,
    _idea_request,
    _mandate_lines,
    daily_technicals_asked,
    is_us_equity,
    leveraged_fund_asked,
    parse_notional,
    stated_limits,
    unread_holdings,
    without_hedges,
)
from argus.lui.research.quote import (
    _PRICE_ASKED,
    _daily_technicals,
    _implied_open_line,
    _lead_with,
    _quote_extras,
    _rtoken_market_lines,
    _session_line,
    _sized_round_trip,
    _technicals_computed,
)
from argus.lui.research.riskmath import (
    _BETA_ASKED,
    _book_beta_line,
    _book_dollar_lines,
    _desk_view,
    _open_columns,
    _resolve_resize,
    _var_lines,
    _vol_multiple,
)
from argus.lui.research.sentiment import (
    _sentiment,
)
from argus.lui.research.session import (
    anchor_is_open,
)
from argus.lui.research.technicals import (
    _answer_the_level_asked,
    _answer_the_state_asked,
    _technicals,
)
from argus.lui.research.text import (
    _question,
    clean_line,
)
from argus.lui.research.venue import (
    _crypto_anchor_line,
    _distribution_line,
    _funding_meaning,
    _leverage,
    _venue,
)
from argus.lui.trace import trace_module
from argus.truth import coverage
from argus.truth.coverage import ContextPool


def run(raw_text: str, request: ResearchRequest, *, ledger: Any = None) -> Answer:
    """Answer a research request, then test any claim the question itself makes about the
    contract's funding or its premium to the stock.

    "Long NVDA perp into earnings — funding looks cheap" is two questions: the earnings one the
    kind answers, and a stated premise about funding that no kind owns. Answered as fundamentals
    alone, the premise went unchecked while a rival desk (optic-bitget, run on the same thesis
    2026-09-25) reported the funding rate beside it. The premise is now measured against the
    contract's own settlement history and the verdict stated: holds, does not hold, or unclear.
    """
    request = _idea_request(raw_text, request)
    with coverage.recording() as reached:
        answer = _run(raw_text, request, ledger=ledger)
        symbol = request.symbols[0] if request.symbols else ""
        if (not answer.refused and symbol and is_us_equity(symbol)
                and _INTO_EARNINGS.search(raw_text)
                and request.kind in (ResearchKind.IMPACT, ResearchKind.STRESS,
                                     ResearchKind.ANALOGUE, ResearchKind.LEVERAGE)):
            # "…into earnings": the release this position would sit through, measured on the
            # name's own history rather than left to the trader to look up.
            night = _earnings_night_lines(symbol, request.size if request.kind is
                                          ResearchKind.IMPACT else None)
            lead = next((i for i, line in enumerate(answer.lines)
                         if bool(LEAD.match(line))), -1)
            answer.lines[lead + 1:lead + 1] = night
            if night:
                answer.sources.append(Source(kind="evidence", ref="SEC EDGAR 8-K item 2.02",
                                             detail=f"{_t(symbol)} results releases; Yahoo "
                                                    f"daily prices"))
        if not answer.refused and (symbol or SESSION_CLAIM.search(raw_text)):
            checked = _claim_check(raw_text, symbol)
            if checked:
                lines, sources = checked
                # The premise leads: it is what the person actually asserted, and an earnings
                # date above "the perp trades at a premium — holds" answered a question nobody
                # asked.
                answer.lines[0:0] = lines
                answer.sources.extend(sources)
            scoped, scoped_sources = _scope_lead(raw_text, symbol, answer.lines)
            answer.sources.extend(scoped_sources)
            if scoped:
                answer.lines[:] = [*scoped, *(re.sub(r"^(?:Actionable|Bottom line)(?: "
                                                     r"\(\w+\))?:\s*(\w)",
                                                     lambda m: m.group(1).upper(), line)
                                              for line in answer.lines)]
            as_of = _as_of(raw_text)
            if as_of is not None and as_of < datetime.now(UTC) - timedelta(days=1):
                _point_in_time(answer, as_of)
            elif symbol and (request.kind in _PREDICTION_KINDS or (
                    request.kind is ResearchKind.ANALOGUE and request.horizon_hours is not None)):
                _add_prediction_markets(answer, symbol)
    # A sentence said twice is noise: "Funding means longs pay shorts every 8h" appeared three
    # times in one funding answer (answer audit, round 3). A line whose whole text already sits
    # inside an earlier line is dropped.
    kept_lines: list[str] = []
    for line in answer.lines:
        bare = LEAD.sub("", line, count=1).strip()
        if bare and any(bare in LEAD.sub("", earlier, count=1)
                        for earlier in kept_lines):
            continue
        kept_lines.append(line)
    answer.lines[:] = kept_lines
    unread = unread_holdings(raw_text) if request.kind in (
        ResearchKind.IMPACT, ResearchKind.STRESS, ResearchKind.BOOK, ResearchKind.COMPARE,
        ResearchKind.HEDGE) else []
    if unread:
        names = ", ".join(f"{name} ({weight:g}%)" for name, weight in unread)
        one = len(unread) == 1
        note = (f"Assumed: {names} {'is not a contract' if one else 'are not contracts'}"
                f" Bitget lists, so {'it was' if len(unread) == 1 else 'they were'} left out of "
                f"every figure above — check the ticker.")
        at = next((i for i, line in enumerate(answer.lines) if line.startswith("Data:")),
                  len(answer.lines))
        answer.lines.insert(at, note)
        answer.lines[:] = [line for line in answer.lines
                           if not line.startswith("Assumed: your holdings add up to")] \
            if any(line.startswith("Assumed: your holdings add up to") for line in answer.lines) \
            else answer.lines
    # Every research answer, just above its Data line: which sources it reached and which did not
    # answer (`truth/coverage.py`), so a reader can tell a complete answer from one built on part
    # of its inputs. Refusals carry it too — "Bitget did not answer" is the reason for many of them.
    closing = reached.line()
    if closing:
        at = next((i for i, line in enumerate(answer.lines) if line.startswith("Data:")),
                  len(answer.lines))
        answer.lines.insert(at, closing)  # the "Data:" line stays last, as every answer ends
        answer.data["coverage"] = reached.as_dict()
        _honest_data_line(answer, reached)
    return answer


DEFAULT_HEDGED_BOOK = "SPYUSDT"
"""The book a hedge question is measured against when it names no holdings: the S&P 500."""

BETA_TIE = 0.05
"""Two session betas closer than this are said as the same market risk: an hourly beta over 30
days carries a standard error of several hundredths, so a gap inside it names no winner."""


def _loss_lead(lines: list[str], odds: Mapping[str, Any] | None,
               request: ResearchRequest) -> list[str]:
    """A loss question's own answer as the lead: how much a position lost over the period one time
    in ten, at the close and at the worst point inside it, in percent and on $10,000.

    "how much can i lose on tsla this week" led with whether TSLA would be higher, a question it
    did not ask (first-user audit, 2026-09-29). The figures are the odds engine's own tenth
    percentile and worst-point percentile; nothing new is computed here."""
    if not odds or odds.get("adverse_p90_bps") is None or not request.symbols:
        return lines
    short = odds.get("side") == "short"
    close_bps = float(odds["p90_bps"] if short else odds["p10_bps"])
    worst_bps = abs(float(odds["adverse_p90_bps"]))
    hours = request.horizon_hours or 24
    span = {24: "day", 72: "weekend", 168: "week", 720: "month"}.get(hours, f"{hours}-hour")
    # A long loses when the price falls and a short when it rises; a tenth percentile on the
    # winning side means no loss at the end one time in ten.
    close_loss = max(0.0, close_bps if short else -close_bps) / 100.0
    name = request.symbols[0].removesuffix("USDT")
    side = "short" if short else "long"
    lead = (f"Bottom line: one {span} in ten, a {side} {name} position lost more than "
            f"{close_loss:.1f}% by the end, and at its worst point inside the {span} it was more "
            f"than {worst_bps / 100:.1f}% under water — on $10,000, about ${close_loss * 100:,.0f} "
            f"and ${worst_bps:,.0f}. That is how often in the past, not a forecast; the loss "
            f"beyond it is the one-in-ten tail, not a cap.")
    return [lead, *(unlead(line) if i == 0 else line for i, line in enumerate(lines))]


_MOMENTUM_ASKED = re.compile(
    r"\bmomentum\b|\b(?:strong|weak)er\b|\bstronger\s+trend\b|\bperform\w*\b|\boutperform\w*\b|"
    r"\btrending\b|\bup\s+more\b|\b(?:doing|done|did)\s+better\b|\bbeat(?:s|ing|en)?\b|"
    r"\bbetter\s+than\b|\bworse\s+than\b|\blagg?(?:ing|ed)?\b", re.I)
"""A comparison that asks which name is moving the better way, not which is riskier: "which one has
better momentum" was answered with volatility (a judge's audit, 2026-09-29)."""


_PERIOD_ASKED = re.compile(
    r"\b(?:over|in|during)\s+the\s+(?:last|past)\s+(?:month|week|30\s+days|7\s+days|few\s+weeks)\b|"
    r"\b(?:this|last)\s+(?:month|week)\b|\bhow\s+(?:did|has|have)\b|\breturns?\b", re.I)
_RISK_ASKED = re.compile(r"\brisk\w*|\bvolatil\w*|\bbeta\b|\bsafe\w*|\bcorrelat\w*|\bdrawdown",
                         re.I)
_WEEK_ASKED = re.compile(r"\bweek\b|\b7\s+days\b", re.I)


def _asks_for_the_move(raw_text: str) -> bool:
    """Whether a comparison asks how the names moved rather than how risky they are: "BTC vs ETH
    over the last month" led on volatility and never said which rose (a first-user audit,
    2026-09-30). A named period counts unless the question also names a risk measure."""
    return bool(_MOMENTUM_ASKED.search(raw_text)
                or (_PERIOD_ASKED.search(raw_text) and not _RISK_ASKED.search(raw_text)))


def _standalone_lead(lines: list[str], add: str, raw: Mapping[str, Mapping[Any, float]],
                     request: ResearchRequest, raw_text: str) -> list[str]:
    """The lead for a question about one name held alone, when a view or an amount was asked.

    "Give me a quick take on ETH" led with a sizing rule for a $10,000 position, and "how much
    should i put in nvda if i have 3k" never used the $3,000 (the round-7 audits, 2026-09-30). Both
    are read here from every hour of the name's own history: the move over 30 days and a week, how
    much it swings, and the worst 24 hours, which the amount is sized against."""
    from argus.desk.portfolio import worst_window
    from argus.lui.research.parse import RISKS_OF, TAKE_ON
    from argus.lui.research.sizing import HOW_MUCH_IN, capital_lines

    series = [v for _, v in sorted((raw.get(add) or {}).items())]
    if len(series) < 48:
        return lines
    worst = worst_window(weights={add: 1.0}, columns={add: series})
    worst_day = None if worst.move_pct is None else worst.move_pct / 100.0
    rest = [LEAD.sub("", line, count=1) if bool(LEAD.match(line)) else line for line in lines]
    name = _t(add)
    if HOW_MUCH_IN.search(raw_text) and worst_day is not None:
        capital = float(request.notional) if request.notional is not None else 10_000.0
        return [*capital_lines(name, capital, worst_day), *rest]
    should = re.search(r"\bshould\s+i\s+(?:buy|get\s+into|invest\s+in|add)\b|\bis\s+\S+\s+a\s+"
                       r"(?:good\s+)?buy\b|\bworth\s+buying\b", raw_text, re.I)
    if TAKE_ON.search(raw_text) or RISKS_OF.search(raw_text) or should:
        month, week = _compounded(series[-720:]), _compounded(series[-168:])
        swing = statistics.pstdev(series[-720:]) * math.sqrt(24 * 365)
        if month is None or week is None:
            return lines
        # "should i buy bitcoin" opened on a sizing rule and never said the console makes no buy
        # calls (the round-8 first-user audit, 2026-09-30).
        call = ("this console makes no buy or sell call, so here is what holding it has meant — "
                if should else "")
        return [f"Bottom line: {call}{name} is {month:+.1%} over the last 30 days and "
                f"{week:+.1%} over "
                f"the last week, swinging about {swing:.0%} a year"
                + (f"; its worst 24 hours in that time was {worst_day:+.1%}"
                   if worst_day is not None else "")
                + ". That is the move so far, not a forecast; the figures below are what it does "
                  "to a book.", *rest]
    return lines


def _compounded(hourly: Sequence[Any]) -> float | None:
    """The compounded return of a run of hourly returns, or None when there are none."""
    values = [float(r) for r in hourly if r is not None]
    if not values:
        return None
    growth = 1.0
    for r in values:
        growth *= 1.0 + r
    return growth - 1.0


def _momentum_lead(rows: Sequence[Mapping[str, Any]], *, week: bool = False) -> str:
    """Which of the compared names has risen more over 30 days, with the last week beside it, or
    over the last week when the question named a week."""
    if week:
        by_week = sorted((r for r in rows if r.get("ret_7d") is not None),
                         key=lambda r: -float(r["ret_7d"]))
        if len(by_week) >= 2:
            others = ", ".join(f"{_t(r['symbol'])} {float(r['ret_7d']):+.1%}"
                               for r in by_week[1:])
            return (f"Bottom line: {_t(by_week[0]['symbol'])} did better over the last week — "
                    f"{float(by_week[0]['ret_7d']):+.1%} against {others}. A week is short; "
                    f"it describes the move so far, not the next one")
    ranked = sorted((r for r in rows if r.get("ret_30d") is not None),
                    key=lambda r: -float(r["ret_30d"]))
    if len(ranked) < 2:
        return _compare_lead(rows)
    top, rest = ranked[0], ranked[1:]
    week_top = top.get("ret_7d")
    same_week = all(r.get("ret_7d") is not None and week_top is not None
                    and float(week_top) >= float(r["ret_7d"]) for r in rest)
    others = ", ".join(f"{_t(r['symbol'])} {float(r['ret_30d']):+.1%}" for r in rest)
    return (f"Bottom line: {_t(top['symbol'])} has the stronger momentum — "
            f"{float(top['ret_30d']):+.1%} over 30 days against {others}; "
            + ("it also leads over the last week." if same_week else
               "over the last week the order is different, so the lead is fading.")
            + " Momentum describes the move so far, not the next one")


def _compare_lead(rows: Sequence[Mapping[str, Any]]) -> str:
    """Which of the compared names carries the most market risk per dollar, and which swings the
    most on its own. Betas inside :data:`BETA_TIE` of each other are said as the same: 0.79 against
    0.78 named gold the riskier of gold and bitcoin (stranger QA, 2026-09-29)."""
    by_beta = sorted(rows, key=lambda r: -abs(r["beta_open"] or 0))
    riskiest, second = by_beta[0], by_beta[1]
    wildest = max(rows, key=lambda r: r["realised_vol"] or 0.0)
    if abs(riskiest["beta_open"] or 0) - abs(second["beta_open"] or 0) < BETA_TIE:
        lead = (f"Bottom line: {_t(riskiest['symbol'])} and {_t(second['symbol'])} carry about "
                f"the same market risk per dollar (beta {riskiest['beta_open'] or 0:.2f} and "
                f"{second['beta_open'] or 0:.2f})")
    else:
        lead = (f"Bottom line: {_t(riskiest['symbol'])} carries the most market risk per dollar "
                f"of the {len(rows)}")
    if wildest["symbol"] != riskiest["symbol"]:
        lead += (f", but {_t(wildest['symbol'])} swings the most on its own — most of its risk is "
                 f"its own news, which a QQQ hedge will not touch")
    return lead


_INTO_EARNINGS = re.compile(r"\b(?:into|over|through|across|before|ahead\s+of)\s+(?:its\s+|the\s+|"
                            r"their\s+|next\s+)?(?:earnings|results|report)\b|财报前|财报期间",
                            re.I)


def _earnings_night_lines(symbol: str, weight: float | None) -> list[str]:
    """How this name's own results have moved the stock, release by release: every 8-K item 2.02
    on EDGAR (acceptance time) against the stock's split-adjusted daily prices (Yahoo). A release
    accepted before the 09:30 New York open moves that session; one after, the next. The perp keeps
    trading through the night, so a position held into the release carries the same gap."""
    from zoneinfo import ZoneInfo

    from argus.market import equity_history
    from argus.market.evidence import EdgarSource

    ticker = _t(symbol)
    try:
        days = equity_history.daily(ticker)
        filings = EdgarSource().filings(ticker, since=datetime.now(UTC) - timedelta(days=5 * 366),
                                        limit=400)
    except Exception:
        return []
    releases = sorted({f.accepted for f in filings if f.form == "8-K" and "2.02" in f.items})
    index = {d.day: i for i, d in enumerate(days)}
    new_york = ZoneInfo("America/New_York")
    moves: list[tuple[date, float, float]] = []
    for accepted in releases:
        local = accepted.astimezone(new_york)
        day = local.date() if local.hour * 60 + local.minute < 9 * 60 + 30 else (
            local.date() + timedelta(days=1))
        while days and day not in index and day <= days[-1].day:
            day += timedelta(days=1)  # a release before a weekend or holiday prices after it
        at = index.get(day)
        if at is None or at == 0:
            continue
        before = days[at - 1].close
        if before <= 0:
            continue
        moves.append((day, days[at].open / before - 1, days[at].close / before - 1))
    if len(moves) < 4:
        return []
    sessions = sorted(abs(m[2]) for m in moves)
    median = sessions[len(sessions) // 2] if len(sessions) % 2 else (
        sessions[len(sessions) // 2 - 1] + sessions[len(sessions) // 2]) / 2
    up = max(m[2] for m in moves)
    down = min(m[2] for m in moves)
    big = sum(1 for m in moves if abs(m[2]) > 0.05)
    lead = (f"Into earnings: {ticker}'s own results moved the stock a median ±{median:.1%} on the "
            f"session that priced them, over its last {len(moves)} releases (largest "
            f"{up:+.1%} and {down:+.1%}; more than 5% in {big} of {len(moves)})")
    if weight:
        lead += (f" — at {weight:.0%} of the book that is about ±{median * weight:.1%} of the "
                 f"whole book on the night, and as much as {abs(down) * weight:.1%} on the worst "
                 f"one seen")
    last = moves[-1]
    return [lead + ".",
            f"The most recent one, {last[0]:%d %b %Y}: opened {last[1]:+.1%} and closed "
            f"{last[2]:+.1%} against the prior close. Held into the release, the perpetual "
            f"carries this gap too — it trades through the night the stock is shut "
            f"(SEC EDGAR 8-K item 2.02 acceptance times; Yahoo split-adjusted daily prices)."]


def _honest_data_line(answer: Answer, reached: Any) -> None:
    """Rewrite a "Data:" line that credits a server which did not answer this time.

    Each kind names its sources up front ("Data: Bitget's bitget-mcp-server (US equity data), Yahoo
    Finance and SEC EDGAR, live"), which is true when the server answers. On 2026-09-25 it
    answered 503 on every tool for hours, the figures came from SEC filings and Yahoo Finance,
    and the line still credited it — false provenance in a product whose point is that every
    number names its source (readiness audit, finding 32). The rewritten line names what did
    answer and says plainly which named source did not."""
    servers: dict[str, bool] = {}
    for name, ok in reached.answered.items():
        server = name.partition(" ")[0] if name.startswith(("bitget-mcp-server ",
                                                              "bitget-signal ")) else name
        servers[server] = servers.get(server, False) or ok
    at = next((i for i, line in enumerate(answer.lines) if line.startswith("Data:")), None)
    if at is None:
        return
    data_line = answer.lines[at]
    dead = [s for s, ok in servers.items() if not ok and s in data_line]
    if not dead:
        return
    body = " ".join(answer.lines[:at])
    used = [s for s, ok in servers.items() if ok]
    for name, marker in (("Yahoo Finance", "Yahoo"), ("Polymarket", "Polymarket"),
                         ("FRED", "FRED")):
        if marker in body and name not in used:
            used.append(name)
    tail = (" This is analysis, not advice — you make the call."
            if "This is analysis" in data_line else "")
    answer.lines[at] = (
        "Data: " + (", ".join(used) + ", live" if used else "no source answered") + "; "
        + " and ".join(dead) + " did not answer this time, so nothing above comes from "
        + ("it" if len(dead) == 1 else "them") + "." + tail)


_TODAYS = re.compile(r"^(?:\w+ reports in \d+ day|Next report:|Analyst price targets|"
                     r"Analyst consensus|Prediction market|Polymarket)", re.I)


def _point_in_time(answer: Answer, as_of: datetime) -> None:
    """Drop every line an answer "as of" a past date could not have known.

    "What was NVDA's net income as of 1 March 2026" withheld the later filings correctly and then
    appended today's earnings calendar, today's analyst targets and an earnings surprise filed in
    August — a look-ahead leak in the one capability this console claims point-in-time
    correctness for (readiness audit, finding 42). Lines that describe today are removed; a line
    that names a filing date after the as-of date is removed; the answer says what was left out."""
    kept, dropped = [], 0
    for line in answer.lines:
        filed = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC)
                 for d in re.findall(r"\bfiled (\d{4}-\d{2}-\d{2})\b", line)]
        bare = LEAD.sub("", line, count=1)
        if _TODAYS.search(bare) or any(f > as_of for f in filed):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        at = next((i for i, line in enumerate(kept) if line.startswith("Data:")), len(kept))
        kept.insert(at, f"Point in time: {dropped} line(s) about today — the next report date, "
                        f"current analyst targets and consensus, and anything filed after "
                        f"{as_of:%d %b %Y} — were left out, because the question asks what was "
                        f"known on that date.")
    answer.lines[:] = kept


_PREDICTION_KINDS = frozenset({ResearchKind.FUNDAMENTALS, ResearchKind.NEWS,
                               ResearchKind.SENTIMENT})
"""The answers about what might happen to a name — its company events, its news, its crowd, and
whether it will be higher — are the ones a prediction market prices beside."""


def _add_prediction_markets(answer: Answer, symbol: str) -> None:
    """Polymarket's busiest informative markets on the name (`market/prediction.py`), placed
    before the closing "Data:" line. Nothing is added when none is open and liquid."""
    from argus.market import prediction

    lines = prediction.lines_for(symbol, name=_t(symbol))
    if not lines:
        return
    at = next((i for i, line in enumerate(answer.lines) if line.startswith("Data:")),
              len(answer.lines))
    answer.lines[at:at] = lines
    answer.sources.append(Source(kind="venue", ref="Polymarket (gamma-api public-search)",
                                 detail=f"open markets naming {_t(symbol)}, over "
                                        f"${prediction.VOLUME_FLOOR:,.0f} traded"))


def _decay_answer(raw_text: str, question: Question) -> Answer | None:
    from argus.market import equity_history
    from argus.research import leveraged_decay

    fund = leveraged_fund_asked(raw_text)
    if fund is None:
        return None
    index, _ = leveraged_decay.FUNDS[fund]
    found = re.search(r"\b(\d+)\s*(trading\s+days?|days?|weeks?|months?|years?)\b", raw_text, re.I)
    if found:
        unit = found.group(2).lower()
        days = int(found.group(1)) * (5 if unit.startswith("w") else 21 if unit.startswith("m")
                                      else 252 if unit.startswith("y") else 1)
    else:
        days = (5 if re.search(r"\bweek\b", raw_text, re.I) else
                252 if re.search(r"\byear\b|long[\s-]term", raw_text, re.I) else 21)
    days = max(2, min(days, 504))
    try:
        fund_days = equity_history.daily(fund)
        index_days = equity_history.daily(index)
    except Exception:
        return Answer(question=question, refused=True, reason="daily history did not arrive",
                      lines=[f"{fund} and {index}'s daily history did not arrive just now, so the "
                             f"decay cannot be measured. Try again shortly."])
    result = leveraged_decay.measure(fund, {d.day: d.close for d in fund_days},
                                     {d.day: d.close for d in index_days}, days)
    lines = leveraged_decay.lines(result)
    lines.append(f"Data: {fund} and {index} split-adjusted daily closes (Yahoo), "
                 f"{min(len(fund_days), len(index_days))} days; {index}'s last {60} days for the "
                 f"volatility. This is analysis, not advice — you make the call.")
    return Answer(question=question, lines=lines,
                  sources=[Source(kind="computation", ref="argus.research.leveraged_decay",
                                  detail=f"{fund} vs {index}, {days}-day sideways windows")],
                  data={"decay": {"fund": fund, "index": index, "days": days,
                                  "windows": result.windows, "median": result.median_fund,
                                  "formula": result.formula}})


def _agent_hub_lines(symbol: str, request: ResearchRequest, plan: Any,
                     ticker: Any, raw_text: str) -> list[str]:
    """The execution plan's first child as the Agent Hub dry-run that previews it
    (`lui/agenthub.py`), or the reason there is none; never raises."""
    from argus.lui import agenthub
    from argus.market import universe

    try:
        if ticker is None or not plan.slices or request.notional is None:
            return []
        contract = universe.contracts().get(symbol)
        child = agenthub.child_order(
            # the side the depth lines walk the book on (`_depth_lines`), so the two agree
            symbol, "sell" if _SELL_WORDS.search(raw_text) else "buy",
            Decimal(str(request.notional)) * plan.slices[0].fraction, ticker.last,
            contract.size_step if contract else None, contract.min_qty if contract else None)
        return agenthub.lines_for(child)
    except Exception:
        return []


def _positioning_lead(found: list[str], raw_text: str, symbol: str) -> list[str]:
    """Lead a sentiment answer with the listed-market line the question asked for — the options
    chain, dark pools or short volume (`lui/research/positioning.py`) — or say it is missing.

    "show me dark pool activity for TSLA" reached the desk's track record, and once routed here
    the dark-pool line sat five lines under a funding verdict (stranger QA, 2026-09-29)."""
    from argus.market.bitget import ANCHOR_OF

    ticker = ANCHOR_OF.get(symbol, _t(symbol))
    for pattern, prefix, what in (
            (DARK_POOL_Q, "Dark pools (FINRA", "off-exchange (FINRA ATS) volume"),
            (SHORT_FLOW_Q, "Short volume (FINRA", "FINRA's daily short volume"),
            (OPTIONS_POSITIONING_Q, f"Options on {ticker} (Cboe", "Cboe's listed options chain")):
        if not pattern.search(raw_text):
            continue
        led = _lead_with(found, prefix)
        if led is not found:
            if pattern is SHORT_FLOW_Q and re.search(r"\bshort\s+interest\b", raw_text, re.I):
                # the figure read is the daily flow; the stock of open shorts is not read
                led.insert(1, "Short interest itself — FINRA's twice-monthly count of shares "
                              "held short — is not read here; the daily short-sale volume above "
                              "is the flow, not the stock of shorts.")
            return led
        reason = ("it is read only for the US-listed underlying of a Bitget stock perpetual"
                  if symbol not in ANCHOR_OF else "the source did not answer just now")
        what = what[0].upper() + what[1:]
        return [f"Bottom line: {what} was not read for {_t(symbol)} — {reason}; the "
                f"positioning that was read follows.", *(unlead(line) for line in found)]
    return found


def _run(raw_text: str, request: ResearchRequest, *, ledger: Any = None) -> Answer:
    """Answer a research request from the desk's engines. Refuses by name rather than guessing."""
    question = _question(raw_text, request)
    decay = (_decay_answer(raw_text, question) if request.kind is not ResearchKind.COMPARE
             else None)
    if decay is not None:
        return decay
    if (request.kind in (ResearchKind.STRESS, ResearchKind.IMPACT, ResearchKind.BOOK)
            and request.cash >= 0.999 and not request.book):
        # "how much VaR do I have at 95% if I hold nothing but stablecoins" was answered with the
        # desk's own track record (2026-09-25 audit).
        return Answer(question=question, sources=[], lines=[
            "Bottom line: a book held entirely in cash or stablecoins has no market move to "
            "stress: its value at risk and every scenario here are zero against the dollar. A "
            "stablecoin's own risk is a de-peg, which this desk does not model.",
            *(f"Assumed: {note}." for note in request.notes)])
    if request.kind in (ResearchKind.MACRO, ResearchKind.SENTIMENT):
        found, extra, backdrop = (_macro(request.symbols[0] if request.symbols else None,
                                         request.book or None, raw_text)
                                  if request.kind is ResearchKind.MACRO
                                  else _sentiment(request.symbols[0] if request.symbols
                                                  else None))
        premise = fed_premise_line(raw_text) if request.kind is ResearchKind.MACRO else None
        if premise is not None and found:
            # A Fed move stated as fact is checked first, in the lead (a judge's audit).
            found = [f"Bottom line: {premise[0]}",
                     *(unlead(line) if i == 0 else line for i, line in enumerate(found))]
            extra = [*extra, premise[1]]
        if not found:
            return Answer(question=question, refused=True,
                          reason="the backdrop sources did not answer",
                          lines=["Neither FRED nor the sentiment source answered just now, so "
                                 "there is nothing honest to report. Try again shortly."])
        crowd = (_crowd_lines(request.symbols[0])
                 if request.kind is ResearchKind.SENTIMENT and request.symbols else [])
        flows = _flow_lines(request.symbols[0] if request.symbols else "BTCUSDT"
                            if request.kind is ResearchKind.SENTIMENT else "")
        if flows:
            found.extend(flows)
            if CRYPTO_ETF_QUESTION.search(raw_text):
                # the funds' own flows answer an ETF question; the positioning follows
                found = _lead_with(found, "US spot ")
            extra.append(Source(kind="venue", ref="SoSoValue US spot ETF flows",
                                detail="creations less redemptions, daily after the US close"))
        if crowd:
            found.extend(crowd)
            extra.append(Source(kind="computation", ref="argus.market.social_pulse",
                                detail="X and Reddit posts grouped into stories by the desk's "
                                       "coordination detector; a dated snapshot"))
        if request.kind is ResearchKind.SENTIMENT and OPEN_INTEREST_QUESTION.search(raw_text):
            # "open interest on ETH futures" opened on the sentiment summary with the open
            # interest four lines down (live, 2026-09-25): the figure asked for leads.
            found = _lead_with(found, "Open interest:")
        if (request.kind is ResearchKind.SENTIMENT and re.search(r"\bfunding\b", raw_text, re.I)
                and not re.search(r"\bopen[\s-]+interest\b|\bOI\b", raw_text)):
            # "What are funding rates telling us about crowd positioning in ETH" led with open
            # interest (a judge's audit, 2026-09-30): funding was asked, and its reading leads.
            found = _lead_with(found, "Funding means ")
        if request.kind is ResearchKind.SENTIMENT and request.symbols:
            found = _positioning_lead(found, raw_text, request.symbols[0])
        if (request.kind is ResearchKind.MACRO and request.symbols
                and any("standing for it" in note for note in request.notes)):
            # "how does that affect crypto" after a Fed question: the name's own sensitivity to
            # rates is the answer, not the rates dashboard (answer audit, round 3)
            found = _lead_with(found, f"{_t(request.symbols[0])} has ")
        if request.kind is ResearchKind.SENTIMENT and LONG_SHORT_QUESTION.search(raw_text):
            led = _lead_with(found, "Long/short on Bitget")
            found = led if led is not found else [
                *found, "Missing: Bitget's long/short series did not answer just now, so no "
                        "ratio is given."]
        found.extend(f"Assumed: {note}." for note in request.notes)
        found.append(f"Data: {extra[0].detail if extra else 'public sources'}. This is analysis, "
                     f"not advice — you make the call.")
        return Answer(question=question, lines=found, sources=extra,
                      data={"request": request.as_dict(), request.kind.value: backdrop})
    if request.kind is ResearchKind.EVENT and request.symbols:
        found, extra = _event_reaction(request.symbols[0], raw_text)
        if not found:
            return Answer(question=question, refused=True,
                          reason="the event study has not been computed",
                          lines=["The event-reaction study is computed once a day and is not "
                                 "available right now, so there is nothing measured to report."])
        found.append("Data: CPI dates from the BLS release archive, Fed decisions from the "
                     "Federal Reserve calendar, earnings from SEC 8-K item 2.02, prices from "
                     "Bitget hourly candles. This is analysis, not advice — you make the call.")
        return Answer(question=question, lines=found, sources=extra,
                      data={"request": request.as_dict()})
    if request.kind is ResearchKind.HEDGE and request.spot:
        return _spot_hedge_answer(question, request)
    request = without_hedges(request, raw_text)
    if request.kind is ResearchKind.HEDGE and not request.book and request.symbols:
        # "what's the best hedge for my NVDA overnight exposure?" reached here with a name and no
        # weights and returned only the footer lines (2026-09-25 audit). The named holding is the
        # position to hedge; the answer says so.
        named = request.symbols[: max(1, len(request.symbols))]
        request = replace(request, book={s: 1.0 / len(named) for s in named}, notes=(
            *request.notes, f"no weights were given, so {', '.join(_t(s) for s in named)} "
                            f"{'is' if len(named) == 1 else 'are'} hedged as the whole position"))
    if request.kind is ResearchKind.HEDGE and not request.book and not request.symbols:
        # "Should I hedge with gold or with TLT?" names only the hedges, and the README lists it
        # as a question to try verbatim; it was refused for want of holdings (judge audit,
        # 2026-09-29). The honest default is the broad US stock market, stated on the answer.
        request = replace(request, book={DEFAULT_HEDGED_BOOK: 1.0},
                          symbols=(DEFAULT_HEDGED_BOOK,), notes=(
            *request.notes, f"no holdings were stated, so the hedges are measured against a "
                            f"position in the S&P 500 ({_t(DEFAULT_HEDGED_BOOK)}) — say what you "
                            f"hold, or fill in My book, for your own figures"))
    if (request.kind is ResearchKind.HEDGE and len(request.symbols) == 1
            and _WITH_ITS_PERP.search(raw_text)):
        same = _same_name_perp_hedge(request.symbols[0], raw_text)
        if same is not None:
            return Answer(question=question, lines=same[0], sources=same[1],
                          data={"request": request.as_dict()})
    if request.kind is ResearchKind.HEDGE and request.book:
        value = request.notional or HEDGE_BOOK_VALUE
        found, extra, hedge_found = _hedge_plan(request.book, value, raw_text)
        if not found:
            return Answer(question=question, refused=True,
                          reason="the hedge candidates could not be measured",
                          lines=["The candidate hedges' prices or order books did not arrive, so "
                                 "no hedge can be measured just now. Try again shortly."])
        found.extend(f"Assumed: {note}." for note in request.notes)
        found.append("Data: live Bitget hourly candles (30 days), order books and funding rates. "
                     "This is analysis, not advice — you make the call.")
        return Answer(question=question, lines=found, sources=extra,
                      data={"request": request.as_dict(), "hedge": hedge_found})
    if (request.kind is ResearchKind.STRESS and not request.symbols and request.shock_on
            and not request.cash):
        # "whats my drawdown if btc drops 20%" names the holding the shock is on and no book; it
        # asked for holdings twice in a row (answer audit, round 3). Read as a position in that
        # name, and said.
        request = replace(request, book={request.shock_on: 1.0}, symbols=(request.shock_on,),
                          notes=(*request.notes, f"no other holdings were stated, so the book is "
                                                 f"read as a position in {_t(request.shock_on)} "
                                                 f"itself — say what you hold for your whole "
                                                 f"book's figure"))
    if not request.symbols:
        return Answer(
            question=question, refused=True,
            reason="the scenario needs your holdings, and none were named",
            lines=[
                "I can run that the moment I know what you hold — say it with weights, e.g. "
                "\"what if the Nasdaq drops 10%? I hold 50% NVDA, 30% MSFT, 20% AAPL\".",
                "Any contract Bitget lists can be named — the twelve stock perpetuals the desk "
                "trades, other US stocks and ETFs, gold, oil, index products and crypto.",
            ],
        )
    try:
        data = (_NO_CANDLES[request.kind] if request.kind in _NO_CANDLES
                else _analogue_data(request.symbols[0])
                if request.kind is ResearchKind.ANALOGUE
                else load((*request.symbols, request.shock_on) if request.shock_on
                          else request.symbols))
    except Exception as exc:
        return Answer(
            question=question, refused=True,
            reason=f"market data for {', '.join(request.symbols)} could not be loaded "
                   f"({type(exc).__name__})",
            lines=[f"I could not load market data for {', '.join(request.symbols)} just now, live "
                   f"or frozen, so I will not guess at the risk. Try again in a minute."],
        )
    is_open = anchor_is_open()
    lines: list[str] = []
    agent_hub_lines: list[str] = []
    sources: list[Source] = [data.source]
    payload: dict[str, Any] = {"request": request.as_dict(), "live_data": data.live}

    try:
        if request.kind is ResearchKind.IMPACT:
            add = request.symbols[0]
            before = {s: w for s, w in request.book.items()}
            if add in before and len(before) == 1 and not request.cash:
                before = {}
            resized = _resolve_resize(request, before)
            if isinstance(resized, str):
                return Answer(question=question, refused=True,
                              reason="a resize needs the rest of the book", lines=[resized])
            if resized is not None:
                before, target, resize_line = resized
                request = replace(request, book=before, size=target, size_stated=True,
                                  target=target)
            # Sector and factor exposure before and after, fetched beside the copilot so it
            # adds no wait (Yahoo daily closes; about four seconds cold, cached after).
            exposure_future = _exposure_future(add, before, request)
            report = copilot(add=add, before=before, size=request.size or DEFAULT_SIZE,
                             raw=data.raw, benchmark=BENCHMARK, is_open=is_open,
                             target=request.target)
            columns = _open_columns(data.raw, is_open)
            lines = _impact_lines(report, request, columns)
            if not before:
                lines = _standalone_lead(lines, add, data.raw, request, raw_text)
            exposure_lines, exposure_sources = _exposure_lines(exposure_future)
            lines.extend(exposure_lines)
            sources.extend(exposure_sources)
            if resized is not None:
                lines.insert(0, resize_line)
            if _VAR.search(raw_text):
                var_lines, var_sources = _var_lines(before, report.weights_after, raw_text)
                lines[1:1] = var_lines
                sources.extend(var_sources)
            payload["report"] = report.as_dict()
            payload["sizing"] = impact_sizing(report, request, columns)
            sources.append(Source(kind="computation", ref="argus.desk.portfolio.copilot",
                                  detail="session beta, Euler risk decomposition, beta stress, "
                                         "realised worst 24h window"))
            anchor = _crypto_anchor_line(add, data.raw, columns)
            if anchor is not None:
                lines.insert(min(2, len(lines)), anchor)
                sources.append(Source(kind="computation", ref="argus.desk.portfolio.beta",
                                      detail=f"{add} against {CRYPTO_ANCHOR}, every aligned hour"))
            if not before:
                profile = _distribution_line(add, data.raw, columns)
                if profile is not None:
                    lines.append(profile)
            else:
                record = _beta_track_record()
                if record is not None:
                    lines.append(record)
                    sources.append(Source(kind="computation", ref="argus.eval.copilot_rivals",
                                          detail="post-trade beta scored against the next 28 "
                                                 "days, 1,800 books, vs weekend-copilot (S2)"))
            desk_lines, desk_sources = _desk_view(add, ledger)
            lines.extend(desk_lines)
            sources.extend(desk_sources)
            mandate = _mandate_lines(add, request.size, data.raw, raw_text,
                                     request.mandate_text, request.mandate_capital)
            if mandate:
                # The trader's own mandate answers "should I" first; the risk view follows it.
                # The engine's lead is not always first in its own list (the final sort puts it
                # there), so it is found by its prefix rather than by position.
                lead = next((i for i, line in enumerate(lines)
                             if bool(LEAD.match(line))), None)
                risk_view = unlead(lines.pop(lead)) if lead is not None else ""
                lines = [mandate[0], *([risk_view] if risk_view else []), *mandate[1:], *lines]
                sources.append(Source(kind="computation", ref="argus.desk.personalisation.judge",
                                      detail="your stated mandate against the trade; the "
                                             "opposite preset on the same trade"))

        elif request.kind is ResearchKind.VENUE:
            lines, extra = _venue(request.symbols[0], is_open, request.spot)
            if request.spot and re.search(r"\btrack|1\s*:\s*1|one[\s-]to[\s-]one|peg", raw_text,
                                          re.I):
                lines = _lead_with(lines, "How closely it tracks:")
            elif request.spot and re.search(r"dividend|backed|redeem|redemption|voting|rights?",
                                            raw_text, re.I):
                lines = _lead_with(lines, "Whether an rToken pays dividends")
            sources.extend(extra)

        elif request.kind is ResearchKind.CONSTRUCT:
            # Crypto alone is weighted on every hour, as `book._book_report` reads it (2026-09-30).
            columns = _open_columns(
                data.raw, (lambda _t: True) if all(
                    not is_us_equity(s) and s not in TRADED_SYMBOLS for s in request.symbols)
                else is_open)
            names = tuple(s for s in request.symbols if s in columns)
            weights = _equal_risk_weights(names, columns) if len(names) >= 2 else None
            if weights is None:
                return Answer(question=question, refused=True,
                              reason="the names share too little history to weight by risk",
                              lines=["Not enough shared hourly history across those names to "
                                     "balance their risk. Try names that trade on Bitget every "
                                     "hour, or fewer of them."])
            cap, count, risk_cap = stated_limits(raw_text)
            equal: dict[str, float] = weights
            headline = (
                "Bottom line: an equal-risk book of these names holds "
                + ", ".join(f"{_t(s)} {equal[s]:.0%}"
                            for s in sorted(equal, key=lambda s: -equal[s]))
                + f" — each carries about {1 / len(equal):.0%} of the risk, the quieter names "
                  f"held larger so no single one dominates.")
            if request.notional is not None:
                # The sum asked about, divided: "$30,000 between BTC and ETH" answered in
                # percentages alone left the arithmetic to the reader (2026-09-30).
                total = float(request.notional)
                headline = headline.removesuffix(".") + (
                    f"; of ${total:,.0f} that is "
                    + ", ".join(f"{_t(s)} ${total * equal[s]:,.0f}"
                                for s in sorted(equal, key=lambda s: -equal[s]))
                    + f", against ${total / len(equal):,.0f} each split evenly.")
            limited: dict[str, Any] | None = None
            if cap is not None or count is not None or risk_cap is not None:
                fitted = _within_limits(equal, columns, cap, count, risk_cap)
                if fitted["status"] == "infeasible":
                    return Answer(question=question, refused=True, reason=fitted["reason"],
                                  lines=[f"Those limits cannot all hold for "
                                         f"{', '.join(_t(s) for s in names)}: "
                                         f"{fitted['plain']}. Loosen one and ask again."])
                weights, headline, limited = fitted["weights"], fitted["headline"], fitted
                sources.append(Source(kind="computation", ref="argus.desk.constrained.allocate",
                                      detail="the equal-risk book, bent as little as possible "
                                             "to meet the stated limits"))
            built = replace(request, kind=ResearchKind.BOOK, book=weights, budget_stated=True,
                            budget=1.0)
            book_lines, extra, book_payload = _book_report(built, data, is_open)
            lines = [headline, *([limited["limits_line"]] if limited else []),
                     *[line for line in book_lines if not bool(LEAD.match(line))]]
            sources.extend(extra)
            payload["construct"] = {"weights": weights, **book_payload}
            if limited:
                payload["construct"]["limits"] = {k: v for k, v in limited.items()
                                                  if k not in ("headline", "limits_line")}

        elif request.kind is ResearchKind.LEVERAGE:
            lines, extra, lever_payload = _leverage(
                request.symbols[0], request.leverage or 10.0, request.side,
                closure=("weekend" if request.weekend else "overnight"
                         if request.horizon_hours else None),
                notional=float(request.notional) if request.notional else None)
            sources.extend(extra)
            if re.search(r"\bliq\w*\s+(?:price|level)|\bliquidat\w*\s+(?:price|level|at)\b|"
                         r"\bprice\b[^?.]{0,20}\bliquidat|\bget\s+liquidated\s+at\b",
                         raw_text, re.I):
                lines = _lead_with(lines, "Liquidation price:")
            payload["leverage"] = lever_payload
            closure_read = lever_payload.get("closure") or {}
            if closure_read:
                data = replace(data, provenance=", ".join(part for part, used in (
                    ("live Bitget hourly candles (highs and lows) for 30 days", True),
                    ("Bitget daily candles for the perpetual's weekends",
                     "perp_weekends" in closure_read),
                    ("the stock's daily history from Yahoo Finance",
                     "stock_weekends" in closure_read),
                    ("Bitget's maintenance-margin tiers",
                     lever_payload.get("maintenance_margin_rate") is not None),
                    ("the live funding rate", True)) if used))
            if not lines:
                return Answer(question=question, refused=True,
                              reason="Bitget's candles for this name could not be read",
                              lines=["Bitget's hourly candles did not come back just now, so the "
                                     "liquidation odds cannot be measured. Try again shortly."])

        elif request.kind is ResearchKind.NEWS:
            lines, extra, news_payload = _news(request.symbols[0], is_open,
                                               tone=tone_asked(raw_text))
            sources.extend(extra)
            payload["news"] = news_payload
            moved = next((i for i, line in enumerate(lines)
                          if re.search(r" is [+-][\d.]+% over 24 hours on Bitget", str(line))),
                         None)
            if (moved is not None and lines
                    and re.search(r"\b(?:what|how)\s+(?:is|are|'s)\b[^?]{1,40}\bdoing\b",
                                  raw_text, re.I)):
                # "what is the S&P 500 doing" led with "nothing in 48h names SP500" and gave the
                # move itself second (2026-09-30): the move is what was asked, the news is why.
                head = str(lines[0]).removeprefix("Bottom line: ")
                lines = [f"Bottom line: {lines[moved]} On the news: {head[:1].lower() + head[1:]}",
                         *(line for i, line in enumerate(lines) if i not in (0, moved))]

        elif request.kind is ResearchKind.BOOK:
            lines, extra, book_payload = _book_report(request, data, is_open)
            sources.extend(extra)
            payload["book"] = book_payload
            worth = parse_notional(raw_text)
            if worth and request.book:
                dollar_lines = _book_dollar_lines(request, data, is_open, float(worth))
                if dollar_lines:
                    lines = [dollar_lines[0], *(re.sub(r"^(?:Actionable|Bottom line):\s*(\w)",
                                                       lambda m: m.group(1).upper(), x)
                                                for x in lines), *dollar_lines[1:]]
            if _BOOK_HISTORY_Q.search(raw_text) and request.book:
                history = _book_history_lines(request.book, request.cash, raw_text)
                if history is not None:
                    lines = [*history[0], *(re.sub(r"^(?:Actionable|Bottom line):\s*(\w)",
                                                   lambda m: m.group(1).upper(), x)
                                            for x in lines)]
                    sources.append(history[1])
            if re.search(r"\bcorrelat\w*", raw_text, re.I) and len(request.book) >= 2:
                # "correlation matrix for my book" was declined; every pair, open-session hours
                columns = _open_columns(data.raw, is_open)
                held_names = [s for s in request.book if s in columns]
                pairs_read = [(a, b, correlation(columns[a], columns[b]))
                              for i, a in enumerate(held_names) for b in held_names[i + 1:]]
                pairs_read = [(a, b, r) for a, b, r in pairs_read if r is not None]
                if pairs_read:
                    pairs_read.sort(key=lambda t: -(t[2] or 0))
                    lines = [
                        "Bottom line: correlation between your holdings (hourly returns, "
                        "open-session hours, last 30 days): "
                        + "; ".join(f"{_t(a)}-{_t(b)} {r:+.2f}" for a, b, r in pairs_read)
                        + f" — the tightest pair, {_t(pairs_read[0][0])} and "
                          f"{_t(pairs_read[0][1])}, is the least diversification in the book.",
                        *(unlead(x)
                          for x in lines)]
            if _WORTH_Q.search(raw_text) and not worth:
                lines.insert(0, "Bottom line: the book you saved is weights, not dollars, so what "
                                "it is worth is not known here — say its value (\"my book is "
                                "$50k\") and each holding's dollars, dollar risk and a "
                                "Nasdaq-drop loss in dollars follow.")
                lines[1:] = [unlead(x)
                             for x in lines[1:]]
            if _BETA_ASKED.search(raw_text) and request.book:
                beta_line = _book_beta_line(request.book, data, is_open, raw_text)
                if beta_line is not None:
                    lines = [beta_line[0], *(re.sub(r"^(?:Actionable|Bottom line):\s*(\w)",
                                                    lambda m: m.group(1).upper(), x)
                                             for x in lines)]
                    sources.append(beta_line[1])

        elif request.kind is ResearchKind.STRESS:
            columns = _open_columns(data.raw, is_open)
            vol_multiple = _vol_multiple(raw_text)
            if request.book and (vol_multiple is not None or _VAR.search(raw_text)):
                # "What is the VaR of 50% BTC, 50% ETH" is answered with the book's VaR and
                # expected shortfall first; it used to get the Nasdaq stress lines alone
                # (a sample of answers, 2026-09-25).
                vol_lines, vol_sources = _var_lines({}, request.book, raw_text,
                                                    scale=vol_multiple or 1.0)
                vol_first = [f"Bottom line: {line[0].lower()}{line[1:]}" if i == 0 else line
                             for i, line in enumerate(vol_lines)]
                lines.extend(vol_first)
                sources.extend(vol_sources)
            shocks = [Shock("benchmark -5%", -5.0), Shock("benchmark -10%", -10.0)]
            single = _SINGLE_NAME.search(raw_text) if request.book else None
            if single is not None:
                # "if a single name in my book craters 50%, which one hurts the most?" was
                # answered as the Nasdaq falling 50% (2026-09-25 audit). One name falling on its
                # own costs the book its weight times the fall; the largest weight is the answer.
                fall = abs(request.shock_pct) if request.shock_pct else 50.0
                ranked = sorted(request.book.items(), key=lambda kv: -kv[1])
                lines.append(
                    f"Bottom line: if one name falls {fall:g}% on its own, the one that hurts most "
                    f"is the biggest holding, {_t(ranked[0][0])} — it would cost the book "
                    f"{ranked[0][1] * fall:.1f}%. Each name alone: "
                    + "; ".join(f"{_t(sym)} {-w * fall:+.1f}%" for sym, w in ranked)
                    + (f" (with {request.cash:.0%} in cash already diluting it)"
                       if request.cash else "")
                    + ". A fall that is the market's, not the company's, is the Nasdaq lines "
                      "below.")
            elif request.shock_pct is not None:
                # The shock asked about comes first and leads, even when it is one of the two
                # standard ones: "a 10% drop in gold" opened on the -5% line (2026-09-25 audit).
                shocks = [Shock(f"benchmark {request.shock_pct:+g}%", request.shock_pct),
                          *(sh for sh in shocks if sh.benchmark_move_pct != request.shock_pct)]
            stated_leads = (request.shock_pct is not None and single is None
                            and vol_multiple is None and not _VAR.search(raw_text))
            shocked = request.shock_on or BENCHMARK
            shocked_name = "QQQ" if shocked == BENCHMARK else _t(shocked)
            outcomes = stress_by_beta(weights=request.book, columns=columns,
                                      benchmark=columns[shocked], shocks=shocks)
            for outcome in outcomes:
                if outcome.portfolio_move_pct is None:
                    lines.append(f"{outcome.shock}: unavailable — {outcome.reason}")
                    continue
                worst = outcome.worst_position
                lead_here = stated_leads and outcome is outcomes[0]
                lines.append(
                    ("Bottom line: " if lead_here else "")
                    + f"If {shocked_name} moves {outcome.shock.removeprefix('benchmark ')}: your "
                    f"book moves "
                    f"about {outcome.portfolio_move_pct:+.2f}%"
                    + (f", {'hardest hit' if worst[1] < 0 else 'biggest move'} "
                       f"{worst[0].removesuffix('USDT')} {worst[1]:+.2f}%"
                       if worst else "") + " (market-driven part only, through each beta)."
                )
            book_beta = 0.0
            driven: dict[str, float] = {}
            for symbol, weight in request.book.items():
                symbol_beta = beta(columns.get(symbol, []), columns[shocked])
                if symbol_beta is not None:
                    book_beta += weight * symbol_beta
                    driven[symbol] = weight * symbol_beta
            if len(driven) > 1 and book_beta > 0:
                # The holding to name is the one carrying the most loss RELATIVE to its weight —
                # the biggest position is usually the biggest contributor and naming it tells the
                # trader nothing. A live answer named QQQ (60% of the book, 52% of the loss) while
                # COIN carried 48% of the loss on 40% of the weight.
                top = max(driven, key=lambda s: driven[s] / book_beta - request.book[s])
                share = driven[top] / book_beta
                prefix = ("" if vol_multiple is not None or single is not None or stated_leads
                          else "Bottom line: ")
                if share - request.book[top] >= 0.02:
                    lines.append(
                        f"{prefix}{_t(top)} is {request.book[top]:.0%} of the book but "
                        f"{share:.0%} of its loss when {shocked_name} falls — trimming it cuts "
                        f"the drawdown fastest; the {shocked_name} hedge below is the other lever."
                    )
                else:
                    lines.append(
                        f"{prefix}{'the' if prefix else 'The'} loss is spread roughly in line with "
                        "your weights, so "
                        f"trimming any one name barely helps — the {shocked_name} hedge below is "
                        "the lever."
                    )
            if shocked != BENCHMARK and abs(book_beta) < 0.2:
                # "oil -20%, how does that hit my book" on a tech book printed three move lines and
                # no verdict (a judge-style pass, 2026-09-25). The verdict is that it barely does.
                lines.insert(0, f"Bottom line: this book barely moves with {shocked_name} — its "
                                f"beta to {shocked_name} is {book_beta:+.2f}, so the shock reaches "
                                f"it only faintly and no hedge against it is needed.")
            if shocked == BENCHMARK:
                hedge = _hedge_line(book_beta)
            elif abs(book_beta) >= 0.2:
                side = "short" if book_beta > 0 else "long"
                hedge = (f"Hedge: {side} {shocked_name} worth about {abs(book_beta):.0%} of the "
                         f"book's value offsets the part of this shock that reaches the book "
                         f"through beta (book beta to {shocked_name} {book_beta:.2f}); it does "
                         f"nothing for moves the holdings make on their own.")
            else:
                hedge = (f"Hedge: none needed against {shocked_name} — the book's beta to it is "
                         f"{book_beta:.2f}, so its moves barely reach these holdings.")
            if hedge:
                lines.append(hedge)
            if not any(bool(LEAD.match(line)) for line in lines):
                # A book with one risky name has no loss share to rank, and the answer used to
                # open on a bare scenario line (2026-09-25 audit). The lead is the largest shock.
                worst_case = min((o for o in outcomes if o.portfolio_move_pct is not None),
                                 key=lambda o: o.portfolio_move_pct or 0.0, default=None)
                if worst_case is not None:
                    lines.insert(0, (
                        f"Bottom line: if {shocked_name} moves "
                        f"{worst_case.shock.removeprefix('benchmark ')}, this book moves about "
                        f"{worst_case.portfolio_move_pct:+.2f}% through beta"
                        + (f", its {request.cash:.0%} in cash already diluting it"
                           if request.cash else "")
                        + (f"; shorting {shocked_name} worth about {abs(book_beta):.0%} of the "
                           f"book offsets that part" if abs(book_beta) >= 0.2 else "")
                        + "."))
            # The book as stated, replayed over its own history. This used copilot(add=<first>,
            # before=<rest>, size=<its weight>), and rebalancing scales the rest by 1 - size: a
            # 50/50 QQQ/TSLA book was replayed as 50/25 and shown losing -2.01% where it had lost
            # -4.01% — 28 of 28 multi-name figures wrong in `eval/research_depth.py`'s
            # recomputation (2026-09-25).
            from argus.desk import stress_tree

            realised = stress_tree.book_worst_window(data.raw, request.book)
            payload["stress"] = [o.as_dict() for o in outcomes]
            payload["worst_window"] = realised.as_dict()
            # The scenario tree (`desk/stress_tree.py`): the stated shock grown into what it
            # implies — the hedged tail, each name alone, whether the shock ever happened in this
            # history, the tail share and the trim that fixes it, the shut-session case. Measured
            # on 61 live stress questions: 10 scenario angles against 3, 0 errors in 1,300
            # recomputed figures, +187ms. It carries the realised worst window itself, so the
            # one-pass line is used only when the tree could not be grown.
            tree_lines, tree_sources, tree_payload = stress_tree.research_lines(
                raw=data.raw, is_open=is_open, book=request.book, shocked=shocked,
                shock_pct=request.shock_pct, cash=request.cash, shocked_label=shocked_name,
                provenance=data.provenance)
            if tree_lines:
                lines.extend(tree_lines)
                sources.extend(tree_sources)
                payload["stress_tree"] = tree_payload
            else:
                lines.append(realised.render().replace(
                    "[stress] ", "What actually happened, not a model — "))
            sources.append(Source(kind="computation", ref="argus.desk.portfolio.stress_by_beta",
                                  detail="beta-propagated shock + realised worst window"))

        elif request.kind is ResearchKind.COMPARE:
            columns = _open_columns(data.raw, is_open)
            rows: list[dict[str, Any]] = []
            for symbol in request.symbols:
                rep = copilot(add=symbol, before={}, size=1.0, raw=data.raw,
                              benchmark=BENCHMARK, is_open=is_open)
                betas = {str(k): v.beta for k, v in rep.session_beta.items()}
                ten = next((o.portfolio_move_pct for o in rep.stress
                            if o.shock == "benchmark -10%"), None)
                hourly = list(data.raw.get(symbol, {}).values())
                var = variance(hourly)
                # Annualised on the hours an rToken actually trades: 24 x 365, not 252 sessions.
                vol = None if var is None else math.sqrt(var) * math.sqrt(24 * 365)
                rows.append({"symbol": symbol, "beta_open": betas.get("open"),
                             "beta_shut": betas.get("shut"), "qqq_minus_10": ten,
                             "worst_24h": rep.worst.move_pct, "realised_vol": vol,
                             "ret_30d": _compounded(hourly), "ret_7d": _compounded(hourly[-168:])})
            for row in sorted(rows, key=lambda r: -(r["realised_vol"] or 0.0)):
                vol_text = ("n/a" if row["realised_vol"] is None
                            else f"{row['realised_vol']:.0%}")
                lines.append(
                    f"{row['symbol'].removesuffix('USDT')}: realised vol {vol_text} a year; beta "
                    f"{row['beta_open'] or 0:.2f} open / {row['beta_shut'] or 0:.2f} shut; QQQ "
                    f"-10% implies {(row['qqq_minus_10'] or 0):+.1f}%; worst 24-hour window "
                    f"{(row['worst_24h'] or 0):+.1f}%."
                )
            a, b = request.symbols[0], request.symbols[1]
            # Two contracts without a US-stock anchor trade on one 24/7 clock, so their
            # co-movement is read over every hour; restricting it to US hours dropped 70% of the
            # sample and moved LINK/BTC from 0.72 to 0.79 (2026-09-25 audit, round 2).
            round_clock = not any(x in TRADED_SYMBOLS or is_us_equity(x) for x in (a, b))
            if round_clock:
                _stamps, all_columns = align({k: v for k, v in data.raw.items() if k in (a, b)})
                rho = correlation(all_columns.get(a, []), all_columns.get(b, []))
            else:
                rho = correlation(columns.get(a, []), columns.get(b, []))
            if rho is not None:
                read = ("mostly the same bet" if abs(rho) >= 0.7 else
                        "a genuinely different bet" if abs(rho) <= 0.3 else "partly the same bet")
                when = ("over every hour of the last 30 days" if round_clock else
                        "in the open session")
                lines.insert(0, f"{a.removesuffix('USDT')} and {b.removesuffix('USDT')} move "
                                f"together at {rho:+.2f} {when} — {read}.")
            actionable = (_momentum_lead(rows, week=bool(_WEEK_ASKED.search(raw_text)))
                          if _asks_for_the_move(raw_text) else _compare_lead(rows))
            lines.insert(1, actionable + ". Listed from most to least volatile.")
            if re.search(r"\bcorrelat\w*|\bmove\s+together\b|\bco-?move", raw_text, re.I):
                # "How correlated is LINK to BTC" led on which is riskier; the figure asked
                # for leads (2026-09-25 audit, round 2).
                lines = _lead_with(lines, f"{a.removesuffix('USDT')} and ")
            payload["compare"] = rows
            sources.append(Source(kind="computation", ref="argus.desk.portfolio",
                                  detail="session betas, beta stress, realised worst window"))

        elif request.kind is ResearchKind.QUOTE:
            from argus.cost.model import CostModel
            from argus.market.bitget import fetch_tickers

            try:
                tickers = fetch_tickers()
            except Exception as exc:
                return Answer(
                    question=question, refused=True,
                    reason=f"the venue did not answer ({type(exc).__name__})",
                    lines=["Bitget did not return a quote just now, and a stale price presented "
                           "as current is worse than none. Try again in a minute."],
                )
            now = datetime.now().astimezone()
            anchor_open = is_open(now)
            fee = CostModel.bitget_perp().round_trip_bps()
            quoted = []
            for symbol in request.symbols:
                ticker = tickers.get(symbol)
                if ticker is None:
                    lines.append(f"{_t(symbol)}: no quote returned by Bitget just now.")
                    continue
                change = float(ticker.change_24h) * 100.0
                funding = float(ticker.funding_rate) * 100.0
                volume = ticker.base_volume * ticker.last
                lines.append(
                    f"{_t(symbol)} last {ticker.last} USDT on Bitget ({change:+.2f}% over 24h); "
                    f"bid {ticker.bid} / ask {ticker.ask}, spread {ticker.spread_bps:.1f}bps; "
                    f"24h range {ticker.low_24h} to {ticker.high_24h}; funding {funding:+.4f}% per "
                    f"interval; 24h volume about ${volume:,.0f}."
                )
                quoted.append((symbol, ticker))
                meaning = _funding_meaning(symbol, funding)
                if meaning:
                    lines.append(meaning)
            if quoted:
                symbol, ticker = quoted[0]
                all_in = fee + ticker.spread_bps
                payload["round_trip_bps"] = round(float(all_in), 2)
                lines.insert(0, (
                    f"Bottom line: a round trip in {_t(symbol)} costs about {all_in:.1f}bps "
                    f"({fee:.0f}bps taker fees + the {ticker.spread_bps:.1f}bps spread) — a trade "
                    f"needs a move bigger than that just to break even."
                ))
                if request.notional and _ROUND_TRIP.search(raw_text) and len(quoted) == 1:
                    sized_line = _sized_round_trip(symbol, Decimal(str(request.notional)), fee)
                    if sized_line is not None:
                        lines[0] = lines[0].replace("Bottom line: ", "", 1)
                        lines.insert(0, sized_line[0])
                        sources.append(sized_line[1])
                if (len(quoted) == 1 and (_PRICE_ASKED.search(raw_text)
                                          or PRICE_AT.match(raw_text))
                        and not re.search(r"\bcost|\bspread|\bfees?\b|round[\s-]?trip|"
                                          r"\bbreak[\s-]?even", raw_text, re.I)):
                    # "what is the NVDA price?" opened on the round-trip cost with the price
                    # second (2026-09-25 audit, round 2): the price asked for leads.
                    lines = _lead_with(lines, f"{_t(symbol)} last ")
                premium = _premium_line(symbol, ticker.last, anchor_open)
                if premium is None:
                    data = _no_candles("Bitget live ticker", "bitget /api/v2/mix/market/tickers",
                                       "last, bid, ask, 24h range, funding")
                lines.append(_session_line(symbol, anchor_open=anchor_open,
                                           us_listed=premium is not None))
                if premium is not None:
                    lines.append(premium[0])
                    sources.append(premium[1])
                implied = None if anchor_open else _implied_open_line(symbol, ticker.last)
                if implied is not None:
                    lines.append(implied[0])
                    sources.append(implied[1])
                if OPEN_INTEREST_QUESTION.search(raw_text):
                    # "current funding rate and open interest on ETH perp" is a quote with its
                    # open interest; the positioning lines answer the second half.
                    try:
                        from argus.market import open_interest

                        reading = open_interest.read(symbol)
                    except Exception:
                        reading = None
                    if reading is not None:
                        lines.extend(open_interest.lines(reading, _t(symbol)))
                        sources.append(Source(kind="venue",
                                              ref="bitget /api/v2/mix/market/tickers",
                                              detail="holdingAmount, every USDT perpetual"))
                if request.spot:
                    spot_lines, spot_sources = _rtoken_market_lines(request.spot, symbol,
                                                                    ticker, raw_text)
                    if spot_lines:
                        lines[0] = lines[0].replace("Bottom line: ", "", 1)
                        lines[0:0] = spot_lines
                        sources.extend(spot_sources)
                if implied is not None and IMPLIED_OPEN_QUESTION.search(raw_text):
                    # "where will NVDA open?" opened on the round-trip cost (live, 2026-09-25).
                    lines = _lead_with(lines, "Implied open:")
                extra_lines, extra_sources, asked = _quote_extras(raw_text, quoted, float(fee))
                lines.extend(extra_lines)
                sources.extend(extra_sources)
                if asked is not None:
                    # The question asked for one of these figures by name; it leads, and the
                    # round-trip line follows as context.
                    lines[0] = lines[0].replace("Bottom line: ", "", 1)
                    lines.insert(0, f"Bottom line: {asked}")
                stamp = quoted[0][1].fetched_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                lines.append(f"Quoted {stamp}.")
            payload["quotes"] = {
                s: {"last": str(t.last), "bid": str(t.bid), "ask": str(t.ask),
                    "change_24h": str(t.change_24h), "funding_rate": str(t.funding_rate),
                    "low_24h": str(t.low_24h), "high_24h": str(t.high_24h),
                    "fetched_at": t.fetched_at.isoformat()}
                for s, t in quoted
            }
            sources.append(Source(kind="computation", ref="argus.cost.model.CostModel.bitget_perp",
                                  detail="round-trip taker fee"))

        elif request.kind is ResearchKind.TECHNICALS:
            levels: dict[str, Any] = {}
            lines, extra = _technicals(request.symbols[0], found=levels)
            sources.extend(extra)
            if levels:
                payload["technicals"] = levels
            if daily_technicals_asked(raw_text):
                daily, daily_sources = _daily_technicals(request.symbols[0], raw_text)
                if daily:
                    # the daily figures asked for lead; the 4-hour Skill reading follows as the
                    # shorter-term context, its own lead demoted
                    lines = [*daily, *(re.sub(r"^(?:Actionable|Bottom line):\s*(\w)",
                                              lambda m: "Shorter term (4h): " + m.group(1),
                                              line) for line in lines)]
                    sources.extend(daily_sources)
            lines = _answer_the_state_asked(raw_text, request.symbols[0], lines)
            lines = _answer_the_level_asked(raw_text, lines)
            if not lines:
                # The Skill has no series for some listed contracts (CVXSTOCKUSDT, SP500USDT —
                # "No OHLCV data", measured 2026-09-23). The same indicators are computed from
                # Bitget's own 4h candles instead, and the answer says whose numbers they are.
                lines, extra = _technicals_computed(request.symbols[0])
                sources.extend(extra)
                if lines:
                    data = _no_candles("RSI, MACD and ATR computed by ARGUS from Bitget's live 4h "
                                       "candles, because bitget-signal has no series for this "
                                       "contract", "bitget /api/v3/market/candles",
                                       "4H; Wilder RSI(14), MACD(12,26,9), Wilder ATR(14)")
            if not lines:
                return Answer(
                    question=question, refused=True,
                    reason="bitget-signal's technical-analysis Skill did not answer and Bitget's "
                           "candles could not be read",
                    lines=["Neither Bitget's technical-analysis Skill nor Bitget's own candles "
                           "answered just now, so there is nothing to read the tape from. Try "
                           "again shortly."],
                )

        elif request.kind is ResearchKind.FUNDAMENTALS:
            if len(request.symbols) > 1:
                with ContextPool(max_workers=len(request.symbols)) as pool:
                    parts = list(pool.map(lambda sym: _fundamentals(sym, raw_text),
                                          request.symbols))
                lines, extra = [], []
                for symbol, (part_lines, part_sources) in zip(request.symbols, parts,
                                                               strict=True):
                    lines.extend(LEAD.sub(f"Bottom line ({_t(symbol)}): ", line, count=1)
                                 if LEAD.match(line) else f"{_t(symbol)} — {line}"
                                 for line in part_lines)
                    extra.extend(part_sources)
            else:
                figures: dict[str, Any] = {}
                lines, extra = _fundamentals(request.symbols[0], raw_text, found=figures)
                if figures:
                    payload["fundamentals"] = figures
            lines = _lead_with_what_was_asked(lines, raw_text)
            if len(request.symbols) > 1 and _VALUATION.search(raw_text):
                try:
                    compared = _valuation_compare(request.symbols)
                except Exception:
                    compared = []
                if compared:
                    lines = [compared[0], *(re.sub(r"^(?:Actionable|Bottom line)(?: "
                                                   r"\(\w+\))?:\s*", "", line)
                                            for line in lines)]
            sources.extend(extra)
            if not extra:
                data = _no_candles("Bitget's contract list, which flags this contract as not a "
                                   "company's shares — no equity data was requested",
                                   "bitget /api/v2/mix/market/contracts", "isRwa + asset class")

        elif request.kind is ResearchKind.ANALOGUE:
            lines, extra, report_dict = _analogue(request.symbols[0], data)
            sources.extend(extra)
            payload["analogue"] = report_dict
            long_lines, long_sources, long_run = _long_run(request.symbols[0])
            if long_lines:
                lines.extend(long_lines)
                sources.extend(long_sources)
                payload["long_run"] = long_run
            if request.horizon_hours is not None:
                odds_lines, odds_sources, odds_dict = _odds_lines(request, data)
                if odds_lines:
                    if (request.horizon_hours or 24) >= 24 or request.weekend:
                        data = replace(data, provenance=(
                            f"Bitget daily closes up to 500 days for the {odds_dict.get('windows')}"
                            f" past windows, and {data.provenance} for the comparable states"))
                    lines = [*odds_lines, *(
                        "Comparable states, next 24h: "
                        + LEAD.sub("", line, count=1) if bool(LEAD.match(line))
                        else line for line in lines)]
                    sources[0:0] = odds_sources
                    payload["odds"] = odds_dict
                    # the line the question asked for leads: a stop, a take-profit, a gap
                    if _TAKE_PROFIT_Q.search(raw_text):
                        lines = _lead_with(lines, "Where a take-profit sits")
                    elif _STOP_QUESTION.search(raw_text):
                        lines = _lead_with(lines, "Where a stop sits")
                    elif _LOSS_OVER_PERIOD.search(raw_text):
                        lines = _loss_lead(lines, odds_dict, request)
                    elif _WEEKEND_GAP_Q.search(raw_text):
                        lines = _lead_with(lines, "Typical ")

        elif request.kind is ResearchKind.EXECUTION:
            from argus.desk.workbench import plan_execution
            from argus.market.bitget import fetch_tickers

            symbol = request.symbols[0]
            assert request.notional is not None
            adv: Decimal | None = None
            try:
                ticker = fetch_tickers().get(symbol)
                if ticker is not None:
                    adv = ticker.base_volume * ticker.last
            except Exception:
                adv = None
            # Only a contract with an off-chain anchor sleeps: "the stock's own market is shut"
            # was said of a BTC order (2026-09-25 audit).
            asleep = (not is_open(datetime.now().astimezone())
                      and (symbol in TRADED_SYMBOLS or is_us_equity(symbol)))
            if adv is None or adv <= 0:
                return Answer(
                    question=question, refused=True,
                    reason="the venue's 24h volume could not be read, and a plan without it "
                           "would state a participation rate it does not know",
                    lines=[f"I could not read {symbol}'s 24h volume from Bitget just now, and "
                           f"splitting an order without it means guessing its footprint. Try "
                           f"again in a minute."],
                )
            plan = plan_execution(symbol=symbol, notional=request.notional, adv_notional=adv,
                                  urgency="high" if request.urgent else "low",
                                  anchor_asleep=asleep)
            fees = sum((s.fraction * s.expected_cost_bps for s in plan.slices), Decimal(0))
            own_sigma = _daily_volatility_bps(symbol)
            from argus.cost.model import CostModel

            modelled = CostModel.bitget_perp().impact_bps(plan.participation_rate, own_sigma)
            lines = [
                f"A ${request.notional:,.0f} order is {plan.participation_rate:.2%} of "
                f"{symbol.removesuffix('USDT')}'s 24h volume (${adv:,.0f}); the fees come to "
                f"about {fees:.1f}bps, and the square-root impact law — calibrated on Bitget's "
                f"own books"
                + (f", at {symbol.removesuffix('USDT')}'s own {own_sigma:.0f}bps daily volatility"
                   if own_sigma is not None else "")
                + f" — puts this size's market impact near {modelled:.1f}bps; the live book "
                  f"below shows what taking it at once costs.",
                f"Plan: {plan.rationale}.",
            ]
            measured: dict[str, float] = {}
            lines.extend(_depth_lines(symbol, request.notional, adv, plan, raw_text,
                                      urgent=request.urgent, modelled=float(modelled),
                                      measured=measured))
            payload["execution"] = {
                "participation": str(plan.participation_rate),
                "expected_cost_bps": str(plan.expected_total_cost_bps),
                # The live book's all-in figures, as the lines above state them; absent when the
                # book did not answer or could not hold the order.
                **{key: round(value, 2) for key, value in measured.items()},
                "slices": [{"fraction": str(s.fraction), "style": s.style,
                            "cost_bps": str(s.expected_cost_bps)} for s in plan.slices],
            }
            sources.append(Source(kind="computation", ref="argus.desk.workbench.plan_execution",
                                  detail="superlinear impact; passive slices quoted at taker"))
            # The first child as the Agent Hub command that previews it (audit finding 113),
            # kept out of the cleaning below, which would strip USDT from the symbol it names.
            agent_hub_lines = _agent_hub_lines(symbol, request, plan, ticker, raw_text)
            if agent_hub_lines:
                sources.append(Source(kind="computation", ref="argus.lui.agenthub",
                                      detail="bgc order --dry-run, checked in "
                                             "data/agenthub_preview.json"))
    except PortfolioError as exc:
        return Answer(question=question, refused=True, reason=str(exc),
                      lines=[f"The risk engine refused this one: {exc}"], sources=[data.source])

    # The conclusion a trader acts on leads; the evidence follows it. Engine lines are written
    # lower-case for the CLI's indented layout, so they are sentence-cased for reading here.
    lines = [clean_line(line) for line in lines]
    lines = [line[:1].upper() + line[1:] for line in lines]
    lines.sort(key=lambda line: 0 if bool(LEAD.match(line)) else 1)
    lines.extend(agent_hub_lines)
    # A "read as" note names both contracts on purpose (CVXSTOCKUSDT vs CVXUSDT); stripping the
    # suffixes there would make the two identical and the note meaningless.
    lines.extend(f"Assumed: {note if ' read as ' in note else clean_line(note)}."
                 for note in request.notes)
    lines.append(f"Data: {data.provenance}. This is analysis, not advice — you make the call.")
    return Answer(question=question, lines=lines, sources=sources, data=payload)


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
