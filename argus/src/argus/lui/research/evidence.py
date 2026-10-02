"""Supporting reads an answer adds: treasury holdings, dividends, the desk's own record, flows,
crowd and Skill readings."""

from __future__ import annotations

import re
from typing import Any

from argus.lui.answer import Answer, Source
from argus.lui.question import (
    Question,
)
from argus.lui.research.anchor import (
    _premium_line,
)
from argus.lui.research.kinds import (
    ResearchRequest,
    _t,
)
from argus.lui.research.parse import (
    _unit_notional,
)
from argus.lui.trace import trace_module
from argus.truth.coverage import ContextPool

CRYPTO_SENTIMENT = frozenset({"COINUSDT", "MSTRUSDT"})
"""Stocks whose price follows crypto closely enough that the crypto backdrop belongs in their
sentiment answer: an exchange and a bitcoin treasury company."""


def _bitcoin_treasury_line(ticker: str, market_cap: float) -> str | None:
    """A listed company's bitcoin and what its market value pays for it, from Bitget's service
    and Bitget's own BTC price."""
    from argus.market.bitget import fetch_tickers
    from argus.market.bitget_positioning import bitcoin_treasury

    try:
        btc = fetch_tickers().get("BTCUSDT")
        price = float(btc.last) if btc is not None else None
    except Exception:
        price = None
    try:
        return bitcoin_treasury(ticker, market_cap, price)
    except Exception:
        return None


def _ex_dividend_line(ticker: str) -> str | None:
    from argus.market.bitget_positioning import next_ex_dividend

    try:
        return next_ex_dividend(ticker)
    except Exception:
        return None


def _desk_integrity_read(symbol: str) -> dict[str, Any] | None:
    """The desk's own last sentiment-integrity read of ``symbol``: the analyst panel's consensus
    after `SourceIndependenceGraph` discounted agreement that shared its sources, from the notes the
    desk wrote at that decision. None when the desk has not decided on the name."""
    import json as _json

    from argus.lui.answer import desk_notes_path

    try:
        lines = desk_notes_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    panel = re.compile(r"^panel: (\d+) analysts?, (\d+) distinct sources?, independence "
                       r"([\d.]+) -> (\w+) at ([\d.]+) after provenance discount")
    for line in reversed(lines):
        try:
            row = _json.loads(line)
        except ValueError:
            continue
        if row.get("symbol") != symbol:
            continue
        for note in row.get("notes", []):
            found = panel.match(str(note))
            if found:
                return {"seq": row.get("seq"), "at": row.get("at"),
                        "analysts": int(found.group(1)), "sources": int(found.group(2)),
                        "signal": found.group(4).replace("_", " "),
                        "confidence": float(found.group(5))}
        return None
    return None


SPOT_HEDGE_DAYS = 240
"""Nights of history a spot hedge is fitted and tested on: about 165 sessions, enough for a held-out
third of fifty nights and a dozen weekends, fetched in a few seconds."""


_WITH_ITS_PERP = re.compile(r"\bwith\s+(?:the|its|a|an|my)?\s*(?:\w+\s+)?perp(?:etual)?s?\b|"
                            r"\busing\s+(?:the|its)\s+perp(?:etual)?\b", re.I)


def _same_name_perp_hedge(symbol: str, raw: str) -> tuple[list[str], list[Source]] | None:
    """Shares hedged with the same company's perpetual: one-for-one, and what it costs and misses.
    "how do i hedge my 500 shares of aapl with the perp" was answered with SPY, QQQ and SMH on a
    $100,000 default book (answer audit, round 3)."""
    from argus.market.bitget import fetch_tickers

    try:
        ticker = fetch_tickers()[symbol]
    except Exception:
        return None
    base = _t(symbol)
    sized = _unit_notional(raw, symbol)
    units = None
    if sized is not None:
        units = float(sized[0]) / float(ticker.last)
    last = float(ticker.last)
    rate = float(ticker.funding_rate) * 100
    hours = 8
    week = rate * (24 / hours) * 7
    lines = [
        "Bottom line: "
        + (f"short {units:,.0f} {base} on Bitget's {base} perpetual — about ${units * last:,.0f} "
           f"at {last:,.2f} — to hedge {units:,.0f} shares one for one."
           if units else
           f"short the same quantity of {base} on Bitget's {base} perpetual as the shares you "
           f"hold — one for one.")
        + " The perpetual tracks the same company, so the hedge removes the share's price risk, "
          "news included — unlike an index hedge, which leaves the company's own moves open.",
        ("Carrying it: funding is flat right now, so the short costs nothing to hold beyond "
         "about 12bps to open and close it." if abs(rate) < 0.00005 else
         f"Carrying it: a short {'receives' if rate > 0 else 'pays'} funding at the current "
         f"{rate:+.4f}% per {hours}h, about {abs(week):.2f}% of the hedge a week"
         + (f" (${abs(week) / 100 * units * last:,.0f} on this size)" if units else "")
         + ", plus about 12bps to open and close it."),
    ]
    premium = _premium_line(symbol, ticker.last, False)
    if premium is not None:
        lines.append(premium[0])
    lines.append("What it misses: while the US market is shut the perpetual keeps trading and the "
                 "shares do not, so the two can diverge overnight and at weekends and meet again "
                 "at the open; the hedge is exact at the close, not at every hour. A perpetual "
                 "short also needs margin and can be liquidated if the stock rallies hard.")
    sources = [Source(kind="venue", ref="bitget /api/v2/mix/market/tickers",
                      detail=f"{symbol} last and funding")]
    return lines, sources


def _spot_hedge_answer(question: Question, request: ResearchRequest) -> Answer:
    """The same-company perpetual hedge for a spot rToken holder, tested on nights it had not seen,
    set beside the index hedge ARGUS offered before it knew the spot token existed
    (`eval/copilot_hedge.py`: 10.1% against 99.7% of the overnight swing on Ballast's nights)."""
    from argus.desk.rtoken_hedge import HedgeUnavailable, overnight_hedge, render
    from argus.market.rtoken_spot import SpotError, hourly_bars, overnight_returns

    spot = request.spot or ""
    perp = request.symbols[0]
    ticker = perp.removesuffix("USDT")
    legs = {spot: True, perp: False, "QQQUSDT": False}
    with ContextPool(max_workers=3) as pool:
        futures = {name: pool.submit(hourly_bars, name, spot=is_spot, days=SPOT_HEDGE_DAYS)
                   for name, is_spot in legs.items()}
        nights: dict[str, Any] = {}
        failures: dict[str, str] = {}
        for name, future in futures.items():
            try:
                nights[name] = overnight_returns(future.result())
            except (SpotError, OSError) as exc:
                failures[name] = str(exc)
    try:
        if spot in failures or perp in failures:
            raise HedgeUnavailable(failures.get(spot) or failures.get(perp) or "")
        hedge = overnight_hedge(spot, perp, nights[spot], nights[perp])
    except HedgeUnavailable as exc:
        return Answer(question=question, refused=True,
                      reason=f"no overnight hedge could be measured for {spot}: {exc}",
                      lines=[f"I could not measure a hedge for {spot} ({exc}). Either it is not a "
                             f"Bitget rToken with a {ticker} stock perpetual beside it, or its "
                             f"history is too short to test a hedge on. Nothing is guessed."])
    lines = render(hedge, ticker)
    if "QQQUSDT" in nights:
        try:
            index = overnight_hedge(spot, "QQQUSDT", nights[spot], nights["QQQUSDT"])
            lines.insert(2, f"Why not an index hedge: shorting QQQUSDT against the same token "
                            f"removed only {index.held_out_variance_removed:.1%} of its "
                            f"overnight swing on the same held-out nights — the other "
                            f"{1 - index.held_out_variance_removed:.0%} is {ticker}'s own.")
        except HedgeUnavailable:
            pass
    lines.extend(f"Assumed: {note}." for note in request.notes)
    lines.append(f"Data: live Bitget hourly candles for {spot} (spot) and {perp} (perpetual), "
                 f"{SPOT_HEDGE_DAYS} days, each night measured from the US close to the next "
                 f"open. Method from Ballast (an S2 entry, MIT). This is analysis, not advice — "
                 f"you make the call.")
    sources = [
        Source(kind="venue", ref=f"bitget history-candles {spot} (spot) + {perp}",
               detail="public hourly candles, both legs"),
        Source(kind="computation", ref="argus.desk.rtoken_hedge.overnight_hedge",
               detail="minimum-variance ratio; fitted on 70% of nights, scored on the rest"),
    ]
    return Answer(question=question, lines=lines, sources=sources,
                  data={"request": request.as_dict(), "hedge": hedge.as_dict()})


def _beta_track_record() -> str | None:
    """How far this copilot's post-trade beta has been from what books then did, measured against
    weekend-copilot (an S2 entry answering the same question) in `data/copilot_rivals.json`. Both
    of that run's losses are carried in the sentence, not only the win."""
    import json as _json

    from argus.lui.answer import desk_notes_path

    try:
        report = _json.loads((desk_notes_path().parent / "copilot_rivals.json")
                             .read_text(encoding="utf-8"))
        err = report["beta_error"]["bitget_daily"]["mean_abs_error"]
        books = report["beta_error"]["bitget_daily"]["books_scored"]
        shipped = report["comparisons"]["bitget_daily"]["argus_open vs rival_spy"]
        same = report["comparisons"]["bitget_daily"]["argus_open vs rival_qqq"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    text = (f"How far to trust that beta: on {books:,} test books, it missed the next four "
            f"weeks' realised beta by {err['argus_open']:.2f} on average, against "
            f"{err['rival_spy']:.2f} for the best other tool measured on the same books (a "
            f"Season 2 entry, named on /proof) "
            f"(p = {shipped['wilcoxon_p']:.2f}; {err['rival_qqq']:.2f} on the same benchmark, "
            f"p = {same['wilcoxon_p']:.2f}, not significant) and {err['naive_one']:.2f} for "
            f"simply assuming a beta of 1.0 — the comparison is on /proof")
    try:
        stress = _json.loads((desk_notes_path().parent / "copilot_stress.json")
                             .read_text(encoding="utf-8"))["down_1pct"]
        miss = stress["mean_abs_error_pp"]
        text += (f". The QQQ-fall lines were off by {miss['argus_beta']:.2f} points on the "
                 f"{stress['days']} days QQQ really fell 1% or more ({miss['skfolio_vine']:.2f} "
                 f"for the strongest statistical model tested), and they are a centre, not a "
                 f"worst case")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return text


def _coordination_test(symbol: str | None) -> tuple[str, dict[str, Any]] | None:
    """The measured coordinated-posting result, from `data/sentiment_comparison.json`: finBERT
    against the desk's sentiment analyst on near-identical unsourced posts. The narrative about
    ``symbol`` is quoted when the test has one."""
    import json as _json

    from argus.lui.answer import desk_notes_path

    try:
        report = _json.loads((desk_notes_path().parent / "sentiment_comparison.json")
                             .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    runs = report.get("narratives") or []
    if not runs:
        return None
    louder = sum(1 for r in runs if r.get("finbert_signal_scales_with_repetition"))
    held = sum(1 for r in runs if r.get("argus_discounts_coordination"))
    mine = next((r for r in runs if symbol and symbol in str(r.get("narrative", ""))), None)
    posts = (mine or runs[0]).get("finbert_coordinated", {}).get("n_posts", 5)
    when = str(report.get("generated_at", ""))[:10]
    text = (f"Tested, not assumed ({when}, data/sentiment_comparison.json): {posts} near-identical "
            f"unsourced posts per narrative across {len(runs)} narratives. finBERT, the standard "
            f"finance sentiment model, grew louder with every repeat on {louder} of {len(runs)}; "
            f"this desk's sentiment analyst stayed non-actionable on {held} of {len(runs)}")
    if mine is not None:
        coordinated = mine.get("finbert_coordinated", {})
        reason = str(mine.get("argus_coordinated", {}).get("reasoning", ""))
        first = re.split(r"(?<=[.!?])\s", reason, maxsplit=1)[0]
        text += (f". On the {_t(symbol or '')} narrative finBERT counted "
                 f"{coordinated.get('matching_count')} of {coordinated.get('n_posts')} posts as "
                 f"{coordinated.get('dominant_label')}; the analyst's reason: \u201c{first}\u201d")
    # The fair rival too, so the line does not rest on the weakest version of the general tool:
    # collapsing near-duplicates before finBERT stops the template copies, not the retellings
    # (`eval/sentiment_dedup_rival.py`).
    try:
        dedup = _json.loads((desk_notes_path().parent / "sentiment_dedup_rival.json")
                            .read_text(encoding="utf-8"))["summary"]
        text += (f". Deduplicating first fixes the copies (finBERT then moved on "
                 f"{dedup['template']['finbert_dedup_moved']} of {dedup['template']['cases']}, "
                 f"the analyst on {dedup['template']['argus_moved']}), not five accounts retelling "
                 f"one rumour in their own words ({dedup['diverse']['finbert_dedup_moved']} of "
                 f"{dedup['diverse']['cases']} against the analyst's "
                 f"{dedup['diverse']['argus_moved']})")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return text + ".", {"narratives": len(runs), "finbert_louder": louder, "held": held}


def _flow_lines(symbol: str) -> list[str]:
    """Spot bitcoin and ether ETF flows, and Strategy's bitcoin buying, for the crypto-linked
    names, from the latest `market/etf_flows` snapshot."""
    if not symbol:
        return []
    from argus.lui.answer import desk_notes_path
    from argus.market import etf_flows

    return etf_flows.lines_for(symbol, etf_flows.load(desk_notes_path().parent / "etf_flows.json"))


def _crowd_lines(symbol: str) -> list[str]:
    """What X and Reddit carry about ``symbol``, from the latest `market/social_pulse` snapshot,
    with its age. The snapshot is collected on the desk's machine (the platforms need a logged-in
    session the hosted console cannot hold) and published beside the other slow sweeps."""
    from argus.lui.answer import desk_notes_path
    from argus.market import social_pulse

    return social_pulse.lines_for(symbol, social_pulse.load(
        desk_notes_path().parent / "social_pulse.json"))


BASIS_AGREE_BPS = 50.0
"""A spot reading from the Skill further than this from Bitget's perpetual is called a
disagreement: BTC's perpetual basis on Bitget is normally a few basis points."""


def _skill_btc_line(job: Any, perp: Any) -> tuple[str, Source] | None:
    """One line checking bitget-signal's crypto_derivatives BTC/USDT reading against Bitget's own
    perpetual ticker, or None when the Skill did not answer in time (it has no mirror, and a
    missing cross-check is not worth a line)."""
    if job is None or perp is None:
        return None
    try:
        routed = job.result(timeout=6.0)
        payload = routed.payload if routed.via == "skill" else None
        spot = float(payload["last"]) if isinstance(payload, dict) else 0.0
        change = payload.get("change_pct") if isinstance(payload, dict) else None
        last = float(perp.last)
    except Exception:
        return None
    if spot <= 0 or last <= 0:
        return None
    basis = (last / spot - 1.0) * 1e4
    verdict = ("the Skill and Bitget's own ticker agree" if abs(basis) <= BASIS_AGREE_BPS
               else "a gap that size is not a basis — one of the two readings is stale")
    line = (f"Cross-check: bitget-signal's crypto_derivatives reads BTC/USDT spot on Bitget at "
            f"{spot:,.0f}" + (f" ({float(change):+.2f}% over 24h)" if change is not None else "")
            + f"; the perpetual trades at {last:,.0f}, a {basis:+.1f}bp basis — {verdict}.")
    return line, routed.source()


def _skill_fear_greed(payload: Any) -> dict[str, Any] | None:
    """The index value and its label from bitget-signal's sentiment_index, whichever of the shapes
    its upstream uses ("value"/"value_classification", "classification", or a ``data`` list), or
    None when neither is there — a reading this cannot parse is not presented as one."""
    node = payload
    if isinstance(node, dict) and isinstance(node.get("data"), list) and node["data"]:
        node = node["data"][0]
    if isinstance(node, dict) and isinstance(node.get("current"), dict):
        node = node["current"]
    if not isinstance(node, dict):
        return None
    try:
        value = int(float(node["value"]))
    except (KeyError, TypeError, ValueError):
        return None
    label = str(node.get("value_classification") or node.get("classification") or "")
    return {"value": value, "value_classification": label}


# Every engine here is a traced step from import on (lui/trace.py, trace_module).
trace_module(globals())
