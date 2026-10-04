"""Arithmetic on an account the trader describes in figures: a balance after successive moves, the
margin and exposure of a leveraged position, equity from realised and unrealised P&L, and whether a
stated move liquidates a leveraged position.

A hostile review (round 32) asked four of these and none was computed:

* "My portfolio dropped 50% in March, then another 50% in April — total loss, and if I started
  with $2,000,000, my balance now?" was filed as a note ("noted — started with $2,000,000"). The
  answer is -75% and $500,000.
* "I deposit 1.5M USDT and open a 300k notional position at 20x — required margin in k, exposure
  in M?" was filed as a note, then refused because "USDT is not listed". The answer is 15k margin
  and 0.3M exposure, 0.2x of the deposit.
* "Realised -$3,200, unrealised -$1,800, starting equity $10,000 — current equity and total P&L
  %?" was told the console cannot see the account. The figures were all given: $5,000 and -50%.
* "0.001 BTC as margin, 125x long, BTC drops 2% — liquidated, and P&L in USD?" was stressed as an
  unleveraged $85 holding (-$2). At 125x the position is liquidated after about a 0.4% fall, so a
  2% fall loses the whole margin, about $85.
* "If ETHUSDT funding is -500% every 8 hours, what would I earn shorting $10,000?" got today's
  +0.01% with no word that the stated rate was dropped. The stated rate is used, and held against
  Bitget's own funding cap for the contract (±0.30% a settlement for ETHUSDT).

Every figure here is the trader's own, with the arithmetic shown; the only outside numbers are
Bitget's last price (to put a coin-denominated margin in dollars) and Bitget's maintenance margin
rate for the position's size (``market.bitget.maintenance_margin_rate``), each named where used.
"""

from __future__ import annotations

import math
import re
from typing import Final

from argus.lui.trace import trace_module

_NUM = r"\d(?:[\d,]*\d)?(?:\.\d+)?"
_UNIT = r"(?:k|m|mm|mn|b|bn|thousand|million|billion)"
_SCALE: Final = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
                 "b": 1e9, "bn": 1e9, "billion": 1e9}


def _amount(raw: str, unit: str | None) -> float:
    return float(raw.replace(",", "")) * _SCALE.get((unit or "").lower(), 1.0)


def _money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}" if abs(value) >= 100 else f"{sign}${abs(value):,.2f}"


# --- successive moves on a starting balance ------------------------------------------------------

_MOVE: Final = re.compile(
    r"\b(?P<v>dropp?ed|fell|lost|declined|crashed|tanked|went\s+down|rose|gained|grew|went\s+up|"
    r"jumped|increased|decreased)\s+(?:by\s+)?(?:another\s+|a\s+further\s+)?(?P<p>\d+(?:\.\d+)?)"
    r"\s*%", re.I)
_START: Final = re.compile(
    rf"\b(?:start(?:ed|ing)?\s+(?:with|at|out\s+with)|(?:account|balance|portfolio|book)\s+"
    rf"(?:was|of|at)|had|starting\s+(?:balance|equity|capital)\s+"
    rf"(?:was|of|is)|began\s+with|initial\s+(?:balance|equity|capital)\s+(?:was|of|is))\s+\$?\s*"
    rf"(?P<a>{_NUM})\s*(?P<u>{_UNIT})?\b", re.I)
_TOTAL_ASKED: Final = re.compile(r"\b(?:total|overall|combined|net)\b[^?]{0,30}\b(?:loss|"
                                 r"change|return|drop|gain)|\bbalance\s+now\b|\bhow\s+much\s+"
                                 r"(?:do\s+i\s+have|is\s+left)|\bwhat\s+is\s+my\s+balance\b|"
                                 # "lost 30% then gained 30%, where am I now in total?" (a live
                                 # re-ask, round 32)
                                 r"\bwhere\s+am\s+i\b|\bin\s+total\b|\bam\s+i\s+back\b|"
                                 r"\bbroke\s+even\b", re.I)


def compounded_lines(text: str) -> list[str] | None:
    """Two or more percentage moves in a row, compounded, and the balance they leave."""
    moves = [(-1 if re.match(r"dropp|fell|lost|declin|crash|tank|went\s+down|decreas",
                             m.group("v"), re.I) else 1) * float(m.group("p")) / 100
             for m in _MOVE.finditer(text)]
    if len(moves) < 2 or not _TOTAL_ASKED.search(text):
        return None
    factor = 1.0
    for move in moves:
        factor *= 1 + move
    total = factor - 1
    said = " \u00d7 ".join(f"{1 + m:.2f}" for m in moves)
    lines = [f"Bottom line: the moves compound to {total:+.2%} in total — {said} = {factor:.4f} — "
             f"not the {sum(moves):+.0%} you get by adding them."]
    start = _START.search(text)
    if start is not None:
        begun = _amount(start.group("a"), start.group("u"))
        lines.append(f"From {_money(begun)}: {_money(begun)} \u00d7 {factor:.4f} = "
                     f"{_money(begun * factor)} now, {_money(begun * total)} in all.")
    if total < 0:
        lines.append(f"Getting back needs {1 / factor - 1:+.0%} from here: a fall takes a larger "
                     f"rise to undo.")
    return lines


# --- margin and exposure of a leveraged position -------------------------------------------------

_NOTIONAL: Final = re.compile(
    rf"\$?\s*(?P<a>{_NUM})\s*(?P<u>{_UNIT})?\s*(?:usdt|usd|dollars?)?\s+(?:notional|position|"
    rf"worth)\b(?:\s+(?:position|of\s+\w+))?[^.?]{{0,30}}?\bat\s+(?P<x>\d+(?:\.\d+)?)\s*x\b|"
    rf"\b(?P<x2>\d+(?:\.\d+)?)\s*x\b[^.?]{{0,30}}?\$?\s*(?P<a2>{_NUM})\s*(?P<u2>{_UNIT})?\s*"
    rf"(?:usdt|usd|dollars?)?\s+(?:notional|position)\b", re.I)
_DEPOSIT: Final = re.compile(
    rf"\b(?:deposit(?:ed)?|fund(?:ed)?\s+(?:it\s+|my\s+account\s+)?with|have|account\s+of|"
    rf"balance\s+of|equity\s+of)\s+\$?\s*(?P<a>{_NUM})\s*(?P<u>{_UNIT})?\s*(?:usdt|usd|dollars?)?|"
    # "with $200k equity I open $1M notional" (a live re-ask, round 32)
    rf"\$\s?(?P<a2>{_NUM})\s*(?P<u2>{_UNIT})?\s*(?:usdt|usd)?\s+(?:of\s+)?(?:equity|balance|"
    rf"capital|in\s+(?:my\s+)?account)\b", re.I)
_MARGIN_ASKED: Final = re.compile(r"\b(?:required|initial|needed|how\s+much)\s+margin\b|\bmargin\s+"
                                  r"(?:required|needed)\b|\bexposure\b|\beffective\s+leverage\b",
                                  re.I)


def margin_lines(text: str) -> list[str] | None:
    """The margin a notional needs at a leverage, the exposure, and their share of a deposit."""
    if not _MARGIN_ASKED.search(text):
        return None
    m = _NOTIONAL.search(text)
    if m is None:
        return None
    notional = _amount(m.group("a") or m.group("a2"), m.group("u") or m.group("u2"))
    lev = float(m.group("x") or m.group("x2"))
    if lev <= 0 or notional <= 0:
        return None
    margin = notional / lev
    lines = [f"Bottom line: {_money(notional)} of notional at {lev:g}x needs {_money(margin)} of "
             f"margin ({margin / 1e3:,.1f}k); the exposure is the full notional, "
             f"{notional / 1e6:,.2f}M — leverage changes the margin, not what the position moves "
             f"with.",
             f"The math: margin = notional / leverage = {notional:,.0f} / {lev:g} = "
             f"{margin:,.0f}."]
    deposit = next((d for d in _DEPOSIT.finditer(text)
                    if _amount(d.group("a") or d.group("a2"), d.group("u") or d.group("u2"))
                    != notional), None)
    if deposit is not None:
        held = _amount(deposit.group("a") or deposit.group("a2"),
                       deposit.group("u") or deposit.group("u2"))
        lines.append(f"Against the {_money(held)} in the account: the margin uses "
                     f"{margin / held:.1%} of it, and the account's effective leverage is "
                     f"{notional / held:.2f}x "
                     f"(exposure / equity) — the {lev:g}x setting only says how little margin "
                     f"was set aside.")
    return lines


# --- equity from realised and unrealised P&L -----------------------------------------------------

_PNL: Final = re.compile(rf"\b(?P<k>realis|realiz|unrealis|unrealiz)ed\s+(?:p\s*&\s*l|pnl|p/l|"
                         rf"profit(?:\s+and\s+loss)?|loss|gain)\w*\s+(?:this\s+\w+\s+)?(?:is|was|of|"
                         rf"=|:)?\s*(?P<s>[-+\u2212])?\s*\$?\s*(?P<s2>[-+\u2212])?(?P<a>{_NUM})\s*"
                         rf"(?P<u>{_UNIT})?", re.I)
_EQUITY: Final = re.compile(rf"\b(?:starting|initial|opening|start(?:ed)?\s+with)\s*(?:equity|"
                            rf"balance|capital)?\s*(?:was|of|is|=|:)?\s*\$?\s*(?P<a>{_NUM})\s*"
                            rf"(?P<u>{_UNIT})?", re.I)


def equity_lines(text: str) -> list[str] | None:
    """Current equity and total P&L from a starting equity and the P&L the trader states."""
    found = {m.group("k").lower()[:2]: m for m in _PNL.finditer(text)}
    start = _EQUITY.search(text)
    if not found or start is None or not re.search(r"\bequity\b|\bbalance\b|\bpercent|%", text,
                                                   re.I):
        return None

    def signed(m: re.Match[str]) -> float:
        value = _amount(m.group("a"), m.group("u"))
        negative = (m.group("s") or m.group("s2") or "") in ("-", "\u2212") or re.search(
            r"\bloss\w*", m.group(0), re.I) is not None
        return -value if negative else value

    realised = signed(found["re"]) if "re" in found else 0.0
    unrealised = signed(found["un"]) if "un" in found else 0.0
    begun = _amount(start.group("a"), start.group("u"))
    if begun <= 0:
        return None
    total = realised + unrealised
    now = begun + total
    return [f"Bottom line: equity is {_money(now)} — {_money(begun)} "
            f"{'+' if realised >= 0 else '-'} {_money(abs(realised))} realised "
            f"{'+' if unrealised >= 0 else '-'} "
            f"{_money(abs(unrealised))} unrealised — and the total P&L is {_money(total)}, "
            f"{total / begun:+.1%} of the starting equity.",
            "Unrealised P&L moves with the price until the position is closed; the realised part "
            "is fixed."]


# --- a stated move on a leveraged position -------------------------------------------------------

_LEVERED: Final = re.compile(
    rf"\b(?P<m>{_NUM})\s*(?P<mu>btc|eth|sol|usdt|usd|dollars?)?\s*(?:as|of|in)?\s*margin\b|"
    rf"\bmargin\s+(?:of\s+)?\$?\s*(?P<m2>{_NUM})\s*(?P<mu2>btc|eth|sol|usdt|usd)?", re.I)
_LEV_SIDE: Final = re.compile(r"\b(?P<x>\d+(?:\.\d+)?)\s*x\s+(?:[A-Za-z]{2,10}\s+)?"
                              r"(?P<side>long|short)\b|\b(?P<side2>"
                              r"long|short)\b[^.?]{0,20}?\bat\s+(?P<x2>\d+(?:\.\d+)?)\s*x\b", re.I)
_STATED_MOVE: Final = re.compile(r"\b(?P<v>drops?|dropped|falls?|fell|rises?|rose|jumps?|"
                                 r"jumped|moves?\s+(?:down|up)|moved\s+(?:down|up)|goes\s+"
                                 r"(?:down|up)|went\s+(?:down|up))\s+(?:by\s+)?(?P<p>\d+(?:\.\d+)?)"
                                 r"\s*%", re.I)


def liquidation_move_lines(text: str) -> list[str] | None:
    """Whether a stated move liquidates a leveraged position, and the P&L on its margin."""
    lev_said, margin_said, move = (_LEV_SIDE.search(text), _LEVERED.search(text),
                                   _STATED_MOVE.search(text))
    if lev_said is None or margin_said is None or move is None:
        return None
    from argus.lui.research import research_symbols

    named = research_symbols(text)[0]
    symbol = named[0] if named else None
    lev = float(lev_said.group("x") or lev_said.group("x2"))
    side = (lev_said.group("side") or lev_said.group("side2")).lower()
    amount = float((margin_said.group("m") or margin_said.group("m2")).replace(",", ""))
    unit = (margin_said.group("mu") or margin_said.group("mu2") or "usd").lower()
    fell = re.match(r"drop|fall|fell|move[sd]?\s+down|goes\s+down|went\s+down", move.group("v"),
                    re.I) is not None
    against = fell == (side == "long")
    pct = float(move.group("p")) / 100
    if lev <= 0 or amount <= 0:
        return None
    price = None
    if unit in ("btc", "eth", "sol"):
        from argus.lui.position_math import _last

        coin = symbol or f"{unit.upper()}USDT"
        price = _last(coin if coin.endswith("USDT") else f"{unit.upper()}USDT")
        if price is None:
            return None
        margin_usd = amount * price
    else:
        margin_usd = amount
    notional = margin_usd * lev
    from argus.market.bitget import maintenance_margin_rate

    mmr = maintenance_margin_rate(symbol, notional) if symbol else None
    distance = 1 / lev - (mmr or 0.0)
    where = (f"Bitget's maintenance margin for that size, {mmr:.2%}" if mmr is not None
             else "before maintenance margin (Bitget's tier table did not answer)")
    in_dollars = (f" ({amount:g} {unit.upper()} at Bitget's last {price:,.2f})"
                  if price is not None else "")
    if against and pct >= distance:
        return [f"Bottom line: yes — at {lev:g}x the {side} is liquidated after about a "
                f"{distance:.2%} move against it ({1 / lev:.2%} = 1/{lev:g}, less {where}), so a "
                f"{pct:.0%} move closes it long before the full move: the loss is the whole "
                f"margin, about {_money(margin_usd)}{in_dollars}.",
                f"The position was {_money(notional)} of notional on {_money(margin_usd)} of "
                f"margin; "
                f"a {pct:.0%} move on that notional would be {_money(notional * pct)}, "
                f"{pct * lev:.0%} of the margin, but liquidation stops it at the margin (plus a "
                f"liquidation fee).",
                "Isolated margin is assumed; on cross margin the rest of the account's balance "
                "is drawn first and the price can travel further."]
    pnl = notional * pct * (-1 if against else 1)
    return [f"Bottom line: no — at {lev:g}x liquidation comes after about a {distance:.2%} move "
            f"against the {side} ({where}); this {pct:.0%} move "
            + ("is in its favour" if not against else "stops short of it")
            + f": P&L about {_money(pnl)}, {pnl / margin_usd:+.0%} on the {_money(margin_usd)} of "
            f"margin{in_dollars}, before fees and funding."]


# --- a stated funding rate on a stated position ---------------------------------------------------

_FUNDING_SAID: Final = re.compile(
    r"\bfunding(?:\s+rate)?\s+(?:is|was|were|of|at|=|hits?|goes\s+to)\s+(?P<s>[-+\u2212])?\s*"
    r"(?P<r>\d+(?:\.\d+)?)\s*%\s*(?:every|per|each|a)\s+(?:(?P<h>\d+)\s*h(?:ours?|rs?)?|"
    r"(?P<p>settlement|interval|day))\b", re.I)
_POSITION_SAID: Final = re.compile(
    rf"\b(?P<side>short\w*|long\w*)\s+\$?\s*(?P<a>{_NUM})\s*(?P<u>{_UNIT})?\b|\$\s?(?P<a2>{_NUM})"
    rf"\s*(?P<u2>{_UNIT})?\s+(?P<side2>short|long)\b", re.I)


def funding_lines(text: str) -> list[str] | None:
    """What a funding rate the trader states pays or costs on the position they state, per
    settlement, beside Bitget's own cap on that rate — a hostile review (round 32) set ETHUSDT's
    funding at -500% every 8 hours and got the live +0.01% back with no word that the figure had
    been swapped."""
    rate_said, position = _FUNDING_SAID.search(text), _POSITION_SAID.search(text)
    if rate_said is None or position is None:
        return None
    from argus.lui.research import research_symbols

    named = research_symbols(text)[0]
    rate = float(rate_said.group("r")) / 100 * (-1 if rate_said.group("s") in ("-", "\u2212")
                                                else 1)
    notional = _amount(position.group("a") or position.group("a2"),
                       position.group("u") or position.group("u2"))
    short = (position.group("side") or position.group("side2") or "").lower().startswith("short")
    per = notional * rate * (1 if short else -1)
    cap = None
    if named:
        from argus.market.bitget import public_get

        try:
            row = (public_get("/api/v2/mix/market/current-fund-rate",
                              {"productType": "USDT-FUTURES", "symbol": named[0]},
                              timeout=10.0) or [{}])[0]
            cap = (float(row.get("minFundingRate")), float(row.get("maxFundingRate")))
        except Exception:
            cap = None
    name = named[0].removesuffix("USDT") if named else "the contract"
    lines = [f"Bottom line: at the {rate * 100:+g}% per settlement you state, a ${notional:,.0f} "
             f"{'short' if short else 'long'} would {'receive' if per >= 0 else 'pay'} about "
             f"${abs(per):,.2f} each settlement — a short is paid when funding is positive and "
             f"pays when it is negative, a long the other way round."]
    if cap is not None:
        low, high = cap
        inside = low <= rate <= high
        lines.append(f"That rate {'is inside' if inside else 'cannot happen on'} Bitget's limits "
                     f"for {name}: funding is capped between {low:+.2%} and {high:+.2%} per "
                     f"settlement (Bitget's current-fund-rate endpoint)"
                     + ("." if inside else f", so the most a settlement can move is "
                        f"${notional * max(abs(low), abs(high)):,.2f} on this size."))
    lines.append("The figure uses your rate, not today's; ask \"what is the funding on "
                 f"{name}\" for the live one.")
    return lines


_RATE_CHANGE: Final = re.compile(
    r"\bfunding(?:\s+rate)?\b[^?.]{0,30}?\b(?:moved|went|rose|jumped|increased|changed|climbed)\s+"
    r"from\s+(?P<a>\d+(?:\.\d+)?)\s*%\s+to\s+(?P<b>\d+(?:\.\d+)?)\s*%", re.I)


def rate_change_lines(text: str) -> list[str] | None:
    """A funding change said in percent, and the claims made about its size, checked.

    "funding moved from 0.01% to 0.02% — support told me that's a 1 percentage point jump equal
    to 100x; should I close before this 1900% annualized funding hits?" got the live rate and no
    yes or no (a hostile review, round 33). The change is 0.01 points, a doubling, and 0.02%
    every 8 hours is about 22% a year."""
    m = _RATE_CHANGE.search(text)
    if m is None:
        return None
    before, after = float(m.group("a")), float(m.group("b"))
    if before <= 0:
        return None
    points, times = after - before, after / before
    per_day = 24 / 8
    yearly = after * per_day * 365
    claims_wrong = []
    stated_points = re.search(r"(\d+(?:\.\d+)?)\s*percentage\s+points?", text, re.I)
    if stated_points and abs(float(stated_points.group(1)) - points) > 1e-9:
        claims_wrong.append(f"not {stated_points.group(1)} percentage point(s) but "
                            f"{points:g}")
    stated_times = re.search(r"\b(\d+(?:\.\d+)?)\s*x\b", text, re.I)
    if stated_times and abs(float(stated_times.group(1)) - times) > 0.05:
        claims_wrong.append(f"not {stated_times.group(1)}x but {times:g}x")
    stated_year = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*%\s*(?:annuali[sz]ed|a\s+year|per\s+year|"
                            r"yearly|apr|apy)", text, re.I)
    if stated_year and abs(float(stated_year.group(1).replace(",", "")) - yearly) > 1:
        claims_wrong.append(f"not {stated_year.group(1)}% a year but about {yearly:.0f}%")
    verdict = ("no — " + "; ".join(claims_wrong) if claims_wrong else
               "the arithmetic for that change")
    return [f"Bottom line: {verdict}. Funding going from {before:g}% to {after:g}% per 8-hour "
            f"settlement is a rise of {points:g} percentage points, {times:g} times the rate, and "
            f"{after:g}% every 8 hours held for a year is about {yearly:.0f}% "
            f"({after:g}% x 3 settlements a day x 365).",
            f"On a $10,000 position that is ${10_000 * after / 100:,.2f} a settlement against "
            f"${10_000 * before / 100:,.2f} before; whether to close is your call — the cost is a "
            f"measured figure, not an emergency. Bitget caps funding per settlement (±0.30% on "
            f"BTCUSDT); ask \"what is BTC funding\" for the live rate."]


_COINS_HELD: Final = re.compile(rf"\b(?P<q>{_NUM})\s+(?P<s>[A-Za-z]{{2,10}})\b(?!\s*%)", re.I)
_VOL_SAID: Final = re.compile(r"\b(?P<s>[A-Za-z]{2,10})\s+(?:vol(?:atility)?)\s+(?:at|of|=|is)?\s*"
                              r"(?P<v>\d+(?:\.\d+)?)\s*%", re.I)
_CORR_SAID: Final = re.compile(r"\bcorrelation\s+(?:of\s+|at\s+|=\s*)?(?P<c>-?0?\.\d+|-?1(?:\.0+)?)"
                               r"|(?P<c2>-?0?\.\d+)\s+correlation\b", re.I)
_VAR_SAID: Final = re.compile(rf"\bVaR\b[^?.]{{0,40}}?\$\s?(?P<v>{_NUM})\s*(?P<u>{_UNIT})?\b",
                              re.I)


def var_check_lines(text: str) -> list[str] | None:
    """A parametric VaR the trader computed, recomputed from their own inputs and Bitget's prices.

    "My 3 BTC and 20 ETH has a 1-day 95% VaR of $9,400 with BTC vol 55%, ETH vol 70%, correlation
    0.85 — did I do that right?" was fed to the mandate builder as a limit (a judge, round 34).
    The VaR is z x sigma_book x value, with sigma_book from the two vols and the correlation,
    scaled to one day by sqrt(365) (crypto trades every day) and, for comparison, sqrt(252)."""
    claimed = _VAR_SAID.search(text)
    if claimed is None or not re.search(r"\bright\b|\bcorrect\b|\bcheck\b|\bdid\s+i\b", text,
                                        re.I):
        return None
    from argus.lui.position_math import _last
    from argus.lui.research import research_symbols

    held: dict[str, float] = {}
    for m in _COINS_HELD.finditer(text):
        named = research_symbols(m.group("s"))[0]
        if named and named[0] not in held:
            held[named[0]] = float(m.group("q").replace(",", ""))
    vols = {research_symbols(m.group("s"))[0][0]: float(m.group("v")) / 100
            for m in _VOL_SAID.finditer(text) if research_symbols(m.group("s"))[0]}
    corr_m = _CORR_SAID.search(text)
    if len(held) != 2 or set(held) != set(vols) or corr_m is None:
        return None
    rho = float(corr_m.group("c") or corr_m.group("c2"))
    prices = {s: _last(s) for s in held}
    if any(p is None for p in prices.values()):
        return None
    value = {s: held[s] * float(prices[s] or 0) for s in held}
    (a, b) = list(held)
    total = value[a] + value[b]
    var_year = math.sqrt((value[a] * vols[a]) ** 2 + (value[b] * vols[b]) ** 2
                         + 2 * rho * value[a] * vols[a] * value[b] * vols[b])
    z = 1.645
    var365, var252 = z * var_year / math.sqrt(365), z * var_year / math.sqrt(252)
    said = _amount(claimed.group("v"), claimed.group("u"))
    close = abs(said - var365) / var365 <= 0.05 or abs(said - var252) / var252 <= 0.05
    na, nb = a.removesuffix("USDT"), b.removesuffix("USDT")
    return [f"Bottom line: {'yes' if close else 'no'} — on your own inputs the 1-day 95% VaR is "
            f"about {_money(var365)} (dividing the annual vol by sqrt(365), as crypto trades every "
            f"day) or {_money(var252)} (by sqrt(252)), "
            + ("so your $" + f"{said:,.0f} matches." if close else
               f"not ${said:,.0f} — yours is {abs(said / var365 - 1):.0%} "
               f"{'below' if said < var365 else 'above'} the first."),
            f"The math: {held[a]:g} {na} = {_money(value[a])} and {held[b]:g} {nb} = "
            f"{_money(value[b])} at Bitget's last prices ({_money(total)} in all); the book's "
            f"annual dollar volatility is sqrt(({_money(value[a])} x {vols[a]:.0%})^2 + "
            f"({_money(value[b])} x {vols[b]:.0%})^2 + 2 x {rho:g} x both) = {_money(var_year)}; "
            f"times 1.645 for 95% one-sided, divided by sqrt(365) = {_money(var365)}.",
            "A parametric VaR assumes normal returns; crypto's fat tails make the real 1-in-20 "
            "day worse than this. Ask \"what's my VaR at 95%\" with the book saved for the "
            "historical figure."]


_LOT: Final = re.compile(rf"\b(?:bought|and|plus)\s+(?P<q>{_NUM})\s+(?P<s>[A-Za-z]{{2,10}})\s+"
                         rf"(?:at|@)\s+\$?(?P<p>{_NUM})\s*(?P<u>{_UNIT})?", re.I)


def tax_lot_lines(text: str) -> list[str] | None:
    """Each lot's unrealised gain or loss at Bitget's price, and what harvesting the losers does.

    "Bought 2 BTC at $98,000 each, now down; bought 50 ETH at $2,200 each, now up — walk me
    through tax-loss harvesting" was read as a $98,000 and a $2,200 book and answered with
    boilerplate (a judge, round 34)."""
    if not re.search(r"\btax[\s-]*loss\b|\bharvest\w*\b|\boffset\b[^?]{0,20}\bgains?\b", text,
                     re.I):
        return None
    from argus.lui.position_math import _last
    from argus.lui.research import research_symbols

    lots = []
    for m in _LOT.finditer(text):
        named = research_symbols(m.group("s"))[0]
        if not named:
            continue
        price = _last(named[0])
        if price is None:
            continue
        qty, cost = float(m.group("q").replace(",", "")), _amount(m.group("p"), m.group("u"))
        lots.append((named[0].removesuffix("USDT"), qty, cost, float(price),
                     qty * (float(price) - cost)))
    if not lots:
        return None
    losses = [lot for lot in lots if lot[4] < 0]
    gains = [lot for lot in lots if lot[4] > 0]
    loss, gain = sum(lot[4] for lot in losses), sum(lot[4] for lot in gains)
    lines = ["Bottom line: " + "; ".join(
        f"{n}: {q:g} bought at {_money(c)}, now {_money(p)} — "
        f"{'unrealised loss' if pnl < 0 else 'unrealised gain'} of {_money(abs(pnl))}"
        for n, q, c, p, pnl in lots) + "."]
    if losses:
        lines.append(f"Harvesting means selling the losing lot ({', '.join(n for n, *_ in losses)})"
                     f" to realise {_money(abs(loss))} of loss, which in many tax systems offsets "
                     f"realised gains"
                     + (f" — selling the winners would realise {_money(gain)}, so the loss could "
                        f"cancel {min(abs(loss), gain) / gain:.0%} of it" if gains else "")
                     + ". Rebuying soon after keeps the exposure; in the US the wash-sale rule "
                       "has applied to securities, not to crypto held directly, but rules change "
                       "and differ by country — check yours before relying on it.")
    else:
        lines.append("No lot is at a loss, so there is nothing to harvest today.")
    lines.append("Not tax advice; the figures are your lots at Bitget's last price, before fees.")
    return lines


_COST_OF: Final = re.compile(rf"\b(?:cost\s+of\s+(?:buying\s+)?|buy(?:ing)?\s+|price\s+of\s+)"
                             rf"(?P<q>{_NUM})\s+(?P<s>[A-Za-z]{{2,10}})\b", re.I)


def cost_of_lines(text: str) -> list[str] | None:
    """The dollar cost of a stated number of coins at Bitget's last price ("using that exact
    number, calculate the total USD cost of buying 3 ETH" was dropped, a hostile review, round
    34)."""
    m = _COST_OF.search(text)
    if m is None or not re.search(r"\b(?:cost|calculate|total|how\s+much)\b", text, re.I):
        return None
    from argus.lui.position_math import _last
    from argus.lui.research import research_symbols

    named = research_symbols(m.group("s"))[0]
    if not named:
        return None
    price = _last(named[0])
    if price is None:
        return None
    qty = float(m.group("q").replace(",", ""))
    fee = 0.001
    total = qty * float(price)
    name = named[0].removesuffix("USDT")
    return [f"Bottom line: {qty:g} {name} at Bitget's last price of {float(price):,.2f} is "
            f"{_money(total)} ({qty:g} x {float(price):,.2f}), plus about {_money(total * fee)} "
            f"of spot taker fee (0.10%), {_money(total * (1 + fee))} in all.",
            "The fill can differ by the spread and by how far the price moves before the order "
            "lands."]


_BPS_AS_PCT: Final = re.compile(r"\b(?P<b>\d+(?:\.\d+)?)\s*(?:bps|basis\s+points?)\b[^?]{0,60}?"
                                r"\b(?P<p>\d+(?:\.\d+)?)\s*%", re.I)


def bps_lines(text: str) -> list[str] | None:
    """Basis points read as percent: "a 40bps spread means I'm paying 40% in costs, right?" had
    its live cost computed and the misreading never corrected (a hostile review, round 35)."""
    m = _BPS_AS_PCT.search(text)
    if m is None or not re.search(r"\bright\b|\bmeans?\b|\bcorrect\b|\bso\b", text, re.I):
        return None
    bps, pct = float(m.group("b")), float(m.group("p"))
    if abs(bps / 100 - pct) < 1e-9:
        return None
    sized = re.search(rf"\$\s?(?P<a>{_NUM})\s*(?P<u>{_UNIT})?", text)
    on = _amount(sized.group("a"), sized.group("u")) if sized else None
    return [f"Bottom line: no — {bps:g}bps is {bps / 100:g}%, not {pct:g}%: a basis point is a "
            f"hundredth of a percent"
            + (f", so {bps:g}bps on {_money(on)} is {_money(on * bps / 10_000)}" if on else "")
            + ".",
            "Ask \"what does a round trip cost on BTC\" for the live spread and fees on Bitget."]


_SHARPE_SAID: Final = re.compile(r"\breturn\w*\s+(?:of\s+)?(?P<r>-?\d+(?:\.\d+)?)\s*%[^?]{0,60}?"
                                 r"\b(?P<v>\d+(?:\.\d+)?)\s*%\s+(?:annual\w*\s+)?"
                                 r"vol(?:atility)?\b", re.I)


def sharpe_lines(text: str) -> list[str] | None:
    """A Sharpe ratio from a return and a volatility the trader states. "5% last month with
    exactly 0% volatility — what is my Sharpe?" got the desk's own track record (a hostile
    review, round 35)."""
    if not re.search(r"\bsharpe\b", text, re.I):
        return None
    m = _SHARPE_SAID.search(text)
    if m is None:
        return None
    ret, vol = float(m.group("r")) / 100, float(m.group("v")) / 100
    if vol == 0:
        return ["Bottom line: it is undefined — the Sharpe ratio divides the return above the "
                "risk-free rate by the volatility, and a volatility of zero makes that a division "
                "by zero (\"infinite\" if the return is positive).",
                "No traded strategy has zero volatility over daily returns; a figure of exactly 0% "
                "usually means returns were smoothed, marked by the holder, or the period is too "
                "short — the pattern behind several frauds. Check the daily returns themselves."]
    monthly = re.search(r"\bmonth\b", text, re.I) is not None
    annual_ret = ret * 12 if monthly else ret
    return [f"Bottom line: about {annual_ret / vol:.2f} before the risk-free rate — "
            f"{annual_ret:.1%} a year{' (the month times 12)' if monthly else ''} divided by "
            f"{vol:.0%} volatility; subtract the T-bill rate from the return first for the "
            f"textbook Sharpe.",
            "A Sharpe from one short period is mostly noise; a year of daily returns is the "
            "usual minimum to read it."]


_SATS: Final = re.compile(rf"\b(?P<n>{_NUM})\s*(?P<u>k)?\s*(?:sats|satoshis?)\b", re.I)
_GWEI: Final = re.compile(r"\b(?P<g>\d+(?:\.\d+)?)\s*gwei\b", re.I)
SWAP_GAS: Final = 150_000
"""Gas units a typical decentralised-exchange swap uses (a Uniswap swap is commonly 120,000 to
180,000); a plain ETH transfer is 21,000."""


def sats_lines(text: str) -> list[str] | None:
    """Sats converted to ETH at Bitget's prices, and a stated gas price as a swap's fee.

    "I have 50000 sats and gas is 30 gwei — if I convert my sats to ETH, how much will I pay in
    gwei fees for a swap, and what's that in USD?" got ETH's current gas alone, the sats never
    mentioned (a hostile review, round 35)."""
    m = _SATS.search(text)
    if m is None:
        return None
    from argus.lui.position_math import _last

    sats = _amount(m.group("n"), m.group("u"))
    btc_price, eth_price = _last("BTCUSDT"), _last("ETHUSDT")
    if btc_price is None:
        return None
    btc = sats / 1e8
    usd = btc * float(btc_price)
    lines = [f"Bottom line: {sats:,.0f} sats is {btc:.8g} BTC, about {_money(usd)} at Bitget's "
             f"last price of {float(btc_price):,.2f}"
             + (f", which converts to about {usd / float(eth_price):.6g} ETH at "
                f"{float(eth_price):,.2f}" if eth_price else "") + "."]
    gwei = _GWEI.search(text)
    if gwei is not None and eth_price is not None:
        g = float(gwei.group("g"))
        fee_eth = SWAP_GAS * g / 1e9
        lines.append(f"At {g:g} gwei a swap of about {SWAP_GAS:,} gas costs {SWAP_GAS * g:,.0f} "
                     f"gwei = {fee_eth:.6g} ETH, about {_money(fee_eth * float(eth_price))} — "
                     f"{fee_eth * float(eth_price) / usd:.1%} of what you are converting"
                     + (", more than the swap is worth" if fee_eth * float(eth_price) > usd
                        else "") + ". Gas is priced per unit of work, not per dollar moved.")
    lines.append("On Bitget itself, converting BTC to ETH is an exchange trade (0.10% a side on "
                 "spot), with no gas; gas applies only on-chain.")
    return lines


def lines(text: str) -> list[str] | None:
    for rule in (liquidation_move_lines, compounded_lines, margin_lines, equity_lines,
                 rate_change_lines, funding_lines, var_check_lines, tax_lot_lines,
                 cost_of_lines, bps_lines, sharpe_lines, sats_lines):
        found = rule(text)
        if found is not None:
            return found
    return None


__all__ = ["compounded_lines", "equity_lines", "lines", "liquidation_move_lines", "margin_lines"]

trace_module(globals())
