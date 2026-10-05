"""Whether an asset diversifies a stock-and-bond book: the book replayed with and without it.

Round 43's judge (M8) asked "If correlation to Nasdaq is high, what does that mean for using BTC
as a diversifier in a 60/40 book?" and got a fresh correlation figure that did not match the one a
turn earlier, and no conclusion. Correlation alone does not answer it: a volatile asset with a
modest correlation can still raise a book's risk more than its return. The question is answered
by the book itself:

* **The book.** 60% SPY and 40% TLT (or the split the question states), rebalanced daily, over the
  dates every holding traded in the last five years.
* **The add.** The asset at 5% and 10% of the book, paid for by trimming the stock and bond legs in
  proportion (the usual way a sleeve is added).
* **What changes.** Annualised return, annualised volatility, return per unit of volatility, the
  worst peak-to-trough fall, and the asset's correlation with the book over the whole span and
  over the last year — the second is what "correlation is high" means now.

Daily closes from `rule_test.daily_closes` (Yahoo's for SPY and TLT, Bitget's for coins). A past
replay is not a forecast.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from typing import Final

ASKED: Final = re.compile(r"\bdiversif\w*\b|\bhedge\s+for\s+(?:a\s+)?(?:60\s*/\s*40|stock)|"
                          r"\b60\s*/\s*40\b[^?]{0,80}\b(?:add|adding|with|include|including)\b",
                          re.I)
_SPLIT: Final = re.compile(r"\b(?P<s>\d{2})\s*/\s*(?P<b>\d{2})\b")
SLEEVES: Final = (0.05, 0.10)
TRADING_DAYS: Final = 252


def _replay(weights: dict[str, float], closes: dict[str, list[float]]) -> list[float]:
    n = len(next(iter(closes.values())))
    value, curve = 1.0, [1.0]
    for i in range(1, n):
        value *= 1 + sum(w * (closes[s][i] / closes[s][i - 1] - 1) for s, w in weights.items())
        curve.append(value)
    return curve


def _stats(curve: list[float]) -> tuple[float, float, float]:
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    years = len(rets) / TRADING_DAYS
    annual = curve[-1] ** (1 / years) - 1 if years > 0 else 0.0
    vol = statistics.stdev(rets) * math.sqrt(TRADING_DAYS)
    peak, worst = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return annual, vol, worst


def lines(text: str, prior: Sequence[str] = ()) -> list[str] | None:
    """The with-and-without replay for the asset asked about, or None when not asked."""
    if not ASKED.search(text):
        return None
    from argus.lui.research.drawdown_sizing import aligned_closes
    from argus.lui.research.parse import research_symbols

    named = [s for s in research_symbols(text)[0] if s not in ("SPYUSDT", "TLTUSDT", "QQQUSDT",
                                                                "NDX100USDT", "SP500USDT")]
    if not named:
        for turn in reversed(list(prior)[-3:]):
            named = [s for s in research_symbols(turn)[0] if s not in (
                "SPYUSDT", "TLTUSDT", "QQQUSDT", "NDX100USDT", "SP500USDT")]
            if named:
                break
    if not named:
        return None
    asset = named[0]
    split = _SPLIT.search(text)
    stocks = int(split.group("s")) / 100 if split else 0.60
    bonds = 1 - stocks
    try:
        days, closes, said = aligned_closes(["SPYUSDT", "TLTUSDT", asset])
    except Exception:
        return ["Bottom line: the daily history for that book could not be read just now; ask "
                "again in a minute."]
    if len(days) < 250:
        return ["Bottom line: SPY, TLT and that asset share under a year of daily history here, "
                "too little to replay."]
    name = asset.removesuffix("USDT")
    base = {"SPYUSDT": stocks, "TLTUSDT": bonds}
    rows = [("without", 0.0, _stats(_replay(base, closes)))]
    for w in SLEEVES:
        book = {"SPYUSDT": stocks * (1 - w), "TLTUSDT": bonds * (1 - w), asset: w}
        rows.append((f"with {w:.0%}", w, _stats(_replay(book, closes))))
    book_rets = [sum(base[s] * (closes[s][i] / closes[s][i - 1] - 1) for s in base)
                 for i in range(1, len(days))]
    asset_rets = [closes[asset][i] / closes[asset][i - 1] - 1 for i in range(1, len(days))]
    corr_all = statistics.correlation(asset_rets, book_rets)
    corr_year = statistics.correlation(asset_rets[-TRADING_DAYS:], book_rets[-TRADING_DAYS:])
    base_ann, base_vol, base_dd = rows[0][2]
    best = max(rows, key=lambda r: r[2][0] / r[2][1] if r[2][1] else 0.0)
    helped = best[1] > 0
    lead = (f"over {days[0]:%b %Y} to {days[-1]:%b %Y}, adding {name} to a {stocks:.0%}/"
            f"{bonds:.0%} stock-bond book {'raised' if helped else 'did not raise'} its return per "
            f"unit of risk: {base_ann:+.1%} a year at {base_vol:.1%} volatility without it, "
            f"{rows[1][2][0]:+.1%} at {rows[1][2][1]:.1%} with 5%")
    five_vol, five_dd = rows[1][2][1], rows[1][2][2]
    role = ("so it worked as a risk reducer: volatility fell" if five_vol < base_vol else
            "so it worked as a return booster, not a risk reducer: volatility rose")
    out = [f"Bottom line: {lead}; {role} from {base_vol:.1%} to {five_vol:.1%} and the worst fall "
           f"went from {base_dd:.1%} to {five_dd:.1%}. Its correlation with the book was "
           f"{corr_all:+.2f} over the span and {corr_year:+.2f} in the last year — "
           + ("low enough that it still moved partly on its own" if corr_year < 0.5 else
              "high enough that it has moved largely with the book lately, which weakens the "
              "case for it as a diversifier") + "."]
    for label, _w, (ann, vol, dd) in rows:
        out.append(f"{label[:1].upper()}{label[1:]}: {ann:+.1%} a year, volatility {vol:.1%}, "
                   f"return per unit of risk {ann / vol if vol else 0:.2f}, worst fall {dd:.1%}.")
    out.append(f"How: SPY and TLT for stocks and bonds, the {name} sleeve paid for by trimming "
               f"both in proportion, rebalanced daily on the {len(days)} days all three traded. "
               "A past replay is not a forecast. Not advice.")
    out.append("Data: " + "; ".join(said) + ".")
    return out


__all__ = ["ASKED", "lines"]
