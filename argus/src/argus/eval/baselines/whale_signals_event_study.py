# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/zty05070242/whale-signals
# Path:    src/analysis/event_study.py, lines 90-164 (`compute_hit_rates`) and 367-416
#          (`compute_base_rate`) — the two functions between them are this vendoring's own
#          divider, not the real file's next function (`compute_conditioned_hit_rates`,
#          `print_hit_rate_results`, `print_conditioned_hit_rates`, `walk_forward_by_year`, all
#          real but not needed for the mechanism this vendoring measures, so not included).
# Commit:  6be10a598c9319aa6b64bf517618eac2dd2ec2f5 (2026-09-15)
# Licence: MIT (Copyright 2026 Fred Zheng) — full text below, read directly from the repo's own
#          LICENSE file.
#
# This is the real significance-testing core of whale-signals' "are whales smart money?" event
# study: for each whale exchange-deposit/withdrawal, test whether the forward price hit rate
# beats a FIXED null of 50% via `scipy.stats.binomtest(hits, n, p=0.5)` — and, separately, the
# real function the same file itself defines to compute the ACTUAL empirical base rate for a
# given direction/horizon/condition, never wired into the significance test above it.
#
# `argus.eval.eventdriven_comparison` runs this real, unmodified `compute_hit_rates` on real
# ARGUS-fetched ETH candle history (`argus.market.history.fetch_range("ETHUSDT", ...)`) with
# constructed placebo "whale" event timestamps carrying zero true informational edge by
# construction, and confirms two real, measured, structural gaps neither of which is a bug in
# whale-signals' arithmetic — both functions compute exactly what they claim to:
#
# 1. The real 24h base rate on real recent ETH history is 56.7%, not 50% (real drift, not a
#    constructed number — see `compute_base_rate`, vendored below, run on the same real data).
#    Placebo events with no real edge hit at that same real base rate, and
#    `binomtest(hits, n, p=0.5)` reports them as significant "smart money" purely from market
#    drift the fixed null never accounts for — while `compute_base_rate`, in the SAME real file,
#    already computes the number that would have caught it, and is simply never passed to the
#    significance test above it.
# 2. Overlapping, serially-correlated 24h-forward windows from real, closely-clustered placebo
#    events are scored as independent Bernoulli trials, with no equivalent anywhere in this file
#    of the Kolari-Pynnonen clustering deflation ARGUS's own `research/eventstudy.py` applies to
#    every parametric statistic it reports.
#
# `argus.research.eventstudy` already has no fixed null anywhere (the sign test's own null
# proportion is each event's *own estimation-window* empirical rate, never 0.5) and already
# applies the clustering deflation this file has no equivalent of — this comparison is the first
# time either claim was checked by running real code against real data rather than read off the
# module's own docstring.
#
# MIT License
#
# Copyright (c) 2026 Fred Zheng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [1, 6, 24]

# ============================== VENDORED FROM HERE ==============================
def compute_hit_rates(events_df: pd.DataFrame) -> dict:
    """
    Compute hit rates: how often did the whale's action match price direction?

    - Exchange deposit (sell signal): hit = price went DOWN afterward
    - Exchange withdrawal (buy signal): hit = price went UP afterward
    - DeFi interaction: treated as buy signal (deploying capital)

    A hit rate of 50% = random. Above 50% = smart money. Below 50% = dumb money.

    Uses a binomial test to determine if the hit rate is significantly
    different from 50% (random chance).

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of compute_event_returns().

    Returns
    -------
    dict
        results[category][horizon] = {
            'n': int, 'hits': int, 'hit_rate': float,
            'pvalue': float, 'direction': str
        }
    """
    # Define what counts as a "hit" for each category
    # Deposit = sell signal, so a hit is price going DOWN (return < 0)
    # Withdrawal = buy signal, so a hit is price going UP (return > 0)
    category_directions = {
        "exchange_deposit": "down",      # selling before a drop = smart
        "exchange_withdrawal": "up",     # buying before a rise = smart
        "defi_interaction": "up",        # deploying capital = bullish bet
    }

    results = {}

    for cat, expected_dir in category_directions.items():
        cat_data = events_df[events_df["tx_category"] == cat]

        if len(cat_data) < 30:
            print(f"  Skipping {cat}: only {len(cat_data)} events")
            continue

        results[cat] = {}

        for h in HORIZONS:
            col = f"fwd_return_{h}h"
            returns = cat_data[col].values

            if expected_dir == "down":
                # Hit = price dropped (return < 0)
                hits = int((returns < 0).sum())
            else:
                # Hit = price rose (return > 0)
                hits = int((returns > 0).sum())

            n = len(returns)
            hit_rate = hits / n

            # Binomial test: is hit rate significantly different from 50%?
            # binom_test tests if the number of successes in n trials
            # differs from what we'd expect under p=0.5 (random)
            p_value = stats.binomtest(hits, n, p=0.5).pvalue

            results[cat][h] = {
                "n": n,
                "hits": hits,
                "hit_rate": hit_rate,
                "pvalue": p_value,
                "expected_direction": expected_dir,
                "mean_return": np.mean(returns),
            }

    return results



# ---

def compute_base_rate(
    price_df: pd.DataFrame,
    direction: str,
    horizon: int,
    condition_mask: Optional[pd.Series] = None,
) -> float:
    """
    Compute the base rate: what fraction of ALL hours saw price go in the
    expected direction? This is the benchmark a whale must beat to show edge.

    Parameters
    ----------
    price_df : pd.DataFrame
        Hourly prices with timestamp_utc and close columns.
    direction : str
        'up' or 'down'.
    horizon : int
        Hours forward to measure.
    condition_mask : pd.Series, optional
        Boolean mask (positionally aligned with price_df) to restrict
        to a sentiment regime.

    Returns
    -------
    float
        Fraction of hours where price moved in the expected direction.
    """
    price = price_df.copy()
    price["timestamp_utc"] = pd.to_datetime(price["timestamp_utc"], utc=True)
    price = price.sort_values("timestamp_utc").reset_index(drop=True)

    price_series = price.set_index("timestamp_utc")["close"]
    future = price["timestamp_utc"] + pd.Timedelta(hours=horizon)
    future_price = future.map(price_series)
    fwd_return = (future_price - price["close"]) / price["close"]

    # Apply condition mask by position (numpy array) to avoid index mismatch
    if condition_mask is not None:
        mask_arr = condition_mask.values if hasattr(condition_mask, "values") else condition_mask
        fwd_return = fwd_return[mask_arr]

    valid = fwd_return.dropna()

    if len(valid) == 0:
        return 0.5

    if direction == "up":
        return float((valid > 0).sum() / len(valid))
    else:
        return float((valid < 0).sum() / len(valid))


