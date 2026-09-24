# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/oronimbus/tactical-asset-allocation (PyPI: pytaa)
# Path:    src/pytaa/backtest/positions.py, lines 154-190 (function `vigilant_allocation`)
# Commit:  317ad1c6618def4e1dc0fb9879050c6f9f2f026c (2024-12-21)
# Licence: MIT (Copyright 2023 oronimbus) — full text below, read directly from the repo's own
#          LICENSE file.
#
# This is the real, published Vigilant Asset Allocation (VAA) breadth-rule (Keller & Keuning 2017,
# "Breadth Momentum and the Canary Universe: Defensive Asset Allocation (DAA)", SSRN 2543979): count
# how many assets in `data` carry a negative momentum score, and scale the book toward the highest-
# momentum safe asset in `step`-sized increments per negative count, splitting whatever remains
# equally across the top-`top_k` risk assets by rank. No `self.` references — the function reads
# only its own five parameters (`data`, `risk_assets`, `safe_assets`, `top_k`, `step`), so it is
# vendored as a free function, unlike the Lean/maxme excerpts elsewhere in this directory that
# needed a wrapper to detach them from `self`.
#
# `argus.eval.rotation_comparison` runs this real, unmodified `vigilant_allocation` — fed the real
# momentum-score arithmetic from the SAME file's `Signal.momentum_score()` (also vendored verbatim,
# `pytaa_signal.py` in this directory) — against ARGUS's actual, live cross-asset-class candle
# history (tokenized equities, native crypto majors, commodity tokens, all real symbols on Bitget's
# own public futures book) fetched through `argus.market.history.fetch_range`. It has no concept of
# a transaction cost or a minimum-history guard anywhere in its five parameters or its body —
# `argus.desk.rotation` adds both, and the comparison measures by how much that changes the real
# decision on real data: fed a NaN score for the only safe asset (real, measured on XAUUSDT — see
# `pytaa_signal.py`'s header), this real function does not refuse; it silently drops the safe
# asset's entire intended weight, worst exactly when the breadth count says "flee to safety
# hardest" (three of three risk assets negative: 75% of the book left unallocated).
#
# MIT License
#
# Copyright (c) 2023 oronimbus
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
from typing import List

import numpy as np
import pandas as pd


# ============================== VENDORED FROM HERE ==============================
def vigilant_allocation(
    data: pd.Series,
    risk_assets: List[str],
    safe_assets: List[str],
    top_k: int = 5,
    step: float = 0.25,
) -> pd.DataFrame:
    """Allocate assets based on threshold using scores.

    Used in computing the Vigilant portfolios. The allocation works as follows (using $k=5$):
    Determine the number of assets $n$ with negative $Z$, if $n>4$ allocate 100% in safe asset with
    highest momentum score, if $n=3$ put 75% in safest asset, remaining 25% is split equally in 5
    risk assets with highest momentum, if $n=2$ put 50% in safest asset, 50% split evenly top 5
    risk assets etc.

    Args:
        data (pd.Series): dataframe with signals
        risk_assets (List[str]): list of risky assets
        safe_assets (List[str]): list of safety assets
        top_k (int, optional): rank threshold. Defaults to 5.
        step (float, optional): step in allocation to risk assets given signal. Defaults to 0.25.

    Returns:
        pd.DataFrame: dataframe of weights
    """
    is_neg = sum(np.where(data < 0, 1, 0))
    empty = data * np.nan
    safety = pd.concat([data.loc[safe_assets].rank(ascending=False), empty.loc[risk_assets]])
    safety = safety[~safety.index.duplicated()].sort_index()
    risky = pd.concat([data.loc[risk_assets].rank(ascending=False), empty.loc[safe_assets]])
    risky = risky[~risky.index.duplicated()].sort_index()

    # allocate assets based on number of negative scores
    safe_weights = np.where(safety == 1, min([1, step * is_neg]), 0)
    risk_weights = np.where(risky <= top_k, (1 - min([1, step * is_neg])) / top_k, 0)
    weights = safe_weights + risk_weights
    return pd.DataFrame(weights, index=safety.index).T


