"""Vendored, verbatim, unmodified below each marker.

Source:  https://github.com/stefan-jansen/alphalens-reloaded (the maintained fork of the
         original, archived quantopian/alphalens -- both Apache-2.0; this vendoring uses the
         maintained fork's more recent commit)
Commit:  f0a07c22d554e4b4036983cc80320b432714fe7e (2025-06-02)
Licence: Apache License 2.0 (Copyright 2017-2018 Quantopian, Inc.) -- full text already vendored
         in this directory (lean_pairs_ranking.py, quantconnect_sue.py); not repeated here.

Three real functions, from two different files in the same repo, concatenated with a divider --
the same technique already used in this directory for qlib_eval_surface.py and
rdagent_cache_utils.py. factor_information_coefficient() is the real, published Information
Coefficient computation (Spearman rank correlation between a factor's values and forward returns,
the field-standard factor-quality metric); it calls get_forward_returns_columns() internally to
find which of its input DataFrame's columns hold forward-return periods, so that second function
is not optional supporting code -- it runs on every real call to the first. plot_information_table
is the real, DEFAULT significance test the library ships for an IC series -- a plain one-sample
t-test (scipy.stats.ttest_1samp(ic_data, 0)) with no correction anywhere for the serial
correlation a real hourly, cross-sectional IC time series carries. eval/factor_divergence_
comparison.py runs this real, naive test alongside ARGUS's own dependency-aware stationary
bootstrap on the SAME real IC data, rather than assuming the naive test would overclaim --
verified by running both, not by argument.

Apache License 2.0
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
from scipy import stats


# alphalens/performance.py:28-77 -- Copyright 2017 Quantopian, Inc., Apache-2.0 License.
def factor_information_coefficient(factor_data, group_adjust=False, by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group["factor"]
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)].apply(
            lambda x: stats.spearmanr(x, f)[0]
        )
        return _ic

    date_idx = factor_data.index.names.index("date")
    freq = factor_data.index.levels[date_idx].freq

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values("date")]

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data, grouper + ["group"])
    if by_group:
        grouper.append("group")

    ic = factor_data.groupby(grouper, observed=True).apply(src_ic)
    if by_group:
        return ic
    else:
        return ic.asfreq(freq)


# ---

# alphalens/utils.py:916-935 -- Copyright 2018 Quantopian, Inc., Apache-2.0 License.
def get_forward_returns_columns(columns, require_exact_day_multiple=False):
    """
    Utility that detects and returns the columns that are forward returns
    """

    # If exact day multiples are required in the forward return periods,
    # drop all other columns (e.g. drop 3D12h).
    if require_exact_day_multiple:
        pattern = re.compile(r"^(\d+([D]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

        if sum(valid_columns) < len(valid_columns):
            warnings.warn(
                "Skipping return periods that aren't exact multiples" + " of days."
            )
    else:
        pattern = re.compile(r"^(\d+([Dhms]|ms|us|ns]))+$", re.IGNORECASE)
        valid_columns = [(pattern.match(col) is not None) for col in columns]

    return columns[valid_columns]


# ---

# alphalens/plotting.py:180-195 -- Copyright 2017 Quantopian, Inc., Apache-2.0 License.
def plot_information_table(ic_data, return_df=False):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    if return_df:
        return ic_summary_table
    else:
        print("Information Analysis")
        utils.print_table(ic_summary_table.apply(lambda x: x.round(3)).T)
