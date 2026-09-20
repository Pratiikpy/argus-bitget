# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/QuantConnect/Lean
# Path:    Algorithm.Framework/Alphas/PearsonCorrelationPairsTradingAlphaModel.py, lines 62-69
#          (the correlation-ranking core of `on_securities_changed`, wrapped in a function
#          definition this vendoring adds so the excerpt is independently callable — the excerpt
#          itself has no `self.` references at all, only `df`/`stop`/`pearsonr`, which the wrapper
#          below supplies as real parameters instead of instance attributes)
# Commit:  23b735d99a357807dc0df9f4c51d30f05fe0d277 (2026-09-04)
# Licence: Apache License 2.0 (Copyright 2014 QuantConnect Corporation) — full text below.
#
# This is Lean's real pairs-selection algorithm: rank every candidate pair by the real
# `scipy.stats.pearsonr` correlation of their price changes, and (in the real file's own next two
# lines, deliberately NOT vendored — trivial application logic, not a computation worth vendoring)
# pick the single highest-correlation pair above a fixed threshold. `argus.eval.cointegration_
# comparison` runs this real, unmodified ranking function on constructed price data and confirms,
# by construction, that it has no equivalent of ARGUS's own real `benjamini_hochberg`/`bonferroni`
# multiple-testing correction (`backtest/validation.py`) — a correlation ranking with a fixed
# threshold and no correction for how many candidate pairs were compared is exactly the setup that
# produces a predictable, computable rate of spurious selections, which this comparison measures.
#
# The `def rank_pairs_by_correlation(df, stop, pearsonr):` line below the marker, and the
# `return corr` line at the very end, are this vendoring's own wrapper (not present in the
# original file, which reaches this logic via `self` from inside a class method) — added only to
# make the excerpt independently callable; every line between them is the real file's own,
# unedited.
#
# Apache License 2.0
#
# Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
def rank_pairs_by_correlation(df, stop, pearsonr):
    # ============================== VENDORED FROM HERE ==============================
            corr = dict()

            for i in range(0, stop):
                for j in range(i+1, stop):
                    if (j, i) not in corr:
                        corr[(i, j)] = pearsonr(df.iloc[:,i], df.iloc[:,j])[0]

            corr = sorted(corr.items(), key = lambda kv: kv[1])
            return corr
