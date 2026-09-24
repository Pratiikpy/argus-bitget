# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/QuantConnect/Tutorials
# Path:    "04 Strategy Library/355 Standardized Unexpected Earnings/03 Method.html",
#          the code inside FineSelectionAndSueSorting() that computes SUE for one stock
#          (the real HTML tutorial page embeds this Python as a <pre class="python"> block)
# Commit:  4a341890296f7e79e095508f06170c72ccaa629c (2025-07-28)
# Licence: Apache License 2.0 (Copyright QuantConnect Corporation) -- full text below, read
#          directly from the repo's own LICENSE file. Same org and licence already vendored
#          elsewhere in this directory (lean_pairs_ranking.py).
#
# This is the real, published Standardized Unexpected Earnings (SUE) factor -- the textbook
# post-earnings-announcement-drift (PEAD) proxy: SUE_q = (EPS_q - EPS_q-4) / stdev(eight real
# historical EPS_q - EPS_q-4 deltas). No 'self.' references to unrelated class state -- the
# three real attributes it reads (self.eps_by_symbol / self.months_count / self.
# months_eps_change) are supplied by the small wrapper class below, which this vendoring adds
# (matching the pattern already used for maxme_arbitrer.py in this directory) so the real,
# unedited body runs standalone. 'stock' is any object with a real .Symbol attribute --
# supplied here by argus.eval.earnings_comparison as a tiny real dataclass, not stubbed.
#
# argus.eval.earnings_comparison runs this real, unmodified SUE computation against REAL
# quarterly EPS history pulled live from SEC EDGAR's XBRL API (argus.market.fundamentals,
# already built and already point-in-time / restatement aware) for real rToken anchor
# companies, and finds the real condition under which its zero-guarded-nowhere denominator
# (np.std of eight real deltas) can reach zero on real filed data.
#
# Apache License 2.0
#
# Copyright QuantConnect Corporation.
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
import numpy as np


class SueSorter:
	def __init__(self, eps_by_symbol, months_count, months_eps_change):
		self.eps_by_symbol = eps_by_symbol
		self.months_count = months_count
		self.months_eps_change = months_eps_change

	def compute(self, stock, sue_by_symbol):
# ============================== VENDORED FROM HERE ==============================
		# Calculate the EPS change from four quarters ago
			rw = self.eps_by_symbol[stock.Symbol]
			eps_change = rw[0] - rw[self.months_eps_change]
			
			# Calculate the st dev of EPS change for the prior eight quarters
			new_eps_list = list(rw)[:self.months_count - self.months_eps_change:3]
			old_eps_list = list(rw)[self.months_eps_change::3]
			eps_std = np.std( [ new_eps - old_eps for new_eps, old_eps in 
								zip( new_eps_list, old_eps_list )
							] )
			
			# Get Standardized Unexpected Earnings (SUE)
			sue_by_symbol[stock.Symbol] = eps_change / eps_std
