# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/maxme/bitcoin-arbitrage
# Path:    arbitrage/arbitrer.py, lines 52-131 (methods `get_profit_for`, `get_max_depth`,
#          `arbitrage_depth_opportunity` of class `Arbitrer` — the real profit-COMPUTATION core;
#          `arbitrage_opportunity`, the caller that reports this same number to observers, is
#          described but not vendored, since it adds only observer-reporting plumbing around the
#          computation these three methods already decide)
# Commit:  f41684a3226710853096a4e93c3b823f92079abf (2024-10-20)
# Licence: MIT (Copyright (c) 2019 Maxime Biais) — full text below, read directly from LICENSE.
#
# This is the real, complete profit-detection computation a live instance of this project runs on
# every tick (`Arbitrer.tick()`, not vendored, calls `arbitrage_opportunity()`, which calls
# `arbitrage_depth_opportunity()`, which calls `get_profit_for()` in a loop — every step in that
# real chain reachable from here). `get_profit_for()`'s own real return statement,
# `profit = sell_total * w_sellprice - buy_total * w_buyprice`, is the entire profit figure this
# project reports as an "opportunity" — no fee, no spread-crossing cost, no slippage-beyond-depth
# term anywhere in this function or in `arbitrage_depth_opportunity()`'s selection loop
# (`if profit >= 0 and profit >= best_profit`). The only fee-aware code anywhere in this
# repository lives in one OPTIONAL simulator observer
# (`observers/traderbotsim.py::TraderBotSim.__init__(..., fee=0, ...)`, itself defaulting to
# zero) — never in the detection layer this excerpt is. `argus.eval.arbitrage_comparison` runs
# this real, unmodified code on constructed order-book depth and confirms it reports a positive
# "profit" on a spread ARGUS's own real `research/arbitrage_study.py` decomposition — round-trip
# taker fee, quoted-spread crossing, slippage, execution probability, failed-leg survival — would
# correctly refuse, matching the exact class of finding this session already confirmed once for
# RD-Agent's execution surface and once for FinceptTerminal's self-inclusion bias: read the
# competitor's real code, then RUN it, rather than trust the headline it prints.
#
# MIT License
#
# Copyright (c) 2019 Maxime Biais
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
# ============================== VENDORED FROM HERE ==============================
class ArbitrerProfitDetector:
    """This class wrapper (name, `__init__`, and this docstring) is this vendoring's own — it is
    NOT present in the original, where these three methods live on the real `Arbitrer` class
    alongside market/observer plumbing this excerpt does not need. `depths` and `max_tx_volume`
    are real `Arbitrer` instance attributes (`self.depths`, `self.max_tx_volume`,
    `arbitrer.py:16,19`), supplied here as constructor arguments instead. Every method body below
    this point is the real file's own, byte-for-byte, starting at its own `def get_profit_for`."""

    def __init__(self, depths, max_tx_volume):
        self.depths = depths
        self.max_tx_volume = max_tx_volume

    def get_profit_for(self, mi, mj, kask, kbid):
        if self.depths[kask]["asks"][mi]["price"] >= self.depths[kbid]["bids"][mj]["price"]:
            return 0, 0, 0, 0

        max_amount_buy = 0
        for i in range(mi + 1):
            max_amount_buy += self.depths[kask]["asks"][i]["amount"]
        max_amount_sell = 0
        for j in range(mj + 1):
            max_amount_sell += self.depths[kbid]["bids"][j]["amount"]
        max_amount = min(max_amount_buy, max_amount_sell, self.max_tx_volume)

        buy_total = 0
        w_buyprice = 0
        for i in range(mi + 1):
            price = self.depths[kask]["asks"][i]["price"]
            amount = min(max_amount, buy_total + self.depths[kask]["asks"][i]["amount"]) - buy_total
            if amount <= 0:
                break
            buy_total += amount
            if w_buyprice == 0:
                w_buyprice = price
            else:
                w_buyprice = (w_buyprice * (buy_total - amount) + price * amount) / buy_total

        sell_total = 0
        w_sellprice = 0
        for j in range(mj + 1):
            price = self.depths[kbid]["bids"][j]["price"]
            amount = (
                min(max_amount, sell_total + self.depths[kbid]["bids"][j]["amount"]) - sell_total
            )
            if amount < 0:
                break
            sell_total += amount
            if w_sellprice == 0 or sell_total == 0:
                w_sellprice = price
            else:
                w_sellprice = (w_sellprice * (sell_total - amount) + price * amount) / sell_total

        profit = sell_total * w_sellprice - buy_total * w_buyprice
        return profit, sell_total, w_buyprice, w_sellprice

    def get_max_depth(self, kask, kbid):
        i = 0
        if len(self.depths[kbid]["bids"]) != 0 and len(self.depths[kask]["asks"]) != 0:
            while self.depths[kask]["asks"][i]["price"] < self.depths[kbid]["bids"][0]["price"]:
                if i >= len(self.depths[kask]["asks"]) - 1:
                    break
                i += 1
        j = 0
        if len(self.depths[kask]["asks"]) != 0 and len(self.depths[kbid]["bids"]) != 0:
            while self.depths[kask]["asks"][0]["price"] < self.depths[kbid]["bids"][j]["price"]:
                if j >= len(self.depths[kbid]["bids"]) - 1:
                    break
                j += 1
        return i, j

    def arbitrage_depth_opportunity(self, kask, kbid):
        maxi, maxj = self.get_max_depth(kask, kbid)
        best_profit = 0
        best_i, best_j = (0, 0)
        best_w_buyprice, best_w_sellprice = (0, 0)
        best_volume = 0
        for i in range(maxi + 1):
            for j in range(maxj + 1):
                profit, volume, w_buyprice, w_sellprice = self.get_profit_for(i, j, kask, kbid)
                if profit >= 0 and profit >= best_profit:
                    best_profit = profit
                    best_volume = volume
                    best_i, best_j = (i, j)
                    best_w_buyprice, best_w_sellprice = (w_buyprice, w_sellprice)
        return (
            best_profit,
            best_volume,
            self.depths[kask]["asks"][best_i]["price"],
            self.depths[kbid]["bids"][best_j]["price"],
            best_w_buyprice,
            best_w_sellprice,
        )
