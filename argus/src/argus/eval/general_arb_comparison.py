"""Net executable arbitrage against the general-purpose tool for the same job: an LP solver.

**The function in general terms.** "Is there executable arbitrage, how much, at what size" is
constrained optimisation. Buy on one price ladder, sell on another, pay each venue's fee, and do
not take more than either ladder holds. That is a linear programme (see
`research/executable_arb.py`'s docstring for the formulation). The strongest general-purpose
tools for it are LP solvers, and the strongest open-source LP solver by the published record is
**HiGHS**. It is MIT-licensed, it leads the open-source field on Hans Mittelmann's LP benchmarks,
and it has been SciPy's default ``linprog`` backend since SciPy 1.9. It runs here through
``scipy.optimize.linprog(method="highs")``, the SciPy that is already installed; nothing was
vendored. The second general-purpose rival is the textbook graph formulation of arbitrage:
**networkx**'s Bellman-Ford negative-cycle test (BSD-3) on a graph whose edge weights are
``-log`` of fee-adjusted exchange rates. It can say whether a profitable cycle exists at the
touch. It cannot size one.

**Candidates considered and set aside.** OR-Tools GLOP (Apache-2.0) and CVXPY (`cvxpy/cvxpy`,
Apache-2.0) solve the same LP. CVXPY is a modelling layer over solvers of
this kind, and a second solver on a problem this small adds nothing that HiGHS agreeing with an
independent exact walk does not already show. OR-Tools was not installed: the disk had under
2 GB free at the time. Hummingbot (`hummingbot/hummingbot`, Apache-2.0, commit 2bfaccc) is a
trading specialist rather than a general tool, and it was read, not run. Its
``amm_arb`` charges a taker fee on each side (`amm_arb/data_types.py:76-121`) but sizes at one
fixed ``order_amount`` (`amm_arb/utils.py:18-58`). Its ``spot_perpetual_arbitrage`` computes
``(sell - buy) / buy`` with no fee term (`spot_perpetual_arbitrage/arb_proposal.py:56-64`) and
compares it with a user-set threshold (`spot_perpetual_arbitrage.py:204-207`). Running it needs
Hummingbot's Cython build and a live connector, so no number from it is claimed here.

**The same input.** Two inputs, both the capability's own:

* the constructed two-exchange books `eval/arbitrage_comparison.py` scores ARGUS and maxme on
  (its three designed spreads read from `data/arbitrage_study.json`, its 60 swept spreads under
  the same seed, plus a dense 0-40bps grid), with ARGUS's 12bps round trip split 6bps per leg;
* real, simultaneous Bitget books for every spot rToken (``RNVDAUSDT``, base ``rNVDA``) and the
  same company's USDT-margined perpetual (``NVDAUSDT``). This is the sub-theme's own "rToken
  versus native / cross-platform" spread. Each leg's taker fee is read from the venue:
  ``takerFeeRate`` from `/api/v2/spot/public/symbols` and `/api/v2/mix/market/contracts`.
  `data/general_arb_books.json` is the development capture (2026-09-25 16:15-16:22Z), and
  `data/general_arb_books_holdout.json` a second capture ten minutes later (16:32-16:38Z). No
  system scored here has a parameter fitted on either capture, so the second one repeats the
  measurement on fresh books; it is not a guard against tuning, and is not called one.

**Truth.** For a given snapshot and fee schedule, the maximum net profit of the two-leg trade
is the LP's optimum. HiGHS computes it, and ARGUS's new exact walk
(`research/executable_arb.leg_optimum`) computes it independently. The comparison requires the
two to agree to within float tolerance on every snapshot before either is trusted. ARGUS's
deployed `research/arbitrage_study.decompose()` is then scored against that truth, together
with a four-arm ablation of its constants, networkx, and the existing fee-blind maxme baseline.

The results are in `data/general_arb_comparison.json`, and the scope statement is assembled from
them rather than typed in.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.eval.arbitrage_comparison import _depths_for_spread_bps, real_basis
from argus.eval.artefact import write as write_artefact
from argus.eval.baselines.maxme_arbitrer_loader import load_arbitrer_module
from argus.market.depth import DepthError, Level, OrderBook
from argus.market.history import BasisPoint
from argus.research.arbitrage_study import ROUND_TRIP_BPS, Decomposition, decompose
from argus.research.executable_arb import best_arbitrage, leg_optimum
from argus.truth.clocks import DualClock

DATA = Path(__file__).resolve().parents[3] / "data"
BOOKS_PATH = DATA / "general_arb_books.json"
HOLDOUT_PATH = DATA / "general_arb_books_holdout.json"
ARTEFACT_PATH = DATA / "general_arb_comparison.json"

TOLERANCE = 1e-6
"""Quote currency. Below this a net profit is float noise, not money: HiGHS works in doubles and
its default feasibility tolerance is 1e-7, so an exact zero can come back as -3e-9."""

NOTIONAL_CAP = Decimal("10000")
"""A second run of every snapshot with the buy leg capped at $10,000, fee included. This checks
that the walk stays exact once a budget binds, and not only when the book runs out."""

SKEW_LIMIT_MS = 1000
"""Two legs fetched further apart than this are reported separately. The comparison is still
fair, because every system sees the same pair of books, but the pair is less of an instant."""

_BPS = Decimal("10000")
_PER_LEG_FEE = ROUND_TRIP_BPS / 2 / _BPS
"""ARGUS's 12bps round trip, split across the constructed input's two legs."""


class GeneralArbComparisonError(RuntimeError):
    """A solver failed, or the two exact methods disagreed, so no verdict can be trusted."""


# =================================================================================================
# Capture (network). Everything below it runs on the saved books.
# =================================================================================================


def capture(*, rounds: int = 3, levels: int = 25, workers: int = 3,
            out: Path = BOOKS_PATH) -> dict[str, Any]:  # pragma: no cover - network
    """Snapshot every spot rToken and its same-company perpetual, both legs fetched at once.

    Written after every round, so an interrupted capture keeps what it already has.
    """
    from argus.market.rtoken_spot import Pair, SpotError, _get, resolve_pairs

    pairs = resolve_pairs()
    spot_fees = {str(s["symbol"]): str(s.get("takerFeeRate"))
                 for s in _get("/api/v2/spot/public/symbols")}
    perp_fees = {str(c["symbol"]): str(c.get("takerFeeRate"))
                 for c in _get("/api/v2/mix/market/contracts?productType=USDT-FUTURES")}
    started = datetime.now(UTC).isoformat()
    snaps: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def book(category: str, symbol: str) -> Any:
        return _get(f"/api/v3/market/orderbook?category={category}&symbol={symbol}"
                    f"&limit={levels}")

    def one(args: tuple[int, Pair]) -> dict[str, Any]:
        r, pair = args
        with ThreadPoolExecutor(max_workers=2) as legs:
            fs = legs.submit(book, "SPOT", pair.spot)
            fp = legs.submit(book, "USDT-FUTURES", pair.perp)
            try:
                spot, perp = fs.result(), fp.result()
            except SpotError as exc:
                return {"round": r, "ticker": pair.ticker, "error": str(exc)[:160]}
        return {
            "round": r, "ticker": pair.ticker, "spot": pair.spot, "perp": pair.perp,
            "spot_ts": str(spot.get("ts")), "perp_ts": str(perp.get("ts")),
            "spot_fee": spot_fees.get(pair.spot), "perp_fee": perp_fees.get(pair.perp),
            "spot_bids": spot.get("b") or [], "spot_asks": spot.get("a") or [],
            "perp_bids": perp.get("b") or [], "perp_asks": perp.get("a") or [],
        }

    def dump() -> dict[str, Any]:
        blob = {
            "captured_from": started, "captured_to": datetime.now(UTC).isoformat(),
            "rounds": rounds, "levels": levels, "pairs": len(pairs),
            "endpoints": [
                "GET /api/v3/market/orderbook?category=SPOT",
                "GET /api/v3/market/orderbook?category=USDT-FUTURES",
                "GET /api/v2/spot/public/symbols (takerFeeRate)",
                "GET /api/v2/mix/market/contracts?productType=USDT-FUTURES (takerFeeRate)",
            ],
            "snapshots": snaps, "failures": failures,
        }
        out.write_text(json.dumps(blob, allow_nan=False, separators=(",", ":")),
                       encoding="utf-8")
        return blob

    blob: dict[str, Any] = {}
    for r in range(rounds):
        jobs = [(r, pairs[t]) for t in sorted(pairs)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(one, jobs):
                if "error" in row:
                    failures.append({k: str(v) for k, v in row.items()})
                else:
                    snaps.append(row)
        blob = dump()
    return blob


# =================================================================================================
# Loading the saved books.
# =================================================================================================


def _ladder(rows: Sequence[Sequence[Any]], *, descending: bool) -> tuple[Level, ...]:
    out = []
    for row in rows:
        price, qty = Decimal(str(row[0])), Decimal(str(row[1]))
        if price > 0 and qty > 0:
            out.append(Level(price=price, quantity=qty))
    return tuple(sorted(out, key=lambda level: level.price, reverse=descending))


@dataclass(frozen=True)
class TwoBooks:
    """One real instant: a spot rToken's book and its perpetual's, with each venue's fee."""

    ticker: str
    round_: int
    spot: OrderBook
    perp: OrderBook
    spot_fee: Decimal
    perp_fee: Decimal
    skew_ms: int


@dataclass(frozen=True)
class Loaded:
    path: str
    sha256: str
    captured_from: str
    captured_to: str
    books: tuple[TwoBooks, ...]
    skipped: dict[str, int]
    """Why a snapshot could not be scored, and how many times. Recorded, never dropped silently."""


def load(path: Path = BOOKS_PATH) -> Loaded:
    raw = path.read_bytes()
    blob = json.loads(raw)
    books: list[TwoBooks] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for s in blob["snapshots"]:
        if s.get("spot_fee") in (None, "None") or s.get("perp_fee") in (None, "None"):
            skip("venue did not publish a taker fee for one leg")
            continue
        ts = datetime.fromtimestamp(int(s["perp_ts"]) / 1000, UTC)
        spot_bids = _ladder(s["spot_bids"], descending=True)
        spot_asks = _ladder(s["spot_asks"], descending=False)
        if not spot_bids and not spot_asks:
            skip("spot rToken book empty on both sides: nothing quoted, nothing to trade")
            continue
        try:
            spot = OrderBook(s["spot"], ts, spot_bids, spot_asks)
            perp = OrderBook(s["perp"], ts, _ladder(s["perp_bids"], descending=True),
                             _ladder(s["perp_asks"], descending=False))
        except DepthError as exc:
            skip(f"one-sided or crossed book: {str(exc).split(':')[0][:80]}")
            continue
        books.append(TwoBooks(
            ticker=s["ticker"], round_=int(s["round"]), spot=spot, perp=perp,
            spot_fee=Decimal(s["spot_fee"]), perp_fee=Decimal(s["perp_fee"]),
            skew_ms=int(s["perp_ts"]) - int(s["spot_ts"]),
        ))
    return Loaded(
        path=path.name, sha256=hashlib.sha256(raw).hexdigest(),
        captured_from=blob["captured_from"], captured_to=blob["captured_to"],
        books=tuple(books), skipped=skipped,
    )


# =================================================================================================
# The rivals.
# =================================================================================================


def highs_leg(
    buy_asks: Sequence[Level], sell_bids: Sequence[Level], *, buy_fee: Decimal,
    sell_fee: Decimal, max_quantity: Decimal | None = None,
    max_notional: Decimal | None = None,
) -> tuple[float, float]:
    """The LP in `research/executable_arb.py`'s docstring, solved by HiGHS. Returns (net, qty)."""
    from scipy.optimize import linprog

    if not buy_asks or not sell_bids:
        return 0.0, 0.0
    fb, fs = float(buy_fee), float(sell_fee)
    asks = [(float(lv.price), float(lv.quantity)) for lv in buy_asks]
    bids = [(float(lv.price), float(lv.quantity)) for lv in sell_bids]
    c = [a * (1 + fb) for a, _ in asks] + [-b * (1 - fs) for b, _ in bids]
    a_eq = [[1.0] * len(asks) + [-1.0] * len(bids)]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    if max_quantity is not None:
        a_ub.append([1.0] * len(asks) + [0.0] * len(bids))
        b_ub.append(float(max_quantity))
    if max_notional is not None:
        a_ub.append([a * (1 + fb) for a, _ in asks] + [0.0] * len(bids))
        b_ub.append(float(max_notional))
    bounds = [(0.0, q) for _, q in asks] + [(0.0, q) for _, q in bids]
    res = linprog(
        c, A_ub=a_ub or None, b_ub=b_ub or None, A_eq=a_eq, b_eq=[0.0],
        bounds=bounds, method="highs",
    )
    if res.status != 0 or res.x is None or res.fun is None:
        raise GeneralArbComparisonError(f"HiGHS did not reach an optimum: {res.message}")
    qty = 0.0
    for v in res.x[: len(asks)]:
        qty += float(v)
    # `0.0 - fun` rather than `-fun`: when doing nothing is optimal, HiGHS returns +0.0 and the
    # unary minus would report -0.0, which reads as a loss in the artefact.
    return 0.0 - float(res.fun), qty


@dataclass(frozen=True)
class SolverAnswer:
    net: float
    quantity: float
    direction: str


def highs_best(tb: TwoBooks, *, max_notional: Decimal | None = None) -> SolverAnswer:
    a = highs_leg(tb.spot.asks, tb.perp.bids, buy_fee=tb.spot_fee, sell_fee=tb.perp_fee,
                  max_notional=max_notional)
    b = highs_leg(tb.perp.asks, tb.spot.bids, buy_fee=tb.perp_fee, sell_fee=tb.spot_fee,
                  max_notional=max_notional)
    if a[0] >= b[0]:
        return SolverAnswer(a[0], a[1], "buy_spot_sell_perp")
    return SolverAnswer(b[0], b[1], "buy_perp_sell_spot")


def networkx_cycle(tb: TwoBooks) -> bool:
    """Bellman-Ford on ``-log`` of fee-adjusted rates at the touch.

    Nodes: USDT, the asset held on the spot venue, the asset held on the perpetual venue. Moving
    the asset between venues is a zero-weight edge, which is the fungibility every cross-venue
    arbitrage assumes. A negative cycle means that one round of buying and selling returns more
    USDT than it started with.
    """
    import networkx as nx  # type: ignore[import-untyped]

    g = nx.DiGraph()

    def rate_edge(u: str, v: str, rate: Decimal) -> None:
        g.add_edge(u, v, weight=-math.log(float(rate)))

    for venue, book, fee in (("spot", tb.spot, tb.spot_fee), ("perp", tb.perp, tb.perp_fee)):
        rate_edge("USDT", venue, 1 / (book.asks[0].price * (1 + fee)))
        rate_edge(venue, "USDT", book.bids[0].price * (1 - fee))
    g.add_edge("spot", "perp", weight=0.0)
    g.add_edge("perp", "spot", weight=0.0)
    return bool(nx.negative_edge_cycle(g, weight="weight"))


def maxme_profit(tb: TwoBooks) -> float:
    """maxme/bitcoin-arbitrage's real, fee-blind detector on the same two books, both ways."""
    module = load_arbitrer_module()

    def side(levels: Sequence[Level]) -> list[dict[str, float]]:
        return [{"price": float(lv.price), "amount": float(lv.quantity)} for lv in levels]

    depths = {
        "spot": {"asks": side(tb.spot.asks), "bids": side(tb.spot.bids)},
        "perp": {"asks": side(tb.perp.asks), "bids": side(tb.perp.bids)},
    }
    detector = module.ArbitrerProfitDetector(depths, max_tx_volume=1e12)
    a = detector.arbitrage_depth_opportunity("spot", "perp")[0]
    b = detector.arbitrage_depth_opportunity("perp", "spot")[0]
    return float(max(a, b))


def _basis_point(tb: TwoBooks) -> BasisPoint:
    ts = tb.perp.fetched_at
    return BasisPoint(ts=ts, market=tb.perp.mid, index=tb.spot.mid, premium=Decimal("0"))


def argus_deployed(tb: TwoBooks, clock: DualClock) -> Decomposition:
    """ARGUS's decomposition exactly as `arbitrage_study.study()` runs it: mid-to-mid basis, its
    own defaults, nothing from the book."""
    return decompose(_basis_point(tb), tb.perp.symbol, clock)


ABLATION_ARMS: tuple[tuple[str, bool, bool, Decimal], ...] = (
    ("deployed_constants", False, False, Decimal("2.0")),
    ("venue_fees_only", True, False, Decimal("2.0")),
    ("measured_touch_only", False, True, Decimal("2.0")),
    ("venue_fees_and_measured_touch", True, True, Decimal("2.0")),
    ("venue_fees_measured_touch_no_slippage", True, True, Decimal("0")),
)
"""Each deployed constant replaced by the measured value, one at a time and then together.
``venue fees`` means the two legs' own taker rates (10bps spot + 6bps perpetual on 2026-09-25)
in place of the 12bps perpetual round trip. ``measured touch`` means half of each book's quoted
spread, which is what crossing both legs from the mid costs, in place of the flat 0.6bps."""


def argus_arm(tb: TwoBooks, clock: DualClock, *, venue_fees: bool, measured_touch: bool,
              slippage: Decimal) -> Decomposition:
    point = _basis_point(tb)
    fee = (tb.spot_fee + tb.perp_fee) * _BPS if venue_fees else ROUND_TRIP_BPS
    touch = ((tb.spot.spread_bps + tb.perp.spread_bps) / 2 if measured_touch
             else Decimal("0.6"))
    return Decomposition(
        ts=point.ts, symbol=tb.perp.symbol, phase=clock.phase(point.ts),
        apparent_bps=point.basis_bps, fee_bps=fee, spread_cost_bps=touch,
        slippage_bps=slippage, execution_probability=Decimal("0.92"),
        failed_leg_probability=Decimal("0.05"),
    )


# =================================================================================================
# Scoring.
# =================================================================================================


@dataclass
class Confusion:
    """A verdict against the truth. A false acceptance is a trade that loses money."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, said: bool, truth: bool) -> None:
        if said and truth:
            self.tp += 1
        elif said:
            self.fp += 1
        elif truth:
            self.fn += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict[str, Any]:
        said = self.tp + self.fp
        n = self.tp + self.fp + self.fn + self.tn
        return {
            "n": n, "true_accept": self.tp, "false_accept": self.fp,
            "false_refuse": self.fn, "true_refuse": self.tn,
            "accuracy": round((self.tp + self.tn) / n, 4) if n else None,
            "precision": round(self.tp / said, 4) if said else None,
            "errors": self.fp + self.fn,
        }


@dataclass(frozen=True)
class Row:
    ticker: str
    round_: int
    skew_ms: int
    phase: str
    spot_spread_bps: float
    perp_spread_bps: float
    mid_basis_bps: float
    truth_net: float
    truth_quantity: float
    truth_direction: str
    top_of_book_edge_bps: float
    argus_exact_net: float
    argus_exact_quantity: float
    argus_exact_expected_net: float
    capped_highs_net: float
    capped_argus_net: float
    deployed_monetizable: bool
    deployed_expected_bps: float
    deployed_gross_bps: float
    arms: dict[str, bool]
    networkx_cycle: bool
    maxme_profit: float

    @property
    def truth(self) -> bool:
        return self.truth_net > TOLERANCE


def score(tb: TwoBooks, clock: DualClock) -> Row:
    truth = highs_best(tb)
    capped = highs_best(tb, max_notional=NOTIONAL_CAP)
    exact = best_arbitrage(tb.spot, tb.perp, fee_a=tb.spot_fee, fee_b=tb.perp_fee)
    exact_capped = best_arbitrage(tb.spot, tb.perp, fee_a=tb.spot_fee, fee_b=tb.perp_fee,
                                  max_notional=NOTIONAL_CAP)
    deployed = argus_deployed(tb, clock)
    edges = [e for e in (exact.best.top_of_book_edge_bps, exact.other.top_of_book_edge_bps)
             if e is not None]
    return Row(
        ticker=tb.ticker, round_=tb.round_, skew_ms=tb.skew_ms,
        phase=str(clock.phase(tb.perp.fetched_at)),
        spot_spread_bps=float(tb.spot.spread_bps), perp_spread_bps=float(tb.perp.spread_bps),
        mid_basis_bps=float(_basis_point(tb).basis_bps),
        truth_net=truth.net, truth_quantity=truth.quantity, truth_direction=truth.direction,
        top_of_book_edge_bps=float(max(edges)),
        argus_exact_net=float(exact.best.net), argus_exact_quantity=float(exact.best.quantity),
        argus_exact_expected_net=float(exact.expected_net),
        capped_highs_net=capped.net, capped_argus_net=float(exact_capped.best.net),
        deployed_monetizable=deployed.is_monetizable,
        deployed_expected_bps=float(deployed.expected_executable_bps),
        deployed_gross_bps=float(deployed.gross_after_costs_bps),
        arms={name: argus_arm(tb, clock, venue_fees=vf, measured_touch=mt,
                              slippage=sl).is_monetizable
              for name, vf, mt, sl in ABLATION_ARMS},
        networkx_cycle=networkx_cycle(tb),
        maxme_profit=maxme_profit(tb),
    )


def _q(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def _dist(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "median": _round(_q(values, 0.5)), "p90": _round(_q(values, 0.9)),
        "p99": _round(_q(values, 0.99)), "max": _round(max(values) if values else None),
    }


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def ticker_bootstrap(rows: Sequence[Row], *, draws: int = 2000,
                     seed: int = 20260925) -> dict[str, Any]:
    """95% interval for the deployed decomposition's false-acceptance rate and precision.

    Resampled by ticker, not by snapshot: the three rounds of one ticker are minutes apart and
    far from independent, so resampling snapshots would claim a precision the data does not have.
    """
    by: dict[str, list[Row]] = {}
    for r in rows:
        by.setdefault(r.ticker, []).append(r)
    tickers = sorted(by)
    rng = random.Random(seed)
    far, prec = [], []
    for _ in range(draws):
        sample = [r for t in (rng.choice(tickers) for _ in tickers) for r in by[t]]
        neg = [r for r in sample if not r.truth]
        said = [r for r in sample if r.deployed_monetizable]
        if neg:
            far.append(sum(r.deployed_monetizable for r in neg) / len(neg))
        if said:
            prec.append(sum(r.truth for r in said) / len(said))
    return {
        "clusters": len(tickers), "draws": draws, "seed": seed,
        "false_accept_rate_ci95": [_round(_q(far, 0.025)), _round(_q(far, 0.975))],
        "precision_ci95": [_round(_q(prec, 0.025)), _round(_q(prec, 0.975))],
    }


def evaluate(loaded: Loaded) -> tuple[dict[str, Any], list[Row]]:
    clock = DualClock()
    rows = [score(tb, clock) for tb in loaded.books]

    net_diffs = [abs(r.truth_net - r.argus_exact_net) for r in rows]
    capped_diffs = [abs(r.capped_highs_net - r.capped_argus_net) for r in rows]
    verdict_mismatch = sum(1 for r in rows if (r.argus_exact_net > TOLERANCE) != r.truth)
    if verdict_mismatch or (net_diffs and max(net_diffs) > 1e-4):
        raise GeneralArbComparisonError(
            f"HiGHS and the exact walk disagree on {verdict_mismatch} verdicts, max net diff "
            f"{max(net_diffs) if net_diffs else 0}: neither can serve as truth"
        )

    deployed, exact, nx_c, maxme = Confusion(), Confusion(), Confusion(), Confusion()
    arms = {name: Confusion() for name, *_ in ABLATION_ARMS}
    for r in rows:
        deployed.add(r.deployed_monetizable, r.truth)
        exact.add(r.argus_exact_net > TOLERANCE, r.truth)
        nx_c.add(r.networkx_cycle, r.truth)
        maxme.add(r.maxme_profit > TOLERANCE, r.truth)
        for name, said in r.arms.items():
            arms[name].add(said, r.truth)

    false_accepts = [r for r in rows if r.deployed_monetizable and not r.truth]
    positives = [r for r in rows if r.truth]
    report = {
        "input": {
            "path": f"data/{loaded.path}", "sha256": loaded.sha256,
            "captured_from": loaded.captured_from, "captured_to": loaded.captured_to,
            "scored_snapshots": len(rows), "tickers": len({r.ticker for r in rows}),
            "skipped": loaded.skipped,
            "session_phases": sorted({r.phase for r in rows}),
            "fee_schedules": sorted({f"spot {tb.spot_fee} / perp {tb.perp_fee}"
                                     for tb in loaded.books}),
            "leg_skew_ms": _dist([abs(r.skew_ms) for r in rows]),
            "snapshots_over_skew_limit": sum(1 for r in rows if abs(r.skew_ms) > SKEW_LIMIT_MS),
            "spot_touch_spread_bps": _dist([r.spot_spread_bps for r in rows]),
            "perp_touch_spread_bps": _dist([r.perp_spread_bps for r in rows]),
            "abs_mid_basis_bps": _dist([abs(r.mid_basis_bps) for r in rows]),
            "top_of_book_net_edge_bps": _dist([r.top_of_book_edge_bps for r in rows]),
        },
        "baseline_agreement": {
            "highs_vs_argus_exact_max_abs_diff_net": max(net_diffs) if net_diffs else 0.0,
            "highs_vs_argus_exact_capped_max_abs_diff_net": (
                max(capped_diffs) if capped_diffs else 0.0),
            "notional_cap_usdt": str(NOTIONAL_CAP),
            "verdict_mismatches": verdict_mismatch,
        },
        "comparison": {
            "truth_monetizable": len(positives),
            "argus_deployed_decompose": deployed.as_dict(),
            "argus_exact_walk": exact.as_dict(),
            "networkx_bellman_ford_touch": nx_c.as_dict(),
            "maxme_fee_blind": maxme.as_dict(),
        },
        "deployed_false_accepts": {
            "tickers": sorted({r.ticker for r in false_accepts}),
            "claimed_expected_bps": _dist([r.deployed_expected_bps for r in false_accepts]),
            "true_top_of_book_edge_bps": _dist([r.top_of_book_edge_bps for r in false_accepts]),
            "over_skew_limit": sum(1 for r in false_accepts if abs(r.skew_ms) > SKEW_LIMIT_MS),
        },
        "deployed_magnitude_error_bps": _dist(
            [abs(r.deployed_gross_bps - r.top_of_book_edge_bps) for r in rows]),
        "truth_positives": [
            {"ticker": r.ticker, "round": r.round_, "net_usdt": round(r.truth_net, 6),
             "quantity": round(r.truth_quantity, 6), "direction": r.truth_direction,
             "top_of_book_edge_bps": round(r.top_of_book_edge_bps, 4),
             "argus_exact_expected_net_usdt": round(r.argus_exact_expected_net, 6),
             "deployed_said_monetizable": r.deployed_monetizable,
             "deployed_expected_bps": round(r.deployed_expected_bps, 4),
             "skew_ms": r.skew_ms}
            for r in positives
        ],
        "ablation": {name: c.as_dict() for name, c in arms.items()},
        "ticker_bootstrap": ticker_bootstrap(rows),
        "per_snapshot": [
            {"ticker": r.ticker, "round": r.round_, "phase": r.phase, "skew_ms": r.skew_ms,
             "mid_basis_bps": round(r.mid_basis_bps, 4),
             "top_of_book_edge_bps": round(r.top_of_book_edge_bps, 4),
             "truth_net_usdt": round(r.truth_net, 8), "truth_monetizable": r.truth,
             "argus_deployed": r.deployed_monetizable,
             "argus_exact": r.argus_exact_net > TOLERANCE,
             "networkx_cycle": r.networkx_cycle, "maxme": r.maxme_profit > TOLERANCE}
            for r in rows
        ],
    }
    return report, rows


# =================================================================================================
# The capability's own constructed input.
# =================================================================================================


def _constructed_books(spread_bps: float) -> tuple[tuple[Level, ...], tuple[Level, ...]]:
    depths = _depths_for_spread_bps(spread_bps)
    asks = tuple(Level(Decimal(str(x["price"])), Decimal(str(x["amount"])))
                 for x in depths["ex_a"]["asks"])
    bids = tuple(Level(Decimal(str(x["price"])), Decimal(str(x["amount"])))
                 for x in depths["ex_b"]["bids"])
    return asks, bids


def _constructed_basis(spread_bps: float, ts: datetime) -> BasisPoint:
    """Identical to `arbitrage_comparison._basis_point_for_spread_bps`."""
    index = Decimal("100")
    market = index * (Decimal("1") + Decimal(str(spread_bps)) / _BPS)
    return BasisPoint(ts=ts, market=market, index=index, premium=Decimal("0"))


def constructed_case(spread_bps: float, clock: DualClock) -> dict[str, Any]:
    ts = datetime(2026, 6, 15, 14, 0, tzinfo=UTC)
    asks, bids = _constructed_books(spread_bps)
    net, qty = highs_leg(asks, bids, buy_fee=_PER_LEG_FEE, sell_fee=_PER_LEG_FEE)
    exact = leg_optimum(asks, bids, buy_fee=_PER_LEG_FEE, sell_fee=_PER_LEG_FEE)
    deployed = decompose(_constructed_basis(spread_bps, ts), "TESTUSDT", clock)
    if (net > TOLERANCE) != (exact.net > 0) or abs(net - float(exact.net)) > 1e-6:
        raise GeneralArbComparisonError(f"HiGHS and the walk disagree at {spread_bps}bps")
    return {
        "spread_bps": spread_bps, "truth_net": round(net, 8), "truth_quantity": round(qty, 8),
        "truth_monetizable": net > TOLERANCE,
        "argus_deployed_monetizable": deployed.is_monetizable,
        "argus_deployed_expected_bps": float(round(deployed.expected_executable_bps, 4)),
        "argus_exact_net": float(round(exact.net, 8)),
    }


def swept_spreads(*, n: int = 60, seed: int = 20260916) -> list[float]:
    """The exact spread sequence `arbitrage_comparison.run_swept_cases()` draws."""
    rng = random.Random(seed)
    return [abs(rng.gauss(0.0, 4.5)) for _ in range(n)]


def evaluate_constructed() -> dict[str, Any]:
    clock = DualClock()
    median, p95, max_observed = real_basis()
    designed = [constructed_case(x, clock) for x in (median, p95, max_observed)]
    swept = [constructed_case(x, clock) for x in swept_spreads()]
    grid = [constructed_case(round(0.1 * k, 1), clock) for k in range(401)]

    def confusion(cases: list[dict[str, Any]]) -> dict[str, Any]:
        c = Confusion()
        for case in cases:
            c.add(case["argus_deployed_monetizable"], case["truth_monetizable"])
        return c.as_dict()

    refused = [c["spread_bps"] for c in grid
               if c["truth_monetizable"] and not c["argus_deployed_monetizable"]]
    return {
        "per_leg_fee_bps": str(_PER_LEG_FEE * _BPS),
        "designed_cases": designed,
        "designed_confusion": confusion(designed),
        "swept_confusion": confusion(swept),
        "swept_false_refusals_bps": [
            round(c["spread_bps"], 4) for c in swept
            if c["truth_monetizable"] and not c["argus_deployed_monetizable"]
        ],
        "grid_0_to_40bps_step_0_1": {
            "confusion": confusion(grid),
            "false_refuse_band_bps": [min(refused), max(refused)] if refused else None,
            "truth_threshold_bps": min(c["spread_bps"] for c in grid if c["truth_monetizable"]),
            "deployed_threshold_bps": min(c["spread_bps"] for c in grid
                                          if c["argus_deployed_monetizable"]),
        },
    }


# =================================================================================================
# Adversarial and failure cases.
# =================================================================================================


def _book(symbol: str, bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    ts = datetime(2026, 9, 25, 16, 0, tzinfo=UTC)
    return OrderBook(
        symbol, ts,
        tuple(Level(Decimal(p), Decimal(q)) for p, q in bids),
        tuple(Level(Decimal(p), Decimal(q)) for p, q in asks),
    )


def adversarial_cases() -> list[dict[str, Any]]:
    """Two constructed books aimed at the two ways a flat decomposition goes wrong."""
    clock = DualClock()
    out: list[dict[str, Any]] = []
    fee_s, fee_p = Decimal("0.001"), Decimal("0.0006")
    cases = {
        "wide_spot_touch_behind_a_wide_mid_gap": TwoBooks(
            "WIDE", 0,
            _book("RWIDEUSDT", [("99.70", "50")], [("100.30", "50")]),
            _book("WIDEUSDT", [("100.24", "50")], [("100.26", "50")]),
            fee_s, fee_p, 0),
        "profitable_touch_one_hundredth_of_a_share_deep": TwoBooks(
            "THIN", 0,
            _book("RTHINUSDT", [("99.60", "100")], [("100.00", "0.01"), ("100.40", "100")]),
            _book("THINUSDT", [("100.30", "100")], [("100.32", "100")]),
            fee_s, fee_p, 0),
    }
    for name, tb in cases.items():
        truth = highs_best(tb)
        exact = best_arbitrage(tb.spot, tb.perp, fee_a=tb.spot_fee, fee_b=tb.perp_fee)
        deployed = argus_deployed(tb, clock)
        out.append({
            "name": name,
            "mid_basis_bps": float(round(_basis_point(tb).basis_bps, 4)),
            "truth_net_usdt": round(truth.net, 8), "truth_quantity": round(truth.quantity, 8),
            "argus_deployed_monetizable": deployed.is_monetizable,
            "argus_deployed_expected_bps": float(round(deployed.expected_executable_bps, 4)),
            "argus_exact_net_usdt": float(round(exact.best.net, 8)),
            "argus_exact_quantity": float(exact.best.quantity),
            "networkx_cycle": networkx_cycle(tb),
        })
    return out


def failure_cases() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    def run(system: str, input_: str, fn: Callable[[], object]) -> None:
        try:
            outcome = f"returned {fn()!r}"
        except Exception as exc:  # recorded, not swallowed: the outcome IS the finding
            outcome = f"{type(exc).__name__}: {str(exc)[:140]}"
        out.append({"system": system, "input": input_, "outcome": outcome})

    level = (Level(Decimal("100"), Decimal("1")),)
    run("highs", "empty ask ladder",
        lambda: highs_leg((), level, buy_fee=_PER_LEG_FEE, sell_fee=_PER_LEG_FEE))
    run("argus_exact", "empty ask ladder",
        lambda: leg_optimum((), level, buy_fee=_PER_LEG_FEE, sell_fee=_PER_LEG_FEE).net)
    run("argus_exact", "negative fee",
        lambda: leg_optimum(level, level, buy_fee=Decimal("-0.001"), sell_fee=_PER_LEG_FEE))
    run("argus_exact", "zero notional cap on a profitable pair",
        lambda: leg_optimum(level, (Level(Decimal("101"), Decimal("1")),),
                            buy_fee=_PER_LEG_FEE, sell_fee=_PER_LEG_FEE,
                            max_notional=Decimal("0")).net)
    run("order_book", "spot book with no bids (a real state on Bitget)",
        lambda: _book("RXUSDT", [], [("100", "1")]))
    run("order_book", "crossed book", lambda: _book("RXUSDT", [("101", "1")], [("100", "1")]))
    return out


# =================================================================================================
# Costs and reproducibility.
# =================================================================================================


def measure_costs(books: Sequence[TwoBooks], *, n: int = 40) -> dict[str, float]:
    sample = list(books[:n])
    clock = DualClock()

    def per_call(fn: Callable[[TwoBooks], object]) -> float:
        start = time.perf_counter()
        for tb in sample:
            fn(tb)
        return (time.perf_counter() - start) / max(1, len(sample))

    return {
        "highs_both_directions_seconds": per_call(highs_best),
        "argus_exact_both_directions_seconds": per_call(
            lambda tb: best_arbitrage(tb.spot, tb.perp, fee_a=tb.spot_fee, fee_b=tb.perp_fee)),
        "argus_deployed_decompose_seconds": per_call(lambda tb: argus_deployed(tb, clock)),
        "networkx_bellman_ford_seconds": per_call(networkx_cycle),
        "maxme_both_directions_seconds": per_call(maxme_profit),
        "snapshots_timed": float(len(sample)),
    }


def _digest(blob: Any) -> str:
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


# =================================================================================================
# Entry point.
# =================================================================================================


def scope_statement(dev: dict[str, Any], held: dict[str, Any] | None,
                    constructed: dict[str, Any]) -> str:
    def line(label: str, rep: dict[str, Any]) -> str:
        d = rep["comparison"]["argus_deployed_decompose"]
        e = rep["comparison"]["argus_exact_walk"]
        diff = rep["baseline_agreement"]["highs_vs_argus_exact_max_abs_diff_net"]
        return (
            f"{label}: {rep['input']['scored_snapshots']} two-sided snapshots over "
            f"{rep['input']['tickers']} tickers ({', '.join(rep['input']['session_phases'])}). "
            f"The optimum finds {rep['comparison']['truth_monetizable']} monetizable. ARGUS's "
            f"deployed decompose() accepts {d['true_accept'] + d['false_accept']}, of which "
            f"{d['false_accept']} lose money at the book (precision {d['precision']}), and "
            f"refuses {d['false_refuse']} that pay. The exact walk makes {e['errors']} errors, "
            f"and HiGHS matches it to {diff:.2e} USDT."
        )

    grid = constructed["grid_0_to_40bps_step_0_1"]
    band = grid["false_refuse_band_bps"]
    swept_refused: list[float] = constructed["swept_false_refusals_bps"]
    nets = [p["net_usdt"] for rep in (dev, held) if rep is not None
            for p in rep["truth_positives"]]
    worth = (f"The positives net {min(nets):.2f} to {max(nets):.2f} USDT each"
             if nets else "No snapshot was monetizable")
    parts = [
        "Claimed: on real, simultaneous Bitget spot-rToken and perpetual books, scored with "
        "each venue's own taker fee, the general-purpose LP (HiGHS via SciPy) beats ARGUS's "
        "deployed arbitrage decomposition on the capability's own question, which is whether an "
        "apparent spread survives the cost stack.",
        line("Development capture", dev),
    ]
    if held is not None:
        parts.append(line("Second capture, ten minutes later, on fresh books", held))
    parts += [
        "On the capability's own constructed two-exchange books, the deployed decomposition "
        f"makes {constructed['designed_confusion']['errors']} errors on the three designed "
        f"spreads and {constructed['swept_confusion']['errors']} on the 60 swept ones "
        f"(false refusals at {', '.join(f'{s:.2f}' for s in swept_refused) or 'none'}bps). "
        "A dense 0-40bps grid finds it refusing every spread from "
        f"{band[0] if band else '-'} to {band[1] if band else '-'}bps, where the touch already "
        "clears the fee. The constructed spread is already between executable prices, and "
        "0.6bps of spread crossing plus 2bps of slippage are charged on top. The truth "
        f"threshold is {grid['truth_threshold_bps']}bps, and the deployed threshold is "
        f"{grid['deployed_threshold_bps']}bps. That makes one of the 60 disagreements "
        "`eval/arbitrage_comparison.py` counts against maxme a case where maxme's positive "
        "reading survives the fee and ARGUS's refusal is the error.",
        "Adapted: research/executable_arb.py solves the same LP exactly, with a monotone "
        "ladder walk, per-leg venue fees, an optional notional cap, the optimal size and "
        "the net in USDT, and ARGUS's own execution-probability and failed-leg survival on top. "
        "It ties HiGHS on every snapshot. That is a tie with the general tool, not a win: its "
        "only addition is the survival scaling, which no fill data here can test.",
        f"NOT claimed: that any real snapshot held a trade worth doing. {worth}, before "
        "closing costs. NOT claimed: that one snapshot is an instant. The legs "
        "are fetched concurrently, and the skew is reported. NOT claimed: that closing the "
        "position later, funding while it is held, or queue position were modelled. NOT "
        "claimed: that the deployed decomposition is wrong for what it was built on (the "
        "perpetual against its index, where one leg has no book). Its constants are the "
        "perpetual's, and the ablation shows which of them break on spot rTokens. NOT claimed: "
        "any Hummingbot number, because Hummingbot was read and not run. Spot rTokens whose "
        "book was empty on both sides are excluded and counted under input.skipped.",
    ]
    return "\n\n".join(parts)


def _version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


def main(*, include_holdout: bool = True) -> dict[str, Any]:
    dev, dev_rows = evaluate(load(BOOKS_PATH))
    held: dict[str, Any] | None = None
    if include_holdout and HOLDOUT_PATH.is_file():
        held, _ = evaluate(load(HOLDOUT_PATH))
    constructed = evaluate_constructed()
    books = load(BOOKS_PATH).books
    again, _ = evaluate(load(BOOKS_PATH))
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "general_function": (
            "net executable two-book arbitrage: maximise sell proceeds minus buy cost minus "
            "both venues' taker fees over the visible ladders; a linear programme"
        ),
        "rivals": {
            "highs": f"scipy.optimize.linprog(method='highs'), SciPy {_version('scipy')} / "
                     "HiGHS, MIT",
            "networkx": "networkx.negative_edge_cycle (Bellman-Ford), networkx "
                        f"{_version('networkx')}, BSD-3",
            "maxme": "maxme/bitcoin-arbitrage get_profit_for, vendored, MIT (fee-blind)",
            "hummingbot": "read only, not run (Cython build + live connector); Apache-2.0",
        },
        "real_books": dev,
        "out_of_sample": {"held_out": held} if held is not None else {
            "held_out": None, "note": "no held-out capture on disk"},
        "constructed_input": constructed,
        "adversarial": adversarial_cases(),
        "failure_cases": failure_cases(),
        "costs": measure_costs(books),
        "reproducibility": {
            "real_books_rerun_identical": _digest(dev) == _digest(again),
            "digest": _digest(dev),
            "rows": len(dev_rows),
        },
    }
    report["scope_statement"] = scope_statement(dev, held, constructed)
    return report


def render(report: dict[str, Any]) -> str:
    lines = ["GENERAL-PURPOSE RIVAL: HiGHS LP vs ARGUS net executable arbitrage", ""]
    for label, rep in (("development", report["real_books"]),
                       ("held-out", report["out_of_sample"]["held_out"])):
        if rep is None:
            continue
        c = rep["comparison"]
        lines.append(f"-- {label}: {rep['input']['scored_snapshots']} snapshots, "
                     f"truth monetizable {c['truth_monetizable']}")
        for name in ("argus_deployed_decompose", "argus_exact_walk",
                     "networkx_bellman_ford_touch", "maxme_fee_blind"):
            lines.append(f"   {name:30s} {c[name]}")
        lines.append(f"   ablation {rep['ablation']}")
    g = report["constructed_input"]["grid_0_to_40bps_step_0_1"]
    lines.append(f"constructed grid: {g}")
    lines.append(f"costs: {report['costs']}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI
    result = main()
    print(render(result))
    undefined = write_artefact(ARTEFACT_PATH, result)
    print(f"\nsaved -> {ARTEFACT_PATH}" + (f" (null: {undefined})" if undefined else ""))


__all__ = [
    "ABLATION_ARMS",
    "BOOKS_PATH",
    "HOLDOUT_PATH",
    "Confusion",
    "GeneralArbComparisonError",
    "Loaded",
    "Row",
    "TwoBooks",
    "adversarial_cases",
    "argus_arm",
    "argus_deployed",
    "capture",
    "constructed_case",
    "evaluate",
    "evaluate_constructed",
    "failure_cases",
    "highs_best",
    "highs_leg",
    "load",
    "main",
    "maxme_profit",
    "measure_costs",
    "networkx_cycle",
    "render",
    "scope_statement",
    "swept_spreads",
    "ticker_bootstrap",
]
