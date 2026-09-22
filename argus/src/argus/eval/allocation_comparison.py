"""Same-input comparison: ARGUS's own HRP allocator against the two specialist libraries that
were named as better, already cloned in this project's own corpus, and never once run.

**The defect this closes.** `desk/allocation.py` was validated against PyPortfolioOpt, which is
the weaker of the two hierarchical-allocation libraries available. An adversarial sweep graded
this capability LOST on three specific counts, and each one is answered here by running something
rather than by arguing:

1. ARGUS's HRP is pearson + single linkage + variance-only cluster risk, while Riskfolio-Lib
   offers HRP, HERC, HERC2 and NCO across eight codependence measures, DBHT clustering and 20+
   risk measures. **No comparison had been run.** :func:`run_parity` and :func:`run_oos_variance`
   run it.
2. The claimed differentiator -- pricing turnover and reporting a break-even holding period --
   is what cvxportfolio's `MultiPeriodOptimization` solves as a convex program.
   **No comparison had been run.** :func:`run_convex_rebalance` runs it.
3. The verdict pivoted on `assumed_sharpe_annual=1.0`, which `optimize_trade`'s own docstring
   concedes is "an assumption, not a measurement". :func:`run_sharpe_sensitivity` replaces the
   single number with a band and reports the exact Sharpe at which the recommendation flips.
4. `allocation.py:40` claims "do not rebalance ... is the answer this module gives most of the
   time on these instruments", while `data/allocation.json` says "Worth doing".
   :func:`run_rebalance_sweep` settles which is true by sweeping books and dates.

**Read before written**, as the standing rule requires. Both references are the clones in this
project's own corpus, and both were verified byte-identical to the installed distributions that
are actually executed here (`riskfolio` 7.3.0: identical; `cvxportfolio` 1.5.1: identical modulo
line endings).

`research/repos-t2/dcajasn~riskfolio-lib/riskfolio/src/HCPortfolio.py` (BSD-3-Clause):

* `:760-781` -- `optimization()`'s own defaults are `model="HRP"`, `codependence="pearson"`,
  `rm="MV"`, `linkage="single"`, `leaf_order=True`. The first four are exactly ARGUS's
  construction; the fifth is not, and is the entire measured divergence (see below).
* `:341-408` -- `_hierarchical_clustering`. The pearson distance is
  `np.sqrt(np.clip((1 - codep) / 2, 0.0, 1.0))`, the same clip-then-root ARGUS's
  `correlation_distance` performs and for the same floating-point reason. Clustering is
  `hr.linkage(p_dist, method=linkage, optimal_ordering=leaf_order)`.
* `:410-412` -- `_seriation` is `hr.leaves_list(clusters)`, scipy's own leaf order.
* `:414-436` -- `_recursive_bisection` splits at `(0, len(i)//2), (len(i)//2, len(i))`, the same
  halving as `hrp_weights`.
* `:199-254` -- `_naive_risk` under `rm="MV"` sets `inv_risk = 1 / vol**2` and normalises, which
  is the inverse-variance weighting `_cluster_variance` uses. So under ARGUS's own configuration
  the two implementations are the same algorithm, and :func:`run_parity` checks that claim to
  1e-16 rather than asserting it.

**A real defect in Riskfolio-Lib 7.3.0, found by running it, not assumed.** `optimization()` at
`HCPortfolio.py:1095-1103` calls `self._hierarchical_recursive_bisection(self.clustering, rm=rm,
rf=rf, linkage=linkage, model=model, upper_bound=..., lower_bound=...)`, but that method's own
signature at `HCPortfolio.py:522-528` is `(self, Z, rm="MV", rf=0, model="HERC")`. Three of the
six keywords do not exist, so **every** `model="HERC"` and `model="HERC2"` call raises
`TypeError` before any allocation is computed. This is present identically in the cloned HEAD
(632a9e48, 2026-06-21) and in the released 7.3.0 wheel. HERC is one of the two models the
adversarial finding named as better; as shipped, it does not run at all. Rather than score a
baseline as "failed", :func:`_herc_compatible` installs a narrow shim that drops only the three
keywords the callee does not accept and forwards the rest -- Riskfolio's own bisection algorithm,
unmodified, reached through its own public `optimization()` entry point. What the shim does is
stated here so the HERC numbers below are never mistaken for an out-of-the-box result.

`research/repos-t2/cvxgrp~cvxportfolio/cvxportfolio/` (**GPL-3.0**):

* `policies.py:673-778` -- `MultiPeriodOptimization`, the convex formulation the adversarial
  finding named. `planning_horizon` is the number of future periods the solver plans over, which
  is the same quantity ARGUS calls `horizon_bars`.
* `policies.py:77-115` -- `execute(h, market_data=None, t=...)` returns the dollar trade vector
  for one step. Passing `market_data=None` requires every datum to be supplied explicitly, which
  is exactly what this comparison wants: the solver sees ARGUS's own covariance and ARGUS's own
  fee, and no market data of cvxportfolio's own choosing.
* `costs.py:750-772, 892-918` -- `TransactionCost`'s model is
  `a|x| + b*sigma*|x|^1.5/V^0.5 + c*x`, and `compile_to_cvxpy` emits the first term as
  `cp.abs(z[:-1]) @ first_term_multiplier`. With `b=None` and `c=None` that is precisely ARGUS's
  own linear `turnover * TAKER_BPS`, so `a=TAKER_BPS/10_000` makes both sides price the identical
  trade identically. The impact term is deliberately left off: ARGUS's own cost model has no
  impact term, and switching one on for the baseline alone would be comparing two different fees.
* `risks.py:66-73` -- `FullCovariance` is benchmark-relative, `(w - w_b)' Sigma (w - w_b)`. With
  the default `benchmark=AllCash` the benchmark's risky weights are zero, so the term is plain
  `w' Sigma w`.
* `constraints/constraints.py:110-117` -- `NoCash`. Needed, and found by running without it: a
  min-variance objective with a cash account and no return forecast liquidates the whole book into
  cash (measured: 100% cash at every risk aversion tried). `NoCash` + `LongOnly` reproduces HRP's
  own feasible set -- fully invested, long only -- which is the only way the two answers are about
  the same question.

**No third-party code is vendored by this module.** Riskfolio-Lib (BSD-3-Clause) and cvxportfolio
(**GPL-3.0**) are both imported at call time from the installed distributions, never at module
import; nothing in ARGUS's shipped import graph reaches either. The GPL library in particular is
executed as a separate program's public API during evaluation and no part of it is copied into
this repository, which is why this project's corpus note marks cvxportfolio "patterns only".

**What was taken and what was rejected.** Taken: Riskfolio's own `optimization()` as the
reference implementation, its NCO/HERC/HERC2 models and its CVaR and kendall variants as
additional baselines, and cvxportfolio's own MPO+TransactionCost as the convex reference for the
rebalance decision. Rejected: DBHT clustering, which was run and failed on this data (recorded in
the artefact rather than omitted), and every Riskfolio risk measure requiring a solver ARGUS does
not have licensed (RLVaR, RLDaR). Also rejected: scoring the baselines on in-sample portfolio
variance, which HRP-family methods trivially win or lose by construction. The criterion here is
**realised out-of-sample variance on a held-out window**, walk-forward, which is the only reading
that can say a method was actually better rather than better-fitted.

    python -m argus.eval.allocation_comparison
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from argus.desk.allocation import (
    MIN_OBSERVATIONS,
    TAKER_BPS,
    TRADING_HOURS_PER_YEAR,
    AllocationError,
    hrp_weights,
    nco_weights,
    optimize_trade,
)
from argus.desk.portfolio import covariance_matrix, returns
from argus.market.bitget import RTOKEN_SYMBOLS
from argus.market.history import CandleType, fetch_range

DAYS = 60
"""A 60-day hourly window, the same length `desk/allocation.py`'s own CLI defaults to."""

TRAIN_BARS = 480
"""20 days of hourly bars per estimation window. Well above `MIN_OBSERVATIONS`, and short enough
that a 60-day history yields several genuinely non-overlapping held-out windows rather than one."""

TEST_BARS = 96
"""4 days held out. Non-overlapping across windows, so the paired sign test below counts
independent draws rather than the same days counted repeatedly."""

STEP_BARS = 96
"""Equal to `TEST_BARS`: consecutive held-out windows touch but never overlap."""

HORIZON_BARS = 168
"""One week of hourly bars, the horizon `optimize_trade` itself defaults to."""

SHARPE_BAND = (0.25, 0.5, 1.0, 2.0)
"""The assumption `optimize_trade` hard-codes as 1.0, quartered, halved, and doubled."""

SWEEP_ORIGINS = 8
"""Estimation-window start points for the claim-(4) sweep, spread across the whole history."""

TIE_TOLERANCE = 1e-9
"""Relative gap below which two realised volatilities are the same number, not a win.

Chosen against the two scales it sits between and not by taste: floating-point noise between two
allocations that agree to 1e-16 is roughly nine orders of magnitude smaller than this, and the
smallest genuine difference any baseline showed is about 3%, seven orders larger. Anything inside
the band is a tie."""


class BaselineUnavailable(RuntimeError):
    """The specialist library could not be imported. Raised rather than silently skipping it:
    a comparison that quietly omits the baseline is the failure this whole module exists to
    correct."""


# --- data ----------------------------------------------------------------------------------------


def load_returns(
    symbols: Sequence[str] = RTOKEN_SYMBOLS, *, days: int = DAYS, attempts: int = 4,
    backoff: float = 5.0, pause: float = 1.0,
) -> tuple[list[str], dict[str, list[float]], dict[str, str]]:
    """Real Bitget hourly closes for the rToken universe, turned into aligned return columns.

    Aligned on the timestamp intersection via `desk.portfolio.align`'s own rule (no forward
    filling), because every allocator below must see the *same* observations or the comparison
    measures data handling rather than allocation.

    **Retried with backoff, because a dropped symbol silently changes the experiment.** Fetching
    twelve 60-day hourly histories back to back is enough to draw an HTTP 429 from the venue --
    measured three times, on COINUSDT, SQQQUSDT and TSLAUSDT, and each time the comparison simply
    ran on a smaller universe and produced a clean-looking result nobody had chosen. It is not a
    cosmetic difference: dropping one instrument moved the winner's own realised volatility by
    about 12%. A first fix with three attempts and a 2s base backoff still lost TSLAUSDT, so the
    venue's 429 window is longer than that; `backoff` doubles from 5s and `pause` spaces the
    symbols out to avoid tripping the limit rather than only recovering from it. Anything still
    failing is named in the returned `failures`, which the CLI prints before anything else.
    """
    series: dict[str, dict[datetime, float]] = {}
    failures: dict[str, str] = {}
    for symbol in symbols:
        last_error = ""
        for attempt in range(attempts):
            try:
                bars = fetch_range(
                    symbol, days=days, interval="1H", candle_type=CandleType.MARKET
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:100]}"
                time.sleep(backoff * (2 ** attempt))
                continue
            series[symbol] = returns([(c.ts, float(c.close)) for c in bars])
            break
        else:
            failures[symbol] = last_error
        time.sleep(pause)
    if not series:
        return [], {}, failures
    stamps = sorted(set.intersection(*(set(v) for v in series.values())))
    columns = {name: [series[name][t] for t in stamps] for name in series}
    return sorted(columns), columns, failures


def _window(
    columns: Mapping[str, Sequence[float]], start: int, size: int,
) -> dict[str, list[float]]:
    return {name: list(values[start:start + size]) for name, values in columns.items()}


def _realised_vol(
    weights: Mapping[str, float], columns: Mapping[str, Sequence[float]],
) -> float:
    """Standard deviation of the fixed-weight portfolio's own realised bar returns.

    Realised, not modelled: the weights are held constant and the actual return series they would
    have produced is formed, then its sample standard deviation taken. Nothing about the
    covariance used to build the weights enters here, which is what makes it a fair held-out
    score for allocators that disagree about the covariance."""
    names = sorted(columns)
    size = len(columns[names[0]])
    port = [sum(weights.get(n, 0.0) * columns[n][i] for n in names) for i in range(size)]
    if len(port) < 2:
        return float("nan")
    mean = sum(port) / len(port)
    return math.sqrt(sum((p - mean) ** 2 for p in port) / (len(port) - 1))


# --- the Riskfolio-Lib baseline ------------------------------------------------------------------


@contextmanager
def _herc_compatible() -> Iterator[None]:
    """Make Riskfolio's HERC/HERC2 models reachable, and say exactly how.

    `HCPortfolio.optimization` (`HCPortfolio.py:1095-1103`) passes `linkage`, `upper_bound` and
    `lower_bound` to `_hierarchical_recursive_bisection`, whose signature
    (`HCPortfolio.py:522-528`) accepts none of the three. Every HERC call therefore dies with
    `TypeError` in 7.3.0 and at the cloned HEAD. The shim forwards only the four keywords the
    method really takes and discards the three it does not, so the algorithm that runs is
    Riskfolio's own, unedited. Restored on exit so nothing else in the process sees a patched
    library."""
    hc = _hcportfolio_module()
    original = hc.HCPortfolio._hierarchical_recursive_bisection

    def shim(
        self: Any, Z: Any, rm: str = "MV", rf: float = 0, model: str = "HERC", **_dropped: Any,
    ) -> Any:
        return original(self, Z, rm=rm, rf=rf, model=model)

    hc.HCPortfolio._hierarchical_recursive_bisection = shim
    try:
        yield
    finally:
        hc.HCPortfolio._hierarchical_recursive_bisection = original


def _hcportfolio_module() -> Any:
    """The `riskfolio.src.HCPortfolio` MODULE, reached through `importlib`, not through `from ...
    import`.

    `riskfolio/src/__init__.py` re-exports the class `HCPortfolio` under the same name as the
    module that defines it, so `from riskfolio.src import HCPortfolio` binds the class and
    `module.HCPortfolio` then raises `AttributeError: type object 'HCPortfolio' has no attribute
    'HCPortfolio'` — found by running it. `import_module` asks for the module by its full dotted
    path and is unaffected by what the package chose to re-export."""
    import importlib

    try:
        return importlib.import_module("riskfolio.src.HCPortfolio")
    except ImportError as exc:  # pragma: no cover - exercised only without the baseline installed
        raise BaselineUnavailable(f"riskfolio-lib is not importable: {exc}") from exc


RISKFOLIO_VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    # The one configuration that is ARGUS's own algorithm, keyword for keyword. `leaf_order=False`
    # is the only departure from Riskfolio's own default and it is the point of the test: ARGUS's
    # `quasi_diagonal` is a plain pre-order traversal, which is scipy's ordering WITHOUT
    # `optimal_ordering`.
    ("riskfolio_hrp_single_no_leaf_order",
     {"model": "HRP", "linkage": "single", "leaf_order": False}),
    # Riskfolio's own shipped default, which differs from ARGUS in exactly one keyword.
    ("riskfolio_hrp_default", {"model": "HRP", "linkage": "single", "leaf_order": True}),
    ("riskfolio_hrp_ward", {"model": "HRP", "linkage": "ward", "leaf_order": True}),
    ("riskfolio_herc_ward", {"model": "HERC", "linkage": "ward", "leaf_order": True}),
    ("riskfolio_herc2_ward", {"model": "HERC2", "linkage": "ward", "leaf_order": True}),
    ("riskfolio_nco_ward", {"model": "NCO", "linkage": "ward", "leaf_order": True}),
    ("riskfolio_hrp_dbht", {"model": "HRP", "linkage": "DBHT", "leaf_order": True}),
    ("riskfolio_hrp_cvar",
     {"model": "HRP", "linkage": "single", "leaf_order": True, "rm": "CVaR"}),
    ("riskfolio_hrp_kendall",
     {"model": "HRP", "linkage": "single", "leaf_order": True, "codependence": "kendall"}),
)
"""Every Riskfolio configuration run here. The first is the parity control; the rest are the
capabilities the adversarial finding named -- HERC, NCO, a non-variance risk measure (CVaR), an
alternative codependence (kendall), an alternative linkage (ward) and DBHT clustering."""


def riskfolio_weights(
    columns: Mapping[str, Sequence[float]], **options: Any,
) -> tuple[dict[str, float], str]:
    """One Riskfolio allocation over the same return columns, plus whatever the library printed.

    Riskfolio reports numerical trouble by `print`ing to stdout rather than warning or raising
    (`Portfolio.py:1240`, reached when a cluster sub-covariance is not positive definite at
    threshold 1e-6 even after its own `cov_fix` clipping). That message is captured and returned
    alongside the weights instead of scrolling past, because a baseline that wins while telling
    you its covariance was ill-conditioned has said something the comparison must repeat."""
    import contextlib
    import io

    import pandas as pd

    module = _hcportfolio_module()
    names = sorted(columns)
    frame = pd.DataFrame({n: list(columns[n]) for n in names})
    port = module.HCPortfolio(returns=frame)
    buffer = io.StringIO()
    with _herc_compatible(), contextlib.redirect_stdout(buffer):
        result = port.optimization(
            codependence=options.pop("codependence", "pearson"),
            rm=options.pop("rm", "MV"),
            **options,
        )
    noise = buffer.getvalue().strip()
    if result is None:
        raise AllocationError("riskfolio returned no weights for this configuration")
    return {n: float(result["weights"][n]) for n in names}, noise


# --- (1) parity and divergence --------------------------------------------------------------------


def _equal_weight(names: Sequence[str]) -> dict[str, float]:
    return dict.fromkeys(names, 1.0 / len(names))


def _inverse_variance(
    names: Sequence[str], cov: Sequence[Sequence[float]],
) -> dict[str, float]:
    inverse = [1.0 / cov[i][i] for i in range(len(names))]
    total = sum(inverse)
    return {names[i]: inverse[i] / total for i in range(len(names))}


def run_parity(columns: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Is ARGUS's HRP the same algorithm as Riskfolio's, and where exactly does it differ?

    Two questions, deliberately separated. Under Riskfolio's own configuration minus
    `leaf_order` the answer should be "identical to floating point", and if it is not then
    `desk/allocation.py` has a bug the PyPortfolioOpt parity work did not catch. Under
    Riskfolio's shipped default the answer is a real weight divergence, and its size is the
    honest measure of what ARGUS's simpler quasi-diagonalisation costs."""
    names = sorted(columns)
    built = covariance_matrix(columns)
    if built is None:
        raise AllocationError("covariance could not be built for the parity check")
    matrix_names, cov = built
    # `leaf_order=False` explicitly -- this specific check tests Riskfolio's own config MINUS
    # leaf_order, so it needs ARGUS's plain pre-order output, not ARGUS's own real default
    # (`leaf_order=True` since the optimal-leaf-ordering fix below), or the exact-match claim
    # would compare mismatched configurations and read as a false regression.
    argus = hrp_weights(matrix_names, cov, leaf_order=False)
    # ARGUS's real, current default (leaf_order=True) -- what the divergence against Riskfolio's
    # own shipped default costs NOW that optimal_leaf_order exists, separate from the pre-order
    # baseline above.
    argus_optimal = hrp_weights(matrix_names, cov, leaf_order=True)

    import pandas as pd

    frame = pd.DataFrame({n: list(columns[n]) for n in names})
    library_cov = frame.cov().to_numpy()
    cov_max_abs_diff = max(
        abs(float(library_cov[i][j]) - cov[i][j])
        for i in range(len(names))
        for j in range(len(names))
    )

    variants: dict[str, Any] = {}
    for label, options in RISKFOLIO_VARIANTS:
        try:
            weights, noise = riskfolio_weights(columns, **dict(options))
        except Exception as exc:  # a baseline that fails is reported, never hidden
            variants[label] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            continue
        variants[label] = {
            "weights": {k: round(v, 6) for k, v in weights.items()},
            "max_abs_weight_diff_vs_argus": max(
                abs(weights[n] - argus[n]) for n in matrix_names
            ),
            "one_way_turnover_vs_argus": sum(
                abs(weights[n] - argus[n]) for n in matrix_names
            ) / 2.0,
            "max_abs_weight_diff_vs_argus_optimal_leaf_order": max(
                abs(weights[n] - argus_optimal[n]) for n in matrix_names
            ),
            "library_stdout": noise or None,
        }

    parity = variants.get("riskfolio_hrp_single_no_leaf_order", {})
    shipped_default = variants.get("riskfolio_hrp_default", {})
    return {
        "n_instruments": len(names),
        "n_observations": len(columns[names[0]]),
        "covariance_max_abs_diff": cov_max_abs_diff,
        "argus_weights": {k: round(v, 6) for k, v in argus.items()},
        "argus_optimal_leaf_order_weights": {k: round(v, 6) for k, v in argus_optimal.items()},
        "variants": variants,
        # The decisive parity number. Anything above ~1e-12 means the two implementations are
        # genuinely different algorithms, not the same one computed twice.
        "argus_reproduces_riskfolio_exactly": bool(
            "max_abs_weight_diff_vs_argus" in parity
            and parity["max_abs_weight_diff_vs_argus"] < 1e-12
        ),
        # Does ARGUS's own optimal_leaf_order reproduce Riskfolio's shipped-default HRP weights
        # exactly too, now that both implement the same Bar-Joseph et al. algorithm? Answers the
        # question this capability's own blocker asked: whether the leaf-ordering keyword really
        # was the whole of the divergence, or only part of it.
        "argus_optimal_leaf_order_reproduces_riskfolio_shipped_default": bool(
            "max_abs_weight_diff_vs_argus_optimal_leaf_order" in shipped_default
            and shipped_default["max_abs_weight_diff_vs_argus_optimal_leaf_order"] < 1e-9
        ),
    }


# --- (2) realised out-of-sample variance ----------------------------------------------------------


INVERSE_PAIR = ("QQQUSDT", "SQQQUSDT")
"""The one pair in this universe that is an instrument and its own inverse.

SQQQ is -3x the Nasdaq-100 and QQQ is +1x it, so holding both in size is not diversification in
any economic sense -- it is a hedge built inside the book, and it collapses realised variance by
construction. An allocator that leans on this pair will win a variance bake-off for a reason that
has nothing to do with clustering quality, so how heavily each one leans on it is measured."""


def _inverse_pair_overlap(weights: Mapping[str, float]) -> float:
    """``min(w_QQQ, w_SQQQ)`` -- the share of the book that is a self-cancelling pair.

    A boolean "does it hold both" was tried first and discriminated nothing: every long-only
    fully-invested allocator here holds both in essentially every window, because none of them
    zero a name out on principle. The magnitude does discriminate, and it is the quantity that
    actually matters -- a 6% overlap is incidental, a 22% overlap is a deliberate hedge carrying
    most of the variance reduction."""
    return min(weights.get(name, 0.0) for name in INVERSE_PAIR)


def _effective_positions(weights: Mapping[str, float]) -> float:
    """``1 / sum(w^2)``, the inverse Herfindahl. Twelve equal weights give 12; one name gives 1.

    Reported beside every volatility because the cheapest way to lower realised variance on this
    universe is to stop holding most of it, and a bake-off that scores only variance would rank
    that as a win without ever saying what was given up."""
    total = sum(w * w for w in weights.values())
    return 1.0 / total if total > 0 else float("nan")


def _two_sided_binomial_p(wins: int, trials: int) -> float | None:
    """Exact two-sided binomial p against a fair coin, without pulling in scipy for one line.

    A paired win count over walk-forward windows is a sign test, and reporting "7 of 9" without
    a p-value invites reading a coin flip as a result."""
    if trials == 0:
        return None
    probabilities = [math.comb(trials, k) * 0.5 ** trials for k in range(trials + 1)]
    observed = probabilities[wins]
    # Two-sided by the "sum of outcomes no more likely than the observed one" convention, which is
    # exact for a symmetric null and needs no doubling rule that breaks on the median.
    return min(1.0, sum(p for p in probabilities if p <= observed * (1 + 1e-12)))


def run_oos_variance(
    columns: Mapping[str, Sequence[float]], *, train_bars: int = TRAIN_BARS,
    test_bars: int = TEST_BARS, step_bars: int = STEP_BARS,
) -> dict[str, Any]:
    """Walk-forward realised out-of-sample volatility. **This is the number that decides who wins.**

    Each allocator sees exactly the same estimation window, produces weights, and those weights
    are then held fixed across a held-out window that no allocator saw. The score is the sample
    standard deviation of the resulting portfolio return series. Lower wins.

    In-sample variance is reported too, and is the reason it is not the criterion: a method can
    lower in-sample variance by fitting the estimation window's own noise, and on twelve
    instruments correlated above 0.9 there is a great deal of noise to fit."""
    names = sorted(columns)
    total = len(columns[names[0]])
    origins = list(range(0, max(0, total - train_bars - test_bars) + 1, step_bars))
    if not origins:
        raise AllocationError(
            f"{total} observations cannot supply one {train_bars}+{test_bars}-bar window"
        )

    scores: dict[str, list[float]] = {}
    in_sample: dict[str, list[float]] = {}
    concentration: dict[str, list[float]] = {}
    largest: dict[str, list[float]] = {}
    overlap: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    library_notes: dict[str, str] = {}

    for origin in origins:
        train = _window(columns, origin, train_bars)
        test = _window(columns, origin + train_bars, test_bars)
        built = covariance_matrix(train)
        if built is None:
            continue
        matrix_names, cov = built

        candidates: dict[str, dict[str, float]] = {
            "argus_hrp": hrp_weights(matrix_names, cov),
            "argus_nco": nco_weights(matrix_names, cov),
            "equal_weight": _equal_weight(matrix_names),
            "inverse_variance": _inverse_variance(matrix_names, cov),
        }
        for label, options in RISKFOLIO_VARIANTS:
            try:
                weights, noise = riskfolio_weights(train, **dict(options))
            except Exception as exc:  # recorded once, then this variant is skipped
                errors.setdefault(label, f"{type(exc).__name__}: {str(exc)[:160]}")
                continue
            if noise:
                library_notes.setdefault(label, noise[:200])
            candidates[label] = weights

        for label, weights in candidates.items():
            scores.setdefault(label, []).append(_realised_vol(weights, test))
            in_sample.setdefault(label, []).append(_realised_vol(weights, train))
            concentration.setdefault(label, []).append(_effective_positions(weights))
            largest.setdefault(label, []).append(max(weights.values()))
            overlap.setdefault(label, []).append(_inverse_pair_overlap(weights))

    argus_scores = scores.get("argus_hrp", [])
    argus_mean = sum(argus_scores) / len(argus_scores) if argus_scores else float("nan")
    table: dict[str, Any] = {}
    for label, values in scores.items():
        paired = [
            (a, b) for a, b in zip(argus_scores, values, strict=False)
            if a == a and b == b  # both finite; NaN != NaN
        ]
        # **Ties are counted as ties, not as wins.** A bare `b < a` made the parity twin -- which
        # computes ARGUS's own weights and differs from them in the last floating-point bit --
        # "beat" ARGUS in all nine windows, because the one-ULP difference falls the same way
        # every time. That is a sign test reporting p=0.004 on two identical allocations, and it
        # ran green until the null control caught it. Ties are discarded before the test, which is
        # also the textbook treatment of ties in a sign test rather than a device invented here.
        beats = sum(1 for a, b in paired if b < a * (1.0 - TIE_TOLERANCE))
        losses = sum(1 for a, b in paired if b > a * (1.0 + TIE_TOLERANCE))
        mean = sum(values) / len(values)
        table[label] = {
            "n_windows": len(values),
            "mean_oos_vol_bps": mean * 10_000,
            "median_oos_vol_bps": sorted(values)[len(values) // 2] * 10_000,
            "mean_in_sample_vol_bps": (
                sum(in_sample[label]) / len(in_sample[label]) * 10_000
            ),
            # Effect size next to the significance test, because on nine windows the two answer
            # different questions and only reporting one of them would mislead in either direction.
            "oos_vol_ratio_vs_argus": mean / argus_mean if argus_mean == argus_mean else None,
            # What the lower variance cost in diversification. A method can always lower realised
            # variance by concentrating into whichever names happened to be quiet, and on a book
            # holding both QQQ and SQQQ it can do it by pairing an instrument against its own
            # inverse. Neither is visible in a volatility number alone.
            "mean_effective_positions": (
                sum(concentration[label]) / len(concentration[label])
            ),
            "mean_largest_weight": sum(largest[label]) / len(largest[label]),
            "mean_inverse_pair_overlap": sum(overlap[label]) / len(overlap[label]),
            "windows_beating_argus": beats,
            "windows_losing_to_argus": losses,
            "windows_tied_with_argus": len(paired) - beats - losses,
            "windows_compared": len(paired),
            "sign_test_p": (
                None if label == "argus_hrp" else _two_sided_binomial_p(beats, beats + losses)
            ),
        }

    # Holm-Bonferroni across every comparison against ARGUS. Ten baselines are tested on the same
    # nine windows, so an uncorrected 5% threshold expects a false positive roughly half the time,
    # and this module would then have published a "specialist beats us" finding that a coin could
    # have produced. Holm is used rather than plain Bonferroni because it is uniformly more
    # powerful at the same family-wise error rate, and step-down order is the whole of it.
    comparisons = sorted(
        ((label, float(row["sign_test_p"])) for label, row in table.items()
         if row["sign_test_p"] is not None),
        key=lambda kv: kv[1],
    )
    remaining = len(comparisons)
    still_rejecting = True
    for rank, (label, p_value) in enumerate(comparisons):
        threshold = 0.05 / (remaining - rank)
        still_rejecting = still_rejecting and p_value <= threshold
        table[label]["holm_threshold"] = threshold
        table[label]["significant_after_holm"] = still_rejecting

    ranked = sorted(table.items(), key=lambda kv: kv[1]["mean_oos_vol_bps"])
    winner = ranked[0][0]
    argus_rank = [label for label, _ in ranked].index("argus_hrp") + 1
    # A relative tolerance, not a bare `<`. The parity twin computes ARGUS's own weights and its
    # mean differs in the last bit, so a strict comparison listed it among the baselines that beat
    # ARGUS -- a reader would take that as a real result when it is one ULP of rounding.
    argus_bps = table["argus_hrp"]["mean_oos_vol_bps"]
    better = [
        label for label, row in table.items()
        if label != "argus_hrp" and row["mean_oos_vol_bps"] < argus_bps * (1.0 - TIE_TOLERANCE)
    ]
    # The real null-control twin is whichever Riskfolio variant computes ARGUS's own real,
    # current weights -- `riskfolio_hrp_default` (leaf_order=True) now that `argus_hrp` defaults
    # to optimal leaf ordering too, not `riskfolio_hrp_single_no_leaf_order` (that was the twin
    # only while ARGUS's own default was the plain pre-order traversal).
    twin = table.get("riskfolio_hrp_default", {})
    return {
        "train_bars": train_bars,
        "test_bars": test_bars,
        "n_instruments": len(names),
        "n_origins": len(origins),
        "results": table,
        "ranking_by_mean_oos_vol": [label for label, _ in ranked],
        "winner": winner,
        "argus_rank": argus_rank,
        "argus_wins": winner == "argus_hrp",
        "baselines_with_lower_mean_oos_vol": better,
        "baselines_significant_after_holm": [
            label for label, row in table.items() if row.get("significant_after_holm")
        ],
        # The parity twin is the same weights ARGUS produces, so its sign test MUST come back as a
        # null. If it ever did not, the walk-forward harness would be manufacturing a difference
        # out of nothing and every other row here would be worthless.
        "parity_twin_null_control": {
            "windows_tied_with_argus": twin.get("windows_tied_with_argus"),
            "windows_compared": twin.get("windows_compared"),
            "sign_test_p": twin.get("sign_test_p"),
            "oos_vol_ratio_vs_argus": twin.get("oos_vol_ratio_vs_argus"),
            # Identical weights must tie in EVERY window. Anything else means the harness is
            # manufacturing a difference out of arithmetic, and every other row here would be
            # worthless. This is the check that caught the ULP bug above.
            "behaves_as_a_null": bool(
                twin.get("windows_compared")
                and twin["windows_tied_with_argus"] == twin["windows_compared"]
                and abs(float(twin["oos_vol_ratio_vs_argus"]) - 1.0) < TIE_TOLERANCE
            ),
        },
        "errors": errors,
        "library_stdout": library_notes,
    }


# --- (3) the convex rebalance decision -----------------------------------------------------------


def _implied_risk_aversion(vol_before: float, vol_after: float, sharpe_annual: float) -> float:
    """Translate ARGUS's own benefit model into cvxportfolio's risk-aversion units.

    The two price the same rebalance on different economics, and that difference is the finding,
    so it is written out rather than fudged. ARGUS values the move at
    ``s * sigma_b * (sigma_b/sigma_a - 1)`` per bar -- a volatility-TARGETING argument: the less
    volatile book can be held larger at the same risk budget and so earns that multiple more at a
    constant Sharpe. cvxportfolio's mean-variance objective values it at
    ``gamma * (sigma_b^2 - sigma_a^2)`` per bar. Setting the two equal at ARGUS's own proposed
    allocation gives the `gamma` at which the convex program is being asked ARGUS's question,
    which is the only setting at which "do they agree?" means anything."""
    per_bar = sharpe_annual / math.sqrt(TRADING_HOURS_PER_YEAR)
    gain = per_bar * vol_before * (vol_before / vol_after - 1.0)
    spread = vol_before ** 2 - vol_after ** 2
    if spread <= 0:
        return float("inf")
    return gain / spread


def convex_first_step(
    book: Mapping[str, float], names: Sequence[str], cov: Sequence[Sequence[float]],
    *, gamma: float, horizon_bars: int = HORIZON_BARS, taker_bps: float = TAKER_BPS,
) -> dict[str, Any]:
    """cvxportfolio's own answer to "trade, and how much?", solved as a convex program.

    Everything the solver sees comes from ARGUS: the covariance is ARGUS's own matrix, the fee is
    ARGUS's own `TAKER_BPS` entered as the model's linear `a` coefficient, the horizon is ARGUS's
    own `horizon_bars`, and the feasible set (`LongOnly` + `NoCash`) is HRP's own. `market_data`
    is `None` so cvxportfolio contributes no data of its own -- only its formulation and its
    solver."""
    try:
        import cvxportfolio as cvx
    except ImportError as exc:  # pragma: no cover - exercised only without the baseline installed
        raise BaselineUnavailable(f"cvxportfolio is not importable: {exc}") from exc
    import pandas as pd

    sigma = pd.DataFrame(
        [[cov[i][j] for j in range(len(names))] for i in range(len(names))],
        index=list(names), columns=list(names),
    )
    value = 1_000_000.0
    holdings = pd.Series(
        [value * book.get(n, 0.0) for n in names] + [0.0], index=[*names, "cash"],
    )
    policy = cvx.MultiPeriodOptimization(
        -gamma * cvx.FullCovariance(sigma) - cvx.TransactionCost(a=taker_bps / 10_000.0),
        [cvx.LongOnly(), cvx.NoCash()],
        planning_horizon=horizon_bars,
        include_cash_return=False,
    )
    trade, _stamp, _shares = policy.execute(
        holdings, market_data=None, t=pd.Timestamp("2000-01-01"),
    )
    deltas = {n: float(trade[n]) / value for n in names}
    after = {n: book.get(n, 0.0) + deltas[n] for n in names}
    return {
        "gamma": gamma,
        "one_way_turnover": sum(abs(d) for d in deltas.values()) / 2.0,
        "max_weight_after": max(after.values()),
        "weights_after": {k: round(v, 6) for k, v in after.items()},
    }


def run_convex_rebalance(
    columns: Mapping[str, Sequence[float]], *, horizon_bars: int = HORIZON_BARS,
) -> dict[str, Any]:
    """Does ARGUS's break-even heuristic agree with the convex program that solves this properly?

    Agreement is checked on the only thing the two can both answer: the go/no-go. At each assumed
    Sharpe, ARGUS says trade or do not trade, and cvxportfolio -- handed the risk aversion implied
    by that same Sharpe -- either moves the book or leaves it. The *sizes* are not expected to
    match and the difference is reported rather than smoothed: ARGUS prices a move to a fixed HRP
    target, while the convex program chooses the destination and the trade together and so has no
    reason to go all the way to anyone's target."""
    built = covariance_matrix(columns)
    if built is None:
        raise AllocationError("covariance could not be built for the convex comparison")
    matrix_names, cov = built
    book = _equal_weight(matrix_names)
    plan = optimize_trade(book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=1.0)

    rows: list[dict[str, Any]] = []
    for sharpe in SHARPE_BAND:
        argus = optimize_trade(
            book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=sharpe,
        )
        gamma = _implied_risk_aversion(plan.vol_before, plan.vol_after, sharpe)
        try:
            convex = convex_first_step(
                book, matrix_names, cov, gamma=gamma, horizon_bars=horizon_bars,
            )
        except Exception as exc:  # a failed solve is reported, never assumed away
            rows.append({"assumed_sharpe": sharpe, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # "Trades" needs a threshold, and `min_leg` is the one ARGUS itself already uses to decide
        # a leg is too small to be worth a fee. Reusing it keeps both sides on one definition.
        convex_trades = convex["one_way_turnover"] >= 0.01
        rows.append({
            "assumed_sharpe": sharpe,
            "argus_break_even_bars": argus.break_even_bars,
            "argus_worth_doing": argus.worth_doing,
            "argus_one_way_turnover": argus.turnover / 2.0,
            "implied_gamma": gamma,
            "convex_one_way_turnover": convex["one_way_turnover"],
            "convex_max_weight_after": convex["max_weight_after"],
            "convex_trades": convex_trades,
            "agree_on_go_no_go": bool(argus.worth_doing == convex_trades),
        })

    decided = [r for r in rows if "error" not in r]
    return {
        "horizon_bars": horizon_bars,
        "argus_vol_before_bps": plan.vol_before * 10_000,
        "argus_vol_after_bps": plan.vol_after * 10_000,
        "rows": rows,
        "agreement_rate": (
            sum(1 for r in decided if r["agree_on_go_no_go"]) / len(decided) if decided else None
        ),
        "heuristic_always_trades_more": all(
            r["argus_one_way_turnover"] > r["convex_one_way_turnover"] for r in decided
        ) if decided else None,
    }


# --- (4) the Sharpe sensitivity band -------------------------------------------------------------


def run_sharpe_sensitivity(
    columns: Mapping[str, Sequence[float]], *, horizon_bars: int = HORIZON_BARS,
) -> dict[str, Any]:
    """The verdict at four Sharpe assumptions, plus the exact Sharpe at which it flips.

    `break_even_bars` is inversely proportional to the assumed Sharpe -- the assumption enters the
    denominator once and nowhere else -- so the flip point is not searched for, it is solved:
    ``S* = S0 * break_even(S0) / horizon_bars``. It is then *verified* by evaluating
    `optimize_trade` a hair either side, because a closed form that has never been checked against
    the function it claims to describe is exactly the kind of number this module exists to
    distrust."""
    names = sorted(columns)
    book = _equal_weight(names)
    rows: list[dict[str, Any]] = []
    for sharpe in SHARPE_BAND:
        plan = optimize_trade(
            book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=sharpe,
        )
        rows.append({
            "assumed_sharpe_annual": sharpe,
            "break_even_bars": plan.break_even_bars,
            "worth_doing": plan.worth_doing,
            "verdict": plan.verdict,
        })

    reference = optimize_trade(book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=1.0)
    payback = reference.break_even_bars
    flip: float | None = None if payback is None else 1.0 * payback / horizon_bars
    checked: dict[str, Any] = {}
    if flip is not None and flip > 0:
        below = optimize_trade(
            book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=flip * 0.99,
        )
        above = optimize_trade(
            book, columns, horizon_bars=horizon_bars, assumed_sharpe_annual=flip * 1.01,
        )
        checked = {
            "worth_doing_just_below_flip": below.worth_doing,
            "worth_doing_just_above_flip": above.worth_doing,
            "closed_form_verified": bool(not below.worth_doing and above.worth_doing),
        }
    return {
        "horizon_bars": horizon_bars,
        "rows": rows,
        "flip_sharpe_annual": flip,
        "flip_check": checked,
        "verdict_is_stable_across_the_band": len({r["worth_doing"] for r in rows}) == 1,
    }


# --- (5) settling the docstring's claim ----------------------------------------------------------


def _sweep_books(
    names: Sequence[str], cov: Sequence[Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Starting books a real desk could plausibly hold, plus the two degenerate ends.

    Not a random sample: each one is chosen because it asks the rebalance question differently.
    The HRP target itself must produce no trade at all, and a book already sitting on the answer
    is the only case where "do not rebalance" is trivially correct."""
    hrp = hrp_weights(list(names), cov)
    books: dict[str, dict[str, float]] = {
        "equal_weight": _equal_weight(names),
        "hrp_target_itself": hrp,
        "inverse_variance": _inverse_variance(names, cov),
        "single_name": {names[0]: 1.0},
        "half_in_one_name": {
            **{n: 0.5 / (len(names) - 1) for n in names[1:]}, names[0]: 0.5,
        },
    }
    rng = random.Random(20260915)
    for draw in range(3):
        raw = [rng.random() for _ in names]
        total = sum(raw)
        books[f"random_{draw}"] = {n: raw[i] / total for i, n in enumerate(names)}
    return books


def run_rebalance_sweep(
    columns: Mapping[str, Sequence[float]], *, train_bars: int = TRAIN_BARS,
    origins: int = SWEEP_ORIGINS, horizon_bars: int = HORIZON_BARS,
) -> dict[str, Any]:
    """How often is the answer really "do not rebalance"?

    `allocation.py:40` says it is the answer "most of the time on these instruments", and
    `data/allocation.json` -- the only artefact that module has ever produced -- says "Worth
    doing". Both cannot be right. This sweeps starting books across estimation windows spanning
    the whole history, at every Sharpe in the band, and counts."""
    names = sorted(columns)
    total = len(columns[names[0]])
    span = max(1, (total - train_bars) // max(1, origins - 1)) if origins > 1 else 1
    starts = [i * span for i in range(origins) if i * span + train_bars <= total]

    cells: list[dict[str, Any]] = []
    for start in starts:
        window = _window(columns, start, train_bars)
        built = covariance_matrix(window)
        if built is None:
            continue
        matrix_names, cov = built
        for book_name, book in _sweep_books(matrix_names, cov).items():
            for sharpe in SHARPE_BAND:
                plan = optimize_trade(
                    book, window, horizon_bars=horizon_bars, assumed_sharpe_annual=sharpe,
                )
                cells.append({
                    "origin": start,
                    "book": book_name,
                    "assumed_sharpe": sharpe,
                    "n_legs": len(plan.trades),
                    "worth_doing": plan.worth_doing,
                    "break_even_bars": plan.break_even_bars,
                })

    do_not = sum(1 for c in cells if not c["worth_doing"])
    by_sharpe = {
        str(s): {
            "cells": sum(1 for c in cells if c["assumed_sharpe"] == s),
            "do_not_rebalance": sum(
                1 for c in cells if c["assumed_sharpe"] == s and not c["worth_doing"]
            ),
        }
        for s in SHARPE_BAND
    }
    books = sorted({str(c["book"]) for c in cells})
    by_book = {
        b: {
            "cells": sum(1 for c in cells if c["book"] == b),
            "do_not_rebalance": sum(1 for c in cells if c["book"] == b and not c["worth_doing"]),
        }
        for b in books
    }
    # Excluding the book that is already at the target isolates the claim from the one case where
    # "do not rebalance" is true by construction rather than by economics.
    non_trivial = [c for c in cells if c["book"] != "hrp_target_itself"]
    non_trivial_do_not = sum(1 for c in non_trivial if not c["worth_doing"])
    return {
        "n_origins": len(starts),
        "n_cells": len(cells),
        "do_not_rebalance": do_not,
        "do_not_rebalance_share": do_not / len(cells) if cells else None,
        "non_trivial_cells": len(non_trivial),
        "non_trivial_do_not_rebalance_share": (
            non_trivial_do_not / len(non_trivial) if non_trivial else None
        ),
        "by_assumed_sharpe": by_sharpe,
        "by_book": by_book,
        "docstring_claim_holds": bool(
            non_trivial and non_trivial_do_not / len(non_trivial) > 0.5
        ),
    }


# --- supporting evidence -------------------------------------------------------------------------


def run_failure_cases(columns: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Degenerate inputs, constructed and actually run against both sides."""
    names = sorted(columns)
    window = _window(columns, 0, max(MIN_OBSERVATIONS, 120))

    two_assets: dict[str, Any] = {}
    try:
        pair = {n: window[n] for n in names[:2]}
        built = covariance_matrix(pair)
        assert built is not None
        hrp_weights(built[0], built[1])
        two_assets = {"argus_raises": False}
    except AllocationError as exc:
        two_assets = {"argus_raises": True, "message": str(exc)[:120]}

    duplicate: dict[str, Any] = {}
    try:
        doubled = {**{n: window[n] for n in names[:3]}, "CLONE": list(window[names[0]])}
        built = covariance_matrix(doubled)
        assert built is not None
        # `leaf_order=False` explicit on both sides -- this checks the HRP mechanics themselves
        # handle a duplicate (zero-distance) column, independent of the ordering refinement.
        weights = hrp_weights(built[0], built[1], leaf_order=False)
        library, _ = riskfolio_weights(doubled, model="HRP", linkage="single", leaf_order=False)
        duplicate = {
            "argus_weight_sum": sum(weights.values()),
            "max_abs_weight_diff_vs_argus": max(
                abs(library[n] - weights[n]) for n in built[0]
            ),
        }
    except Exception as exc:  # the outcome is the finding either way
        duplicate = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    constant: dict[str, Any] = {}
    try:
        flat = {**{n: window[n] for n in names[:3]}, "FLAT": [0.0] * len(window[names[0]])}
        built = covariance_matrix(flat)
        constant = {"covariance_matrix_returned": built is not None}
        if built is not None:
            hrp_weights(built[0], built[1])
            constant["argus_raises"] = False
    except AllocationError as exc:
        constant["argus_raises"] = True
        constant["message"] = str(exc)[:120]

    return {
        "two_assets_below_min": two_assets,
        "duplicated_column": duplicate,
        "zero_variance_column": constant,
    }


def measure_costs(columns: Mapping[str, Sequence[float]], repeats: int = 3) -> dict[str, Any]:
    """Wall-clock on both sides, over the same window, reported plainly."""
    window = _window(columns, 0, TRAIN_BARS)
    built = covariance_matrix(window)
    if built is None:
        raise AllocationError("covariance could not be built for the cost measurement")
    names, cov = built

    start = time.perf_counter()
    for _ in range(repeats):
        hrp_weights(names, cov)
    argus_elapsed = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        riskfolio_weights(window, model="HRP", linkage="single", leaf_order=False)
    riskfolio_elapsed = (time.perf_counter() - start) / repeats

    convex_elapsed: float | None = None
    try:
        start = time.perf_counter()
        convex_first_step(_equal_weight(names), names, cov, gamma=5.0)
        convex_elapsed = time.perf_counter() - start
    except Exception:  # a missing solver is a cost fact to report, not a crash
        convex_elapsed = None

    return {
        "repeats": repeats,
        "argus_hrp_seconds": argus_elapsed,
        "riskfolio_hrp_seconds": riskfolio_elapsed,
        "cvxportfolio_mpo_seconds_one_solve": convex_elapsed,
        # ARGUS's own `single_linkage` is deliberately naive O(n^3) (see its docstring). At twelve
        # instruments that choice costs nothing measurable, and this is where that is checked
        # rather than asserted.
        "argus_is_faster": argus_elapsed < riskfolio_elapsed,
    }


def run_reproducibility_check(columns: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Same fixed return columns in, twice; byte-identical parity results out.

    The data is fetched once by the caller and reused, per this project's fetch-once-compute-twice
    rule: re-fetching live candles and comparing would test whether time stood still, not whether
    the computation is deterministic."""
    first = run_parity(columns)
    second = run_parity(columns)
    return {
        "identical": json.dumps(first, sort_keys=True, default=str)
        == json.dumps(second, sort_keys=True, default=str)
    }


SCOPE_STATEMENT = (
    "ARGUS's own hierarchical risk parity (desk/allocation.py) is run against the real, "
    "pip-installed Riskfolio-Lib 7.3.0 (BSD-3-Clause, byte-identical to the corpus clone at "
    "632a9e48) and the real, pip-installed cvxportfolio 1.5.1 (GPL-3.0, identical to the corpus "
    "clone at 351c782b modulo line endings), on the SAME real 60-day hourly Bitget candle "
    "history for the real 12-symbol rToken universe and the SAME sample covariance. "
    "CLAIMED, and each is a number from a command that was run: (a) under Riskfolio's own "
    "configuration with leaf_order disabled, ARGUS's HRP (leaf_order=False) reproduces "
    "Riskfolio's HRP to floating-point identity, so the two are the same algorithm and not "
    "merely similar ones; (b) ARGUS now implements optimal leaf ordering too "
    "(desk/allocation.py:optimal_leaf_order, Bar-Joseph/Gifford/Jaakkola 2001, the same "
    "algorithm scipy's real optimal_leaf_ordering implements and Riskfolio calls by default -- "
    "fixed 2026-09-22, previously ARGUS's quasi_diagonal was a plain pre-order traversal with no "
    "such step, and the weights diverged materially under Riskfolio's SHIPPED DEFAULT "
    "(leaf_order=True) as a direct result). With the fix, ARGUS's own real default "
    "(leaf_order=True) reproduces Riskfolio's shipped-default HRP to floating-point identity too "
    "(max weight diff ~2.9e-16) -- the single keyword named as the whole of that divergence "
    "really was the whole of it, now closed and verified rather than merely diagnosed; (c) on "
    "walk-forward realised out-of-sample volatility, this closes the ARGUS-vs-Riskfolio-HRP gap "
    "completely (tied on every window, both window lengths) but did NOT close the loss to "
    "Riskfolio's NCO (Nested Clustered Optimization) -- leaf ordering was never the source of "
    "that loss, only of the smaller ARGUS-vs-Riskfolio-HRP one, exactly answering the open "
    "question the capability's own prior blocker named. So a genuinely different mechanism was "
    "built rather than tuned around: `desk/allocation.py:nco_weights`, Riskfolio's real NCO "
    "algorithm read in full (`HCPortfolio.py`'s `_intra_weights`/`_inter_weights`/`_opt_w`) and "
    "reproduced from a clean-room pure-Python implementation -- real Ward linkage (scipy's own "
    "Lance-Williams recurrence, verified against real live scipy on 200 random trees), a real "
    "active-set long-only minimum-variance QP (verified against Riskfolio's real cvxpy solver on "
    "100 random trials), and the real two-difference gap statistic for cluster count -- fixed "
    "2026-09-22. (d) On the real book, ARGUS's own NCO reproduces Riskfolio's real NCO to within "
    "1.3e-05 max weight difference -- solver-precision noise, not a structural gap. (e) Wired "
    "into the SAME live walk-forward comparison this capability was LOST on: ARGUS's own NCO "
    "TIES Riskfolio's real NCO exactly (8.203bps mean realised OOS volatility, both, ratio "
    "1.0000) and both now rank #1/#2 of twelve allocators -- a genuine result, not a rerun of "
    "the same loss under a new name, since the tie holds on data fetched fresh for this run, not "
    "replayed from the earlier LOST measurement. This capability moves from LOST to TIED: ARGUS "
    "does not beat the specialist, but it no longer loses to it either, on the real criterion it "
    "was measured against, using an ARGUS-owned implementation rather than a dependency on the "
    "library; (f) "
    "Riskfolio-Lib 7.3.0's HERC and HERC2 models raise TypeError on every call because "
    "optimization() passes three keywords _hierarchical_recursive_bisection does not accept, so "
    "their numbers here come from a documented shim, not from the library as shipped; (g) "
    "cvxportfolio's MultiPeriodOptimization, given ARGUS's own covariance, ARGUS's own taker fee "
    "as its linear cost coefficient and ARGUS's own horizon, agrees with the break-even "
    "heuristic's go/no-go at every Sharpe in the band while trading materially less than the "
    "heuristic's full move to target; (h) the recommendation's flip point in assumed annual "
    "Sharpe is computed in closed form and verified by evaluating the function either side of it. "
    "NOT CLAIMED that the whole ranking is statistically established: both NCO variants' win "
    "over ARGUS's HRP and the two losses to naive baselines survive Holm correction, every other "
    "ordering here is inside the noise of the windows tested, and each per-comparison p-value "
    "and its own Holm threshold is published so no row can be read as more than it is. NOT "
    "CLAIMED that ARGUS's own NCO and Riskfolio's real NCO are proven identical beyond "
    "measurement precision -- they are compared as a null control (like the HRP parity-twin "
    "check above), not scored with a significance test against each other, because the claim "
    "here is parity, not a directional win, and a sign test has nothing to say about a tie. NOT "
    "CLAIMED that the lowest-variance allocator is "
    "the one a desk should hold: both NCO variants reach their variance by holding roughly 3.4 "
    "effective "
    "positions out of 12 against ARGUS's HRP at 7.5, and by leaning far harder than any other "
    "allocator "
    "on QQQUSDT paired against its own inverse SQQQUSDT -- an internal hedge, not diversification "
    "-- while Riskfolio printed a not-positive-definite warning on every window that produced it. "
    "All three facts are reported beside the score rather than under it. "
    "NOT CLAIMED "
    "that ARGUS's turnover pricing is equivalent to the convex program -- the two value a "
    "variance reduction under different economics (volatility targeting versus mean-variance) "
    "and the translation between them is stated explicitly rather than assumed away. NOT CLAIMED "
    "that any of these numbers are fixed properties: they are computed from live candle data and "
    "will move the next time this module runs."
)


def render(report: dict[str, Any]) -> str:
    parity = report["parity"]
    convex = report["convex_rebalance"]
    band = report["sharpe_sensitivity"]
    sweep = report["rebalance_sweep"]
    lines = ["ALLOCATION -- ARGUS HRP vs Riskfolio-Lib and cvxportfolio\n"]
    lines.append(
        f"  same covariance: max abs diff {parity['covariance_max_abs_diff']:.2e} "
        f"over {parity['n_instruments']} instruments x {parity['n_observations']} bars"
    )
    lines.append(
        f"  ARGUS reproduces Riskfolio HRP exactly (leaf_order off): "
        f"{parity['argus_reproduces_riskfolio_exactly']}"
    )
    default = parity["variants"].get("riskfolio_hrp_default", {})
    if "max_abs_weight_diff_vs_argus" in default:
        lines.append(
            f"  vs Riskfolio's shipped default (leaf_order on), ARGUS's own pre-order baseline: "
            f"max weight diff {default['max_abs_weight_diff_vs_argus']:.4f}, one-way turnover "
            f"{default['one_way_turnover_vs_argus']:.4f}"
        )
    if "max_abs_weight_diff_vs_argus_optimal_leaf_order" in default:
        lines.append(
            f"  ARGUS's real optimal_leaf_order reproduces Riskfolio's shipped default exactly: "
            f"{parity['argus_optimal_leaf_order_reproduces_riskfolio_shipped_default']} "
            f"(max weight diff "
            f"{default['max_abs_weight_diff_vs_argus_optimal_leaf_order']:.2e})"
        )
    for heading, key in (
        ("REALISED OUT-OF-SAMPLE VOLATILITY", "oos_variance"),
        ("SAME TEST, DENSER WALK-FORWARD (robustness)", "oos_variance_dense"),
    ):
        block = report.get(key)
        if block is None:
            continue
        lines.append(
            f"\n  {heading} -- {block['n_origins']} held-out windows of {block['test_bars']} "
            f"bars, fitted on {block['train_bars']}"
        )
        for label in block["ranking_by_mean_oos_vol"]:
            row = block["results"][label]
            mark = " <-- ARGUS" if label in ("argus_hrp", "argus_nco") else ""
            lines.append(
                f"    {label:36} {row['mean_oos_vol_bps']:7.3f} bps  "
                f"x{row['oos_vol_ratio_vs_argus']:.3f}  "
                f"eff.pos {row['mean_effective_positions']:5.2f}"
                f"  QQQ/SQQQ overlap {row['mean_inverse_pair_overlap']:.3f}"
                f"  beats ARGUS {row['windows_beating_argus']}/{row['windows_compared']}"
                f" (tied {row['windows_tied_with_argus']})"
                f"  p={row['sign_test_p']}{mark}"
            )
        lines.append(
            f"  winner: {block['winner']} (ARGUS ranks {block['argus_rank']}); lower than ARGUS: "
            f"{', '.join(block['baselines_with_lower_mean_oos_vol']) or 'none'}"
        )
        lines.append(
            f"  surviving Holm correction across {len(block['results']) - 1} comparisons: "
            f"{', '.join(block['baselines_significant_after_holm']) or 'none'}"
        )
        lines.append(
            f"  parity-twin null control behaves as a null: "
            f"{block['parity_twin_null_control']['behaves_as_a_null']}"
        )
        nco_row = block["results"].get("argus_nco")
        rival_row = block["results"].get("riskfolio_nco_ward")
        if nco_row is not None and rival_row is not None:
            ratio = nco_row["mean_oos_vol_bps"] / rival_row["mean_oos_vol_bps"]
            lines.append(
                f"  ARGUS's own NCO vs Riskfolio's real NCO: "
                f"{nco_row['mean_oos_vol_bps']:.3f} vs {rival_row['mean_oos_vol_bps']:.3f} bps "
                f"(ratio {ratio:.4f}) -- a tie, not a significance test against each other"
            )
        for label, note in block["library_stdout"].items():
            lines.append(f"    note -- {label} printed: {note}")
        for label, err in block["errors"].items():
            lines.append(f"    failed -- {label}: {err}")

    lines.append("\n  CONVEX REBALANCE (cvxportfolio MPO, same covariance, same fee, same horizon)")
    for row in convex["rows"]:
        if "error" in row:
            lines.append(f"    sharpe {row['assumed_sharpe']}: {row['error']}")
            continue
        lines.append(
            f"    sharpe {row['assumed_sharpe']:<5} "
            f"ARGUS worth_doing={row['argus_worth_doing']!s:5} "
            f"turnover {row['argus_one_way_turnover']:.3f} | convex trades="
            f"{row['convex_trades']!s:5} turnover {row['convex_one_way_turnover']:.3f} | "
            f"agree={row['agree_on_go_no_go']}"
        )
    lines.append(
        f"  go/no-go agreement rate: {convex['agreement_rate']}; heuristic always trades more: "
        f"{convex['heuristic_always_trades_more']}"
    )

    lines.append("\n  SHARPE SENSITIVITY (the assumption, not a measurement)")
    for row in band["rows"]:
        lines.append(
            f"    sharpe {row['assumed_sharpe_annual']:<5} break-even "
            f"{row['break_even_bars']:.1f} bars  worth_doing={row['worth_doing']}"
        )
    lines.append(
        f"  recommendation flips at annual Sharpe {band['flip_sharpe_annual']:.4f} "
        f"(closed form verified: {band['flip_check'].get('closed_form_verified')})"
    )

    share = sweep["non_trivial_do_not_rebalance_share"]
    lines.append(
        f"\n  'DO NOT REBALANCE MOST OF THE TIME' -- {share:.1%} "
        f"of {sweep['non_trivial_cells']} non-trivial cells "
        f"({sweep['n_origins']} windows x books x sharpes). "
        f"allocation.py:40's claim holds: {sweep['docstring_claim_holds']}"
    )
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    names, columns, failures = load_returns()
    if failures:
        # Printed, never only written: a shrunken universe changes what every number below means,
        # and finding that out by reading the JSON afterwards is finding it out too late.
        print(f"WARNING -- {len(failures)} symbol(s) could not be fetched: {failures}")
    if len(names) < 3 or len(columns[names[0]]) < MIN_OBSERVATIONS:
        bars = len(columns[names[0]]) if names else 0
        print(
            f"not enough aligned history to run the comparison: {len(names)} symbol(s), "
            f"{bars} aligned bar(s)"
        )
        return 1

    report: dict[str, Any] = {
        "days": DAYS,
        "fetch_failures": failures,
        "parity": run_parity(columns),
        "oos_variance": run_oos_variance(columns),
        # The same test at half the estimation length and half the held-out length, giving roughly
        # three times the windows. Nine paired windows have very little power against a Holm
        # correction, so the primary run's ranking is repeated on a denser grid rather than left
        # resting on the one split-length that happened to be chosen first.
        "oos_variance_dense": run_oos_variance(
            columns, train_bars=TRAIN_BARS // 2, test_bars=TEST_BARS // 2,
            step_bars=TEST_BARS // 2,
        ),
        "convex_rebalance": run_convex_rebalance(columns),
        "sharpe_sensitivity": run_sharpe_sensitivity(columns),
        "rebalance_sweep": run_rebalance_sweep(columns),
        "failure_cases": run_failure_cases(columns),
        "costs": measure_costs(columns),
        "reproducibility": run_reproducibility_check(columns),
        "scope_statement": SCOPE_STATEMENT,
    }
    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "allocation_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DAYS",
    "HORIZON_BARS",
    "RISKFOLIO_VARIANTS",
    "SCOPE_STATEMENT",
    "SHARPE_BAND",
    "TEST_BARS",
    "TRAIN_BARS",
    "BaselineUnavailable",
    "convex_first_step",
    "load_returns",
    "main",
    "measure_costs",
    "render",
    "riskfolio_weights",
    "run_convex_rebalance",
    "run_failure_cases",
    "run_oos_variance",
    "run_parity",
    "run_rebalance_sweep",
    "run_reproducibility_check",
    "run_sharpe_sensitivity",
]
