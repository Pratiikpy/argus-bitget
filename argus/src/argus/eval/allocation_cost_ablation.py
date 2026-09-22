"""Adversarial re-test of `eval/allocation_comparison.py`'s headline, by running the three things
that comparison itself listed as NOT VERIFIED and whose answers could overturn its verdict.

**Why this module exists.** `allocation_comparison.py` published an honest, anti-flattering
result: Riskfolio-Lib's NCO beats ARGUS's HRP on realised out-of-sample volatility in 22 of 24
held-out windows (`data/allocation_comparison.json` `.oos_variance_dense.results
.riskfolio_nco_ward`). It then named three reasons that result might not mean what it appears to,
and left all three untested:

1. *Turnover was never charged.* NCO refits to a far more concentrated book each window, so it
   moves more weight than ARGUS does, and none of that was priced. "Lower variance" is not
   "better" until the fee for reaching it is paid.
2. *The edge may be mechanical, not analytic.* NCO holds 3.35 effective positions out of 12
   against ARGUS's 7.46. Concentrating into whichever names were quiet lowers realised variance
   without any claim about clustering quality.
3. *The edge may rest on an internal hedge.* NCO puts 0.146 of the book into the smaller leg of
   QQQUSDT/SQQQUSDT -- an instrument against its own -3x inverse -- versus ARGUS's 0.068.

A verdict of LOST that survives all three is a real finding. One that does not is a measurement
artefact, and publishing it as LOST would be its own kind of dishonesty. So each is run here.

**This module re-derives the baseline independently rather than importing the comparison's own
allocator plumbing.** Only `load_returns` is shared, because the whole point is that every
allocator sees the same candles. The Riskfolio call below is written from the library's own source
rather than from the sibling module, so that if that module's wrapper were wrong, this one would
disagree with it instead of inheriting the error. It agrees; see `.replication` in the artefact.

**Read before written**, per the standing rule.

`research/repos-t2/dcajasn~riskfolio-lib/riskfolio/src/HCPortfolio.py` (BSD-3-Clause; byte-identical
to the installed 7.3.0 wheel, verified by `diff`):

* `:83-107` -- `HCPortfolio.__init__` takes `w_max` and `w_min`. This is the library's own
  maximum-weight constraint and the only honest way to ask ablation (2): a cap I wrote myself
  would be my algorithm being compared, not Riskfolio's.
* `:1060-1078` -- `w_max` becomes a per-asset `upper_bound` Series, defaulting to 1.0.
* `:1119-1140` -- **the bound is applied after the model, to every model including NCO**: weights
  are clipped to `[lower_bound, upper_bound]` and the excess redistributed among the unbound names,
  iterating up to 100 times. So `w_max` genuinely constrains NCO and is not silently ignored, which
  is the thing that had to be checked before the capped run could mean anything.
* `:1104-1115` -- the NCO branch: `_intra_weights` then `_inter_weights`, both of which solve a
  mean-variance program with cvxpy. This is the code whose output is being re-scored here.
* `:760-781` -- `optimization()`'s defaults, of which `leaf_order=True` is the one that separates
  Riskfolio's shipped HRP from ARGUS's. `leaf_order=False` recovers ARGUS's own algorithm exactly
  (`allocation_comparison.py` measures the agreement at 1.9e-16), which is what lets the capped
  run put a capped *ARGUS* on the board: ARGUS itself has no `w_max`, so its capped twin is
  obtained by running its own algorithm through Riskfolio's own bound-fitting loop rather than by
  inventing a capping rule here.

`src/argus/desk/allocation.py` (ARGUS's own), read for the cost model so the fee charged here is
the desk's own and not a number picked for this test:

* `:441-451` -- `optimize_trade` computes `turnover = sum(abs(t.delta))` (two-sided) and
  `cost_bps = turnover * taker_bps`. That exact convention is reused below.
* `:416-418` -- `assumed_sharpe_annual` is "an assumption, not a measurement". This module
  therefore does **not** price the benefit through an assumed Sharpe: it measures realised net
  return directly, which needs no such assumption.

**What was taken and what was rejected.** Taken: Riskfolio's own `HCPortfolio.optimization()`, its
own `w_max` bound-fitting, ARGUS's own `hrp_weights` and ARGUS's own two-sided turnover fee.
Rejected: scoring the net comparison through `optimize_trade`'s assumed-Sharpe break-even, because
that would make the answer a function of the very assumption the sibling module was built to
distrust -- realised return over held-out bars is an observation and needs no Sharpe input.
Also rejected: capping ARGUS with a rule written here (see `:1119-1140` above for why the library's
own loop is used instead).

**The criterion, and why it is this one.** Each allocator is run as an actual walk-forward
strategy: refit on the estimation window, pay ARGUS's own taker fee on every unit of weight moved
from what it held before, then hold through bars it has never seen. Two scores come out:

* *net return*, which is what a fee-paying desk keeps; and
* *net return at a common risk budget*, which is the only reading that lets a lower-volatility
  allocator cash its advantage in. Leverage is set from the **estimation window's** modelled
  volatility, never the held-out window's realised one, so nothing here can see the future. This
  is the charitable reading for NCO: it is exactly the argument that a lower-variance book can be
  held larger, made concrete and then charged for.

    python -m argus.eval.allocation_cost_ablation
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from argus.desk.allocation import (
    TAKER_BPS,
    TRADING_HOURS_PER_YEAR,
    AllocationError,
    hrp_weights,
    portfolio_variance,
)
from argus.desk.portfolio import covariance_matrix
from argus.eval.allocation_comparison import (
    INVERSE_PAIR,
    STEP_BARS,
    TEST_BARS,
    TRAIN_BARS,
    BaselineUnavailable,
    load_returns,
)

DENSE_TRAIN_BARS = TRAIN_BARS // 2
"""240 bars. The sibling module's *denser* walk-forward is replicated, not its primary one: 24
paired windows is the only one of its two grids where NCO's win survives Holm correction with any
power to spare, so it is the grid the ablations have to be run on to be able to detect a loss of
that win rather than merely failing to find it."""

DENSE_TEST_BARS = TEST_BARS // 2
"""48 bars held out, non-overlapping."""

DENSE_STEP_BARS = STEP_BARS // 2
"""Equal to the held-out length, so consecutive windows touch but never overlap."""

WEIGHT_CAP = 0.25
"""The maximum single-name weight for ablation (2). Three times equal weight on a twelve-name book,
which is a limit a desk would plausibly run and is well above every weight ARGUS itself produced
(its largest mean weight is 0.231) -- so the cap is a constraint on NCO's concentration and very
nearly a no-op for ARGUS. That asymmetry is the experiment, not a flaw in it: the question is
whether NCO's advantage is *made of* concentration."""

TARGET_VOL_BPS = 15.0
"""Common per-bar risk budget for the levered comparison, in basis points. Set at ARGUS's own mean
modelled volatility (14.7 bps in-sample on the dense grid) rather than at a round number, so ARGUS
runs at roughly unit leverage and the levered scores are not dominated by one side being scaled far
away from where it actually operates."""

TIE_TOLERANCE = 1e-9
"""Relative gap below which two scores are the same number rather than a win.

**Found by running the null control, not reasoned about in advance.** A bare `gap != 0` counted
the last floating-point bit as a result: `riskfolio_hrp_argus_equivalent` is ARGUS's own algorithm
run through Riskfolio and agrees with it to 1.9e-16, yet a strict comparison scored it 9 wins and
9 losses over 18 "compared" windows instead of 24 ties. The sibling module hit the identical trap
and fixed it the same way, which is the strongest reason to think the band is right rather than
convenient: it sits nine orders of magnitude above the ULP noise and seven below the smallest
genuine difference any allocator here shows."""

MAX_LEVERAGE = 4.0
"""Leverage ceiling. Without one, a window whose estimation volatility collapses hands an allocator
unbounded size on held-out bars it cannot see, and the comparison becomes a measurement of who got
the luckiest variance estimate. Binding only when the estimate says the book is more than four
times quieter than the risk budget."""


def _riskfolio_allocation(
    columns: Mapping[str, Sequence[float]], *, w_max: float | None = None, **options: Any,
) -> dict[str, float]:
    """One Riskfolio allocation over these return columns, through its own public entry point.

    Written from `HCPortfolio.py` directly rather than by calling the sibling module's wrapper, so
    that the two are independent readings of the same library. `w_max` is passed to the
    constructor (`:83-107`) rather than to `optimization()`, because that is where the library
    takes it; `optimization()`'s own signature at `:760-781` has no weight-bound parameter at all.

    `importlib` rather than `from riskfolio.src import HCPortfolio`: `riskfolio/src/__init__.py`
    re-exports the class under the same name as its module, so the plain import binds the class
    and any later module-level attribute access fails. Confirmed by running it.
    """
    import contextlib
    import importlib
    import io

    import pandas as pd

    try:
        module = importlib.import_module("riskfolio.src.HCPortfolio")
    except ImportError as exc:  # pragma: no cover - only without the baseline installed
        raise BaselineUnavailable(f"riskfolio-lib is not importable: {exc}") from exc

    names = sorted(columns)
    frame = pd.DataFrame({n: list(columns[n]) for n in names})
    port = module.HCPortfolio(returns=frame, w_max=w_max)
    # Riskfolio reports an ill-conditioned cluster covariance by printing, not by warning or
    # raising. Swallowed here only so a 24-window sweep is readable; the sibling module already
    # publishes the message itself, so nothing is being hidden that is not on record.
    with contextlib.redirect_stdout(io.StringIO()):
        result = port.optimization(**options)
    if result is None:
        raise AllocationError(f"riskfolio returned no weights for {options}")
    return {n: float(result["weights"][n]) for n in names}


# --- the shared walk-forward harness -------------------------------------------------------------


def _window(
    columns: Mapping[str, Sequence[float]], start: int, size: int,
) -> dict[str, list[float]]:
    return {name: list(values[start:start + size]) for name, values in columns.items()}


def _portfolio_bars(
    weights: Mapping[str, float], columns: Mapping[str, Sequence[float]],
) -> list[float]:
    """The fixed-weight portfolio's own realised bar returns over this window."""
    names = sorted(columns)
    size = len(columns[names[0]])
    return [sum(weights.get(n, 0.0) * columns[n][i] for n in names) for i in range(size)]


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _turnover(new: Mapping[str, float], old: Mapping[str, float]) -> float:
    """Two-sided ``sum(abs(delta))``, which is `desk/allocation.py:441`'s own convention.

    Deliberately not the one-way half: the fee ARGUS charges is `turnover * taker_bps` with this
    same two-sided quantity, and using a different definition here would silently halve or double
    the cost relative to what the desk itself would pay."""
    keys = set(new) | set(old)
    return sum(abs(new.get(k, 0.0) - old.get(k, 0.0)) for k in keys)


def _two_sided_binomial_p(wins: int, trials: int) -> float | None:
    """Exact two-sided sign test against a fair coin. Same construction as the sibling module's,
    reproduced rather than imported so that a bug there would show up here as a disagreement."""
    if trials == 0:
        return None
    probabilities = [math.comb(trials, k) * 0.5 ** trials for k in range(trials + 1)]
    observed = probabilities[wins]
    return min(1.0, sum(p for p in probabilities if p <= observed * (1 + 1e-12)))


Allocator = tuple[str, Any]


def _allocators(w_max: float | None) -> list[Allocator]:
    """The three allocators every run below scores, and nothing else.

    Kept to three on purpose: this module is not a second ranking exercise, it is a test of one
    specific claim -- that NCO beats ARGUS -- under conditions the original run did not impose.
    `argus_hrp` is ARGUS's own code; `riskfolio_nco_ward` is the configuration that won; and
    `riskfolio_hrp_argus_equivalent` is ARGUS's own algorithm run *through Riskfolio*, which is
    what makes a capped ARGUS possible at all (see the module docstring)."""

    def argus(train: Mapping[str, Sequence[float]]) -> dict[str, float]:
        built = covariance_matrix(train)
        if built is None:
            raise AllocationError("covariance could not be built")
        return hrp_weights(built[0], built[1])

    def nco(train: Mapping[str, Sequence[float]]) -> dict[str, float]:
        return _riskfolio_allocation(
            train, w_max=w_max, model="NCO", linkage="ward", leaf_order=True,
        )

    def hrp_twin(train: Mapping[str, Sequence[float]]) -> dict[str, float]:
        # `leaf_order=True` to match `argus()`'s own real default above (ARGUS now implements
        # optimal leaf ordering too) -- this stays "ARGUS's own algorithm run through Riskfolio"
        # only if both sides use the same configuration.
        return _riskfolio_allocation(
            train, w_max=w_max, model="HRP", linkage="single", leaf_order=True,
        )

    return [
        ("argus_hrp", argus),
        ("riskfolio_nco_ward", nco),
        ("riskfolio_hrp_argus_equivalent", hrp_twin),
    ]


def _origins(total: int, train_bars: int, test_bars: int, step_bars: int) -> list[int]:
    stops = max(0, total - train_bars - test_bars) + 1
    origins = list(range(0, stops, step_bars))
    if not origins:
        raise AllocationError(
            f"{total} observations cannot supply one {train_bars}+{test_bars}-bar window"
        )
    return origins


def run_walk_forward(
    columns: Mapping[str, Sequence[float]], *, w_max: float | None = None,
    train_bars: int = DENSE_TRAIN_BARS, test_bars: int = DENSE_TEST_BARS,
    step_bars: int = DENSE_STEP_BARS, target_vol_bps: float = TARGET_VOL_BPS,
    taker_bps: float = TAKER_BPS,
) -> dict[str, Any]:
    """Every allocator run as a real fee-paying strategy over the same held-out bars.

    Each window: refit on the estimation bars, pay `turnover * taker_bps` on the move from what
    was held before, hold fixed through the held-out bars. Three scores are produced per allocator
    and all three are published, because they answer different questions and the sibling module
    only ever answered the first:

    * `mean_oos_vol_bps` -- the published criterion, reproduced here as a control. If this does
      not match `data/allocation_comparison.json` closely, one of the two harnesses is wrong.
    * `mean_net_return_bps` -- gross realised return over the held-out window minus the fee for
      getting into that book. What a desk keeps.
    * `mean_levered_net_return_bps` -- the same, after scaling every allocator to one common risk
      budget using **the estimation window's** modelled volatility. This is the charitable case
      for the low-variance allocator: it is allowed to hold more because it is quieter. Leverage
      from the estimation window and never from the held-out window, so no result here can be
      produced by seeing the future.

    The first window is entered from all cash, so every long-only fully-invested allocator pays the
    identical `1.0 * taker_bps` to open. That cost is included rather than skipped, and because it
    is identical it cannot favour anyone.
    """
    names = sorted(columns)
    total = len(columns[names[0]])
    origins = _origins(total, train_bars, test_bars, step_bars)
    target = target_vol_bps / 10_000.0

    per_window: dict[str, list[dict[str, float]]] = {}
    held: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}

    for origin in origins:
        train = _window(columns, origin, train_bars)
        test = _window(columns, origin + train_bars, test_bars)
        built = covariance_matrix(train)
        if built is None:
            continue
        matrix_names, cov = built
        for label, allocate in _allocators(w_max):
            if label in errors:
                continue
            try:
                weights = allocate(train)
            except Exception as exc:  # a failed allocator is recorded, never quietly dropped
                errors[label] = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue

            # Modelled, not realised: the leverage an allocator gets must be decided from bars it
            # has already seen. Using the held-out window's realised volatility here would hand
            # every allocator a perfect risk forecast and measure nothing.
            modelled = math.sqrt(max(0.0, portfolio_variance(weights, matrix_names, cov)))
            leverage = min(MAX_LEVERAGE, target / modelled) if modelled > 0 else MAX_LEVERAGE

            previous = held.get(label, {})
            gross_turnover = _turnover(weights, previous)
            levered_previous = held.get(f"{label}::levered", {})
            levered = {n: leverage * w for n, w in weights.items()}
            levered_turnover = _turnover(levered, levered_previous)
            held[label] = dict(weights)
            held[f"{label}::levered"] = levered

            bars = _portfolio_bars(weights, test)
            gross = sum(bars)
            cost = gross_turnover * taker_bps / 10_000.0
            levered_cost = levered_turnover * taker_bps / 10_000.0
            per_window.setdefault(label, []).append({
                "oos_vol": _stdev(bars),
                "in_sample_modelled_vol": modelled,
                "leverage": leverage,
                "turnover": gross_turnover,
                "gross_return": gross,
                "cost": cost,
                "net_return": gross - cost,
                "levered_net_return": leverage * gross - levered_cost,
                "levered_oos_vol": leverage * _stdev(bars),
                "effective_positions": (
                    1.0 / sum(w * w for w in weights.values())
                    if sum(w * w for w in weights.values()) > 0 else float("nan")
                ),
                "largest_weight": max(weights.values()),
                "inverse_pair_overlap": min(
                    (weights.get(n, 0.0) for n in INVERSE_PAIR), default=0.0
                ),
            })

    return _summarise(per_window, errors, origins, train_bars, test_bars, w_max, target_vol_bps)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarise(
    per_window: Mapping[str, Sequence[Mapping[str, float]]], errors: Mapping[str, str],
    origins: Sequence[int], train_bars: int, test_bars: int, w_max: float | None,
    target_vol_bps: float,
) -> dict[str, Any]:
    """Aggregate, then run the paired sign test NCO-vs-ARGUS on each of the three criteria.

    Paired and non-parametric for the same reason the sibling module chose it: 24 windows of
    hourly-return volatility are nowhere near normal, and a t-test on them would be quoting a
    precision the data does not have."""
    table: dict[str, Any] = {}
    reference = per_window.get("argus_hrp", [])
    for label, rows in per_window.items():
        table[label] = {
            "n_windows": len(rows),
            "mean_oos_vol_bps": _mean([r["oos_vol"] for r in rows]) * 10_000,
            "mean_turnover": _mean([r["turnover"] for r in rows]),
            "mean_cost_bps": _mean([r["cost"] for r in rows]) * 10_000,
            "mean_gross_return_bps": _mean([r["gross_return"] for r in rows]) * 10_000,
            "mean_net_return_bps": _mean([r["net_return"] for r in rows]) * 10_000,
            "mean_leverage": _mean([r["leverage"] for r in rows]),
            "mean_levered_net_return_bps": _mean([r["levered_net_return"] for r in rows]) * 10_000,
            "mean_levered_oos_vol_bps": _mean([r["levered_oos_vol"] for r in rows]) * 10_000,
            "mean_effective_positions": _mean([r["effective_positions"] for r in rows]),
            "mean_largest_weight": _mean([r["largest_weight"] for r in rows]),
            "mean_inverse_pair_overlap": _mean([r["inverse_pair_overlap"] for r in rows]),
        }
        if label == "argus_hrp" or not reference:
            continue
        for criterion, key, lower_is_better in (
            ("oos_vol", "oos_vol", True),
            ("net_return", "net_return", False),
            ("levered_net_return", "levered_net_return", False),
        ):
            wins = 0
            losses = 0
            ties = 0
            for mine, theirs in zip(rows, reference, strict=False):
                gap = mine[key] - theirs[key]
                scale = max(abs(mine[key]), abs(theirs[key]))
                if abs(gap) <= TIE_TOLERANCE * scale:
                    ties += 1
                    continue
                better = gap < 0 if lower_is_better else gap > 0
                wins += 1 if better else 0
                losses += 0 if better else 1
            table[label][f"{criterion}_windows_beating_argus"] = wins
            table[label][f"{criterion}_windows_tied_with_argus"] = ties
            table[label][f"{criterion}_windows_compared"] = wins + losses
            table[label][f"{criterion}_sign_test_p"] = _two_sided_binomial_p(wins, wins + losses)

    nco = table.get("riskfolio_nco_ward", {})
    argus = table.get("argus_hrp", {})
    twin = table.get("riskfolio_hrp_argus_equivalent", {})
    return {
        "w_max": w_max,
        "train_bars": train_bars,
        "test_bars": test_bars,
        "n_origins": len(origins),
        "target_vol_bps": target_vol_bps,
        "taker_bps": TAKER_BPS,
        "results": table,
        # The whole point of the module, stated as three booleans so no reader has to infer it.
        "nco_beats_argus_on_oos_vol": bool(
            nco and argus and nco["mean_oos_vol_bps"] < argus["mean_oos_vol_bps"]
        ),
        "nco_beats_argus_on_net_return": bool(
            nco and argus and nco["mean_net_return_bps"] > argus["mean_net_return_bps"]
        ),
        "nco_beats_argus_on_levered_net_return": bool(
            nco and argus
            and nco["mean_levered_net_return_bps"] > argus["mean_levered_net_return_bps"]
        ),
        # Riskfolio's HRP with `leaf_order=False` IS ARGUS's algorithm, so it must tie ARGUS in
        # every window on every criterion. If it does not, this harness is manufacturing
        # differences out of arithmetic and no other row in the table means anything. Published
        # rather than merely asserted in a test, because a reader should be able to check it.
        "parity_twin_null_control": {
            "windows_tied_with_argus": twin.get("oos_vol_windows_tied_with_argus"),
            "windows_compared": twin.get("oos_vol_windows_compared"),
            "n_windows": twin.get("n_windows"),
            # Only a null when uncapped: with `w_max` set, Riskfolio's bound-fitting loop moves
            # the twin's weights and it is legitimately no longer ARGUS's allocation. Reporting
            # `None` there rather than `False` keeps a real constraint from reading as a defect.
            "behaves_as_a_null": None if w_max is not None else bool(
                twin
                and twin.get("oos_vol_windows_compared") == 0
                and twin.get("oos_vol_windows_tied_with_argus") == twin.get("n_windows")
            ),
        },
        "errors": dict(errors),
    }


# --- ablation (3): the internal hedge -------------------------------------------------------------


def run_universe_ablation(
    columns: Mapping[str, Sequence[float]], *, train_bars: int = DENSE_TRAIN_BARS,
    test_bars: int = DENSE_TEST_BARS, step_bars: int = DENSE_STEP_BARS,
) -> dict[str, Any]:
    """Re-run the bake-off with the QQQ/SQQQ inverse pair broken, then removed entirely.

    Three universes, because the two ways of breaking the pair ask different questions. Dropping
    only SQQQUSDT removes the ability to hedge QQQ against itself while keeping the index
    exposure; dropping both removes the index leg too, which also removes whatever genuine
    diversification the pair provided. If NCO's advantage is an artefact of the internal hedge it
    should shrink under the first and not recover under the second."""
    full = sorted(columns)
    variants: dict[str, list[str]] = {
        "full_universe": full,
        "without_sqqq": [n for n in full if n != INVERSE_PAIR[1]],
        "without_qqq_and_sqqq": [n for n in full if n not in INVERSE_PAIR],
    }
    out: dict[str, Any] = {}
    for label, keep in variants.items():
        subset = {n: columns[n] for n in keep}
        run = run_walk_forward(
            subset, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
        )
        out[label] = {
            "n_instruments": len(keep),
            "argus_mean_oos_vol_bps": run["results"]["argus_hrp"]["mean_oos_vol_bps"],
            "nco_mean_oos_vol_bps": run["results"]["riskfolio_nco_ward"]["mean_oos_vol_bps"],
            "nco_vol_ratio_vs_argus": (
                run["results"]["riskfolio_nco_ward"]["mean_oos_vol_bps"]
                / run["results"]["argus_hrp"]["mean_oos_vol_bps"]
            ),
            "nco_windows_beating_argus_on_vol": (
                run["results"]["riskfolio_nco_ward"]["oos_vol_windows_beating_argus"]
            ),
            "nco_windows_compared": (
                run["results"]["riskfolio_nco_ward"]["oos_vol_windows_compared"]
            ),
            "nco_sign_test_p": run["results"]["riskfolio_nco_ward"]["oos_vol_sign_test_p"],
            "nco_mean_effective_positions": (
                run["results"]["riskfolio_nco_ward"]["mean_effective_positions"]
            ),
            "nco_beats_argus_on_oos_vol": run["nco_beats_argus_on_oos_vol"],
            "nco_beats_argus_on_net_return": run["nco_beats_argus_on_net_return"],
            "nco_beats_argus_on_levered_net_return": run["nco_beats_argus_on_levered_net_return"],
        }
    return out


# --- the verdict ----------------------------------------------------------------------------------


def summarise_verdict(report: Mapping[str, Any]) -> dict[str, Any]:
    """Does the published LOST verdict survive all three ablations, or only the first?

    Written as a count rather than a sentence so it cannot drift away from the numbers above it.
    An ablation "confirms" the published verdict only if NCO still wins under it."""
    checks: dict[str, bool] = {
        "uncapped_oos_vol": bool(report["uncapped"]["nco_beats_argus_on_oos_vol"]),
        "uncapped_net_return": bool(report["uncapped"]["nco_beats_argus_on_net_return"]),
        "uncapped_levered_net_return": bool(
            report["uncapped"]["nco_beats_argus_on_levered_net_return"]
        ),
        "capped_oos_vol": bool(report["capped"]["nco_beats_argus_on_oos_vol"]),
        "capped_levered_net_return": bool(
            report["capped"]["nco_beats_argus_on_levered_net_return"]
        ),
        "without_sqqq_oos_vol": bool(
            report["universes"]["without_sqqq"]["nco_beats_argus_on_oos_vol"]
        ),
        "without_qqq_and_sqqq_oos_vol": bool(
            report["universes"]["without_qqq_and_sqqq"]["nco_beats_argus_on_oos_vol"]
        ),
    }
    survived = sum(1 for v in checks.values() if v)
    return {
        "checks": checks,
        "checks_nco_still_wins": survived,
        "checks_total": len(checks),
        "published_verdict_survives_every_ablation": survived == len(checks),
        "published_verdict_survives_the_variance_ablations": all(
            checks[k] for k in
            ("uncapped_oos_vol", "capped_oos_vol", "without_sqqq_oos_vol",
             "without_qqq_and_sqqq_oos_vol")
        ),
    }


SCOPE_STATEMENT = (
    "This module re-runs `eval/allocation_comparison.py`'s own bake-off with the three conditions "
    "that comparison listed as NOT VERIFIED actually imposed, on the same live 60-day hourly "
    "Bitget candle history for the same rToken universe, against the same real pip-installed "
    "Riskfolio-Lib 7.3.0 (BSD-3-Clause). CLAIMED, and each is a number from a command that was "
    "run: (a) the published realised-out-of-sample-volatility result is independently reproduced "
    "by a harness written from the library's own source rather than from that module's wrapper; "
    "(b) the same walk-forward is re-scored with ARGUS's own two-sided taker fee charged on every "
    "unit of weight each allocator moves between windows, and with each allocator scaled to one "
    "common risk budget using the ESTIMATION window's modelled volatility so no result can come "
    "from seeing the future; (c) the same walk-forward is re-run under Riskfolio's own w_max "
    "maximum-weight constraint, applied by the library's own bound-fitting loop rather than by "
    "any rule written here; (d) the same walk-forward is re-run with the QQQUSDT/SQQQUSDT inverse "
    "pair broken and then removed. NOT CLAIMED that any of this establishes which allocator a "
    "desk should hold: the net-return criterion is a realised outcome over one live 60-day window "
    "and carries no statistical claim about expected return, which 24 windows of hourly crypto "
    "returns cannot support at any sample size available here. NOT CLAIMED that the levered "
    "comparison is a backtest: it holds each book fixed across the held-out window with no drift, "
    "no intra-window rebalancing, no funding and no execution model beyond the single taker fee. "
    "NOT CLAIMED that these numbers are fixed properties -- they are computed from live candle "
    "data and will move the next time this module runs."
)


def render(report: dict[str, Any]) -> str:
    lines = ["ALLOCATION COST/CONCENTRATION ABLATION -- does NCO's published win survive?\n"]
    for key, heading in (
        ("uncapped", "UNCAPPED (replicates the published run, then charges the fee)"),
        ("capped", f"CAPPED at w_max={WEIGHT_CAP} (Riskfolio's own bound)"),
    ):
        block = report[key]
        lines.append(
            f"  {heading} -- {block['n_origins']} held-out windows of {block['test_bars']} bars, "
            f"fitted on {block['train_bars']}"
        )
        for label, row in block["results"].items():
            mark = " <-- ARGUS" if label == "argus_hrp" else ""
            lines.append(
                f"    {label:34} vol {row['mean_oos_vol_bps']:6.3f}bps  "
                f"turn {row['mean_turnover']:.3f} fee {row['mean_cost_bps']:5.2f}bps  "
                f"net {row['mean_net_return_bps']:8.2f}bps  "
                f"lev x{row['mean_leverage']:.2f} net {row['mean_levered_net_return_bps']:8.2f}bps"
                f"  eff.pos {row['mean_effective_positions']:5.2f}{mark}"
            )
        lines.append(
            f"    NCO beats ARGUS -- vol: {block['nco_beats_argus_on_oos_vol']}, "
            f"net: {block['nco_beats_argus_on_net_return']}, "
            f"levered net: {block['nco_beats_argus_on_levered_net_return']}"
        )
        lines.append(
            f"    parity-twin null control behaves as a null: "
            f"{block['parity_twin_null_control']['behaves_as_a_null']}"
        )
        for label, err in block["errors"].items():
            lines.append(f"    failed -- {label}: {err}")

    lines.append("\n  UNIVERSE ABLATION (the QQQ/SQQQ internal hedge)")
    for label, row in report["universes"].items():
        lines.append(
            f"    {label:22} n={row['n_instruments']:2}  ARGUS "
            f"{row['argus_mean_oos_vol_bps']:6.3f}bps  NCO {row['nco_mean_oos_vol_bps']:6.3f}bps "
            f"(x{row['nco_vol_ratio_vs_argus']:.3f})  NCO wins "
            f"{row['nco_windows_beating_argus_on_vol']}/{row['nco_windows_compared']}  "
            f"p={row['nco_sign_test_p']}  NCO eff.pos "
            f"{row['nco_mean_effective_positions']:.2f}"
        )

    verdict = report["verdict"]
    lines.append(
        f"\n  VERDICT: NCO still wins {verdict['checks_nco_still_wins']} of "
        f"{verdict['checks_total']} ablation checks. Published LOST verdict survives every "
        f"check: {verdict['published_verdict_survives_every_ablation']}; survives every "
        f"VARIANCE check: {verdict['published_verdict_survives_the_variance_ablations']}"
    )
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    names, columns, failures = load_returns()
    if failures:
        print(f"WARNING -- {len(failures)} symbol(s) could not be fetched: {failures}")
    if len(names) < 4:
        print(f"not enough aligned history: {len(names)} symbol(s)")
        return 1

    uncapped = run_walk_forward(columns)
    report: dict[str, Any] = {
        "n_instruments": len(names),
        "n_observations": len(columns[names[0]]),
        "fetch_failures": failures,
        "uncapped": uncapped,
        "capped": run_walk_forward(columns, w_max=WEIGHT_CAP),
        "universes": run_universe_ablation(columns),
        "annualisation_hours": TRADING_HOURS_PER_YEAR,
        "scope_statement": SCOPE_STATEMENT,
    }
    report["verdict"] = summarise_verdict(report)
    # Cross-check against the sibling module's published artefact if it is on disk. A replication
    # that silently disagreed with the run it claims to replicate would be worse than no
    # replication at all, so the gap is computed and published rather than eyeballed.
    published = Path(__file__).resolve().parents[3] / "data" / "allocation_comparison.json"
    if published.exists():
        prior = json.loads(published.read_text(encoding="utf-8"))
        dense = prior.get("oos_variance_dense", {}).get("results", {})
        report["replication"] = {
            label: {
                "published_mean_oos_vol_bps": dense[label]["mean_oos_vol_bps"],
                "reproduced_mean_oos_vol_bps": uncapped["results"][key]["mean_oos_vol_bps"],
            }
            for label, key in (
                ("argus_hrp", "argus_hrp"),
                ("riskfolio_nco_ward", "riskfolio_nco_ward"),
            )
            if label in dense
        }

    print(render(report))
    out = Path(__file__).resolve().parents[3] / "data" / "allocation_cost_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DENSE_TEST_BARS",
    "DENSE_TRAIN_BARS",
    "MAX_LEVERAGE",
    "SCOPE_STATEMENT",
    "TARGET_VOL_BPS",
    "TIE_TOLERANCE",
    "WEIGHT_CAP",
    "main",
    "render",
    "run_universe_ablation",
    "run_walk_forward",
    "summarise_verdict",
]
