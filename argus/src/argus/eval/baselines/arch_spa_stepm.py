# mypy: ignore-errors
# ruff: noqa
#
# Vendored EXCERPT of Kevin Sheppard's `arch` package. Every line below a "VERBATIM" marker is a
# byte-exact copy of the named line range of the published PyPI wheel `arch==8.0.0` (tag v8.0.0,
# released 2025-10-21), assembled by copying those ranges rather than retyping them:
#
#   arch/bootstrap/_samplers_python.py   sha256 8a60b203cd165e775fd42ff808c4a5bead34651a49acb23591b205e916191a9e
#   arch/bootstrap/base.py               sha256 97c6e1d8d1f48fc001fcfe229d0b1c1a68a18b32b21ab9dc692c55229b00f869
#   arch/bootstrap/multiple_comparison.py sha256 62aec9258d40c129d883abeb319e91f43a94052ccfa491984a5ef4b79280e887
#   arch/utility/array.py                sha256 de87234cc612ead6315e7bc376302da619008982fe1d2399558aa6e6a5c7a4c0
#
# What is NOT verbatim, and why. `SPA` and `StepM` inherit from arch's `MultipleComparison`, use
# its `StationaryBootstrap` class hierarchy (arch/bootstrap/base.py, ~1,900 lines with a compiled
# sampler) and a docstring-copying metaclass. Vendoring that whole hierarchy to run two classes
# would be most of the package. The STAND-IN section below replaces exactly those pieces with the
# minimum that draws the SAME index sequence for an integer seed: `numpy.random.default_rng(seed)`
# (base.py:399-402), `_get_random_integers` (base.py:63-70, verbatim) and
# `StationaryBootstrap.update_indices` (base.py:1596-1606, verbatim body), iterated the way
# `bootstrap()` (base.py:532-573) and `apply()` iterate. Only the stationary bootstrap and integer
# seeds are supported; anything else raises rather than silently differing.
#
# The stand-in is a claim, and it is checked: `eval/general_factorsafety_comparison.py`'s
# `arch_parity()` runs this excerpt and the REAL installed `arch` package on identical input and
# requires identical p-values, critical values and superior-model sets. That check, not this
# comment, is what makes the excerpt a faithful copy. `tests/test_snooping.py` repeats both checks
# (line-for-line against the installed wheel, and numerically) whenever `arch` 8.0.0 is importable.
#
# Two properties of the real code that this excerpt keeps on purpose, because they are findings:
#   1. `SPA(studentize=True)` — the default — computes a variance per model and never divides by
#      it (`compute` / `_simulate_values` below compare raw means). The flag changes only a label.
#   2. `StepM.compute` in 8.0.0 loops `while better_models and (len(better_models) < self.k)`, so
#      when every model is found superior across two or more rounds it calls `SPA.compute()` on an
#      empty selection and raises (bashtage/arch#862, fixed on main after this release).
#
# Licence (arch's LICENSE.md, retained verbatim as its own terms require):
#
# # License
#
# **Copyright (c) 2017 Kevin Sheppard. All rights reserved.**
#
# Developed by: Kevin Sheppard (<kevin.sheppard@economics.ox.ac.uk>,
# <kevin.k.sheppard@gmail.com>)
# [https://www.kevinsheppard.com](https://www.kevinsheppard.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal with
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimers.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimers in the documentation and/or
# other materials provided with the distribution.
#
# Neither the names of Kevin Sheppard, nor the names of its contributors may be
# used to endorse or promote products derived from this Software without specific
# prior written permission.
#
# **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
# THE SOFTWARE.**

# ---- STAND-IN (ours, not arch's): see the header for what this replaces and why ----------------
import copy
from collections.abc import Hashable, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.random import Generator, RandomState

ArrayLike = ArrayLike2D = BoolArray = Float64Array = Int64Array = Int64Array1D = IntArray = Any
DocStringInheritor = type


class MultipleComparison:
    """Stand-in for arch's abstract base: the attributes SPA/StepM set and `reset`."""

    def __init__(self) -> None:
        self._model = ""
        self._info: dict[str, str] = {}

    def reset(self) -> None:
        self.bootstrap.reset()


class _UnsupportedBootstrap:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("this excerpt vendors the stationary bootstrap only")


CircularBlockBootstrap = MovingBlockBootstrap = _UnsupportedBootstrap


class StationaryBootstrap:
    """The pieces of arch's StationaryBootstrap that SPA/StepM call, for an integer seed."""

    def __init__(self, block_size: int, data: Any, *, seed: int | None = None) -> None:
        if not isinstance(seed, (int, np.integer)):
            raise NotImplementedError("this excerpt reproduces arch's draws for an integer seed")
        self.block_size = block_size
        self._p = 1.0 / block_size
        self._data = np.asarray(data)
        self._num_items = self._data.shape[0]
        self._seed = int(seed)
        self._generator = np.random.default_rng(self._seed)

    def reset(self) -> None:
        self._generator = np.random.default_rng(self._seed)

    def clone(self, data: Any, *, seed: int | None = None) -> "StationaryBootstrap":
        return StationaryBootstrap(self.block_size, data, seed=seed)

    def bootstrap(self, reps: int):
        for _ in range(reps):
            index = self.update_indices()
            yield (self._data[index],), {}

    def apply(self, func: Any, reps: int = 1000) -> Any:
        return np.array([func(pos[0]) for pos, _ in self.bootstrap(reps)])

    def __str__(self) -> str:
        return f"Stationary Bootstrap(block size: {self.block_size})"

# ---- VERBATIM: arch/utility/array.py:124-142 ----------------------------------------------
from pandas import DataFrame, Series  # (import line is ours; the function body is verbatim)
from numpy.typing import NDArray  # (ours)


def ensure2d(
    x: Sequence[float | int] | Sequence[Sequence[float | int]] | ArrayLike,
    name: str,
) -> DataFrame | NDArray:
    if isinstance(x, Series):
        return DataFrame(x)
    elif isinstance(x, DataFrame):
        return x
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return np.asarray([[x]])
        elif x.ndim == 1:
            return x[:, None]
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Variable {name} must be 2d or reshapable to 2d")
    else:
        raise TypeError(f"Variable {name} must be a Series, DataFrame or ndarray.")


# ---- VERBATIM: arch/bootstrap/_samplers_python.py:6-35 ---------------------------------
def stationary_bootstrap_sample_python(
    indices: Int64Array, u: Float64Array, p: float
) -> Int64Array:
    """
    Generate indices for sampling from the stationary bootstrap.

    Parameters
    -------
    indices: ndarray
        Single-dimensional array containing draws from randint with the same
        size as the data in the range of [0,nobs).
    u : ndarray
        Single-dimensional Array of standard uniforms.
    p : float
        Probability that a new block is started in the stationary bootstrap.
        The multiplicative reciprocal of the window length.

    Returns
    -------
    ndarray
        Indices for an iteration of the stationary bootstrap.
    """
    num_items = indices.shape[0]
    for i in range(1, num_items):
        if u[i] > p:
            indices[i] = indices[i - 1] + 1
            if indices[i] == num_items:
                indices[i] = 0

    return indices


stationary_bootstrap_sample = stationary_bootstrap_sample_python  # arch base.py:47-50 falls back to this


# ---- VERBATIM: arch/bootstrap/base.py:63-70 --------------------------------------------
def _get_random_integers(
    prng: Generator | RandomState, upper: int, *, size: int = 1
) -> Int64Array1D:
    if isinstance(prng, Generator):
        return cast("Int64Array1D", prng.integers(upper, size=size, dtype=np.int64))
    else:
        assert isinstance(prng, RandomState)
        return cast("Int64Array1D", prng.randint(upper, size=size, dtype=np.int64))


# ---- VERBATIM: arch/bootstrap/base.py:1596-1606 (StationaryBootstrap.update_indices) ----
class _UpdateIndices:  # (this class line is ours; the method body below is verbatim)
    def update_indices(self) -> Int64Array1D:
        indices = _get_random_integers(
            self._generator, self._num_items, size=self._num_items
        )
        indices = indices.astype(np.int64)
        if isinstance(self._generator, Generator):
            u = self._generator.random(self._num_items)
        else:
            assert isinstance(self._generator, RandomState)
            u = self._generator.random_sample(self._num_items)
        return stationary_bootstrap_sample(indices, u, self._p)


StationaryBootstrap.update_indices = _UpdateIndices.update_indices


# ---- VERBATIM: arch/bootstrap/multiple_comparison.py:355-497 (StepM) -------------------
class StepM(MultipleComparison):
    """
    StepM multiple comparison procedure of Romano and Wolf.

    Parameters
    ----------
    benchmark : {ndarray, Series}
        T element array of benchmark model *losses*
    models : {ndarray, DataFrame}
        T by k element array of alternative model *losses*
    size : float, optional
        Value in (0,1) to use as the test size when implementing the
        comparison. Default value is 0.05.
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap
    studentize : bool, optional
        Flag indicating to studentize loss differentials. Default is True
    nested : bool, optional
        Flag indicating to use a nested bootstrap to compute variances for
        studentization.  Default is False.  Note that this can be slow since
        the procedure requires k extra bootstraps.
    seed : {int, Generator, RandomState}, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Notes
    -----
    The size controls the Family Wise Error Rate (FWER) since this is a
    multiple comparison procedure.  Uses SPA and the consistent selection
    procedure.

    See [1]_ for detail.

    See Also
    --------
    SPA

    References
    ----------
    .. [1] Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as
       formalized data snooping. Econometrica, 73(4), 1237-1282.
    """

    def __init__(
        self,
        benchmark: ArrayLike,
        models: ArrayLike,
        size: float = 0.05,
        block_size: int | None = None,
        reps: int = 1000,
        bootstrap: Literal[
            "stationary", "sb", "circular", "cbb", "moving block", "mbb"
        ] = "stationary",
        studentize: bool = True,
        nested: bool = False,
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.benchmark = ensure2d(benchmark, "benchmark")
        self.models = ensure2d(models, "models")
        self.spa: SPA = SPA(
            benchmark,
            models,
            block_size=block_size,
            reps=reps,
            bootstrap=bootstrap,
            studentize=studentize,
            nested=nested,
            seed=seed,
        )
        self.block_size: int = self.spa.block_size
        self.t: int = self.models.shape[0]
        self.k: int = self.models.shape[1]
        self.reps: int = reps
        self.size: float = size
        self._superior_models: list[int] | None = None
        self.bootstrap: CircularBlockBootstrap = self.spa.bootstrap

        self._model = "StepM"
        if self.spa.studentize:
            method = "bootstrap" if self.spa.nested else "asymptotic"
        else:
            method = "none"
        self._info = {
            "FWER (size)": f"{self.size:0.2f}",
            "studentization": method,
            "bootstrap": str(self.spa.bootstrap),
            "ID": hex(id(self)),
        }

    def compute(self) -> None:
        """
        Compute the set of superior models.
        """
        # 1. Run SPA
        self.spa.compute()
        # 2. If any models superior, store indices, remove and re-run SPA
        better_models = [int(i) for i in self.spa.better_models(self.size)]
        all_better_models = better_models[:]
        # 3. Stop if nothing superior
        while better_models and (len(better_models) < self.k):
            # A. Use Selector to remove better models
            selector = np.ones(self.k, dtype=np.bool_)
            selector[np.array(all_better_models)] = False
            self.spa.subset(selector)
            # B. Rerun
            self.spa.compute()
            better_models = list(self.spa.better_models(self.size))
            all_better_models.extend(better_models)
        # Reset SPA
        selector = np.ones(self.k, dtype=np.bool_)
        self.spa.subset(selector)
        all_better_models.sort()
        self._superior_models = all_better_models

    @property
    def superior_models(self) -> list[int] | Sequence[Hashable]:
        """
        List of the indices or column names of the superior models

        Returns
        -------
        list
            List of superior models.  Contains column indices if models is an
            array or contains column names if models is a DataFrame.
        """
        if self._superior_models is None:
            msg = "compute must be called before accessing superior_models"
            raise RuntimeError(msg)
        if isinstance(self.models, pd.DataFrame):
            return list(self.models.columns[self._superior_models])
        return self._superior_models


# ---- VERBATIM: arch/bootstrap/multiple_comparison.py:500-808 (SPA) --------------------
class SPA(MultipleComparison, metaclass=DocStringInheritor):
    """
    Test of Superior Predictive Ability (SPA) of White and Hansen.

    The SPA is also known as the Reality Check or Bootstrap Data Snooper.

    Parameters
    ----------
    benchmark : {ndarray, Series}
        T element array of benchmark model *losses*
    models : {ndarray, DataFrame}
        T  by k element array of alternative model *losses*
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap
    studentize : bool
        Flag indicating to studentize loss differentials. Default is True
    nested : bool
        Flag indicating to use a nested bootstrap to compute variances for
        studentization.  Default is False.  Note that this can be slow since
        the procedure requires k extra bootstraps.
    seed : {int, Generator, RandomState}, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Notes
    -----
    The three p-value correspond to different re-centering decisions.
        - Upper : Never recenter to all models are relevant to distribution
        - Consistent : Only recenter if closer than a log(log(t)) bound
        - Lower : Never recenter a model if worse than benchmark

    See [1]_ and [2]_ for details.

    See Also
    --------
    StepM

    References
    ----------
    .. [1] Hansen, P. R. (2005). A test for superior predictive ability.
       Journal of Business & Economic Statistics, 23(4), 365-380.
    .. [2] White, H. (2000). A reality check for data snooping. Econometrica,
       68(5), 1097-1126.
    """

    def __init__(
        self,
        benchmark: ArrayLike,
        models: ArrayLike,
        block_size: int | None = None,
        reps: int = 1000,
        bootstrap: Literal[
            "stationary", "sb", "circular", "cbb", "moving block", "mbb"
        ] = "stationary",
        studentize: bool = True,
        nested: bool = False,
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.benchmark = ensure2d(benchmark, "benchmark")
        self.models = ensure2d(models, "models")
        self.reps: int = reps
        if block_size is None:
            self.block_size = int(np.sqrt(benchmark.shape[0]))
        else:
            self.block_size = block_size
        self.studentize: bool = studentize
        self.nested: bool = nested
        self._loss_diff = np.asarray(self.benchmark) - np.asarray(self.models)
        self._loss_diff_var = np.empty(0)
        self.t: int = self._loss_diff.shape[0]
        self.k: int = self._loss_diff.shape[1]
        bootstrap_name = bootstrap.lower().replace(" ", "_")
        if bootstrap_name in ("circular", "cbb"):
            bootstrap_inst = CircularBlockBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        elif bootstrap_name in ("stationary", "sb"):
            bootstrap_inst = StationaryBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        elif bootstrap_name in ("moving_block", "mbb"):
            bootstrap_inst = MovingBlockBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        else:
            raise ValueError(f"Unknown bootstrap: {bootstrap_name}")
        self._seed = seed
        self.bootstrap: CircularBlockBootstrap = bootstrap_inst
        self._pvalues: dict[str, float] = {}
        self._simulated_vals: Float64Array | None = None
        self._selector: BoolArray = np.ones(self.k, dtype=np.bool_)
        self._model = "SPA"
        if self.studentize:
            method = "bootstrap" if self.nested else "asymptotic"
        else:
            method = "none"
        self._info = {
            "studentization": method,
            "bootstrap": str(self.bootstrap),
            "ID": hex(id(self)),
        }

    def reset(self) -> None:
        """
        Reset the bootstrap to its initial state.
        """
        super().reset()
        self._pvalues = {}

    def subset(self, selector: BoolArray) -> None:
        """
        Sets a list of active models to run the SPA on.  Primarily for
        internal use.

        Parameters
        ----------
        selector : ndarray
            Boolean array indicating which columns to use when computing the
            p-values.  This is primarily for use by StepM.
        """
        self._selector = selector

    def compute(self) -> None:
        """
        Compute the bootstrap pvalue.

        Notes
        -----
        Must be called before accessing the pvalue.
        """
        # Plan
        # 1. Compute variances
        if self._simulated_vals is None:
            self._simulate_values()
        simulated_vals = self._simulated_vals
        # Use subset if needed
        assert simulated_vals is not None
        simulated_vals = simulated_vals[self._selector, :, :]
        max_simulated_vals = np.max(simulated_vals, 0)
        loss_diff = self._loss_diff[:, self._selector]

        max_loss_diff = np.max(loss_diff.mean(axis=0))
        pvalues = (max_simulated_vals > max_loss_diff).mean(axis=0)
        self._pvalues = {
            "lower": pvalues[0],
            "consistent": pvalues[1],
            "upper": pvalues[2],
        }

    def _simulate_values(self) -> None:
        self._compute_variance()
        # 2. Compute invalid columns using criteria for consistent
        self._valid_columns = self._check_column_validity()
        # 3. Compute simulated values
        # Upper always re-centers
        upper_mean = self._loss_diff.mean(0)
        consistent_mean = upper_mean.copy()
        consistent_mean[np.logical_not(self._valid_columns)] = 0.0
        lower_mean = upper_mean.copy()
        # Lower does not re-center those that are worse
        lower_mean[lower_mean < 0] = 0.0
        means = [lower_mean, consistent_mean, upper_mean]
        simulated_vals = np.zeros((self.k, self.reps, 3))
        for i, bs_data in enumerate(self.bootstrap.bootstrap(self.reps)):
            pos_arg, _ = bs_data
            loss_diff_star = pos_arg[0]
            for j, mean in enumerate(means):
                simulated_vals[:, i, j] = loss_diff_star.mean(0) - mean
        self._simulated_vals = simulated_vals

    def _compute_variance(self) -> None:
        """
        Estimates the variance of the loss differentials

        Returns
        -------
        var : ndarray
            Array containing the variances of each loss differential
        """
        ld = self._loss_diff
        demeaned = ld - ld.mean(axis=0)
        if self.nested:
            # Use bootstrap to estimate variances
            bs = self.bootstrap.clone(demeaned, seed=copy.deepcopy(self._seed))
            means = bs.apply(lambda x: x.mean(0), reps=self.reps)
            variances = self.t * means.var(axis=0)
        else:
            t = self.t
            p = 1.0 / self.block_size
            variances = np.sum(demeaned**2, 0) / t
            for i in range(1, t):
                kappa = ((1.0 - (i / t)) * ((1 - p) ** i)) + (
                    (i / t) * ((1 - p) ** (t - i))
                )
                variances += (
                    2 * kappa * np.sum(demeaned[: (t - i), :] * demeaned[i:, :], 0) / t
                )
        self._loss_diff_var = cast("np.ndarray", variances)

    def _check_column_validity(self) -> BoolArray:
        """
        Checks whether the loss from the model is too low relative to its mean
        to be asymptotically relevant.

        Returns
        -------
        valid : ndarray
            Boolean array indicating columns relevant for consistent p-value
            calculation
        """
        t, variances = self.t, self._loss_diff_var
        mean_loss_diff = self._loss_diff.mean(0)
        threshold = -1.0 * np.sqrt((variances / t) * 2 * np.log(np.log(t)))
        return mean_loss_diff >= threshold

    @property
    def pvalues(self) -> pd.Series:
        """
        P-values corresponding to the lower, consistent and
        upper p-values.

        Returns
        -------
        pvals : Series
            Three p-values corresponding to the lower bound, the consistent
            estimator, and the upper bound.
        """
        self._check_compute()
        return pd.Series(list(self._pvalues.values()), index=list(self._pvalues.keys()))

    def critical_values(self, pvalue: float = 0.05) -> pd.Series:
        """
        Returns data-dependent critical values

        Parameters
        ----------
        pvalue : float, optional
            P-value in (0,1) to use when computing the critical values.

        Returns
        -------
        crit_vals : Series
            Series containing critical values for the lower, consistent and
            upper methodologies
        """
        self._check_compute()
        if not (0.0 < pvalue < 1.0):
            raise ValueError("pvalue must be in (0,1)")
        # Subset if needed
        assert self._simulated_vals is not None
        simulated_values = self._simulated_vals[self._selector, :, :]
        max_simulated_values = np.max(simulated_values, axis=0)
        crit_vals = np.percentile(max_simulated_values, 100.0 * (1 - pvalue), axis=0)
        return pd.Series(crit_vals, index=list(self._pvalues.keys()))

    def better_models(
        self,
        pvalue: float = 0.05,
        pvalue_type: Literal["lower", "consistent", "upper"] = "consistent",
    ) -> Int64Array1D:
        """
        Returns set of models rejected as being equal-or-worse than the
        benchmark

        Parameters
        ----------
        pvalue : float, optional
            P-value in (0,1) to use when computing superior models
        pvalue_type : str, optional
            String in 'lower', 'consistent', or 'upper' indicating which
            critical value to use.

        Returns
        -------
        indices : list
            List of column names or indices of the superior models.  Column
            names are returned if models is a DataFrame.

        Notes
        -----
        List of superior models returned is always with respect to the initial
        set of models, even when using subset().
        """
        self._check_compute()
        if pvalue_type not in self._pvalues:
            raise ValueError("Unknown pvalue type")
        crit_val = self.critical_values(pvalue=pvalue)[pvalue_type]
        better_models = self._loss_diff.mean(0) > crit_val
        better_models = np.logical_and(better_models, self._selector)
        return np.argwhere(better_models).flatten()

    def _check_compute(self) -> None:
        if self._pvalues:
            return
        msg = "compute must be called before pvalues are available."
        raise RuntimeError(msg)

