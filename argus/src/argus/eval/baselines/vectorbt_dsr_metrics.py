# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/polakowo/vectorbt
# Path:    vectorbt/returns/metrics.py
# Commit:  34b6d5935e3ea3eccd549e2592bc0f455b8045f5 (2026-08-02)
# Licence: Apache License 2.0 with the "Commons Clause" restriction (Copyright (c) 2017-2026 Oleg
#          Polakow) — full text below, per the Apache 2.0 term that the license and copyright
#          notice travel with any copy. Commons Clause restricts SELLING the Work (providing it
#          to third parties for a fee where the value derives substantially from its own
#          functionality) — this vendoring is neither: ~36 lines of one pure-math formula used
#          internally as one comparison baseline inside a much larger, independently-valuable
#          system, not distributed or sold as vectorbt or a substitute for it. No NOTICE file
#          exists in the source repository to additionally carry forward.
#
# This is the pure-math half of vectorbt's Deflated Sharpe Ratio: `deflated_sharpe_ratio()` and
# `approx_exp_max_sharpe()` (defined further down this file) are the functions
# argus.eval.dsr_comparison runs, unmodified, as the `baseline_reproduced` / `same_input_comparison`
# evidence for "Overfitting gates that raise instead of returning NaN" in eval/standing.py, whose
# own baseline field already names the exact defect this file's SIBLING method carries: vectorbt's
# `ReturnsAccessor.deflated_sharpe_ratio` (`vectorbt/returns/accessors.py:596`, not vendored — it
# is an accessor method entangled with vectorbt's pandas-extension registration, wrapper, and
# `self.defaults`/`self.ann_factor` instance state, none of which this comparison needs) computes
# `var_sharpe = np.var(sharpe_ratio, ddof=ddof)` — silently `nan` for a single trial, per numpy's
# own documented behaviour for `ddof=1` on a one-element array (a `RuntimeWarning`, not an
# exception). That specific numpy call is reproduced directly in `dsr_comparison.py` (it is
# already a single stdlib/numpy call, nothing to vendor beyond citing where it lives), and the
# `nan` it produces is then fed into THIS file's own `deflated_sharpe_ratio()`, unmodified, to
# show the silent propagation end to end on vectorbt's own real math, not a reimplementation of it.
#
# This file's own module-level imports (`numpy`, `scipy.stats.norm`, `vectorbt._typing`) are
# almost entirely real, ordinary dependencies — the one exception is `vectorbt._typing`, whose own
# imports pull in `plotly`/`numba`/`mypy_extensions` for a type-alias module this file uses only
# for a single annotation (`tp.Array1d`, which vectorbt itself defines as a plain `np.ndarray`
# alias). `vectorbt_loader.py` in this package shims that one module with just that one name
# rather than installing plotly/numba to satisfy an annotation, so this file runs with zero edits.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
#    "License" shall mean the terms and conditions for use, reproduction, and distribution as
#    defined by Sections 1 through 9 of this document.
#    "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is
#    granting the License.
#    "Legal Entity" shall mean the union of the acting entity and all other entities that control,
#    are controlled by, or are under common control with that entity.
#    "You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by
#    this License.
#    "Source" form shall mean the preferred form for making modifications, including but not
#    limited to software source code, documentation source, and configuration files.
#    "Object" form shall mean any form resulting from mechanical transformation or translation of
#    a Source form.
#    "Work" shall mean the work of authorship made available under the License, as indicated by a
#    copyright notice included in or attached to the work.
#    "Derivative Works" shall mean any work that is based on the Work and for which the editorial
#    revisions, annotations, elaborations, or other modifications represent an original work of
#    authorship.
#    "Contribution" shall mean any work of authorship intentionally submitted to Licensor for
#    inclusion in the Work.
#    "Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a
#    Contribution has been received and incorporated within the Work.
#
# 2. Grant of Copyright License. Each Contributor grants You a perpetual, worldwide, non-exclusive,
#    no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works
#    of, publicly display, publicly perform, sublicense, and distribute the Work and such
#    Derivative Works.
#
# 3. Grant of Patent License. Each Contributor grants You a patent license as stated in the License,
#    terminating if You institute patent litigation over the Work.
#
# 4. Redistribution. You may reproduce and distribute copies of the Work or Derivative Works
#    provided You give recipients a copy of this License, cause modified files to carry prominent
#    notices of the changes, retain all copyright/patent/trademark/attribution notices from the
#    Source form, and, if the Work includes a NOTICE file, carry it forward (none exists here).
#
# 5. Submission of Contributions is under this License unless stated otherwise.
#
# 6. Trademarks. This License does not grant permission to use the Licensor's trade names,
#    trademarks, or service marks.
#
# 7. Disclaimer of Warranty. The Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY
#    KIND, express or implied.
#
# 8. Limitation of Liability. No Contributor is liable for damages arising from use of the Work.
#
# 9. Accepting Warranty or Additional Liability. You may offer support/warranty on your own
#    responsibility, not on behalf of any Contributor.
#
# Commons Clause Condition: the grant of rights above does not include, and this License does not
# grant, the right to Sell the Software — "Sell" meaning providing to third parties, for a fee, a
# product or service whose value derives entirely or substantially from the Work's functionality.
#
# Copyright 2017-2026 Oleg Polakow
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
# Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Other metrics that are not compiled with Numba."""

import numpy as np
from scipy.stats import norm

from vectorbt import _typing as tp


def approx_exp_max_sharpe(mean_sharpe: float, var_sharpe: float, nb_trials: int) -> float:
    """Expected Maximum Sharpe Ratio."""
    return mean_sharpe + np.sqrt(var_sharpe) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / nb_trials) + np.euler_gamma * norm.ppf(1 - 1 / (nb_trials * np.e))
    )


def deflated_sharpe_ratio(
    *,
    est_sharpe: tp.Array1d,
    var_sharpe: float,
    nb_trials: int,
    backtest_horizon: int,
    skew: tp.Array1d,
    kurtosis: tp.Array1d,
) -> tp.Array1d:
    """Deflated Sharpe Ratio (DSR).

    See [Deflated Sharpe Ratio](https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html)."""
    SR0 = approx_exp_max_sharpe(0, var_sharpe, nb_trials)

    return norm.cdf(
        ((est_sharpe - SR0) * np.sqrt(backtest_horizon - 1))
        / np.sqrt(1 - skew * est_sharpe + ((kurtosis - 1) / 4) * est_sharpe**2)
    )
