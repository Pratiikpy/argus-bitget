# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/TDAmeritrade/stumpy
# Path:    stumpy/core.py, lines 1056-1121 (function `_calculate_squared_distance`)
# Commit:  e4caf8a7ba519d1ba04796cd06aa78d91e9ca6ee (2026-09-12)
# Licence: 3-Clause BSD (Copyright 2019 TD Ameritrade) — full text below, read directly from the
#          repo's own LICENSE.txt. GitHub's own API classifies this repository's license as
#          "Other" (`gh repo view TDAmeritrade/stumpy --json licenseInfo`), almost certainly
#          because `LICENSE.txt` opens with an extra trademark sentence ("STUMPY is a trademark
#          of TD Ameritrade IP Company, Inc. All rights reserved.") before the standard BSD
#          text, which trips GitHub's license auto-detector — the actual terms below are
#          textbook 3-Clause BSD, fully permissive for redistribution with attribution.
#          Confirmed by reading the license file directly, not by trusting the GitHub badge.
#
# This is STUMPY's real matrix-profile squared-distance formula — the reference implementation
# `argus.desk.shapematch`'s own module docstring already cites by file:line
# (`stumpy/core.py:_calculate_squared_distance`) and derives its own distance metric FROM
# algebraically (`d = sqrt(2 * (1 - rho))`), before this vendoring ever existed. That citation
# is now backed by running the real function, not just reading it:
# `argus.eval.shapematch_comparison` calls this real, unmodified `_calculate_squared_distance`
# with the five real scalar inputs (`QT`, `mu_Q`, `sigma_Q`, `M_T`, `Sigma_T`) computed the
# ordinary way (dot product, mean, standard deviation) from the SAME z-normalised windows
# `argus.desk.shapematch.distance()` scores, and confirms `sqrt(D_squared / m)` matches ARGUS's
# own real `distance()` output to floating-point precision on identical input — the decisive
# `baseline_reproduced`/`same_input_comparison` evidence for "Path-shape matching with a
# calibrated null" in eval/standing.py.
#
# `@njit(fastmath=config.STUMPY_FASTMATH_FLAGS)` is the function's own real decorator, run with
# the real `numba` package (already installed) and the real `STUMPY_DENOM_THRESHOLD`/
# `STUMPY_FASTMATH_FLAGS` VALUES from `stumpy/config.py:13,21` (copied as values, not vendored as
# a module, since this excerpt needs only the two constants this one function reads) — see
# `eval/baselines/stumpy_squared_distance_loader.py`'s own header for the loader wiring.
#
# BSD 3-Clause License
#
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ============================== VENDORED FROM HERE ==============================
@njit(
    # "f8(i8, f8, f8, f8, f8, f8)",
    fastmath=config.STUMPY_FASTMATH_FLAGS
)
def _calculate_squared_distance(
    m, QT, μ_Q, σ_Q, M_T, Σ_T, Q_subseq_isconstant, T_subseq_isconstant
):
    """
    Compute a single squared distance given all scalar inputs.

    Parameters
    ----------
    m : int
        Window size

    QT : float
        Pre-computed dot product between `Q` and the ith subsequence in `T`, each with
        length `m`

    μ_Q : float
        Mean of `Q`

    σ_Q : float
        Standard deviation of `Q`

    M_T : float
        Mean of the ith subsequence in `T`

    Σ_T : float
        Standard deviation of the ith subsequence in `T`

    Q_subseq_isconstant : bool
        A boolean value that indicates whether the subsequence `Q` is constant (True)

    T_subseq_isconstant : bool
        A boolean value that indicates whether the ith subsequence in `T` is
        constant (True)

    Returns
    -------
    D_squared : float
        Squared distance

    Notes
    -----
    `DOI: 10.1109/ICDM.2016.0179 \
    <https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf>`__

    See Equation on Page 4
    """
    if np.isinf(M_T) or np.isinf(μ_Q):
        D_squared = np.inf
    elif Q_subseq_isconstant and T_subseq_isconstant:
        D_squared = 0
    elif Q_subseq_isconstant or T_subseq_isconstant:
        D_squared = m
    else:
        denom = (σ_Q * Σ_T) * m
        denom = max(denom, config.STUMPY_DENOM_THRESHOLD)

        ρ = (QT - (μ_Q * M_T) * m) / denom
        ρ = min(ρ, 1.0)

        D_squared = np.abs(2 * m * (1.0 - ρ))

    return D_squared
