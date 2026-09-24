# Effective number of bets and minimum-variance hedging — what the field ships

**Written 2026-09-13.** Track 3's Open Theme names a portfolio copilot showing "how a proposed
trade changes beta, sector and factor exposure, correlation, concentration, with stress tests and
**hedge suggestions**". Two items on that list were ones ARGUS could not do, and its own docstring
said so:

> *"Three perfectly correlated positions score 2.71 here, not 1 ... A genuine correlation-aware
> count — Meucci's effective number of bets — needs a principal-component decomposition of the
> covariance matrix, which is **NOT BUILT**."* — `desk/portfolio.py`, `RiskDecomposition`

This is the teardown written before building `desk/diversification.py`.

**A caveat about this document's sourcing.** A parallel reader dispatched to survey the corpus returned a
report whose "best local implementation" for both measures was `argus/desk/diversification.py` —
the file being written at that moment. That is circular and none of its praise is evidence. What
survives is the **external** findings, which are checkable and are what this teardown records.

Trees searched: `research/repos*/`, `research/corpus/repos/`,
`okx/trading/best-of-the-best/repos/`, plus the installed `numpy`, `scipy`, `PyPortfolioOpt`,
`riskfolio-lib` and `cvxportfolio` packages. Patterns: `effective number of bets`, `effective
bets`, `diversification distribution`, `Meucci`, `torsion`, `minimum torsion`, `principal
component`, `eigendecomposition`, `eigh`, `PCA`, `marginal contribution to risk`, `minimum variance
hedge`, `hedge ratio`, `optimal hedge`, `beta hedge`, `cross hedge`.

---

## What the field actually has

| System | Effective bets | Hedge sizing | File:line |
|---|---|---|---|
| **Riskfolio-Lib** | "NEA", and it is **not Meucci's measure** | none | `Portfolio.py:82-85`, constraint at `:3041-3045` |
| **PyPortfolioOpt** | absent | absent | hierarchical risk parity clusters, no entropy measure |
| **mlfinlab** | absent | absent | zero grep matches |
| **qlib** | absent | absent | zero grep matches |
| **Hummingbot** | absent | **fixed ratio from config** | `hedge.py:33-40`, `:77` |
| **Auto-Quant-V2** | tracked as a KPI, never computed | absent | `book_risk_explorer.py:62, 775, 1419` |
| **QuantConnect Lean** | absent | none in the mean-variance optimiser | — |

**The Riskfolio finding is the useful one, because it is the same mistake we had.** Its `NEA` is
the inverse Herfindahl of the weight vector — the docstring says "number of effective assets (NEA)
… inverse of the Herfindahl-Hirschman index" — which measures how evenly *weight* is spread and is
blind to correlation entirely. A published, widely used library ships the measure ARGUS's own
docstring already refused to call Meucci's. Two independent implementations reaching for the same
wrong number is the reason this was worth building properly rather than borrowing.

**Nothing in any tree computes a hedge size from a covariance.** Hummingbot's hedge strategy is the
only thing that hedges at all and it reads the ratio from a config file, which is an application
layer rather than a sizing method.

---

## The two measures, and why both are needed

### Effective number of bets

Rotate the book into uncorrelated directions, take each direction's share of variance, and report
the exponential of the entropy of that distribution:

```
p_i  = (loading_i)^2 * variance_i / sum_j (...)
bets = exp(-sum_i p_i log p_i)
```

Bounds are exact and were used as the acceptance tests: N independent equally-weighted positions
score exactly N, and N perfectly correlated ones score exactly 1 — where the inverse Herfindahl
scores 2.71 for three of them.

**Which rotation, though. This is unsettled in the literature and it matters more than the
literature suggests.** Both variants were implemented and both were measured on an equally-weighted
equicorrelated book, which is the shape twelve rTokens driven by one market produce:

| correlation | principal components | minimum torsion |
|---|---|---|
| 0.0 | 4.00 | 4.00 |
| **0.0001** | **1.00** | 4.00 |
| 0.3 | 1.00 | 4.00 |
| 0.9 | 1.00 | 4.00 |
| 0.99 | 1.00 | 4.00 |

The PCA reading collapses at any positive correlation because the equally-weighted vector *is* an
eigenvector of an equicorrelation matrix, so the book loads on one direction and nothing else. The
torsion reading never moves, because its factors are the assets themselves decorrelated and an
equal book has equal exposure to each by construction. **Each is blind exactly where the other is
informative**, and shipping either alone would be a plausible number that cannot see the thing it
claims to measure.

On an unequally-weighted book — the realistic case — both behave, and in opposite directions:

| correlation | PCA | torsion |
|---|---|---|
| 0.0 | 2.94 | 2.94 |
| 0.3 | 1.37 | 3.50 |
| 0.9 | 1.04 | 3.96 |

They agree exactly at zero correlation, which is the sanity check that says both are rotations of
the same book. **The decision taken: report both, use the conservative one for the warning, and
treat the gap between them as the signal that the book is correlated.**

### Minimum-variance hedge

For a book `p` and a candidate `h`, the size minimising `var(p + r*h)` is `r = -cov(p,h)/var(h)`,
and the variance then falls by exactly the squared correlation. Both come from one covariance, so
the promised reduction is implied by the recommendation rather than estimated beside it — there is
no room for the two to disagree, which is the property that makes a hedge suggestion honest.

Sample covariance, not shrunk. Shrinkage biases toward the diagonal, which understates a real
hedge's benefit; it is the right tool for in-sample optimisation and the wrong one for sizing a
hedge that has to work live.

---

## Numerical choices, and the traps behind each

* **Eigensolver: cyclic Jacobi, pure Python.** This package carries no numeric dependency. Jacobi
  is not a compromise for small symmetric matrices — it is backward-stable, needs no pivoting and
  converges quadratically. Verified against `numpy.linalg.eigvalsh` on random symmetric matrices at
  n = 2, 3, 5 and 8: **worst deviation 3.55e-15**, which is machine precision.
* **Asymmetric input raises rather than being symmetrised.** An asymmetric covariance matrix is a
  transposition bug upstream, and averaging the halves hides it behind a plausible answer.
* **Near-zero eigenvalues are treated as zero, not inverted or logged.** The log of a tiny variance
  share is a large negative number that dominates the entropy — a degenerate book would read as
  maximally diversified. Riskfolio instead repairs non-positive-definite matrices by clipping
  eigenvalues (`risk_models.py:57-113`), which silently hides a singular covariance.
* **A singular correlation matrix makes the torsion undefined and it says so.** Perfect correlation
  between two names means no invertible rotation exists. The PCA reading still works there, so the
  result carries one number and names which — rather than reporting agreement between one measure
  and nothing.
* **An instrument with no variance cannot hedge.** Dividing by its variance gives an infinite size;
  the code refuses instead.

---

## A mistake made and caught in the building

The first working version had both the factor exposure and the factor variance transposed: it used
`t·w` for the exposure where the correct quantity is `(t⁻¹)ᵀw`, since factors `z = t·x` mean a book
`w'x` is `(t⁻¹'w)'z`. The symptom was the opposite of the PCA failure — every equicorrelated book
scored exactly the position count at any correlation. **A measure that is constant in the thing it
measures is wrong in a way that reads as stable**, and it was only caught by testing the
correlation sweep rather than the endpoints.

---

## What is still absent, here and everywhere

Verified by grep across every tree listed above:

* Shrinkage-aware hedge sizing (Ledoit-Wolf covariance for the hedge).
* Risk-parity-aware hedge sizing — capping the hedge so total risk contribution stays balanced.
* Rolling hedge effectiveness: how a ratio drifts over time, and whether it is stable enough to
  trade on.
* Hedge basis stability, for a cross hedge where the instrument is not the thing being hedged.

The third is the one that matters most for this venue and is the natural next build: a hedge ratio
estimated once over 180 days is a number, and a hedge ratio that holds across rolling windows is a
capability.
