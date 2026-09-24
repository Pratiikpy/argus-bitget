# Matrix profile / analogue retrieval — teardown, and what ARGUS does instead

**Sources read, on this machine:**

- `research/repos-themed/TDAmeritrade~stumpy` — cloned 2026-09-14 for this teardown. The reference
  implementation of the matrix profile (Yeh et al., ICDM 2016).
- `research/repos-themed/matrix-profile-foundation~matrixprofile` — the other maintained
  implementation, already in the corpus.
- Our own `argus/src/argus/desk/analogue.py` and `argus/src/argus/desk/stress.py`.

Everything below carries `file:line`. Nothing here is from memory.

---

## 1. What the reference actually computes

`stumpy/core.py:1106-1121`, `_calculate_squared_distance`:

```python
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
```

Three facts fall straight out of that line, and all three matter:

1. **The z-normalised Euclidean distance is a pure function of correlation.** `D² = 2m(1-ρ)`. It
   carries no other information. Anything a shape matcher tells you, correlation told you first.
2. **The scale is therefore fixed and derivable.** Divide by `m` to get a per-bar distance, as ARGUS
   does, and `d = sqrt(2(1-ρ))`: identical shapes 0, uncorrelated **1.414**, perfectly inverted 2.
3. **A flat window against a structured one is `D² = m`**, i.e. exactly 1.0 per bar — a special case
   in stumpy, and in ARGUS a consequence of z-normalising a zero-variance window to zeros
   (`desk/shapematch.py:znormalise`). Pinned by a test rather than assumed
   (`tests/test_shapematch.py::test_a_flat_window_against_a_structured_one_is_one`).

`matrixprofile/core.py:615` states the same relationship from the other direction, converting a
Pearson profile back to Euclidean: `euc_a = np.sqrt(2 * window * (1 - a))`.

**This read found a real defect in our own code.** `WEAK_MATCH_DISTANCE` had been written as `2.0`
with a docstring asserting that uncorrelated shapes "land above 2.0". 2.0 is the *maximum value the
metric can return*. The gate could not fire for any input — a perfectly inverted shape passed it.
It is now `1.0` (`ρ = 0.5`), which is below the uncorrelated level by construction, and a test
asserts `WEAK_MATCH_DISTANCE < sqrt(2)` so the class of error cannot return.

## 2. The exclusion zone, and how wide it should be

| Implementation | Zone | Where |
|---|---|---|
| stumpy | `m / 4` either side | `stumpy/config.py:19` — `"STUMPY_EXCL_ZONE_DENOM": 4` |
| matrixprofile STOMP | `ceil(m / 2)` | `matrixprofile/algorithms/stomp.py:276` |
| matrixprofile SCRIMP | `ceil(m / 4)` | `matrixprofile/algorithms/scrimp.py:312` |
| matrixprofile regimes | `m * 5` | `matrixprofile/algorithms/regimes.py:129` |
| matrixprofile motifs | `floor(m / 2)`, re-applied per neighbour | `algorithms/top_k_motifs.py:109-153` |
| **ARGUS** | **`m`, a full window** | `desk/shapematch.py:_scan` |

Mechanically it is the same idea in all of them: set the distance profile to infinity around an
index you have already taken (`matrixprofile/core.py:apply_exclusion_zone`, `stumpy/core.py:2026-2028`)
so the next `argmin` cannot return a neighbour of it. The width is a judgement call and every
library makes a different one.

ARGUS takes a full window because the output is read by a person, not consumed by a clustering step:
with `m/4`, two "separate" analogues can share three-quarters of their bars, and a top-5 built from
them is one event reported five times with an outcome distribution to match. At `m` no returned
match shares a single bar with the query or with another match. Note also that
`matrixprofile/algorithms/top_k_motifs.py:133-160` bounds neighbours by a *radius* (`radius * min_dist`)
as well as by the zone — a relative gate we deliberately do not copy, because it defines "close" as
a multiple of the best match found, which is circular when the best match is itself noise.

## 3. The thing none of them do, and it is the one that matters

Both libraries stop at a distance. `stumpy.motifs` takes `max_distance` as a caller-supplied number;
`matrixprofile` takes an exclusion zone and a radius. Neither asks **whether a match that close means
anything on this series**, and neither can, because that question is not a property of the matrix
profile — it is a property of the series.

Scanning ~1,400 candidate windows for a minimum produces a small number whether or not the series
has any repeating structure at all. That is the selection effect, and it is why every retrieval demo
can always show you five convincing analogues.

**ARGUS calibrates against the series' own noise.** `desk/shapematch.py:find` reruns the identical
scan (`_scan`, shared code, so the null is not measuring the difference between two implementations)
over `NULL_TRIALS` paths built by randomly reordering the series' own returns
(`shuffled_closes`) — same bars, same volatility, same fat tails, ordering destroyed. The reported
`null_p` is the share of those scans whose best match is at least as close as the observed one, and
`has_precedent` requires `null_p <= 0.05` on top of the distance and count gates. An uncalibrated
report (`trials=0`) never claims a precedent; it says "Uncalibrated" and explains why a distance
alone cannot settle it.

### What the calibration says about our own market

60 days of hourly Bitget rTokens, horizon 24 bars, 50 null scans per cell, seed fixed, measured 2026-09-14 (the live artefact re-measures on each run as history grows, so its p moves by a point or two — this grid is a dated snapshot, not a constant):

| symbol | w=6 | w=12 | w=24 | w=48 |
|---|---|---|---|---|
| NVDAUSDT | **p=0.04** | 0.68 | 0.34 | **p=0.00** |
| TSLAUSDT | 0.92 | 0.96 | 0.36 | 0.58 |
| COINUSDT | 0.86 | 0.82 | 0.46 | 0.16 |
| AAPLUSDT | 0.20 | 0.42 | 0.14 | 0.24 |
| MSTRUSDT | 0.12 | 0.76 | 0.82 | 0.74 |
| GOOGLUSDT | 0.30 | 0.86 | 0.40 | 0.52 |

**22 of 24 cells fail the null.** The live default (NVDAUSDT, 24-bar) finds five non-overlapping
matches at distance 0.371–0.437 and reports **no precedent**, because reordered returns from the
same series beat 0.371 in 36% of 50 scans (median 0.440). The two cells that pass are what 24 tests at
α = 0.05 produce by chance — a reader comparing several window lengths is selecting again, one level
up, and should apply their own correction. The module reports one window per run for exactly that
reason.

The honest conclusion, stated in `data/shape_matches.json` rather than in a slide: on this data, at
this horizon, shape recurrence is not distinguishable from reordered noise. A system that showed the
five matches and their median forward return would have been reporting a coincidence with a decimal
point.

## 4. Why ARGUS keeps two retrievals

`desk/analogue.py` (state vector: trailing return and realised volatility, z-scored across the
corpus, overlapping matches collapsed into episodes before counting) and `desk/shapematch.py` (the
path itself) answer different questions, and `desk/research.py` reports them as separate steps so a
disagreement stays visible. Two windows can carry an identical trailing return and an identical
realised volatility while one is a steady grind and the other a crash that fully retraced; a state
vector has thrown that away, and a trader saying *similar* usually means the ordering.

`desk/stress.py` answers a third question — magnitude, "a 5.1% two-hour fall is the 1st percentile of
718 observed windows" — which is neither of the above and is the right answer to "how bad can it
get".

## 5. What we did not take

- **numpy / numba.** stumpy's speed comes from `njit` kernels and an incremental dot product
  (`stumpy/core.py:_calculate_squared_distance_profile`). ARGUS is pure Python by design; at 1,400
  bars a brute-force scan is ~10⁶ operations and finishes in well under a second, and the null's 50
  extra scans cost ~6s. Buying a dependency to save that would be a bad trade.
- **The relative radius gate** (`top_k_motifs.py:133`) — circular, as above.
- **`max_distance` as a caller argument.** A threshold a caller can set to taste is a threshold that
  will be set to whatever returns results. Ours is derived from the metric's algebra, and the
  binding gate is the measured null rather than the constant.
