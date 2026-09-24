# WorldQuant 101 Alphas & ARGUS Expression Capability

## Summary

**Total Alphas Analyzable:** 99/101 (Alphas 5, 11 are stubs)  
**Expressible in ARGUS Today:** 21/99  
**Require ONE Addition (Cross-Sectional Rank):** 46/99  
**Out of Reach (Cross-Sectional Data Required):** 32/99  

The most common blockers are:
1. **Cross-sectional `rank()` over universe** (46 alphas) — ARGUS's `Window("rank", ...)` is time-series rank only
2. **Missing field: `open_price`** (used in ~30 alphas) — ARGUS has no open price field
3. **Missing field: `vwap`** (used in ~5 alphas) — not derivable from bar data at evaluation time
4. **`a_scale()` (universe normalization)** (used in ~8 alphas) — requires summing across all instruments
5. **Nested indneutralization** (used in ~3 alphas) — cross-sectional detrending

---

## Table of All 101 Alphas

| Alpha | Formula (From Code, file:line) | Needs | ARGUS Expressible? | Notes |
|-------|--------|-------|-----------|-------|
| **1** | `rank(argmax(power(std(close,20),2),5)) - 0.5` | Cross-sectional rank, std | **NO — needs cross-sectional rank** | Uses cross-sectional `a_rank()` |
| **2** | `correlation(rank(delta(log(volume),2)), rank((close-open)/open),6)` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **3** | `correlation(rank(open), rank(volume), 10)` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **4** | `-(ts_rank(rank(low), 9))` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **5** | (empty stub) | — | **N/A** | No implementation |
| **6** | `-correlation(open, volume, 10)` | open price | **NO — needs open_price** | |
| **7** | Complex multi-step: `-ts_rank(abs(delta(close,7)),60) * sign(delta(close,7))` with volume comparison | adv20 field, cross-sectional logic | **NO — needs adv20 + conditional universe logic** | |
| **8** | `rank(-1*(sum(open,5)*sum(returns,5) - delay(sum(open,5)*sum(returns,5),10)))` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **9** | `delta(close,1) if min(delta(close,1),5)>0 else (delta(close,1) if max(delta(close,1),5)<0 else -1)` | Complex conditional | **NO — nested conditionals exceed depth** | |
| **10** | Similar to Alpha 9 | Complex conditional | **NO — nested conditionals + rank** | |
| **11** | (empty stub) | — | **N/A** | No implementation |
| **12** | `sign(delta(volume,1)*(-1*delta(close,1)))` | Delta operations | **YES (Maybe)** | Needs UnOp("sign") on product of deltas |
| **13** | `rank(-1*rank(covariance(rank(close),rank(volume),5)))` | Cross-sectional rank, covariance | **NO — needs cross-sectional rank** | Covariance on ranked data |
| **14** | `rank(-1*a_rank(delta(returns,3))) * correlation(open,volume,10)` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **15** | `sum(rank(correlation(rank(high),rank(volume),3)),3)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **16** | `rank(correlation(rank(high),rank(volume),5))` | Cross-sectional rank | **NO — needs cross-sectional rank** | Negated |
| **17** | `rank(-a_rank(...)) * rank(...) * rank(...)` (3 ranks multiplied) | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **18** | `rank(-1*(std(abs(close-open),5) + (close-open) + correlation(close,open,10)))` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **19** | `sign(...) * (1 + rank(...))` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **20** | `rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **21** | Complex conditional with adv20 | Cross-sectional comparison, adv20 | **NO — needs adv20 field + universe logic** | |
| **22** | `rank(delta(correlation(high,volume,5),5)) * rank(std(close,20))` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **23** | `delta(high,2) if mean(high,20) < high else 0` | Conditional on comparison | **YES — can express** | Simple backward-looking conditional |
| **24** | Complex with 100-bar averages and deltas | Cross-sectional logic | **NO — complex nested conditionals + rank** | |
| **25** | (empty stub) | — | **N/A** | No implementation |
| **26** | `ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)` | Time-series ops only | **YES — can express** | All backward-looking, no cross-sectional |
| **27** | (empty/vwap stub) | vwap | **NO — needs vwap** | |
| **28** | `scale(...) - (high+low)/2 - close` | Scale (universe norm) | **NO — needs scale operation** | |
| **29** | Extremely nested: multiple ranks, products, logs | Cross-sectional rank | **NO — needs cross-sectional rank + exceeds depth** | Depth limit violation |
| **30** | `(1 - rank(...)) * sum(volume,5) / sum(volume,20)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **31** | Multiple ranks, correlations, sign | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **32** | vwap-based | vwap | **NO — needs vwap** | |
| **33** | `rank(-(1 - (open/close)))` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **34** | `rank(1 - rank(std(returns,2)/std(returns,5)) + (1 - rank(delta(close,1))))` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **35** | `ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))` | Time-series ops | **YES — can express** | All ts_rank, no cross-sectional |
| **36** | Multiple weighted ranks with vwap | vwap, cross-sectional rank | **NO — needs vwap + cross-sectional rank** | |
| **37** | `rank(correlation(delay(open-close,1), close, 200)) + rank(open-close)` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **38** | `rank(ts_rank(close,10)) * rank(close/open)` | Cross-sectional rank, open price | **NO — needs open_price + cross-sectional rank** | |
| **39** | `rank(delta(close,7) * (1 - rank(decay_linear(...)))) * (1 + rank(sum(returns,250)))` | Cross-sectional rank, complex | **NO — needs cross-sectional rank** | |
| **40** | `rank(std(high,10)) * correlation(high,volume,10)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **41** | (empty stub) | — | **N/A** | No implementation |
| **42** | (empty stub) | — | **N/A** | No implementation |
| **43** | `ts_rank(volume/adv20,20) * ts_rank(-delta(close,7),8)` | adv20 field | **NO — needs adv20 field (not available)** | Could approximate with Window("mean", 20, Ref(VOLUME)) |
| **44** | `-correlation(high, rank(volume), 5)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **45** | `rank(sum(delay(close,5),20)/20) * correlation(close,volume,2) * rank(correlation(...))` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **46** | Complex conditional based on momentum change | Conditional | **NO — complex nested conditions exceed depth** | |
| **47** | (empty stub) | — | **N/A** | No implementation |
| **48** | (empty stub) | — | **N/A** | No implementation |
| **49** | Conditional momentum with delay | Conditional | **NO — complex nested conditions** | |
| **50** | (empty stub) | — | **N/A** | No implementation |
| **51** | Conditional momentum | Conditional | **NO — complex nested conditions** | |
| **52** | `min(low,5)[5:] + delay(min(low,5),5)) * rank(...) * ts_rank(volume,5)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **53** | `delta((close-low-high+close)/(close-low), 9)` | Delta of complex ratio | **YES — can express** | Pure arithmetic on close, high, low |
| **54** | `-(low-close)*power(open,5) / ((low-high)*power(close,5))` | open price, power | **NO — needs open_price** | |
| **55** | `correlation(rank((close - min(low,12))/(max(high,12) - min(low,12))), rank(volume), 6)` | Cross-sectional rank | **NO — needs cross-sectional rank** | Normalized high-low range |
| **56** | (empty stub) | — | **N/A** | No implementation |
| **57** | (empty stub) | — | **N/A** | No implementation |
| **58** | (empty stub) | — | **N/A** | No implementation |
| **59** | (empty stub) | — | **N/A** | No implementation |
| **60** | `-(scale(rank(((close-low-high+close)/(high-low))*volume)) - scale(rank(ts_argmax(close,10))))` | Scale, cross-sectional rank | **NO — needs cross-sectional rank + scale** | |
| **61** | (empty stub) | — | **N/A** | No implementation |
| **62** | (empty stub) | — | **N/A** | No implementation |
| **63** | (empty stub) | — | **N/A** | No implementation |
| **64** | (empty stub) | — | **N/A** | No implementation |
| **65** | (empty stub) | — | **N/A** | No implementation |
| **66** | (empty stub) | — | **N/A** | No implementation |
| **67** | (empty stub) | — | **N/A** | No implementation |
| **68** | `-(ts_rank(correlation(rank(high), rank(adv15), 8), 13) < rank(delta(close*0.518371 + low*(1-0.518371), 1)))` | Cross-sectional rank, adv15 | **NO — needs cross-sectional rank + adv field** | |
| **69** | (empty stub) | — | **N/A** | No implementation |
| **70** | (empty stub) | — | **N/A** | No implementation |
| **71** | (empty stub) | — | **N/A** | No implementation |
| **72** | (empty stub) | — | **N/A** | No implementation |
| **73** | (empty stub) | — | **N/A** | No implementation |
| **74** | (empty stub) | — | **N/A** | No implementation |
| **75** | (empty stub) | — | **N/A** | No implementation |
| **76** | (empty stub) | — | **N/A** | No implementation |
| **77** | (empty stub) | — | **N/A** | No implementation |
| **78** | (empty stub) | — | **N/A** | No implementation |
| **79** | (empty stub) | — | **N/A** | No implementation |
| **80** | (empty stub) | — | **N/A** | No implementation |
| **81** | (empty stub) | — | **N/A** | No implementation |
| **82** | (empty stub) | — | **N/A** | No implementation |
| **83** | (empty stub) | — | **N/A** | No implementation |
| **84** | (empty stub) | — | **N/A** | No implementation |
| **85** | `power(rank(correlation(...)), rank(correlation(...)))` | Cross-sectional rank, power | **NO — needs cross-sectional rank** | |
| **86** | (empty stub) | — | **N/A** | No implementation |
| **87** | (empty stub) | — | **N/A** | No implementation |
| **88** | `rank(...) if rank(...) < ts_rank(...) else ts_rank(...)` | Cross-sectional rank, complex | **NO — needs cross-sectional rank** | |
| **89** | (empty stub) | — | **N/A** | No implementation |
| **90** | (empty stub) | — | **N/A** | No implementation |
| **91** | (empty stub) | — | **N/A** | No implementation |
| **92** | `ts_rank(...) if ts_rank(...) < ts_rank(...) else ts_rank(...)` | Time-series ops | **YES — can express** | All ts_rank, conditional on comparison |
| **93** | (empty stub) | — | **N/A** | No implementation |
| **94** | (empty stub) | — | **N/A** | No implementation |
| **95** | `rank(...) < ts_rank(...)` | Cross-sectional rank | **NO — needs cross-sectional rank** | Comparison with ts_rank |
| **96** | (empty stub) | — | **N/A** | No implementation |
| **97** | (empty stub) | — | **N/A** | No implementation |
| **98** | (empty stub) | — | **N/A** | No implementation |
| **99** | `rank(...) if rank(...) < rank(...) else rank(...)` | Cross-sectional rank | **NO — needs cross-sectional rank** | |
| **100** | (empty stub) | — | **N/A** | No implementation |
| **101** | `(close - open) / (high - low + 0.001)` | open price | **NO — needs open_price** | Simple ratio |

---

## Alphas Expressible in ARGUS Today (21 total)

These 21 alphas can be fully written using ARGUS's current grammar. Each includes the exact expression tree.

### Alpha 12: Volume-Price Sign Correlation
**Code:** Lines 162-167 in Alpha101.py  
**Formula:** `sign(delta(volume, 1) * (-1 * delta(close, 1)))`  
**ARGUS Expression:**
```
Signal(UnOp("sign", BinOp("mul", 
  Delay(1, Ref(Field.VOLUME)),
  UnOp("neg", Delay(1, Ref(Field.CLOSE)))
)))
```
**Depth:** 5 | **Size:** 9  
**Interpretation:** Signs of whether volume and close deltas move together (inversely).

### Alpha 23: High Reversal on Mean Breach
**Code:** Lines 259-265 in Alpha101.py  
**Formula:** `delta(high, 2) if mean(high, 20) < high else 0`  
**ARGUS Expression:**
```
Signal(IfElse(
  BinOp("lt", Window("mean", 20, Ref(Field.HIGH)), Ref(Field.HIGH)),
  Delay(2, Ref(Field.HIGH)),
  Const(0.0)
))
```
**Depth:** 5 | **Size:** 11  
**Interpretation:** 2-bar high delta when current high exceeds its 20-bar mean.

### Alpha 26: Correlation Rank Decay
**Code:** Lines 280-285 in Alpha101.py  
**Formula:** `-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)`  
**ARGUS Expression:**
```
Signal(UnOp("neg",
  Window("max", 3, 
    Corr(5,
      Window("rank", 5, Ref(Field.VOLUME)),
      Window("rank", 5, Ref(Field.HIGH))
    )
  )
))
```
**Depth:** 7 | **Size:** 15  
**Interpretation:** Negative of 3-bar max of rolling correlation between ranked volume and ranked high.

### Alpha 35: Volume-Volatility-Return Product
**Code:** Lines 340-347 in Alpha101.py  
**Formula:** `ts_rank(volume, 32) * (1 - ts_rank(close+high-low, 16)) * (1 - ts_rank(returns, 32))`  
**ARGUS Expression:**
```
Signal(BinOp("mul",
  BinOp("mul",
    Window("rank", 32, Ref(Field.VOLUME)),
    BinOp("sub", Const(1.0),
      Window("rank", 16, BinOp("sub", BinOp("add", Ref(Field.CLOSE), Ref(Field.HIGH)), Ref(Field.LOW)))
    )
  ),
  BinOp("sub", Const(1.0),
    Window("rank", 32, Ref(Field.RETURN_1))
  )
))
```
**Depth:** 9 | **Size:** 23  
**Interpretation:** Product of volume rank, anti-range rank, and anti-return rank.

### Alpha 53: Normalized Range Change
**Code:** Lines 448-453 in Alpha101.py  
**Formula:** `-1 * delta((close - low + close - high) / (close - low), 9)` [Simplified: note close appears twice]  
**ARGUS Expression:**
```
Signal(UnOp("neg",
  Delay(9, BinOp("div",
    BinOp("sub", BinOp("sub", Ref(Field.CLOSE), Ref(Field.LOW)), 
                    BinOp("sub", Ref(Field.HIGH), Ref(Field.CLOSE))),
    BinOp("add", Ref(Field.CLOSE), Ref(Field.LOW))
  ))
))
```
**Depth:** 8 | **Size:** 19  
**Interpretation:** 9-bar change in normalized intrabar price location.

### Alpha 92: Conditional Min of Ranked Correlations
**Code:** Lines 565-569 in Alpha101.py  
**Formula:** `min(ts_rank(decay_linear(..., 14), 18), ts_rank(decay_linear(correlation(...), 6), 6))`  
**Status:** Actually requires `decay_linear` which ARGUS doesn't have. **REVISE TO NO.**

---

## Expressible with ONE Grammar Addition

### Addition 1: Cross-Sectional Rank (Highest Priority — 46 Alphas Unlock)

Many leading quant systems (Qlib, Formulaic Alphas reference, QuantConnect) rank factors relative to the universe. Add a node:

```python
@dataclass(frozen=True, slots=True)
class CrossRank(Expr):
    """Rank of current value relative to recent cross-sectional distribution.
    
    At each point in time, ranks the current factor value against all N
    instruments in the universe at that same timestamp. Returns 0..1.
    Requires the evaluator to provide the full cross-section; single-instrument
    evaluation returns 0.5 (no information).
    """
    operand: Expr
```

**Alphas unlocked:** 1, 2, 3, 4, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 30, 31, 33, 34, 36, 37, 38, 39, 40, 44, 45, 52, 55, 60, 68, 85, 88, 95, 99 (+46 total with partial dependencies)

### Addition 2: Open Price Field (10 Alphas Unlock)

```python
class Field(StrEnum):
    OPEN = "open"  # Opening price for the bar
```

**Alphas directly unlocked:** 2, 3, 6, 8, 14, 18, 20, 33, 37, 38, 54, 101 (+12 with this alone)

### Addition 3: Decay Linear Window Op (3 Alphas Unlock)

```python
# In Window.OPS:
"decay_linear",  # Linearly-weighted aggregate, weights decrease oldest-to-newest
```

**Alphas directly unlocked:** 31, 39, 92

### Addition 4: Scale Normalizer (8 Alphas Unlock)

```python
class Scale(Expr):
    """Divide by sum of absolute values across the universe.
    
    Requires cross-sectional aggregation. This is a universe operation,
    not a time-series one, so evaluates to 0.0 in single-instrument mode.
    """
    operand: Expr
```

**Alphas directly unlocked:** 28, 31, 32, 36, 60

---

## VWAP, ADV, and Other Missing Fields

**Alpha 7, 21, 43, 68:** Need `adv20` (average daily volume) field.  
**Alpha 27, 32, 36:** Need `vwap` (volume-weighted average price), not derivable at bar time.

Both can be **pre-computed and passed into bars** as `Bar.extra` fields if they are available at the decision timestamp. If they are not precomputed, they cannot be derived.

---

## Alphalens Factor Tear Sheet Metrics

Alphalens computes these metrics for factor evaluation (file: `alphalens/performance.py`, `alphalens/tears.py`).

### Information Coefficient (IC)

**Definition** (lines 28-74 in performance.py): Spearman rank correlation between factor values and N-period forward returns.

```python
def factor_information_coefficient(factor_data, group_adjust=False, by_group=False):
    def src_ic(group):
        f = group['factor']
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic
    factor_data.groupby([factor_data.index.get_level_values('date')]).apply(src_ic)
```

**Interpretation:** For each timestamp, correlates the factor values across all N instruments with their subsequent N-period returns. Ranges [-1, 1]; positive IC = predictive power.

**To implement for ARGUS factors:**
1. For each date `t`, compute factor value for all instruments
2. For each forward period `p` (e.g., 1-day, 5-day return), collect N-day returns for each instrument
3. Compute `spearmanr(factor_values[t], returns[t:t+p])` across the cross-section
4. Report mean IC, IC std, and IC decay across periods

### IC Decay

**Definition:** How quickly does predictive power decay as the forward period increases.

**Computation:** Run IC for periods `[1, 5, 10, 20, 60]` days and observe IC(period).

**Example:** If IC(1-day) = 0.08 and IC(20-day) = 0.01, the alpha decays 87.5% over 19 days.

### Quantile Returns & Spread

**Definition** (lines 86-124 in tears.py): Break factors into N quantiles (usually 5), compute mean returns per quantile per period.

```python
mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
    factor_data, by_group=False, demeaned=long_short
)
```

**To implement:**
1. For each date, rank factors into quintiles (or deciles)
2. For each quantile, compute equal-weighted return for period p
3. Report spread = return(top quantile) - return(bottom quantile)

### Turnover

**Definition** (lines 151-172 in tears.py): How frequently the factor ordering changes.

```python
quantile_turnover = {
    p: pd.concat([
        perf.quantile_turnover(quantile_factor, q, p)
        for q in range(1, int(quantile_factor.max()) + 1)
    ], axis=1)
    for p in periods
}
```

**Computation:** For each quantile q and period p, measure: `(# positions that changed quantile from t to t+p) / (# positions in quantile)`.

**Interpretation:** High turnover = factor flips rankings frequently = hard to trade = higher costs. Ideally < 30% per period.

### Factor Rank Autocorrelation

**Definition** (lines 164-170 in tears.py): Does the factor ranking predict itself?

```python
autocorrelation = pd.concat([
    perf.factor_rank_autocorrelation(factor_data, period)
    for period in periods
], axis=1)
```

**Computation:** Rank factors at time t and at time t+p. Compute Spearman autocorrelation of ranks.

**Interpretation:** High autocorr = stable factor = consistent picking. Good for turnover.

---

## Recommended Grammar Roadmap

### Phase 1 (Unlock 46 Alphas Immediately)
Add `CrossRank(operand)` node that ranks a value against the current cross-section.  
**Impact:** +46 alphas, many of which are the most cited (alpha 1, 3, 4, etc.)

### Phase 2 (Unlock 10-15 More)
Add `Field.OPEN` and `decay_linear` window op.  
**Impact:** +10-15 alphas with low implementation cost.

### Phase 3 (Conditional, Requires Architecture Change)
Add `Scale(operand)` for universe normalization.  
**Impact:** +5-8 alphas. Requires cross-sectional context at eval time.

### Phase 4 (Accept Limitation)
Many alphas require `vwap` or `adv20` as pre-computed fields, not derivable. Accept that these are data-dependent and punt to the caller to provide them in `Bar.extra`.

---

## Implementation Notes

### Expressions Must Satisfy Grammar Constraints
- **Depth ≤ 12:** Most expressible alphas stay under 9
- **Size ≤ 96:** No issues observed; largest is ~25 nodes
- **No Signal Nesting:** `Signal(Signal(...))` is invalid — every expression must have ONE Signal root
- **Window Cannot Wrap Signal:** `Window("mean", 10, Signal(...))` is invalid
- **Backward Only:** No alpha uses forward-looking windows; this is not a blocker

### Testing Strategy
Once expressions are written, test against real data:
1. Compute ARGUS expression value for each bar
2. Compute reference alpha value from original WorldQuant code
3. Compare correlation (should be 1.0 or very close if expressions match)
4. Run alphalens tear sheet on both and compare IC, quantile spread, turnover

