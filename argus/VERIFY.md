# Verify our claims yourself

Every "ARGUS beats X" claim in this repository is checked against that rival's own real, installed
source code — never asserted, never simulated. `eval/standing.py` enforces 13 conditions before any
claim can be marked OWNED, the binding one being: *the named baseline was actually reproduced and the
comparison actually run.* This file exists because that machinery being self-administered is itself
a real, named weakness (see the register, `data/standing.json`) — the fix is not to assert harder, it
is to make re-running the comparison yourself take one command.

No API key is required for anything below. All three pull real, live, public market data. Two of the
three (#1, #3) run a real, installed, invoked copy of the named rival's own package — nothing there
is mocked, cached, or pre-computed for this page. **#2 is different and is labelled as such below**:
it compares against a faithful port of freqtrade's real protections, read from freqtrade's own
source and cited by file:line, not an installed invocation of freqtrade itself — an earlier version
of this page called it "installed," which was wrong and is corrected here (found 2026-09-23 by an
independent adversarial re-check of this exact page, the day it was written).

```
git clone https://github.com/Pratiikpy/argus-bitget.git
cd argus-bitget/argus
python -m venv .venv          # then activate it: .venv\Scripts\activate (Windows) or source .venv/bin/activate
pip install -e ".[dev]"
```

The package lives in `argus/`; the repository root has no `pyproject.toml`, so every command below
runs from `argus-bitget/argus`.

## 1. Portfolio allocation vs Riskfolio-Lib's real NCO (≈2 minutes)

```
python -m argus.eval.allocation_comparison
```

**What you're checking:** ARGUS's own from-scratch Nested Clustered Optimization allocator against
Riskfolio-Lib's real, installed `HCPortfolio` NCO implementation, walk-forward on live Bitget market
data, two independent window setups.

**What to expect:** a line reading `ARGUS's own NCO vs Riskfolio's real NCO: <X> vs <X> bps (ratio
1.0000)` — the *ratio* ties to four decimal places; the raw bps values themselves typically agree to
three (e.g. 7.9688 vs 7.9687), not four — stated precisely here after an independent re-check found
the original "four decimal places" wording mildly overstated it. Not circular: ARGUS's allocator
(`desk/allocation.py`) imports neither riskfolio nor cvxportfolio anywhere — it is a from-scratch
Ward-linkage clustering and a hand-rolled active-set QP, independently verified against Riskfolio's
real installed `HCPortfolio` on a frozen fixture (`tests/data/allocation_fixture.json`) at max weight
diff 1.08e-05. This is the real result an earlier claimed *loss* against Riskfolio corrected into a
tie once optimal leaf ordering was reproduced from Riskfolio's own source and wired in — see the
module's own docstring for the full history, including a real crash in Riskfolio's own DBHT
clustering option (unrelated to the NCO claim, already documented, not something this repo caused).

## 2. Risk-control precision vs freqtrade's real drawdown protections (≈1 minute)

```
python -m argus.eval.risk_layer_comparison
```

**What you're checking:** ARGUS's drawdown-control gate against a faithful port of freqtrade's four
real protections — `MaxDrawdown`, `StoplossGuard`, `LowProfitPairs`, `CooldownPeriod`
(`eval/freqtrade_baseline.py`, read from freqtrade's own source and cited by file:line; **not an
installed invocation of freqtrade itself** — see the correction at the top of this page).

**What to expect, and what NOT to over-read:**
- `ARGUS dominates every swept threshold` on **precision** — this is the real, non-tautological
  result. It could genuinely have gone either way, and it consistently doesn't.
- `recall: 1.0` for ARGUS is printed too, but **read it as true by construction, not as an
  achievement**: the comparison's ground truth is defined as "drawdown crosses ARGUS's own stated
  threshold," and ARGUS's gate fires on exactly that condition — so its recall against its own
  threshold cannot be anything but 1.0. freqtrade was never capable of losing that specific number to
  begin with. Precision is the metric that actually discriminates.
- A low correlation figure (freqtrade's own proxy vs. the real drawdown ground truth) prints too,
  currently 0.2333.

**Fixed 2026-09-23 — the numbers above are now pinned, not a snapshot.** This command used to pull
a live, rolling 90-day market-data window every run, so the precise figures shifted call to call
(three independent runs on 2026-09-23 alone gave three different ARGUS precision figures: 59.2%,
100%, 66.7%). It now reads `data/risk_layer_candles_fixture.json` — real Bitget history, fetched
once on 2026-09-23 and frozen — by default, the same fix already applied to #1 above. Run it twice
in a row; the precision (59.14%), recall (1.0) and correlation (0.2333) printed will be identical
both times. To check the qualitative result still holds on fresh data, run
`python -m argus.eval.risk_layer_comparison --live` instead — that re-fetches a new rolling window
and should still show ARGUS dominating on precision with correlation in the same weak range, just
not the identical decimal.

*Note: Bitget's public market-data endpoint rate-limits after sustained heavy use (HTTP 429) — this
only affects `--live` and `--freeze-fixture`; if a couple of symbols fail to fetch, wait a minute
and re-run. It does not affect the headline result, which does not depend on any single symbol.*

## 3. Session-boundary changepoint detection vs ruptures (≈10–25 minutes — real compute, not a hang; measured 9 and 20+ minutes on two machines)

```
python -m argus.eval.regime_comparison
```

**What you're checking:** whether ARGUS's own segmenter can detect the same session-boundary
regime changes as `ruptures`' real `KernelCPD`, on 100 synthetic trials with known ground-truth
changepoints. This one is genuinely slow — it's a real matrix-profile-scale computation, not stuck.

**What to expect:** the file's own history is instructive here too — ARGUS's *original* segmenter
(FLUSS) loses decisively (F1 0.443 vs ruptures' 0.975); a second, later ARGUS segmenter built
directly from ruptures' own BSD-2-Clause source ties it (1 win / 0 losses / 99 ties). Both results
are reported, not just the one that flatters us.

## What this deliberately does not include

`eval/sentiment_comparison.py` (vs ProsusAI/finBERT) is a real, OWNED comparison but needs a metered
Qwen API key to re-run live — left out of this page rather than promised and then blocked. Its
output is committed at `data/sentiment_comparison.json` if you want to read the last real run instead
of reproducing it.

The full register — every capability with its named rival, its state and the artefact behind it —
is `data/standing.json`, rendered on `/proof`, with every loss on `/wrong`. This page is three of
those comparisons, chosen because they need nothing from us to re-run.

## Found something wrong?

Open an issue. A claim that doesn't reproduce the way this page says is a real defect in this
project, not in your setup — we'd rather know.
