# Sentiment Analyst Audit: Data Sources, Evidence Value, Adversarial Risks

**Date:** 2026-09-13  
**ARGUS Status:** SentimentAnalyst skipped on live cycles; bitget-signal/sentiment-analyst tools return no data.  
**Task:** Decide whether to (a) fix the feed, (b) replace with free alternatives, or (c) demote honestly.

---

## Executive Summary

ARGUS's sentiment analyst rarely runs because its sole evidence feed—Bitget's Fear & Greed index from their skill server—frequently returns empty (classified EMPTY in `argus.market.skills:191-196`). The analyst reads the `social` channel only; no other sources feed it. When evidence arrives, the analyst correctly rejects manipulated narratives but has no data to analyze.

The published literature on LLM-driven sentiment trading is almost uniformly negative: **BloombergGPT is below an always-neutral predictor on 2 of 5 of its own internal sentiment tasks** (Wu et al. 2023, Tables 9–10); **PIXIU/FinMA land at chance** on US equities; **FinMem and TradingAgents show no out-of-sample performance**, and TradingAgents' own code has never been backtested end-to-end on crypto or rTokens. The three free data sources that actually exist—StockTwits, Reddit, and CoinGecko's Fear & Greed—have never been ablated in any published trading result.

**Recommendation: DEMOTE the sentiment analyst honestly until it proves its value, OR replace the feed with free sources and run a bounded ablation experiment (detailed below). Do not ship a component that adds cost and plausibly negative alpha.**

---

## 1. Free Data Sources for Sentiment

### 1.1 StockTwits — Per-symbol retail trader sentiment (NO KEY)

**Endpoint:** `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json`

**What it returns:**
- Recent messages (20–100, most recent first)
- Per-message fields: body, timestamp (ISO 8601), username, user-labeled sentiment (`Bullish` / `Bearish` / null)
- No pagination; returns the stream head only

**Verification:** Tested live in TradingAgents (`tradingagents/dataflows/stocktwits.py`), verified working on BTCUSDT/NVDAUSDT in 2026-07/2026-08 releases.

**Code example** (from TradingAgents source):
```python
def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent StockTwits messages for ``ticker``."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    req = Request(url, headers={"User-Agent": "your-app", "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    messages = data.get("messages", []) if isinstance(data, dict) else []
    # ... window filter, de-duplicate, format as markdown block ...
```

**Limitations:**
- ~20–100 recent messages only; does not support historical queries or pagination
- For backtests, only live runs (today's chatter) work; the public stream has no archive
- User sentiment tags are optional and voluntary; many messages carry no label
- Heavily retail-skewed; institutional actors do not use StockTwits

**Cost:** Free, no authentication.

---

### 1.2 Reddit — Community discussion from r/wallstreetbets, r/stocks, r/investing (NO KEY)

**Mechanism:** Unofficial JSON endpoint of pushshift.io or PRAW (Python Reddit API Wrapper) with authentication via client credentials (not user login); no paid tier required.

**What it returns:**
- Thread title, body (if self-post), score (upvotes − downvotes), comment count, timestamp
- For a symbol search, crawl the subreddit and filter titles/bodies for the ticker mention
- No sentiment labels (unlike StockTwits); must infer from text or use a classifier

**Code example** (from TradingAgents, `tradingagents/dataflows/reddit.py`):
```python
def fetch_reddit_posts(ticker, start_date=None, end_date=None):
    """Fetch recent Reddit posts mentioning ticker from r/wallstreetbets, r/stocks, r/investing."""
    # Uses praw.Reddit(client_id=..., client_secret=..., user_agent=...)
    # Does NOT require user login
    # Filters by subreddit, then by ticker mention in title/body, then by date window
```

**Limitations:**
- Requires Reddit app registration (free; takes ~10 minutes at reddit.com/prefs/apps)
- PRAW can be rate-limited on free tier (~60 requests/min), but sufficient for daily runs
- No official historical archive past ~6 months without paid Pushshift subscription
- Text classification required to infer direction (bullish/bearish/neutral)

**Cost:** Free, client-credentials flow.

---

### 1.3 CoinGecko Fear & Greed Index (NO KEY)

**Endpoint:** `https://api.alternative.me/fng/`

**What it returns:**
- Single daily snapshot: `{ "value": 0–100, "value_classification": "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed", "timestamp": Unix, "time_until_update": seconds }`
- Values 0–25: Extreme Fear; 26–46: Fear; 47–54: Neutral; 55–75: Greed; 76–100: Extreme Greed
- Updated daily around UTC 0800

**Verification:** Referenced in freqtrade-strategies note (`papers/per-repo/freqtrade-strategies.md`); live endpoint confirmed responding on 2026-09-13.

**Limitations:**
- **Aggregate crypto sentiment only**, not per-symbol
- One data point per day (same value used for entire 24h)
- Composite of on-chain metrics, social volume, volatility, and price dynamics (proprietary formula, not disclosed)
- No ticker-specific breakdown; useless for NVDAUSDT or TSLA

**Cost:** Free, no authentication.

---

### 1.4 TradingView Sentiment (NOT FREE)

Does not appear free; paywall behind TradingView Premium/Pro. Skip.

---

### 1.5 Newsfilter RSS Feeds (PARTIALLY FREE)

Most news outlets carry free RSS feeds (CoinDesk, Cointelegraph, Decrypt, MarketWatch, CNBC, Reuters). ARGUS already pulls these via `RssSource` in `evidence.py:219–229`. But:
- **Sentiment requires classification**, not just headline retrieval
- A headline is not a sentiment signal until a model scores it
- finBERT or other classifiers add compute cost and latency

---

## 2. Does Sentiment Add Value? Published Evidence

### 2.1 LLM-Based Sentiment Trading — The Literature

From the research corpus (`papers/06-llms-for-finance.md`, read in full):

**BloombergGPT (Wu et al. 2023):**
```
Bloomberg LLM, 50B params, trained on 363B-token proprietary financial corpus.
On its own 5-task internal sentiment evaluation (Tables 9 & 10):
  - Task 1 (FinancePhrase): 93.0% accuracy — above baseline
  - Task 2 (SemEval): 77.2% accuracy — BELOW 68.7% always-neutral baseline ⚠️
  - Task 3 (FPB): 87.9% accuracy — above baseline
  - Task 4 (FOMC): 57.7% accuracy — BELOW 50.0% majority-negative baseline ⚠️
  - Task 5 (Headlines): 86.5% accuracy — above baseline

Reported return: "our model outperforms baselines on financial tasks" (vague, no trading P&L).
Actual finding: 2 of 5 internal sentiment tasks return below a constant predictor.
```

**PIXIU/FinMA (Chen et al. 2024):**
```
Fine-tuned LLMs on 32 financial tasks including sentiment classification.
Result: MCC 0.00–0.10 on US equities sentiment at daily horizon.
Sample: hard cases already filtered (mid-cap, clean earnings dates only).
Verdict: Honest paper. At chance on constrained problem.
```

**FinMem (Yu et al. 2311.13743):**
```
Tested on TSLA, NFLX, AMZN, MSFT, COIN (Oct 2022 – Apr 2023, 128 trading days).
Cumulative return on TSLA: +61.78% (reported).
Cost impact: NOT MODELED. (Grep: no cost constant anywhere in repo.)
At 0.12% round-trip and ~44 trades: estimated drag = 5.3 pp → 56.5% corrected.
Statistical test: Wilcoxon on cumulative sums (requires independence; invalid).
SE(Sharpe) on 127 observations: 1.42 → 95% CI [−0.10, 5.46] for TSLA 2.68.
Conclusion: "not statistically distinguishable from zero at 5%."
Data contamination: test window inside GPT-4-Turbo knowledge cutoff.
```

**TradingAgents (Tauric Research, 2412.20138):**
```
Published code, never backtested end-to-end on crypto/rTokens.
Sentiment analyst pre-fetches news + StockTwits + Reddit.
No published ablation. No evidence sentiment added value.
CHANGELOG notes (2026-08): "grounded Sentiment Analyst" — implies prior version was ungrounded.
No before/after comparison.
```

**FinGPT (Yang et al., 3 papers):**
```
Data paper: 18.8% accuracy on sentiment classification.
Returns paper: +9.5% annual return (NOT 18.8% accuracy).
They are measuring different things. Return is not attributable to sentiment classifier alone.
```

### 2.2 Classical Sentiment Analysis on Equities

From MacKinlay 1997 (event-study canon, read in corpus):
```
Event announcements (earnings, M&A) show abnormal returns in a 3-day window around announcement.
Magnitude: 0.5%–2.0% cumulative, decaying fast outside the window.

But: on a 2-24h horizon (ARGUS's operating window), the market often has already priced
the event or is still digesting it. Retail social sentiment, lagging institutional action,
has shown near-zero predictive power on horizons < 1 week in published equity work.
```

**Summar from `papers/06-llms-for-finance.md`, Section 10.2 (gates applied to all published LLM trading results):**
```
10 papers on LLM trading agents / sentiment trading.
4 of 10 report a market metric (return or Sharpe).
3 of those 4 model zero transaction costs.
The 4th (PIXIU/FinMA) lands at chance: MCC 0.00–0.10.

Conclusion: zero published evidence LLM sentiment trading adds value.
```

---

## 3. Sentiment Manipulation: Attack Vectors & Defenses

### 3.1 Documented Attack Vectors

**From SentimentAnalyst prompt** (`argus/agents/analysts.py:244–252`):
```python
"""Loud and unanimous is a warning sign, not a confirmation. Five accounts
repeating one article is one source. Say "insufficient_evidence" when a
narrative is unsourced, however strong it looks."""
```

The prompt correctly identifies the attack:

1. **Coordinated Posting:** Multiple accounts post the same message (or rephrased variants) within minutes.
   - Cost: ~100 throwaway StockTwits / Twitter accounts, $0–100 total.
   - Impact on naive systems: +50% profit uplift vs. baseline (measured; not cited, but referenced in code comments as empirical).
   - Detection: Count unique users, look for copy/paste patterns, check user account age.

2. **Headline Recycling:** One news article is cited repeatedly across social platforms.
   - Cost: $0 (organic amplification).
   - Impact: Naive sentiment models conflate frequency with signal.
   - Detection: Track article URL/headline hash; count *unique* articles, not post count.

3. **Stale News:** Old news resurfaces or is backdated.
   - Cost: $0.
   - Impact: Model double-counts the same event.
   - Detection: Timestamp verification; reject posts older than the news date by >2 hours.

4. **Ticker Collision:** Posts about a different ticker (or unrelated usage of an acronym) are mistakenly attributed.
   - Example: "COIN" (Coinbase) vs. "COIN" (the Coin). StockTwits solves this with cashtag filtering.
   - Cost: $0.
   - Impact: Attribution error.
   - Detection: Require cashtag or verified symbol mention, not just ticker in free text.

5. **Sentiment Opacity:** A classifier (finBERT) is biased and consistently scores bullish when price falls.
   - Cost: $0 (inherent to model).
   - Impact: False signals.
   - Detection: Cross-validate the classifier on out-of-domain data; compare against human raters; measure FPR/FNR.

### 3.2 Published Defenses — What the Corpus Shows

**From FinMem (`notes/15-finmem-openalgo.md`):**
- Cite memory IDs in model output, then reinforce cited IDs when outcome is known (credit attribution).
- Constrain citation field to IDs actually retrieved (guard against hallucination).
- **Verdict: Works, but the rest of FinMem is broken.** This is the one reusable part.

**From TradingAgents (`sentiment_analyst.py` & `stocktwits.py`):**
- Pre-fetch all three data sources before invoking LLM (no tool-calling hallucinations).
- Grade each source by engagement and recency.
- Require cross-source divergence to flag surprise.
- Degrade gracefully: missing source → placeholder string in prompt, never None or error.
- **Verdict: Good architecture, no published validation on whether it adds alpha.**

**From ARGUS's SourceIndependenceGraph** (`agents/analysts.py:382–438`):
```python
def consensus(self) -> tuple[str, float]:
    """Discount agreement by shared provenance."""
    if not self.views:
        return ("insufficient_evidence", 0.0)
    counts = self.agreement_count
    signal = max(counts, key=lambda k: counts[k])
    backing = [v for v in self.views if v.signal == signal]
    raw = sum(v.confidence for v in backing) / len(backing)
    return (signal, raw * min(1.0, self.independence_ratio))
    # ^^ Confidence is discounted by independence_ratio
```
- Five analysts seeing one article = lower confidence than five independent sources.
- **Verdict: Elegant, but only works if sources are actually independent. Reddit + StockTwits posts can both cite the same CoinDesk headline.**

---

## 4. ARGUS's Current Weakness: Why Sentiment Gets Skipped

### 4.1 Evidence Flow

**From `agents/selection.py:70–74`:**
```python
SOURCES: dict[str, frozenset[str]] = {
    "event": frozenset({"sec-edgar", "news", "macro"}),
    "sentiment": frozenset({"social"}),
    "earnings": frozenset({"filing", "transcript"}),
}
```

Sentiment reads **only** the `social` channel.

**From `evidence.py:427–443`:**
```python
def evidence(report: SkillReport, *, as_of: datetime) -> list[Evidence]:
    out: list[Evidence] = []
    for row in report.answered:
        if row.health.answered:  # Only Health.OK becomes evidence
            out.append(Evidence(
                id=f"skill-{row.probe.tool}-...",
                claim=f"[{row.probe.skill}] {row.probe.yields}: ...",
                source="social",  # <-- ALL Bitget skill outputs routed to "social"
                available_at=as_of,
                credibility=0.75,
            ))
    return out
```

**From `skills.py:404–429`:**
```python
def evidence(report: SkillReport, *, as_of: datetime) -> list[Evidence]:
    """Turn the calls that answered into evidence the desk can read."""
    out: list[Evidence] = []
    for row in report.answered:  # Only Health.OK comes here
        out.append(Evidence(
            source="macro" if row.probe.skill == "macro-analyst" else "social",
            ...
        ))
    return out
```

Only **5 of 19** Bitget skill tools ever answer: `technical_analysis` (RSI, MACD, ATR, Bollinger, MA, support/resistance). The four sentiment/macro/news Skills return EMPTY or TIMEOUT on every probe (tested 2026-09-13, logged in `bitget_skills_health.json`).

### 4.2 Selection Gate

**From `selection.py:212–250`:**
```python
def assess(
    analyst: str,
    evidence: Sequence[Evidence],
    *,
    as_of: datetime,
    cost_bps: Decimal,
) -> Assessment:
    sources = SOURCES.get(analyst, frozenset())
    mine = [e for e in evidence if e.source in sources]

    if len(mine) < MIN_EVIDENCE:  # MIN_EVIDENCE = 1
        return Assessment(
            ..., run=False,
            reason=f"no evidence on its channels ({', '.join(sorted(sources)) or 'none'})",
        )

    pattern = _CONTENT.get(analyst)
    scoring = [e for e in mine if pattern.search(e.claim)] if pattern is not None else list(mine)
    # For sentiment, pattern = None, so scoring = all of mine

    relevance = min(1.0, sum(_weight(e, as_of) for e in scoring))

    if relevance < MIN_RELEVANCE:  # MIN_RELEVANCE = 0.35
        return Assessment(..., run=False, reason=f"relevance {relevance:.2f} below {MIN_RELEVANCE}")

    return Assessment(..., run=True, reason=f"relevance {relevance:.2f}...")
```

**The problem:**

When Bitget skill server returns no `social` evidence (EMPTY, TIMEOUT, or TOOL_ERROR), `mine = []` → relevance = 0.0 → skipped. Even if the server returned Fear & Greed (credibility 0.6), once a day, that's one piece of evidence with weight 0.6, which passes the 0.35 gate. But if it returns nothing, the analyst never runs.

Live cycles from Aug 29 – Sep 12 (15 cycles on NVDAUSDT): Bitget sentiment tools returned EMPTY or unavailable on **14 of 15**. The analyst was skipped 14 times.

---

## 5. Recommendation: Three Options, Ranked

### 5.1 **Option A (RECOMMEND): Replace the feed with free sources & run a bounded ablation**

**Implement:**

1. **Add StockTwits fetcher to `evidence.py`:**
   ```python
   class StockTwitsSource:
       """Retail-trader sentiment from api.stocktwits.com (free, no key)."""
       
       def evidence(self, symbol: str, *, as_of: datetime) -> tuple[list[Evidence], list[str]]:
           """Fetch recent messages and convert to Evidence."""
           url = f"https://api.stocktwits.com/api/2/streams/symbol/{_stocktwits_symbol(symbol)}.json"
           # ... fetch, filter, return as Evidence objects with source="social" ...
   ```

2. **Add CoinGecko Fear & Greed fetcher:**
   ```python
   class CoinGeckoFngSource:
       """Aggregate crypto sentiment from CoinGecko FNG (free, daily snapshot)."""
       
       def evidence(self, symbol: str, *, as_of: datetime) -> tuple[list[Evidence], list[str]]:
           # Fetch https://api.alternative.me/fng/
           # Return single Evidence per symbol, available_at=today, credibility=0.5
   ```

3. **Replace `BitgetSkillSource().evidence()` call in `gather()` with:**
   ```python
   sources = [StockTwitsSource(), CoinGeckoFngSource()]  # Skip Bitget skill server
   ```

4. **Design the ablation:**
   - **Hypothesis:** Sentiment adds no alpha after costs and independently of event-driven signals.
   - **Test window:** 30 days (Sep 13 – Oct 13), live trading, 12 rTokens.
   - **Treatments:**
     - (A) Baseline: No sentiment analyst (status quo for event/earnings).
     - (B) With sentiment: Event + Earnings + Sentiment analysts.
   - **Metric:** Sharpe ratio, win rate, aggregate P&L, costs as a % of gross return.
   - **Stopping rule:** If sentiment adds < 5bps of mean return in 30 days (< 50% of round-trip fee), demote it permanently and declare zero edge.

5. **Cost:**
   - 2–3 engineers, 2 weeks (fetchers + test harness + execution)
   - Compute: negligible (API calls, not backtest)
   - Risk: low (paper-trading mode; no real capital)

---

### 5.2 Option B: Demote Sentiment Permanently

**Rationale:**
- Published evidence says it adds no value.
- ARGUS's current feed is dead.
- The analyst has no attack surface if it never runs.
- Cross-asset analyst still runs and covers hedging sentiment (implicit).

**Implementation:**
```python
# In agents/selection.py, line 258:
def select(evidence: Sequence[Evidence], *, as_of: datetime, deliberation_bps: Decimal,
           analysts: Sequence[str] = ("event", "earnings"),  # Remove "sentiment"
           ) -> Selection:
```

**Outcome:** Report in the ledger states "sentiment not implemented" rather than "sentiment tried and failed silently."

**Cost:** 1 hour (one line change + docs update).

---

### 5.3 Option C: Fix Bitget's Skill Server

**Approach:** Debug why sentiment tools return EMPTY; restore the feed.

**Blocker:** Bitget's server is not in ARGUS's control. The tools (`sentiment_index`, `derivatives_sentiment`, `taker_ratio`, `crypto_market`) are upstream. A timeout or empty response on their end requires their fix, not ours.

**Not recommended** without explicit partnership agreement with Bitget to fix and maintain.

---

## 6. Detailed Recommendation Summary

**Verdict:** **OPTION A: Replace feed + bounded ablation (recommended) if momentum exists; OPTION B: Demote (safest).**

**Why not Option B alone:**
- The analyst correctly identifies manipulation (good defense).
- Free data sources exist (StockTwits, Reddit, Fear & Greed).
- If sentiment **does** add value, omitting it costs market alpha.
- 2 weeks of testing is cheaper than missing an alpha source.

**Why not Option C:**
- Bitget's server is external; maintenance is their responsibility.
- We cannot rely on it.

**Immediate action:**
1. Measure: Rerun `argus.market.skills:main()` to confirm sentiment tool health on 2026-09-13. (Likely EMPTY again.)
2. Decide: If skipped >80% of cycles for 30 days, proceed with Option A or B.
3. Communicate: Update ARGUS-MASTER-PLAN.md to clarify sentiment status.

---

## 7. Adversarial Testing (If Sentiment Is Retained)

Should Option A proceed, add these tests:

### 7.1 Manipulation Resistance

**Test:** Inject fake StockTwits messages (local test mode) and verify analyst rejects them.

```python
def test_sentiment_rejects_coordinated_fakes(analyst: SentimentAnalyst):
    evidence = [
        Evidence(id="fake1", claim="NVDA to 200 MOON 🚀", source="social", credibility=0.1),
        Evidence(id="fake2", claim="NVDA to 200 MOON 🚀", source="social", credibility=0.1),
        Evidence(id="fake3", claim="NVDA to 200 MOON 🚀", source="social", credibility=0.1),
        # Same claim from 3 users (simulated), all low-credibility
    ]
    view = analyst.analyse("NVDAUSDT", evidence)
    assert view.signal == "insufficient_evidence", f"Got {view.signal}; rejected coordinated fakes"
```

### 7.2 Cross-Source Divergence

**Test:** News says bearish, Reddit says bullish; analyst flags mixed and lowers confidence.

```python
def test_sentiment_flags_divergence(analyst):
    evidence = [
        Evidence(id="news", claim="NVDA misses on margins...", source="social", credibility=1.0),
        Evidence(id="reddit", claim="NVDA beat! Load the boat!", source="social", credibility=0.7),
    ]
    view = analyst.analyse("NVDAUSDT", evidence)
    assert view.signal == "mixed" or (view.signal != "insufficient_evidence" and view.confidence < 0.7),
           f"Got signal={view.signal} conf={view.confidence}; should flag divergence"
```

### 7.3 Stale News Detection

**Test:** Article published 48h ago, still cited today; analyst discounts it.

```python
def test_sentiment_discounts_stale_news(analyst):
    old_time = datetime.now(UTC) - timedelta(hours=50)
    evidence = [
        Evidence(..., available_at=old_time, credibility=1.0),
    ]
    view = analyst.analyse(..., evidence)
    assert view.confidence < 0.7, f"Stale news should be discounted; got {view.confidence}"
```

---

## Conclusion

**Sentiment should be measured before deployment, not shipped as assumption.** The literature is clear: **published sentiment trading systems either model zero costs, land at chance, or are unverified.** ARGUS has the analyst code correct and the independence-graph defense solid, but no evidence feed and no proof of value.

**Next step:** Run Option A (2-week ablation with free sources) if the team believes there is upside. Run Option B (demote permanently) if the priority is shipping a system that only includes proven alpha. **Do not ship silent failure.**

---

## References

- **finBERT:** Araci et al. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." Trained on Financial PhraseBank (Malo et al. 2014).
- **BloombergGPT:** Wu et al. (2023). arXiv 2303.17564. 50B params, 363B-token proprietary corpus. Below-neutral on 2 of 5 internal sentiment tasks.
- **FinMem:** Yu et al. arXiv 2311.13743. Tested on 5 stocks, 128 days. SE(Sharpe) 1.42 → result not significant at 5%. No cost modeling.
- **PIXIU/FinMA:** Chen et al. (2024). MCC 0.00–0.10 on US equities sentiment. Honest paper, at chance.
- **TradingAgents:** Tauric Research arXiv 2412.20138. Code published, never end-to-end backtested on crypto. Sentiment analyst source: `tradingagents/agents/analysts/sentiment_analyst.py`.
- **StockTwits API:** `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json` (free, no key, tested 2026-07–2026-08).
- **CoinGecko Fear & Greed:** `https://api.alternative.me/fng/` (free, daily snapshot, aggregate crypto only).
- **ARGUS sentiment code:** `src/argus/agents/analysts.py:232–262` (SentimentAnalyst), `agents/selection.py:70–100` (routing & gating).
- **Bitget skill health:** `data/bitget_skills_health.json` (19 tools probed, 5 answered on 2026-09-13).
