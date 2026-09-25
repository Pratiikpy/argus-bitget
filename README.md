# ARGUS

**A research desk for Bitget's tokenized US stocks. Ask in plain language; every number is
computed from live data and names its source.**

**Live console: https://deploy-topaz-seven-64.vercel.app** · [one research task, run
live](https://deploy-topaz-seven-64.vercel.app/research) · [what we beat](https://deploy-topaz-seven-64.vercel.app/proof)
· [what we got wrong](https://deploy-topaz-seven-64.vercel.app/wrong) · [the Track 2 agent, live](https://deploy-topaz-seven-64.vercel.app/agent)
· [status](https://deploy-topaz-seven-64.vercel.app/status)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/console-answer-dark.png">
  <img alt="The ARGUS console answering 'I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?': a sized, actionable answer, the risk it adds, a hedge, stress cases, how far to trust the beta against a named rival, and a receipt naming every source." src="docs/img/console-answer-light.png">
</picture>

Built for Bitget AI Base Camp / Genesis Hackathon Season 2. Nothing on this page needs an account
or a key to try.

---

## Two entries, one system

| | What it is | Track |
|---|---|---|
| **ARGUS** (this repository) | A natural-language research workbench: a trader asks, seven engines answer from Bitget's market data, SEC filings, FRED and the news, and the answer ends in a receipt. | **Track 3 — AI Trading Desk** |
| **t2-sentiment-agent** ([live record](https://t2-sentiment-agent-live.vercel.app)) | A market-sentiment trading agent. Qwen decides, a risk kernel that can only reduce stands between it and the venue, and Bitget's Agent Hub places every order on Bitget Demo. Paper run live since 2026-09-24, its log hash-chained from a timestamped genesis. | **Track 2 — Agentic Trading** |

They share one discipline: the model reads and decides, it never writes a number; every figure is
computed and sourced; every loss is published.

---

## Try it in a minute

Open the console and ask any of these. Each one exercises something different.

| Ask | What it shows |
|---|---|
| *I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?* | Your book's risk before and after, sized to a risk budget, a hedge, stress cases, and how far to trust the beta — scored against weekend-copilot, an S2 rival answering the same question. |
| *How should I split a $50k order in NVDA?* | An execution schedule priced on the live order book, measured against Bitget's own TWAP on a full-depth replay. |
| *How does NVDA react to CPI?* | An event study over past releases, with the tests that say whether the reaction is real. |
| *Is the hype on NVDA real?* | Headlines grouped into stories, so five outlets repeating one article count once. |
| *I hold NVDA r-token overnight — how do I hedge it?* | The spot rToken hedged with the same company's perpetual, tested on held-out nights. |
| *Long MSTR perp into earnings — funding looks cheap* | Your own premise tested first — funding against the contract's last 100 settlements — then the earnings half: report date, analyst targets, the last surprise, holders, filings. |
| *Will MSTR be higher in 48 hours?* | Not a forecast: how often it finished higher over every past 48-hour window, with an interval that counts overlapping windows honestly, the cost of holding, and what Polymarket prices. |
| *What if I trim TSLA to 10% of my book?* | A resize, not an add: every other holding rescaled, risk share before and after, and the weight that fits your budget. |

Put your holdings in **My book** once and every answer uses them. Every answer ends by naming the
sources it reached and any that did not answer. If the desk cannot source a figure, it refuses and
says why.

**Who it is for:** a trader holding Bitget's tokenized US equities who wants a second desk that
shows its work. **What it is not:** a signal service. No strategy here has cleared its own
deflation gate, and the console says so.

---

## What we beat, and what beat us

Every capability is held against the specialist that leads its sub-theme, run on the same input,
and graded by a register that opens the evidence rather than trusting a filename
(`python -m argus.eval.standing`). **20 of 42 capabilities are OWNED, 11 are TIED, 11 are
IMPLEMENTED, and 0 are LOST.** OWNED needs all thirteen conditions: the rival's best
implementation read and reproduced, a same-input comparison with costs, out-of-sample, ablation,
an adversarial test, documented failure cases and reproducibility.

Losses are published the moment they are found. Two were found and closed on 2026-09-24: Bitget's
own 60-second TWAP beat the schedule the console printed (6.9 against 12.2bps on a $100k order),
and the S2 entry Ballast hedged an rToken holder's nights far better than ARGUS's index hedge.
Both now tie, and both losses stay on [`/wrong`](https://deploy-topaz-seven-64.vercel.app/wrong).
Two more S2 desks were run on the same input on 2026-09-25. MirrorLine checks a trader's claims
about the tape: on the first run ARGUS gave a verdict on 18 of 38 claims to MirrorLine's 38, the
fix went in the same morning, and the re-run tied at 30 of 30 gradable claims. optic-bitget debates
a thesis with a model judge: it abstained on two of three theses for want of earnings and
positioning data that ARGUS answered, while it still gives one synthesised call ARGUS does not — a
tie, with the losing half named. baserate prices a leveraged weekend hold: ARGUS first answered a
different question, then was rebuilt to read 1,443 NVDA weekends since 1999, the perpetual's own
path through each weekend and Bitget's live margin tier — a tie, since baserate still matches the
current regime to past weekends. Rook ends each thesis with an invalidation price: on six names its
stops sat 0.1-1.7% away and ordinary movement reached them on 62% of held-out days, against 11%
for ARGUS's measured stop, fitted out of sample — ahead on that one measure, six names, one run.
The full table, rival by rival, is on [`/proof`](https://deploy-topaz-seven-64.vercel.app/proof).

---

## Bitget's toolkit, used and measured

| Surface | Where ARGUS uses it | Measured |
|---|---|---|
| Public market API (v3) | Candles, tickers, 50-level books, funding — every price and cost in the console | Answering, checked live on `/status` |
| Order-book websocket (`books5`, trades) | Execution replay and the order-splitting comparison | Recorded locally, RFC 6455 client |
| `bitget-mcp-server` | US fundamentals, 13F holders, analyst estimates, earnings calendar, the stock behind each rToken | 37 of 67 catalog entries answered |
| `bitget-signal` Skills | Technicals (MACD corrected against Bitget's own candles), sentiment, macro, news, market intel | Bitget's server answered 6 of 19; ARGUS read the other 13 from the sources those Skills name — 19 of 19, all 5 Skills |
| Agent Hub (`bgc`) | The Track 2 agent's every order, `--paper-trading`, previewed with `--dry-run` | Demo run live |

When a Skill's hosted server fails, ARGUS reads the source that Skill names (alternative.me,
Binance futures data, FRED, Yahoo, DeFiLlama, CoinGecko, the RSS feeds). Every such answer says
it came from the source, not the Skill, and the two counts are never merged
(`argus/src/argus/market/skill_mirror.py`).

---

## Call it from an agent

ARGUS is also a **Model Context Protocol server**: any MCP client can add
`https://deploy-topaz-seven-64.vercel.app/mcp` (Streamable HTTP) and call six tools — `argus_ask`
(the whole console, any language), `argus_quote`, `argus_portfolio_impact`, `argus_stress` (a shock
on the Nasdaq or on any instrument), `argus_execution_plan` and `argus_scoreboard`. They run the
same engines as the page; none of them writes or trades.

---

## Run it yourself

```bash
cd argus
pip install -e ".[dev]"
python -m argus.status          # module and sub-theme coverage, resolved by import
python -m argus.lui.server      # the console on http://127.0.0.1:8765
pytest -q                       # 6,700 tests collected
```

Nothing above needs a credential. The full run took 1h16m from a fresh GitHub clone on
2026-09-24 — 6,457 passed, 85 skipped, 0 failed; tests that need a rival's source cloned beside
the repository skip and say which. `ARGUS_BLOCK_NETWORK=1` refuses every outbound connection, so
a test that reaches the network shows itself.

| | |
|---|---|
| **The code** | [`argus/`](argus/) — package, tests, runnable studies |
| **How it fits together** | [`ARGUS-ARCHITECTURE.md`](ARGUS-ARCHITECTURE.md) |
| **Every claim, explained** | [`ARGUS-EXPLAINED.md`](ARGUS-EXPLAINED.md) |
| **Reproduce a headline number** | [`argus/VERIFY.md`](argus/VERIFY.md) |
| **What runs and what is blocked** | [`argus/STATUS.md`](argus/STATUS.md) |

---

## What is verified

| | |
|---|---|
| Tests | **6,700 tests collected** — `pytest -q` |
| Types | **`mypy --strict` clean on 343 source files** |
| Lint | `ruff` clean |
| Modules | **152/152 modules importable**, checked by `python -m argus.status` |
| Sub-themes | **18/18 sub-themes**, resolved by import at runtime, not claimed in prose |
| Understanding | 240 questions in 12 languages, written by an agent that never saw this repository, scored without reading its misses: the console, with no language model, read **52.1% → 81.7%** of them correctly after 2026-09-25 (85.0% with a saved book; on the live site Qwen reads first and this is the fallback); the trained question classifier alone reads 91.3%; with Qwen reading first the console reads 85.0% — `data/lui_final_heldout_report.json` |
| Quoted figures | Every figure these documents quote is re-checked against its artefact by `python -m argus.eval.docclaims --tests`, which fails if one has drifted |

## What the code will not let happen

- **A zero-fee backtest cannot be built.** `CostModel` raises rather than defaulting to free.
- **Omitting `as_of` is a `TypeError`.** Point-in-time evidence cannot be queried carelessly.
- **The risk layer may only reduce.** It cannot create, flip or grow a position.
- **A maker fee needs a book.** Passive fills go through a queue model ported from hftbacktest.
- **Search cannot see evaluation.** `ProposerContext` has no field that can hold a score.
- **The ledger is hash-chained**, and a tampered chain is refused rather than scored.

---

## Honest limits

- **ARGUS's own paper desk has no settled trades.** Every decision on its ledger is a refusal, so
  its Sharpe, drawdown and win rate are undefined and printed as such. Each refusal carried a
  direction, hashed before the outcome existed: at about two hours, **194 of 325 directional calls
  were right**, which barely clears a coin flip and does not beat calling "up" every time. The
  median refusal **forgave -4.5bps of net edge** after the 12bps round trip — the trades it passed
  on were mostly unprofitable. The Track 2 agent is the entry that trades.
- **No certified alpha.** 0 of 8 factors and 0 of 12 strategies cleared the deflation gate.
- **Price discovery attenuates about 7x while the anchor market is shut — it does not stop.** The
  weekend effect failed its own out-of-sample split, and the 3–5x off-hours spread widening first
  claimed here was falsified. Both are reported as negatives.

---

## 中文简介

ARGUS 是面向 Bitget 美股代币（rToken）的研究工作台（Track 3 · AI Trading Desk）。用自然语言提问，
七个引擎基于 Bitget 行情、SEC 文件、FRED 与新闻实时计算，每个数字都注明来源；语言模型只理解问题，
从不编写数字。38 项能力逐一与各子赛道领先的专业系统在相同输入上对比：20 项领先（OWNED）、8 项
持平、10 项已实现、0 项落后，所有落败记录公开在 `/wrong`。配套的 Track 2 情绪交易 Agent 由 Qwen
决策、只能减仓的风控内核把关，经 Bitget Agent Hub 在 Demo 环境下单，纸面交易日志自带时间戳哈希链。

---

## Licence

MIT. Components ported from other projects cite their source file and line in the docstring;
everything copied is MIT, BSD or Apache licensed, and anything under a restrictive licence was
rebuilt from its described behaviour rather than copied.
