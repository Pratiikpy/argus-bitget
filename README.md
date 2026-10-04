# ARGUS

[![CI](https://github.com/Pratiikpy/argus-bitget/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratiikpy/argus-bitget/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-11%2C280%20collected-2ea44f)
![mypy](https://img.shields.io/badge/mypy-strict%20clean-2ea44f)
![register](https://img.shields.io/badge/register-6%20owned%20%C2%B7%2033%20tied%20%C2%B7%207%20lost-555)
![languages](https://img.shields.io/badge/answers%20in-8%20languages-555)
![MCP](https://img.shields.io/badge/MCP-10%20tools-555)

**A research desk for Bitget's tokenized US stocks, and its crypto, indices and commodities too.
Ask in plain language, new to trading or not; every number is computed from live data and names
its source.**

**Live console: https://deploy-topaz-seven-64.vercel.app** · [one research task, run
live](https://deploy-topaz-seven-64.vercel.app/research) · [what we beat](https://deploy-topaz-seven-64.vercel.app/proof)
· [what we got wrong](https://deploy-topaz-seven-64.vercel.app/wrong)
· [status](https://deploy-topaz-seven-64.vercel.app/status)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/console-answer-dark.png">
  <img alt="The ARGUS console answering 'I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?': a sized, actionable answer, the risk it adds, a hedge, stress cases, how far to trust the beta against a named rival, and a receipt naming every source." src="docs/img/console-answer-light.png">
</picture>

Built for Bitget AI Base Camp / Genesis Hackathon Season 2. Nothing on this page needs an account
or a key to try.

> **Read this before judging.**
> - ARGUS is this team's **Track 3 — AI Trading Desk** entry, a natural-language research
>   workbench. The team's Track 2 entry is a **separate project**,
>   [t2-sentiment-agent](https://github.com/Pratiikpy/t2-sentiment-agent), with its own code,
>   paper-trading ledger, demo and form (Basic Competition Rules, rule 2). Nothing here is part of it.
> - It is **not a signal service**. No strategy in it has cleared its own deflation gate, and the
>   console says so; the trader makes every call.
> - Every capability is measured against a named rival on the same input, and the losses are
>   published beside the wins: [the full register](#what-we-beat-and-what-beat-us) and
>   [`/wrong`](https://deploy-topaz-seven-64.vercel.app/wrong).

---

## Head to head with Season 2

Every rival below was cloned or called and run on the same input as ARGUS. Ties and losses are
kept; the full table, rival by rival, is on [`/proof`](https://deploy-topaz-seven-64.vercel.app/proof).

| Rival | What it does | Result |
|---|---|---|
| gloaming (S2) | estimates the open while the US market is shut | missed by 81bps over 113 nights and 8 stocks against 30bps for ARGUS (93% of directions right) and 90bps for no gap; ahead on all 8 stocks, narrowest on QQQ (12.5 against 14.3bps). The stock's own pre-market price at 09:00, the general tool, missed by 36bps; ARGUS is ahead of it on all 8 stocks |
| Rook (S2) | ends each thesis with an invalidation price | its stops, 0.1-1.7% away, were reached on 62% of held-out days against 11% for ARGUS's fitted stop: ahead on that measure, six names, one run |
| baserate (S2) | prices a leveraged weekend hold | ARGUS rebuilt to read every NVDA weekend since 1999 (1,444 as of October 2026) and Bitget's live margin tier; its regime match scored worse than ARGUS's volatility-scaled band over 17,044 weekends, but baserate's own forecasts are not yet scored on the same weekends: a tie |
| MirrorLine (S2) | checks a trader's claims about the tape | first run 18 of 38 verdicts to its 38; fixed the same morning, re-run tied at 30 of 30 gradable claims |
| optic-bitget (S2) | debates a thesis with a model judge | abstained on two of three theses for want of earnings and positioning data ARGUS answered; still gives one synthesised call ARGUS does not: a tie |
| Ballast (S2) | hedges an rToken holder's nights | beat ARGUS's index hedge on 2026-09-24; now a tie |
| nocturne (S2) | fades the rToken's weekend move | level on its own question (1.92% against 2.07%); on the stock's real Monday open the weekend move carried through (slope +0.83, 160 stock-weekends) |
| Bitget's 60-second TWAP | slices an order | beat the console's schedule (6.9 against 12.2bps on $100k) on 2026-09-24; now a tie |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/proof-dark.png">
  <img alt="The /proof page: 46 capabilities measured against a named rival — 6 OWNED, 33 TIED, 0 IMPLEMENTED, 7 LOST — each linked to its evidence." src="docs/img/proof-light.png" width="720">
</picture>

---

## Where each judging line is answered

| Track 3 judges | Where to look | What to check |
|---|---|---|
| **Feature depth** — data sources, Skill integration count and effectiveness | [Bitget's toolkit, used and measured](#bitgets-toolkit-used-and-measured); `/status` | Which Bitget surfaces answer today and which do not, counted live; data breadth against OpenBB on the same stocks the same day |
| **Research quality** | [`/research`](https://deploy-topaz-seven-64.vercel.app/research); [what we beat](#what-we-beat-and-what-beat-us) | One full research task from question to verdict, eight engines side by side; every capability graded against a named rival |
| **LUI fluency** | [Try it in a minute](#try-it-in-a-minute); the console | Plain questions, slang, typos, follow-ups, Chinese and seven other languages; 81.7% of 240 blind questions read correctly with no model |
| **Personalised thesis** | [It remembers you](#it-remembers-you); ask *test my thesis* | A stated thesis tested reason by reason against live data; your loss limit, horizon, account and book applied to every later answer |

---

## Try it in a minute

Open the console and ask any of these. Each one exercises something different.

| Ask | What it shows |
|---|---|
| *I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?* | Your book's risk before and after, sized to a risk budget, a hedge, stress cases, and how far to trust the beta — scored against weekend-copilot, an S2 rival answering the same question (the answer calls it the best other tool measured; [/proof](https://deploy-topaz-seven-64.vercel.app/proof) names it). |
| *I'm bullish on NVDA because AI capex keeps accelerating. Test my thesis.* | Each reason tested on its own measurement — hyperscaler capex and NVDA's revenue from SEC filings; ask *what would prove it wrong?* next. |
| *How should I split a $50k order in NVDA?* | An execution schedule priced on the live order book, measured against Bitget's own TWAP on a full-depth replay, and the first child written as the Agent Hub `bgc --dry-run` command that previews it without sending. |
| *Long rNVDA over the weekend at 3x* | The stock's weekends since 1999, a band for Monday's open scaled to its volatility now, which covered 79% of 17,044 weekends out of sample against an 80% target, and the perpetual's own path to the liquidation line. |
| *Where will NVDA open?* | While the US market is shut: the price the stock's own perpetual implies for the next open, with that reading's measured miss beside gloaming's (an S2 rival) and a no-gap baseline; while the market is open it says when the next open is. |
| *I have $1,000 and I'm new — where should I start?* | No pick: what that sum went through in a year in three broad markets, and how to place a first trade on Bitget. |

<details>
<summary><b>Thirteen more questions, one per engine</b></summary>

| Ask | What it shows |
|---|---|
| *What are my sector and factor exposures if I add 10% XOM?* | Sectors, industries and eight factor betas with their t-statistics — market, size, momentum, value, quality, low volatility, rates and crypto, widened from four after the eight explained 9 of 13 names better out of sample. |
| *Review my trades: bought NVDA at 188, sold at 176 …* | Your habits across the trades and a checklist, kept and run again the next time you ask about adding a name. |
| *What's trending right now?* | The coins drawing the crowd, checked against Bitget's board: listed or not, the 24-hour move, the funding rate. Also: *top DeFi protocols by TVL*, *how much is ETH gas*. |
| *How does NVDA react to CPI?* | An event study over past releases, with the tests that say whether the reaction is real. |
| *Is the hype on NVDA real?* | Headlines grouped into stories, so five outlets repeating one article count once. |
| *I hold NVDA r-token overnight — how do I hedge it?* | The spot rToken hedged with the same company's perpetual, tested on held-out nights. |
| *Long MSTR perp into earnings — funding looks cheap* | Your own premise tested first — funding against the contract's last 100 settlements — then the earnings half: report date, analyst targets, the last surprise, holders, filings. |
| *Will MSTR be higher in 48 hours?* | Not a forecast: how often it finished higher over every past 48-hour window, with an interval that counts overlapping windows honestly, the cost of holding, and what Polymarket prices. |
| *What if I trim TSLA to 10% of my book?* | A resize, not an add: every other holding rescaled, risk share before and after, and the weight that fits your budget. |
| *ETH open interest* | How much is held open, how many days of today's volume that is, and where it ranks among every liquid Bitget perpetual. |
| *What is the long/short ratio on SOL?* | Bitget's own split for any perpetual: the share of accounts long, its change on the day, and the share of position size — the crowd against the larger money. |
| *How much does TQQQ decay if QQQ goes sideways for a month?* | What the fund actually lost against 3× its index in every sideways month of its history, beside the compounding formula at today's volatility. |
| *Should I hedge with gold or with TLT?* | The hedges you named measured against your book first, then the listed leg that does the job better, sized and costed. |

</details>

### Verify it in 60 seconds

No account, no key, nothing to install but `curl` and `jq`. Each line checks one claim on the live
deployment.

```bash
B=https://deploy-topaz-seven-64.vercel.app
# 1. A live answer from Bitget's own market data, not a cached demo
curl -sG "$B/ask" --data-urlencode "q=What is BTC trading at right now?" | jq -r '.lines[0]'
# 2. The research desk's decision record: entries, and whether its hash chain is intact
curl -s "$B/status?format=json" | jq '{entries, chain_intact, stale}'
# 3. The MCP server any agent can call, and its ten tools
curl -s -X POST "$B/mcp" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq -r '.result.tools[].name'
```

**Who it is for:** a trader holding or weighing Bitget's tokenized US equities who wants a second
desk that shows its work, and a newcomer who wants to see what a sum of money would have been
through before risking it. **Who it is not for:** anyone wanting signals or trades placed for them —
it places no orders, holds no money and says so.

### New to trading? Start here

The console's home page has a **New to trading** row of five questions: *I have $1,000, where
should I start?*, *what is a stop loss and do I need one*, *is 10x leverage ok for a small
account*, *how much could I lose on BTC in a bad week?* and *what is this site and who is it for*.
Each works as written, in plain words, typos and all, and so do the follow-ups a newcomer reaches
for:

- *is that good?* after a loss figure says plainly that it is a loss and what share of the money
  it is; *explain simpler* restates the last answer's terms in plain words.
- *where should my stop go on TSLA?* gives a price to put it at, and why closer gets stopped out
  by ordinary moves.
- *so what should I do?* gets no pick, but three checks that make the choice yours.

Put what you hold in **My book** (`bought 3 NVDA at 180, 0.01 BTC`) and *how's my stuff doing?*
answers against what you paid. Never give anyone your login or API keys; this console never asks
for them and cannot trade.

---

## What makes an answer trustworthy

### Every line says where it came from

Every line of an answer is tagged with where it comes from: **live** (read just now),
**computed** (worked out for this answer), **record** (a measured past result, its sample
named), **desk** (quoted from the agent's logged decision), **assumed** (a default the question
left open) or **missing** (could not be read). On 18 questions never seen while the tags were
written, 94% of content lines carried one; a line no rule recognises carries none rather than a
guess (`python -m argus.eval.provenance_audit`). Every answer ends by naming the sources it reached
and any that did not answer. If the desk cannot source a figure, it refuses and says why.

### It remembers you

Put your holdings in **My book** once and every answer uses them. Tell it about yourself — *I
can't lose more than 10%*, *I'm a swing trader*, *I think NVDA runs on AI capex* — and it remembers,
in your browser only: later answers apply your loss limit, holding period and risk budget, and show
your thesis beside every answer about that name with the move since you stated it, each time on a
**memory** line saying so. Measured on 20 two-session tasks: 20 answers changed by memory, none
with memory off, none reaching another trader (`python -m argus.eval.memory_eval`).

### It answers in your language

Ask in Chinese, Japanese, Korean, Spanish, Portuguese, French or German and the answer comes back in
that language on the console page: the engines write English, which shows at once, and Qwen's
translation replaces it a few seconds later (the page's second call, `/translate`); every number
in each translated line is checked against the English, and a line whose figures do not match
stays English. A direct `POST /ask` returns the English with a one-line note in the asker's
language, and the signed token that `/translate` accepts.

### It is in Telegram, and you can run your own

The same desk answers in Telegram at [@argusbitgetbot](https://t.me/argusbitgetbot): `/book` saves
your holdings for the chat, and orders are refused there as everywhere. Like OpenClaw or Hermes
Agent, it also runs as your own Telegram bot, on your machine, with your token — nothing of yours
passes through our servers:

```bash
pip install -e argus                                  # or: uv sync --frozen, in argus/
argus-bot setup --token <token from @BotFather>       # checks it, keeps it in ~/.argus
argus-bot run                                         # long polling: no server, no public URL
```

Message your bot once and approve the code it gives you (`argus-bot pairing approve <CODE>`): you
become its owner, and a stranger gets a code and nothing else until you approve them too
(`--policy allowlist --allow <id>` locks it to named user ids instead). Your book, what you told
the desk and your alerts stay in `~/.argus` across restarts, and alerts fire while it runs:
`/watch NVDA below 170`, `/watch MSTR funding 0.05%`. No model key is needed; the desk reads
questions with its own classifier.

### Any agent can call it

ARGUS is also a **Model Context Protocol server**: any MCP client can add
`https://deploy-topaz-seven-64.vercel.app/mcp` (Streamable HTTP) and call ten tools — `argus_ask`
(the whole console, any language), `argus_quote`, `argus_portfolio_impact`, `argus_stress` (a shock
on the Nasdaq or on any instrument, long or short book), `argus_execution_plan`,
`argus_scoreboard`, `argus_research_task` (the eight-engine report), `argus_week_ahead`,
`argus_exposures` and `argus_review_trades`. They run the same engines as the page; none of them
writes or trades.

---

## What we beat, and what beat us

Every capability is graded against the specialist that leads its sub-theme, by a register that
opens the evidence rather than trusting a filename (`python -m argus.eval.standing`), and may claim
OWNED only after that rival has been run on the same input and beaten. OWNED needs all thirteen
conditions: the rival's best implementation read and reproduced, a same-input comparison with
costs, out-of-sample, ablation, an adversarial test, documented failure cases and reproducibility.

**6 of 46 capabilities are OWNED, 33 are TIED, 0 are IMPLEMENTED, and 7 are LOST.**

| OWNED | Against |
|---|---|
| Earnings-surprise ranking | QuantConnect's own formula on 7,258 SEC filers with a holdout the rule was never fitted on, and the real edgartools package |
| Research workbench data breadth | OpenBB's keyless providers, on two days and on the categories both can reach |
| Point-in-time filings | OpenBB's platform, edgartools and Qlib, on 619 real questions |
| The perpetual-implied overnight open | gloaming as a trade entered at the 09:00 price, net of costs, and the stock's own pre-market price |
| Per-profile mandate that changes the verdict | Vibe-Trading, the one system of sixteen whose mandate changes a verdict on identical state |
| Risk layer proved by domain sweep | Nautilus Trader's risk engine, QuantConnect's brokerage models and freqtrade's protections |

| LOST | Lost to | By how much |
|---|---|---|
| Crowd sentiment classification | RoB-RT on TweetEval | 0.544 against 0.729 macro-recall on the same 12,284 tweets |
| Session-aware execution | Bitget's own TWAP on replayed order books | 0.62 bps dearer on boundary-crossing orders, interval [0.01, 1.53] |
| The price of thinking time | a trailing realised-move estimator, on 23,003 real decision instants | production's excess error 9.89, interval [9.20, 10.57] |
| Cointegration screening | 128 general pipelines | its false-discovery rate misses its own target where they hold it |
| Perception breadth (the trading desk's own feed) | OpenBB's keyless providers | 12.4 data categories per stock against the desk's 6.2, same stocks, same day |
| Path-shape matching | a stock's own unconditional range | forecast band worse on 2,698 held-out queries (Winkler 16.4% against 15.1%, p = 0.001) |
| Injection screening on attacks it was not written for | a trained classifier | it withholds 41% of an external corpus's attacks to ARGUS's 16% (while withholding 12.4% of the desk's real text, which ARGUS never does) |

Losses are published the moment they are found, on
[`/wrong`](https://deploy-topaz-seven-64.vercel.app/wrong). The register has been made stricter
three times, and the OWNED count fell each time before it rose again on new evidence; that history,
date by date, is in [`ARGUS-EXPLAINED.md`](ARGUS-EXPLAINED.md#how-the-register-got-stricter).

---

## Bitget's toolkit, used and measured

| Surface | Where ARGUS uses it | Measured |
|---|---|---|
| Public market API (v2 market data, v3 for the account and order preview) | Candles, tickers, 50-level books, funding — every price and cost in the console | Answering, checked live on `/status` |
| Order-book websocket (`books5`, trades) | Execution replay and the order-splitting comparison | Recorded locally, RFC 6455 client |
| `bitget-mcp-server` | US fundamentals, 13F holders, analyst estimates, earnings calendar, the stock behind each rToken | 37 of 67 catalog entries answer in the latest sweep (`data/data_coverage.json`, dated inside and on `/status`); on 26 Sep every entry refused while its upstream answered 503 |
| `bitget-signal` Skills | The console asks the Skill first for technicals (MACD corrected against Bitget's own candles), crypto sentiment and the rates curve, and checks BTC against `crypto_derivatives` | 2 of bitget-signal's 19 tools answered all three attempts on 2026-09-26 (crypto_derivatives, technical_analysis), from 1 of its 5 Skills (technical-analysis) plus 1 tool no SKILL.md names; when a Skill is silent the answer names the public source it read instead |
| Agent Hub (`bgc`) | The first child of every execution plan, written as the `--dry-run` command that previews it | Mapping checked against `bgc`'s own dry-run output (`data/agenthub_preview.json`) |

When a Skill's hosted server fails, ARGUS reads the source that Skill names (alternative.me,
Binance futures data, FRED, Yahoo, DeFiLlama, CoinGecko, the RSS feeds). Every such answer says
it came from the source, not the Skill, and the two counts are never merged
(`argus/src/argus/market/skill_mirror.py`).

**How much it can see, against OpenBB, measured on the same stocks the same day.** Every keyless
OpenBB provider was installed and called through OpenBB's own interface for the ten stocks behind
the rTokens, beside ARGUS's own answers (`argus/src/argus/eval/perception_breadth.py`). Counted as
data categories that came back with data, the research workbench answers **16.0 per stock against
OpenBB's 12.4**, ahead on 9 of 10 and level on QQQ; read again the next day with nothing refitted,
15.3 against 11.7 and ahead on all twelve underlyings (+3.6 over both days, bootstrap interval +3.1
to +4.1, `argus/data/groupwise_audit.json`). Most of the lead is what OpenBB's free tier has no
endpoint for (X and Reddit, Bitget funding and positioning, prediction markets); scored only on the
categories OpenBB can reach, the workbench is still ahead by 1.0 and 1.2 a stock (interval +0.7 to
+1.5), from the upcoming earnings date and corporate actions. The trading desk's own evidence panel
is still narrower, **6.2 against 12.4**, and that loss is on the record. With paid keys OpenBB is
wider; those twelve providers are named, not counted.

---

## Run it yourself

```bash
cd argus
pip install -e ".[dev]"
python -m argus.status          # module and sub-theme coverage, resolved by import
python -m argus.lui.server      # the console on http://127.0.0.1:8765
pytest -q                       # 11,615 tests collected
```

Nothing above needs a credential. On 2026-10-03, with every outbound connection refused, 11,012
passed, 75 skipped and 0 failed of the 11,615 tests collected; the other 238 read a live
venue, feed or model and run apart
(`pytest -m network`). Tests that need a rival's source cloned beside the repository skip
and say which. `ARGUS_BLOCK_NETWORK=1` refuses every outbound connection, so
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
| Tests | **11,615 tests collected** — `pytest -q` |
| Types | **`mypy --strict` clean on 597 source files** |
| Lint | `ruff` clean |
| Modules | **152/152 modules importable**, checked by `python -m argus.status` |
| Sub-themes | **18/18 sub-themes** each resolve to an importable module and a test file (`python -m argus.status`). That is coverage, not a claim to lead them: 6 of 46 capabilities are OWNED against a named rival, and `/proof` says which |
| Register self-check | **Not clean, and said so**: `data/standing.json` lists 52 conditions claimed but not evidenced, every one on a TIED (41) or LOST (11) row; none on the 6 OWNED rows, which is what OWNED requires (`python -m argus.eval.standing`) |
| Understanding | **81.7%** of 240 questions in 12 languages, written blind by an agent that never saw this repository and scored once, were read correctly by the console with no language model (52.1% before the 2026-09-25 fixes); with Qwen reading first, as on the live site, 85.0% — `data/lui_final_heldout_report.json`. Which engine each question reaches is re-scored on every change by `eval/kind_routing.py` |
| Quoted figures | Every figure these documents quote is re-checked against its artefact by `python -m argus.eval.docclaims --tests`, which fails if one has drifted |

## What the code will not let happen

- **A zero-fee backtest cannot be built.** `CostModel` raises rather than defaulting to free.
- **Omitting `as_of` is a `TypeError`.** Point-in-time evidence cannot be queried carelessly.
- **The risk layer may only reduce.** It cannot create, flip or grow a position.
- **A maker fee needs a book.** Passive fills go through a queue model ported from hftbacktest.
- **Search cannot see evaluation.** `ProposerContext` has no field that can hold a score.
- **The ledger is hash-chained**, and a tampered chain is refused rather than scored. A chain we
  write we could also rewrite, and once we did: on 2026-09-12 two concurrent runs duplicated seven
  sequence numbers at entry 41, and the repair renumbered them and recomputed their links, with the
  before-image and every changed link published in `argus/data/paper_ledger_incidents.json`. So the
  chain's head also goes to four OpenTimestamps calendars, Bitcoin-backed, with every register
  anchor (`argus/data/anchors/`); a later edit to an anchored row no longer matches a dated proof.
  A calendar's proof is pending until Bitcoin confirms it, a few hours after submission, so the
  newest anchor is usually still pending: `python -m argus.register.anchorcheck` says which are.

---

## Honest limits

- **ARGUS's own paper desk has settled 2 trades, both losses.** BUY 1 NVDAUSDT, opened 2026-09-28
  at 232.45 and closed 2026-09-29 at 230.96: a net loss of 1.78 USDT, 0.29 of it costs. SELL 1
  METAUSDT, opened 2026-09-30 at 728.17 and closed 2026-10-01 at 731.55: a net loss of 4.45 USDT,
  1.07 of it costs. Two rows (seq 264, 265) are void: they record fills the risk layer had
  refused. Every other decision on its ledger is a refusal. Two settled trades are not a
  sample, so its Sharpe and drawdown are undefined and printed as such, and its win rate of 0%
  says nothing yet. Each refusal carried a direction, hashed before the
  outcome existed: at about two hours, **331 of 598 directional calls were right**, which clears a
  coin flip and does not beat calling "up" every time. The median refusal **forgave -9.01bps of
  net edge** after the 12bps round trip — the trades it passed
  on were mostly unprofitable. This workbench does not trade; trading is the separate Track 2
  project's job.
- **No certified alpha.** 0 of 8 factors and 0 of 12 strategies cleared the deflation gate.
- **Price discovery attenuates about 7x while the anchor market is shut — it does not stop.** The
  weekend effect failed its own out-of-sample split, and the 3–5x off-hours spread widening first
  claimed here was falsified. Both are reported as negatives.

---

## 中文简介

ARGUS 是面向 Bitget 美股代币（rToken）的研究工作台（Track 3 · AI Trading Desk）。用自然语言提问，
八个引擎基于 Bitget 行情、SEC 文件、FRED 与新闻实时计算，每个数字都注明来源；语言模型只理解问题，
从不编写数字。全部 46 项能力均已与各子赛道领先的专业系统在相同输入上对比：6 项领先（OWNED）、33 项
持平、0 项已实现、7 项落后。落后的七项：

- 人群情绪分类：在 TweetEval 上不及 RoB-RT；
- 跨时段执行：在回放订单簿上不及 Bitget 自带的 TWAP；
- 思考时间定价：在 23,003 个真实决策时点上不及滚动实现波动估计；
- 协整筛选：错误发现率未达自身目标；
- 数据覆盖面（交易台自身数据）：不及 OpenBB 的免密钥数据源；
- 走势形态匹配：预测区间不及个股自身的无条件区间；
- 对未针对性编写的注入攻击：拦截率不及训练好的分类器。

所有落败记录公开在 `/wrong`。本团队的 Track 2 参赛作品是另一个独立项目
（t2-sentiment-agent，独立的代码库、交易日志与演示），不属于本仓库。

---

## Licence

MIT. Components ported from other projects cite their source file in the docstring, and each
source's licence, with what was taken from it, is in `argus/licenses/`. Everything copied into the
product is MIT, BSD or Apache licensed; one evaluation baseline, vectorbt's deflated-Sharpe code, is
Apache-2.0 with the Commons Clause, runs only as a rival in `eval/` and must be dropped before any
commercial redistribution. Anything under a copyleft licence was rebuilt from its described
behaviour rather than copied.
