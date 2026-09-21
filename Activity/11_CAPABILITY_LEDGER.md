# Capability ledger — one row per atomic part, one named rival each

**The unit of work is one capability x one rival x one measurement.** Not one repo, not one
feature. the owner's own framing: find the specialist who does *exactly this one thing*, run it on
our input, and publish who won.

These 20 parts are **Bitget's taxonomy, not ours** — the six sub-themes and four judging criteria
per track, straight out of the handbook. That is deliberate: a decomposition we invented would
flatter whatever we happen to have built.

**Status vocabulary (never conflate):** LOST · TIED · IMPLEMENTED · OWNED. OWNED requires all
thirteen conditions in `eval/standing.py`, of which the binding one is: *the named baseline was
actually reproduced and the comparison actually run.* **Nothing is OWNED until the rival column
carries a result.**

| # | part | tr | kind | our status | named rival | rival run | who won | last checked |
|---|---|----|------|-----------|-------------|-----------|---------|--------------|
| 1 | `event-driven` | T2 | subtheme | IMPLEMENTED | whale-signals | **yes** | methodological — see note | 2026-09-21 |
| 2 | `sentiment` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 3 | `earnings` | T2 | subtheme | IMPLEMENTED | QuantConnect (vendored SUE) | **yes** | ARGUS (refuses the artefact QC ranks top) | 2026-09-21 |
| 4 | `cross-asset-execution` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 5 | `factor-discovery` | T2 | subtheme | IMPLEMENTED | Microsoft RD-Agent | **yes** | ARGUS (no execution surface; RD-Agent ran attacker code) | 2026-09-21 |
| 6 | `t2-open-evaluation` | T2 | subtheme | IMPLEMENTED (weak) | — | no | — | — |
| 7 | `t2-sharpe-mdd-winrate` | T2 | judging | NOT DEMONSTRATED | — | no | — | — |
| 8 | `t2-explainability` | T2 | judging | IMPLEMENTED | — | no | — | — |
| 9 | `t2-risk-control` | T2 | judging | IMPLEMENTED (weak) | freqtrade PrecisionRecallProtection | **yes** | ARGUS (precision 59.2% vs 19.3% at full recall, dominates every threshold) | 2026-09-21 |
| 10 | `t2-architecture` | T2 | judging | IMPLEMENTED | RD-Agent + AgentDojo (two rivals) | **yes** | ARGUS (no execution surface vs RD-Agent's real `eval()`; 0 production FPs after fix vs 6 before) | 2026-09-21 |
| 11 | `info-extraction` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 12 | `review-self-evolution` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 13 | `stress-testing` | T3 | subtheme | **LOST** | stumpy FLUSS + ruptures + incumbent rule | **yes** | specialist | 2026-09-21 |
| 14 | `personal-workbench` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 15 | `execution-assistance` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 16 | `portfolio-copilot` | T3 | subtheme | **LOST** | Riskfolio-Lib NCO | **yes** | specialist (p=3.6e-05) | 2026-09-21 |
| 17 | `t3-source-depth` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 18 | `t3-research-quality` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 19 | `t3-lui` | T3 | judging | **LOST** on accuracy, **WON** on safety | RasaHQ/rasa DIET (Apache-2.0) | **yes** | split — Rasa (accuracy, p=0.0014); ARGUS (OOS refusal, p=0.031) | 2026-09-21 |
| 20 | `t3-personal-thesis` | T3 | judging | IMPLEMENTED | — | no | — | — |

## The column that decides everything

**`rival run` is `yes` on 8 of 20 as of 2026-09-21: 3 LOST (one of them split — LUI intent
accuracy loses to Rasa's DIET classifier, p=0.0014, while ARGUS wins out-of-scope refusal
separately and significantly, p=0.031, neither laundering the other), 4 real wins (earnings vs
QuantConnect, factor-discovery vs Microsoft RD-Agent, risk-control vs freqtrade's real
PrecisionRecallProtection — ARGUS's precision at full recall is 59.2% against freqtrade's best
swept 19.3%, dominating every threshold, and freqtrade's own proxy correlates only weakly
(r=0.26) with the real drawdown ground truth it approximates — and architecture vs two rivals at
once, RD-Agent's real `eval()`-executed injection against ARGUS's zero execution surface, plus
AgentDojo's real attack corpus behind the production false-positive count that went from 6 to 0),
1 methodological (event-driven vs whale-signals — neither system establishes a real effect on
this data; the finding is that whale-signals tests against the wrong null, not that ARGUS's own
number beats theirs, and no fabricated score is recorded for it).** That is this ledger's honest
scoreboard — it names whether
*this specific Bitget-taxonomy row* has a rival comparison wired in, which is a narrower and
different question from `eval/standing.py`'s own register (27 capabilities across the whole
codebase, 17 OWNED, 3 LOST, independently verified — do not read "0" here as "0 OWNED anywhere,"
an earlier session conflated the two and said so wrongly to the owner). What this ledger's
`rival run` column still says: of 20 judged rows, 13 more already
have a real comparison sitting in `data/` unwired (found by the CONNECT lens, 2026-09-21 — see
`Activity/10_LOOP_DESIGN.md`). Every other number in this
project — 60 commits, 5,000+ tests, 96 pinned doc claims — measures how carefully we checked
*ourselves*; wiring the remaining 16 rows is the highest-value CONNECT work still queued.

## MISSING — parts no row covers

Grown from evidence, never from imagination. **When a rival is read, record what it does that we
do not do at all.** A feature we never thought of does not appear above as LOST; it does not
appear at all, and that is the single most likely way this ledger is wrong.

**9 comparisons on disk map to no Bitget-taxonomy row** (all Track-1 or cross-cutting; found by the
CONNECT lens, 2026-09-21): `arbitrage_comparison.json` (maxme/bitcoin-arbitrage), `rotation_comparison.json`
(pytaa), `cointegration_comparison.json` (statsmodels/Lean/FinceptTerminal), `afterhours_comparison.json`
(Lean market-hours), `crosssection_comparison.json` (qlib CSRankNorm), `dsr_comparison.json` (vectorbt
deflated Sharpe), `factor_divergence_comparison.json` (Alphalens/WorldQuant Alpha#101), `deliberation_comparison.json`
(LatencySensitiveBench, arXiv 2505.19481), `search_bakeoff.json` (internal, no external specialist).

**Genuine capability gaps — no comparison exists at all:** `t2-explainability` (no rival for
reasoning-trail quality), `t2-sharpe-mdd-winrate` (structurally blocked — the desk has no settled
trades, not a missing comparison). `t3-lui` closed 2026-09-21 — see row 19 and
`Activity/10_LOOP_DESIGN.md` iteration 6.

**Closed 2026-09-21 — the three never-run modules.** `feedlist_comparison.py`,
`grammar_comparison.py`, `journal_comparison.py` all ran clean and produced real, substantive
first-ever artefacts (Windows CRLF caused the rival journal library to false-positive a tamper
alert on untampered data; the rival factor-eval library executes real injected code via `eval()`
where ARGUS's grammar parser has no execution surface at all). Two `costs_included` citations
upgraded from ATTESTED (cited the module's own source) to VERIFIED (cited the new artefact) — and
in doing so caught a real, previously-unchecked wrong number: `journal_comparison`'s citation
claimed "~0.014s to verify 250 entries," never checked against a real run; the real figure is
0.021s, about 50% off. Fixed. `failure_cases_documented` on the journal capability stays ATTESTED
on purpose — the new artefact's keys genuinely don't carry that condition's vocabulary, and
forcing the citation would have turned ATTESTED into UNPROVEN, a regression, not an improvement.

## AUDIT sweep, 2026-09-21 — internal contradictions found and fixed

A full sweep of the 26 `SCOPE_STATEMENT`-bearing comparison modules (dispatched from the CONNECT
lens; processed in the next AUDIT iteration — full detail in `Activity/10_LOOP_DESIGN.md`, iterations
2–3) found and fixed **five modules carrying a stale or self-contradictory claim in their own
shipped artefact**: `sentiment_comparison.py` (repeated a finding `standing.py` had already
reversed), `mandate_comparison.py` (three different counts for one quantity, plus two wrong
`desk.py` line citations — backs an already-OWNED capability), `dsr_comparison.py` and
`crosssection_comparison.py` (hardcoded sweep sizes that drifted after a grid dimension was
added — both converted to functions fed by the live count, matching last iteration's
`quarantine_comparison.py` fix), and `review_comparison.py` (a real coverage gap — "four
thresholds ablated" claimed, only two actually were). **The queued BUILD work is done**: the
missing two (`NEVER_FIRES`, `MIN_FIRINGS`) were implemented in iteration 5, both verified to flip
against the live artefact, "four thresholds" is now genuinely true. `sentiment_comparison.json`
could not be regenerated — its module needs
`BITGET_QWEN_API_KEY`, genuinely tested and genuinely blocked by the standing rule against
spending that metered key from a loop.

**Resolved 2026-09-21, the "hung" background suite.** Two F's had sat unidentified for four
iterations. Batched, individually-timed diagnosis (queued three iterations ago, run this
iteration) found: one genuine bug — `test_regime_groundtruth_audit.py` was a deliberate tripwire
asserting `regime_comparison.json`'s scope_statement was stale, and this loop's own earlier fix
(regenerating that artefact) correctly flipped the tripwire without anyone completing its own
documented follow-up. Fixed exactly as the test's own docstring instructed. The rest of the
slowness is not brokenness: `test_regime_comparison.py` and `test_riskproof.py` are each
individually 5-10+ minutes of legitimate heavy computation (matrix-profile sweep; a 2.18M-state
risk-domain sweep), pre-existing and untouched by this session.

**Still queued from the same sweep:** `arbitrage_comparison.py` (flagged as the highest-risk
latent case — six live-data literals in one prose block, methodology switched from signed to
absolute basis), `factor_divergence_comparison.py`, `eventdriven_comparison.py`,
`cointegration_comparison.py`, `rotation_comparison.py` (all need a re-run to confirm no further
drift, none confirmed broken yet).

## Live-surface defects found by the JUDGE lens

| surface | defect | found |
|---------|--------|-------|
| repo visibility | `github.com/Pratiikpy/argus-bitget` is **PRIVATE** (`gh repo view` confirms `isPrivate: true`) — a judge cannot clone it at all, before anything else matters | 2026-09-21 |
| fresh install | `git clone` + fresh venv + `pip install -e ".[dev]"` from a genuinely empty machine state: **clean, exit 0** | 2026-09-21 |
| `/status` | serves 447 entries against a local 507; reports `stale: true` about itself | 2026-09-21 |
| `/wrong` | **404** — the corrections page is absent from the deployed build | 2026-09-21 |
| `/research` | **404** — the research surface is absent from the deployed build | 2026-09-21 |
| `abstention_why` answer | Two of three reasoning blocks in one real answer cut off mid-word ("...unlikely to m"). Traced to `paper/ledger.py`'s `thesis[:500]` — silent, no ellipsis. **350 of 519 ledger rows (67%) affected.** Fixed forward-only (historical rows are hash-chained, cannot be repaired) — `MAX_THESIS_LENGTH` raised to 4000, documented, tested. | 2026-09-21 |

`argus/README.md:211` claims the console is live and working. Two of its five routes are not.
