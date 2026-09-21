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
| 9 | `t2-risk-control` | T2 | judging | IMPLEMENTED (weak) | — | no | — | — |
| 10 | `t2-architecture` | T2 | judging | IMPLEMENTED | — | no | — | — |
| 11 | `info-extraction` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 12 | `review-self-evolution` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 13 | `stress-testing` | T3 | subtheme | **LOST** | stumpy FLUSS + ruptures + incumbent rule | **yes** | specialist | 2026-09-21 |
| 14 | `personal-workbench` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 15 | `execution-assistance` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 16 | `portfolio-copilot` | T3 | subtheme | **LOST** | Riskfolio-Lib NCO | **yes** | specialist (p=3.6e-05) | 2026-09-21 |
| 17 | `t3-source-depth` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 18 | `t3-research-quality` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 19 | `t3-lui` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 20 | `t3-personal-thesis` | T3 | judging | IMPLEMENTED | — | no | — | — |

## The column that decides everything

**`rival run` is `yes` on 5 of 20 as of 2026-09-21: 2 LOST, 2 real wins (earnings vs QuantConnect,
factor-discovery vs Microsoft RD-Agent), 1 methodological (event-driven vs whale-signals — neither
system establishes a real effect on this data; the finding is that whale-signals tests against the
wrong null, not that ARGUS's own number beats theirs, and no fabricated score is recorded for it).**
That is this ledger's honest scoreboard — it names whether *this specific Bitget-taxonomy row* has
a rival comparison wired in, which is a narrower and different question from `eval/standing.py`'s
own register (26 capabilities across the whole codebase, 17 OWNED, 2 LOST, independently verified —
do not read "0" here as "0 OWNED anywhere," an earlier session conflated the two and said so wrongly
to the owner). What this ledger's `rival run` column still says: of 20 judged rows, 13 more already
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
reasoning-trail quality), `t3-lui` (no rival for conversational/LUI fluency), `t2-sharpe-mdd-winrate`
(structurally blocked — the desk has no settled trades, not a missing comparison).

**Three artefacts named as rivals in `eval/standing.py` prose have no persisted output on disk:**
`feedlist_comparison.py`, `grammar_comparison.py`, `journal_comparison.py` — the modules exist, the
comparisons apparently ran once (cited by name in standing.py), but `data/*.json` for none of them
exists now. Next AUDIT iteration: re-run all three and confirm they still work, or mark honestly.

## Live-surface defects found by the JUDGE lens

| surface | defect | found |
|---------|--------|-------|
| repo visibility | `github.com/Pratiikpy/argus-bitget` is **PRIVATE** (`gh repo view` confirms `isPrivate: true`) — a judge cannot clone it at all, before anything else matters | 2026-09-21 |
| fresh install | `git clone` + fresh venv + `pip install -e ".[dev]"` from a genuinely empty machine state: **clean, exit 0** | 2026-09-21 |
| `/status` | serves 447 entries against a local 507; reports `stale: true` about itself | 2026-09-21 |
| `/wrong` | **404** — the corrections page is absent from the deployed build | 2026-09-21 |
| `/research` | **404** — the research surface is absent from the deployed build | 2026-09-21 |

`argus/README.md:211` claims the console is live and working. Two of its five routes are not.
