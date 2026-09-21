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
| 1 | `event-driven` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 2 | `sentiment` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 3 | `earnings` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 4 | `cross-asset-execution` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 5 | `factor-discovery` | T2 | subtheme | IMPLEMENTED | — | no | — | — |
| 6 | `t2-open-evaluation` | T2 | subtheme | IMPLEMENTED (weak) | — | no | — | — |
| 7 | `t2-sharpe-mdd-winrate` | T2 | judging | NOT DEMONSTRATED | — | no | — | — |
| 8 | `t2-explainability` | T2 | judging | IMPLEMENTED | — | no | — | — |
| 9 | `t2-risk-control` | T2 | judging | IMPLEMENTED (weak) | — | no | — | — |
| 10 | `t2-architecture` | T2 | judging | IMPLEMENTED | — | no | — | — |
| 11 | `info-extraction` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 12 | `review-self-evolution` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 13 | `stress-testing` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 14 | `personal-workbench` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 15 | `execution-assistance` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 16 | `portfolio-copilot` | T3 | subtheme | IMPLEMENTED | — | no | — | — |
| 17 | `t3-source-depth` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 18 | `t3-research-quality` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 19 | `t3-lui` | T3 | judging | IMPLEMENTED | — | no | — | — |
| 20 | `t3-personal-thesis` | T3 | judging | IMPLEMENTED | — | no | — | — |

## The column that decides everything

**`rival run` is `no` on all 20.** That is the honest scoreboard, and it is why OWNED is 0.
Every other number in this project — 57 commits, 5,000+ tests, 96 pinned doc claims — measures
how carefully we checked *ourselves*.

## MISSING — parts no row covers

Grown from evidence, never from imagination. **When a rival is read, record what it does that we
do not do at all.** A feature we never thought of does not appear above as LOST; it does not
appear at all, and that is the single most likely way this ledger is wrong.

_(empty — first RIVAL iteration fills it)_

## Live-surface defects found by the JUDGE lens

| surface | defect | found |
|---------|--------|-------|
| `/status` | serves 447 entries against a local 507; reports `stale: true` about itself | 2026-09-21 |
| `/wrong` | **404** — the corrections page is absent from the deployed build | 2026-09-21 |
| `/research` | **404** — the research surface is absent from the deployed build | 2026-09-21 |

`argus/README.md:211` claims the console is live and working. Two of its five routes are not.
