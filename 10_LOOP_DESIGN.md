# The loop, redesigned — `Activity/LOOP_STATE.md`

The 5-minute cron is deleted. It failed in four specific ways, and each one has a fix.

| What broke | Why it broke | Fix |
|---|---|---|
| Circling — iterations 19 and 20 both found bugs in `docclaims.py`, a module we wrote | one prompt, fired forever, with no memory of what was last examined | **stance rotation enforced on disk**, not in my head |
| Zero new capabilities in 20 iterations | nothing in the prompt required leaving the repo | **two of four lenses are outward-facing** |
| Live console fell 60 decisions behind, unnoticed, for two days | the loop only ever read the repository | **JUDGE lens is mandatory in every cycle of four** |
| Progress measured in commits | commits are not the scoreboard | **OWNED count is the scoreboard**, still 0 |
| 5 minutes | a real specialist head-to-head takes an hour; the clock fired anyway | **self-paced**, fires when work is ready |

## The four lenses — rotate, never the same one twice running

1. **JUDGE** — open the live demo cold as a stranger. Drive one complete flow end to end,
   screenshot it, read it critically, check the source of truth behind the screen. Fix what it
   finds. *This is the lens that would have caught the stale console on day one.*
2. **RIVAL** — take one named specialist, install it, run it on our input, publish who won.
   **The only lens that can move a capability to OWNED**, because OWNED requires the baseline
   actually reproduced and the comparison actually run.
3. **BUILD** — add or deepen one capability. Track 3 scores *"feature depth (data sources / Skill
   integration count and effectiveness)"* by name, and we have added nothing in 20 iterations.
4. **AUDIT** — internal verification hardening. **Capped at one in four.** It is genuinely our
   strongest work and it is where the loop kept getting stuck.

## The scoreboard each iteration must move

- **OWNED capabilities: 0.** Against our own 13-condition standard. This is the real number.
- **Live-surface health** — every documented route answering, ledger not stale.
- **Judged-criteria coverage** — the four Track 3 criteria, each with a named rival and a result.
- **Blocked on the owner** — kept short, kept current, never silently grown.

## The rule that ends an iteration

A number on the scoreboard moved, **or** a written reason it could not. Never "committed some work."

## Pacing

Self-paced, not clocked. Each iteration schedules the next from what it actually started: a long
fallback when a build or backtest is running, ~20–30 minutes when idle. No wake-up that exists only
to prove the loop is alive.
