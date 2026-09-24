# Generated HTML reports: how the numbers get in, and what stops them going stale

**Question.** For a self-contained HTML report a reader opens with no server: how is it assembled,
how do figures get in, what happens to a metric that cannot be computed, and is there anything
stopping the page disagreeing with its own source of truth?

**Answer on the last point: nothing, in any of them.** Both serious implementations render once and
never check. Every citation below was opened.

---

## 1. `ranaroussi/quantstats` — the canonical tearsheet

`okx/trading/best-of-the-best/repos/quantstats/quantstats/reports.py`, with an on-disk
`quantstats/report.html` template.

* **Assembly:** the template is read as raw text and `{{key}}` placeholders are string-replaced.
  Unused placeholders are stripped with a regex afterwards.
* **Self-contained:** yes. CSS inline; charts embedded by `_embed_figure()` — SVG inlined as text,
  PNG/JPG base64-encoded into a `data:` URI.
* **Numbers:** computed at render time by `_get_stats()`, formatted, turned into tables by
  `_html_table()`, then injected.
* **Undefined metrics: `df = df.fillna(0)` at `reports.py:1289`.** A strategy with no trades reports
  a Sharpe of **0.0**. This is the single most important line in this teardown: it is the difference
  between "no result" and "a flat result", and the page cannot tell a reader which it is looking at.
* **Graceful degradation:** good — benchmark-dependent blocks are skipped entirely when
  `benchmark is None`, and DataFrame vs Series input branches to different tables.
* **Stale numbers:** no guard of any kind. The figures live in the rendered string.

## 2. `agent-backtest-lab` — the cleanest engineering of the two

`okx/trading/best-of-the-best/repos/agent-backtest-lab/abl/scorecard/html_render.py`.

* **Assembly:** programmatic. A list of `parts` appended and joined at `:230`; CSS is a `_CSS`
  string constant. No template file, so no file I/O at render time.
* **Numbers:** **passed in, not computed.** It receives a `Scorecard` whose fields are already
  calculated and only formats them. The separation is right and we copied it.
* **Undefined metrics:** explicit guards — `np.isfinite(...)` at `:123`, `:139` and
  `hr = f"{r.hit_rate:.3f}" if np.isfinite(r.hit_rate) else "—"` at `:160`. Honest, and **silent
  about why**: an em-dash reads as a formatting choice rather than as a finding.
* **Degradation:** conditional sections throughout, with a literal `<i>none provided</i>` fallback.
* **Escaping:** `_esc()` at `:49` on all user data.
* **Stale numbers:** no guard.

## 3. Others, briefly

* `freqtrade` — `optimize_reports.py` builds metric dicts and stops there; rendering is left to the
  web UI. Good preparation, no report.
* `qlib` — `contrib/report/analysis_position/report.py` emits Plotly graph objects, which need a
  notebook or a server. Not self-contained.
* `nautilus_trader` — `analysis/reporter.py` returns pandas DataFrames. No HTML.

---

## What ARGUS does instead

`argus/src/argus/demo/cockpit.py`.

1. **Data first, HTML second** — adopted from agent-backtest-lab and taken further. `build()`
   returns `Panel`/`Metric` objects; `render()` only formats them. A generator that interpolates
   numbers straight into a template can only be tested by parsing its own output, which is a test
   of the parser.

2. **`None` means unavailable, and the page says the words.** Not `0.0` (quantstats) and not an
   em-dash (agent-backtest-lab): the cell reads **"not available"**, styled as a finding, and
   `Cockpit.undefined` lists every such figure by name so the CLI prints them at build time. Track
   2's Sharpe, max drawdown and win rate are undefined because nothing has settled, and on this
   project's most-read page a `0.00` there would be its strongest false claim.

3. **The staleness guard neither has.** `tests/test_cockpit.py` rebuilds every figure from the
   artefacts and asserts each appears in the committed `cockpit.html`. Verified by corrupting the
   committed page — changing one ledger count from 178 to 2 — and confirming two tests fail by name.
   This exists because the hand-written page it replaces said *"The paper ledger is 2 decisions
   old"* while the ledger held 178, and nothing in the project could catch it.

4. **Every figure names its artefact**, rendered on the card, so a reader can go and check.

**Where they are ahead.** quantstats embeds fifteen-plus charts; the cockpit has none. Equity curves
would need a plotting dependency ARGUS does not have (it is pure Python, pydantic and dateutil only)
and there is no equity curve to draw until something settles. When there is, `_embed_figure()`'s
inline-SVG path is the pattern to copy, not base64 PNG: SVG stays readable in the page source, which
matters for an artefact whose whole claim is auditability.

**Licences.** quantstats Apache-2.0, agent-backtest-lab unlicensed (study only), freqtrade GPL-3.0,
qlib MIT, Nautilus LGPL-3.0. Nothing was copied; the palette in `cockpit.py` is carried over from
our own previous page.
