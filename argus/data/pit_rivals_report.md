# Point-in-time correctness: the general-purpose specialists, run on the same 619 questions

Record for capability 31 (`src/argus/eval/capabilities/31-*.toml`), 2026-09-28. Every rival here
was run from its own cloned source, and every score comes from that code's own output.
`data/pit_rivals.json` holds the rows; this file explains how each arm was built and what its
numbers mean.

## 1. Two new rival arms: Qlib PIT and edgartools, both run from their real, cloned code

**Qlib** (microsoft/qlib, MIT, `research/repos/qlib-upstream`): `scripts/dump_pit.py::DumpPitData`
builds a real on-disk PIT store from a CSV of `date,period,value,field` rows (Qlib's own documented
format, `docs/advanced/PIT.rst`); `qlib/utils/__init__.py::read_period_data(index, data, period,
cur_date_int, quarterly)` is called directly, scanning periods newest-first for the first non-NaN
value — the plain way to use this low-level function for "the latest quarter as of X" (it answers
one named period, not "the latest one"). New venv `research/_venvs/qlib-pit` (plain venv, no Cython
build — `qlib.utils`/`qlib.config`/`scripts/dump_pit.py` are pure Python; only `numpy pandas pyyaml
redis requests packaging pydantic-settings ruamel.yaml setuptools-scm loguru fire tqdm` were
needed).

**edgartools** (dgunning/edgartools, MIT, `research/repos-themed/dgunning~edgartools`):
`edgar/entity/parser.py::EntityFactsParser.parse_company_facts` builds a real `EntityFacts` object
directly from the snapshot's companyfacts JSON (same shape SEC's own API returns — no monkeypatched
transport needed, unlike the other runners). `edgar/entity/query.py::FactQuery.as_of(date)` +
`by_period_type('quarterly')` + `latest_periods(n=1, annual=False)`, run once per tag in ARGUS's own
alias-tag priority order (`Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`,
`RevenueFromContractWithCustomerIncludingAssessedTax`, `SalesRevenueNet`) and merged. New venv
`research/_venvs/edgartools` (`pip install -e .` from the clone).

**A real defect found and deliberately excluded from scoring:** edgartools' free-text
`by_concept("revenue", exact=False)` — the call an agent unfamiliar with ARGUS's tag list would
naturally make — substring-matches `us-gaap:CostOfRevenue` too. The runner uses exact tag matching
so the measurement is of the as-of mechanism, not this separate search defect; recorded in
`edgartools.json`'s `fuzzy_concept_defect` for the record.

**Probes**: both rivals take a real as-of parameter (unlike Vibe-Trading/OpenBB/LangAlpha/
OpenAlice, none of which do), so each is called once per probe with its real as-of mechanism rather
than dumping a raw series for external gating. `scripts/pit_runners/dump_pit_probes.py` (the one
runner here that *does* import ARGUS, to guarantee the same 620 probes `eval/pit_rivals.py`
generates) writes them once; both rivals answer all 620 (619 scored, matching ARGUS's own count).

### Scores (both sides pooled, then broken out — the fairness caveat)

| arm | right/619 | leak | late | stale | McNemar vs ARGUS |
|---|---|---|---|---|---|
| Qlib PIT | 51 | 179 | 106 | 283 | p = 2.1e-171, rival_only_right = 0 |
| edgartools as_of | 369 | 198 | 52 | 0 | p = 1.1e-75, rival_only_right = 0 |

Held out (nine tickers, 553 questions): Qlib 44/553, edgartools 336/553; both still lose paired and
significantly (p = 1.2e-153, p = 9.5e-66; `rival_only_right = 0` for both).

**Before/after, reported separately per `day_granular_fairness` in `data/pit_rivals.json`:**

| arm | before right-rate | after right-rate |
|---|---|---|
| Qlib PIT | 0.0065 (2/309) | 0.1581 (49/310) |
| edgartools | 0.3592 (111/309) | 0.8323 (258/310) |

Both are day-granular (Qlib's `date` column, edgartools' `filing_date <= as_of_date`), so a probe an
hour before a same-day acceptance and one an hour after are the same question to them — confirmed:
leak is concentrated almost entirely on the "before" side for both (Qlib 179 of 179 leaks are
"before"; edgartools 198 of 198).

**Qlib's raw 51/619 understates its period-selection accuracy — found by running it, sourced.**
`scripts/dump_pit.py::DumpPitData.get_source_data` (line 123) does
`df[value_col] = df[value_col].astype("float32")` before every value is packed into the store,
unconditionally — regardless of the on-disk `value` field being float64
(`qlib/config.py`'s `pit_record_type`). Revenue in the billions loses precision under float32
(NVDA's 2,173,000,000 comes back as 2,172,999,936.0). Checked directly: **all 283**
`stale_after_restatement` outcomes are within 1e-5 relative of the true value — right quarter,
rounded value, not a period error. Ignoring that rounding, Qlib's period-selection accuracy is
334/619 (54%); it is still 285/619 outright wrong-period (179 leak + 106 late), and still loses to
ARGUS on every framing. The scoring keeps the same exact-match "right" definition every other rival
in this file is held to — no per-rival tolerance was added, to keep the numbers comparable; this
paragraph is how the raw number is explained, not adjusted.

**Verdict I would draw from these numbers**: neither Qlib PIT nor edgartools beats or ties ARGUS on
either side. ARGUS wins decisively, paired and out of sample, against all eight rival arms now on
record, exactly as it did against the original six. Nothing here changes the capability's standing
toward OWNED except by removing one more named blocker (both specialists the plan called out as
missing are now run).

## 2. openbb-mcp-server — run, not just read

**Blocked at first by a real dependency wall, opened rather than reported.**
`from openbb_core.api.rest_api import app` (the literal import
`openbb_mcp_server/app/app.py:348` performs, and what the MCP server wraps as tools) raised
`AttributeError: '_IncludedRouter' object has no attribute 'path'` inside
`openbb_core/app/router.py:429`. Root cause: `research/_venvs/pit-rivals` is
`--system-site-packages` and inherited `fastapi==0.139.0`/`starlette==1.3.1` from the global
site-packages (installed there for other, unrelated tools), while `openbb_platform/core/
pyproject.toml` pins `fastapi = "0.136.3"` (Poetry caret, effectively `<0.137`). A local
`pip install "fastapi==0.136.3"` inside that one venv — shadowing the global copy only; pip itself
refused to touch the outside-venv install — fixed it. `starlette>=0.46.0` was already satisfied.

**What actually ran** (`scripts/pit_runners/openbb_mcp_verify.py`): `GET
/api/v1/equity/fundamental/income` on `openbb_core.api.rest_api.app` — the same FastAPI app object
the MCP server imports and wraps, dispatching through the identical
`openbb_equity/fundamental/fundamental_router.py::income` → `OBBject.from_query` →
`SecIncomeStatementFetcher.fetch_data` path `openbb_sec_runner.py` already calls directly — for all
ten tickers, both `pit_mode` values, against the same monkeypatched snapshot transport.

**Result**: identical to the already-recorded ODP output for 16 of 20 ticker/mode pairs (checked
field-for-field; every non-float field byte-identical, `diluted_eps` within 1e-9 relative — float64
JSON round-trip noise, ~1e-15, not a data difference). The other 4 (GOOGL, MSTR × two modes) fail
**identically on both paths**, same exception, same cause: GOOGL/GOOG/GOOGM/GOOGN and MSTR/STRD
each share one SEC CIK, `company_tickers.json`'s last entry for that CIK wins the reverse ticker
lookup in both the REST route and the direct fetcher, and the snapshot has no file for that losing
ticker (a pre-existing snapshot-building gap, not new — GOOGL already read as an error in the
committed `openbb.json`). The identical failure mode is itself further confirmation the two paths
are the same code, not weaker evidence. Full row-for-row record:
`research/pit_rivals/outputs/openbb_mcp_verify.json`.

**Agent Rita**: still not run. It has no fetcher of its own — it reads OpenBB Workspace widgets and
needs a Workspace account. That is a scope decision still open (supply an account, or
drop it from the OWNED condition), not something to guess at.

## 4. The 22:00 ET bound — resolved, not re-derived from NVDA alone

Traced in `Activity/PROGRESS.md`: the acceptance-second gate and the 619-question held-out run both
landed in the same session, "2026-09-26 (UTC, afternoon)". The docstring's own justification for
22:00 ET cites MSFT's 20:44 ET same-day acceptance — MSFT is one of the nine held-out tickers — so
the claim did not provably predate the holdout set if read as a value empirically fit to these ten
tickers.

**Re-verified against SEC.gov directly** (`sec.gov/submit-filings/filer-support-resources/how-do-i-
guides/determine-status-my-filing`, fetched 2026-09-28): "EDGAR's hours of operation are 6:00 a.m.
to 10:00 p.m. ET, Monday through Friday" — a fixed, universal daily window, true for every filer and
form, independent of which tickers this benchmark happens to cover. MSFT/NVDA are illustrations of
it in the docstring, not its derivation. (The same page gives the *nominal* same-day cutoff as
5:30 p.m. ET for most forms, with the 10:00 p.m. exception documented only for Forms 3/4/144 and a
few others — 10-Q/10-K are not on that list; MSFT's real 20:44 ET same-day 10-K shows EDGAR's actual
behaviour for these forms does not strictly follow that nominal rule either, which is exactly why
the conservative institutional-close bound is the right one to keep.)

Checked directly and separately: in the 10-ticker, 2,647-fact revenue snapshot, **zero** facts ever
resolve through the `edgar-close` fallback — every accession's exact `acceptanceDateTime` is present
in the filer's submissions index (`basis == "accepted"` for all 2,647). The bound's value cannot
change this benchmark's score either way.

Re-deriving it from NVDA alone (as the plan proposed as the fallback) was tried and rejected, with
the reason now recorded in `market/pit.py`: NVDA's own latest same-day acceptance is 17:30 ET — the
weakest data point of the ten tickers — and narrowing the bound to it would create real leak risk
for filers like MSFT (20:44 ET) in live production without changing anything this benchmark
measures. `market/pit.py`'s docstring carries the full citation and reasoning.

## 5. `same_input_comparison` now cites `data/pit_rivals.json`

The proof previously cited only `test_workbench_comparison.py::TestPointInTimeComparison` (source
grep of the archived openbb-agents repo). It now also carries `artefact = "data/pit_rivals.json"` —
the stronger, primary evidence: eight real rivals asked the identical 619 questions on the identical
frozen snapshot, scored from their own real output. The archived-repo grep stays as the original,
narrower claim (openbb-agents literally has no code path that could accept an as-of query at all).

## How to reproduce

- `python -m argus.eval.pit_rivals --snapshot research/pit_rivals/snapshot --outputs
  research/pit_rivals/outputs` regenerates `data/pit_rivals.json` from the rivals' recorded
  outputs: ARGUS 619/619 with 0 leaks and 29/29 restated quarters; all eight rivals scored.
- Each rival's own run: `scripts/pit_runners/qlib_pit_runner.py`, `edgartools_asof_runner.py` and
  `openbb_mcp_verify.py`, each in its own virtual environment, reading the same frozen snapshot
  (`dump_pit_probes.py` writes the 619 probes they are asked).
- `pytest tests/test_pit.py` checks the scoring of both new arms offline, from their recorded
  outputs, including the before/after split.
