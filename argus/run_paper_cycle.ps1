# ARGUS paper-trading cycle — scheduled entry point.
# Loads credentials from .secrets/*.env (gitignored), runs one cycle on the full rToken universe,
# appends a dated log. Created 2026-09-12.
$ErrorActionPreference = "Continue"
# Derived from this script's own location rather than hardcoded: the scheduled task, a clone
# on another machine and a developer running it by hand must all resolve the same root.
$root = Split-Path -Parent $PSScriptRoot
Set-Location "$root\argus"
foreach ($f in @("$root\.secrets\qwen.env", "$root\.secrets\bitget.env")) {
  if (Test-Path $f) {
    Get-Content $f | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=' -and $_ -notmatch '^\s*#' } | ForEach-Object {
      $k, $v = $_ -split '=', 2
      [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
    }
  }
}
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = "$root\argus\data\paper_runs\cycle_$stamp.log"
# Windows PowerShell writes UTF-16LE by default, which makes the log unreadable with grep, tail and
# every other ordinary tool. -Encoding utf8 on Out-File was not enough: Tee-Object has no -Encoding
# parameter in Windows PowerShell at all and always writes Unicode, and Tee-Object is the line that
# carries the entire cycle output. Every run after that "fix" was still UTF-16. Add-Content with an
# explicit encoding is used throughout instead, and there is no Tee.
"=== ARGUS paper cycle $stamp ===" | Out-File $log -Encoding utf8

# Refresh the hedge-effectiveness measurement BEFORE the cycle reads it. `risk/effectiveness.py`
# treats a measurement older than 36 hours as absent, and the runner then prices the hedge with the
# old constants and says so. Running it here means the live path is measured on every cycle rather
# than whenever somebody remembers; if this step fails the cycle still runs, with ASSUMED factors in
# the record. That degradation is deliberate and visible, never silent.
python -m argus.market.markout --symbol MSTRUSDT --seconds 90 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"markout_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.market.depth 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"depth_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# Append one real order-book snapshot per instrument to the tape. This is the only input to
# `eval/queueproof.py` that cannot be produced by writing more code: a distribution of how a level
# *changes* needs snapshots separated in time, so the measurement improves by running and by
# nothing else. The first tape put a near-touch level at a median 1.95 contracts turning over ~70%
# of itself a minute — nothing like the large, slow levels the queue experiment had assumed.
python -m argus.eval.bookcalib --record 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"bookcalib_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.risk.session_risk --days 90 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"session_risk_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# Re-verify that every symbol we trade still behaves like a tokenized equity. `market/bitget.py:41`
# claimed membership was "verified behaviourally by session attenuation" while nothing called the
# validator, so the sentence described an intention. A hardcoded list is still right — a regex over
# ticker names swept in FARTCOINUSDT — but an unchecked one is how SPXUSDT got in and produced a
# 179% backtest on memecoin drift. Non-zero exit means a symbol lost its anchor; the cycle
# continues and the log carries it, because this is a warning about the universe, not a reason to
# stop measuring it.
python -m argus.market.validation --days 90 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"universe_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# The scheduled CPI and FOMC dates the console cites, from the BLS and Federal Reserve schedules.
python -m argus.market.calendar 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"calendar_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# How each traded name reacts to CPI, Fed decisions and its own earnings (research/eventstudy.py).
# A year of hourly bars for twelve names takes minutes, and a new event arrives at most daily, so
# it refreshes only when the artefact is over 20 hours old.
python -m argus.research.event_reactions --if-stale 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"event_reactions_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# The macro series the hosted console falls back to when FRED does not answer from its network.
python -c "from argus.lui.research import write_macro_snapshot; print(write_macro_snapshot(), 'macro readings written')" 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"macro_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.risk.effectiveness --days 60 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"effectiveness_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.paper.runner --once --symbols NVDAUSDT,TSLAUSDT,AAPLUSDT,MSFTUSDT,METAUSDT,GOOGLUSDT,AMZNUSDT,COINUSDT,MSTRUSDT,QQQUSDT,TQQQUSDT,SQQQUSDT 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

# Verify the cycle did what it was supposed to, while the evidence is fresh. The exit code above is
# 0 whether twelve symbols were decided or two, so it is not a quality signal on its own.
# `cyclecheck` reads the log and the ledger and writes data/cycle_check.json; on the first cycle
# with price discovery it also reports whether the autopsy's falsifier fired.
"" | Add-Content -Path $log -Encoding utf8
python -m argus.register.cadence 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"cadence_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.register.resolve 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"register_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

python -m argus.eval.cyclecheck 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"check_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8
$checkExit = $LASTEXITCODE

# Publish the new decisions to the hosted console. The hosted console reads a bundled, read-only copy
# of the ledger, so without this step it froze at whatever the last manual deploy carried — it was 40
# hours and 42 decisions behind on 2026-09-23 — and a judge compares the page against the record.
# Added 2026-09-23 with the project owner's approval. Gated on the cycle check passing, so a broken
# cycle or a broken chain is never shipped: the page keeps the last good record and the log says why.
# `deploysync` copies only its two manifests and never `.secrets/`; the model key lives in Vercel's
# server environment, not in the bundle.
# The cockpit page is generated from the artefacts, and `test_cockpit` fails the moment the ledger
# grows past the figures it prints -- which every cycle does, whether or not its check passes.
# Regenerated on every cycle, before the publish gate, so the committed page never lags the record.
# The refusal grading goes first: the cockpit prints its figures, and a grading run by hand after
# the page was built left the two disagreeing.
python -m argus.eval.refusal 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"refusal_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8
python -m argus.demo.cockpit 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"cockpit_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8

if ($checkExit -eq 0) {
  python -m argus.demo.deploysync 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
  $syncExit = $LASTEXITCODE
  "deploysync_exit=$syncExit" | Add-Content -Path $log -Encoding utf8
  if ($syncExit -eq 0) {
    Push-Location "$root\deploy"
    & "$env:APPDATA\npm\vercel.cmd" deploy --prod --yes 2>&1 | ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
    "publish_exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8
    Pop-Location
  }
} else {
  "publish_skipped: cycle check exit $checkExit" | Add-Content -Path $log -Encoding utf8
}
