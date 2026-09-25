# ARGUS Skill-effectiveness sweep — scheduled every three hours.
# Calls every bitget-signal tool and every bitget-mcp-server catalog entry once, keyless, and
# appends the sweep to data/skill_matrix_history.jsonl (`eval/skill_matrix.py`). One sweep is a
# snapshot of one moment — the first two, on 2026-09-25, landed in an outage — so the answer rate
# worth quoting is the one across many sweeps. Created 2026-09-25. Reads no credentials.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location "$root\argus"
$env:PYTHONUTF8 = "1"
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
New-Item -ItemType Directory -Force "$root\argus\data\skill_runs" | Out-Null
$log = "$root\argus\data\skill_runs\sweep_$stamp.log"
"=== ARGUS skill sweep $stamp ===" | Out-File $log -Encoding utf8
& "$root\argus\.venv\Scripts\python.exe" -m argus.eval.skill_matrix --rounds 1 2>&1 |
  ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
"exit=$LASTEXITCODE" | Add-Content -Path $log -Encoding utf8
