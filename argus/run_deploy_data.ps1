# Refresh the hosted console's record after each desk cycle — data only, never code.
# Copies the artefacts the deployed console already serves (the ledger, desk notes, studies) into
# deploy/data and redeploys that folder, so /status and every ledger answer show the desk's latest
# decisions rather than the ones from the last code deploy. The deployed code is left exactly as
# it was: code ships when it is ready, the record should not wait for it. Created 2026-09-25.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location "$root\argus"
$env:PYTHONUTF8 = "1"
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
New-Item -ItemType Directory -Force "$root\argus\data\deploy_runs" | Out-Null
$log = "$root\argus\data\deploy_runs\data_$stamp.log"
"=== ARGUS data-only deploy $stamp ===" | Out-File $log -Encoding utf8
& "$root\argus\.venv\Scripts\python.exe" -m argus.demo.deploysync --data-only 2>&1 |
  ForEach-Object { $_ | Out-String -Stream } | Add-Content -Path $log -Encoding utf8
Set-Location "$root\deploy"
$out = & vercel deploy --prod --yes --archive=tgz 2>&1 | Out-String
$out | Add-Content -Path $log -Encoding utf8
# A deploy with no "Aliased" line did not reach the public URL; say so in the log's last line
# rather than let a failed refresh look like a finished one.
$result = if ($out -match "Aliased") { "result=published" } else { "result=FAILED" }
Add-Content -Path $log -Value $result -Encoding utf8
