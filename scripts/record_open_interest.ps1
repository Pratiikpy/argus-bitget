# Every hour: record every liquid Bitget USDT perpetual's open interest (argus.market.open_interest)
# and copy the history into the console's deploy bundle. Bitget publishes only the current figure,
# so this record is the only way the console can say how open interest changed over a day. The
# bundle is redeployed by refresh_social.ps1; each attempt is logged to argus/data/open_interest.log.

param([int]$RecordEvery = 3600)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$argus = Join-Path $root "argus"
$deploy = Join-Path $root "deploy"
$log = Join-Path $argus "data\open_interest.log"
$python = Join-Path $argus ".venv\Scripts\python.exe"

while ($true) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    try {
        $env:PYTHONUTF8 = "1"
        Push-Location $argus
        $out = & $python -m argus.market.open_interest 2>&1 | Out-String
        $code = $LASTEXITCODE
        Pop-Location
        if ($code -eq 0) {
            Copy-Item (Join-Path $argus "data\open_interest_history.jsonl") `
                (Join-Path $deploy "data\open_interest_history.jsonl") -Force
            Add-Content -Path $log -Value "$stamp $($out.Trim())"
        } else {
            Add-Content -Path $log -Value "$stamp record failed (exit $code): $($out.Trim())"
        }
    } catch {
        Add-Content -Path $log -Value "$stamp record failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $RecordEvery
}
