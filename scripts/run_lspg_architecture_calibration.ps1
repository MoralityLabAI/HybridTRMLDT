param(
    [string]$Output = "experiments/loop_schedule_architecture_discovery_v1/calibration_runs"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Profile = Get-Content -Raw -LiteralPath (Join-Path $Repo "configs/lsa/architecture_resource_profile_v1.json") | ConvertFrom-Json
$Budget = 50L * [long]$Profile.effective_batch_size * [long]$Profile.sequence_length * 8L
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
$Scales = @("S0", "S1", "S2")
foreach ($Scale in $Scales) {
    $CellId = "LSAD-C-K2L8-calibration-$Scale-s397"
    $Existing = @(Get-ChildItem -LiteralPath (Join-Path $OutputPath "resource_receipts") -Filter "$CellId.attempt-*.resource_receipt.json" -ErrorAction SilentlyContinue)
    $Completed = @($Existing | Where-Object { (Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json).status -eq "completed" })
    if ($Completed.Count -gt 0 -and (Test-Path -LiteralPath (Join-Path $OutputPath "$CellId\result.json"))) { continue }
    $Attempts = @($Existing | ForEach-Object { if ($_.Name -match "\.attempt-([0-9]+)\.") { [int]$Matches[1] } })
    $Attempt = if ($Attempts.Count) { ($Attempts | Measure-Object -Maximum).Maximum + 1 } else { 1 }
    if ($Attempt -gt 5) { throw "calibration exhausted registered attempts at $Scale" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_lspg_architecture_cell.ps1") `
        -ProposalId "LSAD-C-K2L8" `
        -Stage "calibration" `
        -Scale $Scale `
        -Seed 397 `
        -TokenVisitBudget $Budget `
        -Output $Output `
        -Attempt $Attempt `
        -ResourceOnly `
        -MeasurementWarmupSteps 20 `
        -EvaluationLimitPerFamily 0
    if ($LASTEXITCODE -ne 0) { throw "resource calibration failed at $Scale" }
}
python -m research_gym.scripts.select_lspg_architecture_budget --runs (Join-Path $Repo $Output)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
