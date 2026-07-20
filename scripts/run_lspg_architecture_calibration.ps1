param(
    [string]$Output = "experiments/loop_schedule_architecture_discovery_v1/calibration_runs"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Profile = Get-Content -Raw -LiteralPath (Join-Path $Repo "configs/lsa/architecture_resource_profile_v1.json") | ConvertFrom-Json
$Budget = 50L * [long]$Profile.effective_batch_size * [long]$Profile.sequence_length * 8L
$Scales = @("S0", "S1", "S2")
foreach ($Scale in $Scales) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_lspg_architecture_cell.ps1") `
        -ProposalId "LSAD-C-K2L8" `
        -Stage "calibration" `
        -Scale $Scale `
        -Seed 397 `
        -TokenVisitBudget $Budget `
        -Output $Output `
        -ResourceOnly `
        -MeasurementWarmupSteps 20 `
        -EvaluationLimitPerFamily 0
    if ($LASTEXITCODE -ne 0) { throw "resource calibration failed at $Scale" }
}
python -m research_gym.scripts.select_lspg_architecture_budget --runs (Join-Path $Repo $Output)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
