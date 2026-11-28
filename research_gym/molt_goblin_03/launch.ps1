# MOLT-GOBLIN-03 reproducible launcher (Windows PowerShell, CPU only).
# Stages: tests, audit, calibration, campaign (<= 6 h), freeze + hidden finals, analysis, export.
# The Tinker credential is read from $env:TINKER_API_KEY or from the file named by $env:TINKER_API_KEY_FILE.
param(
    [string]$Stage = "all",
    [double]$CampaignHours = 5.5,
    [double]$FinalsHours = 4.0,
    [int]$Workers = 3
)
$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
$R = "research_gym/molt_goblin_03"
$Rc = "$R/receipts"
Set-Location (Resolve-Path "$PSScriptRoot/../..")

function Stage($name) { return ($Stage -eq "all") -or ($Stage -eq $name) }

if (Stage "tests")     { python -m pytest -q research_gym/molt_goblin_03 }
if (Stage "audit")     { python -m research_gym.molt_goblin_03.audit --out "$Rc/audit" }
if (Stage "calibrate") { python -m research_gym.molt_goblin_03.calibrate3 --out "$Rc/calibration/calib_panel.jsonl" }
if (Stage "campaign")  { python -m research_gym.molt_goblin_03.campaign3 --out "$Rc/campaign" --hours $CampaignHours --workers $Workers }
if (Stage "finals")    { python -m research_gym.molt_goblin_03.finals3 --state "$Rc/campaign/state.json" --out "$Rc/finals" --hours $FinalsHours --workers $Workers }
if (Stage "analyze") {
    $runs = (Get-ChildItem "$Rc/finals/final_runs*.jsonl").FullName
    python -m research_gym.molt_goblin_03.analysis3 --runs $runs --multitask "$Rc/finals/multitask_routing.jsonl" --out "$Rc/finals"
    python -m research_gym.molt_goblin_03.export3 --state "$Rc/campaign/state.json" --out "$Rc/campaign/datagoblin" --analysis "$Rc/finals/analysis.json" --freeze "$Rc/finals/freeze.json"
}
