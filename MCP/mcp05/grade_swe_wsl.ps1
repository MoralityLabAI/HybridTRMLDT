param(
    [Parameter(Mandatory)] [string]$Campaign,
    [Parameter(Mandatory)] [string]$DatasetDirectory,
    [string]$Distro = 'Ubuntu-24.04',
    [string]$HarnessPython = '/root/swebench-venv/bin/python',
    [switch]$ProbeOnly
)

$ErrorActionPreference = 'Stop'
$DatasetDirectory = (Resolve-Path -LiteralPath $DatasetDirectory).Path
$Campaign = (Resolve-Path -LiteralPath $Campaign).Path
$grade = Join-Path $DatasetDirectory 'cuda'
$ids = @('pallets__flask-4045', 'pallets__flask-4992')
New-Item -ItemType Directory -Path $grade -Force | Out-Null

function Set-GradeStatus([string]$stage, [string]$detail = '') {
    [ordered]@{stage=$stage; detail=$detail; utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $grade 'status.json')
}

if (-not $ProbeOnly) {
    foreach ($arm in @('A', 'D')) {
        $predictions = Join-Path $campaign "predictions-$arm.jsonl"
        if (-not (Test-Path -LiteralPath $predictions)) {
            Set-GradeStatus 'waiting-for-predictions' "Missing predictions for $arm"
            exit 0
        }
        $registered = @(Get-Content -LiteralPath $predictions | ForEach-Object { (ConvertFrom-Json $_).instance_id })
        if ($registered.Count -ne $ids.Count -or (Compare-Object ($ids | Sort-Object) ($registered | Sort-Object))) {
            Set-GradeStatus 'waiting-for-predictions' "The $arm predictions do not yet cover both Flask assignments"
            exit 0
        }
        Copy-Item -LiteralPath $predictions -Destination (Join-Path $grade "predictions-$arm.jsonl")
    }
}

$pinned = Join-Path $datasetDirectory 'swe-bench-lite-test.parquet'
$modelTestSet = Join-Path $grade 'pinned-flask.json'
$env:MCP05_GRADE_PARQUET = $pinned
$env:MCP05_GRADE_JSON = $modelTestSet
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $datasetDirectory 'reader'
try {
    @'
import hashlib
import json
import os
from pathlib import Path
import duckdb

source = Path(os.environ["MCP05_GRADE_PARQUET"])
target = Path(os.environ["MCP05_GRADE_JSON"])
if hashlib.sha256(source.read_bytes()).hexdigest() != "438e281d80587aa7be470896ce410557002fde02d2ceee3e099331d308f62dd3":
    raise ValueError("SWE-bench Lite parquet checksum mismatch")
with duckdb.connect(":memory:") as db:
    rows = db.execute(
        """SELECT to_json(row) FROM (
               SELECT * FROM read_parquet(?)
               WHERE instance_id IN ('pallets__flask-4045', 'pallets__flask-4992')
               ORDER BY instance_id
           ) row""",
        [str(source)],
    ).fetchall()
if len(rows) != 2:
    raise ValueError("Unexpected number of pinned Flask rows")
instances = [json.loads(raw) for raw, in rows]
for instance in instances:
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(instance[key], list):
            instance[key] = json.dumps(instance[key])
target.write_text(json.dumps(instances, indent=2) + "\n", encoding="utf-8")
'@ | python -B -
    if ($LASTEXITCODE -ne 0) { throw 'Pinned SWE-bench test projection failed' }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}

$wsl = @('-d', $Distro, '--cd', $grade, '--exec')
$images = @(Get-Content -Raw -LiteralPath $modelTestSet | ConvertFrom-Json | Select-Object -ExpandProperty image)
$diagnostic = & wsl.exe @wsl docker image inspect @images --format '{{.Id}}' 2>&1
$probeExit = $LASTEXITCODE
$message = ($diagnostic | Out-String).Replace([string][char]0, '').Trim()
$message | Set-Content -LiteralPath (Join-Path $grade 'container-probe.log')
if ($probeExit -ne 0) {
    Set-GradeStatus 'waiting-for-container-environment' $message
    exit 0
}
if ($ProbeOnly) {
    Set-GradeStatus 'container-images-available'
    exit 0
}

$armStatus = [ordered]@{}
foreach ($arm in @('A', 'D')) {
    Set-GradeStatus 'grading' "Arm $arm"
    $argsForHarness = @($HarnessPython, '-m', 'swebench.harness.run_evaluation',
                       '--dataset_name', 'pinned-flask.json', '--split', 'test',
                       '--predictions_path', "predictions-$arm.jsonl", '--instance_ids') +
                      $ids + @('--max_workers', '1', '--run_id', "mcp05-inky-cuda-$arm", '--timeout', '900')
    & wsl.exe @wsl @argsForHarness *> (Join-Path $grade "harness-$arm.log")
    $armStatus[$arm] = $LASTEXITCODE
}
$armStatus | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $grade 'arm-exit-codes.json')
if ($armStatus.A -eq 0 -and $armStatus.D -eq 0) {
    Set-GradeStatus 'harness-exited-successfully'
} else {
    Set-GradeStatus 'harness-error' "A exited $($armStatus.A) and D exited $($armStatus.D)"
}
