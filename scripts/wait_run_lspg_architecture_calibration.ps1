param(
    [int]$MaximumWaitSeconds = 21600,
    [int]$PollSeconds = 30,
    [string]$Output = "experiments/loop_schedule_architecture_discovery_v1/calibration_runs"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReceiptPath = Join-Path $Repo "experiments/loop_schedule_architecture_discovery_v1/calibration/watcher_receipt.json"
$LogPath = Join-Path $Repo "experiments/loop_schedule_architecture_discovery_v1/calibration/watcher.log"
New-Item -ItemType Directory -Force -Path (Split-Path $ReceiptPath) | Out-Null
$Started = Get-Date
$Deadline = $Started.AddSeconds($MaximumWaitSeconds)
$Status = "waiting"
$Failure = $null

function Write-WatcherReceipt {
    $Finished = Get-Date
    $Value = [ordered]@{
        schema_version = 1
        status = $Status
        failure = $Failure
        started_utc = $Started.ToUniversalTime().ToString("o")
        finished_utc = $Finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($Finished - $Started).TotalSeconds, 3)
        maximum_wait_seconds = $MaximumWaitSeconds
        poll_seconds = $PollSeconds
        foreign_process_policy = "observe_only_never_terminate"
        calibration_output = $Output
    }
    [System.IO.File]::WriteAllText(
        $ReceiptPath,
        ($Value | ConvertTo-Json -Depth 5) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

try {
    while ((Get-Date) -lt $Deadline) {
        $Utilization = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        $Apps = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null)
        $ForeignApps = @($Apps | Where-Object { $_.Trim() -match "^[0-9]+$" })
        $Idle = $Utilization -match "^[0-9]+(?:\.[0-9]+)?$" -and [double]$Utilization -le 5.0 -and $ForeignApps.Count -eq 0
        Add-Content -LiteralPath $LogPath -Value "$(Get-Date -Format o) utilization=$Utilization foreign_compute=$($ForeignApps.Count) idle=$Idle"
        if ($Idle) {
            $Status = "launching"
            Write-WatcherReceipt
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_lspg_architecture_calibration.ps1") -Output $Output
            if ($LASTEXITCODE -ne 0) { throw "calibration launcher exited $LASTEXITCODE" }
            $Status = "completed"
            Write-WatcherReceipt
            exit 0
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $Status = "waiting_timeout"
    $Failure = "gpu_did_not_reach_idle_preflight_state"
    Write-WatcherReceipt
    exit 2
} catch {
    $Status = "failed"
    $Failure = $_.Exception.Message
    Write-WatcherReceipt
    exit 1
}
