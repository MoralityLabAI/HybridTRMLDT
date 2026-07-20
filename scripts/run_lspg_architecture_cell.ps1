param(
    [Parameter(Mandatory = $true)][string]$ProposalId,
    [Parameter(Mandatory = $true)][ValidateSet("calibration", "smoke", "A1", "A2", "B", "C", "D")][string]$Stage,
    [Parameter(Mandatory = $true)][ValidateSet("S0", "S1", "S2")][string]$Scale,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][long]$TokenVisitBudget,
    [string]$ProposalDir = "experiments/loop_schedule_architecture_discovery_v1/proposals",
    [string]$DatasetDir = "experiments/loop_schedule_architecture_discovery_v1/datasets",
    [string]$Output = "experiments/loop_schedule_architecture_discovery_v1/runs",
    [string]$ResourceConfig = "configs/lsa/architecture_resource_profile_v1.json",
    [switch]$AllowLockedEvaluation,
    [switch]$ResourceOnly,
    [int]$MeasurementWarmupSteps = 0,
    [ValidateRange(1, 5)][int]$Attempt = 1,
    [int]$TimeoutSecondsOverride = 0,
    [double]$GradientClipNorm = 100.0,
    [double]$LearningRateOverride = 0.0,
    [ValidateSet("amp_fp16", "fp32")][string]$Precision = "amp_fp16",
    [int]$EvaluationLimitPerFamily = 256
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProposalPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $ProposalDir))
$DatasetPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $DatasetDir))
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
$ResourcePath = [System.IO.Path]::GetFullPath((Join-Path $Repo $ResourceConfig))
$Profile = Get-Content -Raw -LiteralPath $ResourcePath | ConvertFrom-Json
$Caps = $Profile.caps
$Microbatch = [int]$Profile.microbatch_by_scale.$Scale
$LearningRate = [double]$Profile.learning_rate_by_scale.$Scale
if ($LearningRateOverride -gt 0.0) { $LearningRate = $LearningRateOverride }
$EffectiveBatch = [int]$Profile.effective_batch_size
$CellName = "$ProposalId-$Stage-$Scale-s$Seed"
$SafeCellName = $CellName -replace "[^A-Za-z0-9_.-]", "_"
$LogDir = Join-Path $OutputPath "logs"
$ReceiptDir = Join-Path $OutputPath "resource_receipts"
New-Item -ItemType Directory -Force -Path $OutputPath, $LogDir, $ReceiptDir | Out-Null
$Stdout = Join-Path $LogDir "$SafeCellName.attempt-$Attempt.stdout.log"
$Stderr = Join-Path $LogDir "$SafeCellName.attempt-$Attempt.stderr.log"
$Receipt = Join-Path $ReceiptDir "$SafeCellName.attempt-$Attempt.resource_receipt.json"
$MemoryLimitBytes = [UInt64]$Caps.ram_bytes
$CpuRate = [uint32]([int]$Caps.cpu_pct * 100)
$IoLimitBytesPerSecond = [double]$Caps.io_bytes_per_second
$VramLimitMb = [double]$Caps.vram_mb
$IoViolationLimit = [int]$Caps.io_sustained_samples
$TimeoutSeconds = [int]$Caps.child_timeout_seconds
if ($TimeoutSecondsOverride -gt 0) {
    if ($TimeoutSecondsOverride -gt $TimeoutSeconds) { throw "timeout override exceeds registered cap" }
    $TimeoutSeconds = $TimeoutSecondsOverride
}
$started = Get-Date

function Write-Receipt([hashtable]$Value) {
    $json = $Value | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($Receipt, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Write-ConstructionFailure([string]$Reason) {
    Write-Receipt ([ordered]@{
        training_task_id = $Profile.training_task_id
        cell_id = $CellName
        attempt = $Attempt
        status = "construction_failure"
        abort_reason = $Reason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        caps = $Caps
        owned_pid = $null
        cleanup_passed = $true
    })
}

trap {
    Write-ConstructionFailure $_.Exception.Message
    exit 1
}

if (-not (Test-Path -LiteralPath $ProposalPath)) { throw "proposal directory does not exist" }
if (-not (Test-Path -LiteralPath $DatasetPath)) { throw "dataset directory does not exist" }

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    for ($sample = 0; $sample -lt [int]$Profile.preflight.idle_samples; $sample++) {
        $utilization = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        if ($utilization -notmatch "^[0-9]+(?:\.[0-9]+)?$") { throw "GPU utilization preflight unavailable" }
        if ([double]$utilization -gt [double]$Profile.preflight.gpu_utilization_max_pct) {
            Write-ConstructionFailure "external_gpu_contention"
            exit 2
        }
        $apps = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null)
        if (@($apps | Where-Object { $_.Trim() -match "^[0-9]+$" }).Count -gt 0) {
            Write-ConstructionFailure "foreign_gpu_compute_process_present"
            exit 2
        }
        Start-Sleep -Seconds 1
    }
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LspgArchitectureJobObject {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int t, IntPtr p, uint n);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
  public const int Extended = 9;
  public const int Cpu = 15;
  public const uint ProcessMemory = 0x100;
  public const uint KillOnClose = 0x2000;
  public const uint CpuEnable = 0x1;
  public const uint CpuHardCap = 0x4;
  [StructLayout(LayoutKind.Sequential)] public struct IO { public ulong a,b,c,d,e,f; }
  [StructLayout(LayoutKind.Sequential)] public struct Basic { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public long f; public uint g,h; }
  [StructLayout(LayoutKind.Sequential)] public struct ExtendedInfo { public Basic basic; public IO io; public UIntPtr processMemory, jobMemory, peakProcess, peakJob; }
  [StructLayout(LayoutKind.Sequential)] public struct CpuInfo { public uint flags, rate; }
}
'@

$job = [LspgArchitectureJobObject]::CreateJobObject([IntPtr]::Zero, "lspg-architecture-$PID-$SafeCellName")
$limit = New-Object LspgArchitectureJobObject+ExtendedInfo
$limit.basic.flags = [LspgArchitectureJobObject]::ProcessMemory -bor [LspgArchitectureJobObject]::KillOnClose
$limit.processMemory = [UIntPtr]::new($MemoryLimitBytes)
$size = [Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
if (-not [LspgArchitectureJobObject]::SetInformationJobObject($job, [LspgArchitectureJobObject]::Extended, $ptr, $size)) { throw "memory cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
$cpu = New-Object LspgArchitectureJobObject+CpuInfo
$cpu.flags = [LspgArchitectureJobObject]::CpuEnable -bor [LspgArchitectureJobObject]::CpuHardCap
$cpu.rate = $CpuRate
$size = [Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr, $false)
if (-not [LspgArchitectureJobObject]::SetInformationJobObject($job, [LspgArchitectureJobObject]::Cpu, $ptr, $size)) { throw "CPU cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

$arguments = @(
    "-m", "research_gym.scripts.run_lspg_architecture_cell",
    "--proposal-dir", $ProposalPath,
    "--proposal-id", $ProposalId,
    "--dataset-dir", $DatasetPath,
    "--stage", $Stage,
    "--scale", $Scale,
    "--seed", [string]$Seed,
    "--token-visit-budget", [string]$TokenVisitBudget,
    "--effective-batch-size", [string]$EffectiveBatch,
    "--microbatch-size", [string]$Microbatch,
    "--learning-rate", [string]$LearningRate,
    "--maximum-gradient-norm", [string]$GradientClipNorm,
    "--precision", $Precision,
    "--evaluation-limit-per-family", [string]$EvaluationLimitPerFamily,
    "--vram-fraction", [string]$Caps.torch_vram_fraction,
    "--out", $OutputPath
)
if ($AllowLockedEvaluation) { $arguments += "--allow-locked-evaluation" }
if ($ResourceOnly) {
    $arguments += "--resource-only"
    $arguments += "--measurement-warmup-steps"
    $arguments += [string]$MeasurementWarmupSteps
}

$status = "running"
$abortReason = $null
$peakRamMb = 0.0
$sumRamMb = 0.0
$peakIoMbS = 0.0
$peakVramMb = 0.0
$sumCpuPct = 0.0
$ioViolations = 0
$samples = 0
$proc = $null
try {
    $proc = Start-Process -FilePath "python" -ArgumentList $arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -NoNewWindow
    if (-not [LspgArchitectureJobObject]::AssignProcessToJobObject($job, $proc.Handle)) { throw "process assignment to job failed" }
    $lastIo = 0.0
    $lastCpuSeconds = 0.0
    $lastSample = Get-Date
    $logicalProcessors = [Environment]::ProcessorCount
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 1
        $proc.Refresh()
        $samples++
        $ramMb = $proc.WorkingSet64 / 1MB
        $peakRamMb = [Math]::Max($peakRamMb, $ramMb)
        $sumRamMb += $ramMb
        $now = Get-Date
        $elapsed = [Math]::Max(0.001, ($now - $lastSample).TotalSeconds)
        $cpuSeconds = $proc.TotalProcessorTime.TotalSeconds
        $cpuPct = 100.0 * ($cpuSeconds - $lastCpuSeconds) / ($elapsed * $logicalProcessors)
        $sumCpuPct += [Math]::Max(0.0, $cpuPct)
        $lastCpuSeconds = $cpuSeconds
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -ErrorAction SilentlyContinue
        if ($null -ne $cim) {
            $currentIo = [double]$cim.ReadTransferCount + [double]$cim.WriteTransferCount
            if ($lastIo -gt 0) {
                $ioRate = ($currentIo - $lastIo) / $elapsed
                $peakIoMbS = [Math]::Max($peakIoMbS, $ioRate / 1MB)
                $ioViolations = if ($ioRate -gt $IoLimitBytesPerSecond) { $ioViolations + 1 } else { 0 }
            }
            $lastIo = $currentIo
        }
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $apps = @(& nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null)
            foreach ($line in $apps) {
                $parts = $line -split ","
                if ($parts.Count -ge 2 -and $parts[0].Trim() -eq [string]$proc.Id -and $parts[1].Trim() -match "^[0-9]+(?:\.[0-9]+)?$") {
                    $peakVramMb = [Math]::Max($peakVramMb, [double]$parts[1].Trim())
                }
            }
        }
        $lastSample = $now
        if ($ioViolations -ge $IoViolationLimit) { $abortReason = "sustained_io_cap_exceeded"; break }
        if ($peakVramMb -gt $VramLimitMb) { $abortReason = "vram_cap_exceeded"; break }
        if (($now - $started).TotalSeconds -gt $TimeoutSeconds) { $abortReason = "timeout"; break }
    }
    if ($null -ne $abortReason -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit()
        $status = "aborted"
    } else {
        $proc.WaitForExit()
        $status = if ($proc.ExitCode -eq 0) { "completed" } else { "failed" }
        if ($proc.ExitCode -ne 0) { $abortReason = "process_exit_$($proc.ExitCode)" }
    }
} catch {
    $status = "construction_failure"
    $abortReason = $_.Exception.Message
    if ($null -ne $proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
} finally {
    [LspgArchitectureJobObject]::CloseHandle($job) | Out-Null
    $finished = Get-Date
    Write-Receipt ([ordered]@{
        training_task_id = $Profile.training_task_id
        cell_id = $CellName
        attempt = $Attempt
        status = $status
        abort_reason = $abortReason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = $finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        caps = $Caps
        hard_caps = [ordered]@{ process_memory = $true; cpu_rate = $true; torch_allocator_vram = $true; io_abort = $true }
        peak_ram_mb = [Math]::Round($peakRamMb, 3)
        avg_ram_mb = if ($samples) { [Math]::Round($sumRamMb / $samples, 3) } else { 0 }
        peak_io_mb_s = [Math]::Round($peakIoMbS, 3)
        peak_vram_mb = [Math]::Round($peakVramMb, 3)
        avg_cpu_pct = if ($samples) { [Math]::Round($sumCpuPct / $samples, 3) } else { 0 }
        samples = $samples
        owned_pid = if ($null -eq $proc) { $null } else { $proc.Id }
        lingering_owned_process = if ($null -eq $proc) { $false } else { -not $proc.HasExited }
        cleanup_passed = if ($null -eq $proc) { $true } else { $proc.HasExited }
        foreign_process_policy = "observed but never terminated"
    })
}
if ($status -ne "completed") { exit 1 }
