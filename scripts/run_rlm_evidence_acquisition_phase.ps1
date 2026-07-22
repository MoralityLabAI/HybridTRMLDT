param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("calibrate", "validate", "run", "finalize")]
    [string]$Phase,
    [string]$Config = "configs/rlm_evidence_acquisition_mesh_v0.json",
    [string]$Output = "experiments/rlm_evidence_acquisition_mesh_v0",
    [string]$Module = "research_gym.scripts.bench_rlm_evidence_acquisition_v0",
    [string]$PythonExe = "python",
    [ValidateRange(0, 2147483647)]
    [double]$ExternalPriorElapsedSeconds = 0,
    [ValidateRange(1, 2147483647)]
    [int]$Attempt = 1
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "lspg_gpu_process_policy.ps1")

function Resolve-RepoPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

function Get-MemoryAudit {
    $os = Get-CimInstance Win32_OperatingSystem
    $top = @(
        Get-Process -ErrorAction SilentlyContinue |
        Sort-Object -Property WorkingSet64 -Descending |
        Select-Object -First 5 |
        ForEach-Object {
            [ordered]@{
                pid = $_.Id
                name = $_.ProcessName
                working_set_mb = [Math]::Round($_.WorkingSet64 / 1MB, 3)
                private_memory_mb = [Math]::Round($_.PrivateMemorySize64 / 1MB, 3)
            }
        }
    )
    $gpu = @()
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = @(& nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null)
    }
    return [ordered]@{
        available_ram_mb = [Math]::Round([double]$os.FreePhysicalMemory / 1024.0, 3)
        total_visible_ram_mb = [Math]::Round([double]$os.TotalVisibleMemorySize / 1024.0, 3)
        git_commit = (& git -C $Repo rev-parse HEAD).Trim()
        top_memory_processes = $top
        gpu_compute_apps = $gpu
    }
}

$ConfigPath = Resolve-RepoPath $Config
$OutputPath = Resolve-RepoPath $Output
$PythonPath = if ([System.IO.Path]::IsPathRooted($PythonExe)) {
    [System.IO.Path]::GetFullPath($PythonExe)
} else {
    (Get-Command $PythonExe -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Python executable does not exist" }
if (-not (Test-Path -LiteralPath $ConfigPath)) { throw "registered config does not exist" }
$Protocol = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$Caps = $Protocol.resources
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$Stdout = Join-Path $OutputPath "$Phase.attempt-$Attempt.stdout.log"
$Stderr = Join-Path $OutputPath "$Phase.attempt-$Attempt.stderr.log"
$Receipt = Join-Path $OutputPath "$Phase.resource_receipt.json"
$AttemptReceipt = Join-Path $OutputPath "$Phase.attempt-$Attempt.resource_receipt.json"
$MemoryLimitBytes = [UInt64]([double]$Caps.ram_mb * 1MB)
$CpuRate = [uint32]([int]$Caps.cpu_pct * 100)
$IoLimitBytesPerSecond = [double]$Caps.io_abort_mb_s * 1MB
$VramLimitMb = [double]$Caps.vram_mb
$IoViolationLimit = [int]$Caps.io_sustained_samples
$TimeoutSeconds = [int]$Caps.phase_timeout_seconds
$AggregateLimitSeconds = [double]$Caps.aggregate_gpu_hours * 3600.0
$started = Get-Date
$memoryBefore = Get-MemoryAudit
$IgnoredCpuOnlyRegistryPids = @()
$PriorElapsedSeconds = [double]$ExternalPriorElapsedSeconds
$PriorReceipts = @(
    Get-ChildItem -LiteralPath $OutputPath -Filter "*.attempt-*.resource_receipt.json" -File -ErrorAction SilentlyContinue
)
foreach ($path in $PriorReceipts) {
    if ($path.FullName -eq $AttemptReceipt) { continue }
    $prior = Get-Content -Raw -LiteralPath $path.FullName | ConvertFrom-Json
    if (
        $prior.phase -notin @("validate", "finalize") -and
        $null -ne $prior.elapsed_seconds
    ) {
        $PriorElapsedSeconds += [double]$prior.elapsed_seconds
    }
}

function Write-Receipt([hashtable]$Value) {
    $json = ($Value | ConvertTo-Json -Depth 10).Replace("`r`n", "`n")
    foreach ($path in ($Receipt, $AttemptReceipt)) {
        [System.IO.File]::WriteAllText($path, $json + "`n", [System.Text.UTF8Encoding]::new($false))
    }
}

function Write-ConstructionFailure([string]$Reason) {
    Write-Receipt ([ordered]@{
        protocol_id = $Protocol.protocol_id
        module = $Module
        phase = $Phase
        attempt = $Attempt
        status = "construction_failure"
        abort_reason = $Reason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        caps = $Caps
        prior_elapsed_seconds = [Math]::Round($PriorElapsedSeconds, 3)
        memory_before = $memoryBefore
        memory_after = Get-MemoryAudit
        ignored_cpu_only_gpu_registry_pids = @($IgnoredCpuOnlyRegistryPids | Sort-Object -Unique)
        owned_pid = $null
        cleanup_passed = $true
    })
}

trap {
    Write-ConstructionFailure $_.Exception.Message
    exit 1
}

if ($PriorElapsedSeconds -ge $AggregateLimitSeconds) {
    Write-ConstructionFailure "aggregate_gpu_hour_cap_already_exhausted"
    exit 2
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    for ($sample = 0; $sample -lt 3; $sample++) {
        $utilization = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()
        if ($utilization -notmatch "^[0-9]+(?:\.[0-9]+)?$") { throw "GPU utilization preflight unavailable" }
        if ([double]$utilization -gt 20.0) {
            Write-ConstructionFailure "external_gpu_contention"
            exit 2
        }
        $apps = @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null)
        $foreignApps = @()
        foreach ($line in $apps) {
            $candidatePid = $line.Trim()
            if ($candidatePid -notmatch "^[0-9]+$") { continue }
            $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$candidatePid" -ErrorAction SilentlyContinue
            if (
                $null -ne $candidate -and
                -not [string]::IsNullOrWhiteSpace($candidate.CommandLine) -and
                (Test-LspgCpuOnlyNvidiaRegistration -ProcessName $candidate.Name -CommandLine $candidate.CommandLine)
            ) {
                $IgnoredCpuOnlyRegistryPids += [int]$candidatePid
            } else {
                $foreignApps += [int]$candidatePid
            }
        }
        if ($foreignApps.Count -gt 0) {
            Write-ConstructionFailure "foreign_gpu_compute_process_present"
            exit 2
        }
        Start-Sleep -Seconds 1
    }
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RlmEvidenceAcquisitionJobObject {
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

$job = [RlmEvidenceAcquisitionJobObject]::CreateJobObject([IntPtr]::Zero, "rlm-evidence-acquisition-$PID-$Phase")
$limit = New-Object RlmEvidenceAcquisitionJobObject+ExtendedInfo
$limit.basic.flags = [RlmEvidenceAcquisitionJobObject]::ProcessMemory -bor [RlmEvidenceAcquisitionJobObject]::KillOnClose
$limit.processMemory = [UIntPtr]::new($MemoryLimitBytes)
$size = [Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
if (-not [RlmEvidenceAcquisitionJobObject]::SetInformationJobObject($job, [RlmEvidenceAcquisitionJobObject]::Extended, $ptr, $size)) {
    throw "memory cap setup failed"
}
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
$cpu = New-Object RlmEvidenceAcquisitionJobObject+CpuInfo
$cpu.flags = [RlmEvidenceAcquisitionJobObject]::CpuEnable -bor [RlmEvidenceAcquisitionJobObject]::CpuHardCap
$cpu.rate = $CpuRate
$size = [Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr, $false)
if (-not [RlmEvidenceAcquisitionJobObject]::SetInformationJobObject($job, [RlmEvidenceAcquisitionJobObject]::Cpu, $ptr, $size)) {
    throw "CPU cap setup failed"
}
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

$arguments = @(
    "-m", $Module,
    "--phase", $Phase,
    "--config", $ConfigPath,
    "--output", $OutputPath,
    "--vram-fraction", [string]$Caps.torch_vram_fraction
)

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
    $proc = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -NoNewWindow
    if (-not [RlmEvidenceAcquisitionJobObject]::AssignProcessToJobObject($job, $proc.Handle)) {
        throw "process assignment to job failed"
    }
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
        if (($now - $started).TotalSeconds -gt $TimeoutSeconds) { $abortReason = "phase_timeout"; break }
        if ($PriorElapsedSeconds + ($now - $started).TotalSeconds -gt $AggregateLimitSeconds) {
            $abortReason = "aggregate_gpu_hour_cap_exceeded"
            break
        }
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
    if ($null -ne $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
} finally {
    [RlmEvidenceAcquisitionJobObject]::CloseHandle($job) | Out-Null
    $finished = Get-Date
    $memoryAfter = Get-MemoryAudit
    $ownedGpuAfter = if ($null -eq $proc) { @() } else { @($memoryAfter.gpu_compute_apps | Where-Object { $_ -match "^\s*$($proc.Id)\s*," }) }
    $cleanupPassed = if ($null -eq $proc) { $true } else { $proc.HasExited -and $ownedGpuAfter.Count -eq 0 }
    Write-Receipt ([ordered]@{
        protocol_id = $Protocol.protocol_id
        module = $Module
        python_executable = $PythonPath
        phase = $Phase
        attempt = $Attempt
        status = $status
        abort_reason = $abortReason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = $finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        prior_elapsed_seconds = [Math]::Round($PriorElapsedSeconds, 3)
        aggregate_elapsed_seconds = [Math]::Round($PriorElapsedSeconds + ($finished - $started).TotalSeconds, 3)
        caps = $Caps
        hard_caps = [ordered]@{
            process_memory = $true
            cpu_rate = $true
            torch_allocator_vram = $true
            vram_abort = $true
            io_abort = $true
            aggregate_time_abort = $true
        }
        peak_ram_mb = [Math]::Round($peakRamMb, 3)
        avg_ram_mb = if ($samples) { [Math]::Round($sumRamMb / $samples, 3) } else { 0 }
        peak_io_mb_s = [Math]::Round($peakIoMbS, 3)
        peak_vram_mb = [Math]::Round($peakVramMb, 3)
        avg_cpu_pct = if ($samples) { [Math]::Round($sumCpuPct / $samples, 3) } else { 0 }
        samples = $samples
        owned_pid = if ($null -eq $proc) { $null } else { $proc.Id }
        lingering_owned_process = if ($null -eq $proc) { $false } else { -not $proc.HasExited }
        lingering_owned_gpu_app = $ownedGpuAfter.Count -gt 0
        cleanup_passed = $cleanupPassed
        memory_before = $memoryBefore
        memory_after = $memoryAfter
        foreign_process_policy = "observed but never terminated"
        ignored_cpu_only_gpu_registry_pids = @($IgnoredCpuOnlyRegistryPids | Sort-Object -Unique)
    })
}
if ($status -ne "completed") { exit 1 }
