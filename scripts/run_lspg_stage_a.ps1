param(
    [string]$ProposalDir = "experiments/loop_schedule_prognostic_gym_v0/proposals",
    [string]$Output = "experiments/loop_schedule_prognostic_gym_v0/runs/stage_a"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProposalPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $ProposalDir))
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$Stdout = Join-Path $OutputPath "stage_a.stdout.log"
$Stderr = Join-Path $OutputPath "stage_a.stderr.log"
$Receipt = Join-Path $OutputPath "stage_a.resource_receipt.json"
$MemoryLimitBytes = 2048MB
$CpuRate = 5000
$IoLimitBytesPerSecond = 50MB
$VramLimitMb = 1500
$IoViolationLimit = 3
$TimeoutSeconds = 1800
$started = Get-Date

trap {
    [ordered]@{
        stage = "A"
        status = "construction_failure"
        abort_reason = $_.Exception.Message
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        caps = [ordered]@{ ram_mb = 2048; cpu_pct = 50; io_abort_mb_s = 50; vram_mb = 1500; timeout_seconds = 1800 }
        owned_pid = $null
        cleanup_passed = $true
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $Receipt
    exit 1
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LspgJobObject {
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

$job = [LspgJobObject]::CreateJobObject([IntPtr]::Zero, "lspg-$PID-stage-a")
$limit = New-Object LspgJobObject+ExtendedInfo
$limit.basic.flags = [LspgJobObject]::ProcessMemory -bor [LspgJobObject]::KillOnClose
$limit.processMemory = [UIntPtr]::new([UInt64]$MemoryLimitBytes)
$size = [Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
if (-not [LspgJobObject]::SetInformationJobObject($job, [LspgJobObject]::Extended, $ptr, $size)) { throw "memory cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
$cpu = New-Object LspgJobObject+CpuInfo
$cpu.flags = [LspgJobObject]::CpuEnable -bor [LspgJobObject]::CpuHardCap
$cpu.rate = $CpuRate
$size = [Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr, $false)
if (-not [LspgJobObject]::SetInformationJobObject($job, [LspgJobObject]::Cpu, $ptr, $size)) { throw "CPU cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

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
    $proc = Start-Process -FilePath "python" -ArgumentList @(
        "-m", "research_gym.scripts.execute_lsa_proposal",
        "--proposal-dir", $ProposalPath,
        "--out", $OutputPath
    ) -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -NoNewWindow
    if (-not [LspgJobObject]::AssignProcessToJobObject($job, $proc.Handle)) { throw "process assignment to job failed" }
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
            $apps = & nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null
            foreach ($line in @($apps)) {
                $parts = $line -split ","
                if ($parts.Count -ge 2 -and $parts[0].Trim() -eq [string]$proc.Id) {
                    $used = [double]$parts[1].Trim()
                    $peakVramMb = [Math]::Max($peakVramMb, $used)
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
    [LspgJobObject]::CloseHandle($job) | Out-Null
    $finished = Get-Date
    [ordered]@{
        stage = "A"
        status = $status
        abort_reason = $abortReason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = $finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        caps = [ordered]@{ ram_mb = 2048; cpu_pct = 50; io_abort_mb_s = 50; io_sustained_samples = 3; vram_mb = 1500; timeout_seconds = 1800 }
        hard_caps = [ordered]@{ process_memory = $true; cpu_rate = $true; io = $false; vram = $false }
        telemetry_enforcement = "abort on sustained IO or sampled VRAM threshold"
        peak_ram_mb = [Math]::Round($peakRamMb, 3)
        avg_ram_mb = if ($samples) { [Math]::Round($sumRamMb / $samples, 3) } else { 0 }
        peak_io_mb_s = [Math]::Round($peakIoMbS, 3)
        peak_vram_mb = [Math]::Round($peakVramMb, 3)
        avg_cpu_pct = if ($samples) { [Math]::Round($sumCpuPct / $samples, 3) } else { 0 }
        samples = $samples
        owned_pid = if ($null -eq $proc) { $null } else { $proc.Id }
        lingering_owned_process = if ($null -eq $proc) { $false } else { -not $proc.HasExited }
        cleanup_passed = if ($null -eq $proc) { $true } else { $proc.HasExited }
    } | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $Receipt
}
if ($status -ne "completed") { exit 1 }
