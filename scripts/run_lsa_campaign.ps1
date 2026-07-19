param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gamma", "boundary")]
    [string]$Phase,
    [string]$Output = "experiments/loop_schedule_algebra_v0"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Repo $Output))
New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
$Stdout = Join-Path $OutputPath "$Phase.stdout.log"
$Stderr = Join-Path $OutputPath "$Phase.stderr.log"
$Receipt = Join-Path $OutputPath "$Phase.resource_receipt.json"
$MemoryLimitBytes = 2048MB
$CpuRate = 5000
$IoLimitBytesPerSecond = 50MB
$IoViolationLimit = 3
$TimeoutSeconds = 1800
$started = Get-Date

trap {
    $failure = [ordered]@{
        phase = $Phase
        status = "construction_failure"
        abort_reason = $_.Exception.Message
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = (Get-Date).ToUniversalTime().ToString("o")
        caps = [ordered]@{ ram_mb = 2048; cpu_pct = 50; io_abort_mb_s = 50; io_sustained_samples = 3; timeout_seconds = 1800 }
        owned_pid = $null
        cleanup_passed = $true
    }
    $failure | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $Receipt
    exit 1
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LsaJobObject {
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

$job = [LsaJobObject]::CreateJobObject([IntPtr]::Zero, "lsa-$PID-$Phase")
$limit = New-Object LsaJobObject+ExtendedInfo
$limit.basic.flags = [LsaJobObject]::ProcessMemory -bor [LsaJobObject]::KillOnClose
$limit.processMemory = [UIntPtr]::new([UInt64]$MemoryLimitBytes)
$size = [Runtime.InteropServices.Marshal]::SizeOf($limit)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
if (-not [LsaJobObject]::SetInformationJobObject($job, [LsaJobObject]::Extended, $ptr, $size)) { throw "memory cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
$cpu = New-Object LsaJobObject+CpuInfo
$cpu.flags = [LsaJobObject]::CpuEnable -bor [LsaJobObject]::CpuHardCap
$cpu.rate = $CpuRate
$size = [Runtime.InteropServices.Marshal]::SizeOf($cpu)
$ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
[Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr, $false)
if (-not [LsaJobObject]::SetInformationJobObject($job, [LsaJobObject]::Cpu, $ptr, $size)) { throw "CPU cap setup failed" }
[Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)

$status = "running"
$abortReason = $null
$peakRamMb = 0.0
$peakIoMbS = 0.0
$ioViolations = 0
$samples = 0
$proc = $null
try {
    $proc = Start-Process -FilePath "python" -ArgumentList @(
        "-m", "research_gym.scripts.bench_loop_schedule_algebra",
        "--phase", $Phase,
        "--output", $OutputPath
    ) -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -NoNewWindow
    if (-not [LsaJobObject]::AssignProcessToJobObject($job, $proc.Handle)) { throw "process assignment to job failed" }
    $lastIo = 0.0
    $lastSample = Get-Date
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 1
        $proc.Refresh()
        $samples++
        $peakRamMb = [Math]::Max($peakRamMb, $proc.WorkingSet64 / 1MB)
        $now = Get-Date
        $elapsed = [Math]::Max(0.001, ($now - $lastSample).TotalSeconds)
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
        $lastSample = $now
        if ($ioViolations -ge $IoViolationLimit) { $abortReason = "sustained_io_cap_exceeded"; break }
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
    [LsaJobObject]::CloseHandle($job) | Out-Null
    $finished = Get-Date
    $receiptBody = [ordered]@{
        phase = $Phase
        status = $status
        abort_reason = $abortReason
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = $finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        caps = [ordered]@{ ram_mb = 2048; cpu_pct = 50; io_abort_mb_s = 50; io_sustained_samples = 3; timeout_seconds = 1800 }
        hard_caps = [ordered]@{ process_memory = $true; cpu_rate = $true; io = $false }
        io_enforcement = "abort after three consecutive one-second telemetry samples above threshold"
        peak_ram_mb = [Math]::Round($peakRamMb, 3)
        peak_io_mb_s = [Math]::Round($peakIoMbS, 3)
        samples = $samples
        owned_pid = if ($null -eq $proc) { $null } else { $proc.Id }
        lingering_owned_process = if ($null -eq $proc) { $false } else { -not $proc.HasExited }
        cleanup_passed = if ($null -eq $proc) { $true } else { $proc.HasExited }
    }
    $receiptBody | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $Receipt
}
if ($status -ne "completed") { exit 1 }
