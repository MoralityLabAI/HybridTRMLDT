# S0 step 2b runner (see PREREG.md). Windows PowerShell 5.1.
# Runs the three llama-perplexity passes in order and appends one JSON line per step to runner.log.
param(
    [Parameter(Mandatory)] [string] $Bin,
    [Parameter(Mandatory)] [string] $Ptq,
    [Parameter(Mandatory)] [string] $Pq2,
    [Parameter(Mandatory)] [string] $Text,
    [Parameter(Mandatory)] [string] $Work,
    [string] $Vramorama = "$env:LOCALAPPDATA\Programs\vramorama\vramorama.exe",
    [int] $Threads = 8,
    [double] $MinFreeRamGB = 10,
    # Amendment A1: extra flags for passes 1-2, e.g. '--device none --no-op-offload' for a CPU-only reference
    [string] $CpuArgs = ''
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $Work 'runner.log'
$base = Join-Path $Work 'base_ptq1_cpu.kld'
$ppl = Join-Path $Bin 'llama-perplexity.exe'

function Note($obj) { ($obj | ConvertTo-Json -Compress) | Add-Content -Encoding utf8 $log }

function Wait-Ram {
    while ($true) {
        $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
        if ($free -ge $MinFreeRamGB) { return }
        Note @{ event = 'wait_ram'; free_gb = [math]::Round($free, 2); at = (Get-Date -Format s) }
        Start-Sleep -Seconds 300
    }
}

function Step($name, $cmdline) {
    $out = Join-Path $Work "$name.log"
    $t0 = Get-Date
    Note @{ event = 'start'; step = $name; at = ($t0.ToString('s')); cmd = $cmdline }
    cmd /c "$cmdline > `"$out`" 2>&1"
    $code = $LASTEXITCODE
    Note @{ event = 'end'; step = $name; exit = $code; elapsed_s = [int]((Get-Date) - $t0).TotalSeconds }
    if ($code -ne 0) { Note @{ event = 'abort'; step = $name }; exit $code }
}

$expected = '3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1'
$got = (Get-FileHash -Algorithm SHA256 $Pq2).Hash.ToLower()
Note @{ event = 'pq2_hash'; sha256 = $got; match = ($got -eq $expected) }
if ($got -ne $expected) { Note @{ event = 'abort'; reason = 'PQ2_0 hash mismatch' }; exit 3 }

Wait-Ram
Step '01_base_ptq1_cpu' "`"$ppl`" -m `"$Ptq`" -f `"$Text`" -c 512 --chunks 32 -ngl 0 $CpuArgs -t $Threads --kl-divergence-base `"$base`""
Wait-Ram
Step '02_pq2_cpu' "`"$ppl`" -m `"$Pq2`" -c 512 --chunks 32 -ngl 0 $CpuArgs -t $Threads --kl-divergence-base `"$base`" --kl-divergence"
Step '03_ptq1_cuda' "`"$Vramorama`" run --vram 9G --owner s0-logit-parity --guard-power 600 -- `"$ppl`" -m `"$Ptq`" -c 512 --chunks 32 -ngl 99 -t $Threads --kl-divergence-base `"$base`" --kl-divergence"
Note @{ event = 'done'; at = (Get-Date -Format s) }
