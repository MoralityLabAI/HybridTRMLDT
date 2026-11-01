# Q3c GPU inference baseline: llama-bench at context depth 8K and 32K, full offload, with 500 ms VRAM sampling.
# Windows PowerShell 5.1. Run inside a vramorama lease.
param(
    [Parameter(Mandatory)] [string] $Bin,
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Work,
    [int[]] $Depths = @(8192, 32768)
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Work | Out-Null
$smi = Join-Path $Work 'vram_samples.csv'
$sampler = Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,memory.used,power.draw,clocks.sm', '--format=csv,noheader,nounits', '-lms', '500', '-f', $smi -PassThru -WindowStyle Hidden
try {
    foreach ($d in $Depths) {
        $t0 = Get-Date -Format o
        $out = Join-Path $Work "bench_d$d.jsonl"
        $err = Join-Path $Work "bench_d$d.err"
        # cmd /c keeps native stderr out of PowerShell's error stream (NativeCommandError under Stop)
        cmd /c "`"$(Join-Path $Bin 'llama-bench.exe')`" -m `"$Model`" -ngl 99 -fa on -p 512 -n 128 -d $d -r 3 -o jsonl > `"$out`" 2> `"$err`""
        $code = $LASTEXITCODE
        @{ depth = $d; start = $t0; end = (Get-Date -Format o); exit = $code } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 (Join-Path $Work 'runner.log')
        if ($code -ne 0) { exit $code }
    }
} finally {
    Stop-Process -Id $sampler.Id -ErrorAction SilentlyContinue
}
