function Test-LspgCpuOnlyNvidiaRegistration {
    param(
        [Parameter(Mandatory = $true)][string]$ProcessName,
        [Parameter(Mandatory = $true)][string]$CommandLine
    )

    return (
        $ProcessName -ieq "llama-server.exe" -and
        $CommandLine -match "(?i)(?:^|\s)--n-gpu-layers\s+0(?:\s|$)"
    )
}
