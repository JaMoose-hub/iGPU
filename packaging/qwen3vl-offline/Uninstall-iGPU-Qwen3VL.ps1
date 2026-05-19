param(
    [string]$InstallDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

foreach ($port in 8000, 18080) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) {
        continue
    }
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
        if ("$($process.CommandLine)" -match "llama_vulkan_api_server\.py|llama-server\.exe") {
            Stop-Process -Id $conn.OwningProcess -Force
        }
    }
    catch {
    }
}

$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "iGPU Qwen3VL.lnk"
if (Test-Path -LiteralPath $desktopShortcut) {
    Remove-Item -LiteralPath $desktopShortcut -Force
}

$startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "iGPU Qwen3VL"
if (Test-Path -LiteralPath $startMenu) {
    Remove-Item -LiteralPath $startMenu -Recurse -Force
}

if (Test-Path -LiteralPath $InstallDir) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}

Write-Host "Removed iGPU Qwen3VL from: $InstallDir"
