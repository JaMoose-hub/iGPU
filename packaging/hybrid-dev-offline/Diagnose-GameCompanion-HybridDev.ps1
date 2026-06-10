param(
    [string]$Root = $PSScriptRoot,
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [int]$RouterPort = 18081
)

$ErrorActionPreference = "Continue"

$Python = Join-Path $Root "runtime\python\python.exe"
$Backend = Join-Path $Root "app\llama_vulkan_api_server.py"
$Overlay = Join-Path $Root "app\overlay-chat.exe"
$LlamaServer = Join-Path $Root "tools\llama.cpp-vulkan\llama-server.exe"
$Model = Join-Path $Root "models\Qwen3.5-4B-Q4_K_M.gguf"
$ConfigFile = Join-Path $Root "config\game_companion.env"

function Show-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail = ""
    )
    $status = if ($Ok) { "OK" } else { "MISSING" }
    Write-Host ("{0,-32} {1} {2}" -f $Name, $status, $Detail)
}

Show-Check "Python runtime" (Test-Path -LiteralPath $Python) $Python
Show-Check "Backend script" (Test-Path -LiteralPath $Backend) $Backend
Show-Check "Overlay executable" (Test-Path -LiteralPath $Overlay) $Overlay
Show-Check "llama-server Vulkan" (Test-Path -LiteralPath $LlamaServer) $LlamaServer
Show-Check "Qwen router model" (Test-Path -LiteralPath $Model) $Model
Show-Check "Hermes env file" (Test-Path -LiteralPath $ConfigFile) $ConfigFile

if (Test-Path -LiteralPath $LlamaServer) {
    Write-Host ""
    Write-Host "Vulkan devices reported by llama.cpp:"
    & $LlamaServer --list-devices
}

Write-Host ""
Write-Host "Listening ports:"
foreach ($port in @(8000, 18080, $RouterPort)) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object LocalAddress, LocalPort, OwningProcess |
        Format-Table -AutoSize
}

Write-Host ""
Write-Host "Backend health:"
try {
    Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 3 | ConvertTo-Json -Depth 6
}
catch {
    Write-Host "Backend is not reachable at $ApiUrl/health"
}

Write-Host ""
Write-Host "Hermes note: this package uses option B. Hermes itself must be installed/configured by each developer."
