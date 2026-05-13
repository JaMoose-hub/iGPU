param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "C:\Users\james\miniconda3\envs\igpu\python.exe",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [int]$LlamaPort = 18080,
    [int]$LlamaCtxSize = 32768,
    [int]$LlamaParallel = 0,
    [int]$LlamaCacheRamMiB = -1,
    [string]$VulkanDevice = "0",
    [string]$ChatBackend = "hermes",
    [string]$HermesWslDistro = "Ubuntu-24.04",
    [int]$TimeoutSeconds = 360
)

$ErrorActionPreference = "Stop"

function Test-BackendReady {
    param([string]$Url)
    try {
        $health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2
        return $health.status -eq "ok"
    }
    catch {
        return $false
    }
}

$backendScript = Join-Path $Root "llama_vulkan_api_server.py"
$overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\release\overlay-chat.exe"
$logDir = Join-Path $Root "logs"
$stdoutLog = Join-Path $logDir "llama-vulkan-api.log"
$stderrLog = Join-Path $logDir "llama-vulkan-api.err.log"

function Start-OverlayIfNeeded {
    param([string]$ExePath)

    $resolved = (Resolve-Path -LiteralPath $ExePath).Path
    $running = Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $resolved } |
        Select-Object -First 1

    if ($running) {
        Write-Host "Overlay already running. PID: $($running.Id)"
        return
    }

    Start-Process -FilePath $ExePath -WorkingDirectory (Split-Path -Parent $ExePath) | Out-Null
    Write-Host "Overlay started."
}

if (-not (Test-Path -LiteralPath $backendScript)) {
    throw "Missing backend script: $backendScript"
}
if (-not (Test-Path -LiteralPath $overlayExe)) {
    throw "Missing overlay executable: $overlayExe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing Python executable: $Python"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:GGML_VK_VISIBLE_DEVICES = $VulkanDevice
$env:LLAMA_PORT = "$LlamaPort"
$env:LLAMA_CTX_SIZE = "$LlamaCtxSize"
$env:LLAMA_ARG_FLASH_ATTN = "1"
$env:IGPU_API_HOST = $ApiHost
$env:IGPU_CHAT_BACKEND = $ChatBackend
$env:HERMES_WSL_DISTRO = $HermesWslDistro
if ($LlamaParallel -gt 0) {
    $env:LLAMA_PARALLEL = "$LlamaParallel"
}
else {
    Remove-Item Env:\LLAMA_PARALLEL -ErrorAction SilentlyContinue
}
if ($LlamaCacheRamMiB -ge 0) {
    $env:LLAMA_CACHE_RAM = "$LlamaCacheRamMiB"
}
else {
    Remove-Item Env:\LLAMA_CACHE_RAM -ErrorAction SilentlyContinue
}

if (-not (Test-BackendReady -Url $ApiUrl)) {
    if (Test-Path -LiteralPath $stdoutLog) { Clear-Content -LiteralPath $stdoutLog }
    if (Test-Path -LiteralPath $stderrLog) { Clear-Content -LiteralPath $stderrLog }

    Write-Host "Starting Gemma4 Vulkan bridge backend..."
    Start-Process `
        -FilePath $Python `
        -ArgumentList @($backendScript) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog | Out-Null
}
else {
    Write-Host "Backend is already ready."
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-BackendReady -Url $ApiUrl) {
        Write-Host "Backend ready: $ApiUrl"
        Start-OverlayIfNeeded -ExePath $overlayExe
        exit 0
    }
    Start-Sleep -Seconds 2
}

throw "Backend did not become ready in $TimeoutSeconds seconds. Check $stdoutLog and $stderrLog"
