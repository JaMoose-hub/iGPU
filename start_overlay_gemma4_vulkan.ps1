param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [int]$LlamaPort = 18080,
    [int]$LlamaCtxSize = 8192,
    [int]$LlamaGpuLayers = 99,
    [int]$LlamaParallel = 0,
    [int]$LlamaCacheRamMiB = -1,
    [string]$VulkanDevice = "0",
    [string]$ChatBackend = "llama",
    [string]$LlamaHfRepo = "unsloth/Qwen3-VL-8B-Instruct-GGUF",
    [string]$LlamaHfFile = "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
    [string]$LlamaModelAlias = "qwen3-vl-8b-instruct-ud-q4_k_xl",
    [int]$LlamaImageMinTokens = 256,
    [int]$LlamaImageMaxTokens = 512,
    [string]$HermesWslDistro = "Ubuntu-24.04",
    [int]$TimeoutSeconds = 360
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = Join-Path $env:USERPROFILE "Miniconda3\envs\igpu\python.exe"
    }
}

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

function Stop-ListenerOnPort {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force
        Start-Sleep -Seconds 1
    }
}

function Get-LlamaModelIds {
    param([int]$Port)
    try {
        $models = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2
        return @($models.data | ForEach-Object { "$($_.id)" })
    }
    catch {
        return @()
    }
}

function Get-ListenerProcessCommandLine {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) {
        return ""
    }
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
        return "$($process.CommandLine)"
    }
    catch {
        return ""
    }
}

$backendScript = Join-Path $Root "llama_vulkan_api_server.py"
$overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\release\overlay-chat.exe"
if (-not (Test-Path -LiteralPath $overlayExe)) {
    $overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\debug\overlay-chat.exe"
}
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
$env:LLAMA_GPU_LAYERS = "$LlamaGpuLayers"
$env:LLAMA_ARG_FLASH_ATTN = "1"
$env:IGPU_API_HOST = $ApiHost
$env:IGPU_CHAT_BACKEND = $ChatBackend
$env:LLAMA_MODEL_ALIAS = $LlamaModelAlias
$env:LLAMA_HF_REPO = $LlamaHfRepo
$env:LLAMA_HF_FILE = $LlamaHfFile
$env:LLAMA_CHAT_TEMPLATE_KWARGS = ""
$env:LLAMA_ARG_IMAGE_MIN_TOKENS = "$LlamaImageMinTokens"
$env:LLAMA_ARG_IMAGE_MAX_TOKENS = "$LlamaImageMaxTokens"
$env:LLAMA_VISION_LONG_EDGE = "960"
$env:LLAMA_VISION_IMAGE_FORMAT = "JPEG"
$env:LLAMA_VISION_IMAGE_QUALITY = "65"
$env:IGPU_SCREENSHOT_LONG_EDGE = "960"
$env:IGPU_SCREENSHOT_FORMAT = "JPEG"
$env:IGPU_SCREENSHOT_QUALITY = "65"
$env:LLAMA_IMAGE_RESPONSE_TOKENS = "64"
$env:LLAMA_OVERLAY_GRID_LONG_EDGE = "960"
$env:LLAMA_OVERLAY_GRID_IMAGE_QUALITY = "68"
$env:LLAMA_OVERLAY_RESPONSE_TOKENS = "128"
$env:LLAMA_SKIP_CHAT_PARSING = "1"
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

$runningModelIds = Get-LlamaModelIds -Port $LlamaPort
if ($runningModelIds.Count -gt 0 -and ($runningModelIds -notcontains $LlamaModelAlias)) {
    Write-Host "Stopping llama-server on port $LlamaPort to switch model to $LlamaModelAlias."
    Stop-ListenerOnPort -Port $LlamaPort
    Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
}
elseif ($runningModelIds.Count -gt 0) {
    $llamaCommandLine = Get-ListenerProcessCommandLine -Port $LlamaPort
    if ($llamaCommandLine -and $llamaCommandLine -notmatch "--n-gpu-layers\s+$LlamaGpuLayers(\s|$)") {
        Write-Host "Stopping llama-server on port $LlamaPort to switch GPU layers to $LlamaGpuLayers."
        Stop-ListenerOnPort -Port $LlamaPort
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
}

try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 2
    if ($health.model -and "$($health.model)" -ne $LlamaModelAlias) {
        Write-Host "Stopping backend to switch model alias from $($health.model) to $LlamaModelAlias."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.llama_gpu_layers -and "$($health.llama_gpu_layers)" -ne "$LlamaGpuLayers") {
        Write-Host "Stopping backend to switch GPU layers from $($health.llama_gpu_layers) to $LlamaGpuLayers."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.llama_image_min_tokens -and "$($health.llama_image_min_tokens)" -ne "$LlamaImageMinTokens") {
        Write-Host "Stopping backend and llama-server to switch image min tokens from $($health.llama_image_min_tokens) to $LlamaImageMinTokens."
        Stop-ListenerOnPort -Port $LlamaPort
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.llama_image_max_tokens -and "$($health.llama_image_max_tokens)" -ne "$LlamaImageMaxTokens") {
        Write-Host "Stopping backend and llama-server to switch image max tokens from $($health.llama_image_max_tokens) to $LlamaImageMaxTokens."
        Stop-ListenerOnPort -Port $LlamaPort
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.vision_long_edge -and "$($health.vision_long_edge)" -ne "960") {
        Write-Host "Stopping backend to switch vision long edge from $($health.vision_long_edge) to 960."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.screenshot_long_edge -and "$($health.screenshot_long_edge)" -ne "960") {
        Write-Host "Stopping backend to switch screenshot long edge from $($health.screenshot_long_edge) to 960."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.image_response_tokens -and "$($health.image_response_tokens)" -ne "64") {
        Write-Host "Stopping backend to switch image response tokens from $($health.image_response_tokens) to 64."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.overlay_grid_long_edge -and "$($health.overlay_grid_long_edge)" -ne "960") {
        Write-Host "Stopping backend to switch overlay grid long edge from $($health.overlay_grid_long_edge) to 960."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
    elseif ($health.overlay_response_tokens -and "$($health.overlay_response_tokens)" -ne "128") {
        Write-Host "Stopping backend to switch overlay response tokens from $($health.overlay_response_tokens) to 128."
        Stop-ListenerOnPort -Port ([uri]$ApiUrl).Port
    }
}
catch {
}

if (-not (Test-BackendReady -Url $ApiUrl)) {
    if (Test-Path -LiteralPath $stdoutLog) { Clear-Content -LiteralPath $stdoutLog }
    if (Test-Path -LiteralPath $stderrLog) { Clear-Content -LiteralPath $stderrLog }

    Write-Host "Starting Vulkan bridge backend for $LlamaModelAlias..."
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
