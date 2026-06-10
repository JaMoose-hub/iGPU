param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [int]$RouterPort = 18081,
    [string]$RouterModelPath = "",
    [string]$RouterAlias = "qwen3.5-4b-q4_k_m",
    [string]$VulkanDevice = "0",
    [int]$RouterCtxSize = 8192,
    [int]$RouterGpuLayers = 99,
    [int]$RouterStartupTimeoutSeconds = 300,
    [string]$HermesWslDistro = "Ubuntu-24.04",
    [int]$HermesTimeoutSeconds = 900,
    [int]$HermesMaxTokens = 160,
    [int]$HermesContextLength = 65536,
    [string]$CloudModelAlias = "gpt-5.5-hermes",
    [int]$TimeoutSeconds = 90,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = Join-Path $env:USERPROFILE "Miniconda3\envs\igpu\python.exe"
    }
}

if (-not $RouterModelPath) {
    $RouterModelPath = Join-Path $Root "models\Qwen3.5-4B-Q4_K_M.gguf"
}

$backendScript = Join-Path $Root "llama_vulkan_api_server.py"
$overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\release\overlay-chat.exe"
if (-not (Test-Path -LiteralPath $overlayExe)) {
    $overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\debug\overlay-chat.exe"
}
$logDir = Join-Path $Root "logs"
$routerStdoutLog = Join-Path $logDir "hybrid-qwen35-4b-router.log"
$routerStderrLog = Join-Path $logDir "hybrid-qwen35-4b-router.err.log"
$backendStdoutLog = Join-Path $logDir "hybrid-cloud-api.log"
$backendStderrLog = Join-Path $logDir "hybrid-cloud-api.err.log"
$apiPort = ([uri]$ApiUrl).Port
$routerUrl = "http://127.0.0.1:$RouterPort"

function Resolve-LlamaServer {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "llama-gemma4-e4b\tools\llama.cpp-vulkan\llama-server.exe"),
        (Join-Path $Root "dist\iGPU-Qwen3VL-NoModel-Installer\payload\tools\llama.cpp-vulkan\llama-server.exe"),
        (Join-Path $Root "dist\iGPU-Qwen3VL-Offline-Installer\payload\tools\llama.cpp-vulkan\llama-server.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    $winget = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages") -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($winget) {
        return $winget.FullName
    }
    throw "Missing llama-server.exe. Install llama.cpp Vulkan tools first."
}

function Stop-ListenerOnPort {
    param([int]$Port)
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $listeners) {
        $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $proc.Id -Force
        }
    }
}

function Stop-Overlay {
    Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force
    }
}

function Test-OpenAIServiceReady {
    param([string]$BaseUrl)
    try {
        $models = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 2
        return $null -ne $models
    }
    catch {
        return $false
    }
}

function Test-HybridBackendReady {
    try {
        $health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 3
        return (
            $health.status -eq "ok" -and
            "$($health.chat_backend)" -eq "hermes" -and
            "$($health.llama_auto_start)".ToLowerInvariant() -eq "false" -and
            "$($health.hermes_use_config_model)".ToLowerInvariant() -eq "true" -and
            "$($health.local_router_enabled)".ToLowerInvariant() -eq "true" -and
            "$($health.local_router_ready)".ToLowerInvariant() -eq "true" -and
            "$($health.local_router_model)" -eq $RouterAlias
        )
    }
    catch {
        return $false
    }
}

function Start-Overlay {
    $resolved = (Resolve-Path -LiteralPath $overlayExe).Path
    $running = Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $resolved } |
        Select-Object -First 1
    if ($running) {
        Write-Host "Overlay already running. PID: $($running.Id)"
        return
    }
    Start-Process -FilePath $overlayExe -WorkingDirectory (Split-Path -Parent $overlayExe) | Out-Null
    Write-Host "Overlay started."
}

function Ensure-RouterModel {
    if (Test-Path -LiteralPath $RouterModelPath) {
        return
    }
    if ($SkipModelDownload) {
        throw "Missing router model: $RouterModelPath"
    }

    $hfCli = Join-Path $env:USERPROFILE "Miniconda3\Scripts\huggingface-cli.exe"
    if (-not (Test-Path -LiteralPath $hfCli)) {
        $cmd = Get-Command huggingface-cli -ErrorAction SilentlyContinue
        if ($cmd) {
            $hfCli = $cmd.Source
        }
    }
    if (-not (Test-Path -LiteralPath $hfCli)) {
        throw "Missing huggingface-cli. Cannot download $RouterAlias."
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RouterModelPath) | Out-Null
    $env:PYTHONIOENCODING = "utf-8"
    Write-Host "Downloading Qwen3.5 4B Q4_K_M router model..."
    & $hfCli download jc-builds/Qwen3.5-4B-Q4_K_M-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir (Join-Path $Root "models")
    if (-not (Test-Path -LiteralPath $RouterModelPath)) {
        throw "Model download finished but file was not found: $RouterModelPath"
    }
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

$llamaServer = Resolve-LlamaServer
Ensure-RouterModel
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "Starting Game Companion hybrid mode."
Write-Host "Cloud: Hermes config in WSL"
Write-Host "Local router: $RouterAlias on $routerUrl"
Write-Host "llama.cpp auto-start inside backend: disabled"

Stop-Overlay
Stop-ListenerOnPort -Port $apiPort
Stop-ListenerOnPort -Port 18080
Stop-ListenerOnPort -Port $RouterPort
Start-Sleep -Seconds 1

foreach ($log in @($routerStdoutLog, $routerStderrLog, $backendStdoutLog, $backendStderrLog)) {
    if (Test-Path -LiteralPath $log) {
        Clear-Content -LiteralPath $log -ErrorAction SilentlyContinue
    }
}

$routerArgs = @(
    "--model", $RouterModelPath,
    "--host", "127.0.0.1",
    "--port", "$RouterPort",
    "--ctx-size", "$RouterCtxSize",
    "--device", "Vulkan0",
    "--n-gpu-layers", "$RouterGpuLayers",
    "--alias", $RouterAlias,
    "--jinja",
    "--reasoning", "off",
    "--flash-attn", "off",
    "--temp", "0.2",
    "--top-p", "0.8",
    "--top-k", "20"
)

$routerEnvDevice = $VulkanDevice
$env:GGML_VK_VISIBLE_DEVICES = $routerEnvDevice
$env:LLAMA_ARG_FLASH_ATTN = "0"

$routerProcess = Start-Process `
    -FilePath $llamaServer `
    -ArgumentList $routerArgs `
    -WorkingDirectory (Split-Path -Parent $llamaServer) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $routerStdoutLog `
    -RedirectStandardError $routerStderrLog `
    -PassThru

$routerDeadline = (Get-Date).AddSeconds($RouterStartupTimeoutSeconds)
while ((Get-Date) -lt $routerDeadline) {
    if ($routerProcess.HasExited) {
        Write-Host "Router stdout:"
        Get-Content -LiteralPath $routerStdoutLog -Tail 80 -ErrorAction SilentlyContinue
        Write-Host "Router stderr:"
        Get-Content -LiteralPath $routerStderrLog -Tail 80 -ErrorAction SilentlyContinue
        throw "Local router exited early with code $($routerProcess.ExitCode)."
    }
    if (Test-OpenAIServiceReady -BaseUrl $routerUrl) {
        Write-Host "Local Qwen router ready: $routerUrl"
        break
    }
    Start-Sleep -Seconds 1
}
if (-not (Test-OpenAIServiceReady -BaseUrl $routerUrl)) {
    throw "Local router did not become ready in $RouterStartupTimeoutSeconds seconds. Check $routerStdoutLog"
}

$env:IGPU_CHAT_BACKEND = "hermes"
$env:LLAMA_AUTO_START = "0"
$env:HERMES_USE_CONFIG_MODEL = "1"
$env:IGPU_ENABLE_LOCAL_TOOLS = "0"
$env:HERMES_WSL_DISTRO = $HermesWslDistro
$env:HERMES_TIMEOUT_SECONDS = "$HermesTimeoutSeconds"
$env:HERMES_API_TIMEOUT = "$HermesTimeoutSeconds"
$env:HERMES_API_CALL_STALE_TIMEOUT = "$HermesTimeoutSeconds"
$env:HERMES_MAX_TOKENS = "$HermesMaxTokens"
$env:HERMES_AGENT_WEB_ENABLED = "1"
$env:HERMES_AGENT_TOOLSETS = "web"
$env:HERMES_AGENT_MAX_TOKENS = "360"
$env:HERMES_CONTEXT_LENGTH = "$HermesContextLength"
$env:LLAMA_MODEL_ALIAS = $CloudModelAlias
$env:IGPU_API_HOST = $ApiHost
$env:IGPU_LOCAL_ROUTER_ENABLED = "1"
$env:IGPU_LOCAL_ROUTER_URL = $routerUrl
$env:IGPU_LOCAL_ROUTER_MODEL = $RouterAlias
$env:IGPU_LOCAL_ROUTER_ROLE = "user_intent_router"
$env:IGPU_LOCAL_ROUTER_TIMEOUT = "20"
$env:IGPU_LOCAL_ROUTER_GAMEPATH_GATE = "1"
$env:IGPU_LOCAL_ROUTER_ALWAYS_ROUTE = "1"
$env:IGPU_LOCAL_ROUTER_GAMEPATH_MAX_CHARS = "280"
$env:IGPU_LOCAL_ROUTER_RETRIEVAL_EVAL = "1"
$env:IGPU_LOCAL_ROUTER_CACHE_TTL = "600"
Remove-Item Env:\LLAMA_OPENAI_MAX_TOKENS_CAP -ErrorAction SilentlyContinue

Start-Process `
    -FilePath $Python `
    -ArgumentList @($backendScript) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $backendStdoutLog `
    -RedirectStandardError $backendStderrLog | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-HybridBackendReady) {
        Write-Host "Hybrid backend ready: $ApiUrl"
        Start-Overlay
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Host "Backend stdout:"
Get-Content -LiteralPath $backendStdoutLog -Tail 80 -ErrorAction SilentlyContinue
Write-Host "Backend stderr:"
Get-Content -LiteralPath $backendStderrLog -Tail 80 -ErrorAction SilentlyContinue
throw "Hybrid backend did not become ready in $TimeoutSeconds seconds."
