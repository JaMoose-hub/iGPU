param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [string]$HermesWslDistro = "Ubuntu-24.04",
    [int]$HermesTimeoutSeconds = 900,
    [int]$HermesMaxTokens = 160,
    [int]$HermesContextLength = 65536,
    [string]$ModelAlias = "gpt-5.5-hermes",
    [int]$TimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = Join-Path $env:USERPROFILE "Miniconda3\envs\igpu\python.exe"
    }
}

$backendScript = Join-Path $Root "llama_vulkan_api_server.py"
$overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\release\overlay-chat.exe"
if (-not (Test-Path -LiteralPath $overlayExe)) {
    $overlayExe = Join-Path $Root "overlay-chat\src-tauri\target\debug\overlay-chat.exe"
}
$logDir = Join-Path $Root "logs"
$stdoutLog = Join-Path $logDir "cloud-hermes-api.log"
$stderrLog = Join-Path $logDir "cloud-hermes-api.err.log"
$apiPort = ([uri]$ApiUrl).Port

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

function Test-CloudBackendReady {
    try {
        $health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 2
        return (
            $health.status -eq "ok" -and
            "$($health.chat_backend)" -eq "hermes" -and
            "$($health.llama_auto_start)".ToLowerInvariant() -eq "false" -and
            "$($health.hermes_use_config_model)".ToLowerInvariant() -eq "true"
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

Write-Host "Starting Game Companion cloud Hermes mode."
Write-Host "Hermes model comes from WSL ~/.hermes/config.yaml."
Write-Host "llama.cpp auto-start: disabled"

Stop-Overlay
Stop-ListenerOnPort -Port $apiPort
Stop-ListenerOnPort -Port 18080
Start-Sleep -Seconds 1
if (Test-Path -LiteralPath $stdoutLog) { Clear-Content -LiteralPath $stdoutLog -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $stderrLog) { Clear-Content -LiteralPath $stderrLog -ErrorAction SilentlyContinue }

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
$env:LLAMA_MODEL_ALIAS = $ModelAlias
$env:IGPU_API_HOST = $ApiHost
Remove-Item Env:\LLAMA_OPENAI_MAX_TOKENS_CAP -ErrorAction SilentlyContinue

Start-Process `
    -FilePath $Python `
    -ArgumentList @($backendScript) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-CloudBackendReady) {
        Write-Host "Cloud Hermes backend ready: $ApiUrl"
        Start-Overlay
        $llama = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
        if ($llama) {
            Stop-Process -Name "llama-server" -Force
            throw "llama-server was found and has been stopped. Cloud mode should not run llama.cpp."
        }
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Host "Backend did not become ready. stdout:"
Get-Content -LiteralPath $stdoutLog -Tail 80 -ErrorAction SilentlyContinue
Write-Host "stderr:"
Get-Content -LiteralPath $stderrLog -Tail 80 -ErrorAction SilentlyContinue
throw "Cloud Hermes backend did not become ready in $TimeoutSeconds seconds."
