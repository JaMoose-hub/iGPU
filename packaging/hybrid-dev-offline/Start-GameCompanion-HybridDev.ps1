param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [int]$RouterPort = 18081,
    [string]$RouterAlias = "qwen3.5-4b-q4_k_m",
    [string]$VulkanDevice = "0",
    [int]$RouterCtxSize = 8192,
    [int]$RouterGpuLayers = 99,
    [int]$RouterStartupTimeoutSeconds = 300,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

$Backend = Join-Path $Root "app\llama_vulkan_api_server.py"
$Overlay = Join-Path $Root "app\overlay-chat.exe"
$LlamaServer = Join-Path $Root "tools\llama.cpp-vulkan\llama-server.exe"
$RouterModelPath = Join-Path $Root "models\Qwen3.5-4B-Q4_K_M.gguf"
$ConfigFile = Join-Path $Root "config\game_companion.env"
$LogDir = Join-Path $Root "logs"
$DataDir = Join-Path $Root "data"
$GamePathDir = Join-Path $DataDir "gamepath"
$RouterUrl = "http://127.0.0.1:$RouterPort"
$ApiPort = ([uri]$ApiUrl).Port
$RouterStdoutLog = Join-Path $LogDir "hybrid-qwen35-4b-router.log"
$RouterStderrLog = Join-Path $LogDir "hybrid-qwen35-4b-router.err.log"
$BackendStdoutLog = Join-Path $LogDir "hybrid-cloud-api.log"
$BackendStderrLog = Join-Path $LogDir "hybrid-cloud-api.err.log"
$LauncherLog = Join-Path $LogDir "hybrid-launcher.log"

if (-not $Python) {
    $Python = Join-Path $Root "runtime\python\python.exe"
}

function Write-LauncherLog {
    param([string]$Message)
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Add-Content -LiteralPath $LauncherLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -Encoding UTF8
}

function Assert-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($Path -eq $Python -and (Test-Path -LiteralPath (Join-Path $Root "runtime\python-env.zip"))) {
            throw "This Start script is inside the installer payload or an incomplete install. Run Install-GameCompanion-HybridDev.cmd first so runtime\python-env.zip is extracted to runtime\python."
        }
        throw "Missing required file: $Path"
    }
}

function Import-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    Get-Content -LiteralPath $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) {
            return
        }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

function Get-EnvValue {
    param(
        [string]$Name,
        [string]$Default
    )
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Set-ProcessEnv {
    param(
        [string]$Name,
        [string]$Value
    )
    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Stop-OwnedListenerOnPort {
    param(
        [int]$Port,
        [string]$CommandPattern
    )
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $listeners) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid"
            $cmd = "$($proc.CommandLine)"
            if ($cmd -match $CommandPattern) {
                Stop-Process -Id $ownerPid -Force
                Write-LauncherLog "Stopped listener PID $ownerPid on port $Port"
            }
        }
        catch {
            Write-LauncherLog "Could not inspect listener $ownerPid on port ${Port}: $($_.Exception.Message)"
        }
    }
}

function Assert-PortAvailable {
    param([int]$Port)
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($listener) {
        throw "Port $Port is already in use by PID $($listener.OwningProcess). Stop that process or change the port."
    }
}

function Stop-Overlay {
    Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Stop-Process -Id $_.Id -Force
            Write-LauncherLog "Stopped overlay PID $($_.Id)"
        }
        catch {
            Write-LauncherLog "Could not stop overlay PID $($_.Id): $($_.Exception.Message)"
        }
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

function Start-OverlayWindow {
    Start-Process -FilePath $Overlay -WorkingDirectory (Split-Path -Parent $Overlay) | Out-Null
    Write-LauncherLog "Started overlay window"
}

Assert-File $Python
Assert-File $Backend
Assert-File $Overlay
Assert-File $LlamaServer
Assert-File $RouterModelPath
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $GamePathDir | Out-Null

Import-DotEnv -Path $ConfigFile

$HermesWslDistro = Get-EnvValue "HERMES_WSL_DISTRO" "Ubuntu-24.04"
$HermesTimeoutSeconds = [int](Get-EnvValue "HERMES_TIMEOUT_SECONDS" "900")
$HermesMaxTokens = [int](Get-EnvValue "HERMES_MAX_TOKENS" "160")
$HermesContextLength = [int](Get-EnvValue "HERMES_CONTEXT_LENGTH" "65536")
$CloudModelAlias = Get-EnvValue "LLAMA_MODEL_ALIAS" "gpt-5.5-hermes"
$RouterTimeout = Get-EnvValue "IGPU_LOCAL_ROUTER_TIMEOUT" "8"
$RouterGateMaxChars = Get-EnvValue "IGPU_LOCAL_ROUTER_GAMEPATH_MAX_CHARS" "280"
$RouterCacheTtl = Get-EnvValue "IGPU_LOCAL_ROUTER_CACHE_TTL" "600"
$HermesAgentMaxTokens = Get-EnvValue "HERMES_AGENT_MAX_TOKENS" "360"
$HermesToolsets = Get-EnvValue "HERMES_AGENT_TOOLSETS" "web"

Write-Host "Starting Game Companion Hybrid Dev."
Write-Host "Local router model: $RouterModelPath"
Write-Host "Local router device: Vulkan$VulkanDevice"
Write-Host "Hermes mode: external self-installed Hermes via WSL distro '$HermesWslDistro'"
Write-Host "Config file: $ConfigFile"

Stop-Overlay
Stop-OwnedListenerOnPort -Port $ApiPort -CommandPattern "llama_vulkan_api_server\.py"
Stop-OwnedListenerOnPort -Port 18080 -CommandPattern "llama-server\.exe|llama_vulkan_api_server\.py"
Stop-OwnedListenerOnPort -Port $RouterPort -CommandPattern "llama-server\.exe"
Start-Sleep -Seconds 1
Assert-PortAvailable -Port $ApiPort
Assert-PortAvailable -Port $RouterPort

foreach ($log in @($RouterStdoutLog, $RouterStderrLog, $BackendStdoutLog, $BackendStderrLog)) {
    if (Test-Path -LiteralPath $log) {
        Clear-Content -LiteralPath $log -ErrorAction SilentlyContinue
    }
}

Set-ProcessEnv "GGML_VK_VISIBLE_DEVICES" $VulkanDevice
Set-ProcessEnv "LLAMA_ARG_FLASH_ATTN" "0"

$routerArgs = @(
    "--model", $RouterModelPath,
    "--host", "127.0.0.1",
    "--port", "$RouterPort",
    "--ctx-size", "$RouterCtxSize",
    "--device", "Vulkan$VulkanDevice",
    "--n-gpu-layers", "$RouterGpuLayers",
    "--alias", $RouterAlias,
    "--jinja",
    "--reasoning", "off",
    "--flash-attn", "off",
    "--temp", "0.2",
    "--top-p", "0.8",
    "--top-k", "20"
)

$routerProcess = Start-Process `
    -FilePath $LlamaServer `
    -ArgumentList $routerArgs `
    -WorkingDirectory (Split-Path -Parent $LlamaServer) `
    -WindowStyle Hidden `
    -RedirectStandardOutput $RouterStdoutLog `
    -RedirectStandardError $RouterStderrLog `
    -PassThru

$routerDeadline = (Get-Date).AddSeconds($RouterStartupTimeoutSeconds)
while ((Get-Date) -lt $routerDeadline) {
    if ($routerProcess.HasExited) {
        Write-Host "Router stdout:"
        Get-Content -LiteralPath $RouterStdoutLog -Tail 80 -ErrorAction SilentlyContinue
        Write-Host "Router stderr:"
        Get-Content -LiteralPath $RouterStderrLog -Tail 80 -ErrorAction SilentlyContinue
        throw "Local router exited early with code $($routerProcess.ExitCode)."
    }
    if (Test-OpenAIServiceReady -BaseUrl $RouterUrl) {
        Write-Host "Local Qwen router ready: $RouterUrl"
        break
    }
    Start-Sleep -Seconds 1
}
if (-not (Test-OpenAIServiceReady -BaseUrl $RouterUrl)) {
    throw "Local router did not become ready in $RouterStartupTimeoutSeconds seconds. Check $RouterStdoutLog"
}

Set-ProcessEnv "IGPU_CHAT_BACKEND" "hermes"
Set-ProcessEnv "LLAMA_AUTO_START" "0"
Set-ProcessEnv "HERMES_USE_CONFIG_MODEL" "1"
Set-ProcessEnv "IGPU_ENABLE_LOCAL_TOOLS" "0"
Set-ProcessEnv "HERMES_WSL_DISTRO" $HermesWslDistro
Set-ProcessEnv "HERMES_TIMEOUT_SECONDS" "$HermesTimeoutSeconds"
Set-ProcessEnv "HERMES_API_TIMEOUT" "$HermesTimeoutSeconds"
Set-ProcessEnv "HERMES_API_CALL_STALE_TIMEOUT" "$HermesTimeoutSeconds"
Set-ProcessEnv "HERMES_MAX_TOKENS" "$HermesMaxTokens"
Set-ProcessEnv "HERMES_AGENT_WEB_ENABLED" "1"
Set-ProcessEnv "HERMES_AGENT_TOOLSETS" $HermesToolsets
Set-ProcessEnv "HERMES_AGENT_MAX_TOKENS" "$HermesAgentMaxTokens"
Set-ProcessEnv "HERMES_CONTEXT_LENGTH" "$HermesContextLength"
Set-ProcessEnv "LLAMA_MODEL_ALIAS" $CloudModelAlias
Set-ProcessEnv "IGPU_API_HOST" $ApiHost
Set-ProcessEnv "IGPU_LOG_DIR" $LogDir
Set-ProcessEnv "IGPU_GAMEPATH_DIR" $GamePathDir
Set-ProcessEnv "IGPU_LOCAL_ROUTER_ENABLED" "1"
Set-ProcessEnv "IGPU_LOCAL_ROUTER_URL" $RouterUrl
Set-ProcessEnv "IGPU_LOCAL_ROUTER_MODEL" $RouterAlias
Set-ProcessEnv "IGPU_LOCAL_ROUTER_ROLE" "gamepath_router"
Set-ProcessEnv "IGPU_LOCAL_ROUTER_TIMEOUT" $RouterTimeout
Set-ProcessEnv "IGPU_LOCAL_ROUTER_GAMEPATH_GATE" "1"
Set-ProcessEnv "IGPU_LOCAL_ROUTER_GAMEPATH_MAX_CHARS" $RouterGateMaxChars
Set-ProcessEnv "IGPU_LOCAL_ROUTER_RETRIEVAL_EVAL" "1"
Set-ProcessEnv "IGPU_LOCAL_ROUTER_CACHE_TTL" $RouterCacheTtl
Remove-Item Env:\LLAMA_OPENAI_MAX_TOKENS_CAP -ErrorAction SilentlyContinue

Start-Process `
    -FilePath $Python `
    -ArgumentList @($Backend) `
    -WorkingDirectory (Join-Path $Root "app") `
    -WindowStyle Hidden `
    -RedirectStandardOutput $BackendStdoutLog `
    -RedirectStandardError $BackendStderrLog | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-HybridBackendReady) {
        Write-Host "Hybrid backend ready: $ApiUrl"
        Start-OverlayWindow
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Host "Backend stdout:"
Get-Content -LiteralPath $BackendStdoutLog -Tail 80 -ErrorAction SilentlyContinue
Write-Host "Backend stderr:"
Get-Content -LiteralPath $BackendStderrLog -Tail 80 -ErrorAction SilentlyContinue
throw "Hybrid backend did not become ready in $TimeoutSeconds seconds."
