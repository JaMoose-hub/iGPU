param(
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [int]$LlamaPort = 18080,
    [string]$VulkanDevice = "0"
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Python = Join-Path $Root "runtime\python\python.exe"
$Backend = Join-Path $Root "app\llama_vulkan_api_server.py"
$Overlay = Join-Path $Root "app\overlay-chat.exe"
$LlamaTools = Join-Path $Root "tools\llama.cpp-vulkan"
$ConfigDir = Join-Path $Root "config"
$ModelConfig = Join-Path $ConfigDir "model-paths.json"
$DefaultModel = Join-Path $Root "models\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
$DefaultMmproj = Join-Path $Root "models\mmproj-BF16.gguf"
$Model = $DefaultModel
$Mmproj = $DefaultMmproj
$LogDir = Join-Path $Root "logs"
$StdoutLog = Join-Path $LogDir "llama-vulkan-api.log"
$StderrLog = Join-Path $LogDir "llama-vulkan-api.err.log"
$LauncherLog = Join-Path $LogDir "launcher.log"

function Assert-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required file: $Path"
    }
}

function Write-LauncherLog {
    param([string]$Message)
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    Add-Content -LiteralPath $LauncherLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -Encoding UTF8
}

function Start-OverlayWindow {
    $runningOverlay = Get-CimInstance Win32_Process -Filter "Name = 'overlay-chat.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ExecutablePath -eq $Overlay }

    foreach ($process in $runningOverlay) {
        try {
            Stop-Process -Id $process.ProcessId -Force
            Write-LauncherLog "Stopped stale overlay process $($process.ProcessId)"
        }
        catch {
            Write-LauncherLog "Could not stop stale overlay process $($process.ProcessId): $($_.Exception.Message)"
        }
    }

    Start-Process -FilePath $Overlay -WorkingDirectory (Split-Path -Parent $Overlay) | Out-Null
    Write-LauncherLog "Started overlay window"
}

function Read-ModelConfig {
    if (-not (Test-Path -LiteralPath $ModelConfig)) {
        return
    }
    try {
        $config = Get-Content -Raw -LiteralPath $ModelConfig | ConvertFrom-Json
        if ($config.model_path) {
            $script:Model = [string]$config.model_path
        }
        if ($config.mmproj_path) {
            $script:Mmproj = [string]$config.mmproj_path
        }
        Write-LauncherLog "Loaded model config from $ModelConfig"
    }
    catch {
        Write-LauncherLog "Could not read model config: $($_.Exception.Message)"
    }
}

function Test-BackendReady {
    param([string]$Url)
    try {
        $health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2
        return $health.status -eq "ok" -and "$($health.model)" -eq "qwen3-vl-8b-instruct-ud-q4_k_xl"
    }
    catch {
        return $false
    }
}

function Stop-OwnedListener {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) {
        return
    }
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
        $cmd = "$($process.CommandLine)"
        if ($cmd -match "llama_vulkan_api_server\.py|llama-server\.exe") {
            Stop-Process -Id $conn.OwningProcess -Force
            Start-Sleep -Seconds 1
        }
    }
    catch {
    }
}

Assert-File $Python
Assert-File $Backend
Assert-File $Overlay
Assert-File (Join-Path $LlamaTools "llama-server.exe")
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-LauncherLog "Launcher started from $Root"
Read-ModelConfig

$env:LLAMA_GEMMA4_HOME = $Root
$env:LLAMA_MODEL_PATH = $Model
$env:LLAMA_MMPROJ_PATH = $Mmproj
$env:LLAMA_FORCE_LOCAL_MODEL = "1"
$env:LLAMA_HF_REPO = ""
$env:LLAMA_HF_FILE = ""
$env:LLAMA_MODEL_ALIAS = "qwen3-vl-8b-instruct-ud-q4_k_xl"
$env:GGML_VK_VISIBLE_DEVICES = $VulkanDevice
$env:LLAMA_PORT = "$LlamaPort"
$env:LLAMA_CTX_SIZE = "4096"
$env:LLAMA_GPU_LAYERS = "99"
$env:LLAMA_ARG_FLASH_ATTN = "1"
$env:LLAMA_ARG_IMAGE_MIN_TOKENS = "128"
$env:LLAMA_ARG_IMAGE_MAX_TOKENS = "256"
$env:LLAMA_VISION_LONG_EDGE = "768"
$env:LLAMA_VISION_IMAGE_FORMAT = "JPEG"
$env:LLAMA_VISION_IMAGE_QUALITY = "60"
$env:LLAMA_RETRY_VISION_LONG_EDGE = "640"
$env:LLAMA_RETRY_VISION_QUALITY = "50"
$env:LLAMA_RETRY_IMAGE_RESPONSE_TOKENS = "64"
$env:IGPU_SCREENSHOT_LONG_EDGE = "768"
$env:IGPU_SCREENSHOT_FORMAT = "JPEG"
$env:IGPU_SCREENSHOT_QUALITY = "60"
$env:LLAMA_IMAGE_RESPONSE_TOKENS = "64"
$env:LLAMA_OVERLAY_GRID_LONG_EDGE = "768"
$env:LLAMA_OVERLAY_GRID_IMAGE_QUALITY = "62"
$env:LLAMA_OVERLAY_RESPONSE_TOKENS = "128"
$env:LLAMA_SKIP_CHAT_PARSING = "1"
$env:IGPU_API_HOST = "127.0.0.1"
$env:IGPU_LOG_DIR = $LogDir
$env:IGPU_CHAT_BACKEND = "llama"
$env:IGPU_STT_MODEL = "base"
$env:IGPU_STT_DEVICE = "cpu"
$env:IGPU_STT_COMPUTE_TYPE = "int8"
$env:IGPU_STT_LANGUAGE = "zh"

Start-OverlayWindow
try {
    Assert-File $Model
    Assert-File $Mmproj
}
catch {
    Write-LauncherLog $_.Exception.Message
    throw "$($_.Exception.Message)`nPut the model files under $Root\models, or run Set-iGPU-Qwen3VL-Model.cmd to point to an external model folder."
}

if (-not (Test-BackendReady -Url $ApiUrl)) {
    Stop-OwnedListener -Port $LlamaPort
    Stop-OwnedListener -Port ([uri]$ApiUrl).Port
}

if (-not (Test-BackendReady -Url $ApiUrl)) {
    if (Test-Path -LiteralPath $StdoutLog) { Clear-Content -LiteralPath $StdoutLog }
    if (Test-Path -LiteralPath $StderrLog) { Clear-Content -LiteralPath $StderrLog }
    Start-Process `
        -FilePath $Python `
        -ArgumentList @($Backend) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog | Out-Null
    Write-LauncherLog "Started backend process"
}

$deadline = (Get-Date).AddMinutes(6)
while ((Get-Date) -lt $deadline) {
    if (Test-BackendReady -Url $ApiUrl) {
        Write-LauncherLog "Backend ready"
        exit 0
    }
    Start-Sleep -Seconds 2
}

Write-LauncherLog "Backend did not become ready in time"
throw "Backend did not become ready. Check logs in $LogDir"
