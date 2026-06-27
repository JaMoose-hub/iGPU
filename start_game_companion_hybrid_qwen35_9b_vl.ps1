param(
    [string]$Root = $PSScriptRoot,
    [string]$Python = "",
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [string]$ApiHost = "127.0.0.1",
    [int]$RouterPort = 18081,
    [string]$VulkanDevice = "0",
    [int]$RouterCtxSize = 8192,
    [int]$RouterGpuLayers = 99,
    [int]$RouterStartupTimeoutSeconds = 420,
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

function Find-FirstFile {
    param(
        [string[]]$Roots,
        [string]$Filter
    )
    foreach ($candidateRoot in $Roots) {
        if (-not $candidateRoot -or -not (Test-Path -LiteralPath $candidateRoot)) {
            continue
        }
        $hit = Get-ChildItem -Path $candidateRoot -Recurse -File -Filter $Filter -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) {
            return $hit.FullName
        }
    }
    return ""
}

$qwen35CacheRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--unsloth--Qwen3.5-9B-GGUF"
$modelRoots = @(
    (Join-Path $Root "models\Qwen3.5-9B-GGUF"),
    (Join-Path $Root "models"),
    $qwen35CacheRoot,
    (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
)
$mmprojRoots = @(
    (Join-Path $Root "models\Qwen3.5-9B-GGUF"),
    $qwen35CacheRoot,
    (Join-Path $Root "models"),
    (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
)

$routerModel = Find-FirstFile -Roots $modelRoots -Filter "Qwen3.5-9B-Q4_K_M.gguf"
$routerMmproj = Find-FirstFile -Roots $mmprojRoots -Filter "mmproj-BF16.gguf"

if (-not $routerModel) {
    throw "Missing Qwen3.5 9B model. Expected Qwen3.5-9B-Q4_K_M.gguf under models or Hugging Face cache."
}
if (-not $routerMmproj) {
    throw "Missing Qwen3.5 9B mmproj. Expected mmproj-BF16.gguf under models or Hugging Face cache."
}

$baseLauncher = Join-Path $Root "start_game_companion_hybrid_qwen35_4b.ps1"
if (-not (Test-Path -LiteralPath $baseLauncher)) {
    throw "Missing base launcher: $baseLauncher"
}

Write-Host "Starting Game Companion hybrid local vision mode."
Write-Host "Local vision router model: $routerModel"
Write-Host "Local vision router mmproj: $routerMmproj"

$env:IGPU_LIVE_STATE_TIMEOUT_SECONDS = "45"
$env:IGPU_LIVE_STATE_MAX_LONG_EDGE = "384"
$env:IGPU_LIVE_STATE_JPEG_QUALITY = "50"
$env:IGPU_LIVE_STATE_INTERVAL_MS = "10000"
$env:IGPU_LIVE_STATE_MIN_ANALYZE_GAP_MS = "15000"

& $baseLauncher `
    -Root $Root `
    -Python $Python `
    -ApiUrl $ApiUrl `
    -ApiHost $ApiHost `
    -RouterPort $RouterPort `
    -RouterModelPath $routerModel `
    -RouterMmprojPath $routerMmproj `
    -RouterAlias "qwen3.5-9b-q4_k_m-vl" `
    -RouterLogName "hybrid-qwen35-9b-vl-router" `
    -VulkanDevice $VulkanDevice `
    -RouterCtxSize $RouterCtxSize `
    -RouterGpuLayers $RouterGpuLayers `
    -RouterImageMinTokens 128 `
    -RouterImageMaxTokens 384 `
    -RouterStartupTimeoutSeconds $RouterStartupTimeoutSeconds `
    -TimeoutSeconds $TimeoutSeconds `
    -SkipModelDownload
