param(
    [string]$ModelPath,
    [string]$MmprojPath
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$ConfigDir = Join-Path $Root "config"
$ConfigPath = Join-Path $ConfigDir "model-paths.json"
$DefaultModel = Join-Path $Root "models\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
$DefaultMmproj = Join-Path $Root "models\mmproj-BF16.gguf"

if (-not $ModelPath) {
    if (Test-Path -LiteralPath $DefaultModel) {
        $ModelPath = $DefaultModel
    } else {
        $ModelPath = Read-Host "Path to Qwen3-VL GGUF model"
    }
}

if (-not $MmprojPath) {
    if (Test-Path -LiteralPath $DefaultMmproj) {
        $MmprojPath = $DefaultMmproj
    } else {
        $MmprojPath = Read-Host "Path to mmproj-BF16.gguf"
    }
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model file not found: $ModelPath"
}
if (-not (Test-Path -LiteralPath $MmprojPath)) {
    throw "Vision projector file not found: $MmprojPath"
}

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$config = [ordered]@{
    model_path = (Resolve-Path -LiteralPath $ModelPath).Path
    mmproj_path = (Resolve-Path -LiteralPath $MmprojPath).Path
}
$config | ConvertTo-Json | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

Write-Host "Saved model config:"
Write-Host "  Model:  $($config.model_path)"
Write-Host "  Mmproj: $($config.mmproj_path)"
Write-Host "Launch with: $Root\Start-iGPU-Qwen3VL.cmd"
