param(
    [int]$TimeoutSeconds = 900,
    [string]$VulkanDevice = "0",
    [int]$LlamaCtxSize = 32768,
    [int]$LlamaGpuLayers = 99,
    [ValidateSet("llama", "hermes")]
    [string]$ChatBackend = "llama",
    [int]$HermesTimeoutSeconds = 600,
    [int]$HermesMaxTokens = 160,
    [switch]$SkipChatParsing
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$Launcher = Join-Path $Root "start_overlay_gemma4_vulkan.ps1"

if (-not (Test-Path -LiteralPath $Launcher)) {
    throw "Missing launcher: $Launcher"
}

& $Launcher `
    -Root $Root `
    -LlamaHfRepo "unsloth/Qwen3.5-9B-GGUF" `
    -LlamaHfFile "Qwen3.5-9B-Q4_K_M.gguf" `
    -LlamaModelAlias "qwen3.5-9b-q4_k_m" `
    -LlamaCtxSize $LlamaCtxSize `
    -LlamaGpuLayers $LlamaGpuLayers `
    -ChatBackend $ChatBackend `
    -HermesTimeoutSeconds $HermesTimeoutSeconds `
    -HermesMaxTokens $HermesMaxTokens `
    -LlamaSkipChatParsing:$SkipChatParsing.IsPresent `
    -VulkanDevice $VulkanDevice `
    -TimeoutSeconds $TimeoutSeconds
