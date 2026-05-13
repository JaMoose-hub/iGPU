param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$HermesBin = "~/.local/bin/hermes",
    [string]$Model = "gemma-4-E4B-it-Q4_K_M",
    [int]$ApiPort = 8000,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"

function Invoke-Wsl {
    param([string]$Command)
    wsl -d $Distro -- bash -lc $Command
}

$gatewayLine = Invoke-Wsl "ip route | sed -n 's/^default via \([^ ]*\).*/\1/p' | head -1"
$windowsHost = ($gatewayLine | Select-Object -First 1).Trim()
if (-not $windowsHost) {
    throw "Could not detect the Windows host IP from WSL default route."
}

$baseUrl = "http://${windowsHost}:${ApiPort}/v1"
Write-Host "Configuring Hermes for $baseUrl"

Invoke-Wsl "$HermesBin config set model.provider custom"
Invoke-Wsl "$HermesBin config set model.base_url $baseUrl"
Invoke-Wsl "$HermesBin config set model.default $Model"
Invoke-Wsl "$HermesBin config set model.api_key no-key-required"
Invoke-Wsl "$HermesBin config set model.api_mode chat_completions"

Write-Host "Checking OpenAI-compatible model endpoint..."
Invoke-Wsl "curl -fsS --max-time 10 $baseUrl/models >/tmp/igpu-hermes-models.json && cat /tmp/igpu-hermes-models.json"

if ($SmokeTest) {
    $scriptPath = "/mnt/c/Project/iGPU/scripts/hermes_no_tools_chat.py"
    Write-Host "Running Hermes no-tools smoke test..."
    Invoke-Wsl "cd /mnt/c/Project/iGPU && echo 'Reply with exactly: HERMES_OK' | OPENAI_API_KEY=no-key-required ~/.hermes/hermes-agent/venv/bin/python $scriptPath --base-url $baseUrl --model $Model"
}

Write-Host ""
Write-Host "Hermes is configured. For lightweight verification:"
Write-Host "  powershell -ExecutionPolicy Bypass -File C:\Project\iGPU\configure_hermes_gemma4_wsl.ps1 -SmokeTest"
Write-Host ""
Write-Host "For full Hermes agent mode, use a small toolset first, for example:"
Write-Host "  wsl -d $Distro -- bash -lc 'cd /mnt/c/Project/iGPU && $HermesBin -z ""Create hello.txt with one friendly line."" --toolsets file --ignore-rules'"
