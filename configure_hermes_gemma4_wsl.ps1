param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$HermesBin = "~/.local/bin/hermes",
    [string]$Model = "qwen3.5-9b-q4_k_m",
    [int]$ApiPort = 8000,
    [int]$ContextLength = 32768,
    [int]$MaxTokens = 160,
    [int]$RequestTimeoutSeconds = 600,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"

function Invoke-Wsl {
    param([string]$Command)
    wsl -d $Distro -- bash -lc $Command
}

$baseUrl = "http://127.0.0.1:${ApiPort}/v1"
Write-Host "Configuring Hermes for $baseUrl"

Invoke-Wsl "mkdir -p ~/.hermes && cat > ~/.hermes/config.yaml <<'YAML'
model:
  provider: custom
  base_url: $baseUrl
  default: $Model
  api_key: no-key-required
  api_mode: chat_completions
  context_length: $ContextLength
  max_tokens: $MaxTokens
terminal:
  backend: local
  cwd: '.'
  timeout: $RequestTimeoutSeconds
  docker_mount_cwd_to_workspace: false
  lifetime_seconds: 300
onboarding:
  seen:
    openclaw_residue_cleanup: true
YAML"
Invoke-Wsl "cat ~/.hermes/config.yaml"

Write-Host "Checking OpenAI-compatible model endpoint..."
Invoke-Wsl "curl -fsS --max-time 10 $baseUrl/models >/tmp/igpu-hermes-models.json && cat /tmp/igpu-hermes-models.json"

if ($SmokeTest) {
    $scriptPath = "/mnt/c/Projects/iGPU/scripts/hermes_no_tools_chat.py"
    Write-Host "Running Hermes no-tools smoke test..."
    Invoke-Wsl "cd /mnt/c/Projects/iGPU && echo 'Reply with exactly: HERMES_OK' | OPENAI_API_KEY=no-key-required HERMES_API_TIMEOUT=$RequestTimeoutSeconds HERMES_API_CALL_STALE_TIMEOUT=$RequestTimeoutSeconds HERMES_MAX_TOKENS=$MaxTokens ~/.hermes/hermes-agent/venv/bin/python $scriptPath --base-url $baseUrl --model $Model --max-tokens $MaxTokens --context-length $ContextLength --api-timeout $RequestTimeoutSeconds --api-call-stale-timeout $RequestTimeoutSeconds"
}

Write-Host ""
Write-Host "Hermes is configured. For lightweight verification:"
Write-Host "  powershell -ExecutionPolicy Bypass -File C:\Projects\iGPU\configure_hermes_gemma4_wsl.ps1 -SmokeTest"
Write-Host ""
Write-Host "For full Hermes agent mode, use a small toolset first, for example:"
Write-Host "  wsl -d $Distro -- bash -lc 'cd /mnt/c/Projects/iGPU && $HermesBin -z ""Create hello.txt with one friendly line."" --toolsets file --ignore-rules'"
