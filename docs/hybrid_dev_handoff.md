# Game Companion Hybrid Dev Handoff

This note is for developers who install or clone the hybrid Game Companion build.

## What This Build Runs

- Frontend overlay: `overlay-chat`
- Backend API: `llama_vulkan_api_server.py`
- Local router: llama.cpp Vulkan + `Qwen3.5-4B-Q4_K_M.gguf`
- Cloud agent: external Hermes installation, configured per developer
- GamePath: local SQLite + Markdown RAG Lite knowledge base

The local Qwen router is used for user-intent routing and GamePath retrieval evaluation. Main chat and web search are handled through Hermes.

## Developer Repo Startup

From the repo root:

```bat
start_game_companion_hybrid_qwen35_4b.bat
```

Defaults:

- Backend: `http://127.0.0.1:8000`
- Local Qwen router: `http://127.0.0.1:18081`
- Local router model: `models\Qwen3.5-4B-Q4_K_M.gguf`
- Vulkan device: `0`
- Hermes WSL distro: `Ubuntu-24.04`

Stop everything started by the companion:

```bat
stop_game_companion.bat
```

## Packaged Installer Startup

For the hybrid dev offline package:

```bat
Install-GameCompanion-HybridDev.cmd
Start-GameCompanion-HybridDev.cmd
Diagnose-GameCompanion-HybridDev.cmd
Stop-GameCompanion-HybridDev.cmd
```

Hermes is option B: it is not bundled. Each developer installs Hermes separately, then edits:

```text
config\game_companion.env
```

No API keys or Hermes secrets should be committed.

## Health Checks

Backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected key fields:

- `status: ok`
- `chat_backend: hermes`
- `local_router_enabled: true`
- `local_router_ready: true`
- `local_router_model: qwen3.5-4b-q4_k_m`
- `gamepath_enabled: true`

Router:

```powershell
Invoke-RestMethod http://127.0.0.1:18081/v1/models
```

Expected model:

- `qwen3.5-4b-q4_k_m`

Confirm iGPU/Vulkan device in:

```text
logs\hybrid-qwen35-4b-router.err.log
```

Look for a line like:

```text
using device Vulkan0 (Intel(R) ... Graphics Controller)
```

On another machine, Vulkan device ordering can differ. Run the diagnose script and change `VulkanDevice` or `config\game_companion.env` if needed.

## GamePath Smoke Test

```powershell
$body = @{
  game_id = "RESIDENT_EVIL_requiem"
  query = "廚房的怪物怎麼打"
  limit = 3
  spoiler_level = "low"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/gamepath/search -ContentType "application/json; charset=utf-8" -Body $body
```

Expected:

- `evaluation.confidence` is usually `direct` or `summarize` if matching data exists.
- Results contain RAG Lite metadata such as `match_coverage`, `core_overlap`, `retrieval_score`, and `rag_lite`.

## Common Issues

- GUI does not show: confirm `overlay-chat.exe` exists and no old overlay process is stuck.
- Backend unreachable: check port `8000` and `logs\hybrid-cloud-api.err.log`.
- Router unreachable: check port `18081`, model path, and `logs\hybrid-qwen35-4b-router.err.log`.
- Wrong GPU: run diagnose, inspect llama.cpp device list, then set the desired Vulkan device.
- Hermes timeout: confirm Hermes is installed, the WSL distro name is correct, and web/API secrets are configured outside this repo.
- GamePath empty: this repo does not commit `gamepath.sqlite` or `gamepath/notes`; those are local runtime data.

## Verified On This Machine

- Python compile: passed
- Frontend JS parse: passed
- Tauri/Rust `cargo check`: passed
- Backend health: `status=ok`
- Local router: `qwen3.5-4b-q4_k_m`
- Vulkan device: `Vulkan0 (Intel(R) RaptorLake-S Mobile Graphics Controller)`
- llama.cpp offload: `33/33 layers`
