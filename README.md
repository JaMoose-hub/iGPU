# Game Companion / iGPU

Game Companion 是一套 Windows 遊戲陪伴 overlay。它讓玩家不用頻繁 Alt+Tab，也能用文字、語音、截圖、GamePath 攻略庫和 Hermes Agent 取得遊戲提示。

目前主要架構是「雲地混合」：

- 本地端：llama.cpp Vulkan + Qwen router，優先使用 Intel iGPU。
- 雲端端：Hermes Agent / GPT-5.5，負責完整回答、Web search、截圖視覺分析。
- 前端：Tauri + React overlay。
- 後端：FastAPI。
- 本地知識庫：GamePath RAG Lite，使用 SQLite + Markdown runtime data。

## Quick Start

推薦使用雲地混合版：

```powershell
.\start_game_companion_hybrid_qwen35_4b.bat
```

這個啟動器會自動啟動：

1. 本地 Qwen router。
2. FastAPI backend。
3. Tauri overlay GUI。

停止程式：

```powershell
.\stop_game_companion.bat
```

## Environment

建議環境：

- Windows 11
- Python 3.12+
- Node.js 20+
- Rust stable
- Visual Studio Build Tools / Windows SDK
- llama.cpp Vulkan 版 `llama-server.exe`
- Intel GPU driver / Vulkan runtime
- 可選：WSL Ubuntu + Hermes Agent 設定

Python 環境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Frontend 環境：

```powershell
cd overlay-chat
npm ci
npm run build
npm run tauri build
```

## Model

目前 hybrid launcher 預設使用：

```text
models/Qwen3.5-2B-Q4_K_M.gguf
```

`models/` 不會進 Git。若模型不存在，啟動器會嘗試使用 `huggingface-cli` 下載。

Vulkan device 預設為：

```powershell
-VulkanDevice auto
```

啟動器會用 `llama-server --list-devices` 自動找 Intel/iGPU。若偵測失敗，會 fallback 到 `Vulkan0`。

手動指定 iGPU：

```powershell
.\start_game_companion_hybrid_qwen35_4b.bat -VulkanDevice 0
```

## Hermes Agent

Hybrid 版需要 Hermes Agent 才能使用雲端回答、Web search 和雲端 vision。

可參考：

```powershell
.\configure_hermes_qwen35_wsl.ps1
```

或：

```text
packaging/hybrid-dev-offline/README-HERMES-B.txt
packaging/hybrid-dev-offline/game_companion.env.example
```

沒有 Hermes 時，本地 router 和部分 GUI 仍可啟動，但雲端回答、Web search、截圖視覺分析會不可用或降級。

## Project Structure

```text
iGPU/
  llama_vulkan_api_server.py          Backend API
  start_game_companion_hybrid_*.bat   Main launchers
  stop_game_companion.bat             Stop services
  game_profiles.json                  Game detection profiles
  requirements.txt                    Python dependencies

  overlay-chat/                       Tauri + React frontend
    src/                              Window UI, standby UI, tools panel
    src-tauri/                        Tauri Rust shell

  docs/                               GamePath reports and architecture docs
  scripts/                            Benchmarks, importers, utility scripts
  packaging/                          Offline/dev package scripts

  gamepath/                           Runtime GamePath data, ignored by Git
  models/                             Runtime model files, ignored by Git
  logs/                               Runtime logs, ignored by Git
  runtime/                            Runtime state cache, ignored by Git
```

## Git Policy

可以進 Git：

- Backend / frontend source code
- Tauri, npm, Rust lockfiles
- 啟動腳本與打包腳本
- `docs/` 技術報告
- `packaging/` installer/dev package scripts

不要進 Git：

- `models/`
- `logs/`
- `runtime/`
- `gamepath/*.sqlite`
- `gamepath/notes/`
- `memory_cache/`
- `guide_cache/`
- `overlay-chat/node_modules/`
- `overlay-chat/src-tauri/target/`
- `dist/`

GamePath DB、玩家記憶、模型和 log 都屬於本機 runtime data，不應直接提交。

## Health Check

Backend health：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

常用靜態檢查：

```powershell
python -m py_compile llama_vulkan_api_server.py
node --check overlay-chat/src/main.js
node --check overlay-chat/src/standby.js
node --check overlay-chat/src/tools.js
git diff --check
```

## Troubleshooting

### GUI 沒出現

先確認 backend 是否活著：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

再看 logs：

```text
logs/hybrid-cloud-api.log
logs/hybrid-cloud-api.err.log
logs/hybrid-qwen35-2b-router.log
logs/hybrid-qwen35-2b-router.err.log
```

### 模型跑到 dGPU

檢查 router log：

```text
using device Vulkan0 (...)
```

若自動偵測不對，手動指定：

```powershell
.\start_game_companion_hybrid_qwen35_4b.bat -VulkanDevice 0
```

### GamePath 沒資料

GamePath 是本機 runtime data。第一次使用可能是空的，程式會在需要時建立：

```text
gamepath/gamepath.sqlite
gamepath/notes/
```

這些資料預設不進 Git。
