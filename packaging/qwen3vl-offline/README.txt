iGPU Qwen3VL Offline Installer
==============================

This offline package installs:
- Game Companion overlay
- Local FastAPI backend
- llama.cpp Vulkan runtime
- Qwen3-VL-8B-Instruct GGUF model
- mmproj-BF16 vision projector
- Python runtime with required packages
- faster-whisper base cache for voice fallback

Install
-------
1. Copy this whole folder to the target Windows PC.
2. Double-click Install-iGPU-Qwen3VL.cmd.

If Windows blocks the script, right-click Install-iGPU-Qwen3VL.ps1 and run with PowerShell, or run:

   powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-iGPU-Qwen3VL.ps1

Default install path:
%LOCALAPPDATA%\iGPU-Qwen3VL

Launch
------
Use the desktop shortcut "iGPU Qwen3VL", or run:

   "%LOCALAPPDATA%\iGPU-Qwen3VL\Start-iGPU-Qwen3VL.cmd"

The overlay window opens immediately. The model backend can still take a few minutes
to finish loading the first time.

Uninstall
---------
Run:

   powershell -NoProfile -ExecutionPolicy Bypass -File "%LOCALAPPDATA%\iGPU-Qwen3VL\Uninstall-iGPU-Qwen3VL.ps1"

Notes
-----
- The package is large because it includes the local Qwen3-VL model.
- Vulkan support depends on the target PC GPU driver.
- If startup fails, check logs under:
  %LOCALAPPDATA%\iGPU-Qwen3VL\logs
- Screenshot analysis automatically restarts llama-server once and retries with a
  smaller image if the model service resets the connection.
