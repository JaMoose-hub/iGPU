@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Starting Game Companion local llama.cpp Qwen3.5 mode...
echo This mode starts llama-server.exe with Qwen3.5-9B GGUF.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_qwen35_9b_q4km_vulkan.ps1" -ChatBackend llama %*

if errorlevel 1 (
  echo.
  echo Local llama.cpp startup failed. Check logs under "%SCRIPT_DIR%logs".
  pause
  exit /b %errorlevel%
)

endlocal
