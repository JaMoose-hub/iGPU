@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Starting Game Companion hybrid mode...
echo Cloud chat: Hermes config in WSL
echo Local router: llama.cpp Qwen3.5-2B Q4_K_M
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_game_companion_hybrid_qwen35_4b.ps1" %*

if errorlevel 1 (
  echo.
  echo Hybrid startup failed. Check logs under "%SCRIPT_DIR%logs".
  pause
  exit /b %errorlevel%
)

endlocal
