@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Starting Game Companion Cloud Hermes mode...
echo This mode uses WSL Hermes config and does NOT start llama.cpp.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_game_companion_cloud_hermes.ps1" %*

if errorlevel 1 (
  echo.
  echo Cloud Hermes startup failed. Check logs under "%SCRIPT_DIR%logs".
  pause
  exit /b %errorlevel%
)

endlocal
