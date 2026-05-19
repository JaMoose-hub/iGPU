@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Start-iGPU-Qwen3VL.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. Check the logs folder next to this file.
  echo Press any key to close this window.
  pause >nul
)
endlocal
