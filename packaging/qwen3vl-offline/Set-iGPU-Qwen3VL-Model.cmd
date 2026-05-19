@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Set-iGPU-Qwen3VL-Model.ps1" %*
if errorlevel 1 (
  echo.
  echo Model path setup failed. Press any key to close this window.
  pause >nul
)
endlocal
