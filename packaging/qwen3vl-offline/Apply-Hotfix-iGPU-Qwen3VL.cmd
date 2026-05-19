@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Apply-Hotfix-iGPU-Qwen3VL.ps1" %*
if errorlevel 1 (
  echo.
  echo Hotfix failed. Press any key to close this window.
  pause >nul
)
endlocal
