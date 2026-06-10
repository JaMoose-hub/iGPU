@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-GameCompanion-HybridDev.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo Start failed with exit code %EXITCODE%.
  echo Check logs under "%~dp0logs" if this is an installed copy.
  echo This window is staying open so you can read the error.
  pause
)
exit /b %EXITCODE%
