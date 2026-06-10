@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-GameCompanion-HybridDev.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo Install failed with exit code %EXITCODE%.
  echo This window is staying open so you can read the error.
  pause
)
exit /b %EXITCODE%
