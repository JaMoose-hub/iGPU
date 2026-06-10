@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Diagnose-GameCompanion-HybridDev.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo Diagnose failed with exit code %EXITCODE%.
)
pause
exit /b %EXITCODE%
