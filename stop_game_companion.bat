@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Stopping iGPU Game Companion...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$stopped=@();" ^
  "Get-Process overlay-chat -ErrorAction SilentlyContinue | ForEach-Object { $stopped += ('overlay-chat:' + $_.Id); Stop-Process -Id $_.Id -Force };" ^
  "Get-NetTCPConnection -LocalPort 8000,18080 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $proc = Get-Process -Id $_ -ErrorAction SilentlyContinue; if ($proc) { $stopped += ($proc.ProcessName + ':' + $proc.Id); Stop-Process -Id $proc.Id -Force } };" ^
  "Start-Sleep -Seconds 1;" ^
  "if ($stopped.Count) { 'Stopped: ' + ($stopped -join ', ') } else { 'Nothing was running.' };" ^
  "$remaining = @(Get-Process overlay-chat,python,llama-server -ErrorAction SilentlyContinue) + @(Get-NetTCPConnection -LocalPort 8000,18080 -State Listen -ErrorAction SilentlyContinue);" ^
  "if ($remaining.Count) { 'Some processes/listeners may still be shutting down.' } else { 'All game companion services are stopped.' }"

if errorlevel 1 (
  echo.
  echo Stop command failed.
  pause
  exit /b %errorlevel%
)

endlocal
