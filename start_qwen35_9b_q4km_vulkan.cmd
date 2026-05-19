@echo off
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_qwen35_9b_q4km_vulkan.ps1" %*
