param(
    [string]$Root = $PSScriptRoot,
    [int[]]$Ports = @(8000, 18080, 18081)
)

$ErrorActionPreference = "SilentlyContinue"

function Stop-OwnedListenerOnPort {
    param(
        [int]$Port,
        [string]$CommandPattern
    )
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $listeners) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid"
        $cmd = "$($proc.CommandLine)"
        if ($cmd -match $CommandPattern) {
            Stop-Process -Id $ownerPid -Force
            Write-Host "Stopped PID $ownerPid on port $Port"
        }
    }
}

Get-Process -Name "overlay-chat" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.Id -Force
    Write-Host "Stopped overlay PID $($_.Id)"
}

foreach ($port in $Ports) {
    if ($port -eq 8000) {
        Stop-OwnedListenerOnPort -Port $port -CommandPattern "llama_vulkan_api_server\.py"
    } else {
        Stop-OwnedListenerOnPort -Port $port -CommandPattern "llama-server\.exe|llama_vulkan_api_server\.py"
    }
}

Write-Host "Game Companion Hybrid Dev stopped. External Hermes was not stopped."
