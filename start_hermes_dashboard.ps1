param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 9119,
    [switch]$Setup,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

function Invoke-Wsl {
    param([string]$Command)
    wsl -d $Distro -- bash -lc $Command
}

if ($Setup) {
    Write-Host "Installing Hermes dashboard dependencies..."
    Invoke-Wsl "cd ~/.hermes/hermes-agent && env -u VIRTUAL_ENV ~/.local/bin/uv pip install --python venv/bin/python -e ''.[web]''"

    Write-Host "Building Hermes dashboard frontend..."
    Invoke-Wsl "cd ~/.hermes/hermes-agent/web && CI=1 npm install --silent && CI=1 npm run build"
}

$url = "http://${HostName}:${Port}"
$existing = Invoke-Wsl "pgrep -af '[h]ermes dashboard' || true"
if ($existing -match "hermes dashboard") {
    Write-Host "Hermes dashboard already appears to be running."
}
else {
    $dashCommand = "exec ~/.local/bin/hermes dashboard --host $HostName --port $Port --no-open"
    $dashArgs = @("-d", $Distro, "--", "bash", "-lc", $dashCommand)
    Start-Process -FilePath "wsl.exe" -ArgumentList $dashArgs -WindowStyle Hidden -WorkingDirectory $PSScriptRoot
    Start-Sleep -Seconds 4
}

try {
    & curl.exe -fsS --max-time 10 $url | Out-Null
    Write-Host "Hermes dashboard ready: $url"
}
catch {
    Write-Host "Dashboard process was started, but the URL is not ready yet: $url"
    Write-Host "Check status with: wsl -d $Distro -- bash -lc '~/.local/bin/hermes dashboard --status'"
}

if (-not $NoOpen) {
    Start-Process $url
}
