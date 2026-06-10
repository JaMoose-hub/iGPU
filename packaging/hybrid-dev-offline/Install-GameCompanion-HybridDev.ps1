param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "GameCompanion-Hybrid-Dev"),
    [switch]$NoShortcut,
    [switch]$StartAfterInstall
)

$ErrorActionPreference = "Stop"

$InstallerRoot = $PSScriptRoot
$Payload = Join-Path $InstallerRoot "payload"
$RuntimeZip = Join-Path $Payload "runtime\python-env.zip"

function Assert-Path {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing installer payload: $Path"
    }
}

function Copy-DirectoryMirror {
    param(
        [string]$Source,
        [string]$Destination
    )
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $result = robocopy $Source $Destination /MIR /R:2 /W:2 /NFL /NDL /NP
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed from $Source to $Destination with exit code $LASTEXITCODE"
    }
}

function Copy-FileIfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Source) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function New-LauncherShortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetScript,
        [string]$IconPath
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $targetCmd = [System.IO.Path]::ChangeExtension($TargetScript, ".cmd")
    if (Test-Path -LiteralPath $targetCmd) {
        $shortcut.TargetPath = $targetCmd
        $shortcut.Arguments = ""
    } else {
        $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
        $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$TargetScript`""
    }
    $shortcut.WorkingDirectory = Split-Path -Parent $TargetScript
    if (Test-Path -LiteralPath $IconPath) {
        $shortcut.IconLocation = "$IconPath,0"
    }
    $shortcut.Save()
}

Assert-Path $Payload
Assert-Path (Join-Path $Payload "app\overlay-chat.exe")
Assert-Path (Join-Path $Payload "app\llama_vulkan_api_server.py")
Assert-Path (Join-Path $Payload "app\scripts\hermes_agent_web_chat.py")
Assert-Path (Join-Path $Payload "tools\llama.cpp-vulkan\llama-server.exe")
Assert-Path (Join-Path $Payload "models\Qwen3.5-4B-Q4_K_M.gguf")
Assert-Path (Join-Path $Payload "Start-GameCompanion-HybridDev.cmd")
Assert-Path $RuntimeZip

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Copy-DirectoryMirror (Join-Path $Payload "app") (Join-Path $InstallDir "app")
Copy-DirectoryMirror (Join-Path $Payload "tools") (Join-Path $InstallDir "tools")
Copy-DirectoryMirror (Join-Path $Payload "models") (Join-Path $InstallDir "models")

if (Test-Path -LiteralPath (Join-Path $Payload "docs")) {
    Copy-DirectoryMirror (Join-Path $Payload "docs") (Join-Path $InstallDir "docs")
}

$configSource = Join-Path $Payload "config"
$configDest = Join-Path $InstallDir "config"
New-Item -ItemType Directory -Force -Path $configDest | Out-Null
Copy-FileIfExists (Join-Path $configSource "game_companion.env.example") (Join-Path $configDest "game_companion.env.example")
Copy-FileIfExists (Join-Path $configSource "README-HERMES-B.txt") (Join-Path $configDest "README-HERMES-B.txt")
if (-not (Test-Path -LiteralPath (Join-Path $configDest "game_companion.env"))) {
    Copy-Item -LiteralPath (Join-Path $configSource "game_companion.env.example") -Destination (Join-Path $configDest "game_companion.env") -Force
}

foreach ($name in @(
    "Start-GameCompanion-HybridDev.ps1",
    "Start-GameCompanion-HybridDev.cmd",
    "Stop-GameCompanion-HybridDev.ps1",
    "Stop-GameCompanion-HybridDev.cmd",
    "Diagnose-GameCompanion-HybridDev.ps1",
    "Diagnose-GameCompanion-HybridDev.cmd",
    "Uninstall-GameCompanion-HybridDev.ps1"
)) {
    Copy-Item -LiteralPath (Join-Path $Payload $name) -Destination (Join-Path $InstallDir $name) -Force
}

$PythonRoot = Join-Path $InstallDir "runtime\python"
if (-not (Test-Path -LiteralPath (Join-Path $PythonRoot "python.exe"))) {
    if (Test-Path -LiteralPath $PythonRoot) {
        Remove-Item -LiteralPath $PythonRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $PythonRoot | Out-Null
    Expand-Archive -LiteralPath $RuntimeZip -DestinationPath $PythonRoot -Force
    $CondaUnpack = Join-Path $PythonRoot "Scripts\conda-unpack.exe"
    if (Test-Path -LiteralPath $CondaUnpack) {
        & $CondaUnpack
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "data\gamepath") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "logs") | Out-Null

if (-not $NoShortcut) {
    $launcher = Join-Path $InstallDir "Start-GameCompanion-HybridDev.ps1"
    $icon = Join-Path $InstallDir "app\overlay-chat.exe"
    $desktop = [Environment]::GetFolderPath("Desktop")
    New-LauncherShortcut `
        -ShortcutPath (Join-Path $desktop "Game Companion Hybrid Dev.lnk") `
        -TargetScript $launcher `
        -IconPath $icon

    $startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "Game Companion Hybrid Dev"
    New-Item -ItemType Directory -Force -Path $startMenu | Out-Null
    New-LauncherShortcut `
        -ShortcutPath (Join-Path $startMenu "Game Companion Hybrid Dev.lnk") `
        -TargetScript $launcher `
        -IconPath $icon
}

Write-Host "Installed Game Companion Hybrid Dev to: $InstallDir"
Write-Host "Bundled local model: $InstallDir\models\Qwen3.5-4B-Q4_K_M.gguf"
Write-Host "Hermes is not bundled. Configure: $InstallDir\config\game_companion.env"
Write-Host "Launch with: $InstallDir\Start-GameCompanion-HybridDev.cmd"

if ($StartAfterInstall) {
    & (Join-Path $InstallDir "Start-GameCompanion-HybridDev.cmd")
}
