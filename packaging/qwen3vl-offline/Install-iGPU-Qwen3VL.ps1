param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "iGPU-Qwen3VL"),
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

function Copy-Directory {
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

function New-LauncherShortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetScript,
        [string]$IconPath
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $TargetCmd = [System.IO.Path]::ChangeExtension($TargetScript, ".cmd")
    if (Test-Path -LiteralPath $TargetCmd) {
        $shortcut.TargetPath = $TargetCmd
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
Assert-Path (Join-Path $Payload "tools\llama.cpp-vulkan\llama-server.exe")
Assert-Path (Join-Path $Payload "Start-iGPU-Qwen3VL.cmd")
Assert-Path $RuntimeZip

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Copy-Directory (Join-Path $Payload "app") (Join-Path $InstallDir "app")
Copy-Directory (Join-Path $Payload "tools") (Join-Path $InstallDir "tools")
if (Test-Path -LiteralPath (Join-Path $Payload "models")) {
    Copy-Directory (Join-Path $Payload "models") (Join-Path $InstallDir "models")
} else {
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "models") | Out-Null
}

Copy-Item -LiteralPath (Join-Path $Payload "Start-iGPU-Qwen3VL.ps1") -Destination (Join-Path $InstallDir "Start-iGPU-Qwen3VL.ps1") -Force
Copy-Item -LiteralPath (Join-Path $Payload "Start-iGPU-Qwen3VL.cmd") -Destination (Join-Path $InstallDir "Start-iGPU-Qwen3VL.cmd") -Force
Copy-Item -LiteralPath (Join-Path $Payload "Uninstall-iGPU-Qwen3VL.ps1") -Destination (Join-Path $InstallDir "Uninstall-iGPU-Qwen3VL.ps1") -Force
if (Test-Path -LiteralPath (Join-Path $Payload "Set-iGPU-Qwen3VL-Model.ps1")) {
    Copy-Item -LiteralPath (Join-Path $Payload "Set-iGPU-Qwen3VL-Model.ps1") -Destination (Join-Path $InstallDir "Set-iGPU-Qwen3VL-Model.ps1") -Force
}
if (Test-Path -LiteralPath (Join-Path $Payload "Set-iGPU-Qwen3VL-Model.cmd")) {
    Copy-Item -LiteralPath (Join-Path $Payload "Set-iGPU-Qwen3VL-Model.cmd") -Destination (Join-Path $InstallDir "Set-iGPU-Qwen3VL-Model.cmd") -Force
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

if (-not $NoShortcut) {
    $Launcher = Join-Path $InstallDir "Start-iGPU-Qwen3VL.ps1"
    $Icon = Join-Path $InstallDir "app\overlay-chat.exe"
    $Desktop = [Environment]::GetFolderPath("Desktop")
    New-LauncherShortcut `
        -ShortcutPath (Join-Path $Desktop "iGPU Qwen3VL.lnk") `
        -TargetScript $Launcher `
        -IconPath $Icon

    $StartMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "iGPU Qwen3VL"
    New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null
    New-LauncherShortcut `
        -ShortcutPath (Join-Path $StartMenu "iGPU Qwen3VL.lnk") `
        -TargetScript $Launcher `
        -IconPath $Icon
}

Write-Host "Installed iGPU Qwen3VL to: $InstallDir"
Write-Host "Launch with: $InstallDir\Start-iGPU-Qwen3VL.cmd"
if (-not (Test-Path -LiteralPath (Join-Path $InstallDir "models\Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"))) {
    Write-Host "No bundled Qwen3-VL model was installed."
    Write-Host "Place model files in $InstallDir\models or run Set-iGPU-Qwen3VL-Model.cmd."
}

if ($StartAfterInstall) {
    & (Join-Path $InstallDir "Start-iGPU-Qwen3VL.cmd")
}
