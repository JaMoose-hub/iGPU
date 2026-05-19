param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "iGPU-Qwen3VL")
)

$ErrorActionPreference = "Stop"

$HotfixRoot = $PSScriptRoot
$Backend = Join-Path $HotfixRoot "app\llama_vulkan_api_server.py"
$StartPs1 = Join-Path $HotfixRoot "Start-iGPU-Qwen3VL.ps1"
$StartCmd = Join-Path $HotfixRoot "Start-iGPU-Qwen3VL.cmd"

function Assert-Path {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required path: $Path"
    }
}

function Stop-InstalledProcess {
    param([string]$InstallDir)
    $escaped = [regex]::Escape($InstallDir)
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @("overlay-chat.exe", "python.exe", "pythonw.exe", "llama-server.exe") -and
            "$($_.CommandLine)" -match $escaped
        } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.ProcessId -Force
            }
            catch {
            }
        }
    Start-Sleep -Seconds 1
}

function New-LauncherShortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetCmd,
        [string]$IconPath
    )
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetCmd
    $shortcut.Arguments = ""
    $shortcut.WorkingDirectory = Split-Path -Parent $TargetCmd
    if (Test-Path -LiteralPath $IconPath) {
        $shortcut.IconLocation = "$IconPath,0"
    }
    $shortcut.Save()
}

Assert-Path $InstallDir
Assert-Path (Join-Path $InstallDir "app")
Assert-Path $Backend
Assert-Path $StartPs1
Assert-Path $StartCmd

Stop-InstalledProcess -InstallDir $InstallDir

Copy-Item -LiteralPath $Backend -Destination (Join-Path $InstallDir "app\llama_vulkan_api_server.py") -Force
Copy-Item -LiteralPath $StartPs1 -Destination (Join-Path $InstallDir "Start-iGPU-Qwen3VL.ps1") -Force
Copy-Item -LiteralPath $StartCmd -Destination (Join-Path $InstallDir "Start-iGPU-Qwen3VL.cmd") -Force

$Launcher = Join-Path $InstallDir "Start-iGPU-Qwen3VL.cmd"
$Icon = Join-Path $InstallDir "app\overlay-chat.exe"
$Desktop = [Environment]::GetFolderPath("Desktop")
New-LauncherShortcut `
    -ShortcutPath (Join-Path $Desktop "iGPU Qwen3VL.lnk") `
    -TargetCmd $Launcher `
    -IconPath $Icon

$StartMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "iGPU Qwen3VL"
New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null
New-LauncherShortcut `
    -ShortcutPath (Join-Path $StartMenu "iGPU Qwen3VL.lnk") `
    -TargetCmd $Launcher `
    -IconPath $Icon

Write-Host "Hotfix applied to: $InstallDir"
Write-Host "Launch with: $Launcher"
