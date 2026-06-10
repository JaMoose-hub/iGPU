param(
    [string]$InstallDir = $PSScriptRoot,
    [switch]$KeepData
)

$ErrorActionPreference = "Stop"

$stopScript = Join-Path $InstallDir "Stop-GameCompanion-HybridDev.ps1"
if (Test-Path -LiteralPath $stopScript) {
    & $stopScript -Root $InstallDir
}

$desktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "Game Companion Hybrid Dev.lnk"
$startMenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) "Game Companion Hybrid Dev"
Remove-Item -LiteralPath $desktopShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $startMenuDir -Recurse -Force -ErrorAction SilentlyContinue

if ($KeepData) {
    $backupDir = Join-Path ([Environment]::GetFolderPath("Desktop")) ("GameCompanion-Hybrid-Dev-data-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    foreach ($name in @("data", "config")) {
        $source = Join-Path $InstallDir $name
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $backupDir $name) -Recurse -Force
        }
    }
    Write-Host "Data/config backup created at: $backupDir"
}

Remove-Item -LiteralPath $InstallDir -Recurse -Force
Write-Host "Uninstalled Game Companion Hybrid Dev from: $InstallDir"
