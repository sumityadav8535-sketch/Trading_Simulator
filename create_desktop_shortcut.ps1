# Creates a Desktop shortcut that runs start_app.bat
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$batPath = Join-Path $projectRoot "start_app.bat"
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Trading Simulator.lnk"
$pythonExe = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $batPath)) {
    throw "start_app.bat not found at $batPath"
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $batPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle = 1
$shortcut.Description = "Start Trading Simulator and open in browser"

# Prefer Python icon if venv exists; otherwise shell default
if (Test-Path $pythonExe) {
    $shortcut.IconLocation = "$pythonExe,0"
} else {
    $shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,14"
}

$shortcut.Save()
Write-Host "Desktop shortcut created:"
Write-Host "  $shortcutPath"
Write-Host ""
Write-Host "Double-click 'Trading Simulator' on your Desktop to open the app."
