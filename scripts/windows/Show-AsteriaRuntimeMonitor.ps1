[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Launcher.psm1") -Force
$Host.UI.RawUI.WindowTitle = "Asteria Runtime Monitor"

Write-Host "Asteria Runtime Monitor"
Write-Host "Generation progress and runtime messages will appear below."
Write-Host "Closing this window does not stop Asteria."
Write-Host ""

while (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
    Start-Sleep -Milliseconds 250
}

Get-Content -LiteralPath $LogPath -Encoding utf8 -Tail 30 -Wait |
    ForEach-Object { ConvertTo-AsteriaRuntimeMonitorLine -Line $_ }
