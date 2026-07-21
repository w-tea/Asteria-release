[CmdletBinding()]
param(
    [string]$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$TempRoot = "",
    [string]$StatePath = "",
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Launcher.psm1") -Force

function Complete-Stop {
    param([string]$Status, [string]$Code, [int]$ExitCode, [string]$Message)
    $payload = [ordered]@{ status = $Status; code = $Code; exit_code = $ExitCode; message = $Message }
    if ($Json) { Write-Output ($payload | ConvertTo-Json -Compress) } else { Write-Output "[Asteria] $Message" }
    exit $ExitCode
}

if (-not (Test-AsteriaFullyQualifiedPath $AppRoot)) { Complete-Stop "invalid" "launcher_path_not_absolute" 64 "The application folder must be absolute." }
foreach ($explicitPath in @($TempRoot, $StatePath)) {
    if ($explicitPath -and -not (Test-AsteriaFullyQualifiedPath $explicitPath)) { Complete-Stop "invalid" "launcher_path_not_absolute" 64 "Every explicit launcher path must be absolute." }
}
$resolvedAppRoot = [System.IO.Path]::GetFullPath($AppRoot)
if ($TempRoot) {
    $resolvedTempRoot = [System.IO.Path]::GetFullPath($TempRoot)
}
else {
    $layoutPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-layout.yaml")
    try {
        $resolvedTempRoot = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "writable_roots" "temp")
    }
    catch {
        # Stop and repair remain usable when the packaged layout itself is damaged.
        $resolvedTempRoot = [System.IO.Path]::Combine($resolvedAppRoot, "temp")
    }
}
$processRoot = [System.IO.Path]::Combine($resolvedTempRoot, "launcher")
$resolvedStatePath = if ($StatePath) { [System.IO.Path]::GetFullPath($StatePath) } else { [System.IO.Path]::Combine($processRoot, "launcher-state.json") }
$comfyPidPath = [System.IO.Path]::Combine($processRoot, "comfyui.pid")

function Remove-AsteriaLauncherRecords {
    foreach ($path in @($resolvedStatePath, $comfyPidPath)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path -Force }
    }
}

try { $state = Read-AsteriaLauncherState -Path $resolvedStatePath }
catch { Complete-Stop "error" "launcher_state_invalid" 74 "The launcher ownership record is invalid and was left unchanged." }
if ($null -eq $state) {
    Remove-AsteriaLauncherRecords
    Complete-Stop "stopped" "launcher_already_stopped" 0 "Asteria has no launcher-owned processes to stop."
}

$stateBelongsToRoot = ([System.IO.Path]::GetFullPath([string]$state.app_root)).Equals($resolvedAppRoot, [System.StringComparison]::OrdinalIgnoreCase)
if (-not $stateBelongsToRoot) {
    $owned = @($state.processes | Where-Object { (Get-AsteriaOwnedProcessState -Record $_) -eq "owned" })
    if ($owned.Count -gt 0) {
        Complete-Stop "blocked" "launcher_ownership_mismatch" 75 "The launcher record belongs to another application folder. No process was stopped."
    }
    Remove-AsteriaLauncherRecords
    Complete-Stop "stopped" "launcher_stale_state_removed" 0 "A stopped launcher record from the previous folder was removed."
}

$mismatches = @()
$staleRecordFound = $false
$records = @($state.processes)
[array]::Reverse($records)
foreach ($record in $records) {
    $ownership = Get-AsteriaOwnedProcessState -Record $record
    if ($ownership -ne "owned") {
        # A reused PID is not the recorded Asteria process. Leave that process alone,
        # but do not retain a launcher record that can no longer identify its owner.
        $staleRecordFound = $true
        continue
    }
    if (-not (Stop-AsteriaOwnedProcess -Record $record -TimeoutSeconds 10)) {
        $finalOwnership = Get-AsteriaOwnedProcessState -Record $record
        if ($finalOwnership -eq "owned") { $mismatches += [string]$record.name }
        else { $staleRecordFound = $true }
    }
}
if ($mismatches.Count -gt 0) {
    Complete-Stop "blocked" "launcher_ownership_mismatch" 75 "One or more process records no longer match. Unmatched processes were not stopped."
}
Remove-AsteriaLauncherRecords
if ($staleRecordFound) {
    Complete-Stop "stopped" "launcher_stale_state_removed" 0 "Stopped launcher records were removed."
}
Complete-Stop "stopped" "launcher_stopped" 0 "Asteria launcher-owned processes were stopped."
