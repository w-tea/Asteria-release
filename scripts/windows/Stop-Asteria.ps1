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

try { $state = Read-AsteriaLauncherState -Path $resolvedStatePath }
catch { Complete-Stop "error" "launcher_state_invalid" 74 "The launcher ownership record is invalid and was left unchanged." }
if ($null -eq $state) { Complete-Stop "stopped" "launcher_already_stopped" 0 "Asteria has no launcher-owned processes to stop." }

$stateBelongsToRoot = ([System.IO.Path]::GetFullPath([string]$state.app_root)).Equals($resolvedAppRoot, [System.StringComparison]::OrdinalIgnoreCase)
if (-not $stateBelongsToRoot) {
    $runningOrMismatched = @($state.processes | Where-Object { (Get-AsteriaOwnedProcessState -Record $_) -ne "not-running" })
    if ($runningOrMismatched.Count -gt 0) {
        Complete-Stop "blocked" "launcher_ownership_mismatch" 75 "The launcher record belongs to another application folder. No process was stopped."
    }
    Remove-Item -LiteralPath $resolvedStatePath -Force
    Complete-Stop "stopped" "launcher_stale_state_removed" 0 "A stopped launcher record from the previous folder was removed."
}

$mismatches = @()
$records = @($state.processes)
[array]::Reverse($records)
foreach ($record in $records) {
    $ownership = Get-AsteriaOwnedProcessState -Record $record
    if ($ownership -eq "not-running") { continue }
    if ($ownership -ne "owned") { $mismatches += [string]$record.name; continue }
    if (-not (Stop-AsteriaOwnedProcess -Record $record -TimeoutSeconds 10)) { $mismatches += [string]$record.name }
}
if ($mismatches.Count -gt 0) {
    Complete-Stop "blocked" "launcher_ownership_mismatch" 75 "One or more process records no longer match. Unmatched processes were not stopped."
}
Remove-Item -LiteralPath $resolvedStatePath -Force
Complete-Stop "stopped" "launcher_stopped" 0 "Asteria launcher-owned processes were stopped."
