param(
    [string]$AppRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [switch]$ConfirmDownload,
    [string[]]$StarterPack = @(),
    [switch]$AcceptAnimaLicense,
    [switch]$PlanOnly,
    [switch]$Json
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Maintenance.psm1") -Force
Import-Module (Join-Path $PSScriptRoot "Asteria.Launcher.psm1") -Force
$resolvedAppRoot = [System.IO.Path]::GetFullPath($AppRoot)
$maintenanceJournalPath = [System.IO.Path]::Combine("$resolvedAppRoot.maintenance", "journal.json")
if (Test-Path -LiteralPath $maintenanceJournalPath -PathType Leaf) {
    try {
        $maintenanceRecovery = Repair-AsteriaMaintenanceTransaction -JournalPath $maintenanceJournalPath -ExpectedAppRoot $resolvedAppRoot
    }
    catch {
        if ($Json) {
            Write-Output (@{ status = "blocked"; code = "launcher_maintenance_recovery_failed"; exit_code = 75 } | ConvertTo-Json -Compress)
        }
        else {
            Write-Output "[Asteria] The previous update could not be recovered automatically. Keep the application folders unchanged and check the maintenance record."
        }
        exit 75
    }
    if ($maintenanceRecovery.status -eq "recovery_required") {
        if ($Json) {
            Write-Output (@{ status = "blocked"; code = "launcher_maintenance_recovery_required"; exit_code = 75 } | ConvertTo-Json -Compress)
        }
        else {
            Write-Output "[Asteria] The previous update requires manual recovery."
        }
        exit 75
    }
}
$releaseSignaturePath = [System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.sig.json")
$trustedKeysPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-public-keys.json")
$releaseManifestPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.yaml")
$hasSigningInput = (Test-Path -LiteralPath $releaseSignaturePath -PathType Leaf) -or (Test-Path -LiteralPath $trustedKeysPath -PathType Leaf)
Import-Module (Join-Path $PSScriptRoot "Asteria.Bootstrap.psm1") -Force
if ($hasSigningInput) {
    $exitCode = Invoke-AsteriaBootstrap -AppRoot $AppRoot -Mode Repair -ConfirmDownload:$ConfirmDownload -StarterPack $StarterPack -AcceptAnimaLicense:$AcceptAnimaLicense -PlanOnly:$PlanOnly -Json:$Json -OfficialManifestPath $releaseManifestPath -OfficialSignaturePath $releaseSignaturePath -OfficialTrustedKeysPath $trustedKeysPath
}
else {
    $exitCode = Invoke-AsteriaBootstrap -AppRoot $AppRoot -Mode Repair -ConfirmDownload:$ConfirmDownload -StarterPack $StarterPack -AcceptAnimaLicense:$AcceptAnimaLicense -PlanOnly:$PlanOnly -Json:$Json
}
exit $exitCode
