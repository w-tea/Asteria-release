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
    $maintenanceRecovery = Repair-AsteriaMaintenanceTransaction -JournalPath $maintenanceJournalPath
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
if ((Test-Path -LiteralPath $releaseSignaturePath -PathType Leaf) -or (Test-Path -LiteralPath $trustedKeysPath -PathType Leaf)) {
    try {
        $layoutPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-layout.yaml")
        $backendRoot = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "backend_root")
        $pythonPath = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "python")
        Test-AsteriaOfficialManifest -PythonPath $pythonPath -BackendRoot $backendRoot -ManifestPath ([System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.yaml")) -SignaturePath $releaseSignaturePath -TrustedKeysPath $trustedKeysPath | Out-Null
    }
    catch {
        if ($Json) {
            Write-Output (@{ status = "blocked"; code = "launcher_release_signature_invalid"; exit_code = 76 } | ConvertTo-Json -Compress)
        }
        else {
            Write-Output "[Asteria] The official release signature is missing or invalid."
        }
        exit 76
    }
}
Import-Module (Join-Path $PSScriptRoot "Asteria.Bootstrap.psm1") -Force
$exitCode = Invoke-AsteriaBootstrap -AppRoot $AppRoot -Mode Repair -ConfirmDownload:$ConfirmDownload -StarterPack $StarterPack -AcceptAnimaLicense:$AcceptAnimaLicense -PlanOnly:$PlanOnly -Json:$Json
exit $exitCode
