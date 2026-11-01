Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:AsteriaMaintenanceModulePath = $PSCommandPath

function Get-AsteriaMaintenanceFullPath {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not [System.IO.Path]::IsPathRooted($Path)) {
        throw "maintenance_path_not_absolute"
    }
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function Test-AsteriaMaintenancePathContainedBy {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Candidate
    )

    $resolvedRoot = (Get-AsteriaMaintenanceFullPath $Root) + [System.IO.Path]::DirectorySeparatorChar
    $resolvedCandidate = Get-AsteriaMaintenanceFullPath $Candidate
    return $resolvedCandidate.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)
}

function Write-AsteriaMaintenanceJournal {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Journal
    )

    $resolved = Get-AsteriaMaintenanceFullPath $Path
    $parent = [System.IO.Path]::GetDirectoryName($resolved)
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = [System.IO.Path]::Combine($parent, ".$([System.IO.Path]::GetFileName($resolved)).$([Guid]::NewGuid().ToString('N')).tmp")
    $backup = "$temporary.bak"
    try {
        $Journal | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
        if ([System.IO.File]::Exists($resolved)) {
            [System.IO.File]::Replace($temporary, $resolved, $backup, $true)
        }
        else {
            [System.IO.File]::Move($temporary, $resolved)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
        if (Test-Path -LiteralPath $backup -PathType Leaf) {
            Remove-Item -LiteralPath $backup -Force
        }
    }
}

function Read-AsteriaMaintenanceJournal {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        $journal = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch {
        throw "maintenance_journal_invalid"
    }
    if ($journal.version -ne 1 -or -not $journal.operation_id -or -not $journal.phase -or -not $journal.paths) {
        throw "maintenance_journal_invalid"
    }
    return $journal
}

function Copy-AsteriaMaintenanceTree {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $resolvedSource = Get-AsteriaMaintenanceFullPath $Source
    $resolvedDestination = Get-AsteriaMaintenanceFullPath $Destination
    $sourceItem = Get-Item -LiteralPath $resolvedSource -Force
    if (-not $sourceItem.PSIsContainer -or ($sourceItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "maintenance_source_invalid"
    }
    [System.IO.Directory]::CreateDirectory($resolvedDestination) | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $resolvedSource -Force) {
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "maintenance_reparse_refused"
        }
        $target = [System.IO.Path]::Combine($resolvedDestination, $item.Name)
        if ($item.PSIsContainer) {
            Copy-AsteriaMaintenanceTree -Source $item.FullName -Destination $target
        }
        else {
            [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($target)) | Out-Null
            Copy-Item -LiteralPath $item.FullName -Destination $target -Force
        }
    }
}

function Remove-AsteriaMaintenanceTree {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$AllowedExactPaths
    )

    $resolved = Get-AsteriaMaintenanceFullPath $Path
    $allowed = @($AllowedExactPaths | ForEach-Object { Get-AsteriaMaintenanceFullPath $_ })
    if ($allowed -notcontains $resolved) {
        throw "maintenance_delete_refused"
    }
    if (Test-Path -LiteralPath $resolved) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

function Test-AsteriaMaintenancePair {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$DataRoot
    )

    return (
        (Test-Path -LiteralPath $AppRoot -PathType Container) -and
        (Test-Path -LiteralPath ([System.IO.Path]::Combine($AppRoot, "release-layout.yaml")) -PathType Leaf) -and
        (Test-Path -LiteralPath ([System.IO.Path]::Combine($AppRoot, "app", "backend", "app", "main.py")) -PathType Leaf) -and
        (Test-Path -LiteralPath $DataRoot -PathType Container)
    )
}

function Test-AsteriaOfficialManifest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$SignaturePath,
        [Parameter(Mandatory = $true)][string]$TrustedKeysPath,
        [string]$CurrentDataApplicationVersion = "",
        [switch]$CoordinatedDataRestore
    )

    foreach ($file in @($PythonPath, $ManifestPath, $SignaturePath, $TrustedKeysPath)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "official_manifest_verification_input_missing"
        }
    }
    if (-not (Test-Path -LiteralPath $BackendRoot -PathType Container)) {
        throw "official_manifest_verification_input_missing"
    }
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $BackendRoot
        $arguments = @(
            "-m", "app.cli.maintenance", "verify-release",
            "--manifest", $ManifestPath,
            "--signature", $SignaturePath,
            "--trusted-keys", $TrustedKeysPath
        )
        if ($CurrentDataApplicationVersion) {
            $arguments += @("--current-data-application-version", $CurrentDataApplicationVersion)
        }
        if ($CoordinatedDataRestore) {
            $arguments += "--coordinated-data-restore"
        }
        $output = @(& $PythonPath @arguments 2>$null)
        if ($LASTEXITCODE -ne 0) {
            throw "official_manifest_signature_invalid"
        }
        $result = ($output | Select-Object -Last 1) | ConvertFrom-Json
        if ($result.status -ne "verified") {
            throw "official_manifest_signature_invalid"
        }
        return $result
    }
    finally {
        $env:PYTHONPATH = $previousPythonPath
    }
}

function Set-AsteriaMaintenancePhase {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$JournalPath,
        [Parameter(Mandatory = $true)]$Journal,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$FaultAfterPhase = ""
    )

    $Journal.phase = $Phase
    $Journal.status = $Status
    $Journal.updated_at_utc = [DateTime]::UtcNow.ToString("o")
    Write-AsteriaMaintenanceJournal -Path $JournalPath -Journal $Journal
    if ($FaultAfterPhase -and $FaultAfterPhase -eq $Phase) {
        throw "maintenance_fault_injected_$Phase"
    }
}

function Stop-AsteriaMaintenanceOwnedRuntime {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$LauncherStatePath,
        [Parameter(Mandatory = $true)][string]$LauncherModulePath
    )

    $resolvedApp = Get-AsteriaMaintenanceFullPath $AppRoot
    $resolvedState = Get-AsteriaMaintenanceFullPath $LauncherStatePath
    $resolvedModule = Get-AsteriaMaintenanceFullPath $LauncherModulePath
    if (-not (Test-Path -LiteralPath $resolvedModule -PathType Leaf)) {
        throw "maintenance_launcher_module_missing"
    }
    Import-Module $resolvedModule -Force
    try {
        $state = Read-AsteriaLauncherState -Path $resolvedState
    }
    catch {
        throw "maintenance_launcher_state_invalid"
    }
    if ($null -eq $state) {
        return 0
    }
    if (-not $state.app_root -or $null -eq $state.PSObject.Properties["processes"]) {
        throw "maintenance_launcher_state_invalid"
    }
    $stateApp = Get-AsteriaMaintenanceFullPath ([string]$state.app_root)
    if (-not $stateApp.Equals($resolvedApp, [StringComparison]::OrdinalIgnoreCase)) {
        throw "maintenance_launcher_ownership_mismatch"
    }

    $stopped = 0
    $records = @($state.processes)
    [array]::Reverse($records)
    foreach ($record in $records) {
        $ownership = Get-AsteriaOwnedProcessState -Record $record
        if ($ownership -eq "mismatch") {
            throw "maintenance_launcher_ownership_mismatch"
        }
        if ($ownership -eq "owned") {
            if (-not (Stop-AsteriaOwnedProcess -Record $record -TimeoutSeconds 10)) {
                throw "maintenance_launcher_stop_failed"
            }
            $stopped += 1
        }
    }
    if (@($state.processes | Where-Object { (Get-AsteriaOwnedProcessState -Record $_) -eq "owned" }).Count -gt 0) {
        throw "maintenance_launcher_stop_failed"
    }
    if (Test-Path -LiteralPath $resolvedState -PathType Leaf) {
        Remove-Item -LiteralPath $resolvedState -Force
    }
    return $stopped
}

function Invoke-AsteriaMaintenanceTransaction {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$DataRoot,
        [Parameter(Mandatory = $true)][string]$StagedAppRoot,
        [Parameter(Mandatory = $true)][string]$StagedDataRoot,
        [Parameter(Mandatory = $true)][string]$TransactionRoot,
        [string]$LauncherStatePath = "",
        [string]$LauncherModulePath = (Join-Path $PSScriptRoot "Asteria.Launcher.psm1"),
        [ValidateSet("", "ownership_acquired", "backup_complete", "application_staged", "data_staged", "application_committed", "data_committed", "readiness_verified")]
        [string]$FaultAfterPhase = ""
    )

    $app = Get-AsteriaMaintenanceFullPath $AppRoot
    $data = Get-AsteriaMaintenanceFullPath $DataRoot
    $stagedApp = Get-AsteriaMaintenanceFullPath $StagedAppRoot
    $stagedData = Get-AsteriaMaintenanceFullPath $StagedDataRoot
    $transaction = Get-AsteriaMaintenanceFullPath $TransactionRoot
    foreach ($root in @($app, $data, $stagedApp, $stagedData)) {
        if ((Test-AsteriaMaintenancePathContainedBy -Root $root -Candidate $transaction) -or
            (Test-AsteriaMaintenancePathContainedBy -Root $transaction -Candidate $root)) {
            throw "maintenance_transaction_root_unsafe"
        }
    }
    if (-not (Test-AsteriaMaintenancePair -AppRoot $app -DataRoot $data) -or
        -not (Test-AsteriaMaintenancePair -AppRoot $stagedApp -DataRoot $stagedData)) {
        throw "maintenance_pair_invalid"
    }
    [System.IO.Directory]::CreateDirectory($transaction) | Out-Null
    $lockPath = [System.IO.Path]::Combine($transaction, "maintenance.lock")
    try {
        $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    }
    catch {
        throw "maintenance_busy"
    }
    $journalPath = [System.IO.Path]::Combine($transaction, "journal.json")
    $backupApp = [System.IO.Path]::Combine($transaction, "backup-app")
    $backupData = [System.IO.Path]::Combine($transaction, "backup-data")
    $candidateApp = [System.IO.Path]::Combine($transaction, "candidate-app")
    $candidateData = [System.IO.Path]::Combine($transaction, "candidate-data")
    $retiredApp = [System.IO.Path]::Combine($transaction, "retired-app")
    $retiredData = [System.IO.Path]::Combine($transaction, "retired-data")
    $ownerRoot = [System.IO.Path]::Combine($transaction, "owner")
    $journal = [ordered]@{
        version = 1
        operation_id = [Guid]::NewGuid().ToString("N")
        phase = "created"
        status = "running"
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        paths = [ordered]@{
            app_root = $app
            data_root = $data
            transaction_root = $transaction
            backup_app = $backupApp
            backup_data = $backupData
            candidate_app = $candidateApp
            candidate_data = $candidateData
            retired_app = $retiredApp
            retired_data = $retiredData
        }
    }
    try {
        [System.IO.Directory]::CreateDirectory($ownerRoot) | Out-Null
        Copy-Item -LiteralPath $script:AsteriaMaintenanceModulePath -Destination ([System.IO.Path]::Combine($ownerRoot, "Asteria.Maintenance.psm1")) -Force
        if ($LauncherStatePath) {
            Stop-AsteriaMaintenanceOwnedRuntime -AppRoot $app -LauncherStatePath $LauncherStatePath -LauncherModulePath $LauncherModulePath | Out-Null
        }
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "ownership_acquired" -Status "running" -FaultAfterPhase $FaultAfterPhase

        Copy-AsteriaMaintenanceTree -Source $app -Destination $backupApp
        Copy-AsteriaMaintenanceTree -Source $data -Destination $backupData
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "backup_complete" -Status "running" -FaultAfterPhase $FaultAfterPhase

        Copy-AsteriaMaintenanceTree -Source $stagedApp -Destination $candidateApp
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "application_staged" -Status "running" -FaultAfterPhase $FaultAfterPhase

        Copy-AsteriaMaintenanceTree -Source $stagedData -Destination $candidateData
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "data_staged" -Status "running" -FaultAfterPhase $FaultAfterPhase

        Move-Item -LiteralPath $app -Destination $retiredApp
        Move-Item -LiteralPath $candidateApp -Destination $app
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "application_committed" -Status "running" -FaultAfterPhase $FaultAfterPhase

        Move-Item -LiteralPath $data -Destination $retiredData
        Move-Item -LiteralPath $candidateData -Destination $data
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "data_committed" -Status "running" -FaultAfterPhase $FaultAfterPhase

        if (-not (Test-AsteriaMaintenancePair -AppRoot $app -DataRoot $data)) {
            throw "maintenance_readiness_failed"
        }
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "readiness_verified" -Status "running" -FaultAfterPhase $FaultAfterPhase

        foreach ($retired in @($retiredApp, $retiredData)) {
            Remove-AsteriaMaintenanceTree -Path $retired -AllowedExactPaths @($retiredApp, $retiredData)
        }
        Set-AsteriaMaintenancePhase -JournalPath $journalPath -Journal $journal -Phase "completed" -Status "completed"
        return [pscustomobject][ordered]@{ status = "completed"; journal_path = $journalPath }
    }
    finally {
        if ($null -ne $lock) {
            $lock.Dispose()
        }
    }
}

function Restore-AsteriaMaintenancePair {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Journal)

    $paths = $Journal.paths
    $allowed = @($paths.app_root, $paths.data_root, $paths.retired_app, $paths.retired_data)
    foreach ($current in @($paths.app_root, $paths.data_root)) {
        Remove-AsteriaMaintenanceTree -Path $current -AllowedExactPaths $allowed
    }
    Copy-AsteriaMaintenanceTree -Source $paths.backup_app -Destination $paths.app_root
    Copy-AsteriaMaintenanceTree -Source $paths.backup_data -Destination $paths.data_root
    foreach ($retired in @($paths.retired_app, $paths.retired_data)) {
        Remove-AsteriaMaintenanceTree -Path $retired -AllowedExactPaths $allowed
    }
}

function Repair-AsteriaMaintenanceTransaction {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$JournalPath,
        [Parameter(Mandatory = $true)][string]$ExpectedAppRoot
    )

    $journal = Read-AsteriaMaintenanceJournal -Path $JournalPath
    if ($null -eq $journal) {
        return [pscustomobject][ordered]@{ status = "no_action" }
    }
    $expected = Get-AsteriaMaintenanceFullPath $ExpectedAppRoot
    $recorded = Get-AsteriaMaintenanceFullPath $journal.paths.app_root
    if (-not $expected.Equals($recorded, [StringComparison]::OrdinalIgnoreCase)) {
        throw "maintenance_journal_ownership_mismatch"
    }
    if ($journal.status -eq "completed") {
        return [pscustomobject][ordered]@{ status = $(if ($journal.phase -eq "rolled_back") { "rolled_back" } else { "completed" }) }
    }
    $paths = $journal.paths
    $newPairCommitted = $journal.phase -in @("data_committed", "readiness_verified")
    if ($newPairCommitted -and (Test-AsteriaMaintenancePair -AppRoot $paths.app_root -DataRoot $paths.data_root)) {
        Set-AsteriaMaintenancePhase -JournalPath $JournalPath -Journal $journal -Phase "completed" -Status "completed"
        return [pscustomobject][ordered]@{ status = "completed" }
    }

    $beforeApplicationCommit = $journal.phase -in @("ownership_acquired", "backup_complete", "application_staged", "data_staged")
    if ($beforeApplicationCommit -and (Test-AsteriaMaintenancePair -AppRoot $paths.app_root -DataRoot $paths.data_root)) {
        Set-AsteriaMaintenancePhase -JournalPath $JournalPath -Journal $journal -Phase "rolled_back" -Status "completed"
        return [pscustomobject][ordered]@{ status = "rolled_back" }
    }

    if (Test-AsteriaMaintenancePair -AppRoot $paths.backup_app -DataRoot $paths.backup_data) {
        Restore-AsteriaMaintenancePair -Journal $journal
        Set-AsteriaMaintenancePhase -JournalPath $JournalPath -Journal $journal -Phase "rolled_back" -Status "completed"
        return [pscustomobject][ordered]@{ status = "rolled_back" }
    }

    Set-AsteriaMaintenancePhase -JournalPath $JournalPath -Journal $journal -Phase "recovery_required" -Status "recovery_required"
    return [pscustomobject][ordered]@{ status = "recovery_required" }
}

Export-ModuleMember -Function @(
    "Invoke-AsteriaMaintenanceTransaction",
    "Repair-AsteriaMaintenanceTransaction",
    "Read-AsteriaMaintenanceJournal",
    "Stop-AsteriaMaintenanceOwnedRuntime",
    "Test-AsteriaOfficialManifest",
    "Test-AsteriaMaintenancePair",
    "Write-AsteriaMaintenanceJournal"
)
