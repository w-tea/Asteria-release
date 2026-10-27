[CmdletBinding()]
param(
    [string]$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [ValidateSet("development", "packaged")][string]$TargetMode = "packaged",
    [string]$ProfileId = "sdxl",
    [string[]]$Feature = @(),
    [string]$PythonPath = "",
    [string]$NodePath = "",
    [string]$DataRoot = "",
    [string]$ModelRoot = "",
    [string]$LogsRoot = "",
    [string]$TempRoot = "",
    [string]$ComfyRoot = "",
    [ValidateRange(1, 600)][int]$HealthTimeoutSeconds = 120,
    [switch]$SafeMode,
    [switch]$NoOpen,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Launcher.psm1") -Force
Import-Module (Join-Path $PSScriptRoot "Asteria.Maintenance.psm1") -Force

function Complete-Start {
    param([string]$Status, [string]$Code, [int]$ExitCode, [string]$Message)
    $payload = [ordered]@{ status = $Status; code = $Code; exit_code = $ExitCode; message = $Message }
    if ($Json) { Write-Output ($payload | ConvertTo-Json -Compress) } else { Write-Output "[Asteria] $Message" }
    exit $ExitCode
}

function Write-StartProgress {
    param([string]$Message)
    if (-not $Json) {
        Write-Output "[Asteria] $Message"
    }
}

function Test-SamePath {
    param([string]$First, [string]$Second)
    return ([System.IO.Path]::GetFullPath($First)).Equals([System.IO.Path]::GetFullPath($Second), [StringComparison]::OrdinalIgnoreCase)
}

if (-not (Test-AsteriaFullyQualifiedPath $AppRoot)) { Complete-Start "invalid" "launcher_path_not_absolute" 64 "The application folder must be absolute." }
foreach ($explicitPath in @($PythonPath, $NodePath, $DataRoot, $ModelRoot, $LogsRoot, $TempRoot, $ComfyRoot)) {
    if ($explicitPath -and -not (Test-AsteriaFullyQualifiedPath $explicitPath)) { Complete-Start "invalid" "launcher_path_not_absolute" 64 "Every explicit launcher path must be absolute." }
}
$resolvedAppRoot = [System.IO.Path]::GetFullPath($AppRoot)
$maintenanceJournalPath = [System.IO.Path]::Combine("$resolvedAppRoot.maintenance", "journal.json")
if (Test-Path -LiteralPath $maintenanceJournalPath -PathType Leaf) {
    try {
        $maintenanceRecovery = Repair-AsteriaMaintenanceTransaction -JournalPath $maintenanceJournalPath -ExpectedAppRoot $resolvedAppRoot
        if ($maintenanceRecovery.status -eq "recovery_required") {
            Complete-Start "blocked" "launcher_maintenance_recovery_required" 75 "The previous update could not be recovered automatically. Run Repair-Asteria.cmd."
        }
    }
    catch {
        Complete-Start "blocked" "launcher_maintenance_recovery_failed" 75 "The previous update could not be recovered automatically. Run Repair-Asteria.cmd."
    }
}
$layoutPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-layout.yaml")
try {
    $backendRoot = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "backend_root")
    $frontendEntry = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "frontend_entry")
    $packagePython = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "python")
    $packageNode = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "node")
    $packageComfy = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "package" "comfy_root")
    $packageData = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "writable_roots" "data")
    $packageModels = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "writable_roots" "models")
    $packageLogs = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "writable_roots" "logs")
    $packageTemp = Resolve-AsteriaPackagePath $resolvedAppRoot (Get-AsteriaReleaseLayoutValue $layoutPath "writable_roots" "temp")
}
catch { Complete-Start "blocked" "launcher_layout_invalid" 67 "The release layout is missing or invalid." }

if ($TargetMode -eq "development") {
    $backendRoot = [System.IO.Path]::Combine($resolvedAppRoot, "backend")
    $frontendEntry = [System.IO.Path]::Combine($resolvedAppRoot, "frontend", ".output", "server", "index.mjs")
}

$resolvedPython = if ($PythonPath) { [System.IO.Path]::GetFullPath($PythonPath) } elseif ($TargetMode -eq "packaged") { $packagePython } else { [System.IO.Path]::Combine($resolvedAppRoot, ".venv", "Scripts", "python.exe") }
if ($NodePath) {
    $resolvedNode = [System.IO.Path]::GetFullPath($NodePath)
}
elseif ($TargetMode -eq "packaged") {
    $resolvedNode = $packageNode
}
else {
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    $resolvedNode = if ($null -ne $nodeCommand) { $nodeCommand.Source } else { "" }
}
$resolvedDataRoot = if ($DataRoot) { [System.IO.Path]::GetFullPath($DataRoot) } else { $packageData }
$resolvedModelRoot = if ($ModelRoot) { [System.IO.Path]::GetFullPath($ModelRoot) } else { $packageModels }
$defaultOutputRoot = [System.IO.Path]::Combine($resolvedDataRoot, "outputs")
$resolvedLogsRoot = if ($LogsRoot) { [System.IO.Path]::GetFullPath($LogsRoot) } else { $packageLogs }
$resolvedTempRoot = if ($TempRoot) { [System.IO.Path]::GetFullPath($TempRoot) } else { $packageTemp }
$resolvedComfyRoot = if ($ComfyRoot) { [System.IO.Path]::GetFullPath($ComfyRoot) } else { $packageComfy }

if ($TargetMode -eq "packaged") {
    $actualPaths = @($resolvedPython, $resolvedNode, $resolvedComfyRoot, $resolvedDataRoot, $resolvedModelRoot, $resolvedLogsRoot, $resolvedTempRoot)
    $packagePaths = @($packagePython, $packageNode, $packageComfy, $packageData, $packageModels, $packageLogs, $packageTemp)
    for ($index = 0; $index -lt $actualPaths.Count; $index++) {
        if (-not (Test-SamePath $actualPaths[$index] $packagePaths[$index])) {
            Complete-Start "invalid" "launcher_packaged_path_override" 65 "Packaged launches must use paths contained by the application folder."
        }
    }

    $releaseManifestPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.yaml")
    $releaseSignaturePath = [System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.sig.json")
    $trustedKeysPath = [System.IO.Path]::Combine($resolvedAppRoot, "release-public-keys.json")
    $hasSigningInput = (Test-Path -LiteralPath $releaseSignaturePath -PathType Leaf) -or (Test-Path -LiteralPath $trustedKeysPath -PathType Leaf)
    Initialize-AsteriaModelDirectories -ModelRoot $resolvedModelRoot
    if ($hasSigningInput) {
        $installPath = Join-Path $PSScriptRoot "Install-Asteria.ps1"
        & $installPath -AppRoot $resolvedAppRoot -ConfirmDownload -Json:$Json -OfficialManifestPath $releaseManifestPath -OfficialSignaturePath $releaseSignaturePath -OfficialTrustedKeysPath $trustedKeysPath
    }
    else {
        $installPath = Join-Path $PSScriptRoot "Install-Asteria.ps1"
        & $installPath -AppRoot $resolvedAppRoot -ConfirmDownload -Json:$Json
    }
    $installExit = $LASTEXITCODE
    if ($installExit -ne 0) { Complete-Start "blocked" "launcher_install_failed" $installExit "Required setup is incomplete. Run Repair-Asteria.cmd and retry." }
}
else {
    Initialize-AsteriaModelDirectories -ModelRoot $resolvedModelRoot
}

if (-not (Test-Path -LiteralPath $resolvedPython -PathType Leaf) -or
    -not (Test-Path -LiteralPath $resolvedNode -PathType Leaf) -or
    -not (Test-Path -LiteralPath $resolvedComfyRoot -PathType Container) -or
    -not (Test-Path -LiteralPath $backendRoot -PathType Container) -or
    -not (Test-Path -LiteralPath $frontendEntry -PathType Leaf)) {
    Complete-Start "blocked" "launcher_runtime_missing" 67 "A required packaged runtime or application file is missing."
}
foreach ($path in @($resolvedDataRoot, $resolvedModelRoot, $defaultOutputRoot, $resolvedLogsRoot, $resolvedTempRoot)) {
    [System.IO.Directory]::CreateDirectory($path) | Out-Null
}
$env:YOLO_CONFIG_DIR = [System.IO.Path]::Combine($resolvedDataRoot, "config")
[System.IO.Directory]::CreateDirectory($env:YOLO_CONFIG_DIR) | Out-Null

$extensionStartupJournal = [System.IO.Path]::Combine($resolvedDataRoot, "extensions", "startup-journal.json")
if (-not $SafeMode -and (Test-Path -LiteralPath $extensionStartupJournal -PathType Leaf)) {
    try {
        $extensionStartupState = Get-Content -LiteralPath $extensionStartupJournal -Raw | ConvertFrom-Json
        if ($extensionStartupState.status -eq "loading") {
            Write-Warning "The previous extension load did not finish. Retry normally, or run Start-Asteria.cmd --safe-mode if startup fails again."
        }
    }
    catch {
        Write-Warning "Extension startup recovery information is unreadable. Use Start-Asteria.cmd --safe-mode if startup fails."
    }
}

$preflightPath = Join-Path $PSScriptRoot "Test-AsteriaReadiness.ps1"
$enabledFeatures = @($Feature | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$preflightParameters = @{
    AppRoot = $resolvedAppRoot
    PythonPath = $resolvedPython
    TargetMode = $TargetMode
    ProfileId = $ProfileId
    Feature = $enabledFeatures
    DataRoot = $resolvedDataRoot
    ModelRoot = $resolvedModelRoot
    ComfyRoot = $resolvedComfyRoot
    Json = $true
}
$preflightOutput = & $preflightPath @preflightParameters
$preflightExit = $LASTEXITCODE
if ($preflightExit -notin @(0, 12)) { Complete-Start "blocked" "launcher_preflight_blocked" $preflightExit "Startup readiness is blocked. Run Repair-Asteria.cmd and retry." }

$processRoot = [System.IO.Path]::Combine($resolvedTempRoot, "launcher")
$logsRoot = [System.IO.Path]::Combine($resolvedLogsRoot, "launcher")
$statePath = [System.IO.Path]::Combine($processRoot, "launcher-state.json")
$failurePath = [System.IO.Path]::Combine($logsRoot, "launcher-failure.json")
[System.IO.Directory]::CreateDirectory($logsRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($processRoot) | Out-Null
$launcherPreferences = Get-AsteriaLauncherPreferences -SettingsPath ([System.IO.Path]::Combine($resolvedDataRoot, "user-settings.json"))
if ($launcherPreferences.ReasonId) {
    "reason_id=$($launcherPreferences.ReasonId) field=launcher.disable_pinned_memory" | Set-Content -LiteralPath (Join-Path $logsRoot "launcher-memory-settings.log") -Encoding utf8
}
$existing = Read-AsteriaLauncherState -Path $statePath
function Test-AsteriaOptionalProcessRecord {
    param($Record)
    return ($Record.PSObject.Properties.Name -contains "optional") -and [bool]$Record.optional
}
if ($null -ne $existing) {
    $required = @($existing.processes | Where-Object { -not (Test-AsteriaOptionalProcessRecord -Record $_) })
    $ownedRequired = @($required | Where-Object { (Get-AsteriaOwnedProcessState -Record $_) -eq "owned" })
    if ($required.Count -gt 0 -and $ownedRequired.Count -eq $required.Count) {
        Complete-Start "running" "launcher_already_running" 0 "Asteria is already running under this launcher."
    }
    Complete-Start "blocked" "launcher_state_stale" 74 "A stale launcher record must be stopped or repaired before starting."
}
foreach ($port in @(8188, 8000, 3000)) {
    if (Test-AsteriaTcpPortInUse -Port $port) { Complete-Start "blocked" "launcher_port_in_use" 73 "A required local port is already in use. No process was started." }
}

$comfyInput = [System.IO.Path]::Combine($resolvedDataRoot, "comfy-input")
$comfyUser = [System.IO.Path]::Combine($resolvedDataRoot, "comfy-user")
$comfyDefaultUser = [System.IO.Path]::Combine($comfyUser, "default")
$comfyDatabase = [System.IO.Path]::Combine($comfyUser, "comfyui.db")
$comfyDatabaseUrl = "sqlite:///$($comfyDatabase.Replace('\', '/'))"
$comfyOutput = [System.IO.Path]::Combine($resolvedTempRoot, "comfy-output")
foreach ($path in @($comfyInput, $comfyUser, $comfyDefaultUser, $comfyOutput)) { [System.IO.Directory]::CreateDirectory($path) | Out-Null }
$extraModels = [System.IO.Path]::Combine($processRoot, "extra_model_paths.yaml")
$externalModelRegistry = [System.IO.Path]::Combine($resolvedDataRoot, "config", "external-model-libraries.json")
Write-AsteriaExtraModelPaths -Path $extraModels -ModelRoot $resolvedModelRoot -ExternalRegistryPath $externalModelRegistry

$env:PYTHONPATH = $backendRoot
$env:ASTERIA_ROOT_DIR = $resolvedAppRoot
$env:ASTERIA_DATA_DIR = $resolvedDataRoot
$env:ASTERIA_MODEL_DIR = $resolvedModelRoot
$env:ASTERIA_MODEL_SELECTIONS_PATH = [System.IO.Path]::Combine($resolvedDataRoot, "model-selections.json")
$env:ASTERIA_RELEASE_MANIFEST_PATH = [System.IO.Path]::Combine($resolvedAppRoot, "release-manifest.yaml")
$env:ASTERIA_DEFAULTS_DIR = [System.IO.Path]::Combine($resolvedAppRoot, "defaults")
$env:ASTERIA_COMFY_DIR = $resolvedComfyRoot
$env:ASTERIA_COMFY_PYTHON = $resolvedPython
$env:ASTERIA_COMFY_PID_PATH = [System.IO.Path]::Combine($processRoot, "comfyui.pid")
$env:ASTERIA_COMFY_OUTPUT_DIR = $comfyOutput
$env:ASTERIA_AUTO_BOOTSTRAP_MODELS = "0"
$env:ASTERIA_TARGET_MODE = $TargetMode
$env:ASTERIA_DISABLE_EXTENSIONS = if ($SafeMode) { "1" } else { "0" }
$env:NITRO_HOST = "127.0.0.1"
$env:NITRO_PORT = "3000"
$env:NUXT_PUBLIC_API_BASE = "http://127.0.0.1:8000"
$env:NUXT_PUBLIC_TARGET_MODE = $TargetMode

$state = [ordered]@{ version = 2; app_root = $resolvedAppRoot; created_at_utc = [DateTime]::UtcNow.ToString("o"); processes = @() }
function Add-OwnedProcess {
    param($Record)
    $state.processes += $Record
    Write-AsteriaLauncherState -Path $statePath -State $state
}
function Stop-NewProcesses {
    $allStopped = $true
    $records = @($state.processes); [array]::Reverse($records)
    foreach ($record in $records) {
        if ((Get-AsteriaOwnedProcessState -Record $record) -eq "owned") {
            Stop-AsteriaOwnedProcess -Record $record -TimeoutSeconds 5 | Out-Null
        }
        if ((Get-AsteriaOwnedProcessState -Record $record) -ne "not-running") { $allStopped = $false }
    }
    return $allStopped
}

$comfyUrl = "http://127.0.0.1:8188/system_stats"
$backendUrl = "http://127.0.0.1:8000/api/health"
$frontendUrl = "http://127.0.0.1:3000/"
$comfyErrorLog = Join-Path $logsRoot "comfyui.err.log"
$comfyArguments = @("main.py", "--listen", "127.0.0.1", "--port", "8188", "--preview-method", "auto", "--extra-model-paths-config", "`"$extraModels`"", "--input-directory", "`"$comfyInput`"", "--output-directory", "`"$comfyOutput`"", "--user-directory", "`"$comfyUser`"", "--database-url", "`"$comfyDatabaseUrl`"")
if ($launcherPreferences.DisablePinnedMemory -and ($comfyArguments -notcontains "--disable-pinned-memory")) {
    $comfyArguments += "--disable-pinned-memory"
}
$launchStage = "comfy_start"
try {
    Write-StartProgress "Starting ComfyUI..."
    $comfyRecord = Start-AsteriaOwnedProcess -Name "comfyui" -FilePath $resolvedPython -ArgumentList $comfyArguments -WorkingDirectory $resolvedComfyRoot -StandardOutputPath (Join-Path $logsRoot "comfyui.out.log") -StandardErrorPath $comfyErrorLog
    Add-OwnedProcess $comfyRecord
    Set-Content -LiteralPath $env:ASTERIA_COMFY_PID_PATH -Value $comfyRecord.pid -Encoding ascii
    $launchStage = "comfy_health"
    if (-not (Wait-AsteriaHttpReady -Uri $comfyUrl -TimeoutSeconds $HealthTimeoutSeconds)) { throw "ComfyUI health timeout." }
    Write-StartProgress "ComfyUI is ready."

    $launchStage = "backend_start"
    Write-StartProgress "Starting the Asteria backend..."
    $backendRecord = Start-AsteriaOwnedProcess -Name "backend" -FilePath $resolvedPython -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info") -WorkingDirectory $backendRoot -StandardOutputPath (Join-Path $logsRoot "backend.out.log") -StandardErrorPath (Join-Path $logsRoot "backend.err.log")
    Add-OwnedProcess $backendRecord
    $launchStage = "backend_health"
    if (-not (Wait-AsteriaHttpReady -Uri $backendUrl -TimeoutSeconds $HealthTimeoutSeconds)) { throw "Backend health timeout." }
    Write-StartProgress "Asteria backend is ready."

    $launchStage = "frontend_start"
    Write-StartProgress "Starting the Asteria frontend..."
    $frontendRecord = Start-AsteriaOwnedProcess -Name "frontend" -FilePath $resolvedNode -ArgumentList @("`"$frontendEntry`"") -WorkingDirectory ([System.IO.Path]::GetDirectoryName($frontendEntry)) -StandardOutputPath (Join-Path $logsRoot "frontend.out.log") -StandardErrorPath (Join-Path $logsRoot "frontend.err.log")
    Add-OwnedProcess $frontendRecord
    $launchStage = "frontend_health"
    if (-not (Wait-AsteriaHttpReady -Uri $frontendUrl -TimeoutSeconds $HealthTimeoutSeconds)) { throw "Frontend health timeout." }
    Write-StartProgress "Asteria frontend is ready."
}
catch {
    $launchFailure = $_
    $rollbackComplete = Stop-NewProcesses
    [ordered]@{
        occurred_at_utc = [DateTime]::UtcNow.ToString("o")
        launch_stage = $launchStage
        error_type = $launchFailure.Exception.GetType().FullName
        message = $launchFailure.Exception.Message
        rollback_complete = $rollbackComplete
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $failurePath -Encoding utf8
    if ($rollbackComplete) {
        if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath -Force }
        if (Test-Path -LiteralPath $env:ASTERIA_COMFY_PID_PATH) { Remove-Item -LiteralPath $env:ASTERIA_COMFY_PID_PATH -Force }
    }
    else {
        Complete-Start "blocked" "launcher_rollback_incomplete" 75 "A started process could not be safely stopped. Run Stop-Asteria.cmd and review the preserved logs."
    }
    Complete-Start "error" "launcher_health_failed" 76 "Asteria did not become ready in time. Started processes were stopped; review the preserved logs."
}

if ($launcherPreferences.RuntimeMonitorOnStart) {
    try {
        $monitorScript = Join-Path $PSScriptRoot "Show-AsteriaRuntimeMonitor.ps1"
        $monitorPowerShell = (Get-Command powershell.exe -ErrorAction Stop | Select-Object -First 1).Source
        $monitorRecord = Start-AsteriaOwnedProcess -Name "runtime-monitor" -FilePath $monitorPowerShell -ArgumentList @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$monitorScript`"", "-LogPath", "`"$comfyErrorLog`"") -WorkingDirectory $resolvedAppRoot -WindowStyle Normal -Optional
        Add-OwnedProcess $monitorRecord
    }
    catch {
        Write-Warning "Asteria started, but the optional Runtime Monitor could not be opened."
    }
}

if (-not $NoOpen) { Start-Process $frontendUrl }
Complete-Start "running" "launcher_started" 0 "Asteria is ready."
