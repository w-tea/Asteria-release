[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AppRoot,

    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [ValidateSet("development", "packaged")]
    [string]$TargetMode = "development",

    [string]$ProfileId = "sdxl",

    [string[]]$Feature = @(),

    [string]$DataRoot = "",

    [string]$ModelRoot = "",

    [string]$OutputRoot = "",

    [string]$ComfyRoot = "",

    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Launcher.psm1") -Force

function Stop-Preflight {
    param(
        [string]$Status,
        [string]$Code,
        [int]$ExitCode,
        [string]$Message
    )

    $payload = [ordered]@{
        status = $Status
        code = $Code
        exit_code = $ExitCode
        message = $Message
        report = $null
    }
    if ($Json) {
        Write-Output ($payload | ConvertTo-Json -Depth 4 -Compress)
    }
    else {
        Write-Output "Asteria readiness: $Status"
        Write-Output $Message
    }
    exit $ExitCode
}

function Resolve-AbsolutePath {
    param([string]$Value)

    if (-not (Test-AsteriaFullyQualifiedPath $Value)) {
        return $null
    }
    try {
        return [System.IO.Path]::GetFullPath($Value)
    }
    catch {
        return $null
    }
}

function Test-DocumentationLocation {
    param([string]$Value)

    $segments = $Value -split '[\\/]'
    foreach ($segment in $segments) {
        if ($segment.ToLowerInvariant() -in @("docs", "documentation", "reference", "references")) {
            return $true
        }
    }
    return $false
}

function Test-ContainedPath {
    param(
        [string]$Parent,
        [string]$Child
    )

    $separator = [System.IO.Path]::DirectorySeparatorChar
    $parentPrefix = $Parent.TrimEnd([char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)) + $separator
    return $Child.StartsWith($parentPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-WritableDirectory {
    param([string]$Value)

    if (-not [System.IO.Directory]::Exists($Value)) {
        return $false
    }
    $probePath = [System.IO.Path]::Combine($Value, ".asteria-readiness-$([System.Guid]::NewGuid().ToString('N')).tmp")
    $writable = $false
    try {
        $stream = [System.IO.File]::Open($probePath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        $stream.Dispose()
        $writable = $true
    }
    catch {
        $writable = $false
    }
    finally {
        try {
            if ([System.IO.File]::Exists($probePath)) {
                [System.IO.File]::Delete($probePath)
            }
        }
        catch {
            $writable = $false
        }
    }
    return $writable
}

$appRootResolved = Resolve-AbsolutePath $AppRoot
if ($null -eq $appRootResolved) {
    Stop-Preflight "invalid" "startup_path_not_absolute" 64 "AppRoot and every target path must be absolute."
}
if (Test-DocumentationLocation $appRootResolved) {
    Stop-Preflight "invalid" "startup_execution_location_unsafe" 65 "Choose an installed application folder, not a documentation folder."
}
if (-not [System.IO.Directory]::Exists($appRootResolved)) {
    Stop-Preflight "blocked" "startup_app_root_missing" 67 "The application folder is missing."
}

$pythonResolved = Resolve-AbsolutePath $PythonPath
if ($null -eq $pythonResolved) {
    Stop-Preflight "invalid" "startup_path_not_absolute" 64 "AppRoot and every target path must be absolute."
}
if (Test-DocumentationLocation $pythonResolved) {
    Stop-Preflight "invalid" "startup_execution_location_unsafe" 65 "Choose an installed Python runtime, not a documentation folder."
}
if (-not [System.IO.File]::Exists($pythonResolved)) {
    Stop-Preflight "blocked" "startup_python_missing" 66 "The configured Python runtime is missing."
}
if ($TargetMode -eq "packaged" -and -not (Test-ContainedPath $appRootResolved $pythonResolved)) {
    Stop-Preflight "invalid" "startup_execution_location_unsafe" 65 "The packaged Python runtime must be inside the application folder."
}

$frontendEntry = $null
if ($TargetMode -eq "packaged") {
    $layoutPath = [System.IO.Path]::Combine($appRootResolved, "release-layout.yaml")
    try {
        $backendRoot = Resolve-AsteriaPackagePath $appRootResolved (Get-AsteriaReleaseLayoutValue $layoutPath "package" "backend_root")
        $frontendEntry = Resolve-AsteriaPackagePath $appRootResolved (Get-AsteriaReleaseLayoutValue $layoutPath "package" "frontend_entry")
    }
    catch {
        Stop-Preflight "blocked" "startup_layout_invalid" 67 "The release layout is missing or invalid."
    }
}
else {
    $backendRoot = Resolve-AbsolutePath (Join-Path $appRootResolved "backend")
}
if ($null -eq $backendRoot -or -not (Test-ContainedPath $appRootResolved $backendRoot) -or -not [System.IO.Directory]::Exists([System.IO.Path]::Combine($backendRoot, "app"))) {
    Stop-Preflight "blocked" "startup_backend_missing" 67 "The Asteria backend is missing from the application folder."
}
if ($TargetMode -eq "packaged") {
    if ($null -eq $frontendEntry -or -not (Test-ContainedPath $appRootResolved $frontendEntry) -or -not [System.IO.File]::Exists($frontendEntry)) {
        Stop-Preflight "blocked" "startup_frontend_build_missing" 67 "The packaged frontend build is missing."
    }
}

$resolvedRoots = [ordered]@{}
$rootInputs = [ordered]@{
    DataRoot = if ($DataRoot) { $DataRoot } else { [System.IO.Path]::Combine($appRootResolved, "data") }
    ModelRoot = if ($ModelRoot) { $ModelRoot } else { [System.IO.Path]::Combine($appRootResolved, "models") }
    OutputRoot = if ($OutputRoot) { $OutputRoot } else { [System.IO.Path]::Combine($appRootResolved, "outputs") }
}
foreach ($entry in $rootInputs.GetEnumerator()) {
    $resolved = Resolve-AbsolutePath $entry.Value
    if ($null -eq $resolved) {
        Stop-Preflight "invalid" "startup_path_not_absolute" 64 "AppRoot and every target path must be absolute."
    }
    if (Test-DocumentationLocation $resolved) {
        Stop-Preflight "invalid" "startup_execution_location_unsafe" 65 "Choose writable application data folders, not documentation folders."
    }
    if (-not (Test-WritableDirectory $resolved)) {
        Stop-Preflight "blocked" "startup_writable_root_unavailable" 68 "A required writable data folder is missing or unavailable."
    }
    $resolvedRoots[$entry.Key] = $resolved
}

$comfyResolved = if ($ComfyRoot) { Resolve-AbsolutePath $ComfyRoot } else { Resolve-AbsolutePath ([System.IO.Path]::Combine($appRootResolved, "runtime", "ComfyUI")) }
if ($null -eq $comfyResolved) {
    Stop-Preflight "invalid" "startup_path_not_absolute" 64 "AppRoot and every target path must be absolute."
}
if (Test-DocumentationLocation $comfyResolved) {
    Stop-Preflight "invalid" "startup_execution_location_unsafe" 65 "Choose an installed runtime folder, not a documentation folder."
}

$env:PYTHONPATH = $backendRoot
$env:ASTERIA_ROOT_DIR = $appRootResolved
$env:ASTERIA_DATA_DIR = $resolvedRoots.DataRoot
$env:ASTERIA_MODEL_DIR = $resolvedRoots.ModelRoot
$env:ASTERIA_OUTPUT_DIR = $resolvedRoots.OutputRoot
$env:ASTERIA_MODEL_SELECTIONS_PATH = [System.IO.Path]::Combine($resolvedRoots.DataRoot, "model-selections.json")
$env:ASTERIA_RELEASE_MANIFEST_PATH = [System.IO.Path]::Combine($appRootResolved, "release-manifest.yaml")
$env:ASTERIA_DEFAULTS_DIR = [System.IO.Path]::Combine($appRootResolved, "defaults")
$env:ASTERIA_COMFY_DIR = $comfyResolved
$env:ASTERIA_COMFY_PYTHON = $pythonResolved
$env:ASTERIA_AUTO_BOOTSTRAP_MODELS = "0"

$arguments = @(
    "-m", "app.cli.startup_readiness",
    "--target-mode", $TargetMode,
    "--profile-id", $ProfileId
)
foreach ($featureId in $Feature) {
    if ($featureId.Trim()) {
        $arguments += @("--feature", $featureId.Trim())
    }
}

$cliOutput = & $pythonResolved @arguments 2>$null
$cliExitCode = $LASTEXITCODE
$cliLines = @($cliOutput | ForEach-Object { "$_" } | Where-Object { $_.Trim() })
$cliText = if ($cliLines.Count) { $cliLines[-1] } else { "" }
try {
    $payload = $cliText | ConvertFrom-Json
}
catch {
    Stop-Preflight "error" "startup_readiness_execution_failed" 70 "The readiness check could not be completed."
}
if ($null -eq $payload -or $null -eq $payload.report -or [int]$payload.exit_code -ne $cliExitCode) {
    Stop-Preflight "error" "startup_readiness_execution_failed" 70 "The readiness check could not be completed."
}

if ($Json) {
    Write-Output ($payload | ConvertTo-Json -Depth 12 -Compress)
}
else {
    $summary = $payload.report.summary
    $appState = if ($summary.app_ready) { "Ready" } else { "Needs setup" }
    $runtimeState = if ($summary.runtime_ready) { "Ready" } else { "Needs setup" }
    $generationState = if ($summary.generation_ready) { "Ready" } else { "Needs setup" }
    Write-Output "Asteria readiness: $($payload.status)"
    Write-Output "Application: $appState"
    Write-Output "Generation runtime: $runtimeState"
    Write-Output "Image generation: $generationState"
    Write-Output "Blockers: $($summary.blocking_count); Warnings: $($summary.warning_count); Actions: $($summary.action_count)"
}
exit $cliExitCode
