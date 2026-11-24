[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$AppRoot,
    [Parameter(Mandatory = $true)][string]$StatePath,
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [ValidateSet('mcp-bridge', 'lan-gateway')][string]$Kind = 'mcp-bridge'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Import-Module (Join-Path $PSScriptRoot 'Asteria.Launcher.psm1') -Force

if (-not (Test-AsteriaFullyQualifiedPath $AppRoot) -or -not (Test-AsteriaFullyQualifiedPath $StatePath)) {
    throw 'An absolute application folder and launcher record are required.'
}
$root = [IO.Path]::GetFullPath($AppRoot)
$state = Read-AsteriaLauncherState -Path $StatePath
if ($null -eq $state -or $state.version -ne 2 -or -not ([IO.Path]::GetFullPath([string]$state.app_root)).Equals($root, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Start this Asteria installation before connecting an MCP client.'
}
$backend = @($state.processes | Where-Object { $_.name -eq 'backend' })
if ($backend.Count -ne 1 -or (Get-AsteriaOwnedProcessState -Record $backend[0]) -ne 'owned') {
    throw 'Start Asteria before connecting an MCP client.'
}
# Only the Python process that invoked this registrar may register itself.
$caller = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -Property ParentProcessId
if ([int]$caller.ParentProcessId -ne $ProcessId) { throw 'The bridge process identity does not match its caller.' }
$process = Get-Process -Id $ProcessId -ErrorAction Stop
$record = [ordered]@{
    name = $Kind; pid = $process.Id; executable_path = $process.Path
    started_at_utc = $process.StartTime.ToUniversalTime().ToString('o'); optional = $true
}
$connectionDirectory = if ($Kind -eq 'lan-gateway') { 'lan-gateways' } else { 'mcp-bridges' }
$directory = Join-Path ([IO.Path]::GetDirectoryName($StatePath)) $connectionDirectory
[void][IO.Directory]::CreateDirectory($directory)
$path = Join-Path $directory ("{0}-{1}.json" -f $process.Id, $process.StartTime.ToUniversalTime().Ticks)
$temporary = Join-Path $directory ([Guid]::NewGuid().ToString('N') + '.tmp')
try {
    $payload = [ordered]@{ version = 1; app_root = $root; process = $record }
    [IO.File]::WriteAllText($temporary, ($payload | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary, $path)
    # Stop may have started between the first check and the sidecar publication.
    if (-not (Test-Path -LiteralPath $StatePath) -or (Get-AsteriaOwnedProcessState -Record $backend[0]) -ne 'owned') {
        Remove-Item -LiteralPath $path -Force
        throw 'Asteria stopped while the MCP client was connecting.'
    }
    [ordered]@{ path = $path } | ConvertTo-Json -Compress
}
finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
}
