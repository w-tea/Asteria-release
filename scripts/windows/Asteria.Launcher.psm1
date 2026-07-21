Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-AsteriaFullyQualifiedPath {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::IsPathRooted($Path) -and ($Path -match '^[A-Za-z]:[\\/]' -or $Path.StartsWith('\\'))
}

function Get-AsteriaReleaseLayoutValue {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$Section,
        [Parameter(Mandatory = $true)][string]$Key
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "The release layout is missing."
    }
    $current = ""
    foreach ($line in Get-Content -LiteralPath $ManifestPath -Encoding utf8) {
        if ($line -match '^([A-Za-z_]+):\s*$') {
            $current = $Matches[1]
            continue
        }
        if ($current -eq $Section -and $line -match '^\s{2}([A-Za-z_]+):\s*(.+?)\s*$' -and $Matches[1] -eq $Key) {
            return $Matches[2].Trim('"', "'")
        }
    }
    throw "The release layout is missing $Section.$Key."
}

function Resolve-AsteriaPackagePath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    if ((Test-AsteriaFullyQualifiedPath $RelativePath) -or ($RelativePath -split '[\\/]' | Where-Object { $_ -eq '..' })) {
        throw "The release layout contains an unsafe path."
    }
    $root = [System.IO.Path]::GetFullPath($AppRoot).TrimEnd([char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar))
    $resolved = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($root, $RelativePath))
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "The release layout contains an unsafe path."
    }
    return $resolved
}

function Start-AsteriaOwnedProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StandardOutputPath,
        [Parameter(Mandatory = $true)][string]$StandardErrorPath
    )

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $StandardOutputPath -RedirectStandardError $StandardErrorPath -WindowStyle Hidden -PassThru
    try {
        $process.Refresh()
        return [pscustomobject][ordered]@{
            name = $Name
            pid = $process.Id
            executable_path = [System.IO.Path]::GetFullPath($FilePath)
            started_at_utc = $process.StartTime.ToUniversalTime().ToString("o")
        }
    }
    catch {
        try {
            if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction Stop }
        }
        catch {}
        throw
    }
}

function Get-AsteriaOwnedProcessState {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)]$Record)

    try {
        $process = Get-Process -Id ([int]$Record.pid) -ErrorAction Stop
        $process.Refresh()
    }
    catch {
        return "not-running"
    }
    try {
        $actualPath = [System.IO.Path]::GetFullPath($process.Path)
        $expectedPath = [System.IO.Path]::GetFullPath([string]$Record.executable_path)
        $expectedStart = if ($Record.started_at_utc -is [DateTime]) {
            ([DateTime]$Record.started_at_utc).ToUniversalTime()
        }
        else {
            [DateTime]::Parse([string]$Record.started_at_utc, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        }
        $actualStart = $process.StartTime.ToUniversalTime()
        $samePath = $actualPath.Equals($expectedPath, [System.StringComparison]::OrdinalIgnoreCase)
        $sameStart = $actualStart.Ticks -eq $expectedStart.Ticks
        if ($samePath -and $sameStart) { return "owned" }
    }
    catch {
        return "mismatch"
    }
    return "mismatch"
}

function Wait-AsteriaOwnedProcessExit {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Record,
        [int]$TimeoutSeconds
    )

    $watch = [Diagnostics.Stopwatch]::StartNew()
    do {
        if ((Get-AsteriaOwnedProcessState -Record $Record) -ne "owned") { return $true }
        Start-Sleep -Milliseconds 50
    } while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds)
    return (Get-AsteriaOwnedProcessState -Record $Record) -ne "owned"
}

function Stop-AsteriaOwnedProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Record,
        [int]$TimeoutSeconds = 10
    )

    if ((Get-AsteriaOwnedProcessState -Record $Record) -ne "owned") { return $false }
    try {
        Stop-Process -Id ([int]$Record.pid) -ErrorAction Stop
    }
    catch {
        if ((Get-AsteriaOwnedProcessState -Record $Record) -ne "owned") { return $true }
        throw
    }
    if (Wait-AsteriaOwnedProcessExit -Record $Record -TimeoutSeconds $TimeoutSeconds) { return $true }
    if ((Get-AsteriaOwnedProcessState -Record $Record) -eq "owned") {
        Stop-Process -Id ([int]$Record.pid) -Force -ErrorAction Stop
    }
    return Wait-AsteriaOwnedProcessExit -Record $Record -TimeoutSeconds $TimeoutSeconds
}

function Test-AsteriaTcpPortInUse {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        return $task.Wait(300) -and $client.Connected
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Wait-AsteriaHttpReady {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 60
    )

    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
            if ([int]$response.StatusCode -ge 200 -and [int]$response.StatusCode -lt 400) { return $true }
        }
        catch {}
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Write-AsteriaLauncherState {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$State
    )

    $parent = [System.IO.Path]::GetDirectoryName($Path)
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Read-AsteriaLauncherState {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json }
    catch { throw "The launcher ownership state is invalid." }
}

Export-ModuleMember -Function Test-AsteriaFullyQualifiedPath, Get-AsteriaReleaseLayoutValue, Resolve-AsteriaPackagePath, Start-AsteriaOwnedProcess, Get-AsteriaOwnedProcessState, Stop-AsteriaOwnedProcess, Test-AsteriaTcpPortInUse, Wait-AsteriaHttpReady, Write-AsteriaLauncherState, Read-AsteriaLauncherState
