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

function Initialize-AsteriaModelDirectories {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$ModelRoot)

    [System.IO.Directory]::CreateDirectory($ModelRoot) | Out-Null
    foreach ($directory in @(
        "checkpoints",
        "diffusion_models",
        "vae",
        "text_encoders",
        "loras",
        "embeddings",
        "controlnet",
        "adetailer",
        "bbox",
        "segm",
        "upscalers",
        "latent_upscale_models"
    )) {
        [System.IO.Directory]::CreateDirectory([System.IO.Path]::Combine($ModelRoot, $directory)) | Out-Null
    }
}

function Write-AsteriaExtraModelPaths {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ModelRoot,
        [string]$ExternalRegistryPath = ""
    )

    $parent = [System.IO.Path]::GetDirectoryName($Path)
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $basePath = $ModelRoot.Replace('\', '/').Replace('"', '\"')
    $lines = @(
        "asteria:"
        "  base_path: `"$basePath`""
        "  checkpoints: |-"
        "    checkpoints"
        "    diffusion_models"
        "    unet"
        "    merged/checkpoints"
        "  diffusion_models: |-"
        "    checkpoints"
        "    diffusion_models"
        "    unet"
        "    merged/checkpoints"
        "  unet: |-"
        "    checkpoints"
        "    diffusion_models"
        "    unet"
        "    merged/checkpoints"
        "  vae: |-"
        "    vae"
        "    vae_approx"
        "  vae_approx: |-"
        "    vae_approx"
        "  text_encoders: |-"
        "    text_encoders"
        "    clip"
        "  clip: |-"
        "    text_encoders"
        "    clip"
        "  tokenizers: |-"
        "    tokenizers"
        "    tokenizer"
        "  tokenizer: |-"
        "    tokenizers"
        "    tokenizer"
        "  loras: |-"
        "    loras"
        "    lora"
        "    merged/lora"
        "  embeddings: |-"
        "    embeddings"
        "    textual_inversion"
        "  controlnet: |-"
        "    controlnet"
        "    controlnets"
        "  model_patches: |-"
        "    controlnet"
        "    controlnets"
        "  ultralytics_bbox: |-"
        "    adetailer"
        "    bbox"
        "    ultralytics/bbox"
        "  ultralytics_segm: |-"
        "    adetailer"
        "    segm"
        "    ultralytics/segm"
        "  upscale_models: |-"
        "    upscalers"
        "    esrgan"
        "    realesrgan"
        "    latent_upscale_models"
    )
    if ($ExternalRegistryPath -and [System.IO.File]::Exists($ExternalRegistryPath)) {
        try {
            $registry = Get-Content -LiteralPath $ExternalRegistryPath -Raw | ConvertFrom-Json
            $keyMap = @{
                checkpoint = @("checkpoints", "diffusion_models", "unet")
                vae = @("vae")
                text_encoder = @("text_encoders", "clip")
                tokenizer = @("tokenizers", "tokenizer")
                lora = @("loras")
                embedding = @("embeddings")
                controlnet = @("controlnet", "model_patches")
                upscaler = @("upscale_models")
            }
            foreach ($root in @($registry.roots | Sort-Object @{ Expression = "priority"; Descending = $true }, display_name)) {
                if (-not $root.enabled -or -not $root.path -or -not [System.IO.Directory]::Exists([string]$root.path)) { continue }
                $section = ([string]$root.root_id -replace '[^A-Za-z0-9_]', '_')
                $externalBase = ([string]$root.path).Replace('\', '/').Replace('"', '\"')
                $lines += "asteria_external_$section`:"
                $lines += "  base_path: `"$externalBase`""
                foreach ($modelType in $keyMap.Keys) {
                    $directories = @($root.mappings.$modelType)
                    if ($directories.Count -eq 0) { continue }
                    foreach ($folderName in $keyMap[$modelType]) {
                        $lines += "  $folderName`: |-"
                        foreach ($directory in $directories) { $lines += "    $directory" }
                    }
                }
                $detailerDirectories = @($root.mappings.detailer)
                if ($detailerDirectories.Count -gt 0) {
                    $bboxDirectories = @()
                    $segmDirectories = @()
                    foreach ($directory in $detailerDirectories) {
                        $parts = ([string]$directory).Replace('\', '/').Split('/')
                        if ($parts -contains "segm") {
                            $segmDirectories += [string]$directory
                        } elseif ($parts -contains "bbox") {
                            $bboxDirectories += [string]$directory
                        } else {
                            $bboxDirectories += [string]$directory
                            $segmDirectories += [string]$directory
                        }
                    }
                    if ($bboxDirectories.Count -gt 0) {
                        $lines += "  ultralytics_bbox: |-"
                        foreach ($directory in $bboxDirectories | Select-Object -Unique) { $lines += "    $directory" }
                    }
                    if ($segmDirectories.Count -gt 0) {
                        $lines += "  ultralytics_segm: |-"
                        foreach ($directory in $segmDirectories | Select-Object -Unique) { $lines += "    $directory" }
                    }
                }
            }
        } catch {
            throw "External model registry could not be projected safely."
        }
    }
        $transactionId = [Guid]::NewGuid().ToString("N")
        $temporary = "$Path.$transactionId.tmp"
        $backup = "$Path.$transactionId.bak"
    try {
        $lines | Set-Content -LiteralPath $temporary -Encoding utf8
        if ([System.IO.File]::Exists($Path)) {
            [System.IO.File]::Replace($temporary, $Path, $backup, $true)
        } else {
            [System.IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        if ([System.IO.File]::Exists($temporary)) {
            [System.IO.File]::Delete($temporary)
        }
          if ([System.IO.File]::Exists($backup)) {
              [System.IO.File]::Delete($backup)
          }
        }
}

function Start-AsteriaOwnedProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [string]$StandardOutputPath = "",
        [string]$StandardErrorPath = "",
        [ValidateSet("Hidden", "Normal", "Minimized", "Maximized")][string]$WindowStyle = "Hidden",
        [switch]$Optional
    )

    # Windows treats environment names case-insensitively, but some parent
    # processes can still pass both Path and PATH. Windows PowerShell 5.1
    # Start-Process rejects that environment block as a duplicate dictionary
    # key, so collapse it before starting any launcher-owned child.
    $processPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("Path", $null, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("PATH", $null, [EnvironmentVariableTarget]::Process)
    if ($null -ne $processPath) {
        [Environment]::SetEnvironmentVariable("Path", $processPath, [EnvironmentVariableTarget]::Process)
    }

    $startParameters = @{
        FilePath = $FilePath
        ArgumentList = $ArgumentList
        WorkingDirectory = $WorkingDirectory
        WindowStyle = $WindowStyle
        PassThru = $true
    }
    if ($StandardOutputPath) { $startParameters.RedirectStandardOutput = $StandardOutputPath }
    if ($StandardErrorPath) { $startParameters.RedirectStandardError = $StandardErrorPath }
    $process = Start-Process @startParameters
    try {
        $process.Refresh()
        return [pscustomobject][ordered]@{
            name = $Name
            pid = $process.Id
            executable_path = [System.IO.Path]::GetFullPath($FilePath)
            started_at_utc = $process.StartTime.ToUniversalTime().ToString("o")
            optional = [bool]$Optional
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

function Get-AsteriaLauncherPreferences {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$SettingsPath)

    $result = [ordered]@{
        RuntimeMonitorOnStart = $false
        DisablePinnedMemory = $false
        ReasonId = $null
    }
    if (-not (Test-Path -LiteralPath $SettingsPath -PathType Leaf)) {
        return [pscustomobject]$result
    }
    try {
        $content = Get-Content -LiteralPath $SettingsPath -Raw -Encoding utf8 -ErrorAction Stop
    }
    catch {
        $result.ReasonId = "access_denied"
        return [pscustomobject]$result
    }
    try {
        $settings = $content | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        $result.ReasonId = "json_invalid"
        return [pscustomobject]$result
    }
    if ($null -eq $settings.launcher) { return [pscustomobject]$result }

    $runtimeProperty = $settings.launcher.PSObject.Properties["runtime_monitor_on_start"]
    if ($null -ne $runtimeProperty) {
        $runtimeMonitor = $runtimeProperty.Value
        $result.RuntimeMonitorOnStart = ($runtimeMonitor -is [bool]) -and $runtimeMonitor
    }
    if ($null -ne $settings.launcher.PSObject.Properties["disable_pinned_memory"]) {
        $memoryPreference = $settings.launcher.disable_pinned_memory
        if ($memoryPreference -is [bool]) {
            $result.DisablePinnedMemory = $memoryPreference
        }
        else {
            $result.ReasonId = "field_type_invalid"
        }
    }
    return [pscustomobject]$result
}

function Get-AsteriaRuntimeMonitorPreference {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$SettingsPath)

    return [bool](Get-AsteriaLauncherPreferences -SettingsPath $SettingsPath).RuntimeMonitorOnStart
}

function ConvertTo-AsteriaRuntimeMonitorLine {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line
    )

    if ($Line -match '^\[ERROR\]\s+\[Inspire Pack\]\s+IPAdapterPlus is not installed\.\s*$') {
        return "[INFO] [Asteria] Inspire Pack's optional IPAdapter integration is not installed. Asteria does not use this feature."
    }
    return $Line
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

    $process = $null
    try {
        $process = [Diagnostics.Process]::GetProcessById([int]$Record.pid)
        $milliseconds = [int][Math]::Min([int]::MaxValue, [Math]::Max(0, [long]$TimeoutSeconds) * 1000)
        return $process.WaitForExit($milliseconds)
    }
    catch [System.ArgumentException] { return $true }
    catch { return $false }
    finally { if ($null -ne $process) { $process.Dispose() } }
}

function Stop-AsteriaOwnedProcessRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Record,
        [int]$TimeoutSeconds = 10
    )

    $ownership = Get-AsteriaOwnedProcessState -Record $Record
    if ($ownership -eq "not-running") {
        try {
            $remaining = [Diagnostics.Process]::GetProcessById([int]$Record.pid)
            $remaining.Dispose()
            return $false
        }
        catch [System.ArgumentException] { return $true }
        catch { return $false }
    }
    if ($ownership -ne "owned") { return $false }
    try {
        Stop-Process -Id ([int]$Record.pid) -ErrorAction Stop
    }
    catch {
        if (Wait-AsteriaOwnedProcessExit -Record $Record -TimeoutSeconds 0) { return $true }
        throw
    }
    if (Wait-AsteriaOwnedProcessExit -Record $Record -TimeoutSeconds $TimeoutSeconds) { return $true }
    if ((Get-AsteriaOwnedProcessState -Record $Record) -eq "owned") {
        Stop-Process -Id ([int]$Record.pid) -Force -ErrorAction Stop
    }
    return Wait-AsteriaOwnedProcessExit -Record $Record -TimeoutSeconds $TimeoutSeconds
}

function Stop-AsteriaOwnedProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]$Record,
        [int]$TimeoutSeconds = 10
    )

    if ((Get-AsteriaOwnedProcessState -Record $Record) -ne "owned") { return $false }
    $snapshot = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, CreationDate -ErrorAction Stop)
    $parents = @($Record)
    $children = @()
    $visited = @{}
    $visited[[int]$Record.pid] = $true
    for ($index = 0; $index -lt $parents.Count; $index++) {
        $parent = $parents[$index]
        if ((Get-AsteriaOwnedProcessState -Record $parent) -ne "owned") { continue }
        $parentStart = (Get-Process -Id ([int]$parent.pid) -ErrorAction Stop).StartTime.ToUniversalTime()
        $parentStartStamp = $parentStart.ToString('yyyyMMddHHmmss.ffffff')
        foreach ($row in $snapshot) {
            $childId = [int]$row.ProcessId
            if ([int]$row.ParentProcessId -ne [int]$parent.pid -or $visited.ContainsKey($childId)) { continue }
            # Parent IDs can outlive their original owner. Reject older or unknown
            # records before querying potentially inaccessible unrelated processes.
            if ($null -eq $row.CreationDate) { continue }
            $snapshotStartStamp = $row.CreationDate.ToUniversalTime().ToString('yyyyMMddHHmmss.ffffff')
            if ($snapshotStartStamp -lt $parentStartStamp) { continue }
            try {
                $child = Get-Process -Id $childId -ErrorAction Stop
                $started = $child.StartTime.ToUniversalTime()
                if ($started -lt $parentStart) { continue }
                if ($started.ToString('yyyyMMddHHmmss.ffffff') -ne $snapshotStartStamp) { continue }
                $childRecord = [pscustomobject]@{
                    name = "$($Record.name)-child-$childId"
                    pid = $childId
                    executable_path = [System.IO.Path]::GetFullPath($child.Path)
                    started_at_utc = $started.ToString('o')
                }
            }
            catch {
                if (-not (Get-Process -Id $childId -ErrorAction SilentlyContinue)) { continue }
                throw
            }
            $visited[$childId] = $true
            $children += $childRecord
            $parents += $childRecord
        }
    }
    # Stop the spawning owner first, then its frozen descendants. Every stop
    # still checks executable and creation time; reused PIDs remain untouched.
    $stopped = Stop-AsteriaOwnedProcessRecord -Record $Record -TimeoutSeconds $TimeoutSeconds
    [array]::Reverse($children)
    foreach ($childRecord in $children) {
        if ((Get-AsteriaOwnedProcessState -Record $childRecord) -ne "owned") { continue }
        if (-not (Stop-AsteriaOwnedProcessRecord -Record $childRecord -TimeoutSeconds $TimeoutSeconds)) { $stopped = $false }
    }
    return $stopped
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

Export-ModuleMember -Function Test-AsteriaFullyQualifiedPath, Get-AsteriaReleaseLayoutValue, Resolve-AsteriaPackagePath, Initialize-AsteriaModelDirectories, Write-AsteriaExtraModelPaths, Start-AsteriaOwnedProcess, Get-AsteriaLauncherPreferences, Get-AsteriaRuntimeMonitorPreference, ConvertTo-AsteriaRuntimeMonitorLine, Get-AsteriaOwnedProcessState, Stop-AsteriaOwnedProcess, Test-AsteriaTcpPortInUse, Wait-AsteriaHttpReady, Write-AsteriaLauncherState, Read-AsteriaLauncherState
