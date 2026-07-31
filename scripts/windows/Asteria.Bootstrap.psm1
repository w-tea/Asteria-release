Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"
Import-Module (Join-Path $PSScriptRoot "Asteria.Maintenance.psm1") -Force

function New-AsteriaBootstrapError {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][string]$Message
    )
    $exception = [System.InvalidOperationException]::new($Message)
    $exception.Data["AsteriaCode"] = $Code
    return $exception
}

if (-not ("System.Net.Http.HttpClient" -as [type])) {
    try {
        Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
    }
    catch {
        throw (New-AsteriaBootstrapError -Code "bootstrap_http_runtime_unavailable" -Message "Windows HTTP support could not be loaded.")
    }
}

function Resolve-AsteriaBootstrapRoot {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $isDrivePath = $AppRoot -match '^[A-Za-z]:[\\/]'
    $isUncPath = $AppRoot.StartsWith('\\')
    if (-not [System.IO.Path]::IsPathRooted($AppRoot) -or (-not $isDrivePath -and -not $isUncPath)) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_path_not_absolute" -Message "Application root must be absolute.")
    }
    try {
        return [System.IO.Path]::GetFullPath($AppRoot).TrimEnd('\', '/')
    }
    catch {
        throw (New-AsteriaBootstrapError -Code "bootstrap_path_invalid" -Message "Application root is invalid.")
    }
}

function Get-AsteriaReleaseArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$ArtifactId
    )
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_missing" -Message "Release manifest is missing.")
    }
    $lines = [System.IO.File]::ReadAllLines($ManifestPath)
    $inArtifact = $false
    $values = [ordered]@{}
    foreach ($line in $lines) {
        if ($line -match '^  - id:\s*(.+?)\s*$') {
            if ($inArtifact) { break }
            $candidateId = $Matches[1].Trim('"', "'")
            if ($candidateId -eq $ArtifactId) {
                $inArtifact = $true
                $values["id"] = $candidateId
            }
            continue
        }
        if (-not $inArtifact) { continue }
        if ($line -match '^\s+url:\s*(https://\S+)\s*$' -and -not $values.Contains("url")) {
            $values["url"] = $Matches[1].Trim('"', "'")
        }
        elseif ($line -match '^\s+relative_path:\s*(\S.+?)\s*$' -and -not $values.Contains("target_relative_path")) {
            $values["target_relative_path"] = $Matches[1].Trim('"', "'")
        }
        elseif ($line -match '^\s+size_bytes:\s*(\d+)\s*$') {
            $values["size_bytes"] = [int64]$Matches[1]
        }
        elseif ($line -match '^\s+checksum_sha256:\s*([0-9a-f]{64})\s*$') {
            $values["checksum_sha256"] = $Matches[1]
        }
    }
    foreach ($required in @("id", "url", "target_relative_path", "size_bytes", "checksum_sha256")) {
        if (-not $values.Contains($required)) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "Release artifact metadata is invalid.")
        }
    }
    return [pscustomobject]$values
}

function Get-AsteriaMinimumNvidiaDriver {
    param([Parameter(Mandatory = $true)][string]$ManifestPath)
    foreach ($line in [System.IO.File]::ReadAllLines($ManifestPath)) {
        if ($line -match '^\s+minimum_nvidia_driver:\s*["'']?([0-9]+(?:\.[0-9]+){1,3})["'']?\s*$') {
            return $Matches[1]
        }
    }
    throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "NVIDIA driver requirement is missing.")
}

function Get-AsteriaBootstrapPlan {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $root = Resolve-AsteriaBootstrapRoot -AppRoot $AppRoot
    $manifestPath = Join-Path $root "release-manifest.yaml"
    $uv = Get-AsteriaReleaseArtifact -ManifestPath $manifestPath -ArtifactId "bootstrap-uv"
    $python = Get-AsteriaReleaseArtifact -ManifestPath $manifestPath -ArtifactId "python-runtime"
    return [pscustomobject][ordered]@{
        app_root = $root
        manifest_path = $manifestPath
        uv = $uv
        python = $python
        minimum_nvidia_driver = Get-AsteriaMinimumNvidiaDriver -ManifestPath $manifestPath
        contained_python = Join-Path $root "runtime\python\python.exe"
    }
}

function Invoke-AsteriaNvidiaSmi {
    param([Parameter(Mandatory = $true)][string]$ExecutablePath)
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $ExecutablePath
    $startInfo.Arguments = "--query-gpu=name,driver_version --format=csv,noheader,nounits"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_probe_failed" -Message "NVIDIA hardware check could not start.")
        }
        if (-not $process.WaitForExit(5000)) {
            try { $process.Kill() } catch { }
            throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_probe_timeout" -Message "NVIDIA hardware check timed out.")
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $process.StandardOutput.ReadToEnd()
        }
    }
    finally {
        $process.Dispose()
    }
}

function Test-AsteriaBootstrapHardware {
    param([Parameter(Mandatory = $true)][string]$MinimumDriver)
    if (-not [System.Environment]::Is64BitOperatingSystem) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_arch_unsupported" -Message "Asteria requires 64-bit Windows.")
    }
    $candidates = @(
        (Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
        (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    )
    $executable = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $executable) {
        $command = Get-Command "nvidia-smi.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) { $executable = $command.Source }
    }
    if (-not $executable) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_nvidia_required" -Message "A supported NVIDIA GPU and driver are required.")
    }
    try {
        $result = Invoke-AsteriaNvidiaSmi -ExecutablePath $executable
    }
    catch {
        if ($_.Exception.Data["AsteriaCode"]) { throw }
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_probe_failed" -Message "NVIDIA hardware check failed.")
    }
    if ($result.ExitCode -ne 0) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_probe_failed" -Message "NVIDIA hardware check failed.")
    }
    $supported = @()
    foreach ($line in @($result.Stdout -split "`r?`n" | Where-Object { $_.Trim() })) {
        $fields = @($line -split ',', 2 | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 2 -or $fields[0].ToLowerInvariant().IndexOf("nvidia") -lt 0 -or $fields[1] -notmatch '^\d+(?:\.\d+){1,3}$') {
            continue
        }
        $supported += ,$fields
    }
    if ($supported.Count -eq 0) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_nvidia_required" -Message "A supported NVIDIA GPU and driver are required.")
    }
    try {
        $minimum = [version]$MinimumDriver
        $drivers = @($supported | ForEach-Object { [version]$_[1] })
        $current = $drivers | Sort-Object | Select-Object -First 1
    }
    catch {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_probe_invalid" -Message "NVIDIA driver information is invalid.")
    }
    if ($current -lt $minimum) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_hardware_driver_outdated" -Message "The NVIDIA driver must be updated before setup.")
    }
    return [pscustomobject]@{ supported = $true; driver_version = $current.ToString() }
}

function Resolve-AsteriaContainedTarget {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    if ([System.IO.Path]::IsPathRooted($RelativePath) -or $RelativePath -match '(^|[\\/])\.\.([\\/]|$)' -or $RelativePath -match ':') {
        throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "Release artifact target is invalid.")
    }
    $root = [System.IO.Path]::GetFullPath($AppRoot).TrimEnd('\', '/')
    $target = [System.IO.Path]::GetFullPath((Join-Path $root $RelativePath))
    $prefix = $root + [System.IO.Path]::DirectorySeparatorChar
    if (-not $target.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "Release artifact target escapes the application root.")
    }
    return $target
}

function Assert-AsteriaManagedWritePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        $candidate = [System.IO.Path]::GetFullPath($Path)
    }
    catch {
        throw (New-AsteriaBootstrapError -Code "bootstrap_managed_path_unsafe" -Message "Managed setup path could not be inspected safely.")
    }
    while (-not [string]::IsNullOrWhiteSpace($candidate)) {
        try {
            $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw (New-AsteriaBootstrapError -Code "bootstrap_managed_path_unsafe" -Message "Managed setup refuses filesystem links and reparse points.")
            }
        }
        catch [System.Management.Automation.ItemNotFoundException] {
            # Nonexistent lexical components are allowed; an existing ancestor is still inspected.
        }
        catch {
            if ($_.Exception.Data["AsteriaCode"]) { throw }
            throw (New-AsteriaBootstrapError -Code "bootstrap_managed_path_unsafe" -Message "Managed setup path could not be inspected safely.")
        }
        $parent = [System.IO.Path]::GetDirectoryName($candidate)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $candidate) { break }
        $candidate = $parent
    }
}

function Get-AsteriaBootstrapReceiptPath {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact
    )
    if ([string]$Artifact.id -notmatch '^[a-z0-9][a-z0-9._-]*$') {
        throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "Release artifact id is invalid.")
    }
    return Resolve-AsteriaContainedTarget -AppRoot $AppRoot -RelativePath "runtime/.asteria-bootstrap/$($Artifact.id).json"
}

function Get-AsteriaBootstrapComponentIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [Parameter(Mandatory = $true)][string]$EntryPointPath
    )
    $root = [System.IO.Path]::GetFullPath($AppRoot).TrimEnd('\', '/')
    $target = Resolve-AsteriaContainedTarget -AppRoot $root -RelativePath ([string]$Artifact.target_relative_path)
    $entry = [System.IO.Path]::GetFullPath($EntryPointPath)
    $entryParent = [System.IO.Path]::GetDirectoryName($entry)
    if (-not $entryParent.Equals($target, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_manifest_invalid" -Message "Bootstrap entry point does not match its managed target.")
    }
    if (-not (Test-Path -LiteralPath $entry -PathType Leaf)) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_component_missing" -Message "Bootstrap entry point is missing.")
    }
    $relativeTarget = ([string]$Artifact.target_relative_path).Replace('\', '/').Trim('/')
    return [ordered]@{
        version = 1
        artifact_id = [string]$Artifact.id
        artifact_size_bytes = [int64]$Artifact.size_bytes
        artifact_sha256 = [string]$Artifact.checksum_sha256
        target_relative_path = $relativeTarget
        entry_relative_path = "$relativeTarget/$([System.IO.Path]::GetFileName($entry))"
        entry_size_bytes = [int64](Get-Item -LiteralPath $entry).Length
        entry_sha256 = (Get-FileHash -LiteralPath $entry -Algorithm SHA256).Hash.ToLowerInvariant()
        state = "complete"
    }
}

function Test-AsteriaBootstrapComponentInstalled {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [Parameter(Mandatory = $true)][string]$EntryPointPath
    )
    try {
        $receiptPath = Get-AsteriaBootstrapReceiptPath -AppRoot $AppRoot -Artifact $Artifact
        if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { return $false }
        $expected = Get-AsteriaBootstrapComponentIdentity -AppRoot $AppRoot -Artifact $Artifact -EntryPointPath $EntryPointPath
        $actual = Get-Content -LiteralPath $receiptPath -Raw -Encoding utf8 | ConvertFrom-Json
        foreach ($key in $expected.Keys) {
            if ([string]$actual.$key -cne [string]$expected[$key]) { return $false }
        }
        return $true
    }
    catch {
        return $false
    }
}

function Write-AsteriaBootstrapComponentReceipt {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [Parameter(Mandatory = $true)][string]$EntryPointPath
    )
    $receiptPath = Get-AsteriaBootstrapReceiptPath -AppRoot $AppRoot -Artifact $Artifact
    Assert-AsteriaManagedWritePath -Path $receiptPath
    $receipt = Get-AsteriaBootstrapComponentIdentity -AppRoot $AppRoot -Artifact $Artifact -EntryPointPath $EntryPointPath
    $parent = [System.IO.Path]::GetDirectoryName($receiptPath)
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = "$receiptPath.$([Guid]::NewGuid().ToString('N')).tmp"
    try {
        Assert-AsteriaManagedWritePath -Path $temporary
        $receipt | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $temporary -Encoding utf8
        Assert-AsteriaManagedWritePath -Path $receiptPath
        Move-Item -LiteralPath $temporary -Destination $receiptPath -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Assert-AsteriaManagedWritePath -Path $temporary
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

function Clear-AsteriaPythonPackageReceipts {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $receiptRoot = Resolve-AsteriaContainedTarget -AppRoot $AppRoot -RelativePath "runtime/.asteria-packages"
    if (Test-Path -LiteralPath $receiptRoot) {
        Assert-AsteriaManagedWritePath -Path $receiptRoot
        Remove-Item -LiteralPath $receiptRoot -Recurse -Force
    }
}

function Test-AsteriaVerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [Parameter(Mandatory = $true)][string]$DownloadPath
    )
    if (-not (Test-Path -LiteralPath $DownloadPath -PathType Leaf)) { return $false }
    try {
        if ((Get-Item -LiteralPath $DownloadPath).Length -ne [int64]$Artifact.size_bytes) { return $false }
        return (Get-FileHash -LiteralPath $DownloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Artifact.checksum_sha256
    }
    catch {
        return $false
    }
}

function Invoke-AsteriaVerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [Parameter(Mandatory = $true)][string]$DownloadPath
    )
    $parent = Split-Path -Parent $DownloadPath
    Assert-AsteriaManagedWritePath -Path $DownloadPath
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $partPath = "$DownloadPath.part"
    Assert-AsteriaManagedWritePath -Path $partPath
    if (Test-AsteriaVerifiedDownload -Artifact $Artifact -DownloadPath $DownloadPath) { return }
    if (Test-Path -LiteralPath $DownloadPath) {
        Assert-AsteriaManagedWritePath -Path $DownloadPath
        Remove-Item -LiteralPath $DownloadPath -Force
    }
    $offset = 0L
    if (Test-Path -LiteralPath $partPath -PathType Leaf) {
        $offset = (Get-Item -LiteralPath $partPath).Length
        if ($offset -gt [int64]$Artifact.size_bytes) {
            Assert-AsteriaManagedWritePath -Path $partPath
            Remove-Item -LiteralPath $partPath -Force
            $offset = 0L
        }
    }
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $true
    $client = [System.Net.Http.HttpClient]::new($handler)
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $Artifact.url)
    $request.Headers.UserAgent.ParseAdd("Asteria-Bootstrap/1.0")
    if ($offset -gt 0) {
        $request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new($offset, $null)
    }
    try {
        $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        try {
            $response.EnsureSuccessStatusCode() | Out-Null
            $append = $offset -gt 0 -and [int]$response.StatusCode -eq 206
            $mode = if ($append) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
            $sourceStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            try {
                Assert-AsteriaManagedWritePath -Path $partPath
                $targetStream = [System.IO.File]::Open($partPath, $mode, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
                try {
                    $sourceStream.CopyToAsync($targetStream).GetAwaiter().GetResult()
                }
                finally {
                    $targetStream.Dispose()
                }
            }
            finally {
                $sourceStream.Dispose()
            }
        }
        finally {
            $response.Dispose()
        }
    }
    catch {
        throw (New-AsteriaBootstrapError -Code "bootstrap_download_interrupted" -Message "Runtime download was interrupted and can be resumed.")
    }
    finally {
        $request.Dispose()
        $client.Dispose()
        $handler.Dispose()
    }
    $download = Get-Item -LiteralPath $partPath
    if ($download.Length -ne [int64]$Artifact.size_bytes) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_download_size_mismatch" -Message "Downloaded artifact size is invalid.")
    }
    $hash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $Artifact.checksum_sha256) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_download_checksum_mismatch" -Message "Downloaded artifact checksum is invalid.")
    }
    Assert-AsteriaManagedWritePath -Path $DownloadPath
    Move-Item -LiteralPath $partPath -Destination $DownloadPath -Force
}

function Write-AsteriaBootstrapProgress {
    param(
        [switch]$Json,
        [Parameter(Mandatory = $true)][string]$Code,
        [Parameter(Mandatory = $true)][pscustomobject]$Artifact,
        [int64]$DownloadedBytes = 0
    )
    $payload = [ordered]@{
        code = $Code
        component_id = [string]$Artifact.id
        downloaded_bytes = $DownloadedBytes
        total_size_bytes = [int64]$Artifact.size_bytes
    }
    if ($Json) {
        [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress))
    }
    else {
        $percent = if ([int64]$Artifact.size_bytes -gt 0) { [Math]::Floor(($DownloadedBytes * 100.0) / [int64]$Artifact.size_bytes) } else { 0 }
        [Console]::Out.WriteLine("[Asteria setup] $($Artifact.id): $percent% ($DownloadedBytes / $($Artifact.size_bytes) bytes)")
    }
}

function Test-AsteriaBootstrapArchive {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][ValidateSet("Zip", "TarGz")][string]$Format
    )
    $entries = @()
    if ($Format -eq "Zip") {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
        try { $entries = @($archive.Entries | ForEach-Object { $_.FullName }) }
        finally { $archive.Dispose() }
    }
    else {
        $tarPath = Join-Path $env:SystemRoot "System32\tar.exe"
        if (-not (Test-Path -LiteralPath $tarPath -PathType Leaf)) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_extractor_missing" -Message "Windows archive support is unavailable.")
        }
        $entries = @(& $tarPath -tf $ArchivePath)
        if ($LASTEXITCODE -ne 0) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_archive_invalid" -Message "Runtime archive is invalid.")
        }
    }
    if ($entries.Count -eq 0) {
        throw (New-AsteriaBootstrapError -Code "bootstrap_archive_invalid" -Message "Runtime archive is empty.")
    }
    foreach ($entry in $entries) {
        $normalized = ([string]$entry).Replace('\', '/')
        $segments = @($normalized.Split('/') | Where-Object { $_ -ne "" -and $_ -ne "." })
        if ($normalized.StartsWith('/') -or $normalized.StartsWith('//') -or $normalized -match '^[A-Za-z]:' -or $segments -contains '..' -or $normalized.Contains(':')) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_archive_unsafe" -Message "Runtime archive contains an unsafe path.")
        }
    }
    return $true
}

function Expand-AsteriaBootstrapArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$ArchivePath,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][ValidateSet("Zip", "TarGz")][string]$Format
    )
    $parent = Split-Path -Parent $Destination
    Assert-AsteriaManagedWritePath -Path $Destination
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    Assert-AsteriaManagedWritePath -Path $Destination
    $staging = "$Destination.staging-$([Guid]::NewGuid().ToString('N'))"
    Assert-AsteriaManagedWritePath -Path $staging
    [System.IO.Directory]::CreateDirectory($staging) | Out-Null
    $backup = "$Destination.backup-$([Guid]::NewGuid().ToString('N'))"
    try {
        Assert-AsteriaManagedWritePath -Path $staging
        Assert-AsteriaManagedWritePath -Path $backup
        Test-AsteriaBootstrapArchive -ArchivePath $ArchivePath -Format $Format | Out-Null
        if ($Format -eq "Zip") {
            Expand-Archive -LiteralPath $ArchivePath -DestinationPath $staging -Force
        }
        else {
            $tarPath = Join-Path $env:SystemRoot "System32\tar.exe"
            if (-not (Test-Path -LiteralPath $tarPath -PathType Leaf)) {
                throw (New-AsteriaBootstrapError -Code "bootstrap_extractor_missing" -Message "Windows archive support is unavailable.")
            }
            & $tarPath -xf $ArchivePath -C $staging --strip-components=1
            if ($LASTEXITCODE -ne 0) {
                throw (New-AsteriaBootstrapError -Code "bootstrap_extract_failed" -Message "Runtime extraction failed.")
            }
        }
        if (Test-Path -LiteralPath $Destination) {
            Assert-AsteriaManagedWritePath -Path $Destination
            Move-Item -LiteralPath $Destination -Destination $backup
        }
        try {
            Assert-AsteriaManagedWritePath -Path $Destination
            Move-Item -LiteralPath $staging -Destination $Destination
        }
        catch {
            if ((Test-Path -LiteralPath $backup) -and -not (Test-Path -LiteralPath $Destination)) {
                Assert-AsteriaManagedWritePath -Path $backup
                Assert-AsteriaManagedWritePath -Path $Destination
                Move-Item -LiteralPath $backup -Destination $Destination
            }
            throw
        }
        if (Test-Path -LiteralPath $backup) {
            Assert-AsteriaManagedWritePath -Path $backup
            Remove-Item -LiteralPath $backup -Recurse -Force
        }
    }
    finally {
        if (Test-Path -LiteralPath $staging) {
            Assert-AsteriaManagedWritePath -Path $staging
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

function Set-AsteriaContainedEnvironment {
    param([Parameter(Mandatory = $true)][string]$AppRoot)
    $env:UV_CACHE_DIR = Join-Path $AppRoot "temp\cache\uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppRoot "runtime\python"
    $env:UV_NO_CONFIG = "1"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PIP_CONFIG_FILE = "NUL"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
}

function Invoke-AsteriaBootstrap {
    param(
        [Parameter(Mandatory = $true)][string]$AppRoot,
        [Parameter(Mandatory = $true)][ValidateSet("Install", "Repair")][string]$Mode,
        [switch]$ConfirmDownload,
        [string[]]$StarterPack = @(),
        [switch]$AcceptAnimaLicense,
        [switch]$PlanOnly,
        [switch]$Json,
        [string]$OfficialManifestPath = "",
        [string]$OfficialSignaturePath = "",
        [string]$OfficialTrustedKeysPath = ""
    )
    try {
        $root = Resolve-AsteriaBootstrapRoot -AppRoot $AppRoot
        if (-not $ConfirmDownload -and -not $PlanOnly) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_download_consent_required" -Message "Download consent is required.")
        }
        $plan = Get-AsteriaBootstrapPlan -AppRoot $root
        if ($PlanOnly) {
            $planPayload = [ordered]@{
                code = "bootstrap_plan_ready"
                artifact_ids = @($plan.uv.id, $plan.python.id)
                total_size_bytes = [int64]$plan.uv.size_bytes + [int64]$plan.python.size_bytes
            }
            if ($Json) { [Console]::Out.WriteLine(($planPayload | ConvertTo-Json -Compress)) }
            return 0
        }
        Test-AsteriaBootstrapHardware -MinimumDriver $plan.minimum_nvidia_driver | Out-Null
        Set-AsteriaContainedEnvironment -AppRoot $plan.app_root
        $downloadRoot = Join-Path $plan.app_root "temp\bootstrap"
        $uvArchive = Join-Path $downloadRoot "uv.zip"
        $pythonArchive = Join-Path $downloadRoot "python.tar.gz"
        $uvDestination = Resolve-AsteriaContainedTarget -AppRoot $plan.app_root -RelativePath $plan.uv.target_relative_path
        $pythonDestination = Resolve-AsteriaContainedTarget -AppRoot $plan.app_root -RelativePath $plan.python.target_relative_path
        $uvEntryPoint = Join-Path $uvDestination "uv.exe"
        $pythonEntryPoint = $plan.contained_python

        if ($Mode -eq "Install" -and (Test-AsteriaBootstrapComponentInstalled -AppRoot $plan.app_root -Artifact $plan.uv -EntryPointPath $uvEntryPoint)) {
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_component_skipped" -Artifact $plan.uv
        }
        else {
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_component_started" -Artifact $plan.uv
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_download_progress" -Artifact $plan.uv
            Invoke-AsteriaVerifiedDownload -Artifact $plan.uv -DownloadPath $uvArchive
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_download_progress" -Artifact $plan.uv -DownloadedBytes ([int64]$plan.uv.size_bytes)
            Expand-AsteriaBootstrapArtifact -ArchivePath $uvArchive -Destination $uvDestination -Format Zip
            Write-AsteriaBootstrapComponentReceipt -AppRoot $plan.app_root -Artifact $plan.uv -EntryPointPath $uvEntryPoint
        }
        if ($Mode -eq "Install" -and (Test-AsteriaBootstrapComponentInstalled -AppRoot $plan.app_root -Artifact $plan.python -EntryPointPath $pythonEntryPoint)) {
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_component_skipped" -Artifact $plan.python
        }
        else {
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_component_started" -Artifact $plan.python
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_download_progress" -Artifact $plan.python
            Invoke-AsteriaVerifiedDownload -Artifact $plan.python -DownloadPath $pythonArchive
            Write-AsteriaBootstrapProgress -Json:$Json -Code "bootstrap_download_progress" -Artifact $plan.python -DownloadedBytes ([int64]$plan.python.size_bytes)
            Expand-AsteriaBootstrapArtifact -ArchivePath $pythonArchive -Destination $pythonDestination -Format TarGz
            Clear-AsteriaPythonPackageReceipts -AppRoot $plan.app_root
            Write-AsteriaBootstrapComponentReceipt -AppRoot $plan.app_root -Artifact $plan.python -EntryPointPath $pythonEntryPoint
        }

        $backendRoot = Join-Path $plan.app_root "app\backend"
        if (-not (Test-Path -LiteralPath $backendRoot -PathType Container)) {
            throw (New-AsteriaBootstrapError -Code "bootstrap_application_missing" -Message "Packaged backend is missing.")
        }
        $officialSigningInputs = @($OfficialManifestPath, $OfficialSignaturePath, $OfficialTrustedKeysPath)
        $hasOfficialSigningInput = @($officialSigningInputs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count -gt 0
        if ($hasOfficialSigningInput) {
            if (@($officialSigningInputs | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
                throw (New-AsteriaBootstrapError -Code "bootstrap_release_signature_invalid" -Message "Official release signing inputs are incomplete.")
            }
            $verifierEnvironmentArguments = @("-E", "-s", "-m", "app.cli.bootstrap_environment", "--app-root", $plan.app_root, "--profile", "release-verifier")
            Push-Location $backendRoot
            try {
                & $plan.contained_python @verifierEnvironmentArguments | ForEach-Object { [Console]::Out.WriteLine([string]$_) }
                if ($LASTEXITCODE -ne 0) {
                    throw (New-AsteriaBootstrapError -Code "bootstrap_release_verifier_failed" -Message "The official release verifier could not be installed.")
                }
            }
            finally {
                Pop-Location
            }
            try {
                Test-AsteriaOfficialManifest -PythonPath $plan.contained_python -BackendRoot $backendRoot -ManifestPath $OfficialManifestPath -SignaturePath $OfficialSignaturePath -TrustedKeysPath $OfficialTrustedKeysPath | Out-Null
            }
            catch {
                throw (New-AsteriaBootstrapError -Code "bootstrap_release_signature_invalid" -Message "The official release signature is missing or invalid.")
            }
        }
        $backendEnvironmentArguments = @("-E", "-s", "-m", "app.cli.bootstrap_environment", "--app-root", $plan.app_root, "--profile", "backend")
        if ($Json) {
            [Console]::Out.WriteLine((([ordered]@{ code = "bootstrap_component_started"; component_id = "asteria-backend-environment" }) | ConvertTo-Json -Compress))
        }
        else {
            [Console]::Out.WriteLine("[Asteria setup] Installing the contained backend environment...")
        }
        Push-Location $backendRoot
        try {
            & $plan.contained_python @backendEnvironmentArguments | ForEach-Object { [Console]::Out.WriteLine([string]$_) }
            if ($LASTEXITCODE -ne 0) {
                throw (New-AsteriaBootstrapError -Code "bootstrap_backend_environment_failed" -Message "The contained backend environment could not be installed.")
            }
        }
        finally {
            Pop-Location
        }
        $arguments = @("-E", "-s", "-m", "app.cli.managed_setup", "--app-root", $plan.app_root, "--mode", $Mode.ToLowerInvariant())
        if ($ConfirmDownload) { $arguments += "--confirm-download" }
        foreach ($pack in $StarterPack) { $arguments += @("--starter-pack", $pack) }
        if ($AcceptAnimaLicense) { $arguments += @("--accept-license", "anima") }
        Push-Location $backendRoot
        try {
            & $plan.contained_python @arguments | ForEach-Object { [Console]::Out.WriteLine([string]$_) }
            $pythonExitCode = $LASTEXITCODE
            return $pythonExitCode
        }
        finally {
            Pop-Location
        }
    }
    catch {
        $code = $_.Exception.Data["AsteriaCode"]
        if (-not $code) { $code = "bootstrap_failed" }
        $payload = [ordered]@{ code = $code }
        if ($Json) {
            [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress))
        }
        else {
            [Console]::Error.WriteLine($code)
        }
        switch ($code) {
            "bootstrap_path_not_absolute" { return 64 }
            "bootstrap_path_invalid" { return 64 }
            "bootstrap_manifest_missing" { return 65 }
            "bootstrap_manifest_invalid" { return 65 }
            "bootstrap_download_consent_required" { return 66 }
            "bootstrap_hardware_arch_unsupported" { return 67 }
            "bootstrap_hardware_nvidia_required" { return 67 }
            "bootstrap_hardware_driver_outdated" { return 67 }
            "bootstrap_hardware_probe_timeout" { return 70 }
            "bootstrap_hardware_probe_failed" { return 70 }
            "bootstrap_hardware_probe_invalid" { return 70 }
            "bootstrap_release_verifier_failed" { return 76 }
            "bootstrap_release_signature_invalid" { return 76 }
            default { return 70 }
        }
    }
}

Export-ModuleMember -Function Get-AsteriaBootstrapPlan, Test-AsteriaBootstrapComponentInstalled, Write-AsteriaBootstrapComponentReceipt, Clear-AsteriaPythonPackageReceipts, Invoke-AsteriaBootstrap
