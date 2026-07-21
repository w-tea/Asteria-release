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
Import-Module (Join-Path $PSScriptRoot "Asteria.Bootstrap.psm1") -Force
$exitCode = Invoke-AsteriaBootstrap -AppRoot $AppRoot -Mode Install -ConfirmDownload:$ConfirmDownload -StarterPack $StarterPack -AcceptAnimaLicense:$AcceptAnimaLicense -PlanOnly:$PlanOnly -Json:$Json
exit $exitCode
