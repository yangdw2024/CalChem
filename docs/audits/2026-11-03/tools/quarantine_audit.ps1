[CmdletBinding(SupportsShouldProcess=$true)]
param()
$ErrorActionPreference = 'Stop'
$audit = (Resolve-Path (Split-Path -Parent $PSScriptRoot)).Path
$expected = 'E:\YDW\Trae_Gaussian\docs\audits\2026-11-03'
if ($audit -ne $expected) { throw 'Unexpected audit directory; refusing to move it.' }
$archiveRoot = 'E:\YDW\_CalChemAuditArchive'
$destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot ('2026-11-03-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))))
if (-not $destination.StartsWith($archiveRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $destination)) {
    throw 'Invalid or existing quarantine destination.'
}
if ($PSCmdlet.ShouldProcess($audit,"Move audit and build outputs into $destination without deletion")) {
    New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
    Move-Item -LiteralPath $audit -Destination $destination
    Write-Output $destination
}
