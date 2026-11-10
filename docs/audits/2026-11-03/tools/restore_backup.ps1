[CmdletBinding(SupportsShouldProcess=$true)]
param([Parameter(Mandatory=$true)][string]$ManifestPath)
$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $audit '..\..\..')).Path.TrimEnd('\')
$manifestFile = (Get-Item -LiteralPath $ManifestPath).FullName
$allowed = Join-Path $audit 'rollback\backups\'
if (-not $manifestFile.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest must be in this audit backup directory.' }
$entries = @(Get-Content -Raw -LiteralPath $manifestFile | ConvertFrom-Json)
foreach ($entry in $entries) {
    $target = [IO.Path]::GetFullPath($entry.Original)
    $source = [IO.Path]::GetFullPath($entry.Backup)
    if (-not $target.StartsWith($repo + '\',[StringComparison]::OrdinalIgnoreCase) -or
        $target.StartsWith($audit + '\',[StringComparison]::OrdinalIgnoreCase) -or
        $target.StartsWith($repo + '\.git\',[StringComparison]::OrdinalIgnoreCase) -or
        -not $source.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Restore path escaped its intended scope.'
    }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $entry.SHA256) { throw "Backup is damaged: $source" }
}
$safety = Join-Path $audit ('rollback\before-restore\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
foreach ($entry in $entries) {
    if ($PSCmdlet.ShouldProcess($entry.Original,'Back up current file and restore verified audit snapshot')) {
        if (Test-Path -LiteralPath $entry.Original) {
            $relative = $entry.Original.Substring($repo.Length + 1)
            $saved = Join-Path $safety $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
            Copy-Item -LiteralPath $entry.Original -Destination $saved
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $entry.Original) -Force | Out-Null
        Copy-Item -LiteralPath $entry.Backup -Destination $entry.Original -Force
        if ((Get-FileHash -LiteralPath $entry.Original -Algorithm SHA256).Hash -ne $entry.SHA256) { throw 'Restore verification failed.' }
    }
}
