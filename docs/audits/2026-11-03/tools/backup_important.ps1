[CmdletBinding()]
param([Parameter(Mandatory=$true)][string[]]$Paths)
$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $audit '..\..\..')).Path.TrimEnd('\')
$resolved = foreach ($item in $Paths) {
    $file = Get-Item -LiteralPath $item
    if ($file.PSIsContainer -or -not $file.FullName.StartsWith($repo + '\',[StringComparison]::OrdinalIgnoreCase)) {
        throw "Backup accepts only explicit files inside this repository: $item"
    }
    if ($file.FullName.StartsWith($audit + '\',[StringComparison]::OrdinalIgnoreCase) -or $file.FullName.StartsWith($repo + '\.git\',[StringComparison]::OrdinalIgnoreCase)) {
        throw "Do not recursively back up audit artifacts or Git internals: $item"
    }
    $file
}
$batch = Join-Path $audit ('rollback\backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
if (Test-Path -LiteralPath $batch) { throw "Backup batch already exists: $batch" }
New-Item -ItemType Directory -Path $batch | Out-Null
$manifest = foreach ($file in $resolved) {
    $relative = $file.FullName.Substring($repo.Length + 1)
    $backup = Join-Path $batch ('files\' + $relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
    $before = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    Copy-Item -LiteralPath $file.FullName -Destination $backup
    $after = (Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash
    if ($before -ne $after) { throw "Backup hash mismatch: $relative" }
    [pscustomobject]@{Original=$file.FullName;Backup=$backup;SHA256=$before;LastWriteTimeUtc=$file.LastWriteTimeUtc.ToString('o')}
}
@($manifest) | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $batch 'manifest.json') -Encoding utf8
Write-Output (Join-Path $batch 'manifest.json')
