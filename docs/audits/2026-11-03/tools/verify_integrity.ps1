$ErrorActionPreference = 'Stop'
$audit = Split-Path -Parent $PSScriptRoot
$repo = (Resolve-Path (Join-Path $audit '..\..\..')).Path
$before = @(Import-Csv -LiteralPath (Join-Path $audit 'evidence\source-hashes-before.csv'))
$changed = @()
foreach ($row in $before) {
    if (-not (Test-Path -LiteralPath $row.Path) -or (Get-FileHash -LiteralPath $row.Path -Algorithm SHA256).Hash -ne $row.Hash) { $changed += $row.Path }
}
$design = Import-Csv -LiteralPath (Join-Path $audit 'evidence\design-hash-before.csv')
$designUnchanged = (Get-FileHash -LiteralPath $design.Path -Algorithm SHA256).Hash -eq $design.Hash
$manifestPath = Get-Content -LiteralPath (Join-Path $audit 'evidence\backup-manifest-path.txt')
$entries = @(Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json)
$badBackups = @()
foreach ($entry in $entries) {
    if ((Get-FileHash -LiteralPath $entry.Backup -Algorithm SHA256).Hash -ne $entry.SHA256) { $badBackups += $entry.Backup }
}
$parseErrors = @()
foreach ($file in Get-ChildItem -LiteralPath $PSScriptRoot -Filter *.ps1) {
    $tokens = $null
    $errorsFound = $null
    [System.Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$tokens,[ref]$errorsFound) | Out-Null
    foreach ($problem in $errorsFound) { $parseErrors += "$($file.Name): $($problem.Message)" }
}
$missingLinks = @()
foreach ($file in Get-ChildItem -LiteralPath $audit -Filter *.md) {
    $content = Get-Content -Raw -LiteralPath $file.FullName
    foreach ($match in [regex]::Matches($content,'\]\(([^)]+\.md)\)')) {
        if (-not (Test-Path -LiteralPath (Join-Path $audit $match.Groups[1].Value))) { $missingLinks += $match.Groups[1].Value }
    }
}
Push-Location $repo
try {
    git diff --quiet HEAD
    $diffExit = $LASTEXITCODE
    git status --short --branch | Set-Content (Join-Path $audit 'evidence\git-status-after.txt') -Encoding utf8
} finally { Pop-Location }
$outcome = [ordered]@{
    Date='2026-11-03';SourceFilesChecked=$before.Count;ChangedSourceFiles=$changed;
    ExistingStage3DesignUnchanged=$designUnchanged;BackupFilesChecked=$entries.Count;DamagedBackups=$badBackups;
    PowerShellParseErrors=$parseErrors;MissingDocumentLinks=$missingLinks;TrackedDiffExitCode=$diffExit;
    Note='Restore was previewed with WhatIf only. No source restoration or deletion was performed.'
}
$outcome | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $audit 'evidence\integrity-after.json') -Encoding utf8
$outcome | ConvertTo-Json -Depth 5
if ($changed.Count -or -not $designUnchanged -or $badBackups.Count -or $parseErrors.Count -or $missingLinks.Count -or $diffExit -ne 0) { throw 'Final integrity check failed.' }
