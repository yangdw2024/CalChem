$ErrorActionPreference='Stop'
$root = 'E:\YDW\Trae_Gaussian'
$audit = Join-Path $root 'docs\audits\2026-11-10_round2'
if(Test-Path -LiteralPath "$audit\rollback\source_snapshot.zip") {throw 'Snapshot already exists. Preserve the immutable audit baseline.'}
$files = @()
foreach($part in @('crates','calchem','tests','tools','config','.github')) {
    $files += Get-ChildItem -LiteralPath (Join-Path $root $part) -File -Recurse | Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' }
}
$files += Get-ChildItem -LiteralPath $root -File | Where-Object { $_.Extension -in @('.toml','.lock','.md','.yaml','.yml','.py') }
$manifest = [ordered]@{}
foreach($f in ($files | Sort-Object FullName -Unique)) {
    $rel=$f.FullName.Substring($root.Length+1).Replace('\','/')
    $manifest[$rel]=@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLower()}
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$audit\evidence\source_before.json" -Encoding utf8
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[System.IO.Compression.ZipFile]::Open("$audit\rollback\source_snapshot.zip",[System.IO.Compression.ZipArchiveMode]::Create)
try { foreach($rel in $manifest.Keys) { [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(Join-Path $root $rel),$rel,[System.IO.Compression.CompressionLevel]::Optimal) | Out-Null } } finally { $zip.Dispose() }
git status --porcelain=v1 -uall | Set-Content -LiteralPath "$audit\evidence\git-status-before.txt" -Encoding utf8
git log -20 --date=iso-strict '--format=%H %ad %s' | Set-Content -LiteralPath "$audit\evidence\git-log.txt" -Encoding utf8
rustc -Vv | Set-Content -LiteralPath "$audit\evidence\toolchain.txt" -Encoding utf8
Write-Output "Snapshot files: $($manifest.Count)"
Get-Item -LiteralPath "$audit\rollback\source_snapshot.zip" | Select-Object FullName,Length

