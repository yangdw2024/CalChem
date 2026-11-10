$ErrorActionPreference='Stop'
. "$PSScriptRoot\env.ps1"
$root='E:\YDW\Trae_Gaussian'
$before=Get-Content -LiteralPath "$auditRoot\evidence\source_before.json" -Raw | ConvertFrom-Json
$changed=@();$missing=@()
foreach($entry in $before.PSObject.Properties) {
    $path=Join-Path $root $entry.Name
    if(-not (Test-Path -LiteralPath $path -PathType Leaf)) {$missing+=$entry.Name;continue}
    $hash=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    if($hash -ne $entry.Value.sha256) {$changed+=@{path=$entry.Name;before=$entry.Value.sha256;after=$hash}}
}
@{checked=@($before.PSObject.Properties).Count;changed=$changed;missing=$missing} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$auditRoot\evidence\integrity-after.json" -Encoding utf8
git status --porcelain=v1 -uall | Set-Content -LiteralPath "$auditRoot\evidence\git-status-after.txt" -Encoding utf8
$externalFiles=@()
foreach($folder in @('external_alignment','external_followup','external_extra')) {
    $externalFiles+=Get-ChildItem -LiteralPath (Join-Path $auditRuntime $folder) -Recurse -File | Where-Object {$_.Extension -in @('.gjf','.inp','.out') -or $_.Name -eq 'launcher.txt'}
}
$externalFiles | ForEach-Object {@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash;bytes=$_.Length}} | ConvertTo-Json -Depth 4 | Set-Content "$auditRoot\evidence\external-artifacts.json" -Encoding utf8
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
if(Test-Path -LiteralPath "$auditRoot\evidence\external_raw.zip") {throw 'external_raw.zip already exists; do not overwrite audit evidence.'}
$zip=[System.IO.Compression.ZipFile]::Open("$auditRoot\evidence\external_raw.zip",[System.IO.Compression.ZipArchiveMode]::Create)
try {foreach($f in $externalFiles) {[System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$f.FullName,$f.FullName.Substring($auditRuntime.Length+1).Replace('\','/'),[System.IO.Compression.CompressionLevel]::Optimal) | Out-Null}}finally{$zip.Dispose()}
Get-FileHash D:\G16W\g16.exe,E:\ORCA\software\ORCA6.1.1\orca.exe | Format-List | Out-File "$auditRoot\evidence\external-executable-hashes.txt"
Get-Content "$auditRoot\evidence\integrity-after.json"
