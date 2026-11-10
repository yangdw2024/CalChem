. "$PSScriptRoot\env.ps1"
$argsList=@('--edition=2021','-C','opt-level=3','-C','panic=abort','-L','dependency=E:\YDW\Trae_Gaussian\build_target\release\deps')
foreach($crate in @('calchem_model','calchem_basis','calchem_int','calchem_dft','calchem_opt')) {
    $lib=Get-ChildItem E:\YDW\Trae_Gaussian\build_target\release\deps -Filter "lib$crate-*.rlib" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $argsList+=@('--extern',"$crate=$($lib.FullName)")
}
$argsList+=@("$PSScriptRoot\core_probe.rs",'-o',"$auditRuntime\core_probe.exe")
& rustc @argsList 2> "$auditRoot\evidence\core-probe-build.log"
if($LASTEXITCODE -ne 0) {Get-Content "$auditRoot\evidence\core-probe-build.log";exit $LASTEXITCODE}
& "$auditRuntime\core_probe.exe" *> "$auditRoot\evidence\core-probe.jsonl"
"exit=$LASTEXITCODE" | Set-Content "$auditRoot\evidence\core-probe.exit.txt"
Get-Content "$auditRoot\evidence\core-probe.jsonl"
