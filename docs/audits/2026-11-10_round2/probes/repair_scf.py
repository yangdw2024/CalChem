from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');S=R/'build_target/runtime/round2_20261110/repair_stage'
changed=[]
for path in [* (R/'crates/calchem-scf/src').glob('*.rs'),R/'crates/calchem-grad/src/lib.rs',R/'crates/calchem-opt/src/lib.rs',R/'crates/calchem-solv/src/scrf.rs']:
    s=path.read_text();old=s
    # All these sites have a BasisSet available; prebuilt user grids remain explicit.
    if path.name in {'rks.rs','uks.rs'} or path.parent.parent.name=='calchem-grad':
        s=s.replace('MolecularGrid::build(mol, grid_level)','MolecularGrid::build_for_basis(mol, basis, grid_level)')
    if path.parent.parent.name=='calchem-opt':
        s=s.replace('MolecularGrid::build(mol, grid_level)','MolecularGrid::build_for_basis(mol, bs, grid_level)')
    if path.name=='scrf.rs':
        s=s.replace('MolecularGrid::build(mol, GridLevel::Fine)','MolecularGrid::build_for_basis(mol, basis, GridLevel::Fine)')
        needle=') -> Result<SolvationResult, CalcError> {'
        # Direct RKS SCRF callers must not bypass the LRC guard.
        a=s.index('pub fn run_scrf_rks');b=s.index('{',s.index('Result<',a))+1
        s=s[:b]+'\n    xc.validate_native()?;'+s[b:]
    if path.parent.name=='src' and path.parent.parent.name=='calchem-scf' and path.name in ['rhf.rs','rks.rs','uhf.rs','uks.rs']:
        # Never fill discarded overlap directions with zero-energy fake MOs.
        s=s.replace('if val <= 0.0 {','if val <= 1e-10 {')
        s=s.replace('Overlap matrix is not positive definite','Overlap matrix is singular or too ill-conditioned for full-rank SCF')
    if path.name=='rks.rs':
        a=s.index('    let fp_density_tol = if n >= 40');b=s.index('    let mut fp_diis',a)
        s=s[:a]+'''    let fp_density_tol = opts.conv_density.min(1e-10);
    let fp_comm_tol = 1e-8;
'''+s[b:]
        s=s.replace('let max_fp_steps = if n >= 40 { 2 } else { 30 };','let max_fp_steps = 50;')
        s=s.replace('let comm_norm2 = frobenius_norm(&comm2).min(comm_norm);','let comm_norm2 = frobenius_norm(&comm2);')
        s=s.replace('let comm_norm = frobenius_norm(&comm);\n        let f_step','let _comm_norm = frobenius_norm(&comm);\n        let f_step')
        s=s.replace('    if !fp_converged {','    {')
        s=s.replace('if comm_norm > 1e-6 {','if !comm_norm.is_finite() || comm_norm > fp_comm_tol {')
        s=s.replace('    let mut fp_converged = false;\n','').replace('            fp_converged = true;\n','')
    if s!=old:
        rel=path.relative_to(R);(S/rel).write_text(s);changed.append(str(rel))
print('\n'.join(changed))
