from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');S=R/'build_target/runtime/round2_20261110/repair_stage'
changed=[]
def save(name,s): (S/name).write_text(s,encoding='utf-8');changed.append(name)
name='crates/calchem-solv/src/scrf.rs';s=(R/name).read_text()
a=s.index('    if is_open_shell {',s.index('pub fn run_scrf('));b=s.index('\n    match m.as_str()',a)
s=s[:a]+'''    if is_open_shell {
        return Err(CalcError::UnsupportedMethod(
            "Self-consistent open-shell PCM is not implemented. A post-SCF correction to a gas UHF/UKS density must not be reported as SCRF; use an external backend.".into()
        ));
    }
'''+s[b:]
a=s.index('        "rks" | "dft" | "b3lyp"',s.index('pub fn run_scrf('));b=s.index('\n        other =>',a)
s=s[:a]+'''        "rks" | "dft" => {
            let func=xc.unwrap_or(XcFunctional::B3lyp);
            run_scrf_rks(mol,basis,scf_opts,opts,func)
        }
        "b3lyp" | "b3lyp5" | "pbe" | "blyp" | "svwn5" | "pbe0" | "camb3lyp"
        | "cam-b3lyp" | "wb97xd" | "wb97x-d" | "wb97x" => {
            let func=XcFunctional::from_name(method)?;
            if xc.is_some_and(|x| x!=func) {return Err(CalcError::UnsupportedMethod("Conflicting method and XC identities".into()));}
            run_scrf_rks(mol,basis,scf_opts,opts,func)
        }'''+s[b:]
# Include the electronic polarization cost in the electrostatic contribution.
for fn,next_fn in [('run_scrf_rhf','run_scrf_rks'),('run_scrf_rks','run_scrf(')]:
    a=s.index('pub fn '+fn);b=s.index('pub fn '+next_fn,a+1)
    part=s[a:b].replace('delta_g_elec_kcal_mol: e_solv_elec * HARTREE_TO_KCAL_MOL,','delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,')
    s=s[:a]+part+s[b:]
save(name,s)

name='calchem/backends/native.py';s=(R/name).read_text();a=s.index('    def irc(');head=s[:a];tail=s[a:]
tail=tail.replace('            fwd_barrier = irc_res.forward.barrier_kcal_mol','            fwd_barrier = irc_res.barrier_forward_kcal_mol')
tail=tail.replace('            rev_barrier = irc_res.reverse.barrier_kcal_mol','            rev_barrier = irc_res.barrier_reverse_kcal_mol')
tail=tail.replace('        if irc_res.forward is not None and irc_res.reverse is not None:\n            reaction_energy = rev_barrier - fwd_barrier','''        reaction_energy = irc_res.reaction_energy_kcal_mol
        branches = [b for b in (irc_res.forward, irc_res.reverse) if b is not None]
        path_complete = bool(branches) and all(b.converged_to_minimum for b in branches)''')
tail=tail.replace('            status="succeeded",','            status="succeeded" if path_complete else "not_converged",',1)
tail=tail.replace('            scf_converged=True,','            scf_converged=True,',1)
tail=tail.replace('            extra={','            extra={\n                "irc_endpoints_converged": path_complete,',1)
save(name,head+tail)

name='calchem/workflows/opv.py';s=(R/name).read_text();a=s.index('    def reaction_barrier(');b=s.index('\n    def ',a+5);part=s[a:b]
part=part.replace('            if irc_res.status == "succeeded":','            if irc_res.status in ("succeeded", "not_converged"):')
part=part.replace('        summary = (','        def _barrier_text(value):\n            return "unavailable (IRC endpoints incomplete)" if value is None else f"{value:8.2f} kcal/mol"\n\n        summary = (')
part=part.replace('{fwd_barrier or 0.0:8.2f} kcal/mol','{_barrier_text(fwd_barrier)}').replace('{rev_barrier or 0.0:8.2f} kcal/mol','{_barrier_text(rev_barrier)}').replace('{rxn_energy or 0.0:8.2f} kcal/mol','{_barrier_text(rxn_energy)}')
save(name,s[:a]+part+s[b:])
print('\n'.join(changed))
