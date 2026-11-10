from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');s=(R/'crates/calchem-opt/src/irc.rs').read_text()
s=s.replace('    let ds = opts.step_size_bohr;\n    let radius = 0.5 * ds;','    let mut ds = opts.step_size_bohr;')
a=s.index('        // Hypersphere center:');b=s.index('        s_acc += ds;',a)
old=s[a:b]
old=old.replace('        // Hypersphere center:', '        let radius=0.5*ds;\n        // Hypersphere center:')
old=old.replace('        let g_k_norm = dot_prod(&g_k, &g_k).sqrt();','''        let g_k_norm = dot_prod(&g_k, &g_k).sqrt();
        let u:Vec<_>=q_trial.iter().zip(&q_star).map(|(q,c)|(q-c)/radius).collect();
        let radial=dot_prod(&g_k,&u);
        let tangent_norm=g_k.iter().zip(&u).map(|(g,u)|(g-radial*u).powi(2)).sum::<f64>().sqrt();
        let previous_e=points.last().map(|p:&IrcPoint|p.energy_hartree).unwrap_or(e_ts);
        if tangent_norm<=opts.convergence_tol && e_k<=previous_e+1e-8 {
            accepted=Some((q_trial,e_k,g_k,pt_mol,g_k_norm));break;
        }
        ds*=0.5;
''')
s=s[:a]+'''        let mut accepted=None;
        for _retry in 0..8 {
'''+old+'''        }
        let Some((q_trial,e_k,g_k,pt_mol,g_k_norm))=accepted else {
            // No accepted sphere endpoint: report an incomplete branch, never a minimum.
            break;
        };
'''+s[b:]
# Don't report reaction barriers as final thermochemistry for incomplete branches.
s=s.replace('let e_prod = forward.as_ref().map(|f| f.final_energy_hartree);','let e_prod = forward.as_ref().filter(|f|f.converged_to_minimum).map(|f| f.final_energy_hartree);')
s=s.replace('let e_react = reverse.as_ref().map(|r| r.final_energy_hartree);','let e_react = reverse.as_ref().filter(|r|r.converged_to_minimum).map(|r| r.final_energy_hartree);')
marker='    let masses = get_masses_3n(ts_mol);'
s=s.replace(marker,'''    let dim=3*ts_mol.atoms.len();
    if transition_vector.len()!=dim || transition_vector.iter().any(|v|!v.is_finite())
        || transition_vector.iter().map(|v|v*v).sum::<f64>()<1e-20
        || !opts.step_size_bohr.is_finite() || opts.step_size_bohr<=0.0
        || !opts.convergence_tol.is_finite() || opts.convergence_tol<=0.0
        || opts.max_gs_iter==0 || opts.max_points==0 {
        return Err(CalcError::InvalidMolecule("IRC requires a nonzero finite 3N mode, positive step/tolerance and nonzero iteration budgets".into()));
    }
'''+marker)
(R/'build_target/runtime/round2_20261110/repair_stage/crates/calchem-opt/src/irc.rs').write_text(s)
