"""Deterministic, baseline-backed repairs. Run from the workspace; no C: writes."""
from pathlib import Path
ROOT = Path(r'E:\YDW\Trae_Gaussian\build_target\runtime\round2_20261110\repair_stage')
def edit(name, old, new):
    p = ROOT / name
    s = p.read_text(encoding='utf-8-sig')
    assert old in s, (name, old[:100])
    p.write_text(s.replace(old, new), encoding='utf-8')

# A failed independent gas reference must propagate, not reuse polarized energy.
edit('crates/calchem-solv/src/scrf.rs', '''match calchem_scf::rhf(mol, basis, scf_opts) {
        Ok(gas_wf) => gas_wf.total_energy_hartree,
        Err(_) => e_prev - e_solv_elec,
    }''', 'calchem_scf::rhf(mol, basis, scf_opts)?.total_energy_hartree')
edit('crates/calchem-solv/src/scrf.rs', '''match calchem_scf::rks_with_grid(mol, basis, xc, &grid, scf_opts) {
        Ok(gas_wf) => gas_wf.total_energy_hartree,
        Err(_) => e_prev - e_solv_elec,
    }''', 'calchem_scf::rks_with_grid(mol, basis, xc, &grid, scf_opts)?.total_energy_hartree')

# Remove the radius cutoff entirely: the radial mapping covers [0,infinity).
# Element radii determine mapping scale, not support of a diffuse density.
edit('crates/calchem-dft/src/grid.rs', '''                    let r_max = match level {
                        GridLevel::Coarse => 25.0,
                        GridLevel::Medium => 35.0,
                        GridLevel::Fine => 50.0,
                    };
                    if r > r_max {
                        continue;
                    }
''', '')

# Independent fresh projected Hessian certification, no error-to-success fallback.
p = ROOT/'crates/calchem-opt/src/prfo.rs'; s=p.read_text()
a=s.index('            // Rigorous Transition State Certification:')
b=s.index('            return Ok(TsResult {', a)
s=s[:a]+'''            // Bofill is a search model, never a certificate of Hessian inertia.
            let exact_hessian = compute_initial_hessian(
                &cur_mol, basis_name, opts.reference, opts.grid_level, &opts.scf_options,
            )?;
            let empty_apt = Matrix2D::zeros(3, dim);
            let vib = calchem_vib::analyze_vibrations(&cur_mol, &exact_hessian, &empty_apt)?;
            // Curvatures within 10 cm^-1 are unresolved at the numerical Hessian accuracy.
            // Reject both flat modes and saddles with more than one negative curvature.
            let imag_modes: Vec<_> = vib.modes.iter().filter(|m| m.frequency_cm < -10.0).collect();
            let is_true_ts = imag_modes.len() == 1
                && vib.modes.iter().all(|m| m.frequency_cm.abs() > 10.0);
            let imag_freq = vib.modes.first().map(|m| m.frequency_cm).unwrap_or(0.0);
            let certified_vector = vib.modes.first()
                .map(|m| m.displacements.clone()).unwrap_or(cur_ts_vec);

'''+s[b:]
s=s.replace('    compute_cartesian_hessian_and_apt, FreqOptions, VibMethod, HESSIAN_EVAL_TO_INV_CM,','    compute_cartesian_hessian_and_apt, FreqOptions, VibMethod,')
s=s.replace('transition_vector: cur_ts_vec,','transition_vector: certified_vector,')
p.write_text(s,encoding='utf-8')

# Quartets are generated lazily; the gradient formerly materialized O(n_shell^4).
edit('crates/calchem-int/src/symmetry.rs', '    let mut out = Vec::new();\n    for i in 0..n_shells {', '    shell_quartets(n_shells).collect()\n}\n\n/// Lazy canonical quartets: constant auxiliary storage.\npub fn shell_quartets(n_shells: usize) -> impl Iterator<Item = [usize; 4]> {\n    (0..n_shells).flat_map(move |i| (i..n_shells).flat_map(move |j| {\n        (i..n_shells).flat_map(move |k| {\n            let start = if k == i { j } else { k };\n            (start..n_shells).map(move |l| [i, j, k, l])\n        })\n    }))\n}\n\n/* Historical enumeration retained only in the baseline snapshot.\n    for i in 0..n_shells {')
edit('crates/calchem-int/src/symmetry.rs', '    out\n}\n\n/// Returns', '    out\n} */\n\n/// Returns')
edit('crates/calchem-grad/src/lib.rs','use calchem_int::symmetry::unique_shell_quartets;','use calchem_int::symmetry::shell_quartets;')
edit('crates/calchem-grad/src/lib.rs','let canonical_quartets = unique_shell_quartets(basis.shells.len());','let canonical_quartets = shell_quartets(basis.shells.len());')
edit('crates/calchem-grad/src/lib.rs','canonical_quartets\n        .into_par_iter()','canonical_quartets\n        .par_bridge()')

# Physical constant comment follows actual APT units.
edit('crates/calchem-vib/src/modes.rs','// IR intensity: I = 42.2561 * |dmu / dQ|^2 (km/mol), with dmu/dQ_p = sum_i P_{p, i} * U_i','// IR intensity in km/mol: APT in atomic units, Q in sqrt(amu)*Bohr.')
print('Applied first deterministic repair set')

