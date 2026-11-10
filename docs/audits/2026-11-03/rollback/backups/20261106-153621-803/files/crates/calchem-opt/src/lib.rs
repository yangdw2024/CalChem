use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient_with_grid, uhf_gradient, uks_gradient_with_grid};
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_scf::{rhf, rks_with_grid, uhf, uks_with_grid};
use calchem_tensor::Matrix2D;
use serde::{Deserialize, Serialize};

/// SCF / Kohn-Sham reference used to evaluate energies and analytical gradients during optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Reference {
    /// RHF for closed-shell singlets, UHF otherwise.
    #[default]
    Auto,
    Rhf,
    Uhf,
    /// Restricted Kohn-Sham DFT with the specified XC functional.
    Rks(XcFunctional),
    /// Unrestricted Kohn-Sham DFT with the specified XC functional.
    Uks(XcFunctional),
    /// Closed-shell RKS for singlets, open-shell UKS otherwise.
    DftAuto(XcFunctional),
}

impl Reference {
    pub fn from_method_and_xc(method: &str, xc: Option<&str>) -> Result<Self, CalcError> {
        let m = method.trim().to_ascii_lowercase();
        let resolve_xc = |default_xc: &str| -> Result<XcFunctional, CalcError> {
            XcFunctional::from_name(xc.unwrap_or(default_xc))
        };
        match m.as_str() {
            "rhf" => Ok(Self::Rhf),
            "uhf" => Ok(Self::Uhf),
            "hf" | "auto" => Ok(Self::Auto),
            "rks" => Ok(Self::Rks(resolve_xc("b3lyp")?)),
            "uks" => Ok(Self::Uks(resolve_xc("b3lyp")?)),
            "dft" => Ok(Self::DftAuto(resolve_xc("b3lyp")?)),
            other => {
                let func = XcFunctional::from_name(other)?;
                Ok(Self::DftAuto(func))
            }
        }
    }

    fn resolve(self, mol: &Molecule) -> Reference {
        match self {
            Reference::Auto => {
                if mol.multiplicity == 1 {
                    Reference::Rhf
                } else {
                    Reference::Uhf
                }
            }
            Reference::DftAuto(xc) => {
                if mol.multiplicity == 1 {
                    Reference::Rks(xc)
                } else {
                    Reference::Uks(xc)
                }
            }
            r => r,
        }
    }
}

fn scf_energy(
    mol: &Molecule,
    bs: &BasisSet,
    r: Reference,
    grid_level: GridLevel,
    o: &ScfOptions,
) -> Result<f64, CalcError> {
    match r.resolve(mol) {
        Reference::Uhf => Ok(uhf(mol, bs, o)?.total_energy_hartree),
        Reference::Rhf | Reference::Auto => Ok(rhf(mol, bs, o)?.total_energy_hartree),
        Reference::Rks(xc) => {
            let grid = MolecularGrid::build(mol, grid_level);
            Ok(rks_with_grid(mol, bs, xc, &grid, o)?.total_energy_hartree)
        }
        Reference::Uks(xc) | Reference::DftAuto(xc) => {
            let grid = MolecularGrid::build(mol, grid_level);
            Ok(uks_with_grid(mol, bs, xc, &grid, o)?.total_energy_hartree)
        }
    }
}

fn scf_energy_gradient(
    mol: &Molecule,
    bs: &BasisSet,
    r: Reference,
    grid_level: GridLevel,
    o: &ScfOptions,
) -> Result<(f64, Vec<[f64; 3]>), CalcError> {
    match r.resolve(mol) {
        Reference::Uhf => {
            let wf = uhf(mol, bs, o)?;
            Ok((wf.total_energy_hartree, uhf_gradient(mol, bs, &wf)?))
        }
        Reference::Rhf | Reference::Auto => {
            let wf = rhf(mol, bs, o)?;
            Ok((wf.total_energy_hartree, rhf_gradient(mol, bs, &wf)?))
        }
        Reference::Rks(xc) => {
            let grid = MolecularGrid::build(mol, grid_level);
            let wf = rks_with_grid(mol, bs, xc, &grid, o)?;
            let g = rks_gradient_with_grid(mol, bs, xc, &grid, &wf)?;
            Ok((wf.total_energy_hartree, g))
        }
        Reference::Uks(xc) | Reference::DftAuto(xc) => {
            let grid = MolecularGrid::build(mol, grid_level);
            let wf = uks_with_grid(mol, bs, xc, &grid, o)?;
            let g = uks_gradient_with_grid(mol, bs, xc, &grid, &wf)?;
            Ok((wf.total_energy_hartree, g))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptOptions {
    pub max_steps: usize,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
    pub max_step_bohr: f64,
    pub scf_options: ScfOptions,
    pub c1: f64,
    pub backtrack_factor: f64,
    pub max_line_searches: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
}

impl Default for OptOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-10;
        Self {
            max_steps: 100,
            max_force: 1e-4,
            rms_force: 5e-5,
            max_displacement: 1e-3,
            max_step_bohr: 0.2,
            scf_options: scf_opts,
            c1: 1e-4,
            backtrack_factor: 0.5,
            max_line_searches: 20,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptStep {
    pub step: usize,
    pub energy_hartree: f64,
    pub max_force: f64,
    pub rms_force: f64,
    pub max_displacement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptResult {
    pub molecule: Molecule,
    pub energy_hartree: f64,
    pub gradient: Vec<[f64; 3]>,
    pub steps: usize,
    pub converged: bool,
    pub history: Vec<OptStep>,
}

fn coords_from_mol(mol: &Molecule) -> Vec<f64> {
    let mut coords = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        coords.extend_from_slice(&atom.position_bohr);
    }
    coords
}

fn mol_with_coords(base_mol: &Molecule, coords: &[f64]) -> Molecule {
    let mut mol = base_mol.clone();
    for (i, atom) in mol.atoms.iter_mut().enumerate() {
        atom.position_bohr = [coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]];
    }
    mol
}

fn mat_vec_mul(mat: &Matrix2D, vec: &[f64]) -> Vec<f64> {
    let n = vec.len();
    let mut res = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += mat.get(i, j) * vec[j];
        }
        res[i] = sum;
    }
    res
}

fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Optimizes the molecular geometry using the requested HF or DFT method and basis set.
/// Returns the optimized `Molecule`.
pub fn optimize_geometry(
    mol: &Molecule,
    basis: &BasisSet,
    method: &str,
) -> Result<Molecule, CalcError> {
    let reference = Reference::from_method_and_xc(method, None)?;
    let opts = OptOptions {
        reference,
        ..OptOptions::default()
    };
    let res = optimize_geometry_with_options(mol, &basis.name, &opts)?;
    Ok(res.molecule)
}

/// Optimizes molecular geometry with custom optimizer options.
pub fn optimize_geometry_with_options(
    mol: &Molecule,
    basis_name: &str,
    opts: &OptOptions,
) -> Result<OptResult, CalcError> {
    let n_atoms = mol.atoms.len();
    if n_atoms == 0 {
        return Err(CalcError::InvalidMolecule("No atoms in molecule".into()));
    }

    if n_atoms == 1 {
        let bs = BasisSet::from_name(basis_name, mol)?;
        let e = scf_energy(mol, &bs, opts.reference, opts.grid_level, &opts.scf_options)?;
        return Ok(OptResult {
            molecule: mol.clone(),
            energy_hartree: e,
            gradient: vec![[0.0, 0.0, 0.0]],
            steps: 0,
            converged: true,
            history: vec![OptStep {
                step: 0,
                energy_hartree: e,
                max_force: 0.0,
                rms_force: 0.0,
                max_displacement: 0.0,
            }],
        });
    }

    let dim = n_atoms * 3;
    let mut x = coords_from_mol(mol);

    let mut h_inv = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        h_inv.set(i, i, 0.7);
    }

    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut cached_eg: Option<(f64, Vec<[f64; 3]>)> = None;

    for step in 0..=opts.max_steps {
        let cur_mol = mol_with_coords(mol, &x);
        let (current_e, current_g_3d) = if let Some(eg) = cached_eg.take() {
            eg
        } else {
            let cur_basis = BasisSet::from_name(basis_name, &cur_mol)?;
            scf_energy_gradient(
                &cur_mol,
                &cur_basis,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?
        };

        let mut g = Vec::with_capacity(dim);
        for atom_g in &current_g_3d {
            g.extend_from_slice(atom_g);
        }

        let max_force = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let sum_sq_g: f64 = g.iter().map(|v| v * v).sum();
        let rms_force = (sum_sq_g / (dim as f64)).sqrt();

        history.push(OptStep {
            step,
            energy_hartree: current_e,
            max_force,
            rms_force,
            max_displacement: last_displacement,
        });

        if step == 0 {
            if max_force < opts.max_force && rms_force < opts.rms_force {
                return Ok(OptResult {
                    molecule: cur_mol,
                    energy_hartree: current_e,
                    gradient: current_g_3d,
                    steps: step,
                    converged: true,
                    history,
                });
            }
        } else if max_force < opts.max_force
            && rms_force < opts.rms_force
            && last_displacement < opts.max_displacement
        {
            return Ok(OptResult {
                molecule: cur_mol,
                energy_hartree: current_e,
                gradient: current_g_3d,
                steps: step,
                converged: true,
                history,
            });
        }

        if step == opts.max_steps {
            break;
        }

        let mut p = mat_vec_mul(&h_inv, &g);
        for v in p.iter_mut() {
            *v = -*v;
        }

        let mut g_dot_p = dot_prod(&g, &p);
        if g_dot_p >= 0.0 {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
            p = g.iter().map(|v| -0.7 * v).collect();
            g_dot_p = dot_prod(&g, &p);
        }

        let p_norm = dot_prod(&p, &p).sqrt();
        if p_norm > opts.max_step_bohr && p_norm > 1e-12 {
            let scale = opts.max_step_bohr / p_norm;
            for v in p.iter_mut() {
                *v *= scale;
            }
            g_dot_p *= scale;
        }

        let mut alpha = 1.0;
        let mut accepted = false;
        let mut x_new = x.clone();

        for _ls in 0..opts.max_line_searches {
            for i in 0..dim {
                x_new[i] = x[i] + alpha * p[i];
            }

            let trial_mol = mol_with_coords(mol, &x_new);
            if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                if let Ok(e_trial) = scf_energy(
                    &trial_mol,
                    &trial_bs,
                    opts.reference,
                    opts.grid_level,
                    &opts.scf_options,
                ) {
                    if e_trial <= current_e + opts.c1 * alpha * g_dot_p {
                        accepted = true;
                        break;
                    }
                }
            }
            alpha *= opts.backtrack_factor;
        }

        if !accepted {
            let mut is_diagonal = true;
            for i in 0..dim {
                for j in 0..dim {
                    if i != j && h_inv.get(i, j).abs() > 1e-10 {
                        is_diagonal = false;
                        break;
                    }
                }
                if !is_diagonal {
                    break;
                }
            }

            if !is_diagonal {
                for i in 0..dim {
                    for j in 0..dim {
                        h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                    }
                }
                let small_step = 0.05;
                for i in 0..dim {
                    x_new[i] = x[i] - small_step * g[i];
                }
                let trial_mol = mol_with_coords(mol, &x_new);
                if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                    if scf_energy(
                        &trial_mol,
                        &trial_bs,
                        opts.reference,
                        opts.grid_level,
                        &opts.scf_options,
                    )
                    .is_ok()
                    {
                        accepted = true;
                    }
                }
            }

            if !accepted {
                return Err(CalcError::Internal(format!(
                    "Geometry optimization line search stalled at step {step}"
                )));
            }
        }

        let mut s = vec![0.0; dim];
        for i in 0..dim {
            s[i] = x_new[i] - x[i];
        }

        let mut max_atom_disp = 0.0;
        for a in 0..n_atoms {
            let dx = s[3 * a];
            let dy = s[3 * a + 1];
            let dz = s[3 * a + 2];
            let disp = (dx * dx + dy * dy + dz * dz).sqrt();
            if disp > max_atom_disp {
                max_atom_disp = disp;
            }
        }
        last_displacement = max_atom_disp;

        let new_mol = mol_with_coords(mol, &x_new);
        let new_bs = BasisSet::from_name(basis_name, &new_mol)?;
        let (new_e, new_g_3d) = scf_energy_gradient(
            &new_mol,
            &new_bs,
            opts.reference,
            opts.grid_level,
            &opts.scf_options,
        )?;
        let mut g_new = Vec::with_capacity(dim);
        for ag in &new_g_3d {
            g_new.extend_from_slice(ag);
        }
        cached_eg = Some((new_e, new_g_3d));

        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        let s_dot_y = dot_prod(&s, &y);
        if s_dot_y > 1e-10 {
            let rho = 1.0 / s_dot_y;

            let mut v = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    v.set(i, j, delta_ij - rho * s[i] * y[j]);
                }
            }

            let mut temp = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += v.get(i, k) * h_inv.get(k, j);
                    }
                    temp.set(i, j, sum);
                }
            }

            let mut h_next = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0;
                    for k in 0..dim {
                        sum += temp.get(i, k) * v.get(j, k);
                    }
                    h_next.set(i, j, sum + rho * s[i] * s[j]);
                }
            }
            h_inv = h_next;
        } else {
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
        }

        x = x_new;
    }

    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}
