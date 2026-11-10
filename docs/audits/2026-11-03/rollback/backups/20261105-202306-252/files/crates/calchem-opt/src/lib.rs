use calchem_basis::BasisSet;
use calchem_grad::rhf_gradient;
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_scf::rhf;
use calchem_tensor::Matrix2D;
use serde::{Deserialize, Serialize};

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

/// Optimizes the molecular geometry using the requested method and basis set.
/// Returns the optimized `Molecule`.
pub fn optimize_geometry(
    mol: &Molecule,
    basis: &BasisSet,
    method: &str,
) -> Result<Molecule, CalcError> {
    if !method.trim().eq_ignore_ascii_case("rhf") {
        return Err(CalcError::UnsupportedMethod(format!(
            "Only RHF geometry optimization is currently supported, got '{method}'"
        )));
    }
    let res = optimize_geometry_with_options(mol, &basis.name, &OptOptions::default())?;
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

    // Single atom case: already stationary
    if n_atoms == 1 {
        let bs = BasisSet::from_name(basis_name, mol)?;
        let wf = rhf(mol, &bs, &opts.scf_options)?;
        return Ok(OptResult {
            molecule: mol.clone(),
            energy_hartree: wf.total_energy_hartree,
            gradient: vec![[0.0, 0.0, 0.0]],
            steps: 0,
            converged: true,
            history: vec![OptStep {
                step: 0,
                energy_hartree: wf.total_energy_hartree,
                max_force: 0.0,
                rms_force: 0.0,
                max_displacement: 0.0,
            }],
        });
    }

    let dim = n_atoms * 3;
    let mut x = coords_from_mol(mol);

    // Initial inverse Hessian approximation H_0 = 0.7 * I
    let mut h_inv = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        h_inv.set(i, i, 0.7);
    }

    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut current_g_3d;
    let mut current_e;

    for step in 0..=opts.max_steps {
        let cur_mol = mol_with_coords(mol, &x);
        let cur_basis = BasisSet::from_name(basis_name, &cur_mol)?;
        let wf = rhf(&cur_mol, &cur_basis, &opts.scf_options)?;
        current_e = wf.total_energy_hartree;
        current_g_3d = rhf_gradient(&cur_mol, &cur_basis, &wf)?;

        // Flatten gradient
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

        // Convergence check:
        // Initial point: displacement is treated as zero.
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

        // Calculate search direction p = - H * g
        let mut p = mat_vec_mul(&h_inv, &g);
        for v in p.iter_mut() {
            *v = -*v;
        }

        let mut g_dot_p = dot_prod(&g, &p);
        if g_dot_p >= 0.0 {
            // Not a descent direction: reset H_inv
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
            p = g.iter().map(|v| -0.7 * v).collect();
            g_dot_p = dot_prod(&g, &p);
        }

        // Cap maximum step size
        let p_norm = dot_prod(&p, &p).sqrt();
        if p_norm > opts.max_step_bohr && p_norm > 1e-12 {
            let scale = opts.max_step_bohr / p_norm;
            for v in p.iter_mut() {
                *v *= scale;
            }
            g_dot_p *= scale;
        }

        // Armijo backtracking line search
        let mut alpha = 1.0;
        let mut accepted = false;
        let mut x_new = x.clone();

        for _ls in 0..opts.max_line_searches {
            for i in 0..dim {
                x_new[i] = x[i] + alpha * p[i];
            }

            let trial_mol = mol_with_coords(mol, &x_new);
            if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                if let Ok(trial_wf) = rhf(&trial_mol, &trial_bs, &opts.scf_options) {
                    let e_trial = trial_wf.total_energy_hartree;
                    if e_trial <= current_e + opts.c1 * alpha * g_dot_p {
                        accepted = true;
                        break;
                    }
                }
            }
            alpha *= opts.backtrack_factor;
        }

        if !accepted {
            // Line search failed: if H is not diagonal, reset H and take a small gradient step
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
                // Try a small step along negative gradient
                let small_step = 0.05;
                for i in 0..dim {
                    x_new[i] = x[i] - small_step * g[i];
                }
                let trial_mol = mol_with_coords(mol, &x_new);
                if let Ok(trial_bs) = BasisSet::from_name(basis_name, &trial_mol) {
                    if let Ok(_trial_wf) = rhf(&trial_mol, &trial_bs, &opts.scf_options) {
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

        // Compute step s = x_new - x
        let mut s = vec![0.0; dim];
        for i in 0..dim {
            s[i] = x_new[i] - x[i];
        }

        // Calculate max atom displacement norm
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

        // Gradient at new position
        let new_mol = mol_with_coords(mol, &x_new);
        let new_bs = BasisSet::from_name(basis_name, &new_mol)?;
        let new_wf = rhf(&new_mol, &new_bs, &opts.scf_options)?;
        let new_g_3d = rhf_gradient(&new_mol, &new_bs, &new_wf)?;
        let mut g_new = Vec::with_capacity(dim);
        for ag in &new_g_3d {
            g_new.extend_from_slice(ag);
        }

        // y = g_new - g
        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        let s_dot_y = dot_prod(&s, &y);
        if s_dot_y > 1e-10 {
            // BFGS update on inverse Hessian:
            // H_new = (I - rho * s * y^T) H (I - rho * y * s^T) + rho * s * s^T
            let rho = 1.0 / s_dot_y;

            // V = I - rho * s * y^T
            let mut v = Matrix2D::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    let delta_ij = if i == j { 1.0 } else { 0.0 };
                    v.set(i, j, delta_ij - rho * s[i] * y[j]);
                }
            }

            // temp = V * H
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

            // H_new = temp * V^T + rho * s * s^T
            // Note V^T[k, j] = V[j, k] = delta_jk - rho * s[j] * y[k]
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
            // Curvature too small: reset H to diagonal
            for i in 0..dim {
                for j in 0..dim {
                    h_inv.set(i, j, if i == j { 0.7 } else { 0.0 });
                }
            }
        }

        x = x_new;
    }

    let _final_mol = mol_with_coords(mol, &x);
    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}
