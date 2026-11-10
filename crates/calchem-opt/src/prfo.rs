//! Partitioned Rational Function Optimization (P-RFO) for Transition State (TS) Search.
//!
//! Features:
//! - Rational Function Optimization with uphill maximization along the reaction coordinate
//!   and downhill minimization in all transverse degrees of freedom (Baker 1986).
//! - Automatic mode-following across steps using maximal eigenvector overlap.
//! - Bofill (1994) mixed MS/PSB Hessian update maintaining correct indefinite curvature.
//! - Trust radius step truncation and convergence monitoring.

use calchem_basis::BasisSet;
use calchem_dft::GridLevel;
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_tensor::{eigh_sym, Matrix2D};
use calchem_vib::{
    compute_cartesian_hessian_and_apt, FreqOptions, VibMethod,
};
use serde::{Deserialize, Serialize};

use crate::{
    coords_from_mol, dot_prod, mat_vec_mul, mol_with_coords, scf_energy_gradient, OptStep,
    Reference,
};

/// Transition state search options (Partitioned Rational Function Optimization, P-RFO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsOptions {
    /// Maximum number of P-RFO macro-iterations.
    pub max_steps: usize,
    /// Maximum absolute Cartesian force threshold (Hartree / Bohr).
    pub max_force: f64,
    /// Root-mean-square Cartesian force threshold (Hartree / Bohr).
    pub rms_force: f64,
    /// Maximum Cartesian displacement threshold (Bohr).
    pub max_displacement: f64,
    /// Maximum trust radius step length (Bohr).
    pub trust_radius_bohr: f64,
    /// Optional normal mode index (0-based) to follow uphill. Default `None` follows lowest eigenvalue.
    pub mode_index: Option<usize>,
    /// Hessian recalculation interval (0 = initial Hessian only, N = recompute every N steps).
    pub recalc_hessian_interval: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    pub scf_options: ScfOptions,
}

impl Default for TsOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-10;
        Self {
            max_steps: 100,
            max_force: 3e-4,
            rms_force: 1.5e-4,
            max_displacement: 1.2e-3,
            trust_radius_bohr: 0.1,
            mode_index: None,
            recalc_hessian_interval: 0,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            scf_options: scf_opts,
        }
    }
}

/// Output of a transition state search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsResult {
    /// Transition state molecular geometry.
    pub molecule: Molecule,
    /// Final electronic energy (Hartree).
    pub energy_hartree: f64,
    /// Final Cartesian gradient ($N \times 3$, Hartree / Bohr).
    pub gradient: Vec<[f64; 3]>,
    /// Number of steps taken.
    pub steps: usize,
    /// Whether convergence criteria were met.
    pub converged: bool,
    /// Signed imaginary frequency of the saddle mode ($\text{cm}^{-1}$, negative for imaginary).
    pub imaginary_frequency_cm: f64,
    /// Normalized Cartesian displacement vector of the transition mode ($3N$).
    pub transition_vector: Vec<f64>,
    /// Step history.
    pub history: Vec<OptStep>,
}

/// Computes the initial exact finite-difference Cartesian Hessian matrix.
fn compute_initial_hessian(
    mol: &Molecule,
    basis_name: &str,
    reference: Reference,
    grid_level: GridLevel,
    scf_options: &ScfOptions,
) -> Result<Matrix2D, CalcError> {
    let vib_method = match reference.resolve(mol) {
        Reference::Rhf | Reference::Auto => VibMethod::Rhf,
        Reference::Uhf => VibMethod::Uhf,
        Reference::Rks(xc) => VibMethod::Rks(xc),
        Reference::Uks(xc) | Reference::DftAuto(xc) => VibMethod::Uks(xc),
    };
    let freq_opts = FreqOptions {
        step_bohr: 0.001,
        grid_level,
        temperature_k: 298.15,
        pressure_pa: 101325.0,
        symmetry_number: None,
        scf_options: scf_options.clone(),
    };
    let (hess, _, _) = compute_cartesian_hessian_and_apt(mol, basis_name, vib_method, &freq_opts)?;
    Ok(hess)
}

/// Solves the 1D transverse P-RFO secular equation for lambda_n <= min_{j != m} Lambda_j:
/// f(lambda_n) = lambda_n - sum_{j != m} (g_tilde_j^2 / (lambda_n - Lambda_j)) = 0
fn solve_transverse_lambda(
    evals: &[f64],
    g_tilde: &[f64],
    ts_mode: usize,
) -> f64 {
    let mut min_lam = f64::INFINITY;
    for (j, &lam) in evals.iter().enumerate() {
        if j != ts_mode && lam < min_lam {
            min_lam = lam;
        }
    }

    if min_lam.is_infinite() {
        return 0.0;
    }

    // Initial guess strictly below min_lam
    let mut lambda = if min_lam > 0.1 {
        0.0
    } else {
        min_lam - 0.1
    };
    if lambda >= min_lam - 1e-8 {
        lambda = min_lam - 0.1;
    }

    for _ in 0..60 {
        let mut f = lambda;
        let mut f_prime = 1.0;
        for (j, &lam) in evals.iter().enumerate() {
            if j == ts_mode {
                continue;
            }
            let denom = lambda - lam;
            let denom2 = denom * denom;
            let g2 = g_tilde[j] * g_tilde[j];
            f -= g2 / denom;
            f_prime += g2 / denom2;
        }

        if f.abs() < 1e-12 {
            break;
        }

        let delta = f / f_prime.max(1.0);
        let mut new_lam = lambda - delta;
        if new_lam >= min_lam - 1e-8 {
            new_lam = 0.5 * (lambda + min_lam);
        }
        if (new_lam - lambda).abs() < 1e-14 {
            lambda = new_lam;
            break;
        }
        lambda = new_lam;
    }

    lambda
}

/// Updates Hessian matrix using the Bofill (1994) formula:
/// H_{k+1} = H_k + (1 - mu) * Delta H_PSB + mu * Delta H_MS
fn update_hessian_bofill(
    h: &mut Matrix2D,
    s: &[f64],
    y: &[f64],
) {
    let dim = s.len();
    let s_dot_s = dot_prod(s, s);
    if s_dot_s < 1e-14 {
        return;
    }

    let hs = mat_vec_mul(h, s);
    let mut xi = vec![0.0; dim];
    for i in 0..dim {
        xi[i] = y[i] - hs[i];
    }

    let xi_dot_xi = dot_prod(&xi, &xi);
    if xi_dot_xi < 1e-14 {
        return;
    }

    let s_dot_xi = dot_prod(s, &xi);
    let mu = ((s_dot_xi * s_dot_xi) / (s_dot_s * xi_dot_xi)).clamp(0.0, 1.0);
    let use_ms = s_dot_xi.abs() > 1e-10;

    for i in 0..dim {
        for j in 0..dim {
            let d_psb = (xi[i] * s[j] + s[i] * xi[j]) / s_dot_s
                - (s_dot_xi * s[i] * s[j]) / (s_dot_s * s_dot_s);

            let d_h = if use_ms {
                let d_ms = (xi[i] * xi[j]) / s_dot_xi;
                (1.0 - mu) * d_psb + mu * d_ms
            } else {
                d_psb
            };

            h.set(i, j, h.get(i, j) + d_h);
        }
    }

    // Symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h.get(i, j) + h.get(j, i));
            h.set(i, j, avg);
            h.set(j, i, avg);
        }
    }
}

/// Searches for a First-Order Saddle Point (Transition State) using Partitioned RFO (P-RFO).
pub fn search_transition_state(
    mol: &Molecule,
    basis_name: &str,
    opts: &TsOptions,
) -> Result<TsResult, CalcError> {
    let n_atoms = mol.atoms.len();
    if n_atoms < 2 {
        return Err(CalcError::InvalidMolecule(
            "Transition state search requires at least 2 atoms".into(),
        ));
    }

    let dim = n_atoms * 3;
    let mut x = coords_from_mol(mol);

    // Initial exact Hessian evaluation
    let mut hessian = compute_initial_hessian(
        mol,
        basis_name,
        opts.reference,
        opts.grid_level,
        &opts.scf_options,
    )?;

    let mut history = Vec::new();
    let mut last_displacement = 0.0;
    let mut target_mode_vector: Option<Vec<f64>> = None;
    let mut cached_eg: Option<(f64, Vec<[f64; 3]>)> = None;

    for step in 0..=opts.max_steps {
        let (current_e, current_g_3d) = if let Some(eg) = cached_eg.take() {
            eg
        } else {
            let cur_mol = mol_with_coords(mol, &x);
            let cur_basis = BasisSet::from_name(basis_name, &cur_mol)?;
            scf_energy_gradient(
                &cur_mol,
                &cur_basis,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?
        };
        let cur_mol = mol_with_coords(mol, &x);

        let mut g = Vec::with_capacity(dim);
        for row in &current_g_3d {
            g.extend_from_slice(row);
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

        // Diagonalize current Hessian
        let (evals, evecs) = eigh_sym(&hessian, 150, 1e-14)?;

        // Identify the uphill transition mode via mode-following
        let ts_mode = if let Some(ref target_vec) = target_mode_vector {
            let mut best_idx = 0;
            let mut best_overlap = -1.0;
            for col in 0..dim {
                let mut ov = 0.0;
                for row in 0..dim {
                    ov += target_vec[row] * evecs.get(row, col);
                }
                let abs_ov = ov.abs();
                if abs_ov > best_overlap {
                    best_overlap = abs_ov;
                    best_idx = col;
                }
            }
            best_idx
        } else if let Some(user_idx) = opts.mode_index {
            user_idx.min(dim - 1)
        } else {
            // Lowest eigenvalue mode
            0
        };

        // Extract transition vector
        let mut cur_ts_vec = vec![0.0; dim];
        for row in 0..dim {
            cur_ts_vec[row] = evecs.get(row, ts_mode);
        }
        target_mode_vector = Some(cur_ts_vec.clone());

        // Check convergence
        let is_force_converged = if step == 0 {
            max_force < opts.max_force && rms_force < opts.rms_force
        } else {
            max_force < opts.max_force
                && rms_force < opts.rms_force
                && (last_displacement < opts.max_displacement || max_force < 0.5 * opts.max_force)
        };

        if is_force_converged {
            // Bofill is a search model, never a certificate of Hessian inertia.
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

            return Ok(TsResult {
                molecule: cur_mol,
                energy_hartree: current_e,
                gradient: current_g_3d,
                steps: step,
                converged: is_true_ts,
                imaginary_frequency_cm: imag_freq,
                transition_vector: certified_vector,
                history,
            });
        }

        if step == opts.max_steps {
            break;
        }

        // Project gradient into Hessian eigenbasis: g_tilde = V^T * g
        let mut g_tilde = vec![0.0; dim];
        for col in 0..dim {
            let mut sum = 0.0;
            for row in 0..dim {
                sum += evecs.get(row, col) * g[row];
            }
            g_tilde[col] = sum;
        }

        // P-RFO step along transition mode (uphill)
        let lam_m = evals[ts_mode];
        let g_m = g_tilde[ts_mode];
        let lambda_p = 0.5 * lam_m + (0.25 * lam_m * lam_m + g_m * g_m).sqrt();
        let s_tilde_m = -g_m / (lam_m - lambda_p).min(-1e-12);

        // P-RFO step along transverse modes (downhill minimization)
        let lambda_n = solve_transverse_lambda(&evals, &g_tilde, ts_mode);

        let mut s_tilde = vec![0.0; dim];
        for j in 0..dim {
            if j == ts_mode {
                s_tilde[j] = s_tilde_m;
            } else {
                let denom = evals[j] - lambda_n;
                s_tilde[j] = -g_tilde[j] / (if denom.abs() < 1e-12 { 1e-12 } else { denom });
            }
        }

        // Transform back to Cartesian coordinates: s = V * s_tilde
        let mut s = vec![0.0; dim];
        for row in 0..dim {
            let mut sum = 0.0;
            for col in 0..dim {
                sum += evecs.get(row, col) * s_tilde[col];
            }
            s[row] = sum;
        }

        // Apply trust radius scaling
        let step_norm = dot_prod(&s, &s).sqrt();
        if step_norm > opts.trust_radius_bohr && step_norm > 1e-14 {
            let scale = opts.trust_radius_bohr / step_norm;
            for val in &mut s {
                *val *= scale;
            }
        }

        let mut x_new = x.clone();
        for i in 0..dim {
            x_new[i] += s[i];
        }

        // Compute max atomic displacement
        let mut max_disp = 0.0;
        for a in 0..n_atoms {
            let dx = s[3 * a];
            let dy = s[3 * a + 1];
            let dz = s[3 * a + 2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > max_disp {
                max_disp = d;
            }
        }
        last_displacement = max_disp;

        // Evaluate gradient at new point
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
        for row in &new_g_3d {
            g_new.extend_from_slice(row);
        }

        let mut y = vec![0.0; dim];
        for i in 0..dim {
            y[i] = g_new[i] - g[i];
        }

        cached_eg = Some((new_e, new_g_3d));

        // Update Hessian (either full recalculation or Bofill update)
        if opts.recalc_hessian_interval > 0 && (step + 1) % opts.recalc_hessian_interval == 0 {
            hessian = compute_initial_hessian(
                &new_mol,
                basis_name,
                opts.reference,
                opts.grid_level,
                &opts.scf_options,
            )?;
        } else {
            update_hessian_bofill(&mut hessian, &s, &y);
        }

        x = x_new;
    }

    Err(CalcError::NotConverged {
        iterations: opts.max_steps,
        delta_e: 0.0,
        rms_d: last_displacement,
    })
}
