//! Gonzalez-Schlegel Constrained Hypersphere Minimization for Intrinsic Reaction Coordinate (IRC).
//!
//! Integrates the steepest-descent reaction path in mass-weighted Cartesian coordinates:
//!   dq / ds = - grad_q(E) / ||grad_q(E)||
//! using Gonzalez & Schlegel's (1989, 1990) implicit circular arc geodesic integrator.

use calchem_basis::BasisSet;
use calchem_dft::GridLevel;
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_vib::masses::{atomic_mass_amu, HARTREE_TO_KCAL_MOL};
use serde::{Deserialize, Serialize};

use crate::{
    coords_from_mol, dot_prod, mol_with_coords, scf_energy_gradient, Reference,
};

/// Direction to follow the Intrinsic Reaction Coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IrcDirection {
    #[default]
    Both,
    Forward,
    Reverse,
}

impl IrcDirection {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "forward" | "fwd" => Self::Forward,
            "reverse" | "rev" | "backward" => Self::Reverse,
            _ => Self::Both,
        }
    }
}

/// Options for Gonzalez-Schlegel IRC integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcOptions {
    /// Step size in mass-weighted coordinates (default 0.1 amu^(1/2) * Bohr).
    pub step_size_bohr: f64,
    /// Maximum number of points per branch (default 30).
    pub max_points: usize,
    /// Direction: Both, Forward, or Reverse.
    #[serde(default)]
    pub direction: IrcDirection,
    /// Gradient norm threshold to terminate branch (default 5e-4 a.u.).
    pub convergence_tol: f64,
    /// Maximum Gonzalez-Schlegel hypersphere minimization iterations per step.
    pub max_gs_iter: usize,
    #[serde(default)]
    pub reference: Reference,
    #[serde(default)]
    pub grid_level: GridLevel,
    pub scf_options: ScfOptions,
}

impl Default for IrcOptions {
    fn default() -> Self {
        let mut scf_opts = ScfOptions::default();
        scf_opts.conv_energy = 1e-11;
        scf_opts.conv_density = 1e-10;
        Self {
            step_size_bohr: 0.1,
            max_points: 30,
            direction: IrcDirection::Both,
            convergence_tol: 5e-4,
            max_gs_iter: 25,
            reference: Reference::Auto,
            grid_level: GridLevel::Medium,
            scf_options: scf_opts,
        }
    }
}

/// A single point along the IRC reaction path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcPoint {
    /// 0-based point index along the branch (0 is the transition state).
    pub point_index: usize,
    /// Arc length s in mass-weighted coordinates (amu^(1/2) * Bohr).
    pub path_coordinate_s: f64,
    /// Electronic energy at this point (Hartree).
    pub energy_hartree: f64,
    /// Energy difference relative to the transition state (kcal/mol, <= 0).
    pub delta_energy_kcal_mol: f64,
    /// Norm of the gradient in mass-weighted coordinates (Hartree / (amu^(1/2) * Bohr)).
    pub gradient_norm: f64,
    /// Molecular geometry at this point.
    pub molecule: Molecule,
}

/// A directional branch of the IRC path (forward to product or reverse to reactant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcBranch {
    pub direction: String,
    pub points: Vec<IrcPoint>,
    pub converged_to_minimum: bool,
    pub final_energy_hartree: f64,
    pub barrier_kcal_mol: f64,
}

/// Full IRC result containing both forward and reverse branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrcResult {
    pub transition_state: IrcPoint,
    pub forward: Option<IrcBranch>,
    pub reverse: Option<IrcBranch>,
    /// Overall reaction energy: E_product - E_reactant (kcal/mol).
    pub reaction_energy_kcal_mol: Option<f64>,
    /// Forward reaction barrier: E_TS - E_reactant (kcal/mol).
    pub barrier_forward_kcal_mol: Option<f64>,
    /// Reverse reaction barrier: E_TS - E_product (kcal/mol).
    pub barrier_reverse_kcal_mol: Option<f64>,
}

fn get_masses_3n(mol: &Molecule) -> Vec<f64> {
    let mut m = Vec::with_capacity(mol.atoms.len() * 3);
    for atom in &mol.atoms {
        let mass = atomic_mass_amu(atom.z);
        m.push(mass);
        m.push(mass);
        m.push(mass);
    }
    m
}

fn to_mass_weighted(x: &[f64], masses: &[f64]) -> Vec<f64> {
    x.iter().zip(masses.iter()).map(|(&xi, &mi)| xi * mi.sqrt()).collect()
}

fn from_mass_weighted(q: &[f64], masses: &[f64]) -> Vec<f64> {
    q.iter().zip(masses.iter()).map(|(&qi, &mi)| qi / mi.sqrt()).collect()
}

fn gradient_mass_weighted(g: &[f64], masses: &[f64]) -> Vec<f64> {
    g.iter().zip(masses.iter()).map(|(&gi, &mi)| gi / mi.sqrt()).collect()
}

fn eval_q_energy_gradient(
    q: &[f64],
    masses: &[f64],
    base_mol: &Molecule,
    basis_name: &str,
    opts: &IrcOptions,
) -> Result<(f64, Vec<f64>, Molecule), CalcError> {
    let x = from_mass_weighted(q, masses);
    let cur_mol = mol_with_coords(base_mol, &x);
    let cur_bs = BasisSet::from_name(basis_name, &cur_mol)?;
    let (e, g_3d) = scf_energy_gradient(
        &cur_mol,
        &cur_bs,
        opts.reference,
        opts.grid_level,
        &opts.scf_options,
    )?;
    let mut g_cart = Vec::with_capacity(q.len());
    for row in g_3d {
        g_cart.extend_from_slice(&row);
    }
    let g_q = gradient_mass_weighted(&g_cart, masses);
    Ok((e, g_q, cur_mol))
}

/// Integrates one branch of the IRC path (forward or reverse) using Gonzalez-Schlegel.
fn integrate_irc_branch(
    ts_q: &[f64],
    ts_tangent: &[f64],
    masses: &[f64],
    base_mol: &Molecule,
    basis_name: &str,
    e_ts: f64,
    direction_sign: f64,
    direction_label: &str,
    opts: &IrcOptions,
) -> Result<IrcBranch, CalcError> {
    let dim = ts_q.len();
    let mut ds = opts.step_size_bohr;

    let mut points = Vec::with_capacity(opts.max_points + 1);

    // Initial launch vector: tangent along transition vector
    let mut p_k = vec![0.0; dim];
    for i in 0..dim {
        p_k[i] = direction_sign * ts_tangent[i];
    }
    let p_k_norm = dot_prod(&p_k, &p_k).sqrt().max(1e-14);
    for v in &mut p_k {
        *v /= p_k_norm;
    }

    let mut q_cur = ts_q.to_vec();
    let mut s_acc = 0.0;
    let mut converged_to_min = false;
    let mut min_energy = e_ts;

    for pt_idx in 1..=opts.max_points {
        let mut accepted=None;
        for _retry in 0..8 {
        let radius=0.5*ds;
        // Hypersphere center: q_star = q_cur + 0.5 * ds * p_k
        let mut q_star = vec![0.0; dim];
        for i in 0..dim {
            q_star[i] = q_cur[i] + radius * p_k[i];
        }

        // Initial guess on the hypersphere: q_cur + ds * p_k
        let mut q_trial = vec![0.0; dim];
        for i in 0..dim {
            q_trial[i] = q_cur[i] + ds * p_k[i];
        }

        // Gonzalez-Schlegel constrained hypersphere minimization
        for _ in 0..opts.max_gs_iter {
            let (_e_eval, g_q, _pt_m) = eval_q_energy_gradient(&q_trial, masses, base_mol, basis_name, opts)?;

            // Radial unit vector from sphere center to current trial point
            let mut u = vec![0.0; dim];
            for i in 0..dim {
                u[i] = (q_trial[i] - q_star[i]) / radius;
            }
            let g_dot_u = dot_prod(&g_q, &u);

            // Perpendicular tangent gradient: g_perp = g_q - (g_q . u) * u
            let mut g_perp = vec![0.0; dim];
            for i in 0..dim {
                g_perp[i] = g_q[i] - g_dot_u * u[i];
            }
            let g_perp_norm = dot_prod(&g_perp, &g_perp).sqrt();
            if g_perp_norm < opts.convergence_tol {
                break;
            }

            // Step in tangent space with curvature-safe damping
            let alpha = (radius / (g_perp_norm + 1e-4)).clamp(0.05, 0.8);
            let mut q_cand = vec![0.0; dim];
            for i in 0..dim {
                q_cand[i] = q_trial[i] - alpha * g_perp[i];
            }

            // Project back onto the sphere of radius R
            let mut diff = vec![0.0; dim];
            for i in 0..dim {
                diff[i] = q_cand[i] - q_star[i];
            }
            let diff_norm = dot_prod(&diff, &diff).sqrt().max(1e-14);
            for i in 0..dim {
                q_trial[i] = q_star[i] + radius * (diff[i] / diff_norm);
            }
        }

        // Always evaluate exact energy & gradient at the final q_trial geometry
        // to guarantee coordinates, energy, and forces are strictly synchronized.
        let (e_k, g_k, pt_mol) = eval_q_energy_gradient(&q_trial, masses, base_mol, basis_name, opts)?;
        let g_k_norm = dot_prod(&g_k, &g_k).sqrt();
        let u:Vec<_>=q_trial.iter().zip(&q_star).map(|(q,c)|(q-c)/radius).collect();
        let radial=dot_prod(&g_k,&u);
        let tangent_norm=g_k.iter().zip(&u).map(|(g,u)|(g-radial*u).powi(2)).sum::<f64>().sqrt();
        let previous_e=points.last().map(|p:&IrcPoint|p.energy_hartree).unwrap_or(e_ts);
        if tangent_norm<=opts.convergence_tol && e_k<=previous_e+1e-8 {
            accepted=Some((q_trial,e_k,g_k,pt_mol,g_k_norm));break;
        }
        ds*=0.5;

        }
        let Some((q_trial,e_k,g_k,pt_mol,g_k_norm))=accepted else {
            // No accepted sphere endpoint: report an incomplete branch, never a minimum.
            break;
        };
        s_acc += ds;

        if e_k < min_energy {
            min_energy = e_k;
        }

        points.push(IrcPoint {
            point_index: pt_idx,
            path_coordinate_s: s_acc,
            energy_hartree: e_k,
            delta_energy_kcal_mol: (e_k - e_ts) * HARTREE_TO_KCAL_MOL,
            gradient_norm: g_k_norm,
            molecule: pt_mol,
        });

        // Termination checks
        if g_k_norm < opts.convergence_tol {
            converged_to_min = true;
            break;
        }

        // Energy rebound: if energy rises above previous step along the descent path.
        // Rigorous condition: only declare converged to minimum if gradient is small (< 3 * tol).
        // Otherwise, the trajectory simply encountered a barrier or step-size overshoot.
        if pt_idx > 1 && e_k > points[pt_idx - 2].energy_hartree + 1e-6 {
            if g_k_norm < 3.0 * opts.convergence_tol {
                converged_to_min = true;
            }
            break;
        }

        // Update tangent vector for next step: p_{k+1} = -g_k / ||g_k||
        if g_k_norm > 1e-14 {
            for i in 0..dim {
                p_k[i] = -g_k[i] / g_k_norm;
            }
        } else {
            converged_to_min = true;
            break;
        }

        q_cur = q_trial;
    }

    let final_e = points.last().map(|p| p.energy_hartree).unwrap_or(e_ts);
    let barrier = (e_ts - min_energy) * HARTREE_TO_KCAL_MOL;

    Ok(IrcBranch {
        direction: direction_label.to_string(),
        points,
        converged_to_minimum: converged_to_min,
        final_energy_hartree: final_e,
        barrier_kcal_mol: barrier.max(0.0),
    })
}

/// Runs full Intrinsic Reaction Coordinate (IRC) integration from a Transition State.
pub fn run_irc(
    ts_mol: &Molecule,
    basis_name: &str,
    transition_vector: &[f64],
    opts: &IrcOptions,
) -> Result<IrcResult, CalcError> {
    let dim=3*ts_mol.atoms.len();
    if transition_vector.len()!=dim || transition_vector.iter().any(|v|!v.is_finite())
        || transition_vector.iter().map(|v|v*v).sum::<f64>()<1e-20
        || !opts.step_size_bohr.is_finite() || opts.step_size_bohr<=0.0
        || !opts.convergence_tol.is_finite() || opts.convergence_tol<=0.0
        || opts.max_gs_iter==0 || opts.max_points==0 {
        return Err(CalcError::InvalidMolecule("IRC requires a nonzero finite 3N mode, positive step/tolerance and nonzero iteration budgets".into()));
    }
    let masses = get_masses_3n(ts_mol);
    let x_ts = coords_from_mol(ts_mol);
    let q_ts = to_mass_weighted(&x_ts, &masses);

    // Evaluate transition state energy and gradient
    let (e_ts, g_ts_q, _) = eval_q_energy_gradient(&q_ts, &masses, ts_mol, basis_name, opts)?;
    let g_ts_norm = dot_prod(&g_ts_q, &g_ts_q).sqrt();

    // Mass-weight and normalize transition vector
    let q_tangent = to_mass_weighted(transition_vector, &masses);
    let norm_tangent = dot_prod(&q_tangent, &q_tangent).sqrt().max(1e-14);
    let norm_q_tangent: Vec<f64> = q_tangent.into_iter().map(|v| v / norm_tangent).collect();

    let ts_point = IrcPoint {
        point_index: 0,
        path_coordinate_s: 0.0,
        energy_hartree: e_ts,
        delta_energy_kcal_mol: 0.0,
        gradient_norm: g_ts_norm,
        molecule: ts_mol.clone(),
    };

    let forward = if matches!(opts.direction, IrcDirection::Forward | IrcDirection::Both) {
        Some(integrate_irc_branch(
            &q_ts,
            &norm_q_tangent,
            &masses,
            ts_mol,
            basis_name,
            e_ts,
            1.0,
            "forward",
            opts,
        )?)
    } else {
        None
    };

    let reverse = if matches!(opts.direction, IrcDirection::Reverse | IrcDirection::Both) {
        Some(integrate_irc_branch(
            &q_ts,
            &norm_q_tangent,
            &masses,
            ts_mol,
            basis_name,
            e_ts,
            -1.0,
            "reverse",
            opts,
        )?)
    } else {
        None
    };

    let e_prod = forward.as_ref().filter(|f|f.converged_to_minimum).map(|f| f.final_energy_hartree);
    let e_react = reverse.as_ref().filter(|r|r.converged_to_minimum).map(|r| r.final_energy_hartree);

    let rxn_energy = match (e_prod, e_react) {
        (Some(p), Some(r)) => Some((p - r) * HARTREE_TO_KCAL_MOL),
        _ => None,
    };

    let barrier_fwd = e_react.map(|r| ((e_ts - r) * HARTREE_TO_KCAL_MOL).max(0.0));
    let barrier_rev = e_prod.map(|p| ((e_ts - p) * HARTREE_TO_KCAL_MOL).max(0.0));

    Ok(IrcResult {
        transition_state: ts_point,
        forward,
        reverse,
        reaction_energy_kcal_mol: rxn_energy,
        barrier_forward_kcal_mol: barrier_fwd,
        barrier_reverse_kcal_mol: barrier_rev,
    })
}
