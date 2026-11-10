//! Semi-numerical Cartesian Hessian matrix ($H_{ij} = \frac{\partial^2 E}{\partial x_i \partial x_j}$)
//! and Atomic Polar Tensor (APT, $P_{qi} = \frac{\partial \mu_q}{\partial x_i}$) from analytical nuclear gradients.

use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient, uhf_gradient, uks_gradient};
use calchem_model::{CalcError, Molecule, ScfOptions};
use calchem_prop::{AU_TO_DEBYE, compute_dipole};
use calchem_scf::{rhf, rks, uhf, uks};
use calchem_tensor::Matrix2D;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Calculation method / functional for vibrational frequency analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VibMethod {
    Rhf,
    Uhf,
    Rks(XcFunctional),
    Uks(XcFunctional),
}

impl VibMethod {
    pub fn is_dft(&self) -> bool {
        matches!(self, Self::Rks(_) | Self::Uks(_))
    }

    pub fn xc_functional(&self) -> Option<XcFunctional> {
        match self {
            Self::Rks(f) | Self::Uks(f) => Some(*f),
            _ => None,
        }
    }
}

/// Options controlling vibrational frequency and Hessian calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreqOptions {
    /// Finite-difference displacement step along nuclear Cartesian coordinates (Bohr).
    /// Default is `0.001` Bohr.
    pub step_bohr: f64,
    /// DFT molecular numerical quadrature grid level (used when method is RKS/UKS).
    pub grid_level: GridLevel,
    /// Standard thermochemical temperature in Kelvin (default `298.15` K).
    pub temperature_k: f64,
    /// Standard thermochemical pressure in Pascal (default `101325.0` Pa = 1 atm).
    pub pressure_pa: f64,
    /// Rotational symmetry number $\sigma$ (optional override; if `None`, estimated or defaulted to 1).
    pub symmetry_number: Option<usize>,
    /// Underlying SCF convergence options.
    pub scf_options: ScfOptions,
}

impl Default for FreqOptions {
    fn default() -> Self {
        Self {
            step_bohr: 0.001,
            grid_level: GridLevel::Medium,
            temperature_k: 298.15,
            pressure_pa: 101325.0,
            symmetry_number: None,
            scf_options: ScfOptions {
                conv_energy: 1e-11,
                conv_density: 1e-9,
                ..Default::default()
            },
        }
    }
}

/// Evaluates analytical nuclear gradient and electric dipole moment (in a.u.) at a geometry.
fn eval_grad_and_dipole(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    grid_level: GridLevel,
    scf_opts: &ScfOptions,
) -> Result<(Vec<f64>, [f64; 3], f64), CalcError> {
    let bs = BasisSet::from_name(basis_name, mol)?;
    let (grad_atoms, dipole_au, energy) = match method {
        VibMethod::Rhf => {
            let wfn = rhf(mol, &bs, scf_opts)?;
            let g = rhf_gradient(mol, &bs, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Uhf => {
            let wfn = uhf(mol, &bs, scf_opts)?;
            let g = uhf_gradient(mol, &bs, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Rks(func) => {
            let wfn = rks(mol, &bs, func, grid_level, scf_opts)?;
            let g = rks_gradient(mol, &bs, func, grid_level, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
        VibMethod::Uks(func) => {
            let wfn = uks(mol, &bs, func, grid_level, scf_opts)?;
            let g = uks_gradient(mol, &bs, func, grid_level, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au, wfn.total_energy_hartree)
        }
    };
    let mut grad_3n = Vec::with_capacity(3 * grad_atoms.len());
    for row in grad_atoms {
        grad_3n.push(row[0]);
        grad_3n.push(row[1]);
        grad_3n.push(row[2]);
    }
    Ok((grad_3n, dipole_au, energy))
}

/// Evaluates the symmetric Cartesian Hessian matrix ($3N \times 3N$, Hartree / $\text{Bohr}^2$)
/// and Atomic Polar Tensor (APT, $3 \times 3N$, $e$) by Rayon-parallel central differentiation
/// of analytical nuclear gradients.
pub fn compute_cartesian_hessian_and_apt(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
) -> Result<(Matrix2D, Matrix2D, f64), CalcError> {
    let n_atoms = mol.atoms.len();
    let dim = 3 * n_atoms;
    if dim == 0 {
        return Err(CalcError::InvalidMolecule(
            "Cannot compute Hessian for empty molecule".into(),
        ));
    }
    let delta = opts.step_bohr;
    if !delta.is_finite() || delta <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "Invalid differentiation step_bohr: {delta}"
        )));
    }

    // Unperturbed reference energy and gradient
    let (_g0, _dip0, e0) = eval_grad_and_dipole(
        mol,
        basis_name,
        method,
        opts.grid_level,
        &opts.scf_options,
    )?;

    let two_delta = 2.0 * delta;

    // Parallel differentiation along all 3N Cartesian coordinates
    let cols: Result<Vec<(Vec<f64>, [f64; 3])>, CalcError> = (0..dim)
        .into_par_iter()
        .map(|k| {
            let atom_idx = k / 3;
            let axis = k % 3;

            // +delta displacement
            let mut mol_plus = mol.clone();
            mol_plus.atoms[atom_idx].position_bohr[axis] += delta;
            let (g_plus, dip_plus, _) = eval_grad_and_dipole(
                &mol_plus,
                basis_name,
                method,
                opts.grid_level,
                &opts.scf_options,
            )?;

            // -delta displacement
            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position_bohr[axis] -= delta;
            let (g_minus, dip_minus, _) = eval_grad_and_dipole(
                &mol_minus,
                basis_name,
                method,
                opts.grid_level,
                &opts.scf_options,
            )?;

            let mut h_col = vec![0.0f64; dim];
            for j in 0..dim {
                h_col[j] = (g_plus[j] - g_minus[j]) / two_delta;
            }

            let apt_col = [
                (dip_plus[0] - dip_minus[0]) / two_delta,
                (dip_plus[1] - dip_minus[1]) / two_delta,
                (dip_plus[2] - dip_minus[2]) / two_delta,
            ];

            Ok((h_col, apt_col))
        })
        .collect();

    let cols = cols?;

    // Assemble Cartesian Hessian H (dim x dim) and enforce exact symmetry H = (H + H^T) / 2
    let mut hess = Matrix2D::zeros(dim, dim);
    for k in 0..dim {
        let (ref h_col, _) = cols[k];
        for j in 0..dim {
            hess.set(j, k, h_col[j]);
        }
    }
    for j in 0..dim {
        for k in (j + 1)..dim {
            let avg = 0.5 * (hess.get(j, k) + hess.get(k, j));
            hess.set(j, k, avg);
            hess.set(k, j, avg);
        }
    }

    // Assemble Atomic Polar Tensor (3 x dim)
    let mut apt = Matrix2D::zeros(3, dim);
    for k in 0..dim {
        let (_, apt_col) = cols[k];
        for q in 0..3 {
            apt.set(q, k, apt_col[q]);
        }
    }

    Ok((hess, apt, e0))
}
