//! Semi-numerical Cartesian Hessian matrix ($H_{ij} = \frac{\partial^2 E}{\partial x_i \partial x_j}$)
//! and Atomic Polar Tensor (APT, $P_{qi} = \frac{\partial \mu_q}{\partial x_i}$) from analytical nuclear gradients.

use calchem_basis::BasisSet;
use calchem_dft::{D3Model, GridLevel, MolecularGrid, XcFunctional};
use calchem_grad::{rhf_gradient, rks_gradient_with_grid, uhf_gradient, uks_gradient_with_grid};
use calchem_model::{CalcError, Molecule, ScfOptions, ScfSummary};
use calchem_prop::{compute_dipole, AU_TO_DEBYE};
use calchem_scf::{
    rhf_with_guess, rks_with_grid_and_guess, uhf_with_guess, uks_with_grid_and_guess, InitialGuess,
};
use calchem_tensor::Matrix2D;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

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

/// Evaluates analytical nuclear gradient and electric dipole moment (in a.u.) at a geometry,
/// reusing a single DFT grid and an optional center-geometry orbital projection guess.
fn eval_grad_and_dipole(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    d3: Option<&D3Model>,
    grid_level: GridLevel,
    scf_opts: &ScfOptions,
    guess: Option<&InitialGuess>,
) -> Result<(Vec<f64>, [f64; 3]), CalcError> {
    let bs = BasisSet::from_name(basis_name, mol)?;
    let (mut grad_atoms, dipole_au) = match method {
        VibMethod::Rhf => {
            let wfn = rhf_with_guess(mol, &bs, scf_opts, guess)?;
            let g = rhf_gradient(mol, &bs, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au)
        }
        VibMethod::Uhf => {
            let wfn = uhf_with_guess(mol, &bs, scf_opts, guess)?;
            let g = uhf_gradient(mol, &bs, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au)
        }
        VibMethod::Rks(func) => {
            let grid = MolecularGrid::build_for_basis(mol, &bs, grid_level);
            let wfn = rks_with_grid_and_guess(mol, &bs, func, &grid, scf_opts, guess)?;
            let g = rks_gradient_with_grid(mol, &bs, func, &grid, &wfn)?;
            let (dip_deb, _) = compute_dipole(mol, &bs, &wfn.density_alpha);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au)
        }
        VibMethod::Uks(func) => {
            let grid = MolecularGrid::build_for_basis(mol, &bs, grid_level);
            let wfn = uks_with_grid_and_guess(mol, &bs, func, &grid, scf_opts, guess)?;
            let g = uks_gradient_with_grid(mol, &bs, func, &grid, &wfn)?;
            let total_dens = wfn.total_density();
            let (dip_deb, _) = compute_dipole(mol, &bs, &total_dens);
            let dip_au = [
                dip_deb[0] / AU_TO_DEBYE,
                dip_deb[1] / AU_TO_DEBYE,
                dip_deb[2] / AU_TO_DEBYE,
            ];
            (g, dip_au)
        }
    };
    if let Some(model) = d3 { model.add_gradient(&mol.atoms, &mut grad_atoms)?; }
    let mut grad_3n = Vec::with_capacity(3 * grad_atoms.len());
    for row in grad_atoms {
        grad_3n.push(row[0]);
        grad_3n.push(row[1]);
        grad_3n.push(row[2]);
    }
    Ok((grad_3n, dipole_au))
}

#[derive(Serialize, Deserialize)]
struct DisplacementRecord {
    model: String,
    coord_index: usize,
    h_col: Vec<f64>,
    apt_col: [f64; 3],
}

fn load_displacement(dir: &Path, model: &str, k: usize, dim: usize) -> Option<(Vec<f64>, [f64; 3])> {
    let path = dir.join(format!("disp_{k:04}.json"));
    let f = fs::File::open(&path).ok()?;
    let rec: DisplacementRecord = serde_json::from_reader(BufReader::new(f)).ok()?;
    if rec.model == model
        && rec.coord_index == k
        && rec.h_col.len() == dim
        && rec.h_col.iter().all(|v| v.is_finite())
        && rec.apt_col.iter().all(|v| v.is_finite())
    {
        Some((rec.h_col, rec.apt_col))
    } else {
        None
    }
}

fn save_displacement(
    dir: &Path,
    model: &str,
    k: usize,
    h_col: &[f64],
    apt_col: [f64; 3],
) -> Result<(), CalcError> {
    fs::create_dir_all(dir).map_err(|e| CalcError::Internal(format!("Cannot create freq checkpoint dir: {e}")))?;
    let path = dir.join(format!("disp_{k:04}.json"));
    let tmp = dir.join(format!("disp_{k:04}.{}.tmp", std::process::id()));
    let write_res = (|| -> std::io::Result<()> {
        let f = fs::File::create(&tmp)?;
        let mut writer = BufWriter::new(f);
        serde_json::to_writer(
            &mut writer,
            &DisplacementRecord {
                model: model.to_string(),
                coord_index: k,
                h_col: h_col.to_vec(),
                apt_col,
            },
        )?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&tmp, &path)
    })();
    write_res.map_err(|e| CalcError::Internal(format!("Cannot save freq displacement {k}: {e}")))
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
    let (h, a, summary, correction) = compute_hessian_with_model(mol, basis_name, method, opts, None, 1)?;
    Ok((h, a, summary.total_energy_hartree + correction))
}

pub fn compute_hessian_with_model(
    mol: &Molecule, basis_name: &str, method: VibMethod, opts: &FreqOptions,
    d3: Option<&D3Model>, displacement_jobs: usize,
) -> Result<(Matrix2D, Matrix2D, ScfSummary, f64), CalcError> {
    compute_hessian_with_checkpoint(mol, basis_name, method, opts, d3, displacement_jobs, None, None)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_hessian_with_checkpoint(
    mol: &Molecule,
    basis_name: &str,
    method: VibMethod,
    opts: &FreqOptions,
    d3: Option<&D3Model>,
    displacement_jobs: usize,
    checkpoint_dir: Option<&Path>,
    model_fingerprint: Option<&str>,
) -> Result<(Matrix2D, Matrix2D, ScfSummary, f64), CalcError> {
    if displacement_jobs == 0 { return Err(CalcError::InvalidMolecule("displacement_jobs must be positive".into())); }
    // A density error is amplified by 1/(2h) in both the Hessian and APT.
    // Energy convergence alone does not control IR intensity accuracy.
    let mut response_options = opts.clone();
    response_options.scf_options.conv_energy = response_options.scf_options.conv_energy.min(1e-13);
    response_options.scf_options.conv_density =
        response_options.scf_options.conv_density.min(1e-12);
    let opts = &response_options;
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

    let center_bs = BasisSet::from_name(basis_name, mol)?;
    let (center, center_guess) = match method {
        VibMethod::Rhf => {
            let wfn = rhf_with_guess(mol, &center_bs, &opts.scf_options, None)?;
            let guess = InitialGuess::Restricted { n_ao: wfn.n_ao, mo_coefficients: wfn.mo_coefficients.clone() };
            (ScfSummary::from(&wfn), guess)
        }
        VibMethod::Uhf => {
            let wfn = uhf_with_guess(mol, &center_bs, &opts.scf_options, None)?;
            let guess = InitialGuess::Unrestricted { n_ao: wfn.n_ao, mo_alpha: wfn.mo_coefficients_alpha.clone(), mo_beta: wfn.mo_coefficients_beta.clone() };
            (ScfSummary::from(&wfn), guess)
        }
        VibMethod::Rks(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, &center_bs, opts.grid_level);
            let wfn = rks_with_grid_and_guess(mol, &center_bs, xc, &grid, &opts.scf_options, None)?;
            let guess = InitialGuess::Restricted { n_ao: wfn.n_ao, mo_coefficients: wfn.mo_coefficients.clone() };
            (ScfSummary::from(&wfn), guess)
        }
        VibMethod::Uks(xc) => {
            let grid = MolecularGrid::build_for_basis(mol, &center_bs, opts.grid_level);
            let wfn = uks_with_grid_and_guess(mol, &center_bs, xc, &grid, &opts.scf_options, None)?;
            let guess = InitialGuess::Unrestricted { n_ao: wfn.n_ao, mo_alpha: wfn.mo_coefficients_alpha.clone(), mo_beta: wfn.mo_coefficients_beta.clone() };
            (ScfSummary::from(&wfn), guess)
        }
    };
    let correction = d3.map(|m| m.energy(&mol.atoms)).transpose()?.unwrap_or(0.0);
    let two_delta = 2.0 * delta;
    let column = |k: usize| -> Result<(Vec<f64>, [f64; 3]), CalcError> {
            if let (Some(dir), Some(fp)) = (checkpoint_dir, model_fingerprint) {
                if let Some(cached) = load_displacement(dir, fp, k, dim) {
                    eprintln!("freq coord={}/{dim} (cached)", k + 1);
                    return Ok(cached);
                }
            }
            let atom_idx = k / 3;
            let axis = k % 3;

            // +delta displacement
            let mut mol_plus = mol.clone();
            mol_plus.atoms[atom_idx].position_bohr[axis] += delta;
            let (g_plus, dip_plus) = eval_grad_and_dipole(
                &mol_plus,
                basis_name,
                method,
                d3,
                opts.grid_level,
                &opts.scf_options,
                Some(&center_guess),
            )?;

            // -delta displacement
            let mut mol_minus = mol.clone();
            mol_minus.atoms[atom_idx].position_bohr[axis] -= delta;
            let (g_minus, dip_minus) = eval_grad_and_dipole(
                &mol_minus,
                basis_name,
                method,
                d3,
                opts.grid_level,
                &opts.scf_options,
                Some(&center_guess),
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

            if let (Some(dir), Some(fp)) = (checkpoint_dir, model_fingerprint) {
                save_displacement(dir, fp, k, &h_col, apt_col)?;
            }
            eprintln!("freq coord={}/{dim} done", k + 1);

            Ok((h_col, apt_col))
    };
    // Each displacement owns a full SCF workspace. Bound independent SCFs while
    // leaving the entire Rayon pool available for its integrals and quadrature.
    let limit = displacement_jobs.min(rayon::current_num_threads()).min(dim);
    let mut cols = Vec::with_capacity(dim);
    for start in (0..dim).step_by(limit) {
        let end = (start + limit).min(dim);
        cols.extend((start..end).into_par_iter().map(&column).collect::<Result<Vec<_>, _>>()?);
    }

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

    Ok((hess, apt, center, correction))
}
