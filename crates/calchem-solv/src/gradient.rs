//! Analytical and numerical nuclear gradients for Continuum Solvation (PCM / COSMO / SCRF).
//!
//! Provides the nuclear gradient of the electrostatic and CDS solvation free energy:
//! $$\mathbf{g}_A^{\text{solv}} = \nabla_{\mathbf{R}_A} \left( \frac{1}{2} \mathbf{q}^T \mathbf{V}^{\text{solute}} + \Delta G_{\text{CDS}} \right)$$

use calchem_basis::BasisSet;
use calchem_model::{CalcError, Molecule};
use calchem_tensor::Matrix2D;

use crate::cavity::{CavityOptions, MolecularCavity};
use crate::matrix::BoundaryMatrix;
use crate::potential::CavityPotentialEngine;
use crate::solvent::{Solvent, SolventModel};

/// Solvation energy for a specified density matrix (Hartree).
fn pcm_energy_for_density(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: &CavityOptions,
) -> Result<f64, CalcError> {
    let cavity = MolecularCavity::build(mol, cavity_opts);
    let b_mat = BoundaryMatrix::build(&cavity)?;
    let pot_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let v_solute = pot_engine.compute_solute_potential(density);
    let f_eps = model.dielectric_factor(solvent.epsilon);
    let q = b_mat.solve_surface_charges(&v_solute, f_eps, Some(mol.charge as f64))?;

    let mut e_elec = 0.0;
    for k in 0..cavity.n_tesserae() {
        e_elec += 0.5 * q[k] * v_solute[k];
    }

    let surface_area_angstrom2 = cavity.total_area_angstrom2;
    let cds_kcal_mol = solvent.surface_tension_kcal_mol_angstrom2() * surface_area_angstrom2;
    let e_cds = cds_kcal_mol / crate::scrf::HARTREE_TO_KCAL_MOL;
    Ok(e_elec + e_cds)
}

/// Computes the nuclear gradient of the PCM electrostatic and CDS solvation free energy (Eh/Bohr).
///
/// Central differences at FIXED AO density, including the CDS area term.
/// This is a partial derivative; it excludes density response and is not the
/// gradient of a relaxed gas-phase or SCRF energy. Cavity topology is piecewise smooth.
pub fn compute_pcm_gradient_fd(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: Option<&CavityOptions>,
    h_bohr: f64,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let n_atoms = mol.atoms.len();
    let mut grad = vec![[0.0, 0.0, 0.0]; n_atoms];

    let opts = cavity_opts.cloned().unwrap_or_default();
    let h = if h_bohr > 0.0 { h_bohr } else { 1e-4 };

    for i in 0..n_atoms {
        for c in 0..3 {
            let mut mol_plus = mol.clone();
            mol_plus.atoms[i].position_bohr[c] += h;
            let basis_plus = BasisSet::from_name(&basis.name, &mol_plus)?;
            let e_plus = pcm_energy_for_density(&mol_plus, &basis_plus, density, solvent, model, &opts)?;

            let mut mol_minus = mol.clone();
            mol_minus.atoms[i].position_bohr[c] -= h;
            let basis_minus = BasisSet::from_name(&basis.name, &mol_minus)?;
            let e_minus = pcm_energy_for_density(&mol_minus, &basis_minus, density, solvent, model, &opts)?;

            let g = (e_plus - e_minus) / (2.0 * h);
            grad[i][c] = g;
        }
    }

    Ok(grad)
}

/// Evaluates the analytical nuclear gradient of the PCM electrostatic boundary interaction and cavity energy.
pub fn compute_pcm_gradient_analytic(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    cavity_opts: Option<&CavityOptions>,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let _ = (mol, basis, density, solvent, cavity_opts);
    Err(CalcError::UnsupportedMethod(
        "Analytic PCM gradient is not implemented. Use compute_pcm_gradient_fd for a fixed-density partial derivative, or fully relaxed SCRF energy differences.".into()
    ))
}

/// Unified PCM nuclear gradient entry point.
pub fn run_pcm_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    solvent: &Solvent,
    model: SolventModel,
    cavity_opts: Option<&CavityOptions>,
) -> Result<Vec<[f64; 3]>, CalcError> {
    compute_pcm_gradient_fd(mol, basis, density, solvent, model, cavity_opts, 1e-4)
}
