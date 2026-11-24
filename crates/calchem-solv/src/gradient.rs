//! Analytical and numerical nuclear gradients for Continuum Solvation (C-PCM / COSMO / IEF-PCM / SMD / Onsager).
//!
//! Evaluates the fixed-AO-density partial finite-difference nuclear gradient on the exact same
//! reaction-field and CDS energy surface as `scrf.rs` (`ReactionFieldSolver` + `compute_cds_energy`):
//! $$\mathbf{g}_A^{\text{solv}} = \nabla_{\mathbf{R}_A} \left( E_{\text{elec}}^{\text{RF}}(\mathbf{R}; \mathbf{P}) + \Delta G_{\text{CDS}}(\mathbf{R}) \right)$$

use calchem_basis::BasisSet;
use calchem_model::{CalcError, Molecule};
use calchem_tensor::Matrix2D;

use crate::scrf::{compute_cds_energy, ReactionFieldSolver, SolvationOptions};

/// Continuum solvation energy for a specified total AO density matrix (Hartree),
/// using the exact same `ReactionFieldSolver` and `compute_cds_energy` surface as `scrf.rs`.
pub fn pcm_energy_for_density(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    opts: &SolvationOptions,
) -> Result<f64, CalcError> {
    opts.model.validate_native()?;
    let n = basis.n_ao;
    if density.rows != n || density.cols != n || density.data.len() != n * n {
        return Err(CalcError::InvalidMolecule(
            "Invalid PCM density dimensions".into(),
        ));
    }

    let rf = ReactionFieldSolver::build(mol, basis, opts)?;
    let (e_elec, _v_rf, _q) = rf.reaction_field(density)?;
    let (cav_area_a2, _) = rf.cavity_area_and_coords();
    let (e_cds, _, _) = compute_cds_energy(mol, opts, cav_area_a2);
    Ok(e_elec + e_cds)
}

/// Computes the nuclear gradient of the continuum solvation energy (Eh/Bohr) via central differences.
///
/// Central differences at FIXED AO density on the unified `SolvationOptions` energy surface
/// (supporting C-PCM, COSMO, IEF-PCM, SMD intrinsic radii + parameterized CDS, and Onsager).
/// This is a partial derivative; it excludes density response and is not the gradient of a
/// fully relaxed SCRF energy. Cavity topology is piecewise smooth.
pub fn compute_pcm_gradient_fd(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    opts: &SolvationOptions,
    h_bohr: f64,
) -> Result<Vec<[f64; 3]>, CalcError> {
    opts.model.validate_native()?;
    let n_atoms = mol.atoms.len();
    let mut grad = vec![[0.0, 0.0, 0.0]; n_atoms];

    if !h_bohr.is_finite() || h_bohr <= 0.0 {
        return Err(CalcError::InvalidMolecule(
            "PCM difference step must be finite and positive".into(),
        ));
    }
    let h = h_bohr;

    for i in 0..n_atoms {
        for c in 0..3 {
            let mut mol_plus = mol.clone();
            mol_plus.atoms[i].position_bohr[c] += h;
            let basis_plus = BasisSet::from_name(&basis.name, &mol_plus)?;
            let e_plus = pcm_energy_for_density(&mol_plus, &basis_plus, density, opts)?;

            let mut mol_minus = mol.clone();
            mol_minus.atoms[i].position_bohr[c] -= h;
            let basis_minus = BasisSet::from_name(&basis.name, &mol_minus)?;
            let e_minus = pcm_energy_for_density(&mol_minus, &basis_minus, density, opts)?;

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
    opts: &SolvationOptions,
) -> Result<Vec<[f64; 3]>, CalcError> {
    let _ = (mol, basis, density, opts);
    Err(CalcError::UnsupportedMethod(
        "Analytic PCM gradient is not implemented. Use compute_pcm_gradient_fd for a fixed-density partial derivative, or fully relaxed SCRF energy differences.".into()
    ))
}

/// Unified continuum solvation nuclear gradient entry point (fixed-AO-density central difference).
pub fn run_pcm_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    density: &Matrix2D,
    opts: &SolvationOptions,
) -> Result<Vec<[f64; 3]>, CalcError> {
    compute_pcm_gradient_fd(mol, basis, density, opts, 1e-4)
}
