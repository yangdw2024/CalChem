use super::Reference;
use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid, XcFunctional};
use calchem_model::{CalcError, Molecule, ScfOptions, ScfSummary, ScfWavefunction, UhfWavefunction};
use calchem_scf::InitialGuess;

/// Owns the converged orbitals and the actual quadrature used for one geometry.
pub enum GasPoint {
    Restricted(ScfWavefunction, Option<(XcFunctional, MolecularGrid)>),
    Unrestricted(UhfWavefunction, Option<(XcFunctional, MolecularGrid)>),
}

impl GasPoint {
    pub fn solve(mol: &Molecule, bs: &BasisSet, reference: Reference, level: GridLevel, opts: &ScfOptions) -> Result<Self, CalcError> {
        Self::solve_with_guess(mol, bs, reference, level, opts, None)
    }

    pub fn solve_with_guess(
        mol: &Molecule,
        bs: &BasisSet,
        reference: Reference,
        level: GridLevel,
        opts: &ScfOptions,
        guess: Option<&InitialGuess>,
    ) -> Result<Self, CalcError> {
        use calchem_scf::{rhf_with_guess, rks_with_grid_and_guess, uhf_with_guess, uks_with_grid_and_guess};
        Ok(match reference.resolve(mol) {
            Reference::Uhf => Self::Unrestricted(uhf_with_guess(mol, bs, opts, guess)?, None),
            Reference::Rhf | Reference::Auto => Self::Restricted(rhf_with_guess(mol, bs, opts, guess)?, None),
            Reference::Rks(xc) => {
                let grid = MolecularGrid::build_for_basis(mol, bs, level);
                Self::Restricted(rks_with_grid_and_guess(mol, bs, xc, &grid, opts, guess)?, Some((xc, grid)))
            }
            Reference::Uks(xc) | Reference::DftAuto(xc) => {
                let grid = MolecularGrid::build_for_basis(mol, bs, level);
                Self::Unrestricted(uks_with_grid_and_guess(mol, bs, xc, &grid, opts, guess)?, Some((xc, grid)))
            }
        })
    }

    pub fn mo_guess(&self) -> InitialGuess {
        match self {
            Self::Restricted(w, _) => InitialGuess::Restricted {
                n_ao: w.n_ao,
                mo_coefficients: w.mo_coefficients.clone(),
            },
            Self::Unrestricted(w, _) => InitialGuess::Unrestricted {
                n_ao: w.n_ao,
                mo_alpha: w.mo_coefficients_alpha.clone(),
                mo_beta: w.mo_coefficients_beta.clone(),
            },
        }
    }

    pub fn energy(&self) -> f64 {
        match self {
            Self::Restricted(w, _) => w.total_energy_hartree,
            Self::Unrestricted(w, _) => w.total_energy_hartree,
        }
    }

    pub fn summary(&self) -> ScfSummary {
        match self {
            Self::Restricted(w, _) => w.into(),
            Self::Unrestricted(w, _) => w.into(),
        }
    }

    pub fn gradient(&self, mol: &Molecule, bs: &BasisSet) -> Result<Vec<[f64; 3]>, CalcError> {
        self.gradient_with_telemetry(mol, bs, false)
            .map(|r| r.gradient)
    }

    pub fn gradient_with_telemetry(
        &self,
        mol: &Molecule,
        bs: &BasisSet,
        use_gpu: bool,
    ) -> Result<calchem_grad::GradientResult, CalcError> {
        use calchem_grad::{
            rhf_gradient_with_gpu, rks_gradient_with_grid_and_gpu, uhf_gradient_with_gpu,
            uks_gradient_with_grid_and_gpu,
        };
        match self {
            Self::Restricted(w, None) => rhf_gradient_with_gpu(mol, bs, w, use_gpu),
            Self::Unrestricted(w, None) => uhf_gradient_with_gpu(mol, bs, w, use_gpu),
            Self::Restricted(w, Some((xc, grid))) => {
                rks_gradient_with_grid_and_gpu(mol, bs, *xc, grid, w, use_gpu)
            }
            Self::Unrestricted(w, Some((xc, grid))) => {
                uks_gradient_with_grid_and_gpu(mol, bs, *xc, grid, w, use_gpu)
            }
        }
    }
}
