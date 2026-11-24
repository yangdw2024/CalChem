//! `calchem-solv`: Pure Rust Continuum Solvation Models (PCM, COSMO, SMD, SCRF).
//!
//! Provides:
//! - Molecular cavity generation using interlocking Bondi van der Waals spheres and Lebedev quadrature (`cavity`)
//! - Common solvent parameter database and dielectric response factors (`solvent`)
//! - Green's function electrostatic boundary interaction matrix $A$ and Klamt self-interaction solver (`matrix`)
//! - Exact quantum-mechanical solute potential and reaction field operator integration (`potential`)
//! - Self-Consistent Reaction Field (SCRF) for RHF and RKS and post-SCF PCM solvation (`scrf`)

pub mod cavity;
pub mod gradient;
pub mod matrix;
pub mod potential;
pub mod scrf;
pub mod smd;
pub mod solvent;

pub use cavity::{
    bondi_vdw_radius, smd_intrinsic_radius, CavityLebedevOrder, CavityOptions, CavityRadiiMode,
    MolecularCavity, Tessera,
};
pub use gradient::{
    compute_pcm_gradient_analytic, compute_pcm_gradient_fd, pcm_energy_for_density,
    run_pcm_gradient,
};
pub use matrix::BoundaryMatrix;
pub use potential::CavityPotentialEngine;
pub use scrf::{
    compute_pcm_solvation, run_scrf, run_scrf_rhf, run_scrf_rks, run_scrf_rks_with_grid,
    run_scrf_unrestricted, run_scrf_unrestricted_with_grid, run_scrf_with_grid, SolvationOptions,
    SolvationResult, HARTREE_TO_KCAL_MOL,
};
pub use smd::compute_smd_cds_energy;
pub use solvent::{SmdDescriptors, Solvent, SolventModel};
