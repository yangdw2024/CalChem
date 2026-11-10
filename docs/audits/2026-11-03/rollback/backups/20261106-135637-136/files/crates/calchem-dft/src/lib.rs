//! `calchem-dft`: Pure Rust Kohn-Sham Density Functional Theory Engine.
//!
//! Provides:
//! - Exact octahedral Lebedev-Laikov spherical quadrature (`lebedev`)
//! - Multicenter Becke ($k=3$) + Murray-Handy-Laming radial molecular grids & weight gradients (`grid`)
//! - Shell-screened AO value, gradient, and Hessian evaluation on DFT grids (`ao_grid`)
//! - Spin-polarized LDA, GGA, and Hybrid XC functionals (`SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`) (`xc`)
//! - Parallel Kohn-Sham XC matrix assembly ($E_{\text{xc}}, V^{\text{xc}}$) and exact analytical nuclear gradients (`vxc`)

pub mod ao_grid;
pub mod grid;
pub mod lebedev;
pub mod vxc;
pub mod xc;

pub use grid::{GridLevel, GridPoint, MolecularGrid};
pub use vxc::{evaluate_vxc_rks, evaluate_vxc_uks, xc_nuclear_gradient};
pub use xc::{XcFunctional, XcPointOut, eval_xc_spin};
