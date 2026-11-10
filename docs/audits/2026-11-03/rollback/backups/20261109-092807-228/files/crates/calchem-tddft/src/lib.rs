//! `calchem-tddft`: Pure Rust Linear-Response Excited-State Engine (TD-DFT, TDA, TDHF, CIS).
//!
//! Capabilities:
//! - Full linear-response Casida TD-DFT / TDHF (`RPA`, `tda = false`) with exact Casida normalization $X^T X - Y^T Y = 1$
//! - Tamm-Dancoff Approximation (`TDA` / `CIS`, `tda = true`, $A X = \omega X$)
//! - Closed-shell RHF and RKS references across all 6 functionals (`SVWN5`, `BLYP`, `PBE`, `B3LYP`, `B3LYP5`, `PBE0`) + ECPs (`LANL2DZ`)
//! - Spin-adapted Singlet ($S_n$), Triplet ($T_n$), and combined (`Both`, yielding $\Delta E_{ST}$) manifolds
//! - Block Davidson subspace solvers with automatic degenerate-shell completion & dense fast-path
//! - Length-gauge transition electric dipole moments $\boldsymbol{\mu}_{0k}$, oscillator strengths $f_k$,
//!   dominant orbital transitions, and Gaussian-broadened UV-Vis molar absorption spectra $\varepsilon(\lambda)$

pub mod davidson;

use calchem_basis::BasisSet;
use calchem_dft::{FxcKernel, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::primitive::{GaussianPair, overlap_primitive};
use calchem_int::compute_eri;
use calchem_model::{CalcError, Molecule, ScfOptions, ScfWavefunction};
use calchem_scf::{rhf, rks_with_grid};
use calchem_tensor::{Matrix2D, Tensor4D, matmul, transpose};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::davidson::{EigenMode, solve_rpa, solve_tda};

/// CODATA 2018 Hartree to electron-volt conversion factor.
pub const HARTREE_TO_EV: f64 = 27.211386245988;
/// CODATA 2018 $hc$ conversion factor (`nm * eV`).
pub const EV_TO_NM: f64 = 1239.8419843320028;
/// Wavenumber per eV (`cm^-1 / eV`).
pub const EV_TO_INV_CM: f64 = 8065.54393734921;
/// IUPAC prefactor for Gaussian-broadened molar decadic extinction coefficient
/// (`epsilon_max = UV_VIS_GAUSS_PREFACTOR * f / FWHM_cm^-1` in `L mol^-1 cm^-1`).
const UV_VIS_GAUSS_PREFACTOR: f64 = 2.1748105e8;

/// Target excited-state spin manifold(s) from a closed-shell singlet reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpinManifold {
    Singlet,
    Triplet,
    Both,
}

impl SpinManifold {
    pub fn from_name(name: &str) -> Result<Self, CalcError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "singlet" | "singlets" | "s" => Ok(Self::Singlet),
            "triplet" | "triplets" | "t" => Ok(Self::Triplet),
            "both" | "all" | "singlet+triplet" | "st" => Ok(Self::Both),
            other => Err(CalcError::UnsupportedMethod(format!(
                "Unknown TD-DFT spin manifold '{other}'; expected 'singlet', 'triplet', or 'both'"
            ))),
        }
    }
}

/// Configuration options for CIS / TDHF / TDA-DFT / TD-DFT excited-state calculations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TddftOptions {
    /// Number of lowest excited states to compute per spin manifold.
    pub n_states: usize,
    /// If `true`, use Tamm-Dancoff Approximation (`TDA` / `CIS`, $B = 0$);
    /// if `false`, solve full linear-response Casida `TD-DFT` / `TDHF` (`RPA`).
    pub tda: bool,
    /// Spin manifold(s) to solve (`Singlet`, `Triplet`, or `Both`).
    pub manifold: SpinManifold,
    /// DFT molecular quadrature grid level (used for RKS ground state and $f_{\text{xc}}$ kernel).
    pub grid_level: GridLevel,
    /// Maximum Block Davidson iterations.
    pub max_iter: usize,
    /// Residual norm convergence threshold (Hartree).
    pub conv_tol: f64,
    /// Force iterative Block Davidson solver even when `n_ov <= 64` (primarily for solver verification).
    pub force_davidson: bool,
    /// Underlying SCF options when running from geometry.
    pub scf_options: ScfOptions,
}

impl Default for TddftOptions {
    fn default() -> Self {
        Self {
            n_states: 5,
            tda: false,
            manifold: SpinManifold::Singlet,
            grid_level: GridLevel::Medium,
            max_iter: 80,
            conv_tol: 1e-6,
            force_davidson: false,
            scf_options: ScfOptions::default(),
        }
    }
}

/// Single occupied -> virtual canonical MO contribution to an excited state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalTransition {
    /// 0-based canonical occupied MO index (`0..n_occ`, where `n_occ - 1` is HOMO).
    pub occ_index: usize,
    /// 0-based canonical virtual MO index (`n_occ..n_mo`, where `n_occ` is LUMO).
    pub virt_index: usize,
    /// Excitation amplitude $X_{ia}$.
    pub amplitude_x: f64,
    /// De-excitation amplitude $Y_{ia}$ (`0.0` in TDA/CIS).
    pub amplitude_y: f64,
    /// Normalized Casida weight $X_{ia}^2 - Y_{ia}^2$ (sums to `1.0` over all OV pairs).
    pub weight: f64,
}

/// Computed electronic excited state and spectroscopic observables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcitedState {
    /// 1-based energy-ordered state index across all computed states.
    pub state_index: usize,
    /// 1-based state index within its spin manifold (e.g. `1` for `S1` or `T1`).
    pub manifold_index: usize,
    /// Spectroscopic state label (`"S1"`, `"T1"`, ...).
    pub label: String,
    /// Spin multiplicity (`1` for Singlet, `3` for Triplet).
    pub multiplicity: usize,
    /// Vertical excitation energy $\omega_k$ in Hartree.
    pub excitation_energy_hartree: f64,
    /// Vertical excitation energy $\omega_k$ in eV.
    pub excitation_energy_ev: f64,
    /// Excited-state total vertical energy $E_0 + \omega_k$ in Hartree.
    pub total_energy_hartree: f64,
    /// Photon wavelength $\lambda_k = hc / \omega_k$ in nm.
    pub wavelength_nm: f64,
    /// Length-gauge electric transition dipole moment $\boldsymbol{\mu}_{0k}$ in atomic units ($e a_0$).
    pub transition_dipole_au: [f64; 3],
    /// Dimensionless dipole-length oscillator strength $f_k = \frac{2}{3} \omega_k |\boldsymbol{\mu}_{0k}|^2$.
    pub oscillator_strength: f64,
    /// Dominant canonical MO pair transitions ordered by descending `|weight|`.
    pub dominant_transitions: Vec<OrbitalTransition>,
}

/// Gaussian-broadened UV-Vis absorption spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvVisSpectrum {
    pub wavelengths_nm: Vec<f64>,
    pub energies_ev: Vec<f64>,
    /// Molar decadic absorption coefficient $\varepsilon(\lambda)$ in $\text{L mol}^{-1}\text{ cm}^{-1}$.
    pub epsilon_l_mol_cm: Vec<f64>,
    pub fwhm_ev: f64,
}

/// Complete output of a CIS / TDHF / TDA-DFT / TD-DFT calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TddftResult {
    pub ground_energy_hartree: f64,
    pub xc_functional: Option<String>,
    pub tda: bool,
    pub n_occ: usize,
    pub n_virt: usize,
    pub homo_energy_hartree: f64,
    pub lumo_energy_hartree: f64,
    pub ks_gap_ev: f64,
    /// Lowest singlet vertical excitation energy $E(S_1)$ in eV (optical gap $E_g^{\text{opt}}$).
    pub optical_gap_ev: Option<f64>,
    /// Lowest triplet vertical excitation energy $E(T_1)$ in eV.
    pub lowest_triplet_ev: Option<f64>,
    /// Singlet-triplet splitting $\Delta E_{ST} = E(S_1) - E(T_1)$ in eV (when both manifolds are computed).
    pub delta_est_ev: Option<f64>,
    pub davidson_iterations_singlet: usize,
    pub davidson_iterations_triplet: usize,
    pub states: Vec<ExcitedState>,
    pub scf_wavefunction: ScfWavefunction,
}

/// Computes the 3 Cartesian AO electric dipole integral matrices $\langle \mu | r_q | \nu \rangle$ ($q \in \{x,y,z\}$).
pub fn compute_ao_dipole_matrices(basis: &BasisSet) -> [Matrix2D; 3] {
    let n = basis.n_ao;
    let mut mats = [
        Matrix2D::zeros(n, n),
        Matrix2D::zeros(n, n),
        Matrix2D::zeros(n, n),
    ];

    for sh_a in &basis.shells {
        let comps_a = sh_a.components();
        for sh_b in &basis.shells {
            if sh_b.ao_offset < sh_a.ao_offset {
                continue;
            }
            let comps_b = sh_b.components();
            for (ia, &ca) in comps_a.iter().enumerate() {
                let mu = sh_a.ao_offset + ia;
                let ca_i32 = ca.map(|v| v as i32);
                for (ib, &cb) in comps_b.iter().enumerate() {
                    let nu = sh_b.ao_offset + ib;
                    if nu < mu {
                        continue;
                    }
                    let cb_i32 = cb.map(|v| v as i32);

                    let mut dip = [0.0f64; 3];
                    for (pa_idx, pa) in sh_a.primitives.iter().enumerate() {
                        let norm_a = sh_a.primitive_norm_for(pa_idx, ca);
                        for (pb_idx, pb) in sh_b.primitives.iter().enumerate() {
                            let norm_b = sh_b.primitive_norm_for(pb_idx, cb);
                            let weight = pa.coeff * pb.coeff * norm_a * norm_b;
                            let pair = GaussianPair::new(
                                pa.exponent,
                                sh_a.center,
                                pb.exponent,
                                sh_b.center,
                            );
                            let s0 = overlap_primitive(ca_i32, cb_i32, &pair);
                            for q in 0..3 {
                                let mut ca_plus = ca_i32;
                                ca_plus[q] += 1;
                                let s_plus = overlap_primitive(ca_plus, cb_i32, &pair);
                                dip[q] += weight * (s_plus + sh_a.center[q] * s0);
                            }
                        }
                    }

                    for q in 0..3 {
                        mats[q].set(mu, nu, dip[q]);
                        mats[q].set(nu, mu, dip[q]);
                    }
                }
            }
        }
    }

    mats
}

/// Rayon-parallel Coulomb contraction $J_{\mu\nu}[M] = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma) M_{\lambda\sigma}$
/// for symmetric $M$.
fn contract_j_sym(eri: &Tensor4D, m_sym: &Matrix2D) -> Matrix2D {
    let n = m_sym.rows;
    let n2 = n * n;
    let m_slice = &m_sym.data;
    let eri_slice = &eri.data;

    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|mu| {
            let mut row_out = vec![0.0f64; n];
            for nu in mu..n {
                let base = (mu * n + nu) * n2;
                let eri_blk = &eri_slice[base..base + n2];
                let mut s = 0.0;
                for idx in 0..n2 {
                    s += eri_blk[idx] * m_slice[idx];
                }
                row_out[nu] = s;
            }
            row_out
        })
        .collect();

    let mut j_mat = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in mu..n {
            let v = rows[mu][nu];
            j_mat.set(mu, nu, v);
            j_mat.set(nu, mu, v);
        }
    }
    j_mat
}

/// Rayon-parallel exact-exchange contraction $K_{\mu\nu}[X] = \sum_{\lambda\sigma} (\mu\lambda|\nu\sigma) X_{\lambda\sigma}$
/// for a general (non-symmetric, symmetric, or antisymmetric) AO transition density $X$.
fn contract_k_gen(eri: &Tensor4D, x_ao: &Matrix2D) -> Matrix2D {
    let n = x_ao.rows;
    let n2 = n * n;
    let x_slice = &x_ao.data;
    let eri_slice = &eri.data;

    let data: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|mu| {
            let mut row_out = vec![0.0f64; n];
            for lam in 0..n {
                let x_row = &x_slice[lam * n..(lam + 1) * n];
                let base_ml = (mu * n + lam) * n2;
                for nu in 0..n {
                    let eri_row = &eri_slice[base_ml + nu * n..base_ml + (nu + 1) * n];
                    let mut s = 0.0;
                    for sig in 0..n {
                        s += eri_row[sig] * x_row[sig];
                    }
                    row_out[nu] += s;
                }
            }
            row_out
        })
        .collect();

    let mut k_mat = Matrix2D::zeros(n, n);
    k_mat.data = data;
    k_mat
}

/// Transforms an OV vector `u` (`n_occ * n_virt`) into AO transition density $D = C_{\text{occ}} U C_{\text{virt}}^T$,
/// plus its symmetric part $D_{\text{sym}} = \frac{1}{2}(D + D^T)$ and antisymmetric part $D_{\text{anti}} = \frac{1}{2}(D - D^T)$.
fn ov_to_ao_densities(
    u: &[f64],
    c_occ: &Matrix2D,
    c_virt_t: &Matrix2D,
    n_occ: usize,
    n_virt: usize,
) -> (Matrix2D, Matrix2D, Matrix2D) {
    let mut u_mat = Matrix2D::zeros(n_occ, n_virt);
    u_mat.data.copy_from_slice(u);
    let tmp = matmul(c_occ, &u_mat);
    let d = matmul(&tmp, c_virt_t);
    let n = d.rows;
    let mut d_sym = Matrix2D::zeros(n, n);
    let mut d_anti = Matrix2D::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let dij = d.get(i, j);
            let dji = d.get(j, i);
            let s = 0.5 * (dij + dji);
            let a = 0.5 * (dij - dji);
            d_sym.set(i, j, s);
            d_sym.set(j, i, s);
            d_anti.set(i, j, a);
            d_anti.set(j, i, -a);
        }
    }
    (d, d_sym, d_anti)
}

/// Projects an AO Fock-like response matrix $G$ (`n_ao x n_ao`) onto the MO OV block: $(C_{\text{occ}}^T G C_{\text{virt}})_{ia}$.
fn ao_to_ov(g_ao: &Matrix2D, c_occ_t: &Matrix2D, c_virt: &Matrix2D) -> Vec<f64> {
    let tmp = matmul(c_occ_t, g_ao);
    matmul(&tmp, c_virt).data
}

/// Canonicalizes the overall $\pm 1$ phase of `(X, Y)` so the largest-magnitude element of `X` is positive.
fn canonicalize_mode_phase(mode: &mut EigenMode) {
    let mut best_idx = 0usize;
    let mut best_abs = -1.0f64;
    for (idx, &v) in mode.x.iter().enumerate() {
        let a = v.abs();
        if a > best_abs + 1e-14 {
            best_abs = a;
            best_idx = idx;
        }
    }
    if !mode.x.is_empty() && mode.x[best_idx] < 0.0 {
        for v in &mut mode.x {
            *v = -*v;
        }
        for v in &mut mode.y {
            *v = -*v;
        }
    }
}

/// Builds an [`ExcitedState`] record from a converged [`EigenMode`].
#[allow(clippy::too_many_arguments)]
fn build_excited_state(
    mut mode: EigenMode,
    manifold_index: usize,
    triplet: bool,
    ground_energy_hartree: f64,
    n_occ: usize,
    n_virt: usize,
    c_occ: &Matrix2D,
    c_virt_t: &Matrix2D,
    ao_dipoles: &[Matrix2D; 3],
) -> ExcitedState {
    canonicalize_mode_phase(&mut mode);
    let omega_eh = mode.omega;
    let omega_ev = omega_eh * HARTREE_TO_EV;
    let wavelength_nm = if omega_ev > 1e-12 {
        EV_TO_NM / omega_ev
    } else {
        f64::INFINITY
    };

    // Length-gauge electric transition dipole moment (zero by spin orthogonality for triplets):
    // mu_{0k, q} = -sqrt(2) * Tr[ d_q^{AO} * C_occ * (X_k + Y_k) * C_virt^T ]
    let mut tdip = [0.0f64; 3];
    if !triplet {
        let mut xpy_mat = Matrix2D::zeros(n_occ, n_virt);
        for idx in 0..(n_occ * n_virt) {
            xpy_mat.data[idx] = mode.x[idx] + mode.y[idx];
        }
        let d_tr = matmul(&matmul(c_occ, &xpy_mat), c_virt_t);
        let pref = -std::f64::consts::SQRT_2;
        for q in 0..3 {
            let s: f64 = ao_dipoles[q]
                .data
                .iter()
                .zip(d_tr.data.iter())
                .map(|(a, b)| a * b)
                .sum();
            tdip[q] = pref * s;
        }
    }
    let dip_sq = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
    let osc = (2.0 / 3.0) * omega_eh * dip_sq;

    // Dominant canonical orbital transitions
    let mut trans: Vec<OrbitalTransition> = Vec::with_capacity(n_occ * n_virt);
    for i in 0..n_occ {
        for a in 0..n_virt {
            let idx = i * n_virt + a;
            let xi = mode.x[idx];
            let yi = mode.y[idx];
            let w = xi * xi - yi * yi;
            trans.push(OrbitalTransition {
                occ_index: i,
                virt_index: n_occ + a,
                amplitude_x: xi,
                amplitude_y: yi,
                weight: w,
            });
        }
    }
    trans.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));

    let mut dominant = Vec::new();
    for (idx, t) in trans.into_iter().enumerate() {
        if idx == 0 || (idx < 6 && t.weight.abs() >= 0.02) {
            dominant.push(t);
        } else {
            break;
        }
    }

    let multiplicity = if triplet { 3 } else { 1 };
    let label = if triplet {
        format!("T{manifold_index}")
    } else {
        format!("S{manifold_index}")
    };

    ExcitedState {
        state_index: 0,
        manifold_index,
        label,
        multiplicity,
        excitation_energy_hartree: omega_eh,
        excitation_energy_ev: omega_ev,
        total_energy_hartree: ground_energy_hartree + omega_eh,
        wavelength_nm,
        transition_dipole_au: tdip,
        oscillator_strength: osc,
        dominant_transitions: dominant,
    }
}

/// Runs CIS (`tda = true`) or TDHF (`tda = false`) from geometry on a closed-shell molecule.
pub fn run_cis(
    mol: &Molecule,
    basis: &BasisSet,
    opts: &TddftOptions,
) -> Result<TddftResult, CalcError> {
    let mut hf_opts = opts.clone();
    let wfn = rhf(mol, basis, &hf_opts.scf_options)?;
    if opts.n_states == 0 {
        hf_opts.n_states = 1;
    }
    run_tddft_from_scf(mol, basis, &wfn, None, &hf_opts)
}

/// Runs TD-DFT (`tda = false`) or TDA-DFT (`tda = true`) — or CIS/TDHF when `xc` is `None` —
/// end-to-end starting from molecular geometry.
pub fn run_tddft(
    mol: &Molecule,
    basis: &BasisSet,
    xc: Option<XcFunctional>,
    opts: &TddftOptions,
) -> Result<TddftResult, CalcError> {
    if opts.n_states == 0 {
        return Err(CalcError::InvalidMolecule(
            "TddftOptions.n_states must be >= 1".into(),
        ));
    }
    match xc {
        None => {
            let wfn = rhf(mol, basis, &opts.scf_options)?;
            run_tddft_from_scf(mol, basis, &wfn, None, opts)
        }
        Some(func) => {
            let grid = MolecularGrid::build(mol, opts.grid_level);
            let wfn = rks_with_grid(mol, basis, func, &grid, &opts.scf_options)?;
            run_tddft_from_scf_with_grid(mol, basis, &wfn, Some((func, &grid)), opts)
        }
    }
}

/// Runs linear-response excited states atop an already-converged closed-shell [`ScfWavefunction`].
pub fn run_tddft_from_scf(
    mol: &Molecule,
    basis: &BasisSet,
    wfn: &ScfWavefunction,
    xc: Option<XcFunctional>,
    opts: &TddftOptions,
) -> Result<TddftResult, CalcError> {
    if opts.n_states == 0 {
        return Err(CalcError::InvalidMolecule(
            "TddftOptions.n_states must be >= 1".into(),
        ));
    }
    match xc {
        None => run_tddft_from_scf_with_grid(mol, basis, wfn, None, opts),
        Some(func) => {
            let grid = MolecularGrid::build(mol, opts.grid_level);
            run_tddft_from_scf_with_grid(mol, basis, wfn, Some((func, &grid)), opts)
        }
    }
}

fn run_tddft_from_scf_with_grid(
    mol: &Molecule,
    basis: &BasisSet,
    wfn: &ScfWavefunction,
    xc_and_grid: Option<(XcFunctional, &MolecularGrid)>,
    opts: &TddftOptions,
) -> Result<TddftResult, CalcError> {
    let (n_alpha, n_beta) = basis.validate_rhf_valence(mol)?;
    if n_alpha != n_beta {
        return Err(CalcError::UnsupportedMethod(
            "Closed-shell TD-DFT / CIS requires n_alpha == n_beta (multiplicity = 1)".into(),
        ));
    }
    let n_occ = n_alpha;
    let n_ao = basis.n_ao;
    let n_mo = wfn.n_mo;
    if n_occ == 0 || n_occ >= n_mo {
        return Err(CalcError::UnsupportedBasis(format!(
            "TD-DFT requires at least 1 occupied and 1 virtual MO (got n_occ={n_occ}, n_mo={n_mo})"
        )));
    }
    let n_virt = n_mo - n_occ;
    let n_ov = n_occ * n_virt;

    // Extract C_occ (n_ao x n_occ) and C_virt (n_ao x n_virt)
    let mut c_occ = Matrix2D::zeros(n_ao, n_occ);
    let mut c_virt = Matrix2D::zeros(n_ao, n_virt);
    for mu in 0..n_ao {
        let row = &wfn.mo_coefficients[mu * n_mo..(mu + 1) * n_mo];
        for i in 0..n_occ {
            c_occ.set(mu, i, row[i]);
        }
        for a in 0..n_virt {
            c_virt.set(mu, a, row[n_occ + a]);
        }
    }
    let c_occ_t = transpose(&c_occ);
    let c_virt_t = transpose(&c_virt);

    // Orbital energy differences diag[i * n_virt + a] = eps_{n_occ + a} - eps_i
    let mut diag = vec![0.0f64; n_ov];
    for i in 0..n_occ {
        let ei = wfn.mo_energies[i];
        for a in 0..n_virt {
            diag[i * n_virt + a] = wfn.mo_energies[n_occ + a] - ei;
        }
    }

    let c_hf = match xc_and_grid {
        None => 1.0,
        Some((func, _)) => func.exact_exchange_fraction(),
    };

    // Precompute 2e ERI tensor and adiabatic f_xc kernel once
    let need_singlet = matches!(opts.manifold, SpinManifold::Singlet | SpinManifold::Both);
    let need_triplet = matches!(opts.manifold, SpinManifold::Triplet | SpinManifold::Both);
    let need_eri = need_singlet || c_hf.abs() > 1e-15;
    let eri_opt: Option<Tensor4D> = if need_eri {
        Some(compute_eri(basis))
    } else {
        None
    };

    let fxc_opt: Option<FxcKernel> = xc_and_grid.map(|(func, grid)| {
        let mut p_tot = Matrix2D::zeros(n_ao, n_ao);
        p_tot.data.copy_from_slice(&wfn.density_alpha);
        FxcKernel::new(basis, grid, func, &p_tot)
    });

    let ao_dipoles = compute_ao_dipole_matrices(basis);

    let solve_manifold = |triplet: bool| -> Result<(Vec<ExcitedState>, usize), CalcError> {
        let (modes, iters) = if opts.tda {
            let apply_a = |vecs: &[Vec<f64>]| -> Vec<Vec<f64>> {
                vecs.iter()
                    .map(|u| {
                        let (d_gen, d_sym, _) =
                            ov_to_ao_densities(u, &c_occ, &c_virt_t, n_occ, n_virt);
                        let mut g_ao = Matrix2D::zeros(n_ao, n_ao);

                        if !triplet {
                            let j_mat = contract_j_sym(
                                eri_opt.as_ref().expect("ERI needed for singlet J"),
                                &d_sym,
                            );
                            for idx in 0..(n_ao * n_ao) {
                                g_ao.data[idx] += 2.0 * j_mat.data[idx];
                            }
                        }
                        if c_hf.abs() > 1e-15 {
                            let k_mat = contract_k_gen(
                                eri_opt.as_ref().expect("ERI needed for hybrid/HF K"),
                                &d_gen,
                            );
                            for idx in 0..(n_ao * n_ao) {
                                g_ao.data[idx] -= c_hf * k_mat.data[idx];
                            }
                        }
                        if let Some(ref fxc) = fxc_opt {
                            let v_fxc = fxc.contract(&d_sym, triplet);
                            for idx in 0..(n_ao * n_ao) {
                                g_ao.data[idx] += v_fxc.data[idx];
                            }
                        }

                        let mut au = ao_to_ov(&g_ao, &c_occ_t, &c_virt);
                        for idx in 0..n_ov {
                            au[idx] += diag[idx] * u[idx];
                        }
                        au
                    })
                    .collect()
            };

            solve_tda(
                &diag,
                opts.n_states,
                opts.max_iter,
                opts.conv_tol,
                opts.force_davidson,
                apply_a,
            )?
        } else {
            let apply_apb_amb = |vecs: &[Vec<f64>]| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
                let mut wp_all = Vec::with_capacity(vecs.len());
                let mut wm_all = Vec::with_capacity(vecs.len());

                for u in vecs {
                    let (_, d_sym, d_anti) =
                        ov_to_ao_densities(u, &c_occ, &c_virt_t, n_occ, n_virt);

                    // (A + B) * u: symmetric transition density response
                    let mut gp_ao = Matrix2D::zeros(n_ao, n_ao);
                    if !triplet {
                        let j_mat = contract_j_sym(
                            eri_opt.as_ref().expect("ERI needed for singlet J"),
                            &d_sym,
                        );
                        for idx in 0..(n_ao * n_ao) {
                            gp_ao.data[idx] += 4.0 * j_mat.data[idx];
                        }
                    }
                    if c_hf.abs() > 1e-15 {
                        let k_sym = contract_k_gen(
                            eri_opt.as_ref().expect("ERI needed for hybrid/HF K"),
                            &d_sym,
                        );
                        for idx in 0..(n_ao * n_ao) {
                            gp_ao.data[idx] -= 2.0 * c_hf * k_sym.data[idx];
                        }
                    }
                    if let Some(ref fxc) = fxc_opt {
                        let v_fxc = fxc.contract(&d_sym, triplet);
                        for idx in 0..(n_ao * n_ao) {
                            gp_ao.data[idx] += 2.0 * v_fxc.data[idx];
                        }
                    }
                    let mut apb_u = ao_to_ov(&gp_ao, &c_occ_t, &c_virt);
                    for idx in 0..n_ov {
                        apb_u[idx] += diag[idx] * u[idx];
                    }

                    // (A - B) * u: purely orbital difference + exact-exchange antisymmetric response!
                    let amb_u = if c_hf.abs() > 1e-15 {
                        let k_anti = contract_k_gen(
                            eri_opt.as_ref().expect("ERI needed for hybrid/HF K"),
                            &d_anti,
                        );
                        let mut gm_ao = Matrix2D::zeros(n_ao, n_ao);
                        for idx in 0..(n_ao * n_ao) {
                            gm_ao.data[idx] = -2.0 * c_hf * k_anti.data[idx];
                        }
                        let mut v = ao_to_ov(&gm_ao, &c_occ_t, &c_virt);
                        for idx in 0..n_ov {
                            v[idx] += diag[idx] * u[idx];
                        }
                        v
                    } else {
                        diag.iter().zip(u.iter()).map(|(&d, &ui)| d * ui).collect()
                    };

                    wp_all.push(apb_u);
                    wm_all.push(amb_u);
                }

                (wp_all, wm_all)
            };

            solve_rpa(
                &diag,
                opts.n_states,
                opts.max_iter,
                opts.conv_tol,
                opts.force_davidson,
                apply_apb_amb,
            )?
        };

        let built: Vec<ExcitedState> = modes
            .into_iter()
            .enumerate()
            .map(|(idx, m)| {
                build_excited_state(
                    m,
                    idx + 1,
                    triplet,
                    wfn.total_energy_hartree,
                    n_occ,
                    n_virt,
                    &c_occ,
                    &c_virt_t,
                    &ao_dipoles,
                )
            })
            .collect();

        Ok((built, iters))
    };

    let mut all_states = Vec::new();
    let mut iters_s = 0usize;
    let mut iters_t = 0usize;
    let mut s1_ev: Option<f64> = None;
    let mut t1_ev: Option<f64> = None;

    if need_singlet {
        let (s_states, it) = solve_manifold(false)?;
        iters_s = it;
        if let Some(first) = s_states.first() {
            s1_ev = Some(first.excitation_energy_ev);
        }
        all_states.extend(s_states);
    }
    if need_triplet {
        let (t_states, it) = solve_manifold(true)?;
        iters_t = it;
        if let Some(first) = t_states.first() {
            t1_ev = Some(first.excitation_energy_ev);
        }
        all_states.extend(t_states);
    }

    // Order combined state list by ascending vertical excitation energy
    all_states.sort_by(|a, b| {
        a.excitation_energy_hartree
            .total_cmp(&b.excitation_energy_hartree)
    });
    for (idx, st) in all_states.iter_mut().enumerate() {
        st.state_index = idx + 1;
    }

    let homo_eh = wfn.mo_energies[n_occ - 1];
    let lumo_eh = wfn.mo_energies[n_occ];
    let ks_gap_ev = (lumo_eh - homo_eh) * HARTREE_TO_EV;
    let delta_est_ev = match (s1_ev, t1_ev) {
        (Some(s), Some(t)) => Some(s - t),
        _ => None,
    };

    Ok(TddftResult {
        ground_energy_hartree: wfn.total_energy_hartree,
        xc_functional: xc_and_grid.map(|(f, _)| f.canonical_name().to_string()),
        tda: opts.tda,
        n_occ,
        n_virt,
        homo_energy_hartree: homo_eh,
        lumo_energy_hartree: lumo_eh,
        ks_gap_ev,
        optical_gap_ev: s1_ev,
        lowest_triplet_ev: t1_ev,
        delta_est_ev,
        davidson_iterations_singlet: iters_s,
        davidson_iterations_triplet: iters_t,
        states: all_states,
        scf_wavefunction: wfn.clone(),
    })
}

/// Generates a Gaussian-broadened UV-Vis molar decadic absorption spectrum $\varepsilon(\lambda)$
/// (`L mol^-1 cm^-1`) over `[min_nm, max_nm]` with `n_points` uniformly spaced wavelengths.
pub fn broaden_uv_vis_spectrum(
    states: &[ExcitedState],
    min_nm: f64,
    max_nm: f64,
    n_points: usize,
    fwhm_ev: f64,
) -> Result<UvVisSpectrum, CalcError> {
    if !min_nm.is_finite() || !max_nm.is_finite() || min_nm <= 0.0 || max_nm <= min_nm {
        return Err(CalcError::InvalidMolecule(format!(
            "Invalid UV-Vis wavelength window [{min_nm}, {max_nm}] nm"
        )));
    }
    if n_points < 2 {
        return Err(CalcError::InvalidMolecule(
            "UV-Vis spectrum requires n_points >= 2".into(),
        ));
    }
    if !fwhm_ev.is_finite() || fwhm_ev <= 0.0 {
        return Err(CalcError::InvalidMolecule(format!(
            "UV-Vis FWHM must be positive and finite, got {fwhm_ev} eV"
        )));
    }

    let fwhm_cm = fwhm_ev * EV_TO_INV_CM;
    let four_ln2 = 4.0 * std::f64::consts::LN_2;
    let inv_fwhm_sq = 1.0 / (fwhm_ev * fwhm_ev);
    let step = (max_nm - min_nm) / ((n_points - 1) as f64);

    let mut wavelengths_nm = Vec::with_capacity(n_points);
    let mut energies_ev = Vec::with_capacity(n_points);
    let mut epsilon = Vec::with_capacity(n_points);

    for k in 0..n_points {
        let lam = min_nm + (k as f64) * step;
        let e_ev = EV_TO_NM / lam;
        let mut eps_val = 0.0f64;
        for st in states {
            if st.oscillator_strength <= 0.0 {
                continue;
            }
            let de = e_ev - st.excitation_energy_ev;
            let gauss = (-four_ln2 * de * de * inv_fwhm_sq).exp();
            eps_val += UV_VIS_GAUSS_PREFACTOR * (st.oscillator_strength / fwhm_cm) * gauss;
        }
        wavelengths_nm.push(lam);
        energies_ev.push(e_ev);
        epsilon.push(eps_val);
    }

    Ok(UvVisSpectrum {
        wavelengths_nm,
        energies_ev,
        epsilon_l_mol_cm: epsilon,
        fwhm_ev,
    })
}

/// Evaluates the closed-shell Singlet (`triplet = false`) or Triplet (`triplet = true`)
/// SCF orbital stability Hessian quadratic form $U^T (A + B) U$ for an OV rotation vector `u_ov`.
pub fn orbital_hessian_apb_quad(
    mol: &Molecule,
    basis: &BasisSet,
    wfn: &ScfWavefunction,
    xc: Option<XcFunctional>,
    grid_level: GridLevel,
    u_ov: &[f64],
    triplet: bool,
) -> Result<f64, CalcError> {
    let (n_alpha, _) = basis.validate_rhf_valence(mol)?;
    let n_occ = n_alpha;
    let n_ao = basis.n_ao;
    let n_mo = wfn.n_mo;
    let n_virt = n_mo - n_occ;
    let n_ov = n_occ * n_virt;
    if u_ov.len() != n_ov {
        return Err(CalcError::InvalidMolecule(format!(
            "Expected OV vector of length {n_ov}, got {}",
            u_ov.len()
        )));
    }

    let mut c_occ = Matrix2D::zeros(n_ao, n_occ);
    let mut c_virt = Matrix2D::zeros(n_ao, n_virt);
    for mu in 0..n_ao {
        let row = &wfn.mo_coefficients[mu * n_mo..(mu + 1) * n_mo];
        for i in 0..n_occ {
            c_occ.set(mu, i, row[i]);
        }
        for a in 0..n_virt {
            c_virt.set(mu, a, row[n_occ + a]);
        }
    }
    let c_occ_t = transpose(&c_occ);
    let c_virt_t = transpose(&c_virt);

    let (_, d_sym, _) = ov_to_ao_densities(u_ov, &c_occ, &c_virt_t, n_occ, n_virt);
    let c_hf = xc.map_or(1.0, |f| f.exact_exchange_fraction());

    let mut gp_ao = Matrix2D::zeros(n_ao, n_ao);
    if !triplet || c_hf.abs() > 1e-15 {
        let eri = compute_eri(basis);
        if !triplet {
            let j_mat = contract_j_sym(&eri, &d_sym);
            for idx in 0..(n_ao * n_ao) {
                gp_ao.data[idx] += 4.0 * j_mat.data[idx];
            }
        }
        if c_hf.abs() > 1e-15 {
            let k_sym = contract_k_gen(&eri, &d_sym);
            for idx in 0..(n_ao * n_ao) {
                gp_ao.data[idx] -= 2.0 * c_hf * k_sym.data[idx];
            }
        }
    }

    if let Some(func) = xc {
        let grid = MolecularGrid::build(mol, grid_level);
        let mut p_tot = Matrix2D::zeros(n_ao, n_ao);
        p_tot.data.copy_from_slice(&wfn.density_alpha);
        let fxc = FxcKernel::new(basis, &grid, func, &p_tot);
        let v_fxc = fxc.contract(&d_sym, triplet);
        for idx in 0..(n_ao * n_ao) {
            gp_ao.data[idx] += 2.0 * v_fxc.data[idx];
        }
    }

    let mut apb_u = ao_to_ov(&gp_ao, &c_occ_t, &c_virt);
    for i in 0..n_occ {
        let ei = wfn.mo_energies[i];
        for a in 0..n_virt {
            let idx = i * n_virt + a;
            apb_u[idx] += (wfn.mo_energies[n_occ + a] - ei) * u_ov[idx];
        }
    }

    Ok(u_ov.iter().zip(apb_u.iter()).map(|(x, y)| x * y).sum())
}
