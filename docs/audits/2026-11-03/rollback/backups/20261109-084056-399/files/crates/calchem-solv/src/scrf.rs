//! Self-Consistent Reaction Field (SCRF) solver and post-SCF PCM solvation engines.

use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_rks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{compute_ecp, compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction};
use calchem_scf::diis::DiisState;
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D, Tensor4D};
use serde::{Deserialize, Serialize};

use crate::cavity::{CavityOptions, MolecularCavity};
use crate::matrix::BoundaryMatrix;
use crate::potential::CavityPotentialEngine;
use crate::solvent::{Solvent, SolventModel};

/// Conversion constant from Hartree to kcal/mol.
pub const HARTREE_TO_KCAL_MOL: f64 = 627.509474063;

/// Options controlling SCRF and continuum solvation calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvationOptions {
    /// Continuum solvent specification (dielectric constant, refractive index, surface tension).
    pub solvent: Solvent,
    /// Solvation boundary element model (C-PCM, COSMO, IEF-PCM, Onsager).
    pub model: SolventModel,
    /// Molecular cavity boundary tessellation parameters.
    pub cavity_opts: CavityOptions,
    /// Whether to renormalize total surface charge to satisfy Gauss's law (-f(eps) * Q_net).
    pub renormalize_charge: bool,
    /// Whether to include non-electrostatic cavitation-dispersion-solvent structural reorganization (CDS).
    pub include_cds: bool,
}

impl Default for SolvationOptions {
    fn default() -> Self {
        Self {
            solvent: Solvent::water(),
            model: SolventModel::Cpcm,
            cavity_opts: CavityOptions::default(),
            renormalize_charge: true,
            include_cds: true,
        }
    }
}

/// Comprehensive outcome of a continuum solvation calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvationResult {
    /// Total free energy in solution (Hartree).
    pub solvated_energy: f64,
    /// Reference gas-phase energy (Hartree).
    pub gas_energy: f64,
    /// Net solvation free energy $\Delta G_{\text{solv}} = G_{\text{sol}} - E_{\text{gas}}$ (Hartree).
    pub delta_g_solv: f64,
    /// Net solvation free energy (kcal/mol).
    pub delta_g_solv_kcal_mol: f64,
    /// Electrostatic solvation interaction free energy $\frac{1}{2} \mathbf{q}^T \mathbf{V}^{\text{solute}}$ (kcal/mol).
    pub delta_g_elec_kcal_mol: f64,
    /// Non-electrostatic cavitation-dispersion free energy (kcal/mol).
    pub delta_g_cds_kcal_mol: f64,
    /// Solvent-accessible cavity surface area (Angstrom^2).
    pub cavity_surface_area_angstrom2: f64,
    /// Net sum of apparent surface charges $\sum_k q_k$ (atomic units).
    pub total_surface_charge: f64,
    /// Discretized apparent surface charges $\{q_k\}$ on boundary tesserae.
    pub surface_charges: Vec<f64>,
    /// Coordinates of boundary tesserae centers (Bohr).
    pub tessera_coords_bohr: Vec<[f64; 3]>,
    /// Number of SCRF self-consistency iterations.
    pub scf_iterations: usize,
    /// Convergence status.
    pub converged: bool,
    /// Solvated converged wavefunction (molecular orbitals and density).
    pub wavefunction: ScfWavefunction,
}

/// Computes fast post-SCF PCM solvation from an existing converged gas-phase wavefunction.
pub fn compute_pcm_solvation(
    mol: &Molecule,
    basis: &BasisSet,
    gas_wf: &ScfWavefunction,
    opts: &SolvationOptions,
) -> Result<SolvationResult, CalcError> {
    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;

    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let n = basis.n_ao;
    let density_mat = Matrix2D {
        rows: n,
        cols: n,
        data: gas_wf.density_alpha.clone(),
    };

    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let v_solute = potential_engine.compute_solute_potential(&density_mat);
    let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;

    // Electrostatic solvation energy: 0.5 * q^T * V_solute
    let e_elec_hartree: f64 = 0.5 * q.iter().zip(v_solute.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();

    // Cavitation-dispersion non-electrostatic energy
    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let delta_g_solv_hartree = e_elec_hartree + cds_hartree;
    let solvated_energy = gas_wf.total_energy_hartree + delta_g_solv_hartree;

    let total_q: f64 = q.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy: gas_wf.total_energy_hartree,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_elec_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: gas_wf.diagnostics.iterations,
        converged: gas_wf.converged,
        wavefunction: gas_wf.clone(),
    })
}

/// Executes full closed-shell Restricted Hartree-Fock SCRF in solution.
pub fn run_scrf_rhf(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
) -> Result<SolvationResult, CalcError> {
    let n = basis.n_ao;
    let n_elec = mol.nelectron()?;
    if n_elec % 2 != 0 {
        return Err(CalcError::InvalidMolecule(
            "Closed-shell RHF SCRF requires an even number of electrons".into(),
        ));
    }
    let n_occ = n_elec / 2;

    // 1. One- and two-electron integrals
    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // 2. Cavity and boundary operator setup
    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    // 3. Symmetric orthogonalizer X = S^{-1/2}
    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 0.0 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is not positive definite ({val:.3e})"
            )));
        }
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    // Initial guess from core Hamiltonian
    let (_, _, mut p) = diagonalize_fock_rhf(&h_core, &x, &xt, n_occ)?;

    let mut diis = DiisState::new(scf_opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut v_solute_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        // Solute potential on cavity tesserae
        let v_solute = potential_engine.compute_solute_potential(&p);
        let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;

        // Reaction field operator V_rf = -sum_k q_k V_k
        let v_rf = potential_engine.compute_reaction_field_matrix(&q);

        // Bare gas-phase Fock build F_0 = H_core + G[P]
        let f_0 = build_fock_rhf(&h_core, &p, &eri);

        // Effective Fock matrix in SCRF: F = F_0 + V_rf
        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            f.data[i] = f_0.data[i] + v_rf.data[i];
        }

        // Total electronic energy: 0.5 * Tr(P * (H_core + F_0)) + 0.5 * q^T * V_solute
        let e_gas_like = compute_electronic_energy_rhf(&p, &h_core, &f_0);
        let e_solv_elec: f64 = 0.5 * q.iter().zip(v_solute.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();
        let e_tot = e_gas_like + e_nuc + e_solv_elec;
        energy_history.push(e_tot);

        // Pulay commutator error: e = F P S - S P F
        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = diis.push_and_extrapolate(&f, &err);
        let (evals, c, p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        v_solute_converged = v_solute;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    // Cavitation-dispersion
    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let e_solv_elec: f64 = 0.5 * q_converged.iter().zip(v_solute_converged.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();
    let delta_g_solv_hartree = e_solv_elec + cds_hartree;
    let solvated_energy = e_prev + cds_hartree;
    let gas_energy = e_prev - e_solv_elec;

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: solvated_energy,
        electronic_energy_hartree: e_prev - e_nuc,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_evals,
        mo_occupations,
        mo_coefficients: final_coeffs.data,
        density_alpha: p.data,
        density_beta: None,
        fock_alpha: final_fock.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: diag,
    };

    let total_q: f64 = q_converged.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_solv_elec * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q_converged,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: used_iter,
        converged,
        wavefunction: wf,
    })
}

/// Executes full closed-shell Restricted Kohn-Sham (RKS) SCRF in solution.
pub fn run_scrf_rks(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
    xc: XcFunctional,
) -> Result<SolvationResult, CalcError> {
    let n = basis.n_ao;
    let n_elec = mol.nelectron()?;
    if n_elec % 2 != 0 {
        return Err(CalcError::InvalidMolecule(
            "Closed-shell RKS SCRF requires an even number of electrons".into(),
        ));
    }
    let n_occ = n_elec / 2;

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let eri = compute_eri(basis);
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let grid = MolecularGrid::build(mol, GridLevel::Fine);

    let cavity = MolecularCavity::build(mol, &opts.cavity_opts);
    let boundary_matrix = BoundaryMatrix::build(&cavity)?;
    let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
    let f_eps = opts.model.dielectric_factor(opts.solvent.epsilon);
    let net_charge = if opts.renormalize_charge {
        Some(mol.charge as f64)
    } else {
        None
    };

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val > 1e-10 {
            s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
        }
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let (_, _, mut p) = diagonalize_fock_rhf(&h_core, &x, &xt, n_occ)?;

    let mut diis = DiisState::new(scf_opts.diis_max_vecs);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut v_solute_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        let v_solute = potential_engine.compute_solute_potential(&p);
        let q = boundary_matrix.solve_surface_charges(&v_solute, f_eps, net_charge)?;
        let v_rf = potential_engine.compute_reaction_field_matrix(&q);

        let g_2e = build_two_electron_rks(&p, &eri, xc.exact_exchange_fraction());
        let (e_xc, v_xc) = evaluate_vxc_rks(basis, &grid, xc, &p);

        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i] + v_rf.data[i];
        }

        // Electronic energy in RKS: Tr(P*H_core) + 0.5*Tr(P*J) - 0.5*c_HF*Tr(P*K) + E_xc + 0.5*q^T*V_solute
        let mut e_1e = 0.0;
        let mut e_2e = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                let p_val = p.get(mu, nu);
                e_1e += p_val * h_core.get(mu, nu);
                e_2e += 0.5 * p_val * g_2e.get(mu, nu);
            }
        }
        let e_solv_elec: f64 = 0.5 * q.iter().zip(v_solute.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();
        let e_tot = e_1e + e_2e + e_xc + e_nuc + e_solv_elec;
        energy_history.push(e_tot);

        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = diis.push_and_extrapolate(&f, &err);
        let (evals, c, p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        v_solute_converged = v_solute;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    let cds_kcal_mol = if opts.include_cds {
        opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity.total_area_angstrom2
    } else {
        0.0
    };
    let cds_hartree = cds_kcal_mol / HARTREE_TO_KCAL_MOL;

    let e_solv_elec: f64 = 0.5 * q_converged.iter().zip(v_solute_converged.iter()).map(|(&qk, &vk)| qk * vk).sum::<f64>();
    let delta_g_solv_hartree = e_solv_elec + cds_hartree;
    let solvated_energy = e_prev + cds_hartree;
    let gas_energy = e_prev - e_solv_elec;

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        converged,
        iterations: used_iter,
        total_energy_hartree: solvated_energy,
        electronic_energy_hartree: e_prev - e_nuc,
        nuclear_repulsion_hartree: e_nuc,
        n_ao: n,
        n_mo: n,
        mo_energies: final_evals,
        mo_occupations,
        mo_coefficients: final_coeffs.data,
        density_alpha: p.data,
        density_beta: None,
        fock_alpha: final_fock.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: diag,
    };

    let total_q: f64 = q_converged.iter().sum();
    let tessera_coords: Vec<[f64; 3]> = cavity.tesserae.iter().map(|t| t.position_bohr).collect();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_solv_elec * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: cavity.total_area_angstrom2,
        total_surface_charge: total_q,
        surface_charges: q_converged,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: used_iter,
        converged,
        wavefunction: wf,
    })
}

/// Unified entrypoint for SCRF solvation calculations across HF and DFT.
pub fn run_scrf(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
    method: &str,
    xc: Option<XcFunctional>,
) -> Result<SolvationResult, CalcError> {
    let m = method.trim().to_ascii_lowercase();
    match m.as_str() {
        "rhf" | "hf" => run_scrf_rhf(mol, basis, scf_opts, opts),
        "rks" | "dft" | "b3lyp" | "pbe" | "blyp" | "svwn5" | "pbe0" => {
            let func = xc.unwrap_or(XcFunctional::B3lyp);
            run_scrf_rks(mol, basis, scf_opts, opts, func)
        }
        other => Err(CalcError::UnsupportedMethod(format!(
            "Unsupported SCRF method: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_fock_rhf(h_core: &Matrix2D, p: &Matrix2D, eri: &Tensor4D) -> Matrix2D {
    let n = h_core.rows;
    let mut f = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut g_munu = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let p_ls = p.get(lam, sig);
                    let j = eri.get(mu, nu, lam, sig);
                    let k = eri.get(mu, lam, nu, sig);
                    g_munu += p_ls * (j - 0.5 * k);
                }
            }
            f.set(mu, nu, h_core.get(mu, nu) + g_munu);
        }
    }
    f
}

fn compute_electronic_energy_rhf(p: &Matrix2D, h_core: &Matrix2D, fock: &Matrix2D) -> f64 {
    let n = h_core.rows;
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * p.get(mu, nu) * (h_core.get(mu, nu) + fock.get(mu, nu));
        }
    }
    e_elec
}

fn diagonalize_fock_rhf(
    fock: &Matrix2D,
    x: &Matrix2D,
    xt: &Matrix2D,
    n_occ: usize,
) -> Result<(Vec<f64>, Matrix2D, Matrix2D), CalcError> {
    let n = fock.rows;
    let fp = matmul(&matmul(xt, fock), x);
    let (eps, cp) = eigh_sym(&fp, 100, 1e-14)?;
    let c = matmul(x, &cp);
    let mut p = Matrix2D::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for m in 0..n_occ {
                sum += c.get(mu, m) * c.get(nu, m);
            }
            p.set(mu, nu, 2.0 * sum);
        }
    }
    Ok((eps, c, p))
}

fn build_two_electron_rks(p: &Matrix2D, eri: &Tensor4D, exx_fraction: f64) -> Matrix2D {
    let n = p.rows;
    let mut g = Matrix2D::zeros(n, n);
    if exx_fraction.abs() < 1e-15 {
        for mu in 0..n {
            for nu in mu..n {
                let mut j_sum = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        j_sum += p.get(lam, sig) * eri.get(mu, nu, lam, sig);
                    }
                }
                g.set(mu, nu, j_sum);
                g.set(nu, mu, j_sum);
            }
        }
    } else {
        let half_c = 0.5 * exx_fraction;
        for mu in 0..n {
            for nu in mu..n {
                let mut g_munu = 0.0;
                for lam in 0..n {
                    for sig in 0..n {
                        let p_ls = p.get(lam, sig);
                        let j = eri.get(mu, nu, lam, sig);
                        let k = eri.get(mu, lam, nu, sig);
                        g_munu += p_ls * (j - half_c * k);
                    }
                }
                g.set(mu, nu, g_munu);
                g.set(nu, mu, g_munu);
            }
        }
    }
    g
}
