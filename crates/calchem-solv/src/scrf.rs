//! Self-Consistent Reaction Field (SCRF) solver and post-SCF PCM solvation engines.

use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_rks, evaluate_vxc_uks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{
    compute_ecp, compute_kinetic, compute_nuclear, compute_overlap, overlap_primitive, GaussianPair,
};
use calchem_model::{CalcError, Molecule, ScfDiagnostics, ScfOptions, ScfWavefunction, BOHR_TO_ANGSTROM};
use calchem_scf::{
    diis::DiisState, should_use_direct, validate_direct_resources_and_options,
    validate_open_shell_stored_resources, validate_resources_and_options, RksFockBuilder,
    UksFockBuilder,
};
use calchem_tensor::{eigh_sym, frobenius_norm, matmul, rms_diff, transpose, Matrix2D};
use serde::{Deserialize, Serialize};

use crate::cavity::{bondi_vdw_radius, CavityOptions, CavityRadiiMode, MolecularCavity};
use crate::matrix::BoundaryMatrix;
use crate::potential::CavityPotentialEngine;
use crate::smd::compute_smd_cds_energy;
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

enum ReactionFieldSolver {
    Boundary {
        cavity: MolecularCavity,
        boundary_matrix: BoundaryMatrix,
        potential_engine: CavityPotentialEngine,
        f_eps_scale: f64,
        net_charge: Option<f64>,
    },
    Onsager {
        cavity_area_a2: f64,
        g_factor: f64,
        dip_x: Matrix2D,
        dip_y: Matrix2D,
        dip_z: Matrix2D,
        nuc_dipole: [f64; 3],
    },
}

impl ReactionFieldSolver {
    fn build(mol: &Molecule, basis: &BasisSet, opts: &SolvationOptions) -> Result<Self, CalcError> {
        if opts.model == SolventModel::Onsager {
            // Spherical cavity centered at nuclear center of charge, radius = max(|r_A - origin| + R_vdW) + 0.5 A
            let mut sum_z = 0.0;
            let mut origin = [0.0f64; 3];
            for atom in &mol.atoms {
                let z = atom.z as f64;
                sum_z += z;
                for k in 0..3 {
                    origin[k] += z * atom.position_bohr[k];
                }
            }
            if sum_z > 0.0 {
                for k in 0..3 {
                    origin[k] /= sum_z;
                }
            }
            let ang_to_bohr = 1.0 / BOHR_TO_ANGSTROM;
            let mut a0_bohr = 1.0f64;
            for atom in &mol.atoms {
                let dx = atom.position_bohr[0] - origin[0];
                let dy = atom.position_bohr[1] - origin[1];
                let dz = atom.position_bohr[2] - origin[2];
                let r_atom = (dx * dx + dy * dy + dz * dz).sqrt()
                    + (bondi_vdw_radius(atom.z) + 0.5) * ang_to_bohr;
                if r_atom > a0_bohr {
                    a0_bohr = r_atom;
                }
            }
            let eps = opts.solvent.epsilon;
            let g_factor = if (eps - 1.0).abs() < 1e-12 {
                0.0
            } else {
                2.0 * (eps - 1.0) / ((2.0 * eps + 1.0) * a0_bohr.powi(3))
            };
            let [dip_x, dip_y, dip_z] = compute_dipole_matrices(basis, origin);
            let mut nuc_dipole = [0.0f64; 3];
            for atom in &mol.atoms {
                let z = atom.z as f64;
                for k in 0..3 {
                    nuc_dipole[k] += z * (atom.position_bohr[k] - origin[k]);
                }
            }
            let a0_ang = a0_bohr * BOHR_TO_ANGSTROM;
            let cavity_area_a2 = 4.0 * std::f64::consts::PI * a0_ang * a0_ang;
            return Ok(Self::Onsager {
                cavity_area_a2,
                g_factor,
                dip_x,
                dip_y,
                dip_z,
                nuc_dipole,
            });
        }

        let eff_cav_opts = if opts.model == SolventModel::Smd {
            CavityOptions {
                scaling_factor: 1.0,
                probe_radius_angstrom: 0.0,
                radii_mode: CavityRadiiMode::SmdIntrinsic {
                    alpha: opts.solvent.smd.alpha,
                },
                ..opts.cavity_opts
            }
        } else {
            opts.cavity_opts
        };

        let cavity = MolecularCavity::build(mol, &eff_cav_opts);
        let boundary_matrix = BoundaryMatrix::build_for_model(&cavity, opts.model, opts.solvent.epsilon)?;
        let potential_engine = CavityPotentialEngine::build(mol, basis, &cavity)?;
        let f_eps_scale = match opts.model {
            SolventModel::IefPcm | SolventModel::Smd => 1.0,
            _ => opts.model.dielectric_factor(opts.solvent.epsilon),
        };
        let net_charge = if opts.renormalize_charge {
            Some(mol.charge as f64)
        } else {
            None
        };
        Ok(Self::Boundary {
            cavity,
            boundary_matrix,
            potential_engine,
            f_eps_scale,
            net_charge,
        })
    }

    fn reaction_field(&self, density_total: &Matrix2D) -> Result<(f64, Matrix2D, Vec<f64>), CalcError> {
        let n = density_total.rows;
        match self {
            Self::Boundary {
                boundary_matrix,
                potential_engine,
                f_eps_scale,
                net_charge,
                ..
            } => {
                let v_solute = potential_engine.compute_solute_potential(density_total);
                let q = boundary_matrix.solve_surface_charges(&v_solute, *f_eps_scale, *net_charge)?;
                let v_rf = potential_engine.compute_reaction_field_matrix(&q);
                let e_solv_elec: f64 = 0.5
                    * q.iter()
                        .zip(v_solute.iter())
                        .map(|(&qk, &vk)| qk * vk)
                        .sum::<f64>();
                Ok((e_solv_elec, v_rf, q))
            }
            Self::Onsager {
                g_factor,
                dip_x,
                dip_y,
                dip_z,
                nuc_dipole,
                ..
            } => {
                let mut mu = *nuc_dipole;
                for k in 0..(n * n) {
                    let p_val = density_total.data[k];
                    mu[0] -= p_val * dip_x.data[k];
                    mu[1] -= p_val * dip_y.data[k];
                    mu[2] -= p_val * dip_z.data[k];
                }
                let mu2 = mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2];
                let e_onsager = -0.5 * g_factor * mu2;
                let mut v_rf = Matrix2D::zeros(n, n);
                // Fock contribution of V_rf = d(E_onsager)/dP = +g * mu . r_AO (since mu_e = -Tr(P r))
                for k in 0..(n * n) {
                    v_rf.data[k] = g_factor
                        * (mu[0] * dip_x.data[k] + mu[1] * dip_y.data[k] + mu[2] * dip_z.data[k]);
                }
                Ok((e_onsager, v_rf, Vec::new()))
            }
        }
    }

    fn cavity_area_and_coords(&self) -> (f64, Vec<[f64; 3]>) {
        match self {
            Self::Boundary { cavity, .. } => (
                cavity.total_area_angstrom2,
                cavity.tesserae.iter().map(|t| t.position_bohr).collect(),
            ),
            Self::Onsager { cavity_area_a2, .. } => (*cavity_area_a2, Vec::new()),
        }
    }
}

fn compute_cds_energy(
    mol: &Molecule,
    opts: &SolvationOptions,
    cavity_area_a2: f64,
) -> (f64, f64, f64) {
    if !opts.include_cds && opts.model != SolventModel::Smd {
        return (0.0, 0.0, cavity_area_a2);
    }
    if opts.model == SolventModel::Smd {
        let (g_cds_eh, sasa_a2) = compute_smd_cds_energy(mol, &opts.solvent);
        return (g_cds_eh, g_cds_eh * HARTREE_TO_KCAL_MOL, sasa_a2);
    }
    let cds_kcal = opts.solvent.surface_tension_kcal_mol_angstrom2() * cavity_area_a2;
    (cds_kcal / HARTREE_TO_KCAL_MOL, cds_kcal, cavity_area_a2)
}

/// Computes fast post-SCF PCM solvation from an existing converged gas-phase wavefunction.
pub fn compute_pcm_solvation(
    mol: &Molecule,
    basis: &BasisSet,
    gas_wf: &ScfWavefunction,
    opts: &SolvationOptions,
) -> Result<SolvationResult, CalcError> {
    opts.model.validate_native()?;
    let rf = ReactionFieldSolver::build(mol, basis, opts)?;

    let n = basis.n_ao;
    let total_density: Vec<f64> = match &gas_wf.density_beta {
        Some(pb) => gas_wf
            .density_alpha
            .iter()
            .zip(pb.iter())
            .map(|(a, b)| a + b)
            .collect(),
        None => gas_wf.density_alpha.clone(),
    };
    let density_mat = Matrix2D {
        rows: n,
        cols: n,
        data: total_density,
    };

    let (e_elec_hartree, _v_rf, q) = rf.reaction_field(&density_mat)?;
    let (cav_area_a2, tessera_coords) = rf.cavity_area_and_coords();
    let (cds_hartree, cds_kcal_mol, reported_area_a2) =
        compute_cds_energy(mol, opts, cav_area_a2);

    let delta_g_solv_hartree = e_elec_hartree + cds_hartree;
    let solvated_energy = gas_wf.total_energy_hartree + delta_g_solv_hartree;
    let total_q: f64 = q.iter().sum();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy: gas_wf.total_energy_hartree,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: e_elec_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: reported_area_a2,
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
    opts.model.validate_native()?;
    let use_direct = should_use_direct(basis, scf_opts);
    if use_direct {
        validate_direct_resources_and_options(basis, scf_opts)?;
    } else {
        validate_resources_and_options(basis, scf_opts)?;
    }
    let n = basis.n_ao;
    let n_elec = mol.nelectron()?;
    if n_elec % 2 != 0 {
        return Err(CalcError::InvalidMolecule(
            "Closed-shell RHF SCRF requires an even number of electrons".into(),
        ));
    }
    let n_occ = n_elec / 2;

    // 1. One-electron integrals and Fock builder (Stored / Direct / RI-J)
    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }
    let fock_builder = RksFockBuilder::new_with_options(
        mol,
        basis,
        use_direct,
        scf_opts.density_fitting,
        scf_opts.use_gpu,
        None,
    )?;
    let e_nuc = basis.nuclear_repulsion_energy(mol);

    // 2. Reaction field solver setup (C-PCM / COSMO / IEF-PCM / SMD / Onsager)
    let rf = ReactionFieldSolver::build(mol, basis, opts)?;

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

    let mut diis = DiisState::from_scf_options(scf_opts);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        let (e_solv_elec, v_rf, q) = rf.reaction_field(&p)?;
        let g_2e = fock_builder.build_2e(basis, &p, 1.0, None);

        let mut f_0 = Matrix2D::zeros(n, n);
        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            let f0_val = h_core.data[i] + g_2e.data[i];
            f_0.data[i] = f0_val;
            f.data[i] = f0_val + v_rf.data[i];
        }

        let e_gas_like = compute_electronic_energy_rhf(&p, &h_core, &f_0);
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

        let f_extrap = diis
            .try_push_and_extrapolate(&f, &err)
            .map_err(|e| CalcError::Internal(format!("SCRF DIIS disk spill I/O error: {e}")))?;
        let (evals, c, mut p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if scf_opts.damping > 0.0 && scf_opts.damping < 1.0 {
            scf_opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    let (cav_area_a2, tessera_coords) = rf.cavity_area_and_coords();
    let (cds_hartree, cds_kcal_mol, reported_area_a2) =
        compute_cds_energy(mol, opts, cav_area_a2);

    let solvated_energy = e_prev + cds_hartree;
    let gas_energy = calchem_scf::rhf(mol, basis, scf_opts)?.total_energy_hartree;
    let delta_g_solv_hartree = solvated_energy - gas_energy;
    let (gpu_k, gpu_fb, rij_trunc) = fock_builder.telemetry();

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
        gpu_kernel_invocations: gpu_k,
        gpu_fallback_invocations: gpu_fb,
        diis_disk_backed: diis.is_disk_backed(),
        rij_metric_truncated_modes: rij_trunc,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        density_fitting: scf_opts.density_fitting,
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

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: reported_area_a2,
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
    opts.model.validate_native()?;
    let use_direct = should_use_direct(basis, scf_opts);
    if use_direct {
        validate_direct_resources_and_options(basis, scf_opts)?;
    } else {
        validate_resources_and_options(basis, scf_opts)?;
    }
    xc.validate_native()?;
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
    let fock_builder = RksFockBuilder::new_with_options(
        mol,
        basis,
        use_direct,
        scf_opts.density_fitting,
        scf_opts.use_gpu,
        xc.range_separated_params(),
    )?;
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let e_disp = if xc == XcFunctional::Wb97xd {
        calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?.0
    } else {
        0.0
    };
    let grid = MolecularGrid::build_for_basis(mol, basis, GridLevel::Fine);

    let rf = ReactionFieldSolver::build(mol, basis, opts)?;

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

    let mut diis = DiisState::from_scf_options(scf_opts);
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let mut e_prev = 0.0;
    let mut converged = false;
    let mut used_iter = 0;
    let mut q_converged = Vec::new();
    let mut final_evals = Vec::new();
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fock = Matrix2D::zeros(n, n);

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;

        let (e_solv_elec, v_rf, q) = rf.reaction_field(&p)?;
        let g_2e = fock_builder.build_2e(
            basis,
            &p,
            xc.exact_exchange_fraction(),
            xc.range_separated_params(),
        );
        let (e_xc, v_xc) = evaluate_vxc_rks(basis, &grid, xc, &p);

        let mut f = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            f.data[i] = h_core.data[i] + g_2e.data[i] + v_xc.data[i] + v_rf.data[i];
        }

        let mut e_1e = 0.0;
        let mut e_2e = 0.0;
        for mu in 0..n {
            for nu in 0..n {
                let p_val = p.get(mu, nu);
                e_1e += p_val * h_core.get(mu, nu);
                e_2e += 0.5 * p_val * g_2e.get(mu, nu);
            }
        }
        let e_tot = e_1e + e_2e + e_xc + e_nuc + e_disp + e_solv_elec;
        energy_history.push(e_tot);

        let fps = matmul(&matmul(&f, &p), &s);
        let spf = matmul(&matmul(&s, &p), &f);
        let mut err = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            err.data[i] = fps.data[i] - spf.data[i];
        }
        let diis_err = frobenius_norm(&err);
        diis_err_history.push(diis_err);

        let f_extrap = diis
            .try_push_and_extrapolate(&f, &err)
            .map_err(|e| CalcError::Internal(format!("SCRF DIIS disk spill I/O error: {e}")))?;
        let (evals, c, mut p_new) = diagonalize_fock_rhf(&f_extrap, &x, &xt, n_occ)?;

        let effective_damping = if scf_opts.damping > 0.0 && scf_opts.damping < 1.0 {
            scf_opts.damping
        } else if n >= 40 {
            0.3
        } else {
            0.0
        };
        if effective_damping > 0.0 {
            for idx in 0..p_new.data.len() {
                p_new.data[idx] =
                    (1.0 - effective_damping) * p_new.data[idx] + effective_damping * p.data[idx];
            }
        }

        let delta_e = (e_tot - e_prev).abs();
        let rms_d = rms_diff(&p_new, &p);
        rms_history.push(rms_d);

        e_prev = e_tot;
        p = p_new;
        q_converged = q;
        final_evals = evals;
        final_coeffs = c;
        final_fock = f;

        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density {
            converged = true;
            break;
        }
    }

    let (cav_area_a2, tessera_coords) = rf.cavity_area_and_coords();
    let (cds_hartree, cds_kcal_mol, reported_area_a2) =
        compute_cds_energy(mol, opts, cav_area_a2);

    let solvated_energy = e_prev + cds_hartree;
    let gas_energy =
        calchem_scf::rks_with_grid(mol, basis, xc, &grid, scf_opts)?.total_energy_hartree;
    let delta_g_solv_hartree = solvated_energy - gas_energy;
    let (gpu_k, gpu_fb, rij_trunc) = fock_builder.telemetry();

    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
        gpu_kernel_invocations: gpu_k,
        gpu_fallback_invocations: gpu_fb,
        diis_disk_backed: diis.is_disk_backed(),
        rij_metric_truncated_modes: rij_trunc,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_occ) {
        *occ = 2.0;
    }

    let wf = ScfWavefunction {
        density_fitting: scf_opts.density_fitting,
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

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: reported_area_a2,
        total_surface_charge: total_q,
        surface_charges: q_converged,
        tessera_coords_bohr: tessera_coords,
        scf_iterations: used_iter,
        converged,
        wavefunction: wf,
    })
}

/// Runs a Self-Consistent Reaction Field (SCRF) unrestricted UHF (`xc = None`) or UKS (`xc = Some(...)`)
/// calculation for open-shell molecules.
pub fn run_scrf_unrestricted(
    mol: &Molecule,
    basis: &BasisSet,
    scf_opts: &ScfOptions,
    opts: &SolvationOptions,
    xc: Option<XcFunctional>,
) -> Result<SolvationResult, CalcError> {
    opts.model.validate_native()?;
    if let Some(func) = xc {
        func.validate_native()?;
    }
    let use_direct = should_use_direct(basis, scf_opts);
    if use_direct {
        validate_direct_resources_and_options(basis, scf_opts)?;
    } else {
        validate_open_shell_stored_resources(basis, scf_opts)?;
    }

    let (n_alpha, n_beta) = basis.validate_uhf_valence(mol)?;
    let n = basis.n_ao;
    if n == 0 || n_alpha > n || n_beta > n {
        return Err(CalcError::UnsupportedBasis(format!(
            "Insufficient AO basis functions ({n}) for ({n_alpha}, {n_beta}) electrons"
        )));
    }

    let grid_opt = xc.map(|_| MolecularGrid::build_for_basis(mol, basis, GridLevel::Medium));
    let gas_wf = match (xc, &grid_opt) {
        (None, _) => calchem_scf::uhf(mol, basis, scf_opts)?,
        (Some(func), Some(grid)) => calchem_scf::uks_with_grid(mol, basis, func, grid, scf_opts)?,
        _ => unreachable!(),
    };
    let gas_energy = gas_wf.total_energy_hartree;

    let s = compute_overlap(basis);
    let t = compute_kinetic(basis);
    let v = compute_nuclear(basis, mol);
    let v_ecp = compute_ecp(basis);
    let mut h_core = Matrix2D::zeros(n, n);
    for i in 0..n * n {
        h_core.data[i] = t.data[i] + v.data[i] + v_ecp.data[i];
    }

    let fock_builder = UksFockBuilder::new_with_options(
        mol,
        basis,
        use_direct,
        scf_opts.density_fitting,
        scf_opts.use_gpu,
        xc.and_then(|f| f.range_separated_params()),
    )?;
    let e_nuc = basis.nuclear_repulsion_energy(mol);
    let e_disp = if xc == Some(XcFunctional::Wb97xd) {
        calchem_dft::wb97xd_chg_d2_energy_and_gradient(&mol.atoms)?.0
    } else {
        0.0
    };

    let rf = ReactionFieldSolver::build(mol, basis, opts)?;

    let (s_vals, s_vecs) = eigh_sym(&s, 100, 1e-14)?;
    let mut s_inv_sqrt = Matrix2D::zeros(n, n);
    for (i, &val) in s_vals.iter().enumerate() {
        if val <= 1e-10 {
            return Err(CalcError::LinearAlgebra(format!(
                "Overlap matrix is singular or too ill-conditioned for SCRF (eigenvalue {val:.3e})"
            )));
        }
        s_inv_sqrt.set(i, i, 1.0 / val.sqrt());
    }
    let x = matmul(&matmul(&s_vecs, &s_inv_sqrt), &transpose(&s_vecs));
    let xt = transpose(&x);

    let (_, _, mut pa) = diagonalize_fock_uhf(&h_core, &x, &xt, n_alpha)?;
    let (_, _, mut pb) = diagonalize_fock_uhf(&h_core, &x, &xt, n_beta)?;

    let mut diis_a = DiisState::from_scf_options(scf_opts);
    let mut diis_b = DiisState::from_scf_options(scf_opts);
    let mut e_prev = 0.0f64;
    let mut converged = false;
    let mut used_iter = 0usize;
    let mut last_delta_e = f64::INFINITY;
    let mut last_rms_d = f64::INFINITY;
    let mut final_evals = vec![0.0; n];
    let mut final_coeffs = Matrix2D::zeros(n, n);
    let mut final_fa = h_core.clone();
    let mut q_converged = Vec::new();
    let mut energy_history = Vec::new();
    let mut rms_history = Vec::new();
    let mut diis_err_history = Vec::new();

    let exx_frac = xc.map_or(1.0, |f| f.exact_exchange_fraction());
    let range_sep = xc.and_then(|f| f.range_separated_params());

    for iter in 1..=scf_opts.max_iter {
        used_iter = iter;
        let (ga, gb) = fock_builder.build_2e(basis, &pa, &pb, exx_frac, range_sep);
        let (e_xc, va_xc, vb_xc) = match (xc, &grid_opt) {
            (Some(func), Some(grid)) => evaluate_vxc_uks(basis, grid, func, &pa, &pb),
            _ => (0.0, Matrix2D::zeros(n, n), Matrix2D::zeros(n, n)),
        };

        let mut p_tot = Matrix2D::zeros(n, n);
        for idx in 0..n * n {
            p_tot.data[idx] = pa.data[idx] + pb.data[idx];
        }
        let (e_solv_elec, v_rf, q) = rf.reaction_field(&p_tot)?;
        q_converged = q;

        let mut fa = Matrix2D::zeros(n, n);
        let mut fb = Matrix2D::zeros(n, n);
        let mut e_1e_2e = 0.0;
        for i in 0..n * n {
            fa.data[i] = h_core.data[i] + ga.data[i] + va_xc.data[i] + v_rf.data[i];
            fb.data[i] = h_core.data[i] + gb.data[i] + vb_xc.data[i] + v_rf.data[i];
            e_1e_2e += pa.data[i] * (h_core.data[i] + 0.5 * ga.data[i])
                + pb.data[i] * (h_core.data[i] + 0.5 * gb.data[i]);
        }
        let e_tot = e_1e_2e + e_xc + e_nuc + e_disp + e_solv_elec;
        energy_history.push(e_tot);
        final_fa = fa.clone();

        let err_a = {
            let fps = matmul(&matmul(&fa, &pa), &s);
            let spf = matmul(&matmul(&s, &pa), &fa);
            let mut err = Matrix2D::zeros(n, n);
            for i in 0..n * n {
                err.data[i] = fps.data[i] - spf.data[i];
            }
            err
        };
        let err_b = {
            let fps = matmul(&matmul(&fb, &pb), &s);
            let spf = matmul(&matmul(&s, &pb), &fb);
            let mut err = Matrix2D::zeros(n, n);
            for i in 0..n * n {
                err.data[i] = fps.data[i] - spf.data[i];
            }
            err
        };
        let diis_err = frobenius_norm(&err_a).max(frobenius_norm(&err_b));
        diis_err_history.push(diis_err);

        let fa_extrap = if scf_opts.use_diis {
            let e = diis_a
                .try_push_and_extrapolate(&fa, &err_a)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            if iter >= 2 {
                e
            } else {
                fa.clone()
            }
        } else {
            fa.clone()
        };
        let fb_extrap = if scf_opts.use_diis {
            let e = diis_b
                .try_push_and_extrapolate(&fb, &err_b)
                .map_err(|e| CalcError::Internal(format!("DIIS disk spill I/O error: {e}")))?;
            if iter >= 2 {
                e
            } else {
                fb.clone()
            }
        } else {
            fb.clone()
        };

        let (eps_a, ca_new, mut pa_new) = diagonalize_fock_uhf(&fa_extrap, &x, &xt, n_alpha)?;
        let (_eps_b, _cb_new, mut pb_new) = diagonalize_fock_uhf(&fb_extrap, &x, &xt, n_beta)?;
        final_evals = eps_a;
        final_coeffs = ca_new;

        let damping = if scf_opts.damping > 0.0 && scf_opts.damping < 1.0 {
            scf_opts.damping
        } else if n >= 40 && (iter <= 10 || diis_err > 0.05) {
            0.3
        } else {
            0.0
        };
        if damping > 0.0 {
            for idx in 0..pa_new.data.len() {
                pa_new.data[idx] = (1.0 - damping) * pa_new.data[idx] + damping * pa.data[idx];
                pb_new.data[idx] = (1.0 - damping) * pb_new.data[idx] + damping * pb.data[idx];
            }
        }

        let delta_e = if iter == 1 {
            f64::INFINITY
        } else {
            (e_tot - e_prev).abs()
        };
        let rms_d = rms_diff(&pa_new, &pa).max(rms_diff(&pb_new, &pb));
        rms_history.push(rms_d);
        last_delta_e = delta_e;
        last_rms_d = rms_d;
        pa = pa_new;
        pb = pb_new;
        e_prev = e_tot;

        let grad_tol = (scf_opts.conv_density * 100.0).max(1e-6);
        if iter > 1 && delta_e < scf_opts.conv_energy && rms_d < scf_opts.conv_density && diis_err < grad_tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(CalcError::NotConverged {
            iterations: used_iter,
            delta_e: last_delta_e,
            rms_d: last_rms_d,
        });
    }

    let (cav_area_a2, tessera_coords) = rf.cavity_area_and_coords();
    let (cds_hartree, cds_kcal_mol, reported_area_a2) =
        compute_cds_energy(mol, opts, cav_area_a2);
    let solvated_energy = e_prev + cds_hartree;
    let delta_g_solv_hartree = solvated_energy - gas_energy;

    let rij_trunc = match &fock_builder {
        UksFockBuilder::DensityFitting(rij, _) => rij.metric_truncated_modes,
        _ => 0,
    };
    let diag = ScfDiagnostics {
        iterations: used_iter,
        energy_history,
        rms_density_history: rms_history,
        diis_error_history: diis_err_history,
        gpu_kernel_invocations: 0,
        gpu_fallback_invocations: 0,
        diis_disk_backed: diis_a.is_disk_backed() || diis_b.is_disk_backed(),
        rij_metric_truncated_modes: rij_trunc,
    };

    let mut mo_occupations = vec![0.0f64; n];
    for occ in mo_occupations.iter_mut().take(n_alpha) {
        *occ = 1.0;
    }

    let wf = ScfWavefunction {
        density_fitting: scf_opts.density_fitting,
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
        density_alpha: pa.data,
        density_beta: Some(pb.data),
        fock_alpha: final_fa.data,
        overlap_matrix: s.data,
        h_core_matrix: h_core.data,
        diagnostics: diag,
    };

    let total_q: f64 = q_converged.iter().sum();

    Ok(SolvationResult {
        solvated_energy,
        gas_energy,
        delta_g_solv: delta_g_solv_hartree,
        delta_g_solv_kcal_mol: delta_g_solv_hartree * HARTREE_TO_KCAL_MOL,
        delta_g_elec_kcal_mol: (delta_g_solv_hartree - cds_hartree) * HARTREE_TO_KCAL_MOL,
        delta_g_cds_kcal_mol: cds_kcal_mol,
        cavity_surface_area_angstrom2: reported_area_a2,
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
    let is_open_shell = mol.multiplicity > 1
        || mol.nelectron().map(|ne| ne % 2 != 0).unwrap_or(false)
        || m == "uhf"
        || m == "uks";

    if is_open_shell {
        return match m.as_str() {
            "uhf" | "rhf" | "hf" => run_scrf_unrestricted(mol, basis, scf_opts, opts, None),
            "uks" | "rks" | "dft" => {
                let func = xc.unwrap_or(XcFunctional::B3lyp);
                run_scrf_unrestricted(mol, basis, scf_opts, opts, Some(func))
            }
            "b3lyp" | "b3lyp5" | "pbe" | "blyp" | "svwn5" | "pbe0" | "camb3lyp" | "cam-b3lyp"
            | "wb97xd" | "wb97x-d" | "wb97x" => {
                let func = XcFunctional::from_name(method)?;
                if xc.is_some_and(|x| x != func) {
                    return Err(CalcError::UnsupportedMethod(
                        "Conflicting method and XC identities".into(),
                    ));
                }
                run_scrf_unrestricted(mol, basis, scf_opts, opts, Some(func))
            }
            other => Err(CalcError::UnsupportedMethod(format!(
                "Unsupported open-shell SCRF method: {other}"
            ))),
        };
    }

    match m.as_str() {
        "rhf" | "hf" => run_scrf_rhf(mol, basis, scf_opts, opts),
        "rks" | "dft" => {
            let func = xc.unwrap_or(XcFunctional::B3lyp);
            run_scrf_rks(mol, basis, scf_opts, opts, func)
        }
        "b3lyp" | "b3lyp5" | "pbe" | "blyp" | "svwn5" | "pbe0" | "camb3lyp" | "cam-b3lyp"
        | "wb97xd" | "wb97x-d" | "wb97x" => {
            let func = XcFunctional::from_name(method)?;
            if xc.is_some_and(|x| x != func) {
                return Err(CalcError::UnsupportedMethod(
                    "Conflicting method and XC identities".into(),
                ));
            }
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

fn diagonalize_fock_uhf(
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
            p.set(mu, nu, sum);
        }
    }
    Ok((eps, c, p))
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

fn compute_dipole_matrices(basis: &BasisSet, origin: [f64; 3]) -> [Matrix2D; 3] {
    let n = basis.n_ao;
    let mut out = [
        Matrix2D::zeros(n, n),
        Matrix2D::zeros(n, n),
        Matrix2D::zeros(n, n),
    ];
    for q in 0..3 {
        for sh_a in &basis.shells {
            let comps_a = sh_a.components();
            for sh_b in &basis.shells {
                let comps_b = sh_b.components();
                for (ia, &ca) in comps_a.iter().enumerate() {
                    let mu = sh_a.ao_offset + ia;
                    for (ib, &cb) in comps_b.iter().enumerate() {
                        let nu = sh_b.ao_offset + ib;
                        let mut sum_prim = 0.0;
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
                                let mut ca_plus = ca.map(|v| v as i32);
                                ca_plus[q] += 1;
                                let cb_i32 = cb.map(|v| v as i32);
                                let s_ab = overlap_primitive(ca.map(|v| v as i32), cb_i32, &pair);
                                let x_ab = overlap_primitive(ca_plus, cb_i32, &pair)
                                    + (sh_a.center[q] - origin[q]) * s_ab;
                                sum_prim += weight * x_ab;
                            }
                        }
                        out[q].set(mu, nu, sum_prim);
                    }
                }
            }
        }
    }
    out
}
