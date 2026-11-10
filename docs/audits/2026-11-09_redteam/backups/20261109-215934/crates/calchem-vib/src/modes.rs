//! Normal mode analysis, Eckart translation-rotation projection,
//! vibrational frequencies ($\text{cm}^{-1}$), and infrared (IR) intensities ($\text{km/mol}$).

use calchem_model::{CalcError, Molecule};
use calchem_tensor::{Matrix2D, eigh_sym, matmul, transpose};
use serde::{Deserialize, Serialize};

use crate::masses::{
    HESSIAN_EVAL_TO_INV_CM, IR_INTENSITY_PREFACTOR_KM_MOL, atomic_mass_amu,
};

const HARTREE_BOHR2_TO_MDYN_ANG: f64 = 15.568931;

/// Single vibrational normal mode with frequency, reduced mass, force constant, and IR intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalMode {
    /// 1-based index among vibrational modes.
    pub mode_index: usize,
    /// Harmonic vibrational frequency $\tilde{\nu}_k$ in $\text{cm}^{-1}$ (positive for real, negative for imaginary).
    pub frequency_cm: f64,
    /// `true` if $\lambda_k < 0$ (imaginary frequency, characteristic of a transition state).
    pub is_imaginary: bool,
    /// Reduced mass $\mu_k$ in atomic mass units (amu).
    pub reduced_mass_amu: f64,
    /// Harmonic force constant $k$ in $\text{mDyn} / \text{Å}$.
    pub force_constant_m_dyn_a: f64,
    /// Infrared absorption intensity $I_k$ in $\text{km} / \text{mol}$.
    pub ir_intensity_km_mol: f64,
    /// Normalized Cartesian displacement vector ($3N$ elements: `[dx_0, dy_0, dz_0, dx_1, ...]`).
    pub displacements: Vec<f64>,
}

/// Output of harmonic vibrational frequency and normal mode analysis.
#[derive(Debug, Clone)]
pub struct VibrationalAnalysis {
    pub n_atoms: usize,
    pub is_linear: bool,
    pub n_trans_rot: usize,
    pub n_vib: usize,
    /// Principal moments of inertia $I_A \le I_B \le I_C$ in $\text{amu}\cdot\text{Bohr}^2$.
    pub moments_of_inertia_amu_bohr2: [f64; 3],
    /// Rotational constants $A, B, C$ in GHz.
    pub rotational_constants_ghz: [f64; 3],
    /// Symmetrized Cartesian Hessian matrix ($3N \times 3N$, Hartree / $\text{Bohr}^2$).
    pub cartesian_hessian: Matrix2D,
    /// Atomic Polar Tensor ($3 \times 3N$, atomic units).
    pub apt_matrix: Matrix2D,
    /// Vibrational normal modes ordered by ascending frequency.
    pub modes: Vec<NormalMode>,
    /// Projected translation and rotation residual eigenvalues (Hartree / ($\text{amu}\cdot\text{Bohr}^2$)).
    pub trans_rot_residuals: Vec<f64>,
}

/// Computes center of mass and coordinates shifted to center of mass.
fn center_of_mass(mol: &Molecule, masses_amu: &[f64]) -> ([f64; 3], Vec<[f64; 3]>) {
    let mut total_m = 0.0;
    let mut com = [0.0; 3];
    for (atom, &m) in mol.atoms.iter().zip(masses_amu) {
        total_m += m;
        for q in 0..3 {
            com[q] += m * atom.position_bohr[q];
        }
    }
    if total_m > 0.0 {
        com[0] /= total_m;
        com[1] /= total_m;
        com[2] /= total_m;
    }
    let shifted: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .map(|a| {
            [
                a.position_bohr[0] - com[0],
                a.position_bohr[1] - com[1],
                a.position_bohr[2] - com[2],
            ]
        })
        .collect();
    (com, shifted)
}

/// Computes the $3 \times 3$ inertia tensor in $\text{amu}\cdot\text{Bohr}^2$ and principal moments.
fn compute_inertia_tensor(shifted_coords: &[[f64; 3]], masses_amu: &[f64]) -> ([f64; 3], [f64; 3], bool) {
    let mut i_mat = Matrix2D::zeros(3, 3);
    for (pos, &m) in shifted_coords.iter().zip(masses_amu) {
        let x = pos[0];
        let y = pos[1];
        let z = pos[2];
        let r2 = x * x + y * y + z * z;
        i_mat.add_assign_at(0, 0, m * (r2 - x * x));
        i_mat.add_assign_at(1, 1, m * (r2 - y * y));
        i_mat.add_assign_at(2, 2, m * (r2 - z * z));
        i_mat.add_assign_at(0, 1, -m * x * y);
        i_mat.add_assign_at(1, 0, -m * x * y);
        i_mat.add_assign_at(0, 2, -m * x * z);
        i_mat.add_assign_at(2, 0, -m * x * z);
        i_mat.add_assign_at(1, 2, -m * y * z);
        i_mat.add_assign_at(2, 1, -m * y * z);
    }

    let (mut evals, _) = eigh_sym(&i_mat, 50, 1e-14).unwrap_or(([0.0, 0.0, 0.0].to_vec(), Matrix2D::identity(3)));
    evals.sort_by(|a, b| a.total_cmp(b));
    let moments = [evals[0].max(0.0), evals[1].max(0.0), evals[2].max(0.0)];

    // Rotational constants B = h / (8 * pi^2 * c * I)
    // 1 amu * Bohr^2 = 1.66053906660e-27 kg * (5.29177210903e-11 m)^2 = 4.6509653e-48 kg*m^2
    // h / (8 * pi^2 * 4.6509653e-48 kg*m^2) in GHz = 6.62607015e-34 / (8 * pi^2 * 4.6509653e-48) * 1e-9 = 505.379006 GHz * amu * Bohr^2
    const INERTIA_TO_GHZ: f64 = 505.379006;
    let b_ghz = [
        if moments[0] > 1e-6 { INERTIA_TO_GHZ / moments[0] } else { 0.0 },
        if moments[1] > 1e-6 { INERTIA_TO_GHZ / moments[1] } else { 0.0 },
        if moments[2] > 1e-6 { INERTIA_TO_GHZ / moments[2] } else { 0.0 },
    ];

    let is_linear = shifted_coords.len() >= 2 && (moments[0] < 1e-4 * moments[2] || moments[0] < 1e-4);
    (moments, b_ghz, is_linear)
}

/// Orthonormalizes vectors using two-pass Modified Gram-Schmidt; drops vectors with norm $< 10^{-6}$.
fn gram_schmidt_ortho(_dim: usize, raw_vecs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for mut v in raw_vecs {
        for _pass in 0..2 {
            for q in &basis {
                let dot: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q.iter()) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-6 {
            let inv_n = 1.0 / norm;
            for x in &mut v {
                *x *= inv_n;
            }
            basis.push(v);
        }
    }
    basis
}

/// Performs normal mode analysis on Cartesian Hessian and Atomic Polar Tensor.
pub fn analyze_vibrations(
    mol: &Molecule,
    cart_hessian: &Matrix2D,
    apt: &Matrix2D,
) -> Result<VibrationalAnalysis, CalcError> {
    let n_atoms = mol.atoms.len();
    let dim = 3 * n_atoms;
    if cart_hessian.rows != dim || cart_hessian.cols != dim {
        return Err(CalcError::InvalidMolecule(format!(
            "Cartesian Hessian dimension {}x{} mismatch for {n_atoms} atoms (expected {dim}x{dim})",
            cart_hessian.rows, cart_hessian.cols
        )));
    }

    let masses_amu: Vec<f64> = mol.atoms.iter().map(|a| atomic_mass_amu(a.z)).collect();
    let mut mass_coord = vec![0.0f64; dim];
    for (i, &m) in masses_amu.iter().enumerate() {
        mass_coord[3 * i] = m;
        mass_coord[3 * i + 1] = m;
        mass_coord[3 * i + 2] = m;
    }

    let (_, shifted) = center_of_mass(mol, &masses_amu);
    let (moments, b_ghz, is_linear) = compute_inertia_tensor(&shifted, &masses_amu);

    // Build translation and rotation vectors in mass-weighted coordinate space:
    // T_q: translation along Cartesian axis q
    // R_q: rotation about Cartesian axis q
    let mut raw_tr: Vec<Vec<f64>> = Vec::with_capacity(6);
    for q in 0..3 {
        let mut t = vec![0.0f64; dim];
        for a in 0..n_atoms {
            t[3 * a + q] = masses_amu[a].sqrt();
        }
        raw_tr.push(t);
    }

    if n_atoms >= 2 {
        let mut rx = vec![0.0f64; dim];
        let mut ry = vec![0.0f64; dim];
        let mut rz = vec![0.0f64; dim];
        for a in 0..n_atoms {
            let sqrt_m = masses_amu[a].sqrt();
            let x = shifted[a][0];
            let y = shifted[a][1];
            let z = shifted[a][2];
            // R_x = (0, -z, y) * sqrt(m)
            rx[3 * a + 1] = -z * sqrt_m;
            rx[3 * a + 2] = y * sqrt_m;
            // R_y = (z, 0, -x) * sqrt(m)
            ry[3 * a] = z * sqrt_m;
            ry[3 * a + 2] = -x * sqrt_m;
            // R_z = (-y, x, 0) * sqrt(m)
            rz[3 * a] = -y * sqrt_m;
            rz[3 * a + 1] = x * sqrt_m;
        }
        raw_tr.push(rx);
        raw_tr.push(ry);
        raw_tr.push(rz);
    }

    // Orthonormalize translation-rotation basis D (dim x n_tr)
    let tr_basis = gram_schmidt_ortho(dim, raw_tr);
    let n_tr = tr_basis.len();
    let n_vib = dim.saturating_sub(n_tr);

    // Mass-weight the Hessian: H_mw_{ij} = H_{ij} / sqrt(m_i * m_j)
    let mut h_mw = Matrix2D::zeros(dim, dim);
    for i in 0..dim {
        let inv_sqrt_mi = 1.0 / mass_coord[i].sqrt();
        for j in 0..dim {
            let inv_sqrt_mj = 1.0 / mass_coord[j].sqrt();
            h_mw.set(i, j, cart_hessian.get(i, j) * inv_sqrt_mi * inv_sqrt_mj);
        }
    }

    // Construct projection matrix P = I - D D^T and project H_mw: H_proj = P H_mw P
    let mut h_proj = h_mw.clone();
    if n_tr > 0 {
        let mut d_mat = Matrix2D::zeros(dim, n_tr);
        for (col, v) in tr_basis.iter().enumerate() {
            for row in 0..dim {
                d_mat.set(row, col, v[row]);
            }
        }
        let dt_h = matmul(&transpose(&d_mat), &h_mw);
        let h_d = matmul(&h_mw, &d_mat);
        let dt_h_d = matmul(&dt_h, &d_mat);
        let d_dt_h = matmul(&d_mat, &dt_h);
        let h_d_dt = matmul(&h_d, &transpose(&d_mat));
        let d_dt_h_d_dt = matmul(&matmul(&d_mat, &dt_h_d), &transpose(&d_mat));

        for i in 0..dim {
            for j in 0..dim {
                let v = h_mw.get(i, j) - d_dt_h.get(i, j) - h_d_dt.get(i, j) + d_dt_h_d_dt.get(i, j);
                h_proj.set(i, j, v);
            }
        }
    }

    // Symmetrize H_proj to prevent floating-point asymmetry
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (h_proj.get(i, j) + h_proj.get(j, i));
            h_proj.set(i, j, avg);
            h_proj.set(j, i, avg);
        }
    }

    let (evals, evecs) = eigh_sym(&h_proj, 150, 1e-14)?;

    // Package eigenvalues and mode eigenvectors (columns of evecs)
    let mut indexed_modes: Vec<(f64, Vec<f64>)> = (0..dim)
        .map(|col| {
            let lam = evals[col];
            let mut v = vec![0.0f64; dim];
            for row in 0..dim {
                v[row] = evecs.get(row, col);
            }
            (lam, v)
        })
        .collect();

    // Sort modes: the n_tr eigenvalues closest to zero are translation/rotation
    indexed_modes.sort_by(|a, b| a.0.abs().total_cmp(&b.0.abs()));
    let trans_rot_modes: Vec<f64> = indexed_modes[..n_tr].iter().map(|m| m.0).collect();
    let mut vib_modes = indexed_modes[n_tr..].to_vec();

    // Sort vibrational modes by ascending signed eigenvalue (imaginary/negative first, then real)
    vib_modes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut modes: Vec<NormalMode> = Vec::with_capacity(n_vib);
    for (idx, (lam, evec_mw)) in vib_modes.into_iter().enumerate() {
        let (freq_cm, is_imag) = if lam >= 0.0 {
            (HESSIAN_EVAL_TO_INV_CM * lam.sqrt(), false)
        } else {
            (-HESSIAN_EVAL_TO_INV_CM * (-lam).sqrt(), true)
        };

        // Cartesian displacement U_i = L_i / sqrt(m_i)
        let mut u_cart = vec![0.0f64; dim];
        for i in 0..dim {
            u_cart[i] = evec_mw[i] / mass_coord[i].sqrt();
        }

        // Reduced mass mu = 1 / sum_i (L_i / sqrt(m_i))^2 = 1 / sum_i U_i^2 (amu)
        let sum_u2: f64 = u_cart.iter().map(|x| x * x).sum();
        let red_mass = if sum_u2 > 1e-15 { 1.0 / sum_u2 } else { 1.0 };

        // Normalize Cartesian displacement: Q_i = U_i / sqrt(sum_j U_j^2)
        let inv_norm = if sum_u2 > 1e-15 { 1.0 / sum_u2.sqrt() } else { 1.0 };
        let q_disp: Vec<f64> = u_cart.iter().map(|&x| x * inv_norm).collect();

        // Harmonic force constant k = lambda * red_mass in mDyn / A
        let k_mdyn_a = lam.abs() * red_mass * HARTREE_BOHR2_TO_MDYN_ANG;

        // IR intensity: I = 42.2561 * |dmu / dQ|^2 (km/mol), with dmu/dQ_p = sum_i P_{p, i} * U_i
        let mut dmu_dq = [0.0f64; 3];
        for p in 0..3 {
            let mut s = 0.0;
            for i in 0..dim {
                s += apt.get(p, i) * u_cart[i];
            }
            dmu_dq[p] = s;
        }
        let dmu_sq = dmu_dq[0] * dmu_dq[0] + dmu_dq[1] * dmu_dq[1] + dmu_dq[2] * dmu_dq[2];
        let ir_intensity = IR_INTENSITY_PREFACTOR_KM_MOL * dmu_sq;

        modes.push(NormalMode {
            mode_index: idx + 1,
            frequency_cm: freq_cm,
            is_imaginary: is_imag,
            reduced_mass_amu: red_mass,
            force_constant_m_dyn_a: k_mdyn_a,
            ir_intensity_km_mol: ir_intensity,
            displacements: q_disp,
        });
    }

    Ok(VibrationalAnalysis {
        n_atoms,
        is_linear,
        n_trans_rot: n_tr,
        n_vib,
        moments_of_inertia_amu_bohr2: moments,
        rotational_constants_ghz: b_ghz,
        cartesian_hessian: cart_hessian.clone(),
        apt_matrix: apt.clone(),
        modes,
        trans_rot_residuals: trans_rot_modes,
    })
}
