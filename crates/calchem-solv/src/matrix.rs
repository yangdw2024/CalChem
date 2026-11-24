//! Green's function electrostatic boundary interaction matrix $A$ and Cholesky solver.

use calchem_int::boys::boys_batch;
use calchem_model::CalcError;
use serde::{Deserialize, Serialize};

use crate::cavity::MolecularCavity;

/// Evaluates the York-Karplus spherical-Gaussian Coulomb kernel `erf(z*r)/r = (2/sqrt(pi))*z*F_0(z^2*r^2)`
/// and its radial derivative factor `-(1/r) d/dr [erf(z*r)/r] = (4/sqrt(pi))*z^3*F_1(z^2*r^2)`.
#[inline]
fn gaussian_coulomb_and_grad(zeta_ij: f64, r2: f64) -> (f64, f64) {
    let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    let t = zeta_ij * zeta_ij * r2;
    let mut f = [0.0f64; 2];
    boys_batch(1, t, &mut f);
    let s_ij = 2.0 * inv_sqrt_pi * zeta_ij * f[0];
    let g_ij = 4.0 * inv_sqrt_pi * zeta_ij * zeta_ij * zeta_ij * f[1];
    (s_ij, g_ij)
}

/// Green's function boundary interaction matrix $S \in \mathbb{R}^{N_s \times N_s}$
/// and inverse / effective response operator stored in `a_inv`.
///
/// Uses the York-Karplus spherical-Gaussian regularized Coulomb kernel:
/// - Diagonal self-interaction ($i = j$):
///   $$S_{ii} = 1.07 \sqrt{\frac{4\pi}{a_i}}, \qquad \zeta_i = \sqrt{\frac{\pi}{2}}\, S_{ii}$$
///   (Klamt-Schüürmann approximation for tessera area $a_i$).
/// - Off-diagonal elements ($i \ne j$):
///   $$S_{ij} = \frac{\operatorname{erf}(\zeta_{ij}\, \|\mathbf{s}_i - \mathbf{s}_j\|)}{\|\mathbf{s}_i - \mathbf{s}_j\|}, \qquad \zeta_{ij} = \frac{\zeta_i \zeta_j}{\sqrt{\zeta_i^2 + \zeta_j^2}}$$
///
/// For C-PCM / COSMO (`build`), `a_inv` stores $S^{-1}$.
/// For IEF-PCM / SMD (`build_iefpcm`), `a_inv` stores the symmetrized effective response matrix
/// $-Q_{\mathrm{eff}} = -\frac{1}{2}\bigl(K^{-1} R + (K^{-1} R)^T\bigr)$, where
/// $K = S - \frac{f(\varepsilon)}{2\pi} D A S$ and $R = -f(\varepsilon)\bigl(I - \frac{1}{2\pi} D A\bigr)$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryMatrix {
    pub dim: usize,
    pub a_inv: Vec<f64>,
}

impl BoundaryMatrix {
    /// Constructs the boundary matrix $A$ and precomputes its symmetric inverse $A^{-1}$ via Cholesky.
    pub fn build(cavity: &MolecularCavity) -> Result<Self, CalcError> {
        let n = cavity.n_tesserae();
        if n == 0 {
            return Err(CalcError::InvalidMolecule(
                "Cavity has zero surface tesserae".into(),
            ));
        }

        let tesserae = &cavity.tesserae;
        let mut a = vec![0.0f64; n * n];

        let klamt_prefactor = 1.07 * (4.0 * std::f64::consts::PI).sqrt();
        let sqrt_pi_over_2 = (0.5 * std::f64::consts::PI).sqrt();
        let mut zetas = vec![0.0f64; n];

        for i in 0..n {
            let a_i = tesserae[i].area_bohr2.max(1e-12);
            // Diagonal self-interaction of spherical Gaussian tessera: A_ii = sqrt(2/pi) * zeta_i
            let a_diag = klamt_prefactor / a_i.sqrt();
            a[i * n + i] = a_diag;
            zetas[i] = sqrt_pi_over_2 * a_diag;
        }

        for i in 0..n {
            let s_i = tesserae[i].position_bohr;
            let zi = zetas[i];
            for j in (i + 1)..n {
                let s_j = tesserae[j].position_bohr;
                let dx = s_i[0] - s_j[0];
                let dy = s_i[1] - s_j[1];
                let dz = s_i[2] - s_j[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let zj = zetas[j];
                let zeta_ij = (zi * zj) / (zi * zi + zj * zj).sqrt();
                let (s_ij, _) = gaussian_coulomb_and_grad(zeta_ij, r2);
                a[i * n + j] = s_ij;
                a[j * n + i] = s_ij;
            }
        }

        // Cholesky decomposition A = L * L^T with adaptive diagonal regularization if required
        let mut l = vec![0.0f64; n * n];
        let mut shift = 0.0;
        let mut factorized = false;

        for _attempt in 0..12 {
            let mut ok = true;
            l.fill(0.0);
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    if i == j {
                        let a_diag = a[i * n + i] + shift;
                        let val = a_diag - sum;
                        if val <= 1e-12 {
                            ok = false;
                            break;
                        }
                        l[i * n + j] = val.sqrt();
                    } else {
                        let lj_diag = l[j * n + j];
                        if lj_diag.abs() < 1e-15 {
                            ok = false;
                            break;
                        }
                        l[i * n + j] = (a[i * n + j] - sum) / lj_diag;
                    }
                }
                if !ok {
                    break;
                }
            }
            if ok {
                factorized = true;
                break;
            }
            shift = if shift == 0.0 { 1e-5 } else { shift * 10.0 };
        }

        if !factorized {
            return Err(CalcError::LinearAlgebra(
                "Boundary matrix A could not be factorized into positive definite form".into(),
            ));
        }

        // Invert L: compute L^{-1} where L is lower triangular
        let mut l_inv = vec![0.0f64; n * n];
        for i in 0..n {
            l_inv[i * n + i] = 1.0 / l[i * n + i];
            for j in 0..i {
                let mut sum = 0.0;
                for k in j..i {
                    sum += l[i * n + k] * l_inv[k * n + j];
                }
                l_inv[i * n + j] = -sum / l[i * n + i];
            }
        }

        // Compute A^{-1} = (L^{-1})^T * L^{-1}
        let mut a_inv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in j..n {
                    sum += l_inv[k * n + i] * l_inv[k * n + j];
                }
                a_inv[i * n + j] = sum;
                a_inv[j * n + i] = sum;
            }
        }

        Ok(Self { dim: n, a_inv })
    }

    /// Constructs the IEF-PCM effective response operator `Q_eff = 0.5 * (K^{-1} R + (K^{-1} R)^T)`
    /// where `K = S - (f_eps / 2pi) D A S` and `R = -f_eps (I - (1 / 2pi) D A)`,
    /// storing `-Q_eff` in `a_inv` so `solve_surface_charges(v, 1.0, q_net)` yields `q = Q_eff V`.
    pub fn build_iefpcm(cavity: &MolecularCavity, epsilon: f64) -> Result<Self, CalcError> {
        use faer::prelude::SpSolver;
        use faer::Mat;
        use std::f64::consts::PI;

        let n = cavity.n_tesserae();
        if n == 0 {
            return Err(CalcError::InvalidMolecule(
                "Cavity has zero surface tesserae".into(),
            ));
        }
        if (epsilon - 1.0).abs() < 1e-12 {
            return Ok(Self {
                dim: n,
                a_inv: vec![0.0; n * n],
            });
        }

        let tesserae = &cavity.tesserae;
        let klamt_prefactor = 1.07 * (4.0 * PI).sqrt();
        let sqrt_pi_over_2 = (0.5 * PI).sqrt();
        let f_eps = (epsilon - 1.0) / (epsilon + 1.0);
        let inv_2pi = 1.0 / (2.0 * PI);

        let mut s_mat = vec![0.0f64; n * n];
        let mut d_mat = vec![0.0f64; n * n];
        let mut areas = vec![0.0f64; n];
        let mut zetas = vec![0.0f64; n];

        for i in 0..n {
            let a_i = tesserae[i].area_bohr2.max(1e-12);
            let r_i = tesserae[i].radius_bohr.max(1e-6);
            areas[i] = a_i;
            let s_diag = klamt_prefactor / a_i.sqrt();
            s_mat[i * n + i] = s_diag;
            d_mat[i * n + i] = -s_diag / (2.0 * r_i);
            zetas[i] = sqrt_pi_over_2 * s_diag;
        }

        for i in 0..n {
            let s_i = tesserae[i].position_bohr;
            let zi = zetas[i];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let s_j = tesserae[j].position_bohr;
                let n_j = tesserae[j].normal;
                let dx = s_i[0] - s_j[0];
                let dy = s_i[1] - s_j[1];
                let dz = s_i[2] - s_j[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let zj = zetas[j];
                let zeta_ij = (zi * zj) / (zi * zi + zj * zj).sqrt();
                let (s_ij, g_ij) = gaussian_coulomb_and_grad(zeta_ij, r2);
                s_mat[i * n + j] = s_ij;
                d_mat[i * n + j] = (dx * n_j[0] + dy * n_j[1] + dz * n_j[2]) * g_ij;
            }
        }

        let da_mat = Mat::<f64>::from_fn(n, n, |i, j| d_mat[i * n + j] * areas[j]);
        let s_faer = Mat::<f64>::from_fn(n, n, |i, j| s_mat[i * n + j]);
        let das = &da_mat * &s_faer;

        let k_faer = Mat::<f64>::from_fn(n, n, |i, j| {
            s_faer.read(i, j) - f_eps * inv_2pi * das.read(i, j)
        });
        let r_faer = Mat::<f64>::from_fn(n, n, |i, j| {
            let eye = if i == j { 1.0 } else { 0.0 };
            -f_eps * (eye - inv_2pi * da_mat.read(i, j))
        });

        let lu = k_faer.partial_piv_lu();
        let q_mat = lu.solve(&r_faer);

        let mut a_inv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let q_ij = q_mat.read(i, j);
                let q_ji = q_mat.read(j, i);
                if !q_ij.is_finite() || !q_ji.is_finite() {
                    return Err(CalcError::LinearAlgebra(
                        "IEF-PCM boundary linear solve produced non-finite entries".into(),
                    ));
                }
                let sym = -0.5 * (q_ij + q_ji);
                a_inv[i * n + j] = sym;
                a_inv[j * n + i] = sym;
            }
        }

        Ok(Self { dim: n, a_inv })
    }

    /// Builds the boundary matrix appropriate for the requested [`crate::solvent::SolventModel`].
    pub fn build_for_model(
        cavity: &MolecularCavity,
        model: crate::solvent::SolventModel,
        epsilon: f64,
    ) -> Result<Self, CalcError> {
        match model {
            crate::solvent::SolventModel::IefPcm | crate::solvent::SolventModel::Smd => {
                Self::build_iefpcm(cavity, epsilon)
            }
            _ => Self::build(cavity),
        }
    }

    /// Solves for apparent surface charges $\mathbf{q} = -f(\varepsilon) A^{-1} \mathbf{V}^{\text{solute}}$.
    ///
    /// If `renormalize_charge` is true and net solute charge `q_solute` is specified,
    /// applies Klamt charge conservation scaling $\sum_k q_k = -f(\varepsilon) Q_{\text{solute}}$.
    pub fn solve_surface_charges(
        &self,
        v_solute: &[f64],
        dielectric_factor: f64,
        solute_net_charge: Option<f64>,
    ) -> Result<Vec<f64>, CalcError> {
        self.solve_surface_charges_with_gauss(
            v_solute,
            dielectric_factor,
            dielectric_factor,
            solute_net_charge,
        )
    }

    /// Solves for apparent surface charges with separate matrix scaling (`matrix_scale`, e.g. `1.0`
    /// when `a_inv` already embeds IEF-PCM dielectric screening) and Gauss-law charge renormalization
    /// factor (`gauss_scale`, e.g. `(eps - 1) / eps`).
    pub fn solve_surface_charges_with_gauss(
        &self,
        v_solute: &[f64],
        matrix_scale: f64,
        gauss_scale: f64,
        solute_net_charge: Option<f64>,
    ) -> Result<Vec<f64>, CalcError> {
        let n = self.dim;
        if v_solute.len() != n {
            return Err(CalcError::LinearAlgebra(format!(
                "Potential vector length {} does not match boundary dimension {n}",
                v_solute.len()
            )));
        }

        // q_raw = -matrix_scale * A^{-1} * V_solute
        let mut q = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            let row_offset = i * n;
            for j in 0..n {
                sum += self.a_inv[row_offset + j] * v_solute[j];
            }
            q[i] = -matrix_scale * sum;
        }

        // Charge conservation / Gauss law renormalization
        if let Some(q_net) = solute_net_charge {
            let target_charge = -gauss_scale * q_net;
            let current_sum: f64 = q.iter().sum();

            if q_net.abs() > 1e-4 {
                if current_sum.abs() > 1e-8 {
                    let scale = target_charge / current_sum;
                    for val in &mut q {
                        *val *= scale;
                    }
                }
            } else {
                // Neutral molecule: ensure sum_k q_k == 0 strictly
                let diff_per_tessera = (target_charge - current_sum) / (n as f64);
                for val in &mut q {
                    *val += diff_per_tessera;
                }
            }
        }

        Ok(q)
    }
}
