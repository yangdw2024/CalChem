//! Adiabatic Kohn-Sham Exchange-Correlation Kernel ($f_{\text{xc}}$) for TD-DFT / TDA Linear Response.
//!
//! Precomputes spin-adapted Singlet ($\Delta P^\alpha = +\Delta P^\beta = M$) and Triplet
//! ($\Delta P^\alpha = -\Delta P^\beta = M$) local second-derivative invariants on a [`MolecularGrid`]
//! around a closed-shell ground state $P_0^t = 2 P_0^\alpha$, enabling ultra-fast Rayon-parallel
//! matrix-vector contractions $\delta V^{\text{xc}}[M]$ inside Davidson iterations with zero
//! transcendental function evaluations per trial vector.

use calchem_basis::BasisSet;
use calchem_tensor::Matrix2D;
use rayon::prelude::*;

use crate::ao_grid::{AoGradVal, PrecomputedAoShell, eval_aos_grad};
use crate::grid::MolecularGrid;
use crate::xc::{XcFunctional, eval_xc_spin};

const FXC_CHUNK_SIZE: usize = 256;
/// Ground-state total density screening cutoff (avoids divergent $\rho^{-2/3}$ / $\rho^{-8/3}$ vacuum tails).
const FXC_RHO_CUTOFF: f64 = 1e-12;
const GAMMA_ANISO_CUTOFF: f64 = 1e-28;

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Precomputed local XC kernel invariants for either Singlet or Triplet channel at a grid point.
#[derive(Debug, Clone, Copy, Default)]
struct SpinChannelKernel {
    k_rr: f64,
    k_rg: f64,
    k_gg: f64,
    u_iso: f64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveFxcPoint {
    coords: [f64; 3],
    weight: f64,
    g_s: [f64; 3],
    singlet: SpinChannelKernel,
    triplet: SpinChannelKernel,
}

/// Precomputed adiabatic $f_{\text{xc}}$ operator on a molecular grid.
pub struct FxcKernel {
    func: XcFunctional,
    n_ao: usize,
    shells: Vec<PrecomputedAoShell>,
    points: Vec<ActiveFxcPoint>,
}

impl FxcKernel {
    /// Precomputes both Singlet and Triplet adiabatic XC kernels around closed-shell `density_total` ($P^t = 2 P^\alpha$).
    pub fn new(
        basis: &BasisSet,
        grid: &MolecularGrid,
        func: XcFunctional,
        density_total: &Matrix2D,
    ) -> Self {
        let n = basis.n_ao;
        let shells = PrecomputedAoShell::from_basis(basis);
        let p_slice = &density_total.data;
        let is_gga = func.is_gga();

        let points: Vec<ActiveFxcPoint> = grid
            .points
            .par_chunks(FXC_CHUNK_SIZE)
            .flat_map_iter(|chunk| {
                let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
                let mut out_pts: Vec<ActiveFxcPoint> = Vec::with_capacity(chunk.len());

                for gp in chunk {
                    let w = gp.weight;
                    if w.abs() < 1e-22 {
                        continue;
                    }
                    eval_aos_grad(&shells, gp.coords, &mut active);
                    if active.is_empty() {
                        continue;
                    }

                    let mut rho_t = 0.0;
                    let mut grad_rho_t = [0.0f64; 3];
                    for a_i in &active {
                        let row = &p_slice[a_i.mu * n..(a_i.mu + 1) * n];
                        let mut s = 0.0;
                        for a_j in &active {
                            s += row[a_j.mu] * a_j.val;
                        }
                        rho_t += a_i.val * s;
                        grad_rho_t[0] += 2.0 * a_i.grad[0] * s;
                        grad_rho_t[1] += 2.0 * a_i.grad[1] * s;
                        grad_rho_t[2] += 2.0 * a_i.grad[2] * s;
                    }

                    if rho_t < FXC_RHO_CUTOFF {
                        continue;
                    }

                    let rho_s = 0.5 * rho_t;
                    let g_s = [
                        0.5 * grad_rho_t[0],
                        0.5 * grad_rho_t[1],
                        0.5 * grad_rho_t[2],
                    ];
                    let gamma_s = dot3(g_s, g_s);
                    let out0 = eval_xc_spin(func, rho_s, rho_s, gamma_s, gamma_s, gamma_s);

                    // Relative density step guarantees rho_s +/- 2*hr > 0 everywhere.
                    let hr = 1e-5 * rho_s;

                    // --- Singlet kernel (spin-symmetric perturbation) ---
                    let eval_s = |r: f64, g: f64| -> (f64, f64) {
                        let o = eval_xc_spin(func, r, r, g, g, g);
                        (o.v_rho_a, 2.0 * o.v_gamma_aa + o.v_gamma_ab)
                    };
                    let (vr_m2, ug_m2) = eval_s(rho_s - 2.0 * hr, gamma_s);
                    let (vr_m1, ug_m1) = eval_s(rho_s - hr, gamma_s);
                    let (vr_p1, ug_p1) = eval_s(rho_s + hr, gamma_s);
                    let (vr_p2, ug_p2) = eval_s(rho_s + 2.0 * hr, gamma_s);
                    let inv_12hr = 1.0 / (12.0 * hr);

                    let k_rr_s = (vr_m2 - 8.0 * vr_m1 + 8.0 * vr_p1 - vr_p2) * inv_12hr;
                    let (k_rg_s, k_gg_s, u_iso_s) = if is_gga {
                        let k_rg = (ug_m2 - 8.0 * ug_m1 + 8.0 * ug_p1 - ug_p2) * inv_12hr;
                        let u_iso = 2.0 * out0.v_gamma_aa + out0.v_gamma_ab;
                        let k_gg = if gamma_s > GAMMA_ANISO_CUTOFF {
                            let hg = 1e-5 * gamma_s;
                            let (_, u_gm2) = eval_s(rho_s, gamma_s - 2.0 * hg);
                            let (_, u_gm1) = eval_s(rho_s, gamma_s - hg);
                            let (_, u_gp1) = eval_s(rho_s, gamma_s + hg);
                            let (_, u_gp2) = eval_s(rho_s, gamma_s + 2.0 * hg);
                            2.0 * (u_gm2 - 8.0 * u_gm1 + 8.0 * u_gp1 - u_gp2) / (12.0 * hg)
                        } else {
                            0.0
                        };
                        (k_rg, k_gg, u_iso)
                    } else {
                        (0.0, 0.0, 0.0)
                    };

                    // --- Triplet kernel (spin-antisymmetric perturbation; odd symmetry halves evaluations!) ---
                    let eval_t = |tau: f64, zeta: f64| -> (f64, f64) {
                        let o = eval_xc_spin(
                            func,
                            rho_s + tau,
                            rho_s - tau,
                            gamma_s + zeta,
                            gamma_s,
                            gamma_s - zeta,
                        );
                        (0.5 * (o.v_rho_a - o.v_rho_b), o.v_gamma_aa - o.v_gamma_bb)
                    };
                    let (vt_1, ut_1) = eval_t(hr, 0.0);
                    let (vt_2, ut_2) = eval_t(2.0 * hr, 0.0);
                    let inv_6hr = 1.0 / (6.0 * hr);

                    let k_rr_t = (8.0 * vt_1 - vt_2) * inv_6hr;
                    let (k_rg_t, k_gg_t, u_iso_t) = if is_gga {
                        let k_rg = (8.0 * ut_1 - ut_2) * inv_6hr;
                        let u_iso = 2.0 * out0.v_gamma_aa - out0.v_gamma_ab;
                        let k_gg = if gamma_s > GAMMA_ANISO_CUTOFF {
                            let hg = 1e-5 * gamma_s;
                            let (_, u_z1) = eval_t(0.0, hg);
                            let (_, u_z2) = eval_t(0.0, 2.0 * hg);
                            2.0 * (8.0 * u_z1 - u_z2) / (6.0 * hg)
                        } else {
                            0.0
                        };
                        (k_rg, k_gg, u_iso)
                    } else {
                        (0.0, 0.0, 0.0)
                    };

                    out_pts.push(ActiveFxcPoint {
                        coords: gp.coords,
                        weight: w,
                        g_s,
                        singlet: SpinChannelKernel {
                            k_rr: k_rr_s,
                            k_rg: k_rg_s,
                            k_gg: k_gg_s,
                            u_iso: u_iso_s,
                        },
                        triplet: SpinChannelKernel {
                            k_rr: k_rr_t,
                            k_rg: k_rg_t,
                            k_gg: k_gg_t,
                            u_iso: u_iso_t,
                        },
                    });
                }

                out_pts
            })
            .collect();

        Self {
            func,
            n_ao: n,
            shells,
            points,
        }
    }

    #[inline]
    pub fn functional(&self) -> XcFunctional {
        self.func
    }

    /// Contracts the precomputed Singlet (`triplet = false`) or Triplet (`triplet = true`)
    /// adiabatic XC kernel with a symmetric spatial transition density matrix $M$ (`n_ao x n_ao`),
    /// returning $\delta V^{\text{xc},\alpha}[P_0^\alpha + M, P_0^\beta \pm M]$.
    pub fn contract(&self, m_sym: &Matrix2D, triplet: bool) -> Matrix2D {
        let n = self.n_ao;
        if n == 0 || self.points.is_empty() {
            return Matrix2D::zeros(n, n);
        }
        let is_gga = self.func.is_gga();
        let m_slice = &m_sym.data;

        let v_data = self
            .points
            .par_chunks(FXC_CHUNK_SIZE)
            .map(|chunk| {
                let mut v_loc = vec![0.0f64; n * n];
                let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
                let mut z_vec: Vec<f64> = Vec::with_capacity(n);

                for pt in chunk {
                    eval_aos_grad(&self.shells, pt.coords, &mut active);
                    if active.is_empty() {
                        continue;
                    }

                    let mut eta = 0.0;
                    let mut xi = [0.0f64; 3];
                    for a_i in &active {
                        let row = &m_slice[a_i.mu * n..(a_i.mu + 1) * n];
                        let mut s = 0.0;
                        for a_j in &active {
                            s += row[a_j.mu] * a_j.val;
                        }
                        eta += a_i.val * s;
                        if is_gga {
                            xi[0] += 2.0 * a_i.grad[0] * s;
                            xi[1] += 2.0 * a_i.grad[1] * s;
                            xi[2] += 2.0 * a_i.grad[2] * s;
                        }
                    }

                    let k = if triplet { &pt.triplet } else { &pt.singlet };
                    let w = pt.weight;

                    z_vec.clear();
                    if is_gga {
                        let sigma = dot3(pt.g_s, xi);
                        let dv_rho = k.k_rr * eta + k.k_rg * sigma;
                        let g_coeff = k.k_rg * eta + k.k_gg * sigma;
                        let du = [
                            k.u_iso * xi[0] + g_coeff * pt.g_s[0],
                            k.u_iso * xi[1] + g_coeff * pt.g_s[1],
                            k.u_iso * xi[2] + g_coeff * pt.g_s[2],
                        ];
                        let half_wdv = 0.5 * w * dv_rho;
                        for a_i in &active {
                            z_vec.push(half_wdv * a_i.val + w * dot3(du, a_i.grad));
                        }
                    } else {
                        let half_wdv = 0.5 * w * (k.k_rr * eta);
                        for a_i in &active {
                            z_vec.push(half_wdv * a_i.val);
                        }
                    }

                    let m_len = active.len();
                    for i in 0..m_len {
                        let mu = active[i].mu;
                        let phi_i = active[i].val;
                        let z_i = z_vec[i];
                        let row_out = &mut v_loc[mu * n..(mu + 1) * n];
                        for j in 0..m_len {
                            let nu = active[j].mu;
                            row_out[nu] += z_i * active[j].val + phi_i * z_vec[j];
                        }
                    }
                }

                v_loc
            })
            .reduce(
                || vec![0.0f64; n * n],
                |mut a, b| {
                    for (dst, src) in a.iter_mut().zip(b.iter()) {
                        *dst += *src;
                    }
                    a
                },
            );

        let mut out = Matrix2D::zeros(n, n);
        out.data = v_data;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{GridLevel, MolecularGrid};
    use crate::vxc::evaluate_vxc_uks;
    use calchem_model::{Atom, Molecule};

    #[test]
    fn test_fxc_singlet_and_triplet_match_uks_vxc_finite_differences_all_functionals() {
        let mol = Molecule::new(
            vec![
                Atom::from_angstrom("O", [0.0, 0.0, 0.1173]).unwrap(),
                Atom::from_angstrom("H", [0.0, 0.7572, -0.4692]).unwrap(),
                Atom::from_angstrom("H", [0.0, -0.7572, -0.4692]).unwrap(),
            ],
            0,
            1,
        );
        let bs = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = bs.n_ao;

        // Strictly positive-definite closed-shell total density P_0 = 2 P_alpha > 0
        let mut p0_tot = Matrix2D::zeros(n, n);
        for i in 0..n {
            p0_tot.set(i, i, 0.25 + 0.01 * (i as f64));
            for j in (i + 1)..n {
                let v = 0.015 * (((i * 7 + j * 3) as f64).sin());
                p0_tot.set(i, j, v);
                p0_tot.set(j, i, v);
            }
        }
        // Symmetric transition density perturbation M
        let mut m_sym = Matrix2D::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let v = 0.02 * (((i * 5 + j * 11 + 1) as f64).cos());
                m_sym.set(i, j, v);
                m_sym.set(j, i, v);
            }
        }

        let eps = 1e-5;
        let build_perturbed = |c_a: f64, c_b: f64| -> (Matrix2D, Matrix2D) {
            let mut pa = Matrix2D::zeros(n, n);
            let mut pb = Matrix2D::zeros(n, n);
            for k in 0..(n * n) {
                pa.data[k] = 0.5 * p0_tot.data[k] + c_a * m_sym.data[k];
                pb.data[k] = 0.5 * p0_tot.data[k] + c_b * m_sym.data[k];
            }
            (pa, pb)
        };

        for func in [
            XcFunctional::Svwn5,
            XcFunctional::Blyp,
            XcFunctional::Pbe,
            XcFunctional::B3lyp,
            XcFunctional::B3lyp5,
            XcFunctional::Pbe0,
        ] {
            let kernel = FxcKernel::new(&bs, &grid, func, &p0_tot);

            for (triplet, sign_b) in [(false, 1.0), (true, -1.0)] {
                let dv_ana = kernel.contract(&m_sym, triplet);
                assert!(
                    dv_ana.sym_frobenius_error() < 1e-13,
                    "Fxc matrix must be symmetric"
                );

                // 4-point central FD of evaluate_vxc_uks
                let (pa_m2, pb_m2) = build_perturbed(-2.0 * eps, -2.0 * sign_b * eps);
                let (pa_m1, pb_m1) = build_perturbed(-eps, -sign_b * eps);
                let (pa_p1, pb_p1) = build_perturbed(eps, sign_b * eps);
                let (pa_p2, pb_p2) = build_perturbed(2.0 * eps, 2.0 * sign_b * eps);

                let (_, va_m2, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_m2, &pb_m2);
                let (_, va_m1, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_m1, &pb_m1);
                let (_, va_p1, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_p1, &pb_p1);
                let (_, va_p2, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_p2, &pb_p2);

                let mut max_err = 0.0f64;
                for k in 0..(n * n) {
                    let fd = (va_m2.data[k] - 8.0 * va_m1.data[k] + 8.0 * va_p1.data[k] - va_p2.data[k])
                        / (12.0 * eps);
                    max_err = max_err.max((dv_ana.data[k] - fd).abs());
                }
                assert!(
                    max_err < 1e-8,
                    "Fxc {func:?} (triplet={triplet}) max error vs UKS Vxc FD = {max_err:.3e}"
                );
            }
        }
    }
}
