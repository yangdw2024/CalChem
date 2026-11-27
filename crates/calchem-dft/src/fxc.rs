//! Adiabatic Kohn-Sham Exchange-Correlation Kernel ($f_{\text{xc}}$) for TD-DFT / TDA Linear Response.
//!
//! Precomputes spin-adapted Singlet ($\Delta P^\alpha = +\Delta P^\beta = M$) and Triplet
//! ($\Delta P^\alpha = -\Delta P^\beta = M$) local second-derivative invariants on a [`MolecularGrid`]
//! around a closed-shell ground state $P_0^t = 2 P_0^\alpha$, enabling ultra-fast Rayon-parallel
//! matrix-vector contractions $\delta V^{\text{xc}}[M]$ inside Davidson iterations with zero
//! transcendental function evaluations per trial vector.

use calchem_basis::BasisSet;
use calchem_tensor::{gemm_nn_into, gemm_nt_into, Matrix2D};
use rayon::prelude::*;

use crate::ao_grid::{eval_aos_grad, eval_aos_grad_filtered, AoGradVal, PrecomputedAoShell};
use crate::grid::{GridPoint, MolecularGrid};
use crate::xc::{eval_xc_spin, XcFunctional};

const FXC_CHUNK_SIZE: usize = 256;
const FXC_GEMM_BATCH: usize = 128;
/// Ground-state total density screening cutoff (avoids divergent $\rho^{-2/3}$ / $\rho^{-8/3}$ vacuum tails).
const FXC_RHO_CUTOFF: f64 = 1e-12;
const GAMMA_ANISO_CUTOFF: f64 = 1e-28;

fn fxc_budget_bytes() -> usize {
    std::env::var("CALCHEM_MAX_FXC_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&m| m > 0)
        .unwrap_or_else(|| {
            let default_cap = 512usize << 20;
            if let Some(max_mem) = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&m| m > 0)
            {
                default_cap.min((max_mem / 4).max(1))
            } else {
                default_cap
            }
        })
}

fn fxc_workers_and_b_chunk(
    n: usize,
    b_total: usize,
    budget_bytes: usize,
) -> (usize, usize) {
    if n == 0 || b_total == 0 {
        return (1, b_total.max(1));
    }
    let per_vec_worker = 3usize
        .saturating_mul(n)
        .saturating_mul(n)
        .saturating_add(2usize.saturating_mul(n).saturating_mul(FXC_GEMM_BATCH))
        .saturating_mul(8)
        .max(1);
    let max_threads = rayon::current_num_threads().max(1);
    let workers = max_threads.min((budget_bytes / per_vec_worker).max(1));
    let b_chunk = (budget_bytes / (workers.saturating_mul(per_vec_worker).max(1)))
        .max(1)
        .min(b_total);
    (workers, b_chunk)
}

struct FxcBatchWorkspace<'a> {
    candidate_shells: Vec<&'a PrecomputedAoShell>,
    ao_indices: Vec<usize>,
    ao_to_local: Vec<usize>,
    active: Vec<AoGradVal>,
    phi: Vec<f64>,
    dphi_x: Vec<f64>,
    dphi_y: Vec<f64>,
    dphi_z: Vec<f64>,
    m_sub_all: Vec<f64>,
    t_all: Vec<f64>,
    z_all: Vec<f64>,
    v_sub_all: Vec<f64>,
    eta: [f64; FXC_GEMM_BATCH],
    gx: [f64; FXC_GEMM_BATCH],
    gy: [f64; FXC_GEMM_BATCH],
    gz: [f64; FXC_GEMM_BATCH],
    half_wdv: [f64; FXC_GEMM_BATCH],
    wu_x: [f64; FXC_GEMM_BATCH],
    wu_y: [f64; FXC_GEMM_BATCH],
    wu_z: [f64; FXC_GEMM_BATCH],
}

impl<'a> FxcBatchWorkspace<'a> {
    fn new(n_ao: usize, n_shells: usize) -> Self {
        Self {
            candidate_shells: Vec::with_capacity(n_shells),
            ao_indices: Vec::with_capacity(n_ao),
            ao_to_local: vec![usize::MAX; n_ao],
            active: Vec::with_capacity(n_ao),
            phi: Vec::new(),
            dphi_x: Vec::new(),
            dphi_y: Vec::new(),
            dphi_z: Vec::new(),
            m_sub_all: Vec::new(),
            t_all: Vec::new(),
            z_all: Vec::new(),
            v_sub_all: Vec::new(),
            eta: [0.0; FXC_GEMM_BATCH],
            gx: [0.0; FXC_GEMM_BATCH],
            gy: [0.0; FXC_GEMM_BATCH],
            gz: [0.0; FXC_GEMM_BATCH],
            half_wdv: [0.0; FXC_GEMM_BATCH],
            wu_x: [0.0; FXC_GEMM_BATCH],
            wu_y: [0.0; FXC_GEMM_BATCH],
            wu_z: [0.0; FXC_GEMM_BATCH],
        }
    }

    fn prepare(&mut self, b_c: usize, m: usize, g_len: usize, is_gga: bool) {
        let mm = m * m;
        let mg = m * g_len;
        if self.phi.len() < mg {
            self.phi.resize(mg, 0.0);
        }
        self.phi[..mg].fill(0.0);
        if is_gga {
            if self.dphi_x.len() < mg {
                self.dphi_x.resize(mg, 0.0);
                self.dphi_y.resize(mg, 0.0);
                self.dphi_z.resize(mg, 0.0);
            }
            self.dphi_x[..mg].fill(0.0);
            self.dphi_y[..mg].fill(0.0);
            self.dphi_z[..mg].fill(0.0);
        }
        let b_mm = b_c * mm;
        if self.m_sub_all.len() < b_mm {
            self.m_sub_all.resize(b_mm, 0.0);
            self.v_sub_all.resize(b_mm, 0.0);
        }
        let b_mg = b_c * mg;
        if self.t_all.len() < b_mg {
            self.t_all.resize(b_mg, 0.0);
            self.z_all.resize(b_mg, 0.0);
        }
    }
}

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

        let eval_chunk = |chunk: &[GridPoint],
                          active: &mut Vec<AoGradVal>,
                          out_pts: &mut Vec<ActiveFxcPoint>| {
            for gp in chunk {
                let w = gp.weight;
                if w.abs() < 1e-22 {
                    continue;
                }
                eval_aos_grad(&shells, gp.coords, active);
                if active.is_empty() {
                    continue;
                }

                let mut rho_t = 0.0;
                let mut grad_rho_t = [0.0f64; 3];
                for a_i in active.iter() {
                    let row = &p_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let mut s = 0.0;
                    for a_j in active.iter() {
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
        };

        let points: Vec<ActiveFxcPoint> = if let Some((ref mol, radial_min)) = grid.streaming {
            if grid.points.is_empty() {
                let tasks = grid.streaming_tasks();
                tasks
                    .par_iter()
                    .flat_map_iter(|&(atom_idx, rad_start, rad_end)| {
                        let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
                        let mut batch_pts: Vec<GridPoint> = Vec::with_capacity(2048);
                        MolecularGrid::generate_task_points(
                            mol,
                            grid.level,
                            radial_min,
                            atom_idx,
                            rad_start,
                            rad_end,
                            &mut batch_pts,
                        );
                        let mut out_pts: Vec<ActiveFxcPoint> = Vec::with_capacity(batch_pts.len());
                        eval_chunk(&batch_pts, &mut active, &mut out_pts);
                        out_pts
                    })
                    .collect()
            } else {
                grid.points
                    .par_chunks(FXC_CHUNK_SIZE)
                    .flat_map_iter(|chunk| {
                        let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
                        let mut out_pts: Vec<ActiveFxcPoint> = Vec::with_capacity(chunk.len());
                        eval_chunk(chunk, &mut active, &mut out_pts);
                        out_pts
                    })
                    .collect()
            }
        } else {
            grid.points
                .par_chunks(FXC_CHUNK_SIZE)
                .flat_map_iter(|chunk| {
                    let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
                    let mut out_pts: Vec<ActiveFxcPoint> = Vec::with_capacity(chunk.len());
                    eval_chunk(chunk, &mut active, &mut out_pts);
                    out_pts
                })
                .collect()
        };

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
    /// adiabatic XC kernel with a batch of symmetric spatial transition density matrices
    /// `d_sym_list` (`n_ao x n_ao`), evaluating AOs once per grid point per memory-bounded batch.
    pub fn contract_batch(&self, d_sym_list: &[Matrix2D], triplet: bool) -> Vec<Matrix2D> {
        self.contract_batch_with_budget(d_sym_list, triplet, fxc_budget_bytes())
    }

    pub(crate) fn contract_batch_with_budget(
        &self,
        d_sym_list: &[Matrix2D],
        triplet: bool,
        budget_bytes: usize,
    ) -> Vec<Matrix2D> {
        if d_sym_list.is_empty() {
            return Vec::new();
        }
        let n = self.n_ao;
        let nn = n * n;
        for m in d_sym_list {
            assert_eq!(m.rows, n, "FxcKernel::contract_batch row mismatch");
            assert_eq!(m.cols, n, "FxcKernel::contract_batch col mismatch");
            assert_eq!(m.data.len(), nn, "FxcKernel::contract_batch data len mismatch");
        }
        if n == 0 || self.points.is_empty() {
            return vec![Matrix2D::zeros(n, n); d_sym_list.len()];
        }

        let is_gga = self.func.is_gga();
        let (workers, b_chunk) = fxc_workers_and_b_chunk(n, d_sym_list.len(), budget_bytes);
        let chunk_len = self.points.len().div_ceil(workers).max(FXC_CHUNK_SIZE);
        let mut results = Vec::with_capacity(d_sym_list.len());

        for b_slice in d_sym_list.chunks(b_chunk) {
            let b_c = b_slice.len();
            let v_all = self
                .points
                .par_chunks(chunk_len)
                .map(|pt_chunk| {
                    let mut v_loc = vec![0.0f64; b_c * nn];
                    let mut ws = FxcBatchWorkspace::new(n, self.shells.len());

                    for sub_chunk in pt_chunk.chunks(FXC_GEMM_BATCH) {
                        let g_len = sub_chunk.len();
                        if g_len == 0 {
                            continue;
                        }

                        let mut min_c = [f64::INFINITY; 3];
                        let mut max_c = [-f64::INFINITY; 3];
                        for pt in sub_chunk {
                            for q in 0..3 {
                                if pt.coords[q] < min_c[q] {
                                    min_c[q] = pt.coords[q];
                                }
                                if pt.coords[q] > max_c[q] {
                                    max_c[q] = pt.coords[q];
                                }
                            }
                        }

                        let center = [
                            0.5 * (min_c[0] + max_c[0]),
                            0.5 * (min_c[1] + max_c[1]),
                            0.5 * (min_c[2] + max_c[2]),
                        ];
                        let r_box = {
                            let dx = max_c[0] - center[0];
                            let dy = max_c[1] - center[1];
                            let dz = max_c[2] - center[2];
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        };

                        ws.candidate_shells.clear();
                        ws.ao_indices.clear();
                        for sh in &self.shells {
                            let dx = sh.center[0] - center[0];
                            let dy = sh.center[1] - center[1];
                            let dz = sh.center[2] - center[2];
                            let d2 = dx * dx + dy * dy + dz * dz;
                            let thresh = r_box + sh.r_cutoff + 1e-12;
                            if d2 <= thresh * thresh {
                                ws.candidate_shells.push(sh);
                                for c in 0..sh.n_functions() {
                                    ws.ao_indices.push(sh.ao_offset + c);
                                }
                            }
                        }

                        let m = ws.ao_indices.len();
                        if m == 0 {
                            continue;
                        }
                        debug_assert!(
                            ws.ao_indices.windows(2).all(|w| w[0] < w[1]),
                            "ao_indices must be strictly ascending"
                        );

                        let mm = m * m;
                        let mg = m * g_len;
                        ws.prepare(b_c, m, g_len, is_gga);

                        for (local_i, &mu) in ws.ao_indices.iter().enumerate() {
                            ws.ao_to_local[mu] = local_i;
                        }

                        for (g, pt) in sub_chunk.iter().enumerate() {
                            eval_aos_grad_filtered(&ws.candidate_shells, pt.coords, &mut ws.active);
                            for a in &ws.active {
                                let li = ws.ao_to_local[a.mu];
                                debug_assert!(li < m);
                                let idx = li * g_len + g;
                                ws.phi[idx] = a.val;
                                if is_gga {
                                    ws.dphi_x[idx] = a.grad[0];
                                    ws.dphi_y[idx] = a.grad[1];
                                    ws.dphi_z[idx] = a.grad[2];
                                }
                            }
                        }

                        for (b, m_mat) in b_slice.iter().enumerate() {
                            let m_slice = &m_mat.data;
                            let dst_base = b * mm;
                            for i in 0..m {
                                let mu = ws.ao_indices[i];
                                let src_row = &m_slice[mu * n..(mu + 1) * n];
                                let dst_row =
                                    &mut ws.m_sub_all[dst_base + i * m..dst_base + (i + 1) * m];
                                for j in 0..m {
                                    dst_row[j] = src_row[ws.ao_indices[j]];
                                }
                            }
                        }

                        gemm_nn_into(
                            b_c * m,
                            g_len,
                            m,
                            &ws.m_sub_all[..b_c * mm],
                            &ws.phi[..mg],
                            &mut ws.t_all[..b_c * mg],
                        );

                        for b in 0..b_c {
                            let t_b = &ws.t_all[b * mg..(b + 1) * mg];
                            let z_b = &mut ws.z_all[b * mg..(b + 1) * mg];

                            ws.eta[..g_len].fill(0.0);
                            if is_gga {
                                ws.gx[..g_len].fill(0.0);
                                ws.gy[..g_len].fill(0.0);
                                ws.gz[..g_len].fill(0.0);
                                for i in 0..m {
                                    let row_off = i * g_len;
                                    let phi_row = &ws.phi[row_off..row_off + g_len];
                                    let dpx_row = &ws.dphi_x[row_off..row_off + g_len];
                                    let dpy_row = &ws.dphi_y[row_off..row_off + g_len];
                                    let dpz_row = &ws.dphi_z[row_off..row_off + g_len];
                                    let t_row = &t_b[row_off..row_off + g_len];
                                    for g in 0..g_len {
                                        let s = t_row[g];
                                        ws.eta[g] += phi_row[g] * s;
                                        ws.gx[g] += 2.0 * dpx_row[g] * s;
                                        ws.gy[g] += 2.0 * dpy_row[g] * s;
                                        ws.gz[g] += 2.0 * dpz_row[g] * s;
                                    }
                                }
                                for (g, pt) in sub_chunk.iter().enumerate() {
                                    let k = if triplet { &pt.triplet } else { &pt.singlet };
                                    let w = pt.weight;
                                    let eta_g = ws.eta[g];
                                    let xi = [ws.gx[g], ws.gy[g], ws.gz[g]];
                                    let sigma = dot3(pt.g_s, xi);
                                    let dv_rho = k.k_rr * eta_g + k.k_rg * sigma;
                                    let g_coeff = k.k_rg * eta_g + k.k_gg * sigma;
                                    ws.half_wdv[g] = 0.5 * w * dv_rho;
                                    ws.wu_x[g] = w * (k.u_iso * xi[0] + g_coeff * pt.g_s[0]);
                                    ws.wu_y[g] = w * (k.u_iso * xi[1] + g_coeff * pt.g_s[1]);
                                    ws.wu_z[g] = w * (k.u_iso * xi[2] + g_coeff * pt.g_s[2]);
                                }
                                for i in 0..m {
                                    let row_off = i * g_len;
                                    let phi_row = &ws.phi[row_off..row_off + g_len];
                                    let dpx_row = &ws.dphi_x[row_off..row_off + g_len];
                                    let dpy_row = &ws.dphi_y[row_off..row_off + g_len];
                                    let dpz_row = &ws.dphi_z[row_off..row_off + g_len];
                                    let z_row = &mut z_b[row_off..row_off + g_len];
                                    for g in 0..g_len {
                                        z_row[g] = ws.half_wdv[g] * phi_row[g]
                                            + ws.wu_x[g] * dpx_row[g]
                                            + ws.wu_y[g] * dpy_row[g]
                                            + ws.wu_z[g] * dpz_row[g];
                                    }
                                }
                            } else {
                                for i in 0..m {
                                    let row_off = i * g_len;
                                    let phi_row = &ws.phi[row_off..row_off + g_len];
                                    let t_row = &t_b[row_off..row_off + g_len];
                                    for g in 0..g_len {
                                        ws.eta[g] += phi_row[g] * t_row[g];
                                    }
                                }
                                for (g, pt) in sub_chunk.iter().enumerate() {
                                    let k = if triplet { &pt.triplet } else { &pt.singlet };
                                    ws.half_wdv[g] = 0.5 * pt.weight * k.k_rr * ws.eta[g];
                                }
                                for i in 0..m {
                                    let row_off = i * g_len;
                                    let phi_row = &ws.phi[row_off..row_off + g_len];
                                    let z_row = &mut z_b[row_off..row_off + g_len];
                                    for g in 0..g_len {
                                        z_row[g] = ws.half_wdv[g] * phi_row[g];
                                    }
                                }
                            }
                        }

                        gemm_nt_into(
                            b_c * m,
                            m,
                            g_len,
                            &ws.z_all[..b_c * mg],
                            &ws.phi[..mg],
                            &mut ws.v_sub_all[..b_c * mm],
                        );

                        for b in 0..b_c {
                            let v_sub = &ws.v_sub_all[b * mm..(b + 1) * mm];
                            let v_mat = &mut v_loc[b * nn..(b + 1) * nn];
                            for i in 0..m {
                                let mu = ws.ao_indices[i];
                                let row_out = &mut v_mat[mu * n..(mu + 1) * n];
                                let row_v = &v_sub[i * m..(i + 1) * m];
                                for j in 0..=i {
                                    let nu = ws.ao_indices[j];
                                    row_out[nu] += row_v[j] + v_sub[j * m + i];
                                }
                            }
                        }

                        for &mu in &ws.ao_indices {
                            ws.ao_to_local[mu] = usize::MAX;
                        }
                    }

                    v_loc
                })
                .collect::<Vec<_>>()
                .into_iter()
                .reduce(|mut a, b| {
                    for (dst, src) in a.iter_mut().zip(b.iter()) {
                        *dst += *src;
                    }
                    a
                })
                .unwrap_or_else(|| vec![0.0f64; b_c * nn]);

            for b in 0..b_c {
                let mut mat = Matrix2D::zeros(n, n);
                mat.data.copy_from_slice(&v_all[b * nn..(b + 1) * nn]);
                for i in 0..n {
                    for j in 0..i {
                        let v = mat.data[i * n + j];
                        mat.data[j * n + i] = v;
                    }
                }
                results.push(mat);
            }
        }

        results
    }

    /// Contracts the precomputed Singlet (`triplet = false`) or Triplet (`triplet = true`)
    /// adiabatic XC kernel with a symmetric spatial transition density matrix $M$ (`n_ao x n_ao`),
    /// returning $\delta V^{\text{xc},\alpha}[P_0^\alpha + M, P_0^\beta \pm M]$.
    pub fn contract(&self, m_sym: &Matrix2D, triplet: bool) -> Matrix2D {
        self.contract_batch(std::slice::from_ref(m_sym), triplet)
            .pop()
            .expect("single matrix")
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
                    let fd = (va_m2.data[k] - 8.0 * va_m1.data[k] + 8.0 * va_p1.data[k]
                        - va_p2.data[k])
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

    fn contract_pointwise_ref(kernel: &FxcKernel, m_sym: &Matrix2D, triplet: bool) -> Matrix2D {
        let n = kernel.n_ao;
        if n == 0 || kernel.points.is_empty() {
            return Matrix2D::zeros(n, n);
        }
        let is_gga = kernel.func.is_gga();
        let m_slice = &m_sym.data;
        let mut v_loc = vec![0.0f64; n * n];
        let mut active: Vec<AoGradVal> = Vec::with_capacity(n);
        let mut z_vec: Vec<f64> = Vec::with_capacity(n);

        for pt in &kernel.points {
            eval_aos_grad(&kernel.shells, pt.coords, &mut active);
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

        let mut out = Matrix2D::zeros(n, n);
        out.data = v_loc;
        out
    }

    fn build_water_fxc_inputs() -> (Molecule, BasisSet, MolecularGrid, Matrix2D, Vec<Matrix2D>) {
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

        let mut p0_tot = Matrix2D::zeros(n, n);
        for i in 0..n {
            p0_tot.set(i, i, 0.25 + 0.01 * (i as f64));
            for j in (i + 1)..n {
                let v = 0.015 * (((i * 7 + j * 3) as f64).sin());
                p0_tot.set(i, j, v);
                p0_tot.set(j, i, v);
            }
        }

        let mut d_list = Vec::with_capacity(5);
        for b in 0..5 {
            let mut m_sym = Matrix2D::zeros(n, n);
            for i in 0..n {
                for j in i..n {
                    let v = 0.02 * (((i * (5 + b) + j * (11 + 2 * b) + b + 1) as f64).cos());
                    m_sym.set(i, j, v);
                    m_sym.set(j, i, v);
                }
            }
            d_list.push(m_sym);
        }
        (mol, bs, grid, p0_tot, d_list)
    }

    #[test]
    fn test_fxc_contract_batch_matches_pointwise_reference_and_fd() {
        let (_mol, bs, grid, p0_tot, d_list) = build_water_fxc_inputs();
        let n = bs.n_ao;
        let eps = 1e-5;

        for func in [
            XcFunctional::Svwn5,
            XcFunctional::Pbe,
            XcFunctional::B3lyp,
            XcFunctional::CamB3lyp,
        ] {
            let kernel = FxcKernel::new(&bs, &grid, func, &p0_tot);
            for (triplet, sign_b) in [(false, 1.0), (true, -1.0)] {
                let batch = kernel.contract_batch(&d_list, triplet);
                assert_eq!(batch.len(), d_list.len());

                let single = kernel.contract_batch(&d_list[..1], triplet);
                let via_contract = kernel.contract(&d_list[0], triplet);
                for k in 0..(n * n) {
                    assert!((batch[0].data[k] - single[0].data[k]).abs() <= 1e-13);
                    assert!((batch[0].data[k] - via_contract.data[k]).abs() <= 1e-13);
                }

                for (b, m_sym) in d_list.iter().enumerate() {
                    assert!(
                        batch[b].sym_frobenius_error() < 1e-13,
                        "Fxc contract_batch output not symmetric for {func:?} b={b}"
                    );
                    let ref_mat = contract_pointwise_ref(&kernel, m_sym, triplet);
                    let max_err = batch[b]
                        .data
                        .iter()
                        .zip(&ref_mat.data)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_err <= 1e-13,
                        "Fxc contract_batch vs pointwise ref error {max_err:.3e} for {func:?} triplet={triplet} b={b}"
                    );
                }

                // Cross-check batch[0] against 4-point central FD of UKS Vxc
                let m0 = &d_list[0];
                let build_perturbed = |c_a: f64, c_b: f64| -> (Matrix2D, Matrix2D) {
                    let mut pa = Matrix2D::zeros(n, n);
                    let mut pb = Matrix2D::zeros(n, n);
                    for k in 0..(n * n) {
                        pa.data[k] = 0.5 * p0_tot.data[k] + c_a * m0.data[k];
                        pb.data[k] = 0.5 * p0_tot.data[k] + c_b * m0.data[k];
                    }
                    (pa, pb)
                };
                let (pa_m2, pb_m2) = build_perturbed(-2.0 * eps, -2.0 * sign_b * eps);
                let (pa_m1, pb_m1) = build_perturbed(-eps, -sign_b * eps);
                let (pa_p1, pb_p1) = build_perturbed(eps, sign_b * eps);
                let (pa_p2, pb_p2) = build_perturbed(2.0 * eps, 2.0 * sign_b * eps);
                let (_, va_m2, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_m2, &pb_m2);
                let (_, va_m1, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_m1, &pb_m1);
                let (_, va_p1, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_p1, &pb_p1);
                let (_, va_p2, _) = evaluate_vxc_uks(&bs, &grid, func, &pa_p2, &pb_p2);
                let mut fd_err = 0.0f64;
                for k in 0..(n * n) {
                    let fd = (va_m2.data[k] - 8.0 * va_m1.data[k] + 8.0 * va_p1.data[k]
                        - va_p2.data[k])
                        / (12.0 * eps);
                    fd_err = fd_err.max((batch[0].data[k] - fd).abs());
                }
                assert!(
                    fd_err < 1e-8,
                    "Fxc contract_batch FD mismatch {fd_err:.3e} for {func:?} triplet={triplet}"
                );
            }
        }
    }

    #[test]
    fn test_fxc_contract_batch_edge_cases_and_b_chunking() {
        let (_mol, bs, grid, p0_tot, d_list) = build_water_fxc_inputs();
        let n = bs.n_ao;

        let kernel = FxcKernel::new(&bs, &grid, XcFunctional::B3lyp, &p0_tot);
        assert!(kernel.contract_batch(&[], false).is_empty());

        let empty_grid = MolecularGrid {
            points: Vec::new(),
            level: GridLevel::Coarse,
            streaming: None,
        };
        let empty_kernel = FxcKernel::new(&bs, &empty_grid, XcFunctional::B3lyp, &p0_tot);
        let zero_outs = empty_kernel.contract_batch(&d_list, false);
        assert_eq!(zero_outs.len(), d_list.len());
        for mat in &zero_outs {
            assert_eq!(mat.rows, n);
            assert_eq!(mat.cols, n);
            assert!(mat.data.iter().all(|&v| v == 0.0));
        }

        // Forced multi-pass B-chunking across LDA (Svwn5) and GGA (B3lyp)
        for func in [XcFunctional::Svwn5, XcFunctional::B3lyp] {
            let k = FxcKernel::new(&bs, &grid, func, &p0_tot);
            for triplet in [false, true] {
                let unchunked = k.contract_batch(&d_list, triplet);
                let chunked = k.contract_batch_with_budget(&d_list, triplet, 32 * 1024);
                assert_eq!(chunked.len(), unchunked.len());
                for (u, c) in unchunked.iter().zip(&chunked) {
                    let max_err = u
                        .data
                        .iter()
                        .zip(&c.data)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_err <= 1e-13,
                        "Forced B-chunking mismatch {max_err:.3e} for {func:?} triplet={triplet}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fxc_contract_batch_wb97xd_and_zero_collinear_trial_vectors() {
        let (_mol, bs, grid, p0_tot, d_list) = build_water_fxc_inputs();
        let n = bs.n_ao;
        let kernel = FxcKernel::new(&bs, &grid, XcFunctional::Wb97xd, &p0_tot);

        let zero_m = Matrix2D::zeros(n, n);
        let mut scaled_m = d_list[0].clone();
        for v in &mut scaled_m.data {
            *v *= 2.5;
        }
        let batch = vec![zero_m, d_list[0].clone(), scaled_m];
        for triplet in [false, true] {
            let out = kernel.contract_batch(&batch, triplet);
            assert_eq!(out.len(), 3);
            assert!(out[0].data.iter().all(|&v| v == 0.0));
            let ref_mat = contract_pointwise_ref(&kernel, &d_list[0], triplet);
            for idx in 0..(n * n) {
                assert!((out[1].data[idx] - ref_mat.data[idx]).abs() < 1e-13);
                assert!((out[2].data[idx] - 2.5 * out[1].data[idx]).abs() < 1e-12);
            }
        }
    }
}
