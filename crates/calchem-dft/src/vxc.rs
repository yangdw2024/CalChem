//! Parallel Kohn-Sham Exchange-Correlation Matrix Assembly ($E_{\text{xc}}, V^{\text{xc}}$)
//! and Exact Analytical XC Nuclear Gradients $\left.\frac{\partial E_{\text{xc}}}{\partial \mathbf{R}_A}\right|_{P^\alpha, P^\beta}$.
//!
//! Includes all three exact Pople-Johnson-Gill (1992/1993) DFT nuclear gradient contributions:
//! 1. Orbital center derivative via AO gradients $\nabla\phi_\mu$ and Hessians $\nabla\nabla\phi_\mu$.
//! 2. Host-atom quadrature point co-moving translation invariance ($\mathbf{r}_g = \mathbf{R}_{H(g)} + \mathbf{u}_g$).
//! 3. Analytical Becke fuzzy-Voronoi quadrature weight derivatives $\frac{d w_g}{d \mathbf{R}_A}$.

use calchem_basis::BasisSet;
use calchem_model::Molecule;
use calchem_tensor::Matrix2D;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::ao_grid::{
    eval_aos_grad, eval_aos_grad_filtered, eval_aos_hess, AoGradVal, AoHessVal, PrecomputedAoShell,
};
use crate::grid::{becke_weight_nuclear_gradient, GridPoint, MolecularGrid};
use crate::xc::{eval_xc_spin, XcFunctional};

const GRID_CHUNK_SIZE: usize = 256;
const DENSITY_SCREEN: f64 = 1e-18;

fn bounded_worker_count(n: usize, channels: usize) -> usize {
    let bytes = n
        .saturating_mul(n)
        .saturating_mul(8)
        .saturating_mul(channels)
        .max(1);
    let max_mem = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&m| m > 0);
    let direct_cap = std::env::var("CALCHEM_MAX_DIRECT_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4 * 1024 * 1024 * 1024);
    let budget = match max_mem {
        Some(m) => direct_cap.min(m / 2),
        None => direct_cap,
    };
    rayon::current_num_threads().min((budget / bytes).max(1))
}

fn bounded_grid_chunk_size(points: usize, n: usize, channels: usize) -> usize {
    let workers = bounded_worker_count(n, channels);
    points.div_ceil(workers).max(GRID_CHUNK_SIZE)
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub struct RksWorkspace<'a> {
    pub active: Vec<AoGradVal>,
    pub local_indices: Vec<usize>,
    pub vals: Vec<f64>,
    pub phi_p: Vec<f64>,
    pub z_vec: Vec<f64>,
    pub candidate_shells: Vec<&'a PrecomputedAoShell>,
    pub ao_indices: Vec<usize>,
    pub ao_to_local: Vec<usize>,
    pub p_sub: Vec<f64>,
    pub v_sub: Vec<f64>,
}

impl<'a> RksWorkspace<'a> {
    pub fn new(n: usize, n_shells: usize) -> Self {
        let init_m = n.min(256);
        Self {
            active: Vec::with_capacity(n),
            local_indices: Vec::with_capacity(n),
            vals: Vec::with_capacity(n),
            phi_p: Vec::with_capacity(n),
            z_vec: Vec::with_capacity(n),
            candidate_shells: Vec::with_capacity(n_shells),
            ao_indices: Vec::with_capacity(n),
            ao_to_local: vec![usize::MAX; n],
            p_sub: vec![0.0f64; init_m * init_m],
            v_sub: vec![0.0f64; init_m * init_m],
        }
    }

    #[inline]
    pub fn ensure_capacity(&mut self, m: usize) {
        let needed = m * m;
        if self.p_sub.len() < needed {
            self.p_sub.resize(needed, 0.0);
            self.v_sub.resize(needed, 0.0);
        }
    }
}

const BATCH_SIZE: usize = 128;

#[allow(clippy::too_many_arguments)]
fn accumulate_rks_points_with<'a, F>(
    chunk: &[GridPoint],
    n: usize,
    shells: &'a [PrecomputedAoShell],
    p_slice: &[f64],
    mut eval_fn: F,
    e_loc: &mut f64,
    e_comp: &mut f64,
    v_loc: &mut [f64],
    ws: &mut RksWorkspace<'a>,
) where
    F: FnMut(f64, f64) -> crate::xc::XcPointOut,
{
    for sub_chunk in chunk.chunks(BATCH_SIZE) {
        let mut min_c = [f64::INFINITY; 3];
        let mut max_c = [-f64::INFINITY; 3];
        let mut has_points = false;

        for gp in sub_chunk {
            if gp.weight.abs() < 1e-24 {
                continue;
            }
            has_points = true;
            for q in 0..3 {
                if gp.coords[q] < min_c[q] { min_c[q] = gp.coords[q]; }
                if gp.coords[q] > max_c[q] { max_c[q] = gp.coords[q]; }
            }
        }

        if !has_points {
            continue;
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

        for sh in shells {
            let dx = sh.center[0] - center[0];
            let dy = sh.center[1] - center[1];
            let dz = sh.center[2] - center[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let thresh = r_box + sh.r_cutoff;
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

        ws.ensure_capacity(m);

        for (local_i, &mu) in ws.ao_indices.iter().enumerate() {
            ws.ao_to_local[mu] = local_i;
        }

        for i in 0..m {
            let mu = ws.ao_indices[i];
            let row_p = &p_slice[mu * n..(mu + 1) * n];
            let sub_p_row = &mut ws.p_sub[i * m..(i + 1) * m];
            for j in 0..m {
                let nu = ws.ao_indices[j];
                sub_p_row[j] = row_p[nu];
            }
        }

        ws.v_sub[..m * m].fill(0.0);

        for gp in sub_chunk {
            let w = gp.weight;
            if w.abs() < 1e-24 {
                continue;
            }
            eval_aos_grad_filtered(&ws.candidate_shells, gp.coords, &mut ws.active);
            if ws.active.is_empty() {
                continue;
            }

            ws.local_indices.clear();
            ws.vals.clear();
            for a in ws.active.iter() {
                ws.local_indices.push(ws.ao_to_local[a.mu]);
                ws.vals.push(a.val);
            }

            ws.phi_p.clear();
            let mut rho_t = 0.0;
            let mut grad_rho_t = [0.0f64; 3];

            for (k_i, a_i) in ws.active.iter().enumerate() {
                let li = ws.local_indices[k_i];
                let sub_p_row = &ws.p_sub[li * m..(li + 1) * m];
                let mut s = 0.0;
                for (&lj, &v) in ws.local_indices.iter().zip(ws.vals.iter()) {
                    s += sub_p_row[lj] * v;
                }
                ws.phi_p.push(s);
                rho_t += a_i.val * s;
                grad_rho_t[0] += 2.0 * a_i.grad[0] * s;
                grad_rho_t[1] += 2.0 * a_i.grad[1] * s;
                grad_rho_t[2] += 2.0 * a_i.grad[2] * s;
            }

            if rho_t < DENSITY_SCREEN {
                continue;
            }

            let rho_s = 0.5 * rho_t;
            let g_ss = 0.25 * dot3(grad_rho_t, grad_rho_t);
            let out = eval_fn(rho_s, g_ss);

            let term = w * out.f_xc - *e_comp;
            let next = *e_loc + term;
            *e_comp = (next - *e_loc) - term;
            *e_loc = next;

            // Closed-shell spin symmetry: U_alpha = (v_gamma_aa + 0.5 * v_gamma_ab) * grad_rho_t
            let u_scale = out.v_gamma_aa + 0.5 * out.v_gamma_ab;
            let u_vec = [
                u_scale * grad_rho_t[0],
                u_scale * grad_rho_t[1],
                u_scale * grad_rho_t[2],
            ];

            ws.z_vec.clear();
            let half_wv = 0.5 * w * out.v_rho_a;
            for a_i in ws.active.iter() {
                let z_i = half_wv * a_i.val + w * dot3(u_vec, a_i.grad);
                ws.z_vec.push(z_i);
            }

            let pt_m = ws.active.len();
            for i in 0..pt_m {
                let li = ws.local_indices[i];
                let phi_i = ws.vals[i];
                let z_i = ws.z_vec[i];
                let sub_v_row = &mut ws.v_sub[li * m..(li + 1) * m];
                for j in 0..=i {
                    let lj = ws.local_indices[j];
                    sub_v_row[lj] += z_i * ws.vals[j] + phi_i * ws.z_vec[j];
                }
            }
        }

        // Scatter V_sub into thread-local v_loc (lower triangle)
        for i in 0..m {
            let mu = ws.ao_indices[i];
            let row_out = &mut v_loc[mu * n..(mu + 1) * n];
            let sub_v_row = &ws.v_sub[i * m..(i + 1) * m];
            for j in 0..=i {
                let nu = ws.ao_indices[j];
                row_out[nu] += sub_v_row[j];
            }
        }

        // Reset ao_to_local for next batch
        for &mu in &ws.ao_indices {
            ws.ao_to_local[mu] = usize::MAX;
        }
    }
}

/// Evaluates closed-shell Kohn-Sham $(E_{\text{xc}}, V^{\text{xc}})$ from spin-summed total density
/// $P^t = 2 P^\alpha$ (`n_ao x n_ao`). Supports both prebuilt and zero-residency streaming grids.
pub fn evaluate_vxc_rks(
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    density_total: &Matrix2D,
) -> (f64, Matrix2D) {
    let n = basis.n_ao;
    if n == 0 || (grid.points.is_empty() && grid.streaming.is_none()) {
        return (0.0, Matrix2D::zeros(n, n));
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let p_slice = &density_total.data;

    let (e_xc, v_upper) = if let Some((ref mol, radial_min)) =
        grid.streaming.as_ref().filter(|_| grid.points.is_empty())
    {
        let tasks = grid.streaming_tasks();
        if tasks.is_empty() {
            return (0.0, Matrix2D::zeros(n, n));
        }
        let workers = bounded_worker_count(n, 1);
        let task_chunk_len = tasks.len().div_ceil(workers).max(1);
        tasks
            .par_chunks(task_chunk_len)
            .map(|task_slice| {
                let mut e_loc = 0.0f64;
                let mut e_comp = 0.0f64;
                let mut v_loc = vec![0.0f64; n * n];
                let mut ws = RksWorkspace::new(n, shells.len());
                let mut batch_pts: Vec<GridPoint> = Vec::with_capacity(2048);

                for &(atom_idx, rad_start, rad_end) in task_slice {
                    batch_pts.clear();
                    MolecularGrid::generate_task_points(
                        mol,
                        grid.level,
                        *radial_min,
                        atom_idx,
                        rad_start,
                        rad_end,
                        &mut batch_pts,
                    );
                    accumulate_rks_points_with(
                        &batch_pts,
                        n,
                        &shells,
                        p_slice,
                        |rs, gss| eval_xc_spin(func, rs, rs, gss, gss, gss),
                        &mut e_loc,
                        &mut e_comp,
                        &mut v_loc,
                        &mut ws,
                    );
                }
                (e_loc, v_loc)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|(ea, mut va), (eb, vb)| {
                for (dst, src) in va.iter_mut().zip(vb.iter()) {
                    *dst += *src;
                }
                (ea + eb, va)
            })
            .expect("nonempty streaming tasks")
    } else {
        grid.points
            .par_chunks(bounded_grid_chunk_size(grid.points.len(), n, 1))
            .map(|chunk| {
                let mut e_loc = 0.0f64;
                let mut e_comp = 0.0f64;
                let mut v_loc = vec![0.0f64; n * n];
                let mut ws = RksWorkspace::new(n, shells.len());

                accumulate_rks_points_with(
                    chunk,
                    n,
                    &shells,
                    p_slice,
                    |rs, gss| eval_xc_spin(func, rs, rs, gss, gss, gss),
                    &mut e_loc,
                    &mut e_comp,
                    &mut v_loc,
                    &mut ws,
                );
                (e_loc, v_loc)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|(ea, mut va), (eb, vb)| {
                for (dst, src) in va.iter_mut().zip(vb.iter()) {
                    *dst += *src;
                }
                (ea + eb, va)
            })
            .expect("nonempty grid")
    };

    let mut v_mat = Matrix2D {rows:n,cols:n,data:v_upper};
    for i in 0..n {
        for j in 0..i { v_mat.data[j*n+i] = v_mat.data[i*n+j]; }
    }
    (e_xc, v_mat)
}

struct CachedAoBatch {
    ao_indices: Vec<usize>,
    weights: Vec<f64>,
    /// Row-major `(m, b)` where `m = ao_indices.len()`, `b = weights.len()`.
    phi: Vec<f64>,
    dphi_x: Vec<f64>,
    dphi_y: Vec<f64>,
    dphi_z: Vec<f64>,
}

struct GpuVxcState {
    grid: calchem_cuda::GpuVxcGrid,
    weights: Vec<f64>,
    rho_grad: Mutex<Vec<f64>>,
    wv: Mutex<Vec<f64>>,
}

/// High-performance GPU + BLAS-3 (`faer` GEMM) RKS exchange-correlation integration engine
/// with spatial-batched AO screening across SCF iterations.
pub struct RksVxcEngine {
    n_ao: usize,
    use_gpu: bool,
    gpu: Option<GpuVxcState>,
    batches: Option<Vec<CachedAoBatch>>,
    pub gpu_kernel_invocations: AtomicUsize,
    pub gpu_fallback_invocations: AtomicUsize,
}

impl RksVxcEngine {
    /// Builds a CPU BLAS-3 RKS $V_{\text{xc}}$ engine for a fixed geometry/basis/grid.
    pub fn new(basis: &BasisSet, grid: &MolecularGrid) -> Self {
        Self::new_with_gpu(basis, grid, false)
    }

    pub fn telemetry(&self) -> (usize, usize) {
        (
            self.gpu_kernel_invocations.load(Ordering::Relaxed),
            self.gpu_fallback_invocations.load(Ordering::Relaxed),
        )
    }

    /// Builds an RKS $V_{\text{xc}}$ engine, optionally uploading compact spatial-screened
    /// grid & basis metadata (~10–20 MB) to CUDA VRAM for on-the-fly FP64 GPU evaluation.
    pub fn new_with_gpu(basis: &BasisSet, grid: &MolecularGrid, use_gpu: bool) -> Self {
        let n_ao = basis.n_ao;
        if n_ao == 0 || grid.points.is_empty() || grid.streaming.is_some() {
            return Self {
                n_ao,
                use_gpu,
                gpu: None,
                batches: None,
                gpu_kernel_invocations: AtomicUsize::new(0),
                gpu_fallback_invocations: AtomicUsize::new(0),
            };
        }

        let shells = PrecomputedAoShell::from_basis(basis);
        let raw_chunks: Vec<&[GridPoint]> = grid.points.chunks(BATCH_SIZE).collect();

        // First pass: screen shells per batch and compute active point indices.
        let batch_meta: Vec<(Vec<usize>, Vec<usize>)> = raw_chunks
            .par_iter()
            .map(|sub_chunk| {
                let mut min_c = [f64::INFINITY; 3];
                let mut max_c = [-f64::INFINITY; 3];
                let mut pt_idxs = Vec::with_capacity(sub_chunk.len());
                for (local_g, gp) in sub_chunk.iter().enumerate() {
                    if gp.weight.abs() < 1e-24 {
                        continue;
                    }
                    pt_idxs.push(local_g);
                    for q in 0..3 {
                        if gp.coords[q] < min_c[q] {
                            min_c[q] = gp.coords[q];
                        }
                        if gp.coords[q] > max_c[q] {
                            max_c[q] = gp.coords[q];
                        }
                    }
                }
                if pt_idxs.is_empty() {
                    return (Vec::new(), pt_idxs);
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
                let mut cand_shell_idxs = Vec::new();
                for (si, sh) in shells.iter().enumerate() {
                    let dx = sh.center[0] - center[0];
                    let dy = sh.center[1] - center[1];
                    let dz = sh.center[2] - center[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    let thresh = r_box + sh.r_cutoff;
                    if d2 <= thresh * thresh {
                        cand_shell_idxs.push(si);
                    }
                }
                (cand_shell_idxs, pt_idxs)
            })
            .collect();

        // Try GPU on-the-fly FP64 Vxc path first when GPU is enabled (< 25 MB VRAM, zero 2 GB RAM cache!)
        let init_gpu_fallback = 0usize;
        if use_gpu {
            if calchem_cuda::is_cuda_available() {
                let mut gpu_shells = Vec::with_capacity(shells.len());
                let mut gpu_prims = Vec::new();
                let mut gpu_comps = Vec::new();
                for sh in &shells {
                    let prim_start = gpu_prims.len() as i32;
                    for pr in &sh.prims {
                        gpu_prims.push(calchem_cuda::AoPrimGpu {
                            exponent: pr.exponent,
                            weighted_norm: pr.weighted_norm,
                        });
                    }
                    let comp_start = gpu_comps.len() as i32;
                    for (ic, &c) in sh.comps.iter().enumerate() {
                        gpu_comps.push(calchem_cuda::AoCompGpu {
                            scale: sh.comp_scales[ic],
                            lx: c[0] as i32,
                            ly: c[1] as i32,
                            lz: c[2] as i32,
                            _pad: 0,
                        });
                    }
                    gpu_shells.push(calchem_cuda::AoShellGpu {
                        center: sh.center,
                        r2_cutoff: sh.r2_cutoff,
                        ao_offset: sh.ao_offset as i32,
                        n_funcs: sh.comps.len() as i32,
                        prim_start,
                        n_prims: sh.prims.len() as i32,
                        comp_start,
                        atom_index: sh.atom_index as i32,
                    });
                }

                let mut gpu_pts = Vec::with_capacity(grid.points.len());
                let mut weights = Vec::with_capacity(grid.points.len());
                let mut cand_offsets = Vec::with_capacity(batch_meta.len() + 1);
                let mut cand_shells_flat = Vec::new();
                cand_offsets.push(0i32);
                let mut batch_id = 0i32;

                for (sub_chunk, (cand_sh, pt_idxs)) in raw_chunks.iter().zip(batch_meta.iter()) {
                    if pt_idxs.is_empty() || cand_sh.is_empty() {
                        continue;
                    }
                    for &si in cand_sh {
                        cand_shells_flat.push(si as i32);
                    }
                    cand_offsets.push(cand_shells_flat.len() as i32);
                    for &lg in pt_idxs {
                        let gp = &sub_chunk[lg];
                        gpu_pts.push(calchem_cuda::GridPtGpu {
                            x: gp.coords[0],
                            y: gp.coords[1],
                            z: gp.coords[2],
                            batch_id,
                            host_atom: gp.host_atom as i32,
                        });
                        weights.push(gp.weight);
                    }
                    batch_id += 1;
                }

                if !gpu_pts.is_empty() && !cand_shells_flat.is_empty() {
                    if let Some(gpu_grid) = calchem_cuda::GpuVxcGrid::upload(
                        &gpu_shells,
                        &gpu_prims,
                        &gpu_comps,
                        &gpu_pts,
                        &cand_offsets,
                        &cand_shells_flat,
                        n_ao,
                    ) {
                        if std::env::var_os("CALCHEM_PROFILE").is_some() {
                            eprintln!(
                                "[vxc_gpu] uploaded pts={} batches={} cand_entries={}",
                                gpu_pts.len(),
                                batch_id,
                                cand_shells_flat.len()
                            );
                        }
                        let n_pts = weights.len();
                        return Self {
                            n_ao,
                            use_gpu,
                            gpu: Some(GpuVxcState {
                                grid: gpu_grid,
                                weights,
                                rho_grad: Mutex::new(vec![0.0f64; n_pts * 4]),
                                wv: Mutex::new(vec![0.0f64; n_pts * 4]),
                            }),
                            batches: None,
                            gpu_kernel_invocations: AtomicUsize::new(0),
                            gpu_fallback_invocations: AtomicUsize::new(0),
                        };
                    }
                }
            }
        }

        let max_mem = std::env::var("CALCHEM_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&m| m > 0);
        let default_cap = 2560usize << 20;
        let mem_cap = match max_mem {
            Some(m) => default_cap.min(m / 2),
            None => default_cap,
        };
        let max_bytes: usize = std::env::var("CALCHEM_MAX_AO_GRID_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| match max_mem {
                Some(m) => v.min(m),
                None => v,
            })
            .unwrap_or(mem_cap);
        if max_bytes == 0 {
            return Self {
                n_ao,
                use_gpu,
                gpu: None,
                batches: None,
                gpu_kernel_invocations: AtomicUsize::new(0),
                gpu_fallback_invocations: AtomicUsize::new(init_gpu_fallback),
            };
        }

        let mut total_bytes: u128 = 0;
        for (cand_shells, pt_idxs) in &batch_meta {
            let m: usize = cand_shells.iter().map(|&si| shells[si].n_functions()).sum();
            let b = pt_idxs.len();
            total_bytes += (m as u128) * (b as u128) * 4 * 8;
        }
        if std::env::var_os("CALCHEM_PROFILE").is_some() {
            eprintln!(
                "[vxc_cache] batches={} bytes={} limit={}",
                batch_meta.len(),
                total_bytes,
                max_bytes
            );
        }
        if total_bytes > max_bytes as u128 {
            return Self {
                n_ao,
                use_gpu,
                gpu: None,
                batches: None,
                gpu_kernel_invocations: AtomicUsize::new(0),
                gpu_fallback_invocations: AtomicUsize::new(init_gpu_fallback),
            };
        }

        // Second pass: evaluate AO values & gradients into contiguous row-major (m x b) blocks.
        let batches: Vec<CachedAoBatch> = raw_chunks
            .par_iter()
            .zip(batch_meta.par_iter())
            .filter_map(|(sub_chunk, (cand_shell_idxs, pt_idxs))| {
                let b = pt_idxs.len();
                if b == 0 || cand_shell_idxs.is_empty() {
                    return None;
                }
                let cand_shells: Vec<&PrecomputedAoShell> =
                    cand_shell_idxs.iter().map(|&si| &shells[si]).collect();
                let mut ao_indices = Vec::new();
                for sh in &cand_shells {
                    for c in 0..sh.n_functions() {
                        ao_indices.push(sh.ao_offset + c);
                    }
                }
                let m = ao_indices.len();
                let mut ao_to_local = vec![usize::MAX; n_ao];
                for (li, &mu) in ao_indices.iter().enumerate() {
                    ao_to_local[mu] = li;
                }

                let mut weights = Vec::with_capacity(b);
                let mut phi = vec![0.0f64; m * b];
                let mut dphi_x = vec![0.0f64; m * b];
                let mut dphi_y = vec![0.0f64; m * b];
                let mut dphi_z = vec![0.0f64; m * b];
                let mut active = Vec::with_capacity(m);

                for (g, &local_g) in pt_idxs.iter().enumerate() {
                    let gp = &sub_chunk[local_g];
                    weights.push(gp.weight);
                    eval_aos_grad_filtered(&cand_shells, gp.coords, &mut active);
                    for a in &active {
                        let li = ao_to_local[a.mu];
                        if li < m {
                            let idx = li * b + g;
                            phi[idx] = a.val;
                            dphi_x[idx] = a.grad[0];
                            dphi_y[idx] = a.grad[1];
                            dphi_z[idx] = a.grad[2];
                        }
                    }
                }

                Some(CachedAoBatch {
                    ao_indices,
                    weights,
                    phi,
                    dphi_x,
                    dphi_y,
                    dphi_z,
                })
            })
            .collect();

        Self {
            n_ao,
            use_gpu,
            gpu: None,
            batches: Some(batches),
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(init_gpu_fallback),
        }
    }

    /// Evaluates closed-shell $(E_{\text{xc}}, V^{\text{xc}})$ using GPU FP64 kernels if resident,
    /// else batched BLAS-3 GEMM on cached AO grid, else streaming CPU fallback.
    pub fn evaluate(
        &self,
        basis: &BasisSet,
        grid: &MolecularGrid,
        func: XcFunctional,
        density_total: &Matrix2D,
    ) -> (f64, Matrix2D) {
        let n = self.n_ao;
        if let Some(ref gpu_state) = self.gpu {
            let mut rho_grad = gpu_state.rho_grad.lock().unwrap();
            let mut wv = gpu_state.wv.lock().unwrap();
            if gpu_state
                .grid
                .eval_rho_grad(&density_total.data, &mut rho_grad)
            {
                self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                let n_pts = gpu_state.weights.len();
                wv.fill(0.0);
                let workers = rayon::current_num_threads().max(1);
                let pt_chunk = n_pts.div_ceil(workers).max(1024);
                let e_xc: f64 = wv
                    .par_chunks_mut(pt_chunk * 4)
                    .zip(rho_grad.par_chunks(pt_chunk * 4))
                    .zip(gpu_state.weights.par_chunks(pt_chunk))
                    .map(|((wv_ch, rg_ch), w_ch)| {
                        let mut e_loc = 0.0f64;
                        let mut e_comp = 0.0f64;
                        for g in 0..w_ch.len() {
                            let o = g * 4;
                            let rt = rg_ch[o];
                            if rt < DENSITY_SCREEN {
                                continue;
                            }
                            let w = w_ch[g];
                            let grad_x = rg_ch[o + 1];
                            let grad_y = rg_ch[o + 2];
                            let grad_z = rg_ch[o + 3];
                            let rho_s = 0.5 * rt;
                            let g_ss =
                                0.25 * (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
                            let out = eval_xc_spin(func, rho_s, rho_s, g_ss, g_ss, g_ss);

                            let term = w * out.f_xc - e_comp;
                            let next = e_loc + term;
                            e_comp = (next - e_loc) - term;
                            e_loc = next;

                            wv_ch[o] = 0.5 * w * out.v_rho_a;
                            let wu_scale = w * (out.v_gamma_aa + 0.5 * out.v_gamma_ab);
                            wv_ch[o + 1] = wu_scale * grad_x;
                            wv_ch[o + 2] = wu_scale * grad_y;
                            wv_ch[o + 3] = wu_scale * grad_z;
                        }
                        e_loc
                    })
                    .sum();

                let mut v_lower = vec![0.0f64; n * n];
                if gpu_state.grid.accumulate_vxc(&wv, &mut v_lower) {
                    self.gpu_kernel_invocations.fetch_add(1, Ordering::Relaxed);
                    let mut v_mat = Matrix2D {
                        rows: n,
                        cols: n,
                        data: v_lower,
                    };
                    for i in 0..n {
                        for j in 0..i {
                            v_mat.data[j * n + i] = v_mat.data[i * n + j];
                        }
                    }
                    return (e_xc, v_mat);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
            }
        } else if self.use_gpu && n > 0 && (!grid.points.is_empty() || grid.streaming.is_some()) {
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        let Some(ref batches) = self.batches else {
            return evaluate_vxc_rks(basis, grid, func, density_total);
        };
        let n = self.n_ao;
        if n == 0 || batches.is_empty() {
            return (0.0, Matrix2D::zeros(n, n));
        }

        let p_slice = &density_total.data;
        let workers = bounded_worker_count(n, 1);
        let chunk_len = batches.len().div_ceil(workers).max(1);

        let (e_xc, v_lower) = batches
            .par_chunks(chunk_len)
            .map(|batch_slice| {
                let mut e_loc = 0.0f64;
                let mut e_comp = 0.0f64;
                let mut v_loc = vec![0.0f64; n * n];
                let mut p_sub = Vec::<f64>::new();
                let mut t_buf = Vec::<f64>::new();
                let mut z_buf = Vec::<f64>::new();
                let mut v_sub = Vec::<f64>::new();
                let mut rho = [0.0f64; BATCH_SIZE];
                let mut gx = [0.0f64; BATCH_SIZE];
                let mut gy = [0.0f64; BATCH_SIZE];
                let mut gz = [0.0f64; BATCH_SIZE];
                let mut w_vrho = [0.0f64; BATCH_SIZE];
                let mut wu_x = [0.0f64; BATCH_SIZE];
                let mut wu_y = [0.0f64; BATCH_SIZE];
                let mut wu_z = [0.0f64; BATCH_SIZE];

                for batch in batch_slice {
                    let m = batch.ao_indices.len();
                    let b = batch.weights.len();
                    if m == 0 || b == 0 {
                        continue;
                    }
                    let mm = m * m;
                    let mb = m * b;
                    if p_sub.len() < mm {
                        p_sub.resize(mm, 0.0);
                        v_sub.resize(mm, 0.0);
                    }
                    if t_buf.len() < mb {
                        t_buf.resize(mb, 0.0);
                        z_buf.resize(mb, 0.0);
                    }

                    // 1. Gather contiguous sub-density P_sub (m x m)
                    for i in 0..m {
                        let mu = batch.ao_indices[i];
                        let row_p = &p_slice[mu * n..(mu + 1) * n];
                        let dst = &mut p_sub[i * m..(i + 1) * m];
                        for j in 0..m {
                            dst[j] = row_p[batch.ao_indices[j]];
                        }
                    }

                    // 2. BLAS-3 GEMM: T(m x b) = P_sub(m x m) * Phi(m x b)
                    calchem_tensor::gemm_nn_into(
                        m,
                        b,
                        m,
                        &p_sub[..mm],
                        &batch.phi[..mb],
                        &mut t_buf[..mb],
                    );

                    // 3. Evaluate rho and grad_rho (contiguous inner loop over g in 0..b -> AVX2!)
                    rho[..b].fill(0.0);
                    gx[..b].fill(0.0);
                    gy[..b].fill(0.0);
                    gz[..b].fill(0.0);
                    for i in 0..m {
                        let off = i * b;
                        let t_row = &t_buf[off..off + b];
                        let p_row = &batch.phi[off..off + b];
                        let dx_row = &batch.dphi_x[off..off + b];
                        let dy_row = &batch.dphi_y[off..off + b];
                        let dz_row = &batch.dphi_z[off..off + b];
                        for g in 0..b {
                            let tg = t_row[g];
                            rho[g] += p_row[g] * tg;
                            gx[g] += dx_row[g] * tg;
                            gy[g] += dy_row[g] * tg;
                            gz[g] += dz_row[g] * tg;
                        }
                    }

                    // 4. Evaluate XC functional at each grid point
                    let mut any_active = false;
                    for g in 0..b {
                        let rt = rho[g];
                        if rt < DENSITY_SCREEN {
                            w_vrho[g] = 0.0;
                            wu_x[g] = 0.0;
                            wu_y[g] = 0.0;
                            wu_z[g] = 0.0;
                            continue;
                        }
                        any_active = true;
                        let w = batch.weights[g];
                        let grad_x = 2.0 * gx[g];
                        let grad_y = 2.0 * gy[g];
                        let grad_z = 2.0 * gz[g];
                        let rho_s = 0.5 * rt;
                        let g_ss = 0.25 * (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
                        let out = eval_xc_spin(func, rho_s, rho_s, g_ss, g_ss, g_ss);

                        let term = w * out.f_xc - e_comp;
                        let next = e_loc + term;
                        e_comp = (next - e_loc) - term;
                        e_loc = next;

                        w_vrho[g] = 0.5 * w * out.v_rho_a;
                        let wu_scale = w * (out.v_gamma_aa + 0.5 * out.v_gamma_ab);
                        wu_x[g] = wu_scale * grad_x;
                        wu_y[g] = wu_scale * grad_y;
                        wu_z[g] = wu_scale * grad_z;
                    }
                    if !any_active {
                        continue;
                    }

                    // 5. Assemble Z(m x b) (contiguous inner loop over g in 0..b -> AVX2!)
                    for i in 0..m {
                        let off = i * b;
                        let z_row = &mut z_buf[off..off + b];
                        let p_row = &batch.phi[off..off + b];
                        let dx_row = &batch.dphi_x[off..off + b];
                        let dy_row = &batch.dphi_y[off..off + b];
                        let dz_row = &batch.dphi_z[off..off + b];
                        for g in 0..b {
                            z_row[g] = w_vrho[g] * p_row[g]
                                + wu_x[g] * dx_row[g]
                                + wu_y[g] * dy_row[g]
                                + wu_z[g] * dz_row[g];
                        }
                    }

                    // 6. BLAS-3 GEMM: V_sub(m x m) = Z(m x b) * Phi(m x b)^T
                    calchem_tensor::gemm_nt_into(
                        m,
                        m,
                        b,
                        &z_buf[..mb],
                        &batch.phi[..mb],
                        &mut v_sub[..mm],
                    );

                    // 7. Scatter symmetrized (V_sub + V_sub^T) into lower triangle of v_loc
                    for i in 0..m {
                        let mu = batch.ao_indices[i];
                        let row_out = &mut v_loc[mu * n..(mu + 1) * n];
                        let vi_off = i * m;
                        for j in 0..=i {
                            let nu = batch.ao_indices[j];
                            row_out[nu] += v_sub[vi_off + j] + v_sub[j * m + i];
                        }
                    }
                }
                (e_loc, v_loc)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|(ea, mut va), (eb, vb)| {
                for (dst, src) in va.iter_mut().zip(vb.iter()) {
                    *dst += *src;
                }
                (ea + eb, va)
            })
            .expect("nonempty batches");

        let mut v_mat = Matrix2D {
            rows: n,
            cols: n,
            data: v_lower,
        };
        for i in 0..n {
            for j in 0..i {
                v_mat.data[j * n + i] = v_mat.data[i * n + j];
            }
        }
        (e_xc, v_mat)
    }
}

/// Evaluates RKS `(e_x_unscaled, e_c, e_xc_total, v_xc)` for a custom `PbeAbParams` variant.
#[doc(hidden)]
pub fn evaluate_vxc_rks_pbe_ab(
    basis: &BasisSet,
    grid: &MolecularGrid,
    params: crate::xc::PbeAbParams,
    x_scale: f64,
    density_total: &Matrix2D,
) -> (f64, f64, f64, Matrix2D) {
    let n = basis.n_ao;
    let shells = PrecomputedAoShell::from_basis(basis);
    let p_slice = &density_total.data;
    let tasks = grid.streaming_tasks();
    let (mol, radial_min) = grid
        .streaming
        .as_ref()
        .expect("evaluate_vxc_rks_pbe_ab requires streaming grid");

    let (e_x, e_c, e_xc, v_upper) = tasks
        .par_iter()
        .map(|&(atom_idx, rad_start, rad_end)| {
            let mut ex_loc = 0.0f64;
            let mut ec_loc = 0.0f64;
            let mut e_loc = 0.0f64;
            let mut e_comp = 0.0f64;
            let mut v_loc = vec![0.0f64; n * n];
            let mut ws = RksWorkspace::new(n, shells.len());
            let mut batch_pts: Vec<GridPoint> = Vec::with_capacity(2048);

            batch_pts.clear();
            MolecularGrid::generate_task_points(
                mol,
                grid.level,
                *radial_min,
                atom_idx,
                rad_start,
                rad_end,
                &mut batch_pts,
            );
            accumulate_rks_points_with(
                &batch_pts,
                n,
                &shells,
                p_slice,
                |rs, gss| {
                    let (ex, ec, out) = crate::xc::eval_pbe_ab(params, rs, rs, gss, gss, gss, x_scale);
                    ex_loc += out.f_xc * 0.0; // avoid unused warning
                    let _ = (ex, ec);
                    out
                },
                &mut e_loc,
                &mut e_comp,
                &mut v_loc,
                &mut ws,
            );
            // Separate X and C energies on the same batch_pts
            for gp in &batch_pts {
                let w = gp.weight;
                if w.abs() < 1e-24 {
                    continue;
                }
                eval_aos_grad(&shells, gp.coords, &mut ws.active);
                if ws.active.is_empty() {
                    continue;
                }
                let mut rho_t = 0.0;
                let mut grad_rho_t = [0.0f64; 3];
                for a_i in ws.active.iter() {
                    let row = &p_slice[a_i.mu * n..(a_i.mu + 1) * n];
                    let mut s = 0.0;
                    for a_j in ws.active.iter() {
                        s += row[a_j.mu] * a_j.val;
                    }
                    rho_t += a_i.val * s;
                    grad_rho_t[0] += 2.0 * a_i.grad[0] * s;
                    grad_rho_t[1] += 2.0 * a_i.grad[1] * s;
                    grad_rho_t[2] += 2.0 * a_i.grad[2] * s;
                }
                if rho_t < DENSITY_SCREEN {
                    continue;
                }
                let rs = 0.5 * rho_t;
                let gss = 0.25 * dot3(grad_rho_t, grad_rho_t);
                let (ex, ec, _) = crate::xc::eval_pbe_ab(params, rs, rs, gss, gss, gss, x_scale);
                ex_loc += w * ex;
                ec_loc += w * ec;
            }
            (ex_loc, ec_loc, e_loc, v_loc)
        })
        .reduce(
            || (0.0, 0.0, 0.0, vec![0.0f64; n * n]),
            |(xa, ca, ea, mut va), (xb, cb, eb, vb)| {
                for (d, s) in va.iter_mut().zip(vb.iter()) {
                    *d += *s;
                }
                (xa + xb, ca + cb, ea + eb, va)
            },
        );

    let mut v_mat = Matrix2D {rows:n,cols:n,data:v_upper};
    for i in 0..n {
        for j in 0..i { v_mat.data[j*n+i] = v_mat.data[i*n+j]; }
    }
    (e_x, e_c, e_xc, v_mat)
}

pub struct UksWorkspace<'a> {
    pub active: Vec<AoGradVal>,
    pub local_indices: Vec<usize>,
    pub vals: Vec<f64>,
    pub za_vec: Vec<f64>,
    pub zb_vec: Vec<f64>,
    pub candidate_shells: Vec<&'a PrecomputedAoShell>,
    pub ao_indices: Vec<usize>,
    pub ao_to_local: Vec<usize>,
    pub pa_sub: Vec<f64>,
    pub pb_sub: Vec<f64>,
    pub va_sub: Vec<f64>,
    pub vb_sub: Vec<f64>,
}

impl<'a> UksWorkspace<'a> {
    pub fn new(n: usize, n_shells: usize) -> Self {
        let init_m = n.min(256);
        Self {
            active: Vec::with_capacity(n),
            local_indices: Vec::with_capacity(n),
            vals: Vec::with_capacity(n),
            za_vec: Vec::with_capacity(n),
            zb_vec: Vec::with_capacity(n),
            candidate_shells: Vec::with_capacity(n_shells),
            ao_indices: Vec::with_capacity(n),
            ao_to_local: vec![usize::MAX; n],
            pa_sub: vec![0.0f64; init_m * init_m],
            pb_sub: vec![0.0f64; init_m * init_m],
            va_sub: vec![0.0f64; init_m * init_m],
            vb_sub: vec![0.0f64; init_m * init_m],
        }
    }

    #[inline]
    pub fn ensure_capacity(&mut self, m: usize) {
        let needed = m * m;
        if self.pa_sub.len() < needed {
            self.pa_sub.resize(needed, 0.0);
            self.pb_sub.resize(needed, 0.0);
            self.va_sub.resize(needed, 0.0);
            self.vb_sub.resize(needed, 0.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_uks_points<'a>(
    chunk: &[GridPoint],
    n: usize,
    shells: &'a [PrecomputedAoShell],
    pa_slice: &[f64],
    pb_slice: &[f64],
    func: XcFunctional,
    e_loc: &mut f64,
    e_comp: &mut f64,
    va_loc: &mut [f64],
    vb_loc: &mut [f64],
    ws: &mut UksWorkspace<'a>,
) {
    for sub_chunk in chunk.chunks(BATCH_SIZE) {
        let mut min_c = [f64::INFINITY; 3];
        let mut max_c = [-f64::INFINITY; 3];
        let mut has_points = false;

        for gp in sub_chunk {
            if gp.weight.abs() < 1e-24 {
                continue;
            }
            has_points = true;
            for q in 0..3 {
                if gp.coords[q] < min_c[q] { min_c[q] = gp.coords[q]; }
                if gp.coords[q] > max_c[q] { max_c[q] = gp.coords[q]; }
            }
        }

        if !has_points {
            continue;
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

        for sh in shells {
            let dx = sh.center[0] - center[0];
            let dy = sh.center[1] - center[1];
            let dz = sh.center[2] - center[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let thresh = r_box + sh.r_cutoff;
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

        ws.ensure_capacity(m);

        for (local_i, &mu) in ws.ao_indices.iter().enumerate() {
            ws.ao_to_local[mu] = local_i;
        }

        for i in 0..m {
            let mu = ws.ao_indices[i];
            let row_pa = &pa_slice[mu * n..(mu + 1) * n];
            let row_pb = &pb_slice[mu * n..(mu + 1) * n];
            let sub_pa_row = &mut ws.pa_sub[i * m..(i + 1) * m];
            let sub_pb_row = &mut ws.pb_sub[i * m..(i + 1) * m];
            for j in 0..m {
                let nu = ws.ao_indices[j];
                sub_pa_row[j] = row_pa[nu];
                sub_pb_row[j] = row_pb[nu];
            }
        }

        ws.va_sub[..m * m].fill(0.0);
        ws.vb_sub[..m * m].fill(0.0);

        for gp in sub_chunk {
            let w = gp.weight;
            if w.abs() < 1e-24 {
                continue;
            }

            eval_aos_grad_filtered(&ws.candidate_shells, gp.coords, &mut ws.active);
            if ws.active.is_empty() {
                continue;
            }

            ws.local_indices.clear();
            ws.vals.clear();
            for a in ws.active.iter() {
                ws.local_indices.push(ws.ao_to_local[a.mu]);
                ws.vals.push(a.val);
            }

            let mut rho_a = 0.0;
            let mut rho_b = 0.0;
            let mut grad_a = [0.0f64; 3];
            let mut grad_b = [0.0f64; 3];

            for (k_i, a_i) in ws.active.iter().enumerate() {
                let li = ws.local_indices[k_i];
                let sub_pa_row = &ws.pa_sub[li * m..(li + 1) * m];
                let sub_pb_row = &ws.pb_sub[li * m..(li + 1) * m];
                let mut sa = 0.0;
                let mut sb = 0.0;
                for (&lj, &v) in ws.local_indices.iter().zip(ws.vals.iter()) {
                    sa += sub_pa_row[lj] * v;
                    sb += sub_pb_row[lj] * v;
                }
                rho_a += a_i.val * sa;
                rho_b += a_i.val * sb;
                for q in 0..3 {
                    grad_a[q] += 2.0 * a_i.grad[q] * sa;
                    grad_b[q] += 2.0 * a_i.grad[q] * sb;
                }
            }

            let ra = rho_a.max(0.0);
            let rb = rho_b.max(0.0);
            if ra + rb < DENSITY_SCREEN {
                continue;
            }

            let gaa = dot3(grad_a, grad_a);
            let gab = dot3(grad_a, grad_b);
            let gbb = dot3(grad_b, grad_b);
            let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);

            let term = w * out.f_xc - *e_comp;
            let next = *e_loc + term;
            *e_comp = (next - *e_loc) - term;
            *e_loc = next;

            let ua = [
                2.0 * out.v_gamma_aa * grad_a[0] + out.v_gamma_ab * grad_b[0],
                2.0 * out.v_gamma_aa * grad_a[1] + out.v_gamma_ab * grad_b[1],
                2.0 * out.v_gamma_aa * grad_a[2] + out.v_gamma_ab * grad_b[2],
            ];
            let ub = [
                2.0 * out.v_gamma_bb * grad_b[0] + out.v_gamma_ab * grad_a[0],
                2.0 * out.v_gamma_bb * grad_b[1] + out.v_gamma_ab * grad_a[1],
                2.0 * out.v_gamma_bb * grad_b[2] + out.v_gamma_ab * grad_a[2],
            ];

            ws.za_vec.clear();
            ws.zb_vec.clear();
            let half_wva = 0.5 * w * out.v_rho_a;
            let half_wvb = 0.5 * w * out.v_rho_b;
            for a_i in ws.active.iter() {
                ws.za_vec.push(half_wva * a_i.val + w * dot3(ua, a_i.grad));
                ws.zb_vec.push(half_wvb * a_i.val + w * dot3(ub, a_i.grad));
            }

            let pt_m = ws.active.len();
            for i in 0..pt_m {
                let li = ws.local_indices[i];
                let phi_i = ws.vals[i];
                let za_i = ws.za_vec[i];
                let zb_i = ws.zb_vec[i];
                let sub_va_row = &mut ws.va_sub[li * m..(li + 1) * m];
                let sub_vb_row = &mut ws.vb_sub[li * m..(li + 1) * m];
                for j in 0..=i {
                    let lj = ws.local_indices[j];
                    let phi_j = ws.vals[j];
                    sub_va_row[lj] += za_i * phi_j + phi_i * ws.za_vec[j];
                    sub_vb_row[lj] += zb_i * phi_j + phi_i * ws.zb_vec[j];
                }
            }
        }

        // Scatter V_sub into thread-local va_loc and vb_loc
        for i in 0..m {
            let mu = ws.ao_indices[i];
            let sub_va_row = &ws.va_sub[i * m..(i + 1) * m];
            let sub_vb_row = &ws.vb_sub[i * m..(i + 1) * m];
            let mu_offset = mu * n;
            for j in 0..=i {
                let nu = ws.ao_indices[j];
                let va_val = sub_va_row[j];
                let vb_val = sub_vb_row[j];
                va_loc[mu_offset + nu] += va_val;
                vb_loc[mu_offset + nu] += vb_val;
                if i != j {
                    va_loc[nu * n + mu] += va_val;
                    vb_loc[nu * n + mu] += vb_val;
                }
            }
        }

        // Reset ao_to_local for next batch
        for &mu in &ws.ao_indices {
            ws.ao_to_local[mu] = usize::MAX;
        }
    }
}

/// Evaluates spin-polarized open-shell Kohn-Sham $(E_{\text{xc}}, V^{\text{xc},\alpha}, V^{\text{xc},\beta})$
/// from per-spin densities $P^\alpha, P^\beta$ (`n_ao x n_ao`). Supports both prebuilt and streaming grids.
pub fn evaluate_vxc_uks(
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> (f64, Matrix2D, Matrix2D) {
    let n = basis.n_ao;
    if n == 0 || (grid.points.is_empty() && grid.streaming.is_none()) {
        return (0.0, Matrix2D::zeros(n, n), Matrix2D::zeros(n, n));
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let pa_slice = &pa.data;
    let pb_slice = &pb.data;

    let (e_xc, va_data, vb_data) = if let Some((ref mol, radial_min)) =
        grid.streaming.as_ref().filter(|_| grid.points.is_empty())
    {
        let tasks = grid.streaming_tasks();
        if tasks.is_empty() {
            return (0.0, Matrix2D::zeros(n, n), Matrix2D::zeros(n, n));
        }
        let workers = bounded_worker_count(n, 2);
        let task_chunk_len = tasks.len().div_ceil(workers).max(1);
        tasks
            .par_chunks(task_chunk_len)
            .map(|task_slice| {
                let mut e_loc = 0.0f64;
                let mut e_comp = 0.0f64;
                let mut va_loc = vec![0.0f64; n * n];
                let mut vb_loc = vec![0.0f64; n * n];
                let mut ws = UksWorkspace::new(n, shells.len());
                let mut batch_pts: Vec<GridPoint> = Vec::with_capacity(2048);

                for &(atom_idx, rad_start, rad_end) in task_slice {
                    batch_pts.clear();
                    MolecularGrid::generate_task_points(
                        mol,
                        grid.level,
                        *radial_min,
                        atom_idx,
                        rad_start,
                        rad_end,
                        &mut batch_pts,
                    );
                    accumulate_uks_points(
                        &batch_pts,
                        n,
                        &shells,
                        pa_slice,
                        pb_slice,
                        func,
                        &mut e_loc,
                        &mut e_comp,
                        &mut va_loc,
                        &mut vb_loc,
                        &mut ws,
                    );
                }
                (e_loc, va_loc, vb_loc)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|(ea, mut va1, mut vb1), (eb, va2, vb2)| {
                for (d, s) in va1.iter_mut().zip(va2.iter()) {
                    *d += *s;
                }
                for (d, s) in vb1.iter_mut().zip(vb2.iter()) {
                    *d += *s;
                }
                (ea + eb, va1, vb1)
            })
            .expect("nonempty streaming tasks")
    } else {
        grid.points
            .par_chunks(bounded_grid_chunk_size(grid.points.len(), n, 2))
            .map(|chunk| {
                let mut e_loc = 0.0f64;
                let mut e_comp = 0.0f64;
                let mut va_loc = vec![0.0f64; n * n];
                let mut vb_loc = vec![0.0f64; n * n];
                let mut ws = UksWorkspace::new(n, shells.len());

                accumulate_uks_points(
                    chunk,
                    n,
                    &shells,
                    pa_slice,
                    pb_slice,
                    func,
                    &mut e_loc,
                    &mut e_comp,
                    &mut va_loc,
                    &mut vb_loc,
                    &mut ws,
                );
                (e_loc, va_loc, vb_loc)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .reduce(|(ea, mut va1, mut vb1), (eb, va2, vb2)| {
                for (d, s) in va1.iter_mut().zip(va2.iter()) {
                    *d += *s;
                }
                for (d, s) in vb1.iter_mut().zip(vb2.iter()) {
                    *d += *s;
                }
                (ea + eb, va1, vb1)
            })
            .expect("nonempty grid")
    };

    let mut va = Matrix2D::zeros(n, n);
    let mut vb = Matrix2D::zeros(n, n);
    va.data = va_data;
    vb.data = vb_data;
    (e_xc, va, vb)
}

/// GPU-accelerated open-shell UKS exchange-correlation integration engine with telemetry.
pub struct UksVxcEngine {
    n_ao: usize,
    use_gpu: bool,
    gpu: Option<GpuVxcState>,
    rg_b: Mutex<Vec<f64>>,
    wv_b: Mutex<Vec<f64>>,
    pub gpu_kernel_invocations: AtomicUsize,
    pub gpu_fallback_invocations: AtomicUsize,
}

impl UksVxcEngine {
    pub fn new(basis: &BasisSet, grid: &MolecularGrid) -> Self {
        Self::new_with_gpu(basis, grid, false)
    }

    pub fn telemetry(&self) -> (usize, usize) {
        (
            self.gpu_kernel_invocations.load(Ordering::Relaxed),
            self.gpu_fallback_invocations.load(Ordering::Relaxed),
        )
    }

    pub fn new_with_gpu(basis: &BasisSet, grid: &MolecularGrid, use_gpu: bool) -> Self {
        let n_ao = basis.n_ao;
        if !use_gpu || n_ao == 0 || grid.points.is_empty() || grid.streaming.is_some() {
            return Self {
                n_ao,
                use_gpu,
                gpu: None,
                rg_b: Mutex::new(Vec::new()),
                wv_b: Mutex::new(Vec::new()),
                gpu_kernel_invocations: AtomicUsize::new(0),
                gpu_fallback_invocations: AtomicUsize::new(0),
            };
        }
        if calchem_cuda::is_cuda_available() {
            let shells = PrecomputedAoShell::from_basis(basis);
            let raw_chunks: Vec<&[GridPoint]> = grid.points.chunks(BATCH_SIZE).collect();
            let batch_meta: Vec<(Vec<usize>, Vec<usize>)> = raw_chunks
                .par_iter()
                .map(|sub_chunk| {
                    let mut min_c = [f64::INFINITY; 3];
                    let mut max_c = [-f64::INFINITY; 3];
                    let mut pt_idxs = Vec::with_capacity(sub_chunk.len());
                    for (local_g, gp) in sub_chunk.iter().enumerate() {
                        if gp.weight.abs() < 1e-24 {
                            continue;
                        }
                        pt_idxs.push(local_g);
                        for q in 0..3 {
                            if gp.coords[q] < min_c[q] {
                                min_c[q] = gp.coords[q];
                            }
                            if gp.coords[q] > max_c[q] {
                                max_c[q] = gp.coords[q];
                            }
                        }
                    }
                    if pt_idxs.is_empty() {
                        return (Vec::new(), pt_idxs);
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
                    let mut cand_shell_idxs = Vec::new();
                    for (si, sh) in shells.iter().enumerate() {
                        let dx = sh.center[0] - center[0];
                        let dy = sh.center[1] - center[1];
                        let dz = sh.center[2] - center[2];
                        let d2 = dx * dx + dy * dy + dz * dz;
                        let thresh = r_box + sh.r_cutoff;
                        if d2 <= thresh * thresh {
                            cand_shell_idxs.push(si);
                        }
                    }
                    (cand_shell_idxs, pt_idxs)
                })
                .collect();

            let mut gpu_shells = Vec::with_capacity(shells.len());
            let mut gpu_prims = Vec::new();
            let mut gpu_comps = Vec::new();
            for sh in &shells {
                let prim_start = gpu_prims.len() as i32;
                for pr in &sh.prims {
                    gpu_prims.push(calchem_cuda::AoPrimGpu {
                        exponent: pr.exponent,
                        weighted_norm: pr.weighted_norm,
                    });
                }
                let comp_start = gpu_comps.len() as i32;
                for (ic, &c) in sh.comps.iter().enumerate() {
                    gpu_comps.push(calchem_cuda::AoCompGpu {
                        scale: sh.comp_scales[ic],
                        lx: c[0] as i32,
                        ly: c[1] as i32,
                        lz: c[2] as i32,
                        _pad: 0,
                    });
                }
                gpu_shells.push(calchem_cuda::AoShellGpu {
                    center: sh.center,
                    r2_cutoff: sh.r2_cutoff,
                    ao_offset: sh.ao_offset as i32,
                    n_funcs: sh.comps.len() as i32,
                    prim_start,
                    n_prims: sh.prims.len() as i32,
                    comp_start,
                    atom_index: sh.atom_index as i32,
                });
            }

            let mut gpu_pts = Vec::with_capacity(grid.points.len());
            let mut weights = Vec::with_capacity(grid.points.len());
            let mut cand_offsets = Vec::with_capacity(batch_meta.len() + 1);
            let mut cand_shells_flat = Vec::new();
            cand_offsets.push(0i32);
            let mut batch_id = 0i32;

            for (sub_chunk, (cand_sh, pt_idxs)) in raw_chunks.iter().zip(batch_meta.iter()) {
                if pt_idxs.is_empty() || cand_sh.is_empty() {
                    continue;
                }
                for &si in cand_sh {
                    cand_shells_flat.push(si as i32);
                }
                cand_offsets.push(cand_shells_flat.len() as i32);
                for &lg in pt_idxs {
                    let gp = &sub_chunk[lg];
                    gpu_pts.push(calchem_cuda::GridPtGpu {
                        x: gp.coords[0],
                        y: gp.coords[1],
                        z: gp.coords[2],
                        batch_id,
                        host_atom: gp.host_atom as i32,
                    });
                    weights.push(gp.weight);
                }
                batch_id += 1;
            }

            if !gpu_pts.is_empty() && !cand_shells_flat.is_empty() {
                if let Some(gpu_grid) = calchem_cuda::GpuVxcGrid::upload(
                    &gpu_shells,
                    &gpu_prims,
                    &gpu_comps,
                    &gpu_pts,
                    &cand_offsets,
                    &cand_shells_flat,
                    n_ao,
                ) {
                    let n_pts = weights.len();
                    return Self {
                        n_ao,
                        use_gpu,
                        gpu: Some(GpuVxcState {
                            grid: gpu_grid,
                            weights,
                            rho_grad: Mutex::new(vec![0.0f64; n_pts * 4]),
                            wv: Mutex::new(vec![0.0f64; n_pts * 4]),
                        }),
                        rg_b: Mutex::new(vec![0.0f64; n_pts * 4]),
                        wv_b: Mutex::new(vec![0.0f64; n_pts * 4]),
                        gpu_kernel_invocations: AtomicUsize::new(0),
                        gpu_fallback_invocations: AtomicUsize::new(0),
                    };
                }
            }
        }

        Self {
            n_ao,
            use_gpu,
            gpu: None,
            rg_b: Mutex::new(Vec::new()),
            wv_b: Mutex::new(Vec::new()),
            gpu_kernel_invocations: AtomicUsize::new(0),
            gpu_fallback_invocations: AtomicUsize::new(0),
        }
    }

    pub fn evaluate(
        &self,
        basis: &BasisSet,
        grid: &MolecularGrid,
        func: XcFunctional,
        pa: &Matrix2D,
        pb: &Matrix2D,
    ) -> (f64, Matrix2D, Matrix2D) {
        let n = self.n_ao;
        if let Some(ref gpu_state) = self.gpu {
            let mut rg_a = gpu_state.rho_grad.lock().unwrap();
            let mut rg_b = self.rg_b.lock().unwrap();
            let mut wv_a = gpu_state.wv.lock().unwrap();
            let mut wv_b = self.wv_b.lock().unwrap();
            if gpu_state.grid.eval_rho_grad(&pa.data, &mut rg_a)
                && gpu_state.grid.eval_rho_grad(&pb.data, &mut rg_b)
            {
                self.gpu_kernel_invocations.fetch_add(2, Ordering::Relaxed);
                let n_pts = gpu_state.weights.len();
                wv_a.fill(0.0);
                wv_b.fill(0.0);
                let workers = rayon::current_num_threads().max(1);
                let pt_chunk = n_pts.div_ceil(workers).max(1024);
                let e_xc: f64 = wv_a
                    .par_chunks_mut(pt_chunk * 4)
                    .zip(wv_b.par_chunks_mut(pt_chunk * 4))
                    .zip(rg_a.par_chunks(pt_chunk * 4))
                    .zip(rg_b.par_chunks(pt_chunk * 4))
                    .zip(gpu_state.weights.par_chunks(pt_chunk))
                    .map(|((((wa_ch, wb_ch), ra_ch), rb_ch), w_ch)| {
                        let mut e_loc = 0.0f64;
                        let mut e_comp = 0.0f64;
                        for g in 0..w_ch.len() {
                            let o = g * 4;
                            let ra = ra_ch[o].max(0.0);
                            let rb = rb_ch[o].max(0.0);
                            if ra + rb < DENSITY_SCREEN {
                                continue;
                            }
                            let w = w_ch[g];
                            let gax = ra_ch[o + 1];
                            let gay = ra_ch[o + 2];
                            let gaz = ra_ch[o + 3];
                            let gbx = rb_ch[o + 1];
                            let gby = rb_ch[o + 2];
                            let gbz = rb_ch[o + 3];
                            let gaa = gax * gax + gay * gay + gaz * gaz;
                            let gab = gax * gbx + gay * gby + gaz * gbz;
                            let gbb = gbx * gbx + gby * gby + gbz * gbz;
                            let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);

                            let term = w * out.f_xc - e_comp;
                            let next = e_loc + term;
                            e_comp = (next - e_loc) - term;
                            e_loc = next;

                            wa_ch[o] = 0.5 * w * out.v_rho_a;
                            wb_ch[o] = 0.5 * w * out.v_rho_b;
                            wa_ch[o + 1] = w * (2.0 * out.v_gamma_aa * gax + out.v_gamma_ab * gbx);
                            wa_ch[o + 2] = w * (2.0 * out.v_gamma_aa * gay + out.v_gamma_ab * gby);
                            wa_ch[o + 3] = w * (2.0 * out.v_gamma_aa * gaz + out.v_gamma_ab * gbz);
                            wb_ch[o + 1] = w * (2.0 * out.v_gamma_bb * gbx + out.v_gamma_ab * gax);
                            wb_ch[o + 2] = w * (2.0 * out.v_gamma_bb * gby + out.v_gamma_ab * gay);
                            wb_ch[o + 3] = w * (2.0 * out.v_gamma_bb * gbz + out.v_gamma_ab * gaz);
                        }
                        e_loc
                    })
                    .sum();

                let mut va_lower = vec![0.0f64; n * n];
                let mut vb_lower = vec![0.0f64; n * n];
                if gpu_state.grid.accumulate_vxc(&wv_a, &mut va_lower)
                    && gpu_state.grid.accumulate_vxc(&wv_b, &mut vb_lower)
                {
                    self.gpu_kernel_invocations.fetch_add(2, Ordering::Relaxed);
                    let mut va = Matrix2D {
                        rows: n,
                        cols: n,
                        data: va_lower,
                    };
                    let mut vb = Matrix2D {
                        rows: n,
                        cols: n,
                        data: vb_lower,
                    };
                    for i in 0..n {
                        for j in 0..i {
                            va.data[j * n + i] = va.data[i * n + j];
                            vb.data[j * n + i] = vb.data[i * n + j];
                        }
                    }
                    return (e_xc, va, vb);
                } else {
                    self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
            }
        } else if self.use_gpu && n > 0 && (!grid.points.is_empty() || grid.streaming.is_some()) {
            self.gpu_fallback_invocations.fetch_add(1, Ordering::Relaxed);
        }

        evaluate_vxc_uks(basis, grid, func, pa, pb)
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_xc_grad_points(
    chunk: &[GridPoint],
    mol: &Molecule,
    n: usize,
    shells: &[PrecomputedAoShell],
    pa_slice: &[f64],
    pb_slice: &[f64],
    func: XcFunctional,
    local_grad: &mut [[f64; 3]],
    active: &mut Vec<AoHessVal>,
    phi_pa: &mut Vec<f64>,
    phi_pb: &mut Vec<f64>,
    gpa: &mut Vec<[f64; 3]>,
    gpb: &mut Vec<[f64; 3]>,
) {
    for gp in chunk {
        if gp.raw_weight.abs() < 1e-24 {
            continue;
        }
        eval_aos_hess(shells, gp.coords, active);
        if active.is_empty() {
            continue;
        }

        phi_pa.clear();
        phi_pb.clear();
        gpa.clear();
        gpb.clear();

        let mut rho_a = 0.0;
        let mut rho_b = 0.0;
        let mut grad_a = [0.0f64; 3];
        let mut grad_b = [0.0f64; 3];

        for a_i in active.iter() {
            let row_a = &pa_slice[a_i.mu * n..(a_i.mu + 1) * n];
            let row_b = &pb_slice[a_i.mu * n..(a_i.mu + 1) * n];
            let mut sa = 0.0;
            let mut sb = 0.0;
            let mut ga_vec = [0.0f64; 3];
            let mut gb_vec = [0.0f64; 3];

            for a_j in active.iter() {
                let p_a = row_a[a_j.mu];
                let p_b = row_b[a_j.mu];
                sa += p_a * a_j.val;
                sb += p_b * a_j.val;
                for q in 0..3 {
                    ga_vec[q] += p_a * a_j.grad[q];
                    gb_vec[q] += p_b * a_j.grad[q];
                }
            }

            phi_pa.push(sa);
            phi_pb.push(sb);
            gpa.push(ga_vec);
            gpb.push(gb_vec);

            rho_a += a_i.val * sa;
            rho_b += a_i.val * sb;
            for q in 0..3 {
                grad_a[q] += 2.0 * a_i.grad[q] * sa;
                grad_b[q] += 2.0 * a_i.grad[q] * sb;
            }
        }

        let ra = rho_a.max(0.0);
        let rb = rho_b.max(0.0);
        if ra + rb < DENSITY_SCREEN {
            continue;
        }

        let gaa = dot3(grad_a, grad_a);
        let gab = dot3(grad_a, grad_b);
        let gbb = dot3(grad_b, grad_b);
        let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);

        let w = gp.weight;
        if w.abs() > 1e-24 {
            let ua = [
                2.0 * out.v_gamma_aa * grad_a[0] + out.v_gamma_ab * grad_b[0],
                2.0 * out.v_gamma_aa * grad_a[1] + out.v_gamma_ab * grad_b[1],
                2.0 * out.v_gamma_aa * grad_a[2] + out.v_gamma_ab * grad_b[2],
            ];
            let ub = [
                2.0 * out.v_gamma_bb * grad_b[0] + out.v_gamma_ab * grad_a[0],
                2.0 * out.v_gamma_bb * grad_b[1] + out.v_gamma_ab * grad_a[1],
                2.0 * out.v_gamma_bb * grad_b[2] + out.v_gamma_ab * grad_a[2],
            ];

            let mut sum_ao_force = [0.0f64; 3];

            for (idx, a_i) in active.iter().enumerate() {
                let atom = a_i.atom_index;
                let sa = phi_pa[idx];
                let sb = phi_pb[idx];
                let ua_dot_gpa = dot3(ua, gpa[idx]);
                let ub_dot_gpb = dot3(ub, gpb[idx]);

                for q in 0..3 {
                    let dphi_dq = a_i.grad[q];
                    let hrow = a_i.hess_row(q);

                    let term_a = out.v_rho_a * dphi_dq * sa
                        + dot3(ua, hrow) * sa
                        + dphi_dq * ua_dot_gpa;
                    let term_b = out.v_rho_b * dphi_dq * sb
                        + dot3(ub, hrow) * sb
                        + dphi_dq * ub_dot_gpb;

                    let f_aq = -2.0 * w * (term_a + term_b);
                    local_grad[atom][q] += f_aq;
                    sum_ao_force[q] += f_aq;
                }
            }

            for q in 0..3 {
                local_grad[gp.host_atom][q] -= sum_ao_force[q];
            }
        }

        becke_weight_nuclear_gradient(
            mol,
            gp.host_atom,
            gp.coords,
            gp.raw_weight,
            local_grad,
            out.f_xc,
        );
    }
}

/// Evaluates the exact analytical nuclear gradient $\left.\frac{\partial E_{\text{xc}}}{\partial \mathbf{R}_A}\right|_{P^\alpha, P^\beta}$
/// in Eh/Bohr for both closed-shell (`pa = pb = 0.5 * P_total`) and open-shell (`pa, pb`) Kohn-Sham DFT.
pub fn xc_nuclear_gradient(
    mol: &Molecule,
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> Vec<[f64; 3]> {
    xc_nuclear_gradient_with_gpu(mol, basis, grid, func, pa, pb, false).0
}

fn xc_nuclear_gradient_cpu(
    mol: &Molecule,
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
) -> Vec<[f64; 3]> {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;
    if n_atoms <= 1 || n == 0 || (grid.points.is_empty() && grid.streaming.is_none()) {
        return vec![[0.0; 3]; n_atoms];
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let pa_slice = &pa.data;
    let pb_slice = &pb.data;

    if let Some((_, radial_min)) = grid.streaming {
        if grid.points.is_empty() {
            let tasks = grid.streaming_tasks();
            return tasks
                .into_par_iter()
                .fold(
                    || vec![[0.0f64; 3]; n_atoms],
                    |mut local_grad, (atom_idx, rad_start, rad_end)| {
                        let mut active: Vec<AoHessVal> = Vec::with_capacity(n);
                        let mut phi_pa: Vec<f64> = Vec::with_capacity(n);
                        let mut phi_pb: Vec<f64> = Vec::with_capacity(n);
                        let mut gpa: Vec<[f64; 3]> = Vec::with_capacity(n);
                        let mut gpb: Vec<[f64; 3]> = Vec::with_capacity(n);
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
                        accumulate_xc_grad_points(
                            &batch_pts,
                            mol,
                            n,
                            &shells,
                            pa_slice,
                            pb_slice,
                            func,
                            &mut local_grad,
                            &mut active,
                            &mut phi_pa,
                            &mut phi_pb,
                            &mut gpa,
                            &mut gpb,
                        );
                        local_grad
                    },
                )
                .reduce(
                    || vec![[0.0f64; 3]; n_atoms],
                    |mut ga, gb| {
                        for i in 0..n_atoms {
                            for q in 0..3 {
                                ga[i][q] += gb[i][q];
                            }
                        }
                        ga
                    },
                );
        }
    }

    grid.points
        .par_chunks(GRID_CHUNK_SIZE)
        .map(|chunk| {
            let mut local_grad = vec![[0.0f64; 3]; n_atoms];
            let mut active: Vec<AoHessVal> = Vec::with_capacity(n);
            let mut phi_pa: Vec<f64> = Vec::with_capacity(n);
            let mut phi_pb: Vec<f64> = Vec::with_capacity(n);
            let mut gpa: Vec<[f64; 3]> = Vec::with_capacity(n);
            let mut gpb: Vec<[f64; 3]> = Vec::with_capacity(n);

            accumulate_xc_grad_points(
                chunk,
                mol,
                n,
                &shells,
                pa_slice,
                pb_slice,
                func,
                &mut local_grad,
                &mut active,
                &mut phi_pa,
                &mut phi_pb,
                &mut gpa,
                &mut gpb,
            );
            local_grad
        })
        .reduce(
            || vec![[0.0f64; 3]; n_atoms],
            |mut ga, gb| {
                for i in 0..n_atoms {
                    for q in 0..3 {
                        ga[i][q] += gb[i][q];
                    }
                }
                ga
            },
        )
}

/// Evaluates the exact analytical XC nuclear gradient with optional CUDA GPU acceleration,
/// returning `(grad, gpu_kernel_invocations, gpu_fallback_invocations)`.
pub fn xc_nuclear_gradient_with_gpu(
    mol: &Molecule,
    basis: &BasisSet,
    grid: &MolecularGrid,
    func: XcFunctional,
    pa: &Matrix2D,
    pb: &Matrix2D,
    use_gpu: bool,
) -> (Vec<[f64; 3]>, usize, usize) {
    let n_atoms = mol.atoms.len();
    let n = basis.n_ao;
    if n_atoms <= 1 || n == 0 || (grid.points.is_empty() && grid.streaming.is_none()) {
        return (vec![[0.0; 3]; n_atoms], 0, 0);
    }
    if !use_gpu {
        return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 0);
    }
    if grid.streaming.is_some() || grid.points.is_empty() || !calchem_cuda::is_cuda_available() {
        return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 1);
    }

    let shells = PrecomputedAoShell::from_basis(basis);
    let raw_chunks: Vec<&[GridPoint]> = grid.points.chunks(BATCH_SIZE).collect();
    let batch_meta: Vec<(Vec<usize>, Vec<usize>)> = raw_chunks
        .par_iter()
        .map(|sub_chunk| {
            let mut min_c = [f64::INFINITY; 3];
            let mut max_c = [-f64::INFINITY; 3];
            let mut pt_idxs = Vec::with_capacity(sub_chunk.len());
            for (local_g, gp) in sub_chunk.iter().enumerate() {
                if gp.raw_weight.abs() < 1e-24 {
                    continue;
                }
                pt_idxs.push(local_g);
                for q in 0..3 {
                    if gp.coords[q] < min_c[q] {
                        min_c[q] = gp.coords[q];
                    }
                    if gp.coords[q] > max_c[q] {
                        max_c[q] = gp.coords[q];
                    }
                }
            }
            if pt_idxs.is_empty() {
                return (Vec::new(), pt_idxs);
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
            let mut cand_shell_idxs = Vec::new();
            for (si, sh) in shells.iter().enumerate() {
                let dx = sh.center[0] - center[0];
                let dy = sh.center[1] - center[1];
                let dz = sh.center[2] - center[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                let thresh = r_box + sh.r_cutoff;
                if d2 <= thresh * thresh {
                    cand_shell_idxs.push(si);
                }
            }
            (cand_shell_idxs, pt_idxs)
        })
        .collect();

    let mut gpu_shells = Vec::with_capacity(shells.len());
    let mut gpu_prims = Vec::new();
    let mut gpu_comps = Vec::new();
    for sh in &shells {
        let prim_start = gpu_prims.len() as i32;
        for pr in &sh.prims {
            gpu_prims.push(calchem_cuda::AoPrimGpu {
                exponent: pr.exponent,
                weighted_norm: pr.weighted_norm,
            });
        }
        let comp_start = gpu_comps.len() as i32;
        for (ic, &c) in sh.comps.iter().enumerate() {
            gpu_comps.push(calchem_cuda::AoCompGpu {
                scale: sh.comp_scales[ic],
                lx: c[0] as i32,
                ly: c[1] as i32,
                lz: c[2] as i32,
                _pad: 0,
            });
        }
        gpu_shells.push(calchem_cuda::AoShellGpu {
            center: sh.center,
            r2_cutoff: sh.r2_cutoff,
            ao_offset: sh.ao_offset as i32,
            n_funcs: sh.comps.len() as i32,
            prim_start,
            n_prims: sh.prims.len() as i32,
            comp_start,
            atom_index: sh.atom_index as i32,
        });
    }

    let mut gpu_pts = Vec::with_capacity(grid.points.len());
    let mut active_pts: Vec<&GridPoint> = Vec::with_capacity(grid.points.len());
    let mut cand_offsets = Vec::with_capacity(batch_meta.len() + 1);
    let mut cand_shells_flat = Vec::new();
    cand_offsets.push(0i32);
    let mut batch_id = 0i32;

    for (sub_chunk, (cand_sh, pt_idxs)) in raw_chunks.iter().zip(batch_meta.iter()) {
        if pt_idxs.is_empty() || cand_sh.is_empty() {
            continue;
        }
        for &si in cand_sh {
            cand_shells_flat.push(si as i32);
        }
        cand_offsets.push(cand_shells_flat.len() as i32);
        for &lg in pt_idxs {
            let gp = &sub_chunk[lg];
            gpu_pts.push(calchem_cuda::GridPtGpu {
                x: gp.coords[0],
                y: gp.coords[1],
                z: gp.coords[2],
                batch_id,
                host_atom: gp.host_atom as i32,
            });
            active_pts.push(gp);
        }
        batch_id += 1;
    }

    if gpu_pts.is_empty() || cand_shells_flat.is_empty() {
        return (vec![[0.0; 3]; n_atoms], 0, 0);
    }

    let Some(gpu_grid) = calchem_cuda::GpuVxcGrid::upload(
        &gpu_shells,
        &gpu_prims,
        &gpu_comps,
        &gpu_pts,
        &cand_offsets,
        &cand_shells_flat,
        n,
    ) else {
        return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 1);
    };

    let n_pts = active_pts.len();
    let is_rks = pa.data == pb.data;
    let mut rg_a = vec![0.0f64; n_pts * 4];
    if !gpu_grid.eval_rho_grad(&pa.data, &mut rg_a) {
        return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 1);
    }
    let mut gpu_k = 1usize;
    let rg_b_vec = if is_rks {
        Vec::new()
    } else {
        let mut rg_b = vec![0.0f64; n_pts * 4];
        if !gpu_grid.eval_rho_grad(&pb.data, &mut rg_b) {
            return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 1);
        }
        gpu_k += 1;
        rg_b
    };

    let mut wxc = vec![0.0f64; n_pts * 8];
    let workers = rayon::current_num_threads().max(1);
    let pt_chunk = n_pts.div_ceil(workers).max(1024);

    let grad_becke = if is_rks {
        wxc.par_chunks_mut(pt_chunk * 8)
            .zip(rg_a.par_chunks(pt_chunk * 4))
            .zip(active_pts.par_chunks(pt_chunk))
            .map(|((wxc_ch, ra_ch), pts_ch)| {
                let mut local_grad = vec![[0.0f64; 3]; n_atoms];
                for g in 0..pts_ch.len() {
                    let o = g * 4;
                    let ra = ra_ch[o].max(0.0);
                    if 2.0 * ra < DENSITY_SCREEN {
                        continue;
                    }
                    let ga = [ra_ch[o + 1], ra_ch[o + 2], ra_ch[o + 3]];
                    let gaa = dot3(ga, ga);
                    let out = eval_xc_spin(func, ra, ra, gaa, gaa, gaa);
                    let gp = pts_ch[g];
                    let w = gp.weight;
                    if w.abs() > 1e-24 {
                        let ua0 = 2.0 * out.v_gamma_aa * ga[0] + out.v_gamma_ab * ga[0];
                        let ua1 = 2.0 * out.v_gamma_aa * ga[1] + out.v_gamma_ab * ga[1];
                        let ua2 = 2.0 * out.v_gamma_aa * ga[2] + out.v_gamma_ab * ga[2];
                        let ub0 = 2.0 * out.v_gamma_bb * ga[0] + out.v_gamma_ab * ga[0];
                        let ub1 = 2.0 * out.v_gamma_bb * ga[1] + out.v_gamma_ab * ga[1];
                        let ub2 = 2.0 * out.v_gamma_bb * ga[2] + out.v_gamma_ab * ga[2];
                        let wo = g * 8;
                        wxc_ch[wo] = w * out.v_rho_a;
                        wxc_ch[wo + 1] = w * ua0;
                        wxc_ch[wo + 2] = w * ua1;
                        wxc_ch[wo + 3] = w * ua2;
                        wxc_ch[wo + 4] = w * out.v_rho_b;
                        wxc_ch[wo + 5] = w * ub0;
                        wxc_ch[wo + 6] = w * ub1;
                        wxc_ch[wo + 7] = w * ub2;
                    }
                    becke_weight_nuclear_gradient(
                        mol,
                        gp.host_atom,
                        gp.coords,
                        gp.raw_weight,
                        &mut local_grad,
                        out.f_xc,
                    );
                }
                local_grad
            })
            .reduce(
                || vec![[0.0f64; 3]; n_atoms],
                |mut ga, gb| {
                    for i in 0..n_atoms {
                        for q in 0..3 {
                            ga[i][q] += gb[i][q];
                        }
                    }
                    ga
                },
            )
    } else {
        wxc.par_chunks_mut(pt_chunk * 8)
            .zip(rg_a.par_chunks(pt_chunk * 4))
            .zip(rg_b_vec.par_chunks(pt_chunk * 4))
            .zip(active_pts.par_chunks(pt_chunk))
            .map(|(((wxc_ch, ra_ch), rb_ch), pts_ch)| {
                let mut local_grad = vec![[0.0f64; 3]; n_atoms];
                for g in 0..pts_ch.len() {
                    let o = g * 4;
                    let ra = ra_ch[o].max(0.0);
                    let rb = rb_ch[o].max(0.0);
                    if ra + rb < DENSITY_SCREEN {
                        continue;
                    }
                    let ga = [ra_ch[o + 1], ra_ch[o + 2], ra_ch[o + 3]];
                    let gb = [rb_ch[o + 1], rb_ch[o + 2], rb_ch[o + 3]];
                    let gaa = dot3(ga, ga);
                    let gab = dot3(ga, gb);
                    let gbb = dot3(gb, gb);
                    let out = eval_xc_spin(func, ra, rb, gaa, gab, gbb);
                    let gp = pts_ch[g];
                    let w = gp.weight;
                    if w.abs() > 1e-24 {
                        let ua0 = 2.0 * out.v_gamma_aa * ga[0] + out.v_gamma_ab * gb[0];
                        let ua1 = 2.0 * out.v_gamma_aa * ga[1] + out.v_gamma_ab * gb[1];
                        let ua2 = 2.0 * out.v_gamma_aa * ga[2] + out.v_gamma_ab * gb[2];
                        let ub0 = 2.0 * out.v_gamma_bb * gb[0] + out.v_gamma_ab * ga[0];
                        let ub1 = 2.0 * out.v_gamma_bb * gb[1] + out.v_gamma_ab * ga[1];
                        let ub2 = 2.0 * out.v_gamma_bb * gb[2] + out.v_gamma_ab * ga[2];
                        let wo = g * 8;
                        wxc_ch[wo] = w * out.v_rho_a;
                        wxc_ch[wo + 1] = w * ua0;
                        wxc_ch[wo + 2] = w * ua1;
                        wxc_ch[wo + 3] = w * ua2;
                        wxc_ch[wo + 4] = w * out.v_rho_b;
                        wxc_ch[wo + 5] = w * ub0;
                        wxc_ch[wo + 6] = w * ub1;
                        wxc_ch[wo + 7] = w * ub2;
                    }
                    becke_weight_nuclear_gradient(
                        mol,
                        gp.host_atom,
                        gp.coords,
                        gp.raw_weight,
                        &mut local_grad,
                        out.f_xc,
                    );
                }
                local_grad
            })
            .reduce(
                || vec![[0.0f64; 3]; n_atoms],
                |mut ga, gb| {
                    for i in 0..n_atoms {
                        for q in 0..3 {
                            ga[i][q] += gb[i][q];
                        }
                    }
                    ga
                },
            )
    };

    let mut grad_ao_flat = vec![0.0f64; n_atoms * 3];
    if !gpu_grid.eval_grad_force(&pa.data, &pb.data, is_rks, &wxc, &mut grad_ao_flat) {
        return (xc_nuclear_gradient_cpu(mol, basis, grid, func, pa, pb), 0, 1);
    }
    gpu_k += 1;

    let mut grad = grad_becke;
    for a in 0..n_atoms {
        for q in 0..3 {
            grad[a][q] += grad_ao_flat[a * 3 + q];
        }
    }
    (grad, gpu_k, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridLevel;
    use calchem_basis::BasisSet;
    use calchem_model::{Atom, Molecule};

    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn distorted_water(atoms_override: Option<Vec<[f64; 3]>>) -> Molecule {
        let coords = atoms_override.unwrap_or_else(|| {
            vec![
                [0.02, -0.04, 0.11],
                [0.81, 0.59, -0.37],
                [-0.74, 0.63, -0.42],
            ]
        });
        Molecule::new(
            vec![
                Atom::from_angstrom("O", coords[0]).unwrap(),
                Atom::from_angstrom("H", coords[1]).unwrap(),
                Atom::from_angstrom("H", coords[2]).unwrap(),
            ],
            0,
            1,
        )
    }

    fn make_positive_density(n: usize, seed_shift: usize) -> Matrix2D {
        // Construct strictly positive-definite P = A * A^T + 0.05 * I so density > 0 everywhere
        // even under off-diagonal finite-difference perturbations.
        let mut a = Matrix2D::zeros(n, 4);
        for mu in 0..n {
            for k in 0..4 {
                let x = ((mu * 7 + k * 13 + seed_shift * 5 + 3) as f64 * 0.37).sin();
                a.set(mu, k, 0.22 * x);
            }
        }
        let mut p = Matrix2D::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut s = if mu == nu { 0.05 } else { 0.0 };
                for k in 0..4 {
                    s += a.get(mu, k) * a.get(nu, k);
                }
                p.set(mu, nu, s);
            }
        }
        p
    }

    #[test]
    fn test_vxc_rks_and_uks_matrix_matches_exact_density_derivative() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).expect("6-31g(d)");
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = basis.n_ao;

        let p_tot = make_positive_density(n, 1);
        let eps = 1e-6;

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
            let (_exc, vxc) = evaluate_vxc_rks(&basis, &grid, func, &p_tot);

            // Check symmetry and exact partial derivative wrt P^t on representative (mu, nu) pairs
            for &(mu, nu) in &[(0, 0), (0, 3), (2, 7), (5, 11), (8, 14), (12, 12)] {
                assert!(
                    (vxc.get(mu, nu) - vxc.get(nu, mu)).abs() < 1e-14,
                    "V_xc not symmetric at ({mu},{nu})"
                );
                let mut p_p = p_tot.clone();
                let mut p_m = p_tot.clone();
                if mu == nu {
                    p_p.set(mu, mu, p_tot.get(mu, mu) + eps);
                    p_m.set(mu, mu, p_tot.get(mu, mu) - eps);
                    let (ep, _) = evaluate_vxc_rks(&basis, &grid, func, &p_p);
                    let (em, _) = evaluate_vxc_rks(&basis, &grid, func, &p_m);
                    let fd = (ep - em) / (2.0 * eps);
                    assert!(
                        (vxc.get(mu, mu) - fd).abs() < 1e-9,
                        "{func:?} RKS diag ({mu},{mu}): ana={} fd={fd}",
                        vxc.get(mu, mu)
                    );
                } else {
                    p_p.set(mu, nu, p_tot.get(mu, nu) + eps);
                    p_p.set(nu, mu, p_tot.get(nu, mu) + eps);
                    p_m.set(mu, nu, p_tot.get(mu, nu) - eps);
                    p_m.set(nu, mu, p_tot.get(nu, mu) - eps);
                    let (ep, _) = evaluate_vxc_rks(&basis, &grid, func, &p_p);
                    let (em, _) = evaluate_vxc_rks(&basis, &grid, func, &p_m);
                    // Perturbing both (mu,nu) and (nu,mu) by +eps changes E_xc by 2 * V_xc[mu,nu] * eps!
                    let fd = (ep - em) / (4.0 * eps);
                    assert!(
                        (vxc.get(mu, nu) - fd).abs() < 1e-9,
                        "{func:?} RKS off-diag ({mu},{nu}): ana={} fd={fd}",
                        vxc.get(mu, nu)
                    );
                }
            }

            // Also verify closed-shell UKS (pa = pb = 0.5 * p_tot) matches RKS identically!
            let mut half_p = p_tot.clone();
            for v in &mut half_p.data {
                *v *= 0.5;
            }
            let (exc_u, va, vb) = evaluate_vxc_uks(&basis, &grid, func, &half_p, &half_p);
            let (exc_r, vr) = evaluate_vxc_rks(&basis, &grid, func, &p_tot);
            assert!((exc_u - exc_r).abs() < 1e-13);
            for idx in 0..n * n {
                assert!((va.data[idx] - vr.data[idx]).abs() < 1e-13);
                assert!((vb.data[idx] - vr.data[idx]).abs() < 1e-13);
            }
        }

        // Open-shell UKS asymmetric spin check (`pa != pb`)
        let pa = make_positive_density(n, 2);
        let pb = make_positive_density(n, 5);
        for func in [XcFunctional::Blyp, XcFunctional::B3lyp, XcFunctional::Pbe0] {
            let (_, va, vb) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb);
            for &(mu, nu) in &[(1, 1), (1, 6), (4, 10), (9, 15)] {
                let mut pa_p = pa.clone();
                let mut pa_m = pa.clone();
                let mut pb_p = pb.clone();
                let mut pb_m = pb.clone();
                let denom = if mu == nu {
                    pa_p.set(mu, mu, pa.get(mu, mu) + eps);
                    pa_m.set(mu, mu, pa.get(mu, mu) - eps);
                    pb_p.set(mu, mu, pb.get(mu, mu) + eps);
                    pb_m.set(mu, mu, pb.get(mu, mu) - eps);
                    2.0 * eps
                } else {
                    pa_p.set(mu, nu, pa.get(mu, nu) + eps);
                    pa_p.set(nu, mu, pa.get(nu, mu) + eps);
                    pa_m.set(mu, nu, pa.get(mu, nu) - eps);
                    pa_m.set(nu, mu, pa.get(nu, mu) - eps);
                    pb_p.set(mu, nu, pb.get(mu, nu) + eps);
                    pb_p.set(nu, mu, pb.get(nu, mu) + eps);
                    pb_m.set(mu, nu, pb.get(mu, nu) - eps);
                    pb_m.set(nu, mu, pb.get(nu, mu) - eps);
                    4.0 * eps
                };
                let (epa, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa_p, &pb);
                let (ema, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa_m, &pb);
                let fda = (epa - ema) / denom;
                assert!(
                    (va.get(mu, nu) - fda).abs() < 1e-9,
                    "{func:?} UKS alpha ({mu},{nu}): ana={} fd={fda}",
                    va.get(mu, nu)
                );

                let (epb, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb_p);
                let (emb, _, _) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb_m);
                let fdb = (epb - emb) / denom;
                assert!(
                    (vb.get(mu, nu) - fdb).abs() < 1e-9,
                    "{func:?} UKS beta ({mu},{nu}): ana={} fd={fdb}",
                    vb.get(mu, nu)
                );
            }
        }
    }

    #[test]
    fn test_xc_nuclear_gradient_matches_exact_finite_differences_with_moving_grid_and_basis() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).expect("6-31g(d)");
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = basis.n_ao;

        let pa = make_positive_density(n, 3);
        let pb = make_positive_density(n, 8);
        let h_bohr = 1e-5;

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
            let ana = xc_nuclear_gradient(&mol, &basis, &grid, func, &pa, &pb);

            // Translational invariance check: sum_A dE_xc/dR_A == 0!
            for q in 0..3 {
                let sum_q: f64 = ana.iter().map(|g| g[q]).sum();
                assert!(
                    sum_q.abs() < 1e-12,
                    "{func:?} translational invariance violated on axis {q}: sum={sum_q:.3e}"
                );
            }

            // Compare every nuclear coordinate against central FD of E_xc(R +- h) at fixed (pa, pb)!
            for atom_idx in 0..mol.atoms.len() {
                for q in 0..3 {
                    let mut mol_p = mol.clone();
                    let mut mol_m = mol.clone();
                    mol_p.atoms[atom_idx].position_bohr[q] += h_bohr;
                    mol_m.atoms[atom_idx].position_bohr[q] -= h_bohr;

                    let basis_p = BasisSet::from_name("6-31g(d)", &mol_p).unwrap();
                    let basis_m = BasisSet::from_name("6-31g(d)", &mol_m).unwrap();
                    let grid_p = MolecularGrid::build(&mol_p, GridLevel::Coarse);
                    let grid_m = MolecularGrid::build(&mol_m, GridLevel::Coarse);

                    let (ep, _, _) = evaluate_vxc_uks(&basis_p, &grid_p, func, &pa, &pb);
                    let (em, _, _) = evaluate_vxc_uks(&basis_m, &grid_m, func, &pa, &pb);
                    let fd = (ep - em) / (2.0 * h_bohr);

                    let diff = (ana[atom_idx][q] - fd).abs();
                    assert!(
                        diff < 1e-8,
                        "{func:?} atom {atom_idx} axis {q}: ana={:.10e} fd={:.10e} diff={diff:.3e}",
                        ana[atom_idx][q],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn test_streaming_grid_matches_materialized_vxc_and_gradient_with_zero_residency() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).expect("6-31g(d)");
        let grid_mat = MolecularGrid::build_for_basis(&mol, &basis, GridLevel::Medium);
        let grid_stream = MolecularGrid::build_streaming_for_basis(&mol, &basis, GridLevel::Medium);

        assert!(!grid_mat.points.is_empty());
        assert!(
            grid_stream.points.is_empty() && grid_stream.is_streaming(),
            "Streaming grid must retain zero full-grid Vec<GridPoint> residency"
        );

        let n = basis.n_ao;
        let p_tot = make_positive_density(n, 1);
        let pa = make_positive_density(n, 3);
        let pb = make_positive_density(n, 8);

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp, XcFunctional::CamB3lyp] {
            let (e_m, v_m) = evaluate_vxc_rks(&basis, &grid_mat, func, &p_tot);
            let (e_s, v_s) = evaluate_vxc_rks(&basis, &grid_stream, func, &p_tot);
            assert!(
                (e_m - e_s).abs() < 1e-13,
                "{func:?} RKS streaming E_xc mismatch: {e_m} vs {e_s}"
            );
            for idx in 0..n * n {
                assert!(
                    (v_m.data[idx] - v_s.data[idx]).abs() < 1e-13,
                    "{func:?} RKS streaming V_xc mismatch at {idx}"
                );
            }

            let (eu_m, va_m, vb_m) = evaluate_vxc_uks(&basis, &grid_mat, func, &pa, &pb);
            let (eu_s, va_s, vb_s) = evaluate_vxc_uks(&basis, &grid_stream, func, &pa, &pb);
            assert!((eu_m - eu_s).abs() < 1e-13);
            for idx in 0..n * n {
                assert!((va_m.data[idx] - va_s.data[idx]).abs() < 1e-13);
                assert!((vb_m.data[idx] - vb_s.data[idx]).abs() < 1e-13);
            }

            let g_m = xc_nuclear_gradient(&mol, &basis, &grid_mat, func, &pa, &pb);
            let g_s = xc_nuclear_gradient(&mol, &basis, &grid_stream, func, &pa, &pb);
            for a in 0..mol.atoms.len() {
                for q in 0..3 {
                    assert!(
                        (g_m[a][q] - g_s[a][q]).abs() < 1e-13,
                        "{func:?} streaming XC grad mismatch at ({a},{q})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_vxc_matches_cpu_to_machine_precision() {
        let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let grid = MolecularGrid::build_for_basis(&mol, &basis, GridLevel::Coarse);
        let n = basis.n_ao;
        let p_tot = make_positive_density(n, 7);

        let gpu_eng = RksVxcEngine::new_with_gpu(&basis, &grid, true);
        assert!(gpu_eng.gpu.is_some(), "GPU Vxc engine should initialize on CUDA device");
        let cpu_eng = RksVxcEngine::new_with_gpu(&basis, &grid, false);

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp, XcFunctional::CamB3lyp] {
            let (e_ref, v_ref) = evaluate_vxc_rks(&basis, &grid, func, &p_tot);
            let (e_cpu, v_cpu) = cpu_eng.evaluate(&basis, &grid, func, &p_tot);
            let (e_gpu, v_gpu) = gpu_eng.evaluate(&basis, &grid, func, &p_tot);

            assert!(
                (e_gpu - e_ref).abs() < 1e-12,
                "{func:?} GPU E_xc mismatch: gpu={e_gpu:.15} ref={e_ref:.15}"
            );
            assert!(
                (e_cpu - e_ref).abs() < 1e-12,
                "{func:?} CPU BLAS-3 E_xc mismatch: cpu={e_cpu:.15} ref={e_ref:.15}"
            );
            let mut max_gpu_err = 0.0f64;
            let mut max_cpu_err = 0.0f64;
            for idx in 0..n * n {
                max_gpu_err = max_gpu_err.max((v_gpu.data[idx] - v_ref.data[idx]).abs());
                max_cpu_err = max_cpu_err.max((v_cpu.data[idx] - v_ref.data[idx]).abs());
            }
            assert!(
                max_gpu_err < 1e-11,
                "{func:?} GPU V_xc max abs error {max_gpu_err:.3e} >= 1e-11"
            );
            assert!(
                max_cpu_err < 1e-12,
                "{func:?} CPU BLAS-3 V_xc max abs error {max_cpu_err:.3e} >= 1e-12"
            );
        }
        let (k, fb) = gpu_eng.telemetry();
        assert!(k >= 8 && fb == 0, "Expected GPU Vxc telemetry k>=8 fb==0, got k={k} fb={fb}");
    }

    #[test]
    fn test_gpu_uks_vxc_matches_cpu_to_machine_precision() {
        let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if !calchem_cuda::is_cuda_available() {
            return;
        }
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let grid = MolecularGrid::build_for_basis(&mol, &basis, GridLevel::Coarse);
        let n = basis.n_ao;
        let pa = make_positive_density(n, 7);
        let pb = make_positive_density(n, 13);

        let gpu_eng = UksVxcEngine::new_with_gpu(&basis, &grid, true);
        assert!(gpu_eng.gpu.is_some(), "GPU UKS Vxc engine should initialize on CUDA device");

        for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp, XcFunctional::CamB3lyp] {
            let (e_ref, va_ref, vb_ref) = evaluate_vxc_uks(&basis, &grid, func, &pa, &pb);
            let (e_gpu, va_gpu, vb_gpu) = gpu_eng.evaluate(&basis, &grid, func, &pa, &pb);
            assert!(
                (e_gpu - e_ref).abs() < 1e-12,
                "{func:?} GPU UKS E_xc mismatch: gpu={e_gpu:.15} ref={e_ref:.15}"
            );
            let mut max_err = 0.0f64;
            for idx in 0..n * n {
                max_err = max_err
                    .max((va_gpu.data[idx] - va_ref.data[idx]).abs())
                    .max((vb_gpu.data[idx] - vb_ref.data[idx]).abs());
            }
            assert!(max_err < 1e-11, "{func:?} GPU UKS V_xc max abs err {max_err:.3e} >= 1e-11");
        }
        let (k, fb) = gpu_eng.telemetry();
        assert!(k >= 16 && fb == 0, "Expected UKS GPU Vxc telemetry k>=16 fb==0, got k={k} fb={fb}");
    }

    #[test]
    fn test_gpu_xc_nuclear_gradient_matches_cpu_and_disable_gpu() {
        let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("6-31g(d)", &mol).unwrap();
        let grid = MolecularGrid::build_for_basis(&mol, &basis, GridLevel::Coarse);
        let n = basis.n_ao;
        let pa = make_positive_density(n, 3);
        let pb = make_positive_density(n, 9);

        if calchem_cuda::is_cuda_available() {
            for func in [XcFunctional::Svwn5, XcFunctional::Pbe, XcFunctional::B3lyp] {
                let g_cpu_rks = xc_nuclear_gradient(&mol, &basis, &grid, func, &pa, &pa);
                let (g_gpu_rks, kr, fbr) =
                    xc_nuclear_gradient_with_gpu(&mol, &basis, &grid, func, &pa, &pa, true);
                assert!(kr == 2 && fbr == 0, "{func:?} RKS XC grad telemetry ({kr}, {fbr})");
                let mut max_r = 0.0f64;
                for (a, b) in g_cpu_rks.iter().zip(&g_gpu_rks) {
                    for q in 0..3 {
                        max_r = max_r.max((a[q] - b[q]).abs());
                    }
                }
                assert!(max_r < 1e-10, "{func:?} RKS GPU XC grad max_err={max_r:.3e}");

                let g_cpu_uks = xc_nuclear_gradient(&mol, &basis, &grid, func, &pa, &pb);
                let (g_gpu_uks, ku, fbu) =
                    xc_nuclear_gradient_with_gpu(&mol, &basis, &grid, func, &pa, &pb, true);
                assert!(ku == 3 && fbu == 0, "{func:?} UKS XC grad telemetry ({ku}, {fbu})");
                let mut max_u = 0.0f64;
                for (a, b) in g_cpu_uks.iter().zip(&g_gpu_uks) {
                    for q in 0..3 {
                        max_u = max_u.max((a[q] - b[q]).abs());
                    }
                }
                assert!(max_u < 1e-10, "{func:?} UKS GPU XC grad max_err={max_u:.3e}");
            }
        }

        struct ResetDisable(Option<std::ffi::OsString>);
        impl Drop for ResetDisable {
            fn drop(&mut self) {
                if let Some(ref v) = self.0 {
                    std::env::set_var("CALCHEM_DISABLE_GPU", v);
                } else {
                    std::env::remove_var("CALCHEM_DISABLE_GPU");
                }
            }
        }
        let _reset = ResetDisable(std::env::var_os("CALCHEM_DISABLE_GPU"));
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");
        let (g_dis, kd, fbd) =
            xc_nuclear_gradient_with_gpu(&mol, &basis, &grid, XcFunctional::B3lyp, &pa, &pa, true);
        let g_cpu = xc_nuclear_gradient(&mol, &basis, &grid, XcFunctional::B3lyp, &pa, &pa);
        assert_eq!((kd, fbd), (0, 1));
        let max_d = g_dis.iter().zip(&g_cpu).flat_map(|(a, b)| (0..3).map(move |q| (a[q] - b[q]).abs())).fold(0.0f64, f64::max);
        assert!(max_d < 1e-14);
    }

    #[test]
    fn test_vxc_hybrid_streaming_with_materialized_points_and_zero_weight_gpu_no_panic() {
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("sto-3g", &mol).unwrap();
        let mut grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        grid.streaming = Some((mol.clone(), 0));
        let n = basis.n_ao;
        let pa = make_positive_density(n, 1);
        let pb = make_positive_density(n, 2);
        let mut p_tot = Matrix2D::zeros(n, n);
        for i in 0..n * n {
            p_tot.data[i] = pa.data[i] + pb.data[i];
        }

        let (er, vr) = evaluate_vxc_rks(&basis, &grid, XcFunctional::B3lyp, &p_tot);
        let (eu, va, vb) = evaluate_vxc_uks(&basis, &grid, XcFunctional::B3lyp, &pa, &pb);
        assert!(er.is_finite() && eu.is_finite());
        assert!(vr.sym_frobenius_error() < 1e-13);
        assert!(va.sym_frobenius_error() < 1e-13 && vb.sym_frobenius_error() < 1e-13);

        let zero_w_grid = MolecularGrid {
            points: vec![GridPoint {
                coords: [0.0, 0.0, 0.0],
                weight: 0.0,
                raw_weight: 0.0,
                host_atom: 0,
            }],
            level: GridLevel::Coarse,
            streaming: None,
        };
        let rks_eng = RksVxcEngine::new_with_gpu(&basis, &zero_w_grid, true);
        let (e0, v0) = rks_eng.evaluate(&basis, &zero_w_grid, XcFunctional::B3lyp, &p_tot);
        assert_eq!(e0, 0.0);
        assert!(v0.data.iter().all(|&x| x == 0.0));
        let uks_eng = UksVxcEngine::new_with_gpu(&basis, &zero_w_grid, true);
        let (eu0, va0, vb0) =
            uks_eng.evaluate(&basis, &zero_w_grid, XcFunctional::B3lyp, &pa, &pb);
        assert_eq!(eu0, 0.0);
        assert!(va0.data.iter().all(|&x| x == 0.0) && vb0.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_vxc_engine_disable_gpu_fallback_telemetry_not_double_counted() {
        let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = distorted_water(None);
        let basis = BasisSet::from_name("sto-3g", &mol).unwrap();
        let grid = MolecularGrid::build(&mol, GridLevel::Coarse);
        let n = basis.n_ao;
        let p_tot = make_positive_density(n, 4);
        let pa = make_positive_density(n, 5);
        let pb = make_positive_density(n, 6);

        let prev = std::env::var_os("CALCHEM_DISABLE_GPU");
        std::env::set_var("CALCHEM_DISABLE_GPU", "1");

        let rks_eng = RksVxcEngine::new_with_gpu(&basis, &grid, true);
        let _ = rks_eng.evaluate(&basis, &grid, XcFunctional::Pbe, &p_tot);
        let (rk, rfb) = rks_eng.telemetry();
        assert_eq!(rk, 0);
        assert_eq!(rfb, 1, "Single RKS evaluate with GPU disabled should record 1 fallback, got {rfb}");

        let uks_eng = UksVxcEngine::new_with_gpu(&basis, &grid, true);
        let _ = uks_eng.evaluate(&basis, &grid, XcFunctional::Pbe, &pa, &pb);
        let (uk, ufb) = uks_eng.telemetry();
        assert_eq!(uk, 0);
        assert_eq!(ufb, 1, "Single UKS evaluate with GPU disabled should record 1 fallback, got {ufb}");

        if let Some(v) = prev {
            std::env::set_var("CALCHEM_DISABLE_GPU", v);
        } else {
            std::env::remove_var("CALCHEM_DISABLE_GPU");
        }
    }
}
