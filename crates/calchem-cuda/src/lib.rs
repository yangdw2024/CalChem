use calchem_model::CalcError;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AuxShellGpu {
    pub center: [f64; 3],
    pub bound: f64,
    pub lp: i32,
    pub ao_offset: i32,
    pub prim_start: i32,
    pub n_prims: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AuxPrimGpu {
    pub gamma_p: f64,
    pub cp_norm: [f64; 6],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShellPair3CGpu {
    pub q_ij: f64,
    pub li: i32,
    pub lj: i32,
    pub same_shell: i32,
    pub col_offset: i32,
    pub prim_start: i32,
    pub n_prims: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrimPair3CGpu {
    pub q: f64,
    pub p_center: [f64; 3],
    pub qc: [f64; 3],
    pub qd: [f64; 3],
    pub two_pi25_kab_over_q: f64,
    pub ci_norm: [f64; 6],
    pub cj_norm: [f64; 6],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AoShellGpu {
    pub center: [f64; 3],
    pub r2_cutoff: f64,
    pub ao_offset: i32,
    pub n_funcs: i32,
    pub prim_start: i32,
    pub n_prims: i32,
    pub comp_start: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AoPrimGpu {
    pub exponent: f64,
    pub weighted_norm: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AoCompGpu {
    pub scale: f64,
    pub lx: i32,
    pub ly: i32,
    pub lz: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GridPtGpu {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub batch_id: i32,
    pub _pad: i32,
}

#[cfg(not(no_cuda))]
extern "C" {
    fn calchem_cuda_is_available() -> i32;
    fn calchem_cuda_get_memory_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
    fn calchem_cuda_rij_contract_coulomb_tagged(
        h_b_block: *const f64,
        h_x_block: *const f64,
        h_j: *mut f64,
        n_p: i32,
        n_ao_sq: i32,
        memory_limit_bytes: usize,
        tensor_id: u64,
    ) -> i32;
    fn calchem_cuda_rij_contract_v_tagged(
        h_b_block: *const f64,
        h_p_block: *const f64,
        h_v: *mut f64,
        n_p: i32,
        n_ao_sq: i32,
        memory_limit_bytes: usize,
        tensor_id: u64,
    ) -> i32;
    fn calchem_cuda_build_eri3c_packed(
        h_aux_shells: *const AuxShellGpu,
        n_aux_shells: i32,
        h_aux_prims: *const AuxPrimGpu,
        n_aux_prims: i32,
        h_shell_pairs: *const ShellPair3CGpu,
        n_shell_pairs: i32,
        h_prim_pairs: *const PrimPair3CGpu,
        n_prim_pairs: i32,
        h_b_out: *mut f64,
        n_aux: i32,
        width: i32,
        memory_limit_bytes: usize,
        tensor_id: u64,
    ) -> i32;
    fn calchem_cuda_vxc_upload_grid(
        h_shells: *const AoShellGpu,
        n_shells: i32,
        h_prims: *const AoPrimGpu,
        n_prims: i32,
        h_comps: *const AoCompGpu,
        n_comps: i32,
        h_pts: *const GridPtGpu,
        n_pts: i32,
        h_cand_offsets: *const i32,
        n_cand_offsets: i32,
        h_cand_shells: *const i32,
        n_cand_shells: i32,
        n_ao: i32,
        grid_id: u64,
    ) -> i32;
    fn calchem_cuda_vxc_eval_rho(
        grid_id: u64,
        h_p_mat: *const f64,
        h_rho_grad_out: *mut f64,
        n_pts: i32,
        n_ao: i32,
    ) -> i32;
    fn calchem_cuda_vxc_accum(
        grid_id: u64,
        h_wv_in: *const f64,
        h_vxc_out: *mut f64,
        n_pts: i32,
        n_ao: i32,
    ) -> i32;
}

/// Hard safety ceiling on device memory allocation (5.0 GiB) to guarantee stability on 8 GB cards.
pub const MAX_GPU_MEMORY_BYTES: usize = 5 * 1024 * 1024 * 1024;

static GPU_ENABLED: AtomicBool = AtomicBool::new(true);

/// Checks whether a functional CUDA device is available and accessible.
pub fn is_cuda_available() -> bool {
    if !GPU_ENABLED.load(Ordering::Relaxed)
        || std::env::var("CALCHEM_DISABLE_GPU").is_ok_and(|v| v == "1")
    {
        return false;
    }
    #[cfg(not(no_cuda))]
    unsafe {
        calchem_cuda_is_available() == 1
    }
    #[cfg(no_cuda)]
    {
        false
    }
}

/// Allows disabling or enabling GPU acceleration dynamically (e.g. forced CPU fallback).
pub fn set_gpu_enabled(enabled: bool) {
    GPU_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Queries free and total device memory in bytes.
pub fn get_cuda_memory_info() -> Result<(usize, usize), CalcError> {
    if !is_cuda_available() {
        return Err(CalcError::UnsupportedMethod(
            "CUDA device unavailable".into(),
        ));
    }
    #[cfg(not(no_cuda))]
    unsafe {
        let mut free = 0usize;
        let mut total = 0usize;
        let ret = calchem_cuda_get_memory_info(&mut free, &mut total);
        if ret == 0 {
            Ok((free, total))
        } else {
            Err(CalcError::Internal(format!(
                "cudaMemGetInfo failed with code {ret}"
            )))
        }
    }
    #[cfg(no_cuda)]
    {
        Err(CalcError::UnsupportedMethod(
            "CUDA not compiled into binary".into(),
        ))
    }
}

fn validate_rij_contract_inputs(
    b_block: &[f64],
    vec_p_len: usize,
    vec_ao_len: usize,
    n_p: usize,
    n_ao_sq: usize,
) -> bool {
    let Some(b_len) = n_p.checked_mul(n_ao_sq) else {
        return false;
    };
    n_p > 0
        && n_ao_sq > 0
        && n_p <= i32::MAX as usize
        && n_ao_sq <= i32::MAX as usize
        && b_block.len() == b_len
        && vec_p_len == n_p
        && vec_ao_len == n_ao_sq
        && is_cuda_available()
}

fn resolve_gpu_memory_limit() -> usize {
    std::env::var("CALCHEM_MAX_GPU_MEM_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .and_then(|v| v.checked_mul(1024 * 1024))
        .unwrap_or(MAX_GPU_MEMORY_BYTES)
        .min(MAX_GPU_MEMORY_BYTES)
}

/// An immutable host allocation with a unique identity for device-side RI-J reuse.
pub struct ResidentRijTensor {
    data: Vec<f64>,
    id: u64,
}

impl ResidentRijTensor {
    fn next_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .expect("RI-J tensor identifier exhausted")
    }

    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            id: Self::next_id(),
        }
    }

    /// Evaluates the packed 3-center RI-J tensor `(P | i j)` directly on the CUDA GPU
    /// (for all `lp <= 2, li <= 1, lj <= 1` shells), caching it in GPU VRAM with a unique
    /// `tensor_id` so subsequent contractions hit VRAM immediately with zero upload overhead.
    pub fn build_eri3c_gpu(
        aux_shells: &[AuxShellGpu],
        aux_prims: &[AuxPrimGpu],
        shell_pairs: &[ShellPair3CGpu],
        prim_pairs: &[PrimPair3CGpu],
        n_aux: usize,
        width: usize,
    ) -> Option<Self> {
        if !is_cuda_available()
            || aux_shells.is_empty()
            || aux_prims.is_empty()
            || shell_pairs.is_empty()
            || prim_pairs.is_empty()
            || n_aux == 0
            || width == 0
            || n_aux > i32::MAX as usize
            || width > i32::MAX as usize
        {
            return None;
        }
        let len = n_aux.checked_mul(width)?;
        let mut data = vec![0.0f64; len];
        let id = Self::next_id();

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_build_eri3c_packed(
                aux_shells.as_ptr(),
                aux_shells.len() as i32,
                aux_prims.as_ptr(),
                aux_prims.len() as i32,
                shell_pairs.as_ptr(),
                shell_pairs.len() as i32,
                prim_pairs.as_ptr(),
                prim_pairs.len() as i32,
                data.as_mut_ptr(),
                n_aux as i32,
                width as i32,
                resolve_gpu_memory_limit(),
                id,
            );
            if ret == 0 {
                return Some(Self { data, id });
            }
        }
        let _ = (data, id);
        None
    }

    pub fn view(&self) -> RijTensorView<'_> {
        RijTensorView {
            data: &self.data,
            id: self.id,
        }
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

#[derive(Copy, Clone)]
pub struct RijTensorView<'a> {
    data: &'a [f64],
    id: u64,
}

impl<'a> From<&'a [f64]> for RijTensorView<'a> {
    fn from(data: &'a [f64]) -> Self {
        Self { data, id: 0 }
    }
}

/// Contracts a chunk of auxiliary 3-center integral block `B[p, mu, nu]` with `x[p]`
/// into Coulomb matrix `J[mu, nu]` using GPU acceleration if available, returning `true`
/// if GPU execution succeeded, or `false` to request automatic CPU fallback.
pub fn cuda_rij_contract_coulomb(
    b_block: &[f64],
    x_block: &[f64],
    j_matrix: &mut [f64],
    n_p: usize,
    n_ao_sq: usize,
) -> bool {
    RijTensorView::from(b_block).contract_coulomb(x_block, j_matrix, n_p, n_ao_sq)
}

impl RijTensorView<'_> {
    pub fn contract_coulomb(
        self,
        x_block: &[f64],
        j_matrix: &mut [f64],
        n_p: usize,
        n_ao_sq: usize,
    ) -> bool {
        let b_block = self.data;
        if !validate_rij_contract_inputs(b_block, x_block.len(), j_matrix.len(), n_p, n_ao_sq) {
            return false;
        }

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_rij_contract_coulomb_tagged(
                b_block.as_ptr(),
                x_block.as_ptr(),
                j_matrix.as_mut_ptr(),
                n_p as i32,
                n_ao_sq as i32,
                resolve_gpu_memory_limit(),
                self.id,
            );
            ret == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = (b_block, x_block, j_matrix, n_p, n_ao_sq);
            false
        }
    }
}

/// Contracts auxiliary 3-center tensor `B[p, mu, nu]` with density matrix `P[mu, nu]`
/// into projection vector `v[p] = sum_{mu,nu} B[p, mu, nu] P[mu, nu]` on the GPU.
pub fn cuda_rij_contract_v(
    b_block: &[f64],
    p_matrix: &[f64],
    v_vec: &mut [f64],
    n_p: usize,
    n_ao_sq: usize,
) -> bool {
    RijTensorView::from(b_block).contract_v(p_matrix, v_vec, n_p, n_ao_sq)
}

impl RijTensorView<'_> {
    pub fn contract_v(
        self,
        p_matrix: &[f64],
        v_vec: &mut [f64],
        n_p: usize,
        n_ao_sq: usize,
    ) -> bool {
        let b_block = self.data;
        if !validate_rij_contract_inputs(b_block, v_vec.len(), p_matrix.len(), n_p, n_ao_sq) {
            return false;
        }

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_rij_contract_v_tagged(
                b_block.as_ptr(),
                p_matrix.as_ptr(),
                v_vec.as_mut_ptr(),
                n_p as i32,
                n_ao_sq as i32,
                resolve_gpu_memory_limit(),
                self.id,
            );
            ret == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = (b_block, p_matrix, v_vec, n_p, n_ao_sq);
            false
        }
    }
}

/// GPU-resident spatial-screened DFT quadrature grid and basis metadata for FP64 $V_{\text{xc}}$.
pub struct GpuVxcGrid {
    grid_id: u64,
    n_pts: usize,
    n_ao: usize,
}

impl GpuVxcGrid {
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        shells: &[AoShellGpu],
        prims: &[AoPrimGpu],
        comps: &[AoCompGpu],
        pts: &[GridPtGpu],
        cand_offsets: &[i32],
        cand_shells: &[i32],
        n_ao: usize,
    ) -> Option<Self> {
        if !is_cuda_available()
            || shells.is_empty()
            || prims.is_empty()
            || comps.is_empty()
            || pts.is_empty()
            || cand_offsets.len() < 2
            || cand_shells.is_empty()
            || n_ao == 0
        {
            return None;
        }
        static NEXT_GRID_ID: AtomicU64 = AtomicU64::new(1);
        let grid_id = NEXT_GRID_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .expect("GPU Vxc grid identifier exhausted");

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_vxc_upload_grid(
                shells.as_ptr(),
                shells.len() as i32,
                prims.as_ptr(),
                prims.len() as i32,
                comps.as_ptr(),
                comps.len() as i32,
                pts.as_ptr(),
                pts.len() as i32,
                cand_offsets.as_ptr(),
                cand_offsets.len() as i32,
                cand_shells.as_ptr(),
                cand_shells.len() as i32,
                n_ao as i32,
                grid_id,
            );
            if ret == 0 {
                return Some(Self {
                    grid_id,
                    n_pts: pts.len(),
                    n_ao,
                });
            }
        }
        let _ = grid_id;
        None
    }

    pub fn eval_rho_grad(&self, p_mat: &[f64], rho_grad_out: &mut [f64]) -> bool {
        if p_mat.len() != self.n_ao * self.n_ao || rho_grad_out.len() != self.n_pts * 4 {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_vxc_eval_rho(
                self.grid_id,
                p_mat.as_ptr(),
                rho_grad_out.as_mut_ptr(),
                self.n_pts as i32,
                self.n_ao as i32,
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            false
        }
    }

    pub fn accumulate_vxc(&self, wv_in: &[f64], vxc_out: &mut [f64]) -> bool {
        if wv_in.len() != self.n_pts * 4 || vxc_out.len() != self.n_ao * self.n_ao {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_vxc_accum(
                self.grid_id,
                wv_in.as_ptr(),
                vxc_out.as_mut_ptr(),
                self.n_pts as i32,
                self.n_ao as i32,
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            false
        }
    }
}
