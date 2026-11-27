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
    pub cp_norm: [f64; 10],
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
    pub atom_index: i32,
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
    pub host_atom: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad3cAuxShellGpu {
    pub center: [f64; 3],
    pub bound: f64,
    pub comp_scales: [f64; 10],
    pub lp: i32,
    pub ao_offset: i32,
    pub prim_start: i32,
    pub n_prims: i32,
    pub atom_index: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad3cAuxPrimGpu {
    pub gamma_p: f64,
    pub cp_lead: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad3cShellPairGpu {
    pub q_ij: f64,
    pub ab_vec: [f64; 3],
    pub comp_scales_i: [f64; 6],
    pub comp_scales_j: [f64; 6],
    pub li: i32,
    pub lj: i32,
    pub same_shell: i32,
    pub atom_i: i32,
    pub atom_j: i32,
    pub ao_off_i: i32,
    pub ao_off_j: i32,
    pub prim_start: i32,
    pub n_prims: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad3cPrimPairGpu {
    pub q: f64,
    pub alpha: f64,
    pub beta: f64,
    pub two_pi25_kab_w_over_q: f64,
    pub p_center: [f64; 3],
    pub qc: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad4cShellGpu {
    pub center: [f64; 3],
    pub comp_scales: [f64; 10],
    pub l: i32,
    pub ao_offset: i32,
    pub n_funcs: i32,
    pub atom_index: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad4cShellPairGpu {
    pub shell_i: i32,
    pub shell_j: i32,
    pub bound: f64,
    pub prim_pair_start: i32,
    pub n_prim_pairs: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Grad4cPrimPairGpu {
    pub gamma: f64,
    pub kab_w: f64,
    pub alpha: f64,
    pub beta: f64,
    pub p: [f64; 3],
    pub pa: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectShellGpu {
    pub center: [f64; 3],
    pub comp_scales: [f64; 10],
    pub l: i32,
    pub ao_offset: i32,
    pub n_funcs: i32,
    pub _pad: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectShellPairGpu {
    pub shell_i: i32,
    pub shell_j: i32,
    pub bound: f64,
    pub prim_pair_start: i32,
    pub n_prim_pairs: i32,
    pub cached_kets: i32,
    pub cache_valid_offset: i32,
    pub cache_data_offset: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectPrimPairGpu {
    pub gamma: f64,
    pub kab_w: f64,
    pub p: [f64; 3],
    pub pa: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectStatsGpu {
    pub computed_quartets: u64,
    pub skipped_quartets: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub fp64_quartets: u64,
    pub fp32_quartets: u64,
}

#[cfg(not(no_cuda))]
extern "C" {
    fn calchem_cuda_is_available() -> i32;
    fn calchem_cuda_get_memory_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
    fn calchem_cuda_total_allocated_bytes() -> usize;
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
    fn calchem_cuda_rij_release(tensor_id: u64) -> i32;
    fn calchem_cuda_rij_stream_upload(
        h_aux_shells: *const AuxShellGpu,
        n_aux_shells: i32,
        h_aux_prims: *const AuxPrimGpu,
        n_aux_prims: i32,
        h_shell_pairs: *const ShellPair3CGpu,
        n_shell_pairs: i32,
        h_prim_pairs: *const PrimPair3CGpu,
        n_prim_pairs: i32,
        max_chunk_n_aux: i32,
        width: i32,
        memory_limit_bytes: usize,
        stream_id: u64,
    ) -> i32;
    fn calchem_cuda_rij_stream_pass1_begin(
        stream_id: u64,
        h_p_packed: *const f64,
        width: i32,
    ) -> i32;
    fn calchem_cuda_rij_stream_pass1_v(
        stream_id: u64,
        h_p_packed: *const f64,
        width: i32,
        shell_start: i32,
        chunk_n_shells: i32,
        ao_start: i32,
        chunk_n_aux: i32,
        h_v_out: *mut f64,
    ) -> i32;
    fn calchem_cuda_rij_stream_pass2_begin(stream_id: u64, width: i32) -> i32;
    fn calchem_cuda_rij_stream_pass2_accum(
        stream_id: u64,
        h_x_chunk: *const f64,
        shell_start: i32,
        chunk_n_shells: i32,
        ao_start: i32,
        chunk_n_aux: i32,
        width: i32,
    ) -> i32;
    fn calchem_cuda_rij_stream_pass2_finish(
        stream_id: u64,
        h_j_packed: *mut f64,
        width: i32,
    ) -> i32;
    fn calchem_cuda_rij_stream_release(stream_id: u64) -> i32;
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
        memory_limit_bytes: usize,
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
    fn calchem_cuda_vxc_release(grid_id: u64) -> i32;
    fn calchem_cuda_direct_recommended_cache_bytes(
        memory_limit_bytes: usize,
        max_cache_cap: usize,
    ) -> usize;
    fn calchem_cuda_direct_upload(
        h_shells: *const DirectShellGpu,
        n_shells: i32,
        h_pairs: *const DirectShellPairGpu,
        n_pairs: i32,
        h_prims: *const DirectPrimPairGpu,
        n_prims: i32,
        h_pair_prefix: *const u64,
        n_prefix: i32,
        n_ao: i32,
        cache_elements: usize,
        valid_slots: usize,
        memory_limit_bytes: usize,
        wl_stride_override: u32,
        direct_id: u64,
    ) -> i32;
    fn calchem_cuda_direct_contract(
        direct_id: u64,
        mode: i32,
        h_p_a: *const f64,
        h_p_b: *const f64,
        h_d_t: *const f64,
        h_d_a: *const f64,
        h_d_b: *const f64,
        active_pairs: i32,
        h_ends: *const i32,
        exx: f64,
        omega: f64,
        has_omega: i32,
        thresh: f64,
        fp32_thresh: f64,
        h_out_a: *mut f64,
        h_out_b: *mut f64,
        out_stats: *mut DirectStatsGpu,
    ) -> i32;
    fn calchem_cuda_direct_td_scratch_bytes(direct_id: u64) -> usize;
    fn calchem_cuda_direct_contract_tddft_batch(
        direct_id: u64,
        b_sym: i32,
        b_anti: i32,
        do_j: i32,
        do_k_sym: i32,
        do_k_anti: i32,
        h_d_sym_packed: *const f64,
        h_d_anti_packed: *const f64,
        h_dj_env: *const f64,
        h_dk_env: *const f64,
        active_pairs: i32,
        h_ends: *const i32,
        j_weight: f64,
        k_weight: f64,
        omega: f64,
        has_omega: i32,
        thresh: f64,
        fp32_thresh: f64,
        h_out_j_packed: *mut f64,
        h_out_ks_packed: *mut f64,
        h_out_ka_packed: *mut f64,
        out_stats: *mut DirectStatsGpu,
    ) -> i32;
    fn calchem_cuda_direct_release(direct_id: u64) -> i32;
    fn calchem_cuda_vxc_grad_force(
        grid_id: u64,
        h_pa: *const f64,
        h_pb: *const f64,
        is_rks: i32,
        h_wxc_in: *const f64,
        h_grad_out: *mut f64,
        n_pts: i32,
        n_ao: i32,
        n_atoms: i32,
        memory_limit_bytes: usize,
    ) -> i32;
    fn calchem_cuda_rij_grad_v(
        h_aux_shells: *const Grad3cAuxShellGpu,
        n_aux_shells: i32,
        h_aux_prims: *const Grad3cAuxPrimGpu,
        n_aux_prims: i32,
        h_shell_pairs: *const Grad3cShellPairGpu,
        n_shell_pairs: i32,
        h_prim_pairs: *const Grad3cPrimPairGpu,
        n_prim_pairs: i32,
        h_p_mat: *const f64,
        h_v_out: *mut f64,
        n_aux: i32,
        n_ao: i32,
        memory_limit_bytes: usize,
    ) -> i32;
    fn calchem_cuda_rij_grad_3c(
        h_aux_shells: *const Grad3cAuxShellGpu,
        n_aux_shells: i32,
        h_aux_prims: *const Grad3cAuxPrimGpu,
        n_aux_prims: i32,
        h_shell_pairs: *const Grad3cShellPairGpu,
        n_shell_pairs: i32,
        h_prim_pairs: *const Grad3cPrimPairGpu,
        n_prim_pairs: i32,
        h_x_vec: *const f64,
        h_p_mat: *const f64,
        h_grad_out: *mut f64,
        n_aux: i32,
        n_ao: i32,
        n_atoms: i32,
        memory_limit_bytes: usize,
    ) -> i32;
    fn calchem_cuda_grad4c_contract(
        h_shells: *const Grad4cShellGpu,
        n_shells: i32,
        h_pairs: *const Grad4cShellPairGpu,
        n_pairs: i32,
        h_prims: *const Grad4cPrimPairGpu,
        n_prims: i32,
        h_pt: *const f64,
        h_pa: *const f64,
        h_pb: *const f64,
        h_d_tot_max: *const f64,
        h_d_spin_max: *const f64,
        h_ends: *const i32,
        n_ao: i32,
        n_atoms: i32,
        do_coul: i32,
        exx: f64,
        omega: f64,
        has_omega: i32,
        thresh: f64,
        h_grad_out: *mut f64,
        memory_limit_bytes: usize,
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

/// Returns total device memory currently tracked by `calchem-cuda` across all active allocations.
pub fn total_allocated_gpu_bytes() -> usize {
    #[cfg(not(no_cuda))]
    unsafe {
        calchem_cuda_total_allocated_bytes()
    }
    #[cfg(no_cuda)]
    {
        0
    }
}

/// Resolves the effective GPU memory limit (<= `MAX_GPU_MEMORY_BYTES`).
pub fn resolve_gpu_memory_limit() -> usize {
    std::env::var("CALCHEM_MAX_GPU_MEM_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .and_then(|v| v.checked_mul(1024 * 1024))
        .unwrap_or(MAX_GPU_MEMORY_BYTES)
        .min(MAX_GPU_MEMORY_BYTES)
}

fn validate_rij_contract_inputs(
    b_block: &[f64],
    device_only: bool,
    vec_p_len: usize,
    vec_ao_len: usize,
    n_p: usize,
    n_ao_sq: usize,
) -> bool {
    let Some(b_len) = n_p.checked_mul(n_ao_sq) else {
        return false;
    };
    let b_ok = if device_only {
        b_block.is_empty()
    } else {
        b_block.len() == b_len
    };
    n_p > 0
        && n_ao_sq > 0
        && n_p <= i32::MAX as usize
        && n_ao_sq <= i32::MAX as usize
        && b_ok
        && vec_p_len == n_p
        && vec_ao_len == n_ao_sq
        && is_cuda_available()
}

fn aux_shell_n_funcs(lp: i32) -> Option<usize> {
    match lp {
        0 => Some(1),
        1 => Some(3),
        2 => Some(6),
        3 => Some(10),
        _ => None,
    }
}

/// An immutable host or device-resident allocation with a unique identity for device-side RI-J reuse.
pub struct ResidentRijTensor {
    data: Vec<f64>,
    id: u64,
    device_only: bool,
}

impl Drop for ResidentRijTensor {
    fn drop(&mut self) {
        if self.id != 0 {
            #[cfg(not(no_cuda))]
            unsafe {
                let _ = calchem_cuda_rij_release(self.id);
            }
        }
    }
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
            device_only: false,
        }
    }

    /// Evaluates the packed 3-center RI-J tensor `(P | i j)` directly on the CUDA GPU
    /// (for all `lp <= 3, li <= 2, lj <= 2` shells), caching it in GPU VRAM with a unique
    /// `tensor_id` and copying it back to host RAM.
    pub fn build_eri3c_gpu(
        aux_shells: &[AuxShellGpu],
        aux_prims: &[AuxPrimGpu],
        shell_pairs: &[ShellPair3CGpu],
        prim_pairs: &[PrimPair3CGpu],
        n_aux: usize,
        width: usize,
    ) -> Option<Self> {
        Self::build_eri3c_gpu_impl(
            aux_shells,
            aux_prims,
            shell_pairs,
            prim_pairs,
            n_aux,
            width,
            true,
        )
    }

    /// Evaluates the packed 3-center RI-J tensor `(P | i j)` directly on the CUDA GPU
    /// (for all `lp <= 3, li <= 2, lj <= 2` shells) and keeps it **device-only** in VRAM
    /// with zero host `B` allocation.
    pub fn build_eri3c_gpu_resident(
        aux_shells: &[AuxShellGpu],
        aux_prims: &[AuxPrimGpu],
        shell_pairs: &[ShellPair3CGpu],
        prim_pairs: &[PrimPair3CGpu],
        n_aux: usize,
        width: usize,
    ) -> Option<Self> {
        Self::build_eri3c_gpu_impl(
            aux_shells,
            aux_prims,
            shell_pairs,
            prim_pairs,
            n_aux,
            width,
            false,
        )
    }

    fn build_eri3c_gpu_impl(
        aux_shells: &[AuxShellGpu],
        aux_prims: &[AuxPrimGpu],
        shell_pairs: &[ShellPair3CGpu],
        prim_pairs: &[PrimPair3CGpu],
        n_aux: usize,
        width: usize,
        copy_to_host: bool,
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
        for sh in aux_shells {
            let nf = aux_shell_n_funcs(sh.lp)?;
            if sh.ao_offset < 0
                || sh.prim_start < 0
                || sh.n_prims <= 0
                || (sh.ao_offset as usize)
                    .checked_add(nf)
                    .is_none_or(|end| end > n_aux)
                || (sh.prim_start as usize)
                    .checked_add(sh.n_prims as usize)
                    .is_none_or(|end| end > aux_prims.len())
            {
                return None;
            }
        }
        let len = n_aux.checked_mul(width)?;
        let mut data = if copy_to_host {
            vec![0.0f64; len]
        } else {
            Vec::new()
        };
        let id = Self::next_id();

        #[cfg(not(no_cuda))]
        unsafe {
            let h_b_ptr = if copy_to_host {
                data.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            };
            let ret = calchem_cuda_build_eri3c_packed(
                aux_shells.as_ptr(),
                aux_shells.len() as i32,
                aux_prims.as_ptr(),
                aux_prims.len() as i32,
                shell_pairs.as_ptr(),
                shell_pairs.len() as i32,
                prim_pairs.as_ptr(),
                prim_pairs.len() as i32,
                h_b_ptr,
                n_aux as i32,
                width as i32,
                resolve_gpu_memory_limit(),
                id,
            );
            if ret == 0 {
                return Some(Self {
                    data,
                    id,
                    device_only: !copy_to_host,
                });
            }
        }
        let _ = (data, id);
        None
    }

    pub fn is_device_only(&self) -> bool {
        self.device_only
    }

    pub fn view(&self) -> RijTensorView<'_> {
        RijTensorView {
            data: &self.data,
            id: self.id,
            device_only: self.device_only,
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
    device_only: bool,
}

impl<'a> From<&'a [f64]> for RijTensorView<'a> {
    fn from(data: &'a [f64]) -> Self {
        Self {
            data,
            id: 0,
            device_only: false,
        }
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
        if !validate_rij_contract_inputs(
            b_block,
            self.device_only,
            x_block.len(),
            j_matrix.len(),
            n_p,
            n_ao_sq,
        ) {
            return false;
        }

        #[cfg(not(no_cuda))]
        unsafe {
            let b_ptr = if self.device_only {
                std::ptr::null()
            } else {
                b_block.as_ptr()
            };
            let ret = calchem_cuda_rij_contract_coulomb_tagged(
                b_ptr,
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
        if !validate_rij_contract_inputs(
            b_block,
            self.device_only,
            v_vec.len(),
            p_matrix.len(),
            n_p,
            n_ao_sq,
        ) {
            return false;
        }

        #[cfg(not(no_cuda))]
        unsafe {
            let b_ptr = if self.device_only {
                std::ptr::null()
            } else {
                b_block.as_ptr()
            };
            let ret = calchem_cuda_rij_contract_v_tagged(
                b_ptr,
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

/// Aux-shell-chunked GPU streaming RI-J evaluator for large systems whose full packed `B`
/// tensor exceeds `max_df_bytes` or GPU VRAM headroom.
pub struct GpuRijStreamer {
    stream_id: u64,
    width: usize,
    n_aux_shells: usize,
    max_chunk_n_aux: usize,
    aux_ao_bounds: Vec<(usize, usize)>,
}

impl Drop for GpuRijStreamer {
    fn drop(&mut self) {
        if self.stream_id != 0 {
            #[cfg(not(no_cuda))]
            unsafe {
                let _ = calchem_cuda_rij_stream_release(self.stream_id);
            }
        }
    }
}

impl GpuRijStreamer {
    pub fn upload(
        aux_shells: &[AuxShellGpu],
        aux_prims: &[AuxPrimGpu],
        shell_pairs: &[ShellPair3CGpu],
        prim_pairs: &[PrimPair3CGpu],
        max_chunk_n_aux: usize,
        width: usize,
    ) -> Option<Self> {
        if !is_cuda_available()
            || aux_shells.is_empty()
            || aux_prims.is_empty()
            || shell_pairs.is_empty()
            || prim_pairs.is_empty()
            || max_chunk_n_aux == 0
            || width == 0
            || aux_shells.len() > i32::MAX as usize
            || aux_prims.len() > i32::MAX as usize
            || shell_pairs.len() > i32::MAX as usize
            || prim_pairs.len() > i32::MAX as usize
            || max_chunk_n_aux > i32::MAX as usize
            || width > i32::MAX as usize
        {
            return None;
        }
        let mut aux_ao_bounds = Vec::with_capacity(aux_shells.len());
        let mut max_ao_end = 0usize;
        let mut total_n_aux = 0usize;
        for sh in aux_shells {
            let nf = aux_shell_n_funcs(sh.lp)?;
            if sh.ao_offset < 0
                || sh.prim_start < 0
                || sh.n_prims <= 0
                || (sh.prim_start as usize)
                    .checked_add(sh.n_prims as usize)
                    .is_none_or(|end| end > aux_prims.len())
            {
                return None;
            }
            let off = sh.ao_offset as usize;
            let end = off.checked_add(nf)?;
            if end > i32::MAX as usize {
                return None;
            }
            max_ao_end = max_ao_end.max(end);
            total_n_aux = total_n_aux.checked_add(nf)?;
            aux_ao_bounds.push((off, nf));
        }
        let n_aux = total_n_aux.max(max_ao_end);
        if aux_ao_bounds
            .iter()
            .any(|&(off, nf)| off.checked_add(nf).is_none_or(|end| end > n_aux))
        {
            return None;
        }
        static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);
        let stream_id = NEXT_STREAM_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .expect("GPU RI-J stream identifier exhausted");

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_rij_stream_upload(
                aux_shells.as_ptr(),
                aux_shells.len() as i32,
                aux_prims.as_ptr(),
                aux_prims.len() as i32,
                shell_pairs.as_ptr(),
                shell_pairs.len() as i32,
                prim_pairs.as_ptr(),
                prim_pairs.len() as i32,
                max_chunk_n_aux as i32,
                width as i32,
                resolve_gpu_memory_limit(),
                stream_id,
            );
            if ret == 0 {
                return Some(Self {
                    stream_id,
                    width,
                    n_aux_shells: aux_shells.len(),
                    max_chunk_n_aux,
                    aux_ao_bounds,
                });
            }
        }
        let _ = (stream_id, aux_ao_bounds);
        None
    }

    pub fn pass1_begin(&self, p_packed: &[f64]) -> bool {
        if p_packed.len() != self.width {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass1_begin(
                self.stream_id,
                p_packed.as_ptr(),
                self.width as i32,
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            false
        }
    }

    fn validate_chunk_bounds(&self, shell_start: usize, chunk_n_shells: usize, ao_start: usize, chunk_n_aux: usize) -> bool {
        if chunk_n_shells == 0
            || chunk_n_aux == 0
            || chunk_n_aux > self.max_chunk_n_aux
            || shell_start > i32::MAX as usize
            || chunk_n_shells > i32::MAX as usize
            || ao_start > i32::MAX as usize
            || chunk_n_aux > i32::MAX as usize
        {
            return false;
        }
        let Some(shell_end) = shell_start.checked_add(chunk_n_shells) else {
            return false;
        };
        if shell_end > self.n_aux_shells {
            return false;
        }
        let Some(ao_end) = ao_start.checked_add(chunk_n_aux) else {
            return false;
        };
        if ao_end > i32::MAX as usize || ao_start != self.aux_ao_bounds[shell_start].0 {
            return false;
        }
        let mut max_end = ao_start;
        let mut sum_nf = 0usize;
        for &(off, nf) in &self.aux_ao_bounds[shell_start..shell_end] {
            let Some(end) = off.checked_add(nf) else {
                return false;
            };
            if off < ao_start || end > ao_end {
                return false;
            }
            if end > max_end {
                max_end = end;
            }
            sum_nf = match sum_nf.checked_add(nf) {
                Some(s) => s,
                None => return false,
            };
        }
        max_end - ao_start == chunk_n_aux && sum_nf == chunk_n_aux
    }

    pub fn pass1_v(
        &self,
        p_packed: &[f64],
        shell_start: usize,
        chunk_n_shells: usize,
        ao_start: usize,
        v_chunk_out: &mut [f64],
    ) -> bool {
        if p_packed.len() != self.width
            || !self.validate_chunk_bounds(shell_start, chunk_n_shells, ao_start, v_chunk_out.len())
        {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass1_v(
                self.stream_id,
                p_packed.as_ptr(),
                self.width as i32,
                shell_start as i32,
                chunk_n_shells as i32,
                ao_start as i32,
                v_chunk_out.len() as i32,
                v_chunk_out.as_mut_ptr(),
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = (shell_start, ao_start);
            false
        }
    }

    /// Evaluates Pass 1 `v_P` chunk using the density already uploaded by `pass1_begin` without re-uploading `p_packed`.
    pub fn pass1_v_resident(
        &self,
        shell_start: usize,
        chunk_n_shells: usize,
        ao_start: usize,
        v_chunk_out: &mut [f64],
    ) -> bool {
        if !self.validate_chunk_bounds(shell_start, chunk_n_shells, ao_start, v_chunk_out.len()) {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass1_v(
                self.stream_id,
                std::ptr::null(),
                self.width as i32,
                shell_start as i32,
                chunk_n_shells as i32,
                ao_start as i32,
                v_chunk_out.len() as i32,
                v_chunk_out.as_mut_ptr(),
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = (shell_start, ao_start);
            false
        }
    }

    pub fn pass2_begin(&self) -> bool {
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass2_begin(self.stream_id, self.width as i32) == 0
        }
        #[cfg(no_cuda)]
        {
            false
        }
    }

    pub fn pass2_accum(
        &self,
        x_chunk: &[f64],
        shell_start: usize,
        chunk_n_shells: usize,
        ao_start: usize,
    ) -> bool {
        if !self.validate_chunk_bounds(shell_start, chunk_n_shells, ao_start, x_chunk.len()) {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass2_accum(
                self.stream_id,
                x_chunk.as_ptr(),
                shell_start as i32,
                chunk_n_shells as i32,
                ao_start as i32,
                x_chunk.len() as i32,
                self.width as i32,
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = (shell_start, ao_start);
            false
        }
    }

    pub fn pass2_finish(&self, j_packed_out: &mut [f64]) -> bool {
        if j_packed_out.len() != self.width {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_rij_stream_pass2_finish(
                self.stream_id,
                j_packed_out.as_mut_ptr(),
                self.width as i32,
            ) == 0
        }
        #[cfg(no_cuda)]
        {
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

impl Drop for GpuVxcGrid {
    fn drop(&mut self) {
        if self.grid_id != 0 {
            #[cfg(not(no_cuda))]
            unsafe {
                let _ = calchem_cuda_vxc_release(self.grid_id);
            }
        }
    }
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
            || shells.len() > i32::MAX as usize
            || prims.len() > i32::MAX as usize
            || comps.len() > i32::MAX as usize
            || pts.len() > i32::MAX as usize
            || cand_offsets.len() > i32::MAX as usize
            || cand_shells.len() > i32::MAX as usize
            || n_ao > i32::MAX as usize
            || n_ao.checked_mul(n_ao).is_none()
            || pts.len().checked_mul(8).is_none()
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
                resolve_gpu_memory_limit(),
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
        let (Some(nn), Some(pts4)) = (self.n_ao.checked_mul(self.n_ao), self.n_pts.checked_mul(4)) else {
            return false;
        };
        if p_mat.len() != nn || rho_grad_out.len() != pts4 {
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
        let (Some(nn), Some(pts4)) = (self.n_ao.checked_mul(self.n_ao), self.n_pts.checked_mul(4)) else {
            return false;
        };
        if wv_in.len() != pts4 || vxc_out.len() != nn {
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

    pub fn eval_grad_force(
        &self,
        pa: &[f64],
        pb: &[f64],
        is_rks: bool,
        wxc_in: &[f64],
        grad_out: &mut [f64],
    ) -> bool {
        let (Some(nn), Some(pts8)) = (self.n_ao.checked_mul(self.n_ao), self.n_pts.checked_mul(8)) else {
            return false;
        };
        if !is_cuda_available()
            || pa.len() != nn
            || (!is_rks && pb.len() != nn)
            || wxc_in.len() != pts8
            || grad_out.is_empty()
            || !grad_out.len().is_multiple_of(3)
        {
            return false;
        }
        let n_atoms = grad_out.len() / 3;
        if n_atoms > i32::MAX as usize {
            return false;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_vxc_grad_force(
                self.grid_id,
                pa.as_ptr(),
                if is_rks { std::ptr::null() } else { pb.as_ptr() },
                if is_rks { 1 } else { 0 },
                wxc_in.as_ptr(),
                grad_out.as_mut_ptr(),
                self.n_pts as i32,
                self.n_ao as i32,
                n_atoms as i32,
                resolve_gpu_memory_limit(),
            ) == 0
        }
        #[cfg(no_cuda)]
        {
            let _ = n_atoms;
            false
        }
    }
}

/// GPU Direct-JK / Direct-K engine with resident VRAM 4-center quartet cache,
/// two-stage Schwarz screening, L2-cached Boys tables, and precision-bounded FP64/FP32 evaluation.
#[derive(Debug)]
pub struct GpuDirectEngine {
    direct_id: u64,
    n_ao: usize,
    n_shells: usize,
    n_pairs: usize,
}

impl Drop for GpuDirectEngine {
    fn drop(&mut self) {
        if self.direct_id != 0 {
            #[cfg(not(no_cuda))]
            unsafe {
                let _ = calchem_cuda_direct_release(self.direct_id);
            }
        }
    }
}

impl GpuDirectEngine {
    pub fn recommended_cache_bytes(max_cache_cap: usize) -> usize {
        if !is_cuda_available() || max_cache_cap == 0 {
            return 0;
        }
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_direct_recommended_cache_bytes(
                resolve_gpu_memory_limit(),
                max_cache_cap,
            )
        }
        #[cfg(no_cuda)]
        {
            0
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        shells: &[DirectShellGpu],
        pairs: &[DirectShellPairGpu],
        prims: &[DirectPrimPairGpu],
        pair_prefix: &[u64],
        n_ao: usize,
        cache_elements: usize,
        valid_slots: usize,
    ) -> Option<Self> {
        if !is_cuda_available()
            || shells.is_empty()
            || pairs.is_empty()
            || prims.is_empty()
            || pair_prefix.len() != pairs.len() + 1
            || n_ao == 0
            || shells.len() > i32::MAX as usize
            || pairs.len() > i32::MAX as usize
            || prims.len() > i32::MAX as usize
            || pair_prefix.len() > i32::MAX as usize
            || n_ao > i32::MAX as usize
            || n_ao
                .checked_mul(n_ao)
                .and_then(|nn| nn.checked_mul(4 * std::mem::size_of::<f64>()))
                .is_none()
            || shells
                .len()
                .checked_mul(shells.len())
                .and_then(|ss| ss.checked_mul(3 * std::mem::size_of::<f64>()))
                .is_none()
            || cache_elements
                .checked_mul(std::mem::size_of::<f64>())
                .is_none()
            || valid_slots
                .checked_mul(std::mem::size_of::<u8>())
                .is_none()
        {
            return None;
        }
        static NEXT_DIRECT_ID: AtomicU64 = AtomicU64::new(1);
        let direct_id = NEXT_DIRECT_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .expect("GPU Direct identifier exhausted");

        let wl_stride_override = std::env::var("CALCHEM_GPU_DIR_WL_STRIDE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(2_000_000))
            .unwrap_or(0);

        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_direct_upload(
                shells.as_ptr(),
                shells.len() as i32,
                pairs.as_ptr(),
                pairs.len() as i32,
                prims.as_ptr(),
                prims.len() as i32,
                pair_prefix.as_ptr(),
                pair_prefix.len() as i32,
                n_ao as i32,
                cache_elements,
                valid_slots,
                resolve_gpu_memory_limit(),
                wl_stride_override,
                direct_id,
            );
            if ret == 0 {
                return Some(Self {
                    direct_id,
                    n_ao,
                    n_shells: shells.len(),
                    n_pairs: pairs.len(),
                });
            }
        }
        let _ = direct_id;
        None
    }

    fn valid_ends(&self, active_pairs: usize, ends: &[i32]) -> bool {
        active_pairs > 0
            && active_pairs <= self.n_pairs
            && active_pairs <= i32::MAX as usize
            && ends.len() == active_pairs
            && ends.iter().all(|&e| e >= 0 && (e as usize) <= self.n_pairs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_rks(
        &self,
        p: &[f64],
        d_l1: &[f64],
        active_pairs: usize,
        ends: &[i32],
        exx: f64,
        thresh: f64,
        fp32_thresh: f64,
        out_g: &mut [f64],
    ) -> Option<DirectStatsGpu> {
        let nn = self.n_ao.checked_mul(self.n_ao)?;
        let ss = self.n_shells.checked_mul(self.n_shells)?;
        if p.len() != nn
            || out_g.len() != nn
            || d_l1.len() != ss
            || !self.valid_ends(active_pairs, ends)
        {
            return None;
        }
        let mut stats = DirectStatsGpu::default();
        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_direct_contract(
                self.direct_id,
                0,
                p.as_ptr(),
                std::ptr::null(),
                d_l1.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                active_pairs as i32,
                ends.as_ptr(),
                exx,
                0.0,
                0,
                thresh,
                fp32_thresh,
                out_g.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut stats,
            );
            if ret == 0 {
                return Some(stats);
            }
        }
        let _ = stats;
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_uks(
        &self,
        pa: &[f64],
        pb: &[f64],
        dt_l1: &[f64],
        da_l1: &[f64],
        db_l1: &[f64],
        active_pairs: usize,
        ends: &[i32],
        exx: f64,
        thresh: f64,
        fp32_thresh: f64,
        out_ga: &mut [f64],
        out_gb: &mut [f64],
    ) -> Option<DirectStatsGpu> {
        let nn = self.n_ao.checked_mul(self.n_ao)?;
        let ss = self.n_shells.checked_mul(self.n_shells)?;
        if pa.len() != nn
            || pb.len() != nn
            || out_ga.len() != nn
            || out_gb.len() != nn
            || dt_l1.len() != ss
            || da_l1.len() != ss
            || db_l1.len() != ss
            || !self.valid_ends(active_pairs, ends)
        {
            return None;
        }
        let mut stats = DirectStatsGpu::default();
        #[cfg(not(no_cuda))]
        unsafe {
            let ret = calchem_cuda_direct_contract(
                self.direct_id,
                3,
                pa.as_ptr(),
                pb.as_ptr(),
                dt_l1.as_ptr(),
                da_l1.as_ptr(),
                db_l1.as_ptr(),
                active_pairs as i32,
                ends.as_ptr(),
                exx,
                0.0,
                0,
                thresh,
                fp32_thresh,
                out_ga.as_mut_ptr(),
                out_gb.as_mut_ptr(),
                &mut stats,
            );
            if ret == 0 {
                return Some(stats);
            }
        }
        let _ = stats;
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_k_only(
        &self,
        p: &[f64],
        d_l1: &[f64],
        active_pairs: usize,
        ends: &[i32],
        omega: Option<f64>,
        thresh: f64,
        fp32_thresh: f64,
        out_k: &mut [f64],
    ) -> Option<DirectStatsGpu> {
        let nn = self.n_ao.checked_mul(self.n_ao)?;
        let ss = self.n_shells.checked_mul(self.n_shells)?;
        if p.len() != nn
            || out_k.len() != nn
            || d_l1.len() != ss
            || !self.valid_ends(active_pairs, ends)
        {
            return None;
        }
        let mut stats = DirectStatsGpu::default();
        #[cfg(not(no_cuda))]
        unsafe {
            let (has_omega, omega_val) = match omega {
                Some(w) => (1, w),
                None => (0, 0.0),
            };
            let ret = calchem_cuda_direct_contract(
                self.direct_id,
                1,
                p.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                d_l1.as_ptr(),
                std::ptr::null(),
                active_pairs as i32,
                ends.as_ptr(),
                1.0,
                omega_val,
                has_omega,
                thresh,
                fp32_thresh,
                out_k.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut stats,
            );
            if ret == 0 {
                return Some(stats);
            }
        }
        let _ = stats;
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_k_pair(
        &self,
        pa: &[f64],
        pb: &[f64],
        da_l1: &[f64],
        db_l1: &[f64],
        active_pairs: usize,
        ends: &[i32],
        omega: Option<f64>,
        thresh: f64,
        fp32_thresh: f64,
        out_ka: &mut [f64],
        out_kb: &mut [f64],
    ) -> Option<DirectStatsGpu> {
        let nn = self.n_ao.checked_mul(self.n_ao)?;
        let ss = self.n_shells.checked_mul(self.n_shells)?;
        if pa.len() != nn
            || pb.len() != nn
            || out_ka.len() != nn
            || out_kb.len() != nn
            || da_l1.len() != ss
            || db_l1.len() != ss
            || !self.valid_ends(active_pairs, ends)
        {
            return None;
        }
        let mut stats = DirectStatsGpu::default();
        #[cfg(not(no_cuda))]
        unsafe {
            let (has_omega, omega_val) = match omega {
                Some(w) => (1, w),
                None => (0, 0.0),
            };
            let ret = calchem_cuda_direct_contract(
                self.direct_id,
                2,
                pa.as_ptr(),
                pb.as_ptr(),
                std::ptr::null(),
                da_l1.as_ptr(),
                db_l1.as_ptr(),
                active_pairs as i32,
                ends.as_ptr(),
                1.0,
                omega_val,
                has_omega,
                thresh,
                fp32_thresh,
                out_ka.as_mut_ptr(),
                out_kb.as_mut_ptr(),
                &mut stats,
            );
            if ret == 0 {
                return Some(stats);
            }
        }
        let _ = stats;
        None
    }

    pub fn td_scratch_bytes(&self) -> usize {
        #[cfg(not(no_cuda))]
        unsafe {
            calchem_cuda_direct_td_scratch_bytes(self.direct_id)
        }
        #[cfg(no_cuda)]
        {
            0
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_tddft_batch_with_code(
        &self,
        b_sym: usize,
        b_anti: usize,
        do_j: bool,
        do_k_sym: bool,
        do_k_anti: bool,
        d_sym_packed: &[f64],
        d_anti_packed: &[f64],
        dj_env: &[f64],
        dk_env: &[f64],
        active_pairs: usize,
        ends: &[i32],
        j_weight: f64,
        k_weight: f64,
        omega: Option<f64>,
        thresh: f64,
        fp32_thresh: f64,
        out_j_packed: &mut [f64],
        out_ks_packed: &mut [f64],
        out_ka_packed: &mut [f64],
    ) -> Result<DirectStatsGpu, i32> {
        let nn = self.n_ao.checked_mul(self.n_ao).ok_or(-1)?;
        let ss = self.n_shells.checked_mul(self.n_shells).ok_or(-1)?;
        let eff_j = do_j && b_sym > 0;
        let eff_ks = do_k_sym && b_sym > 0;
        let eff_ka = do_k_anti && b_anti > 0;
        if !self.valid_ends(active_pairs, ends) {
            return Err(-1);
        }
        if !eff_j && !eff_ks && !eff_ka {
            return Ok(DirectStatsGpu::default());
        }
        let b_sym_nn = b_sym.checked_mul(nn).ok_or(-1)?;
        let b_anti_nn = b_anti.checked_mul(nn).ok_or(-1)?;
        if (eff_j || eff_ks) && d_sym_packed.len() != b_sym_nn {
            return Err(-1);
        }
        if eff_ka && d_anti_packed.len() != b_anti_nn {
            return Err(-1);
        }
        if eff_j && (dj_env.len() != ss || out_j_packed.len() != b_sym_nn) {
            return Err(-1);
        }
        if eff_ks && (dk_env.len() != ss || out_ks_packed.len() != b_sym_nn) {
            return Err(-1);
        }
        if eff_ka && (dk_env.len() != ss || out_ka_packed.len() != b_anti_nn) {
            return Err(-1);
        }
        if b_sym > i32::MAX as usize || b_anti > i32::MAX as usize {
            return Err(-1);
        }

        let mut stats = DirectStatsGpu::default();
        #[cfg(not(no_cuda))]
        unsafe {
            let (has_omega, omega_val) = match omega {
                Some(w) => (1, w),
                None => (0, 0.0),
            };
            let ret = calchem_cuda_direct_contract_tddft_batch(
                self.direct_id,
                b_sym as i32,
                b_anti as i32,
                if eff_j { 1 } else { 0 },
                if eff_ks { 1 } else { 0 },
                if eff_ka { 1 } else { 0 },
                if eff_j || eff_ks { d_sym_packed.as_ptr() } else { std::ptr::null() },
                if eff_ka { d_anti_packed.as_ptr() } else { std::ptr::null() },
                if eff_j { dj_env.as_ptr() } else { std::ptr::null() },
                if eff_ks || eff_ka { dk_env.as_ptr() } else { std::ptr::null() },
                active_pairs as i32,
                ends.as_ptr(),
                j_weight,
                k_weight,
                omega_val,
                has_omega,
                thresh,
                fp32_thresh,
                if eff_j { out_j_packed.as_mut_ptr() } else { std::ptr::null_mut() },
                if eff_ks { out_ks_packed.as_mut_ptr() } else { std::ptr::null_mut() },
                if eff_ka { out_ka_packed.as_mut_ptr() } else { std::ptr::null_mut() },
                &mut stats,
            );
            if ret == 0 {
                return Ok(stats);
            }
            Err(ret)
        }
        #[cfg(no_cuda)]
        {
            let _ = (j_weight, k_weight, omega, thresh, fp32_thresh, &mut stats);
            Err(-5)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_tddft_batch(
        &self,
        b_sym: usize,
        b_anti: usize,
        do_j: bool,
        do_k_sym: bool,
        do_k_anti: bool,
        d_sym_packed: &[f64],
        d_anti_packed: &[f64],
        dj_env: &[f64],
        dk_env: &[f64],
        active_pairs: usize,
        ends: &[i32],
        j_weight: f64,
        k_weight: f64,
        omega: Option<f64>,
        thresh: f64,
        fp32_thresh: f64,
        out_j_packed: &mut [f64],
        out_ks_packed: &mut [f64],
        out_ka_packed: &mut [f64],
    ) -> Option<DirectStatsGpu> {
        self.contract_tddft_batch_with_code(
            b_sym,
            b_anti,
            do_j,
            do_k_sym,
            do_k_anti,
            d_sym_packed,
            d_anti_packed,
            dj_env,
            dk_env,
            active_pairs,
            ends,
            j_weight,
            k_weight,
            omega,
            thresh,
            fp32_thresh,
            out_j_packed,
            out_ks_packed,
            out_ka_packed,
        )
        .ok()
    }
}

/// Evaluates `v_P = sum_{mu,nu} (P|mu nu) P_{mu,nu}` on GPU without allocating packed `B`.
#[allow(clippy::too_many_arguments)]
pub fn cuda_rij_grad_v(
    aux_shells: &[Grad3cAuxShellGpu],
    aux_prims: &[Grad3cAuxPrimGpu],
    shell_pairs: &[Grad3cShellPairGpu],
    prim_pairs: &[Grad3cPrimPairGpu],
    p_mat: &[f64],
    v_out: &mut [f64],
    n_ao: usize,
) -> bool {
    let Some(nn) = n_ao.checked_mul(n_ao) else {
        return false;
    };
    if !is_cuda_available()
        || aux_shells.is_empty()
        || aux_prims.is_empty()
        || shell_pairs.is_empty()
        || prim_pairs.is_empty()
        || n_ao == 0
        || aux_shells.len() > i32::MAX as usize
        || aux_prims.len() > i32::MAX as usize
        || shell_pairs.len() > i32::MAX as usize
        || prim_pairs.len() > i32::MAX as usize
        || v_out.len() > i32::MAX as usize
        || n_ao > i32::MAX as usize
        || p_mat.len() != nn
        || v_out.is_empty()
    {
        return false;
    }
    #[cfg(not(no_cuda))]
    unsafe {
        calchem_cuda_rij_grad_v(
            aux_shells.as_ptr(),
            aux_shells.len() as i32,
            aux_prims.as_ptr(),
            aux_prims.len() as i32,
            shell_pairs.as_ptr(),
            shell_pairs.len() as i32,
            prim_pairs.as_ptr(),
            prim_pairs.len() as i32,
            p_mat.as_ptr(),
            v_out.as_mut_ptr(),
            v_out.len() as i32,
            n_ao as i32,
            resolve_gpu_memory_limit(),
        ) == 0
    }
    #[cfg(no_cuda)]
    {
        false
    }
}

/// Evaluates the 3-center RI-J nuclear gradient `sum_{P,mu,nu} x_P P_{mu,nu} d(P|mu nu)/dR` on GPU.
#[allow(clippy::too_many_arguments)]
pub fn cuda_rij_grad_3c(
    aux_shells: &[Grad3cAuxShellGpu],
    aux_prims: &[Grad3cAuxPrimGpu],
    shell_pairs: &[Grad3cShellPairGpu],
    prim_pairs: &[Grad3cPrimPairGpu],
    x_vec: &[f64],
    p_mat: &[f64],
    grad_out: &mut [f64],
    n_ao: usize,
) -> bool {
    let Some(nn) = n_ao.checked_mul(n_ao) else {
        return false;
    };
    if !is_cuda_available()
        || aux_shells.is_empty()
        || aux_prims.is_empty()
        || shell_pairs.is_empty()
        || prim_pairs.is_empty()
        || n_ao == 0
        || aux_shells.len() > i32::MAX as usize
        || aux_prims.len() > i32::MAX as usize
        || shell_pairs.len() > i32::MAX as usize
        || prim_pairs.len() > i32::MAX as usize
        || x_vec.len() > i32::MAX as usize
        || n_ao > i32::MAX as usize
        || p_mat.len() != nn
        || x_vec.is_empty()
        || grad_out.is_empty()
        || !grad_out.len().is_multiple_of(3)
    {
        return false;
    }
    let n_atoms = grad_out.len() / 3;
    if n_atoms > i32::MAX as usize {
        return false;
    }
    #[cfg(not(no_cuda))]
    unsafe {
        calchem_cuda_rij_grad_3c(
            aux_shells.as_ptr(),
            aux_shells.len() as i32,
            aux_prims.as_ptr(),
            aux_prims.len() as i32,
            shell_pairs.as_ptr(),
            shell_pairs.len() as i32,
            prim_pairs.as_ptr(),
            prim_pairs.len() as i32,
            x_vec.as_ptr(),
            p_mat.as_ptr(),
            grad_out.as_mut_ptr(),
            x_vec.len() as i32,
            n_ao as i32,
            n_atoms as i32,
            resolve_gpu_memory_limit(),
        ) == 0
    }
    #[cfg(no_cuda)]
    {
        let _ = n_atoms;
        false
    }
}

/// Evaluates and contracts 4-center ERI nuclear derivatives in FP64 on CUDA GPU.
#[allow(clippy::too_many_arguments)]
pub fn cuda_grad4c_contract(
    shells: &[Grad4cShellGpu],
    pairs: &[Grad4cShellPairGpu],
    prims: &[Grad4cPrimPairGpu],
    pt: &[f64],
    pa: &[f64],
    pb: &[f64],
    d_tot_max: &[f64],
    d_spin_max: &[f64],
    ends: &[i32],
    n_ao: usize,
    do_coul: bool,
    exx: f64,
    omega: Option<f64>,
    thresh: f64,
    grad_out: &mut [f64],
) -> bool {
    let (Some(nn), Some(ss)) = (n_ao.checked_mul(n_ao), shells.len().checked_mul(shells.len())) else {
        return false;
    };
    if !is_cuda_available()
        || shells.is_empty()
        || pairs.is_empty()
        || prims.is_empty()
        || n_ao == 0
        || shells.len() > i32::MAX as usize
        || pairs.len() > i32::MAX as usize
        || prims.len() > i32::MAX as usize
        || n_ao > i32::MAX as usize
        || pt.len() != nn
        || pa.len() != nn
        || pb.len() != nn
        || d_tot_max.len() != ss
        || d_spin_max.len() != ss
        || ends.len() != pairs.len()
        || ends.iter().any(|&e| e < 0 || (e as usize) > pairs.len())
        || grad_out.is_empty()
        || !grad_out.len().is_multiple_of(3)
    {
        return false;
    }
    let n_atoms = grad_out.len() / 3;
    if n_atoms > i32::MAX as usize {
        return false;
    }
    let (has_omega, omega_val) = match omega {
        Some(w) => (1, w),
        None => (0, 0.0),
    };
    #[cfg(not(no_cuda))]
    unsafe {
        calchem_cuda_grad4c_contract(
            shells.as_ptr(),
            shells.len() as i32,
            pairs.as_ptr(),
            pairs.len() as i32,
            prims.as_ptr(),
            prims.len() as i32,
            pt.as_ptr(),
            pa.as_ptr(),
            pb.as_ptr(),
            d_tot_max.as_ptr(),
            d_spin_max.as_ptr(),
            ends.as_ptr(),
            n_ao as i32,
            n_atoms as i32,
            if do_coul { 1 } else { 0 },
            exx,
            omega_val,
            has_omega,
            thresh,
            grad_out.as_mut_ptr(),
            resolve_gpu_memory_limit(),
        ) == 0
    }
    #[cfg(no_cuda)]
    {
        let _ = (do_coul, exx, omega_val, has_omega, thresh, n_atoms);
        false
    }
}
