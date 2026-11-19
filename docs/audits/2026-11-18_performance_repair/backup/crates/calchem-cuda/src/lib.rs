use calchem_model::CalcError;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(no_cuda))]
extern "C" {
    fn calchem_cuda_is_available() -> i32;
    fn calchem_cuda_get_memory_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
    fn calchem_cuda_rij_contract_coulomb(
        h_b_block: *const f64,
        h_x_block: *const f64,
        h_j: *mut f64,
        n_p: i32,
        n_ao_sq: i32,
        memory_limit_bytes: usize,
    ) -> i32;
    fn calchem_cuda_rij_contract_v(
        h_b_block: *const f64,
        h_p_block: *const f64,
        h_v: *mut f64,
        n_p: i32,
        n_ao_sq: i32,
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
    if !validate_rij_contract_inputs(b_block, x_block.len(), j_matrix.len(), n_p, n_ao_sq) {
        return false;
    }

    #[cfg(not(no_cuda))]
    unsafe {
        let ret = calchem_cuda_rij_contract_coulomb(
            b_block.as_ptr(),
            x_block.as_ptr(),
            j_matrix.as_mut_ptr(),
            n_p as i32,
            n_ao_sq as i32,
            resolve_gpu_memory_limit(),
        );
        ret == 0
    }
    #[cfg(no_cuda)]
    {
        let _ = (b_block, x_block, j_matrix, n_p, n_ao_sq);
        false
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
    if !validate_rij_contract_inputs(b_block, v_vec.len(), p_matrix.len(), n_p, n_ao_sq) {
        return false;
    }

    #[cfg(not(no_cuda))]
    unsafe {
        let ret = calchem_cuda_rij_contract_v(
            b_block.as_ptr(),
            p_matrix.as_ptr(),
            v_vec.as_mut_ptr(),
            n_p as i32,
            n_ao_sq as i32,
            resolve_gpu_memory_limit(),
        );
        ret == 0
    }
    #[cfg(no_cuda)]
    {
        let _ = (b_block, p_matrix, v_vec, n_p, n_ao_sq);
        false
    }
}
