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
}

/// Hard safety ceiling on device memory allocation (5.0 GiB) to guarantee stability on 8 GB cards.
pub const MAX_GPU_MEMORY_BYTES: usize = 5 * 1024 * 1024 * 1024;

static GPU_ENABLED: AtomicBool = AtomicBool::new(true);

/// Checks whether a functional CUDA device is available and accessible.
pub fn is_cuda_available() -> bool {
    if !GPU_ENABLED.load(Ordering::Relaxed) {
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
        return Err(CalcError::UnsupportedMethod("CUDA device unavailable".into()));
    }
    #[cfg(not(no_cuda))]
    unsafe {
        let mut free = 0usize;
        let mut total = 0usize;
        let ret = calchem_cuda_get_memory_info(&mut free, &mut total);
        if ret == 0 {
            Ok((free, total))
        } else {
            Err(CalcError::Internal(format!("cudaMemGetInfo failed with code {ret}")))
        }
    }
    #[cfg(no_cuda)]
    {
        Err(CalcError::UnsupportedMethod("CUDA not compiled into binary".into()))
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
    if !is_cuda_available() {
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
            MAX_GPU_MEMORY_BYTES,
        );
        ret == 0
    }
    #[cfg(no_cuda)]
    {
        let _ = (b_block, x_block, j_matrix, n_p, n_ao_sq);
        false
    }
}
