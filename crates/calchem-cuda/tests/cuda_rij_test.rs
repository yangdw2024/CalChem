use calchem_cuda::{cuda_rij_contract_coulomb, get_cuda_memory_info, is_cuda_available, set_gpu_enabled};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_cuda_device_detection() {
    let _guard = TEST_LOCK.lock().unwrap();
    let available = is_cuda_available();
    println!("CUDA is available: {}", available);
    if available {
        let (free, total) = get_cuda_memory_info().expect("Failed to get CUDA memory info");
        println!(
            "CUDA device memory: {:.2} GB free / {:.2} GB total",
            free as f64 / 1e9,
            total as f64 / 1e9
        );
        assert!(total > 0, "Total VRAM should be greater than 0");
        assert!(free <= total, "Free VRAM cannot exceed total VRAM");
    }
}

#[test]
fn test_cuda_rij_contraction_accuracy() {
    let _guard = TEST_LOCK.lock().unwrap();
    if !is_cuda_available() {
        println!("Skipping CUDA contraction test: device unavailable");
        return;
    }

    let n_p = 128;
    let n_ao = 32;
    let n_ao_sq = n_ao * n_ao;

    // Generate deterministic test tensors
    let mut b_block = vec![0.0f64; n_p * n_ao_sq];
    let mut x_block = vec![0.0f64; n_p];

    for p in 0..n_p {
        x_block[p] = ((p as f64) * 0.123).sin();
        for i in 0..n_ao_sq {
            b_block[p * n_ao_sq + i] = ((p * 37 + i * 13) as f64 * 0.05).cos();
        }
    }

    // Reference CPU contraction
    let mut j_ref = vec![0.0f64; n_ao_sq];
    for p in 0..n_p {
        let xp = x_block[p];
        for i in 0..n_ao_sq {
            j_ref[i] += xp * b_block[p * n_ao_sq + i];
        }
    }

    // CUDA contraction
    let mut j_cuda = vec![0.0f64; n_ao_sq];
    let ok = cuda_rij_contract_coulomb(&b_block, &x_block, &mut j_cuda, n_p, n_ao_sq);
    assert!(ok, "CUDA contraction should succeed");

    // Check numerical agreement
    let mut max_err = 0.0f64;
    for i in 0..n_ao_sq {
        let err = (j_ref[i] - j_cuda[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("CUDA vs CPU max absolute error: {:.3e}", max_err);
    assert!(
        max_err < 1e-12,
        "CUDA contraction must agree with CPU reference to < 1e-12, got {:.3e}",
        max_err
    );
}

#[test]
fn test_cuda_graceful_cpu_fallback() {
    let _guard = TEST_LOCK.lock().unwrap();
    set_gpu_enabled(false);
    assert!(!is_cuda_available(), "Should report false when disabled");

    let mut j_dummy = vec![0.0; 10];
    let b_dummy = vec![0.0; 10];
    let x_dummy = vec![0.0; 1];
    let ok = cuda_rij_contract_coulomb(&b_dummy, &x_dummy, &mut j_dummy, 1, 10);
    assert!(!ok, "Should safely return false indicating CPU fallback");

    // Restore
    set_gpu_enabled(true);
}
