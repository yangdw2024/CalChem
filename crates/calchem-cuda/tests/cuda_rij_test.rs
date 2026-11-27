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

#[test]
fn test_gpu_rij_streamer_pass1_v_nonzero_shell_start_uploads_p_safely() {
    use calchem_cuda::{AuxPrimGpu, AuxShellGpu, GpuRijStreamer, PrimPair3CGpu, ShellPair3CGpu};

    let _guard = TEST_LOCK.lock().unwrap();
    if !is_cuda_available() {
        return;
    }

    // Two s-type auxiliary shells (shell 0 and shell 1) with exponent 1.0 and 0.5
    let aux_shells = [
        AuxShellGpu {
            center: [0.0, 0.0, 0.0],
            bound: 1.0,
            lp: 0,
            ao_offset: 0,
            prim_start: 0,
            n_prims: 1,
        },
        AuxShellGpu {
            center: [0.0, 0.0, 1.5],
            bound: 1.0,
            lp: 0,
            ao_offset: 1,
            prim_start: 1,
            n_prims: 1,
        },
    ];
    let mut cp0 = [0.0; 10];
    cp0[0] = 1.0;
    let aux_prims = [
        AuxPrimGpu { gamma_p: 1.0, cp_norm: cp0 },
        AuxPrimGpu { gamma_p: 0.5, cp_norm: cp0 },
    ];
    let shell_pairs = [ShellPair3CGpu {
        q_ij: 1.0,
        li: 0,
        lj: 0,
        same_shell: 1,
        col_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let mut ci = [0.0; 6];
    ci[0] = 1.0;
    let prim_pairs = [PrimPair3CGpu {
        q: 1.0,
        p_center: [0.0, 0.0, 0.0],
        qc: [0.0, 0.0, 0.0],
        qd: [0.0, 0.0, 0.0],
        two_pi25_kab_over_q: 34.986836655249725,
        ci_norm: ci,
        cj_norm: ci,
    }];

    let streamer = GpuRijStreamer::upload(&aux_shells, &aux_prims, &shell_pairs, &prim_pairs, 1, 1)
        .expect("Streamer upload should succeed");

    // First call starts at shell_start == 1, ao_start == 1 (skipping shell_start == 0) WITHOUT pass1_begin
    let mut v1_direct = [0.0f64; 1];
    assert!(streamer.pass1_v(&[2.5], 1, 1, 1, &mut v1_direct));
    assert!(v1_direct[0].is_finite() && v1_direct[0] > 0.0);

    // Closed-form (s | s s) analytical reference for shell 1:
    // p = 0.5, q = 1.0, zeta = 1.5, rho = 1/3, pq2 = 2.25, t = 0.75
    let boys_t075 = boys_batch_ref(0, 0.75)[0];
    let v1_ref = 2.5 * (34.986836655249725 / (0.5 * 1.5_f64.sqrt())) * boys_t075;
    assert!(
        (v1_direct[0] - v1_ref).abs() < 1e-12,
        "Expected exact lp=0 analytical match: gpu={} ref={}",
        v1_direct[0],
        v1_ref
    );

    // Compare against calling with a different density (must NOT read stale d_stream_p!)
    let mut v1_scaled = [0.0f64; 1];
    assert!(streamer.pass1_v(&[5.0], 1, 1, 1, &mut v1_scaled));
    assert!(
        (v1_scaled[0] - 2.0 * v1_direct[0]).abs() < 1e-12,
        "Expected exact scaling when calling pass1_v with shell_start!=0: {} vs 2*{}",
        v1_scaled[0],
        v1_direct[0]
    );
}

fn boys_batch_ref(max_m: usize, t: f64) -> Vec<f64> {
    let mut out = vec![0.0; max_m + 1];
    if t < 1e-10 {
        for (m, v) in out.iter_mut().enumerate() {
            let dm = (2 * m) as f64;
            *v = 1.0 / (dm + 1.0) - t / (dm + 3.0) + 0.5 * t * t / (dm + 5.0);
        }
        return out;
    }
    let exp_mt = (-t).exp();
    let two_t = 2.0 * t;
    let m_top = max_m + 40;
    let mut fm = 0.0;
    for m in (max_m + 1..=m_top).rev() {
        fm = (two_t * fm + exp_mt) / (2 * m - 1) as f64;
    }
    out[max_m] = fm;
    for m in (1..=max_m).rev() {
        fm = (two_t * fm + exp_mt) / (2 * m - 1) as f64;
        out[m - 1] = fm;
    }
    out
}

#[test]
fn test_gpu_rij_streamer_pass1_v_lp3_matches_cpu_reference() {
    use calchem_cuda::{AuxPrimGpu, AuxShellGpu, GpuRijStreamer, PrimPair3CGpu, ShellPair3CGpu};

    let _guard = TEST_LOCK.lock().unwrap();
    if !is_cuda_available() {
        return;
    }

    let r_p = [0.2, 0.7, -0.6];
    let p_exp = 0.85;
    let cp_f = [0.4, -0.3, 0.2, 0.5, -0.6, 0.7, 0.1, -0.2, 0.35, 0.8];
    let aux_shells = [
        AuxShellGpu {
            center: [0.0, 0.0, 0.0],
            bound: 1.0,
            lp: 0,
            ao_offset: 0,
            prim_start: 0,
            n_prims: 1,
        },
        AuxShellGpu {
            center: r_p,
            bound: 1.0,
            lp: 3,
            ao_offset: 1,
            prim_start: 1,
            n_prims: 1,
        },
    ];
    let mut cp0 = [0.0; 10];
    cp0[0] = 1.0;
    let aux_prims = [
        AuxPrimGpu { gamma_p: 1.0, cp_norm: cp0 },
        AuxPrimGpu { gamma_p: p_exp, cp_norm: cp_f },
    ];
    let shell_pairs = [ShellPair3CGpu {
        q_ij: 1.0,
        li: 0,
        lj: 0,
        same_shell: 1,
        col_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let p_center = [0.3, -0.4, 0.5];
    let q_exp = 1.2;
    let two_pi25_kab_over_q = 29.155697212708104;
    let mut ci = [0.0; 6];
    ci[0] = 0.9;
    let mut cj = [0.0; 6];
    cj[0] = 1.1;
    let prim_pairs = [PrimPair3CGpu {
        q: q_exp,
        p_center,
        qc: [0.0, 0.0, 0.0],
        qd: [0.0, 0.0, 0.0],
        two_pi25_kab_over_q,
        ci_norm: ci,
        cj_norm: cj,
    }];

    let streamer = GpuRijStreamer::upload(&aux_shells, &aux_prims, &shell_pairs, &prim_pairs, 10, 1)
        .expect("Streamer upload for lp=3 should succeed");

    let p_val = 2.5;
    let mut v_gpu = [0.0f64; 10];
    assert!(streamer.pass1_v(&[p_val], 1, 1, 1, &mut v_gpu));

    // Self-contained closed-form (f | s s) Obara-Saika VRR CPU reference in COMPS_3C[10..20] order:
    // [(3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2), (0,3,0), (0,2,1), (0,1,2), (0,0,3)]
    let zeta = p_exp + q_exp;
    let rho = p_exp * q_exp / zeta;
    let inv_2p = 0.5 / p_exp;
    let rho_over_p = rho / p_exp;
    let w = [
        (p_exp * r_p[0] + q_exp * p_center[0]) / zeta,
        (p_exp * r_p[1] + q_exp * p_center[1]) / zeta,
        (p_exp * r_p[2] + q_exp * p_center[2]) / zeta,
    ];
    let wp = [w[0] - r_p[0], w[1] - r_p[1], w[2] - r_p[2]];
    let pq2 = (r_p[0] - p_center[0]).powi(2)
        + (r_p[1] - p_center[1]).powi(2)
        + (r_p[2] - p_center[2]).powi(2);
    let boys = boys_batch_ref(3, rho * pq2);
    let pref = two_pi25_kab_over_q / (p_exp * zeta.sqrt());
    let v0: [f64; 4] = [pref * boys[0], pref * boys[1], pref * boys[2], pref * boys[3]];

    let mut v1 = [[0.0f64; 3]; 3];
    for u in 0..3 {
        for m in 0..3 {
            v1[u][m] = wp[u] * v0[m + 1];
        }
    }
    let pairs_d = [(0usize, 0usize), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let mut v2 = [[0.0f64; 2]; 6];
    for (kl, &(u, v)) in pairs_d.iter().enumerate() {
        for m in 0..2 {
            let mut val = wp[u] * v1[v][m + 1];
            if u == v {
                val += inv_2p * (v0[m] - rho_over_p * v0[m + 1]);
            }
            v2[kl][m] = val;
        }
    }
    // COMPS_3C[10..20]:
    let v3 = [
        wp[0] * v2[0][1] + 2.0 * inv_2p * (v1[0][0] - rho_over_p * v1[0][1]), // (3,0,0)
        wp[0] * v2[1][1] + 1.0 * inv_2p * (v1[1][0] - rho_over_p * v1[1][1]), // (2,1,0)
        wp[2] * v2[0][1],                                                     // (2,0,1)
        wp[0] * v2[3][1],                                                     // (1,2,0)
        wp[0] * v2[4][1],                                                     // (1,1,1)
        wp[0] * v2[5][1],                                                     // (1,0,2)
        wp[1] * v2[3][1] + 2.0 * inv_2p * (v1[1][0] - rho_over_p * v1[1][1]), // (0,3,0)
        wp[2] * v2[3][1],                                                     // (0,2,1)
        wp[1] * v2[5][1],                                                     // (0,1,2)
        wp[2] * v2[5][1] + 2.0 * inv_2p * (v1[2][0] - rho_over_p * v1[2][1]), // (0,0,3)
    ];

    for pf in 0..10 {
        let v_cpu = p_val * cp_f[pf] * ci[0] * cj[0] * v3[pf];
        assert!(
            (v_gpu[pf] - v_cpu).abs() < 1e-12,
            "lp=3 component {pf} mismatch: gpu={} cpu={}",
            v_gpu[pf],
            v_cpu
        );
    }

    let mut v_gpu_2x = [0.0f64; 10];
    assert!(streamer.pass1_v(&[2.0 * p_val], 1, 1, 1, &mut v_gpu_2x));
    for pf in 0..10 {
        assert!(
            (v_gpu_2x[pf] - 2.0 * v_gpu[pf]).abs() < 1e-12,
            "lp=3 component {pf} 2x scaling mismatch: {} vs 2*{}",
            v_gpu_2x[pf],
            v_gpu[pf]
        );
    }
}

struct CudaEnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl CudaEnvGuard {
    fn new(keys: &[&'static str]) -> Self {
        let saved = keys.iter().map(|&k| (k, std::env::var(k).ok())).collect();
        Self { saved }
    }
}

impl Drop for CudaEnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.saved {
            match v {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }
}

#[test]
fn test_resolve_gpu_memory_limit_clamps_and_handles_invalid_values() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _env = CudaEnvGuard::new(&["CALCHEM_MAX_GPU_MEM_MB"]);

    std::env::remove_var("CALCHEM_MAX_GPU_MEM_MB");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), calchem_cuda::MAX_GPU_MEMORY_BYTES);

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "999999");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), calchem_cuda::MAX_GPU_MEMORY_BYTES);

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "18446744073709551615");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), calchem_cuda::MAX_GPU_MEMORY_BYTES);

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "not_a_number");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), calchem_cuda::MAX_GPU_MEMORY_BYTES);

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "128");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), 128 * 1024 * 1024);

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "0");
    assert_eq!(calchem_cuda::resolve_gpu_memory_limit(), 0);
}

#[test]
fn test_gpu_rij_streamer_pass1_v_after_pass1_begin_updates_density() {
    use calchem_cuda::{AuxPrimGpu, AuxShellGpu, GpuRijStreamer, PrimPair3CGpu, ShellPair3CGpu};

    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !is_cuda_available() {
        return;
    }

    let aux_shells = [AuxShellGpu {
        center: [0.0, 0.0, 1.5],
        bound: 1.0,
        lp: 0,
        ao_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let mut cp0 = [0.0; 10];
    cp0[0] = 1.0;
    let aux_prims = [AuxPrimGpu { gamma_p: 0.5, cp_norm: cp0 }];
    let shell_pairs = [ShellPair3CGpu {
        q_ij: 1.0,
        li: 0,
        lj: 0,
        same_shell: 1,
        col_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let mut ci = [0.0; 6];
    ci[0] = 1.0;
    let prim_pairs = [PrimPair3CGpu {
        q: 1.0,
        p_center: [0.0, 0.0, 0.0],
        qc: [0.0, 0.0, 0.0],
        qd: [0.0, 0.0, 0.0],
        two_pi25_kab_over_q: 34.986836655249725,
        ci_norm: ci,
        cj_norm: ci,
    }];

    let streamer = GpuRijStreamer::upload(&aux_shells, &aux_prims, &shell_pairs, &prim_pairs, 1, 1)
        .expect("Streamer upload should succeed");

    assert!(streamer.pass1_begin(&[2.5]));
    let mut v1 = [0.0f64; 1];
    assert!(streamer.pass1_v_resident(0, 1, 0, &mut v1));

    // Subsequent call to pass1_v with a DIFFERENT density WITHOUT calling pass1_begin first
    // MUST upload the new density rather than reading stale d_p (catches HIGH-01).
    let mut v2 = [0.0f64; 1];
    assert!(streamer.pass1_v(&[5.0], 0, 1, 0, &mut v2));
    assert!(
        (v2[0] - 2.0 * v1[0]).abs() < 1e-12,
        "pass1_v read stale d_p after pass1_begin: v1={} v2={}",
        v1[0],
        v2[0]
    );

    // Out-of-bounds shell_start / chunk_n_shells must be rejected cleanly in Rust
    assert!(!streamer.pass1_v(&[5.0], 1, 1, 0, &mut v2));
    assert!(!streamer.pass1_v_resident(1, 1, 0, &mut v2));
    assert!(!streamer.pass2_accum(&[1.0], 1, 1, 0));
}

#[test]
fn test_gpu_rij_streamer_pass2_lp3_and_dd_parity() {
    use calchem_cuda::{
        AuxPrimGpu, AuxShellGpu, GpuRijStreamer, PrimPair3CGpu, ResidentRijTensor, ShellPair3CGpu,
    };

    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !is_cuda_available() {
        return;
    }

    let mut cp_f = [0.0; 10];
    cp_f[0] = 0.5;
    cp_f[4] = -0.35;
    cp_f[9] = 0.7;
    let aux_shells = [AuxShellGpu {
        center: [0.2, -0.3, 0.6],
        bound: 1.0,
        lp: 3,
        ao_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let aux_prims = [AuxPrimGpu { gamma_p: 0.9, cp_norm: cp_f }];
    let shell_pairs = [ShellPair3CGpu {
        q_ij: 1.0,
        li: 2,
        lj: 2,
        same_shell: 0,
        col_offset: 0,
        prim_start: 0,
        n_prims: 1,
    }];
    let ci = [0.4, 0.7, -0.3, 0.5, 0.2, 0.6];
    let cj = [0.3, -0.5, 0.4, 0.25, -0.45, 0.55];
    let prim_pairs = [PrimPair3CGpu {
        q: 1.4,
        p_center: [0.1, 0.2, -0.1],
        qc: [0.25, -0.15, 0.05],
        qd: [-0.2, 0.3, -0.1],
        two_pi25_kab_over_q: 25.0,
        ci_norm: ci,
        cj_norm: cj,
    }];

    let width = 36;
    let n_aux = 10;
    let resident = ResidentRijTensor::build_eri3c_gpu(
        &aux_shells,
        &aux_prims,
        &shell_pairs,
        &prim_pairs,
        n_aux,
        width,
    )
    .expect("build_eri3c_gpu for (f|dd) should succeed");
    let b = resident.as_slice();
    assert!(b.iter().all(|v| v.is_finite()));

    let mut p_vec = vec![0.0f64; width];
    for (k, val) in p_vec.iter_mut().enumerate() {
        *val = ((k as f64 + 1.0) * 0.17).sin();
    }

    let streamer = GpuRijStreamer::upload(&aux_shells, &aux_prims, &shell_pairs, &prim_pairs, n_aux, width)
        .expect("GpuRijStreamer upload for (f|dd) should succeed");

    let mut v_stream = vec![0.0f64; n_aux];
    assert!(streamer.pass1_begin(&p_vec));
    assert!(streamer.pass1_v_resident(0, 1, 0, &mut v_stream));

    let mut x_vec = vec![0.0f64; n_aux];
    for p in 0..n_aux {
        x_vec[p] = v_stream[p] * 0.1 + 0.05 * ((p as f64) + 1.0);
    }
    let mut j_stream = vec![0.0f64; width];
    assert!(streamer.pass2_begin());
    assert!(streamer.pass2_accum(&x_vec, 0, 1, 0));
    assert!(streamer.pass2_finish(&mut j_stream));

    for idx in 0..width {
        let mut j_ref = 0.0f64;
        for p in 0..n_aux {
            j_ref += b[p * width + idx] * x_vec[p];
        }
        assert!(
            (j_stream[idx] - j_ref).abs() < 1e-12,
            "(f|dd) pass2 mismatch at idx={idx}: stream={} ref={}",
            j_stream[idx],
            j_ref
        );
    }
}

#[test]
fn test_cuda_wrappers_reject_overflows_and_tight_vram_safely() {
    use calchem_cuda::{
        cuda_grad4c_contract, cuda_rij_grad_3c, cuda_rij_grad_v, AoCompGpu, AoPrimGpu, AoShellGpu,
        Grad3cAuxPrimGpu, Grad3cAuxShellGpu, Grad3cPrimPairGpu, Grad3cShellPairGpu,
        Grad4cPrimPairGpu, Grad4cShellGpu, Grad4cShellPairGpu, GpuVxcGrid, GridPtGpu,
    };

    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _env = CudaEnvGuard::new(&["CALCHEM_MAX_GPU_MEM_MB"]);

    let mut v_out = [0.0f64; 1];
    assert!(!cuda_rij_grad_v(&[], &[], &[], &[], &[], &mut v_out, 0));
    let mut g_out = [0.0f64; 2];
    assert!(!cuda_rij_grad_3c(&[], &[], &[], &[], &[1.0], &[1.0], &mut g_out, 1));
    assert!(!cuda_grad4c_contract(
        &[], &[], &[], &[1.0], &[1.0], &[1.0], &[], &[], &[], 1, true, 0.2, None, 1e-12, &mut g_out
    ));

    if !is_cuda_available() {
        return;
    }

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "0");

    let ash = [Grad3cAuxShellGpu {
        center: [0.0, 0.0, 0.0],
        bound: 1.0,
        comp_scales: [1.0; 10],
        lp: 0,
        ao_offset: 0,
        prim_start: 0,
        n_prims: 1,
        atom_index: 0,
        _pad: 0,
    }];
    let apr = [Grad3cAuxPrimGpu { gamma_p: 1.0, cp_lead: 1.0 }];
    let sp = [Grad3cShellPairGpu {
        q_ij: 1.0,
        ab_vec: [0.0, 0.0, 0.0],
        comp_scales_i: [1.0; 6],
        comp_scales_j: [1.0; 6],
        li: 0,
        lj: 0,
        same_shell: 1,
        atom_i: 0,
        atom_j: 0,
        ao_off_i: 0,
        ao_off_j: 0,
        prim_start: 0,
        n_prims: 1,
        _pad: 0,
    }];
    let pp = [Grad3cPrimPairGpu {
        q: 1.0,
        alpha: 0.5,
        beta: 0.5,
        two_pi25_kab_w_over_q: 1.0,
        p_center: [0.0, 0.0, 0.0],
        qc: [0.0, 0.0, 0.0],
    }];
    assert!(!cuda_rij_grad_v(&ash, &apr, &sp, &pp, &[1.0], &mut v_out, 1));
    let mut g3 = [0.0f64; 3];
    assert!(!cuda_rij_grad_3c(&ash, &apr, &sp, &pp, &[1.0], &[1.0], &mut g3, 1));

    let g4sh = [Grad4cShellGpu {
        center: [0.0, 0.0, 0.0],
        comp_scales: [1.0; 10],
        l: 0,
        ao_offset: 0,
        n_funcs: 1,
        atom_index: 0,
    }];
    let g4sp = [Grad4cShellPairGpu {
        shell_i: 0,
        shell_j: 0,
        bound: 1.0,
        prim_pair_start: 0,
        n_prim_pairs: 1,
    }];
    let g4pp = [Grad4cPrimPairGpu {
        gamma: 1.0,
        kab_w: 1.0,
        alpha: 0.5,
        beta: 0.5,
        p: [0.0, 0.0, 0.0],
        pa: [0.0, 0.0, 0.0],
    }];
    assert!(!cuda_grad4c_contract(
        &g4sh, &g4sp, &g4pp, &[1.0], &[0.5], &[0.5], &[1.0], &[0.5], &[1], 1, true, 0.2, None, 1e-12, &mut g3
    ));
    // Also reject out-of-range ends even with normal VRAM limit
    std::env::remove_var("CALCHEM_MAX_GPU_MEM_MB");
    assert!(!cuda_grad4c_contract(
        &g4sh, &g4sp, &g4pp, &[1.0], &[0.5], &[0.5], &[1.0], &[0.5], &[2], 1, true, 0.2, None, 1e-12, &mut g3
    ));

    let dsh = [calchem_cuda::DirectShellGpu {
        center: [0.0; 3],
        comp_scales: [1.0; 10],
        l: 0,
        ao_offset: 0,
        n_funcs: 1,
        _pad: 0,
    }];
    let dsp = [calchem_cuda::DirectShellPairGpu {
        shell_i: 0,
        shell_j: 0,
        bound: 1.0,
        prim_pair_start: 0,
        n_prim_pairs: 1,
        cached_kets: 0,
        cache_valid_offset: 0,
        cache_data_offset: 0,
    }];
    let dpp = [calchem_cuda::DirectPrimPairGpu {
        gamma: 1.0,
        kab_w: 1.0,
        p: [0.0; 3],
        pa: [0.0; 3],
    }];
    assert!(
        calchem_cuda::GpuDirectEngine::upload(&dsh, &dsp, &dpp, &[0, 1], 1, usize::MAX / 8 + 1, 0)
            .is_none()
    );

    std::env::set_var("CALCHEM_MAX_GPU_MEM_MB", "0");
    let vxc_none = GpuVxcGrid::upload(
        &[AoShellGpu {
            center: [0.0, 0.0, 0.0],
            r2_cutoff: 10.0,
            ao_offset: 0,
            n_funcs: 1,
            prim_start: 0,
            n_prims: 1,
            comp_start: 0,
            atom_index: 0,
        }],
        &[AoPrimGpu { exponent: 1.0, weighted_norm: 1.0 }],
        &[AoCompGpu { scale: 1.0, lx: 0, ly: 0, lz: 0, _pad: 0 }],
        &[GridPtGpu { x: 0.1, y: 0.2, z: 0.3, batch_id: 0, host_atom: 0 }],
        &[0, 1],
        &[0],
        1,
    );
    assert!(vxc_none.is_none(), "GpuVxcGrid::upload must fail cleanly when VRAM limit is 0");
}

#[test]
fn test_gpu_rij_streamer_rejects_mismatched_ao_start() {
    use calchem_cuda::{AuxPrimGpu, AuxShellGpu, GpuRijStreamer, PrimPair3CGpu, ShellPair3CGpu};

    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !is_cuda_available() {
        return;
    }
    let aux_shells = [
        AuxShellGpu { center: [0.0; 3], bound: 1.0, lp: 0, ao_offset: 0, prim_start: 0, n_prims: 1 },
        AuxShellGpu { center: [0.0, 0.0, 1.5], bound: 1.0, lp: 0, ao_offset: 1, prim_start: 1, n_prims: 1 },
    ];
    let mut cp0 = [0.0; 10];
    cp0[0] = 1.0;
    let aux_prims = [
        AuxPrimGpu { gamma_p: 1.0, cp_norm: cp0 },
        AuxPrimGpu { gamma_p: 0.5, cp_norm: cp0 },
    ];
    let shell_pairs = [ShellPair3CGpu {
        q_ij: 1.0, li: 0, lj: 0, same_shell: 1, col_offset: 0, prim_start: 0, n_prims: 1,
    }];
    let mut ci = [0.0; 6];
    ci[0] = 1.0;
    let prim_pairs = [PrimPair3CGpu {
        q: 1.0, p_center: [0.0; 3], qc: [0.0; 3], qd: [0.0; 3],
        two_pi25_kab_over_q: 34.986836655249725, ci_norm: ci, cj_norm: ci,
    }];
    let streamer = GpuRijStreamer::upload(&aux_shells, &aux_prims, &shell_pairs, &prim_pairs, 1, 1).unwrap();
    let mut v = [7.0f64; 1];
    assert!(!streamer.pass1_v(&[1.0], 0, 1, 999_999, &mut v));
    assert!(!streamer.pass1_v(&[1.0], 1, 1, 0, &mut v));
    assert!(streamer.pass1_begin(&[1.0]));
    assert!(!streamer.pass1_v_resident(1, 1, 0, &mut v));
    assert!(!streamer.pass2_accum(&[1.0], 1, 1, 0));
    assert_eq!(v, [7.0]);
}
