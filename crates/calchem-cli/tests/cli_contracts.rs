use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn cli_bin() -> &'static str {
    env!("CARGO_BIN_EXE_calchem_cli")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "calchem_cli_test_{}_{}_{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_water_xyz(dir: &Path, name: &str, oh_ang: f64) -> PathBuf {
    let p = dir.join(name);
    let content = format!(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 {:.12} 0.0\nH {:.12} {:.12} 0.0\n",
        oh_ang,
        oh_ang * (104.5_f64.to_radians()).sin(),
        oh_ang * (104.5_f64.to_radians()).cos()
    );
    fs::write(&p, content).unwrap();
    p
}

#[test]
fn p0_01_d3bj_energy_gradient_and_fd_closed_loop() {
    let dir = temp_dir("d3_loop");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 1.05);

    let out_nod3 = dir.join("nod3.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "grad", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "medium", "--threads", "2",
            "--json", "--out", out_nod3.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let j_nod3: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_nod3).unwrap()).unwrap();

    let out_d3 = dir.join("d3.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "grad", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "medium", "--disp", "d3bj",
            "--threads", "2", "--json", "--out", out_d3.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let j_d3: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_d3).unwrap()).unwrap();

    // P0-04 contract fields
    assert_eq!(j_d3["status"], "converged");
    assert_eq!(j_d3["initial_geometry_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(j_d3["initial_geometry_sha256"], j_d3["final_geometry_sha256"]);
    assert!(j_d3["scf"]["homo_hartree"].as_f64().is_some());
    assert!(j_d3["scf"]["lumo_hartree"].as_f64().is_some());

    // Energy delta must equal reported dispersion_energy_hartree
    let e_nod3 = j_nod3["energy_hartree"].as_f64().unwrap();
    let e_d3 = j_d3["energy_hartree"].as_f64().unwrap();
    let disp_e = j_d3["dispersion_energy_hartree"].as_f64().unwrap();
    assert!(disp_e < 0.0, "D3BJ energy should be negative, got {disp_e}");
    assert!((e_d3 - e_nod3 - disp_e).abs() < 1e-11);

    // Central finite difference of total energy (with D3BJ) along H1 y-coordinate must match analytical gradient!
    let h_bohr = 1e-4_f64;
    let h_ang = h_bohr * calchem_model::BOHR_TO_ANGSTROM;
    let mut energies = [0.0f64; 2];
    for (idx, sign) in [(0, 1.0), (1, -1.0)] {
        let p = dir.join(format!("h2o_{idx}.xyz"));
        let oh = 1.05;
        let content = format!(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.0 {:.15} 0.0\nH {:.15} {:.15} 0.0\n",
            oh + sign * h_ang,
            oh * (104.5_f64.to_radians()).sin(),
            oh * (104.5_f64.to_radians()).cos()
        );
        fs::write(&p, content).unwrap();
        let out = dir.join(format!("e_{idx}.json"));
        let st = Command::new(cli_bin())
            .args([
                p.to_str().unwrap(),
                "--task", "energy", "--method", "rks", "--xc", "b3lyp",
                "--basis", "sto-3g", "--grid", "medium", "--disp", "d3bj",
                "--conv-e", "1e-12", "--conv-d", "1e-10", "--threads", "2",
                "--json", "--out", out.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st.success());
        let j: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
        energies[idx] = j["energy_hartree"].as_f64().unwrap();
    }
    let fd_grad_h1_y = (energies[0] - energies[1]) / (2.0 * h_bohr);
    let ana_grad_h1_y = j_d3["gradient_atomic_units"][1][1].as_f64().unwrap();
    let nod3_grad_h1_y = j_nod3["gradient_atomic_units"][1][1].as_f64().unwrap();
    assert!((ana_grad_h1_y - nod3_grad_h1_y).abs() > 1e-5, "D3BJ gradient must actually change gradient");
    assert!(
        (ana_grad_h1_y - fd_grad_h1_y).abs() < 2e-5,
        "D3BJ analytical grad {ana_grad_h1_y:.8} vs FD {fd_grad_h1_y:.8}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p0_02_invalid_arguments_and_combinations_fail_cleanly() {
    let dir = temp_dir("invalid_args");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 0.96);

    let cases: &[(&str, &[&str])] = &[
        ("bad_disp", &["--task", "energy", "--disp", "d999"]),
        ("wb97xd_plus_d3", &["--task", "energy", "--xc", "wb97x-d", "--disp", "d3bj"]),
        ("bad_solv", &["--task", "energy", "--solv", "cosmo"]),
        ("solvent_without_solv", &["--task", "energy", "--solvent", "water"]),
        ("bad_solvent_name", &["--task", "energy", "--solv", "pcm", "--solvent", "kryptonite"]),
        ("solv_on_opt", &["--task", "opt", "--solv", "pcm", "--solvent", "water"]),
        ("zero_max_iter", &["--task", "energy", "--max-iter", "0"]),
        ("zero_nstates", &["--task", "tddft", "--nstates", "0"]),
        ("resume_on_energy", &["--task", "energy", "--resume", "dummy.json"]),
        ("rij_on_tddft", &["--task", "tddft", "--rij"]),
    ];

    for (name, extra) in cases {
        let out = dir.join(format!("{name}.json"));
        let output = Command::new(cli_bin())
            .arg(xyz.to_str().unwrap())
            .args(*extra)
            .args(["--json", "--out", out.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "Case {name} unexpectedly succeeded!"
        );
        assert!(
            !out.exists(),
            "Case {name} wrote an output JSON despite failing!"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p0_05_opt_interrupt_checkpoint_resume_matches_continuous_and_rejects_mismatch() {
    let dir = temp_dir("opt_resume");
    let xyz = write_water_xyz(&dir, "h2o_stretched.xyz", 1.15);

    // 1. Uninterrupted optimization
    let out_full = dir.join("full.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "opt", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--disp", "d3bj",
            "--opt-max-cycles", "25", "--threads", "2",
            "--json", "--out", out_full.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let j_full: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_full).unwrap()).unwrap();
    assert_eq!(j_full["converged"], true);
    assert_ne!(j_full["initial_geometry_sha256"], j_full["final_geometry_sha256"]);

    // 2. Deliberately truncated optimization (max-cycles=2 -> fails with non-zero exit & saves checkpoint)
    let cp_path = dir.join("opt.checkpoint.json");
    let out_part = dir.join("part.json");
    let part = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "opt", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--disp", "d3bj",
            "--opt-max-cycles", "2", "--checkpoint", cp_path.to_str().unwrap(),
            "--threads", "2", "--json", "--out", out_part.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!part.status.success(), "Truncated 2-step opt should return non-zero");
    assert!(!out_part.exists(), "Failed opt must not emit success JSON");
    assert!(cp_path.exists(), "Truncated opt must preserve checkpoint file");

    // 3. Resume with mismatched model (different grid) must be rejected
    let bad_resume = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "opt", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "medium", "--disp", "d3bj",
            "--opt-max-cycles", "25", "--resume", cp_path.to_str().unwrap(),
            "--threads", "2", "--json",
        ])
        .output()
        .unwrap();
    assert!(!bad_resume.status.success(), "Mismatched resume must be rejected");

    // 4. Resume with matching model must converge to identical final energy & geometry!
    let out_resumed = dir.join("resumed.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "opt", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--disp", "d3bj",
            "--opt-max-cycles", "25", "--resume", cp_path.to_str().unwrap(),
            "--threads", "2", "--json", "--out", out_resumed.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success(), "Valid resume must succeed");
    let j_res: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_resumed).unwrap()).unwrap();
    assert_eq!(j_res["converged"], true);
    let e_full = j_full["energy_hartree"].as_f64().unwrap();
    let e_res = j_res["energy_hartree"].as_f64().unwrap();
    assert!(
        (e_full - e_res).abs() < 1e-8,
        "Resumed energy {e_res:.12} != full energy {e_full:.12}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p0_03_and_p1_03_direct_tddft_lr_and_freq_checkpoint_resume() {
    let dir = temp_dir("td_and_freq");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 0.96);

    // 1. Direct full-LR TD-DFT via CLI
    let out_td = dir.join("td_lr.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "tddft", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--direct",
            "--td-mode", "lr", "--spin", "singlet", "--nstates", "2",
            "--threads", "2", "--json", "--out", out_td.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let j_td: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_td).unwrap()).unwrap();
    assert_eq!(j_td["task_settings"]["td_mode"], "lr");
    assert_eq!(j_td["direct_scf"], true);
    assert_eq!(j_td["excited_states"].as_array().unwrap().len(), 2);
    assert!(j_td["excited_states"][0]["residual_norm_hartree"].as_f64().unwrap() < 1e-5);

    // 2. Freq with D3BJ and displacement checkpointing + cached rerun
    let freq_cp = dir.join("freq_cp");
    let out_f1 = dir.join("f1.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "freq", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--disp", "d3bj",
            "--checkpoint", freq_cp.to_str().unwrap(), "--freq-jobs", "1",
            "--threads", "2", "--json", "--out", out_f1.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert!(freq_cp.join("disp_0000.json").exists());

    // Rerun using cached displacements
    let out_f2 = dir.join("f2.json");
    let out2 = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "freq", "--method", "rks", "--xc", "b3lyp",
            "--basis", "sto-3g", "--grid", "coarse", "--disp", "d3bj",
            "--checkpoint", freq_cp.to_str().unwrap(), "--freq-jobs", "1",
            "--threads", "2", "--json", "--out", out_f2.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out2.status.success());
    let stderr = String::from_utf8_lossy(&out2.stderr);
    assert!(stderr.contains("(cached)"), "Second run should reuse cached displacements: {stderr}");

    let j_f1: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_f1).unwrap()).unwrap();
    let j_f2: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_f2).unwrap()).unwrap();
    assert_eq!(j_f1["frequencies_cm1"], j_f2["frequencies_cm1"]);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn open_03_cli_scrf_solvation_decomposition_contracts() {
    let dir = temp_dir("solv_scrf");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 0.96);

    for model in ["pcm", "iefpcm", "cpcm", "smd"] {
        let out = dir.join(format!("solv_{model}.json"));
        let st = Command::new(cli_bin())
            .args([
                xyz.to_str().unwrap(),
                "--task", "energy", "--method", "rks", "--xc", "b3lyp",
                "--basis", "sto-3g", "--grid", "coarse",
                "--solv", model, "--solvent", "water",
                "--threads", "2", "--json", "--out", out.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st.success(), "CLI --solv {model} failed");
        let j: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(j["status"], "converged");
        assert_eq!(j["task_settings"]["solvation"], "scrf");
        let dg = j["solvation_energy_kcal_mol"].as_f64().unwrap();
        let elec = j["task_settings"]["electrostatic_kcal_mol"].as_f64().unwrap();
        let cds = j["task_settings"]["cds_kcal_mol"].as_f64().unwrap();
        assert!(elec < 0.0, "{model}: expected negative elec, got {elec}");
        assert!(dg < 0.0, "{model}: expected negative dG_solv for water in water, got {dg}");
        if model == "smd" {
            assert!(cds.abs() > 0.1, "SMD must include non-zero CDS term, got {cds}");
        } else {
            assert_eq!(cds, 0.0, "{model}: PCM must not add spurious macroscopic CDS, got {cds}");
        }
        assert!((dg - (elec + cds)).abs() < 1e-9, "{model}: dG != elec + cds");
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn m5a_tddft_gpu_and_disable_gpu_telemetry_contract() {
    let dir = temp_dir("td_gpu_telemetry");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 0.96);

    // 1. CALCHEM_DISABLE_GPU=1 with --gpu must record zero GPU kernels and > 0 fallbacks
    let out_dis = dir.join("td_disabled.json");
    let st_dis = Command::new(cli_bin())
        .env("CALCHEM_DISABLE_GPU", "1")
        .args([
            xyz.to_str().unwrap(),
            "--task", "tddft", "--method", "rks", "--xc", "b3lyp",
            "--basis", "6-31g(d)", "--grid", "coarse", "--direct", "--gpu",
            "--td-mode", "lr", "--spin", "singlet", "--nstates", "2",
            "--threads", "2", "--json", "--out", out_dis.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st_dis.success());
    let j_dis: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_dis).unwrap()).unwrap();
    assert_eq!(j_dis["gpu_requested"], true);
    assert_eq!(j_dis["gpu_kernel_invocations"], 0);
    assert!(j_dis["gpu_fallback_invocations"].as_u64().unwrap() > 0);
    assert_eq!(j_dis["task_settings"]["tddft_gpu_kernel_invocations"], 0);
    assert!(j_dis["task_settings"]["tddft_gpu_fallback_invocations"].as_u64().unwrap() > 0);

    // 2. If CUDA device is present, --task tddft --gpu must record > 0 TD-DFT kernels and 0 fallbacks
    if calchem_int::is_cuda_available() {
        let out_gpu = dir.join("td_gpu.json");
        let st_gpu = Command::new(cli_bin())
            .env_remove("CALCHEM_DISABLE_GPU")
            .args([
                xyz.to_str().unwrap(),
                "--task", "tddft", "--method", "rks", "--xc", "b3lyp",
                "--basis", "6-31g(d)", "--grid", "coarse", "--direct", "--gpu",
                "--td-mode", "lr", "--spin", "singlet", "--nstates", "2",
                "--threads", "2", "--json", "--out", out_gpu.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st_gpu.success());
        let j_gpu: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_gpu).unwrap()).unwrap();
        assert_eq!(j_gpu["gpu_requested"], true);
        let scf_k = j_gpu["scf"]["diagnostics"]["gpu_kernel_invocations"].as_u64().unwrap();
        let td_k = j_gpu["task_settings"]["tddft_gpu_kernel_invocations"].as_u64().unwrap();
        let tot_k = j_gpu["gpu_kernel_invocations"].as_u64().unwrap();
        assert!(td_k > 0, "Expected TD-DFT GPU kernels > 0, got {td_k}");
        assert_eq!(tot_k, scf_k + td_k);
        assert_eq!(j_gpu["gpu_fallback_invocations"], 0);
        assert_eq!(j_gpu["task_settings"]["tddft_gpu_fallback_invocations"], 0);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn m5b_grad_gpu_and_disable_gpu_telemetry_contract() {
    let dir = temp_dir("grad_gpu_telemetry");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 1.02);

    // 1. CALCHEM_DISABLE_GPU=1 with --gpu must record zero GPU kernels and > 0 fallbacks
    let out_dis = dir.join("grad_disabled.json");
    let st_dis = Command::new(cli_bin())
        .env("CALCHEM_DISABLE_GPU", "1")
        .args([
            xyz.to_str().unwrap(),
            "--task", "gradient", "--method", "rks", "--xc", "b3lyp",
            "--basis", "6-31g(d)", "--grid", "coarse", "--direct", "--gpu",
            "--threads", "2", "--json", "--out", out_dis.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st_dis.success());
    let j_dis: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_dis).unwrap()).unwrap();
    assert_eq!(j_dis["gpu_requested"], true);
    assert_eq!(j_dis["gpu_kernel_invocations"], 0);
    assert!(j_dis["gpu_fallback_invocations"].as_u64().unwrap() > 0);
    assert_eq!(j_dis["task_settings"]["gradient_gpu_kernel_invocations"], 0);
    assert!(j_dis["task_settings"]["gradient_gpu_fallback_invocations"].as_u64().unwrap() > 0);

    // 2. If CUDA device is present, --task gradient --gpu must record > 0 grad kernels, 0 fallbacks,
    // and match CPU analytical gradient within 1e-6 Eh/Bohr.
    if calchem_int::is_cuda_available() {
        let out_cpu = dir.join("grad_cpu.json");
        let st_cpu = Command::new(cli_bin())
            .env_remove("CALCHEM_DISABLE_GPU")
            .args([
                xyz.to_str().unwrap(),
                "--task", "gradient", "--method", "rks", "--xc", "b3lyp",
                "--basis", "6-31g(d)", "--grid", "coarse", "--direct",
                "--threads", "2", "--json", "--out", out_cpu.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st_cpu.success());
        let j_cpu: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_cpu).unwrap()).unwrap();

        let out_gpu = dir.join("grad_gpu.json");
        let st_gpu = Command::new(cli_bin())
            .env_remove("CALCHEM_DISABLE_GPU")
            .args([
                xyz.to_str().unwrap(),
                "--task", "gradient", "--method", "rks", "--xc", "b3lyp",
                "--basis", "6-31g(d)", "--grid", "coarse", "--direct", "--gpu",
                "--threads", "2", "--json", "--out", out_gpu.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st_gpu.success());
        let j_gpu: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_gpu).unwrap()).unwrap();
        assert_eq!(j_gpu["gpu_requested"], true);
        let scf_k = j_gpu["scf"]["diagnostics"]["gpu_kernel_invocations"].as_u64().unwrap();
        let grad_k = j_gpu["task_settings"]["gradient_gpu_kernel_invocations"].as_u64().unwrap();
        let tot_k = j_gpu["gpu_kernel_invocations"].as_u64().unwrap();
        assert!(grad_k > 0, "Expected gradient GPU kernels > 0, got {grad_k}");
        assert_eq!(tot_k, scf_k + grad_k);
        assert_eq!(j_gpu["gpu_fallback_invocations"], 0);
        assert_eq!(j_gpu["task_settings"]["gradient_gpu_fallback_invocations"], 0);

        let g_cpu = j_cpu["gradient_atomic_units"].as_array().unwrap();
        let g_gpu = j_gpu["gradient_atomic_units"].as_array().unwrap();
        let mut max_diff = 0.0_f64;
        for (a_cpu, a_gpu) in g_cpu.iter().zip(g_gpu.iter()) {
            for q in 0..3 {
                let d = (a_cpu[q].as_f64().unwrap() - a_gpu[q].as_f64().unwrap()).abs();
                max_diff = max_diff.max(d);
            }
        }
        assert!(max_diff <= 1e-6, "GPU vs CPU gradient max diff = {max_diff}");
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn tier5_cli_flag_conflicts_and_uppercase_case_insensitivity() {
    let dir = temp_dir("tier5_flags");
    let xyz = write_water_xyz(&dir, "h2o.xyz", 0.96);

    let conflicts: &[(&str, &[&str])] = &[
        ("direct_and_stored", &["--task", "energy", "--direct", "--stored"]),
        ("stored_and_rij", &["--task", "energy", "--stored", "--rij"]),
        ("stored_and_aux", &["--task", "energy", "--stored", "--aux-basis", "auto-jfit"]),
        ("rhf_with_xc", &["--task", "energy", "--method", "rhf", "--xc", "pbe"]),
        ("uhf_with_grid", &["--task", "energy", "--method", "uhf", "--grid", "fine"]),
        ("energy_with_td_mode", &["--task", "energy", "--td-mode", "lr"]),
        ("energy_with_freq_step", &["--task", "energy", "--freq-step", "0.002"]),
        ("energy_with_opt_cycles", &["--task", "energy", "--opt-max-cycles", "10"]),
        ("mult_zero", &["--task", "energy", "--mult", "0"]),
        ("diis_gt_64", &["--task", "energy", "--diis-max-vecs", "65"]),
    ];

    for (name, extra) in conflicts {
        let out = dir.join(format!("{name}.json"));
        let res = Command::new(cli_bin())
            .arg(xyz.to_str().unwrap())
            .args(*extra)
            .args(["--json", "--out", out.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!res.status.success(), "Conflict {name} unexpectedly succeeded!");
        assert!(!out.exists(), "Conflict {name} emitted JSON despite failing!");
    }

    let out_uc = dir.join("uc_rij.json");
    let st = Command::new(cli_bin())
        .args([
            xyz.to_str().unwrap(),
            "--task", "energy", "--method", "rks", "--xc", "pbe",
            "--basis", "sto-3g", "--grid", "coarse", "--aux-basis", "AUTO-JFIT",
            "--threads", "2", "--json", "--out", out_uc.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success());

    // Blank-title .gjf must parse cleanly (GAP-B13)
    let gjf = dir.join("blank_title.gjf");
    fs::write(
        &gjf,
        "#P B3LYP/STO-3G\n\n\n0 1\nO 0.0 0.0 0.0\nH 0.0 0.757 0.586\nH 0.0 -0.757 0.586\n\n",
    )
    .unwrap();
    let out_gjf = dir.join("blank_title.json");
    let st_gjf = Command::new(cli_bin())
        .args([
            gjf.to_str().unwrap(),
            "--task", "energy", "--method", "rks", "--xc", "pbe",
            "--basis", "sto-3g", "--grid", "coarse",
            "--threads", "2", "--json", "--out", out_gjf.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st_gjf.success(), "Blank-title .gjf failed to parse!");

    let _ = fs::remove_dir_all(&dir);
}
