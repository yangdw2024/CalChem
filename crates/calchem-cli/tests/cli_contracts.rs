use std::fs;
use std::path::PathBuf;
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

fn write_water_xyz(dir: &PathBuf, name: &str, oh_ang: f64) -> PathBuf {
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
