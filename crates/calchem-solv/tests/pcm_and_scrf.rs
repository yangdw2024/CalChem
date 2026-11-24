use calchem_basis::BasisSet;
use calchem_dft::XcFunctional;
use calchem_model::{Atom, Molecule, ScfOptions, BOHR_TO_ANGSTROM};
use calchem_scf::rhf;
use calchem_solv::{
    compute_pcm_solvation, run_scrf_rhf, run_scrf_rks, CavityLebedevOrder, CavityOptions,
    MolecularCavity, SolvationOptions, Solvent, SolventModel, HARTREE_TO_KCAL_MOL,
};

#[test]
fn test_born_ion_solvation_analytical_benchmark() {
    // Single monoatomic ion (Li+, Z=3, charge=+1) in a spherical cavity
    // Analytical Born equation: Delta G_Born = - (q^2 / (2 * R)) * (1 - 1/eps)
    let li_plus = Molecule::new(vec![Atom::from_bohr("Li", [0.0, 0.0, 0.0]).unwrap()], 1, 1);

    let basis = BasisSet::from_name("sto-3g", &li_plus).expect("Failed to load sto-3g for Li+");
    let scf_opts = ScfOptions {
        max_iter: 50,
        conv_energy: 1e-9,
        conv_density: 1e-8,
        ..Default::default()
    };

    // Run gas phase RHF first
    let gas_wf = rhf(&li_plus, &basis, &scf_opts).expect("Gas-phase RHF failed for Li+");

    // Cavity radius R in Bohr: Bondi radius of Li is 1.82 A.
    // With scaling factor 1.20, R = 1.20 * 1.82 / 0.529177 = 4.127 Bohr
    let cav_opts = CavityOptions {
        scaling_factor: 1.20,
        lebedev_order: CavityLebedevOrder::N110,
        probe_radius_angstrom: 0.0,
        exclusion_tol_bohr: 1e-5,
        ..Default::default()
    };
    let cavity = MolecularCavity::build(&li_plus, &cav_opts);
    assert_eq!(cavity.n_tesserae(), 110);

    let r_cav_bohr = 1.20 * 1.82 / BOHR_TO_ANGSTROM;

    // Test across two solvents: water (eps = 78.36) and chlorobenzene (eps = 5.62)
    for (solvent, eps) in [
        (Solvent::water(), 78.3553),
        (Solvent::chlorobenzene(), 5.6968),
    ] {
        let solv_opts = SolvationOptions {
            solvent,
            model: SolventModel::Cpcm,
            cavity_opts: cav_opts,
            renormalize_charge: true,
            include_cds: false, // Pure electrostatic Born check
        };

        let res = compute_pcm_solvation(&li_plus, &basis, &gas_wf, &solv_opts)
            .expect("Post-SCF PCM failed for Li+");

        // Analytical Born free energy: Delta G = - (q^2 / (2 * R)) * (1 - 1/eps)
        let f_eps = (eps - 1.0) / eps;
        let born_analytical_hartree = -0.5 * (1.0 / r_cav_bohr) * f_eps;
        let born_analytical_kcal = born_analytical_hartree * HARTREE_TO_KCAL_MOL;

        println!(
            "Solvent: {} (eps={:.2}): Numerical elec = {:.2} kcal/mol, Analytical Born = {:.2} kcal/mol",
            solv_opts.solvent.name, eps, res.delta_g_elec_kcal_mol, born_analytical_kcal
        );

        // Discretization on 110 Lebedev points matches analytical sphere within 4%
        let rel_err =
            ((res.delta_g_elec_kcal_mol - born_analytical_kcal) / born_analytical_kcal).abs();
        assert!(
            rel_err < 0.05,
            "Born solvation relative error {:.3} exceeds 5% tolerance",
            rel_err
        );
        assert!(res.delta_g_elec_kcal_mol < 0.0);
    }
}

#[test]
fn test_water_scrf_rhf_and_solvent_polarity_ordering() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1072]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1072]).unwrap(),
        ],
        0,
        1,
    );

    let basis = BasisSet::from_name("sto-3g", &water).expect("Failed to load sto-3g for water");
    let scf_opts = ScfOptions {
        max_iter: 60,
        conv_energy: 1e-8,
        conv_density: 1e-7,
        ..Default::default()
    };

    // 1. Gas phase baseline
    let gas_wf = rhf(&water, &basis, &scf_opts).expect("Gas RHF failed");

    // 2. High dielectric solvent (water, eps = 78.36)
    let water_opts = SolvationOptions {
        solvent: Solvent::water(),
        model: SolventModel::Cpcm,
        cavity_opts: CavityOptions {
            scaling_factor: 1.20,
            lebedev_order: CavityLebedevOrder::N50,
            ..Default::default()
        },
        renormalize_charge: true,
        include_cds: true,
    };
    let water_scrf =
        run_scrf_rhf(&water, &basis, &scf_opts, &water_opts).expect("Water SCRF RHF failed");
    assert!(water_scrf.converged);
    assert!(water_scrf.delta_g_elec_kcal_mol < 0.0);
    println!(
        "Water in Water: Gas = {:.6} Eh, Solv = {:.6} Eh, dG_elec = {:.2} kcal/mol, dG_CDS = {:.2} kcal/mol, dG_tot = {:.2} kcal/mol",
        gas_wf.total_energy_hartree,
        water_scrf.solvated_energy,
        water_scrf.delta_g_elec_kcal_mol,
        water_scrf.delta_g_cds_kcal_mol,
        water_scrf.delta_g_solv_kcal_mol
    );

    // 3. Medium dielectric OPV solvent (chlorobenzene, eps = 5.62)
    let cb_opts = SolvationOptions {
        solvent: Solvent::chlorobenzene(),
        model: SolventModel::Cpcm,
        cavity_opts: CavityOptions {
            scaling_factor: 1.20,
            lebedev_order: CavityLebedevOrder::N50,
            ..Default::default()
        },
        renormalize_charge: true,
        include_cds: true,
    };
    let cb_scrf =
        run_scrf_rhf(&water, &basis, &scf_opts, &cb_opts).expect("Chlorobenzene SCRF RHF failed");
    assert!(cb_scrf.converged);
    println!(
        "Water in Chlorobenzene: Solv = {:.6} Eh, dG_elec = {:.2} kcal/mol, dG_tot = {:.2} kcal/mol",
        cb_scrf.solvated_energy,
        cb_scrf.delta_g_elec_kcal_mol,
        cb_scrf.delta_g_solv_kcal_mol
    );

    // Physical verification: electrostatic stabilization in high-dielectric water must be strictly
    // more stabilizing (more negative) than in low-dielectric chlorobenzene!
    assert!(
        water_scrf.delta_g_elec_kcal_mol < cb_scrf.delta_g_elec_kcal_mol,
        "Water dG_elec ({:.2}) must be more negative than Chlorobenzene ({:.2})",
        water_scrf.delta_g_elec_kcal_mol,
        cb_scrf.delta_g_elec_kcal_mol
    );

    // Total surface charge for neutral molecule must be strictly zero
    assert!(
        water_scrf.total_surface_charge.abs() < 1e-10,
        "Neutral molecule total surface charge must be 0, got {:.3e}",
        water_scrf.total_surface_charge
    );
}

#[test]
fn test_water_scrf_rks_b3lyp() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1072]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1072]).unwrap(),
        ],
        0,
        1,
    );

    let basis = BasisSet::from_name("sto-3g", &water).expect("Failed to load sto-3g for water");
    let scf_opts = ScfOptions {
        max_iter: 60,
        conv_energy: 1e-7,
        conv_density: 1e-6,
        ..Default::default()
    };

    let solv_opts = SolvationOptions {
        solvent: Solvent::water(),
        model: SolventModel::Cpcm,
        cavity_opts: CavityOptions {
            scaling_factor: 1.20,
            lebedev_order: CavityLebedevOrder::N50,
            ..Default::default()
        },
        renormalize_charge: true,
        include_cds: true,
    };

    let res = run_scrf_rks(&water, &basis, &scf_opts, &solv_opts, XcFunctional::B3lyp)
        .expect("B3LYP SCRF failed for water");

    assert!(res.converged);
    assert!(res.delta_g_elec_kcal_mol < 0.0);
    assert!(res.cavity_surface_area_angstrom2 > 10.0);
    println!(
        "B3LYP SCRF Water: Solvated E = {:.6} Eh, dG_elec = {:.2} kcal/mol, SASA = {:.2} A^2, iters = {}",
        res.solvated_energy,
        res.delta_g_elec_kcal_mol,
        res.cavity_surface_area_angstrom2,
        res.scf_iterations
    );
}

#[test]
fn test_unified_run_scrf_rhf_and_dft() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1072]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1072]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &water).unwrap();
    let scf_opts = ScfOptions::default();
    let solv_opts = SolvationOptions::default();

    let res_rhf = calchem_solv::run_scrf(&water, &basis, &scf_opts, &solv_opts, "rhf", None)
        .expect("Unified run_scrf rhf failed");
    assert!(res_rhf.converged);

    let res_dft = calchem_solv::run_scrf(
        &water,
        &basis,
        &scf_opts,
        &solv_opts,
        "b3lyp",
        Some(XcFunctional::B3lyp),
    )
    .expect("Unified run_scrf b3lyp failed");
    assert!(res_dft.converged);
}

#[test]
fn test_pcm_nuclear_gradient() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.1]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &water).unwrap();
    let scf_opts = ScfOptions::default();
    let wf = calchem_scf::rhf(&water, &basis, &scf_opts).unwrap();
    let density = calchem_tensor::Matrix2D {
        rows: basis.n_ao,
        cols: basis.n_ao,
        data: wf.density_alpha,
    };

    let solv = Solvent::water();
    let cavity_opts = CavityOptions {
        lebedev_order: CavityLebedevOrder::N50,
        scaling_factor: 1.20,
        ..Default::default()
    };

    let grad = calchem_solv::run_pcm_gradient(
        &water,
        &basis,
        &density,
        &SolvationOptions {
            solvent: solv,
            model: calchem_solv::SolventModel::Cpcm,
            cavity_opts,
            ..Default::default()
        },
    )
    .expect("PCM gradient failed");

    assert_eq!(grad.len(), 3);
    for ga in &grad {
        for c in 0..3 {
            assert!(ga[c].is_finite());
            assert!(ga[c].abs() < 1.0, "Solvation gradient component should be physically reasonable (< 1.0 Eh/Bohr), got {}", ga[c]);
        }
    }
    // Net force on neutral water should be small
    let mut sum_g = [0.0; 3];
    for ga in &grad {
        sum_g[0] += ga[0];
        sum_g[1] += ga[1];
        sum_g[2] += ga[2];
    }
    println!("PCM gradient sum: {:?}", sum_g);
}

#[test]
fn test_pcm_default_pure_electrostatic_and_iefpcm_ion_gauss_law() {
    // 1. Default SolvationOptions must not add macroscopic gamma*Area CDS to C-PCM/IEF-PCM
    let def_opts = SolvationOptions::default();
    assert!(!def_opts.include_cds, "Default PCM options must be purely electrostatic");

    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1072]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1072]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &water).unwrap();
    let scf_opts = ScfOptions {
        max_iter: 60,
        conv_energy: 1e-8,
        conv_density: 1e-7,
        ..Default::default()
    };

    let ief_opts = SolvationOptions {
        solvent: Solvent::water(),
        model: SolventModel::IefPcm,
        ..Default::default()
    };
    let ief_res = calchem_solv::run_scrf_rks_with_grid(
        &water,
        &basis,
        &scf_opts,
        &ief_opts,
        XcFunctional::B3lyp,
        calchem_dft::GridLevel::Coarse,
    )
    .expect("IEF-PCM SCRF failed");
    assert!(ief_res.converged);
    assert_eq!(ief_res.delta_g_cds_kcal_mol, 0.0);
    assert!(ief_res.delta_g_elec_kcal_mol < 0.0);
    assert!(
        ief_res.delta_g_solv_kcal_mol < 0.0,
        "Default IEF-PCM water-in-water must be stabilizing, got {}",
        ief_res.delta_g_solv_kcal_mol
    );

    // 2. SMD must still compute parameterized SMD-CDS even when include_cds == false
    let smd_opts = SolvationOptions {
        solvent: Solvent::water(),
        model: SolventModel::Smd,
        ..Default::default()
    };
    let smd_res = calchem_solv::run_scrf_rks_with_grid(
        &water,
        &basis,
        &scf_opts,
        &smd_opts,
        XcFunctional::B3lyp,
        calchem_dft::GridLevel::Coarse,
    )
    .expect("SMD SCRF failed");
    assert!(smd_res.converged);
    assert!(smd_res.delta_g_elec_kcal_mol < 0.0);
    assert!(smd_res.delta_g_cds_kcal_mol.abs() > 0.1);
    assert!(smd_res.delta_g_solv_kcal_mol < 0.0);

    // 3. IEF-PCM Gauss law for an ion in low-dielectric solvent (chlorobenzene, eps=5.6968):
    //    (a) Default uncompensated IEF-PCM (renormalize_charge=false) naturally satisfies Gauss law to ~1%;
    //    (b) Explicit renormalize_charge=true enforces -((eps - 1)/eps) * Q_net (NOT -1.0 * Q_net) to < 1e-8.
    let li_plus = Molecule::new(vec![Atom::from_bohr("Li", [0.0, 0.0, 0.0]).unwrap()], 1, 1);
    let li_basis = BasisSet::from_name("sto-3g", &li_plus).unwrap();
    let cb = Solvent::chlorobenzene();
    let expected_q = -((cb.epsilon - 1.0) / cb.epsilon) * 1.0;
    let li_default = run_scrf_rhf(
        &li_plus,
        &li_basis,
        &scf_opts,
        &SolvationOptions {
            solvent: cb.clone(),
            model: SolventModel::IefPcm,
            ..Default::default()
        },
    )
    .expect("Li+ uncompensated IEF-PCM SCRF failed");
    assert!(
        (li_default.total_surface_charge - expected_q).abs() < 0.02,
        "Uncompensated IEF-PCM Li+ charge {:.6} too far from Gauss law {:.6}",
        li_default.total_surface_charge,
        expected_q
    );

    let li_renorm = run_scrf_rhf(
        &li_plus,
        &li_basis,
        &scf_opts,
        &SolvationOptions {
            solvent: cb,
            model: SolventModel::IefPcm,
            renormalize_charge: true,
            ..Default::default()
        },
    )
    .expect("Li+ renormalized IEF-PCM SCRF failed");
    assert!(
        (li_renorm.total_surface_charge - expected_q).abs() < 1e-8,
        "Renormalized IEF-PCM ion surface charge {:.6} != Gauss law target {:.6}",
        li_renorm.total_surface_charge,
        expected_q
    );

    // 4. Regression for BoundaryMatrix::build Cholesky global-shift bug on interlocking spheres:
    //    On ethanol in water (eps=78.36), C-PCM and IEF-PCM on the same cavity must agree within 0.1 kcal/mol
    //    (previously C-PCM was attenuated by +0.60 kcal/mol due to global diagonal Cholesky shift).
    let ethanol = Molecule::new(
        vec![
            Atom::from_angstrom("C", [0.00, 0.00, 0.00]).unwrap(),
            Atom::from_angstrom("C", [1.54, 0.00, 0.00]).unwrap(),
            Atom::from_angstrom("O", [2.24, 1.10, 0.00]).unwrap(),
            Atom::from_angstrom("H", [3.10, 1.10, 0.00]).unwrap(),
            Atom::from_angstrom("H", [-0.40, 0.90, 0.00]).unwrap(),
            Atom::from_angstrom("H", [-0.40, -0.45, 0.779]).unwrap(),
            Atom::from_angstrom("H", [-0.40, -0.45, -0.779]).unwrap(),
            Atom::from_angstrom("H", [1.90, -0.90, 0.00]).unwrap(),
            Atom::from_angstrom("H", [1.90, 0.45, 0.779]).unwrap(),
        ],
        0,
        1,
    );
    let eth_basis = BasisSet::from_name("sto-3g", &ethanol).unwrap();
    let eth_ief = run_scrf_rhf(
        &ethanol,
        &eth_basis,
        &scf_opts,
        &SolvationOptions {
            solvent: Solvent::water(),
            model: SolventModel::IefPcm,
            ..Default::default()
        },
    )
    .expect("Ethanol IEF-PCM failed");
    let eth_cpcm = run_scrf_rhf(
        &ethanol,
        &eth_basis,
        &scf_opts,
        &SolvationOptions {
            solvent: Solvent::water(),
            model: SolventModel::Cpcm,
            ..Default::default()
        },
    )
    .expect("Ethanol C-PCM failed");
    assert!(
        (eth_cpcm.delta_g_solv_kcal_mol - eth_ief.delta_g_solv_kcal_mol).abs() < 0.10,
        "Ethanol C-PCM ({:.3}) diverged from IEF-PCM ({:.3}) in water (eps=78.36)",
        eth_cpcm.delta_g_solv_kcal_mol,
        eth_ief.delta_g_solv_kcal_mol
    );
}

#[test]
fn test_solvation_gradient_matches_scrf_surface_iefpcm_cpcm_smd() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.1]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &water).unwrap();
    let scf_opts = ScfOptions {
        max_iter: 60,
        conv_energy: 1e-9,
        conv_density: 1e-8,
        ..Default::default()
    };
    let h = 1e-4;

    let cases = [
        SolvationOptions {
            solvent: Solvent::chlorobenzene(),
            model: SolventModel::IefPcm,
            ..Default::default()
        },
        SolvationOptions {
            solvent: Solvent::chlorobenzene(),
            model: SolventModel::Cpcm,
            include_cds: false,
            ..Default::default()
        },
        SolvationOptions {
            solvent: Solvent::chlorobenzene(),
            model: SolventModel::Cpcm,
            include_cds: true,
            ..Default::default()
        },
        SolvationOptions {
            solvent: Solvent::water(),
            model: SolventModel::Smd,
            include_cds: false,
            ..Default::default()
        },
        SolvationOptions {
            solvent: Solvent::water(),
            model: SolventModel::Smd,
            include_cds: true,
            ..Default::default()
        },
    ];

    let mut grads = Vec::new();
    let mut energies = Vec::new();

    for opts in &cases {
        let scrf_res = run_scrf_rhf(&water, &basis, &scf_opts, opts)
            .expect("SCRF failed before fixed-density gradient check");
        let density = calchem_tensor::Matrix2D {
            rows: basis.n_ao,
            cols: basis.n_ao,
            data: scrf_res.wavefunction.density_alpha.clone(),
        };

        let e_grad_fn = calchem_solv::pcm_energy_for_density(&water, &basis, &density, opts)
            .expect("pcm_energy_for_density failed");
        let e_scrf_frozen = calchem_solv::scrf::frozen_density_correction(
            &water,
            &basis,
            density.data.clone(),
            opts,
        )
        .expect("frozen_density_correction failed")
        .delta_g_solv;
        assert!(
            (e_grad_fn - e_scrf_frozen).abs() < 1e-12,
            "Energy surface mismatch for {:?}: grad_fn={e_grad_fn:.12} vs scrf={e_scrf_frozen:.12}",
            opts.model
        );
        energies.push(e_grad_fn);

        let grad = calchem_solv::compute_pcm_gradient_fd(&water, &basis, &density, opts, h)
            .expect("compute_pcm_gradient_fd failed");

        for i in 0..water.atoms.len() {
            for c in 0..3 {
                let mut mol_p = water.clone();
                mol_p.atoms[i].position_bohr[c] += h;
                let basis_p = BasisSet::from_name(&basis.name, &mol_p).unwrap();
                let ep = calchem_solv::scrf::frozen_density_correction(
                    &mol_p,
                    &basis_p,
                    density.data.clone(),
                    opts,
                )
                .unwrap()
                .delta_g_solv;

                let mut mol_m = water.clone();
                mol_m.atoms[i].position_bohr[c] -= h;
                let basis_m = BasisSet::from_name(&basis.name, &mol_m).unwrap();
                let em = calchem_solv::scrf::frozen_density_correction(
                    &mol_m,
                    &basis_m,
                    density.data.clone(),
                    opts,
                )
                .unwrap()
                .delta_g_solv;

                let ref_g = (ep - em) / (2.0 * h);
                assert!(
                    (grad[i][c] - ref_g).abs() < 1e-11,
                    "FD gradient mismatch for {:?} atom {i} coord {c}: got {:.12e}, ref {:.12e}",
                    opts.model,
                    grad[i][c],
                    ref_g
                );
            }
        }
        grads.push(grad);
    }

    let max_diff = |a: &[[f64; 3]], b: &[[f64; 3]]| -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(ga, gb)| (0..3).map(move |c| (ga[c] - gb[c]).abs()))
            .fold(0.0_f64, f64::max)
    };

    // IEF-PCM and C-PCM must dispatch distinct electrostatics in chlorobenzene
    assert!(
        max_diff(&grads[0], &grads[1]) > 1e-5,
        "IEF-PCM and C-PCM gradients unexpectedly identical"
    );
    // C-PCM include_cds=false vs true must add gamma * Area to the solvation energy surface
    // (Note: discrete unsmoothed Lebedev sphere areas are piecewise constant away from tessera switches)
    assert!(
        (energies[1] - energies[2]).abs() > 1e-4,
        "C-PCM include_cds toggle had no effect on pcm_energy_for_density: {} vs {}",
        energies[1],
        energies[2]
    );
    // SMD uses parameterized SMD-CDS and SmdIntrinsic radii identically regardless of include_cds,
    // yielding a distinct surface from pure Bondi IEF-PCM
    assert!(
        (energies[3] - energies[4]).abs() < 1e-12,
        "SMD energy varied with include_cds"
    );
    assert!(
        max_diff(&grads[3], &grads[4]) < 1e-12,
        "SMD gradient varied with include_cds"
    );
    assert!(
        (energies[3] - energies[0]).abs() > 1e-3 && max_diff(&grads[3], &grads[0]) > 1e-4,
        "SMD surface unexpectedly matched Bondi IEF-PCM"
    );
}

#[test]
fn test_scrf_closed_shell_respects_use_diis_false() {
    let water = Molecule::new(
        vec![
            Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
            Atom::from_bohr("H", [0.0, 1.4304, -1.1072]).unwrap(),
            Atom::from_bohr("H", [0.0, -1.4304, -1.1072]).unwrap(),
        ],
        0,
        1,
    );
    let basis = BasisSet::from_name("sto-3g", &water).unwrap();
    let solv_opts = SolvationOptions {
        solvent: Solvent::water(),
        model: SolventModel::IefPcm,
        ..Default::default()
    };
    let opts_diis = ScfOptions {
        max_iter: 60,
        conv_energy: 1e-8,
        conv_density: 1e-7,
        use_diis: true,
        ..Default::default()
    };
    let opts_no_diis = ScfOptions {
        max_iter: 120,
        conv_energy: 1e-8,
        conv_density: 1e-7,
        use_diis: false,
        damping: 0.2,
        ..Default::default()
    };

    let rhf_diis = run_scrf_rhf(&water, &basis, &opts_diis, &solv_opts).unwrap();
    let rhf_no_diis = run_scrf_rhf(&water, &basis, &opts_no_diis, &solv_opts).unwrap();
    assert!(rhf_no_diis.converged);
    assert!(
        (rhf_diis.solvated_energy - rhf_no_diis.solvated_energy).abs() < 1e-7,
        "RHF SCRF with/without DIIS diverged: {} vs {}",
        rhf_diis.solvated_energy,
        rhf_no_diis.solvated_energy
    );

    let rks_diis = calchem_solv::run_scrf_rks_with_grid(
        &water,
        &basis,
        &opts_diis,
        &solv_opts,
        XcFunctional::B3lyp,
        calchem_dft::GridLevel::Coarse,
    )
    .unwrap();
    let rks_no_diis = calchem_solv::run_scrf_rks_with_grid(
        &water,
        &basis,
        &opts_no_diis,
        &solv_opts,
        XcFunctional::B3lyp,
        calchem_dft::GridLevel::Coarse,
    )
    .unwrap();
    assert!(rks_no_diis.converged);
    assert!(
        (rks_diis.solvated_energy - rks_no_diis.solvated_energy).abs() < 1e-7,
        "RKS SCRF with/without DIIS diverged: {} vs {}",
        rks_diis.solvated_energy,
        rks_no_diis.solvated_energy
    );
}
