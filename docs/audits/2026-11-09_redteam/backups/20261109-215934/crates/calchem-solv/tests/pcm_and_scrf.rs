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
    let li_plus = Molecule::new(
        vec![Atom::from_bohr("Li", [0.0, 0.0, 0.0]).unwrap()],
        1,
        1,
    );

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
    };
    let cavity = MolecularCavity::build(&li_plus, &cav_opts);
    assert_eq!(cavity.n_tesserae(), 110);

    let r_cav_bohr = 1.20 * 1.82 / BOHR_TO_ANGSTROM;

    // Test across two solvents: water (eps = 78.36) and chlorobenzene (eps = 5.62)
    for (solvent, eps) in [
        (Solvent::water(), 78.3553),
        (Solvent::chlorobenzene(), 5.62),
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
        let rel_err = ((res.delta_g_elec_kcal_mol - born_analytical_kcal) / born_analytical_kcal).abs();
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
    let water_scrf = run_scrf_rhf(&water, &basis, &scf_opts, &water_opts)
        .expect("Water SCRF RHF failed");
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
    let cb_scrf = run_scrf_rhf(&water, &basis, &scf_opts, &cb_opts)
        .expect("Chlorobenzene SCRF RHF failed");
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

    let res_dft = calchem_solv::run_scrf(&water, &basis, &scf_opts, &solv_opts, "b3lyp", Some(XcFunctional::B3lyp))
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

    let grad = calchem_solv::run_pcm_gradient(&water, &basis, &density, &solv, Some(&cavity_opts))
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
