use calchem_dft::GridLevel;
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_opt::irc::{run_irc, IrcDirection, IrcOptions};
use calchem_opt::prfo::{search_transition_state, TsOptions};
use calchem_opt::Reference;

#[test]
fn test_hcn_isomerization_ts_search_and_irc() {
    // Textbook first-order saddle point: HCN <-> HNC isomerization.
    // At the TS, hydrogen bridges between carbon and nitrogen with a single imaginary frequency (-1248 cm^-1).
    // Start with a slightly perturbed near-TS geometry:
    let atoms = vec![
        Atom::from_bohr("C", [0.0, 0.0, 0.0]).unwrap(),
        Atom::from_bohr("N", [2.310, 0.0, 0.0]).unwrap(),
        Atom::from_bohr("H", [0.685, 2.155, 0.0]).unwrap(),
    ];
    let mol = Molecule::new(atoms, 0, 1);
    let basis_name = "sto-3g";

    let scf_opts = ScfOptions {
        max_iter: 100,
        conv_energy: 1e-11,
        conv_density: 1e-10,
        ..Default::default()
    };

    let ts_opts = TsOptions {
        max_steps: 15,
        max_force: 8e-4,
        rms_force: 5e-4,
        max_displacement: 2e-3,
        trust_radius_bohr: 0.05,
        reference: Reference::Rhf,
        grid_level: GridLevel::Medium,
        scf_options: scf_opts.clone(),
        ..Default::default()
    };

    // 1. Run P-RFO Transition State Search
    let ts_res = search_transition_state(&mol, basis_name, &ts_opts)
        .expect("P-RFO transition state search failed");

    assert!(ts_res.converged, "P-RFO should converge to saddle point");
    assert!(
        ts_res.imaginary_frequency_cm < -500.0,
        "Expected imaginary frequency < -500 cm^-1, got {:.1}",
        ts_res.imaginary_frequency_cm
    );

    // Final geometry should be close to literature TS (R_CN ~ 2.308, R_NH ~ 2.718)
    let ts_mol = &ts_res.molecule;
    let r_cn = ts_mol.atoms[1].position_bohr[0] - ts_mol.atoms[0].position_bohr[0];
    assert!((r_cn - 2.308).abs() < 0.03, "CN distance should be ~2.308, got {}", r_cn);

    // 2. Run Gonzalez-Schlegel IRC from the Transition State
    let irc_opts = IrcOptions {
        step_size_bohr: 0.1,
        max_points: 4,
        direction: IrcDirection::Both,
        convergence_tol: 1e-3,
        max_gs_iter: 6,
        reference: Reference::Rhf,
        grid_level: GridLevel::Medium,
        scf_options: scf_opts,
    };

    let irc_res = run_irc(ts_mol, basis_name, &ts_res.transition_vector, &irc_opts)
        .expect("Gonzalez-Schlegel IRC integration failed");

    // Verify TS point
    assert_eq!(irc_res.transition_state.point_index, 0);
    assert_eq!(irc_res.transition_state.path_coordinate_s, 0.0);
    assert_eq!(irc_res.transition_state.delta_energy_kcal_mol, 0.0);

    // Verify Forward branch (descends in energy towards HCN or HNC)
    let fwd = irc_res.forward.as_ref().expect("Forward branch should exist");
    assert!(!fwd.points.is_empty(), "Forward branch should have points");
    let last_fwd = fwd.points.last().unwrap();
    assert!(
        last_fwd.delta_energy_kcal_mol < 0.0,
        "Forward branch should drop in energy relative to TS"
    );
    assert!(fwd.barrier_kcal_mol > 0.0);

    // Verify Reverse branch (descends in energy towards the opposite isomer)
    let rev = irc_res.reverse.as_ref().expect("Reverse branch should exist");
    assert!(!rev.points.is_empty(), "Reverse branch should have points");
    let last_rev = rev.points.last().unwrap();
    assert!(
        last_rev.delta_energy_kcal_mol < 0.0,
        "Reverse branch should drop in energy relative to TS"
    );
    assert!(rev.barrier_kcal_mol > 0.0);
}
