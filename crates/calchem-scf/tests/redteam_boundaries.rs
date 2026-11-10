use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, XcFunctional};
use calchem_model::{Atom,Molecule,ScfOptions};

#[test]
fn legacy_lrc_enum_is_not_a_backdoor() {
    let mol=Molecule::new(vec![Atom::from_bohr("He",[0.0;3]).unwrap()],0,1);
    let bs=BasisSet::from_name("sto-3g",&mol).unwrap();
    for xc in [XcFunctional::CamB3lyp,XcFunctional::Wb97xd] {
        assert!(calchem_scf::rks(&mol,&bs,xc,GridLevel::Coarse,&ScfOptions::default()).is_err());
        assert!(calchem_scf::uks(&mol,&bs,xc,GridLevel::Coarse,&ScfOptions::default()).is_err());
    }
}

#[test]
fn memory_and_invalid_scf_options_are_rejected_before_integrals() {
    let mol=Molecule::new((0..256).map(|i|Atom::from_bohr("He",[i as f64*4.0,0.0,0.0]).unwrap()).collect(),0,1);
    let bs=BasisSet::from_name("sto-3g",&mol).unwrap();
    let error=calchem_scf::validate_resources_and_options(&bs,&ScfOptions::default()).unwrap_err().to_string();
    assert!(error.contains("memory estimate"),"{error}");
    let invalid=ScfOptions{conv_energy:f64::INFINITY,..Default::default()};
    assert!(calchem_scf::validate_resources_and_options(&bs,&invalid).is_err());
}

#[test]
fn singular_basis_is_rejected_instead_of_zero_energy_ghost_orbitals() {
    let mol=Molecule::new(vec![Atom::from_bohr("He",[0.0;3]).unwrap()],0,1);
    let bs=BasisSet::from_json_str(r#"{"name":"duplicate-s","elements":{"He":[{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]},{"shell_type":"s","exponents":[1.0],"coefficients":[1.0]}]}}"#,&mol).unwrap();
    assert!(calchem_scf::rhf(&mol,&bs,&ScfOptions::default()).is_err());
}
