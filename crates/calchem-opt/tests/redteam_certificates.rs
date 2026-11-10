use calchem_basis::BasisSet;
use calchem_model::{Atom,Molecule,ScfOptions};
use calchem_opt::{Reference,TsOptions,IrcOptions,IrcDirection,search_transition_state,run_irc};
use calchem_vib::masses::atomic_mass_amu;

#[test]
fn irc_coordinates_energy_and_gradient_are_synchronized() {
    let mol=Molecule::new(vec![
        Atom::from_bohr("C",[0.0,0.0,0.0]).unwrap(),
        Atom::from_bohr("N",[2.308,0.0,0.0]).unwrap(),
        Atom::from_bohr("H",[0.672,2.171,0.0]).unwrap(),
    ],0,1);
    let ts=search_transition_state(&mol,"sto-3g",&TsOptions{max_steps:15,..Default::default()}).unwrap();
    assert!(ts.converged && ts.imaginary_frequency_cm < -500.0);
    let scf=ScfOptions{conv_energy:1e-11,conv_density:1e-10,..Default::default()};
    let options=IrcOptions{max_points:3,max_gs_iter:6,convergence_tol:1e-3,scf_options:scf.clone(),..Default::default()};
    let irc=run_irc(&ts.molecule,"sto-3g",&ts.transition_vector,&options).unwrap();
    let mut count=0;
    for branch in [irc.forward.as_ref(),irc.reverse.as_ref()].into_iter().flatten() {
        for point in &branch.points {
            let bs=BasisSet::from_name("sto-3g",&point.molecule).unwrap();
            let (e,g)=calchem_opt::scf_energy_gradient(&point.molecule,&bs,Reference::Rhf,options.grid_level,&scf).unwrap();
            let norm=g.iter().zip(&point.molecule.atoms).map(|(g,a)|g.iter().map(|x|x*x).sum::<f64>()/atomic_mass_amu(a.z)).sum::<f64>().sqrt();
            assert!((e-point.energy_hartree).abs()<1e-10);
            assert!((norm-point.gradient_norm).abs()<1e-10);
            count+=1;
        }
        assert!(!branch.converged_to_minimum,"Only 3 points cannot establish the reaction endpoint here");
    }
    assert!(count>=2);
    assert!(irc.barrier_forward_kcal_mol.is_none());
    assert!(irc.reaction_energy_kcal_mol.is_none());
    let invalid=IrcOptions{step_size_bohr:0.0,direction:IrcDirection::Forward,..options};
    assert!(run_irc(&ts.molecule,"sto-3g",&ts.transition_vector,&invalid).is_err());
}
