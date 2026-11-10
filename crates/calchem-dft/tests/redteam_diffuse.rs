use calchem_basis::BasisSet;
use calchem_dft::{GridLevel,MolecularGrid};
use calchem_model::{Atom,Molecule};
use std::f64::consts::PI;
#[test]
fn diffuse_gaussian_density_conserves_one_electron() {
    let mol=Molecule::new(vec![Atom::from_bohr("H",[0.0;3]).unwrap()],0,2);
    for alpha in [1e-4_f64,1e-6_f64] {
        let json=format!(r#"{{"name":"diffuse","elements":{{"H":[{{"shell_type":"s","exponents":[{alpha}],"coefficients":[1.0]}}]}}}}"#);
        let bs=BasisSet::from_json_str(&json,&mol).unwrap();
        let grid=MolecularGrid::build_for_basis(&mol,&bs,GridLevel::Fine);
        let norm=(2.0*alpha/PI).powf(1.5);
        let q:f64=grid.points.iter().map(|p|p.weight*norm*(-2.0*alpha*p.coords.iter().map(|x|x*x).sum::<f64>()).exp()).sum();
        assert!((q-1.0).abs()<1e-6,"alpha={alpha}, q={q:.16}");
    }
}
