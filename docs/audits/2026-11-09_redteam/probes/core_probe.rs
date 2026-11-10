use calchem_basis::BasisSet;
use calchem_dft::{GridLevel, MolecularGrid};
use calchem_int::{compute_eri_with_options, compute_sparse_eri_with_options, IntegralOptions, SparseSymmetricEri};
use calchem_model::{Atom, Molecule};
use calchem_opt::{OptOptions, Reference, scf_energy_gradient_with_options, scf_energy_with_options};
use std::f64::consts::PI;

fn main() {
    let atoms = (0..65).map(|i| Atom::from_bohr("H", [i as f64 * 4.0, 0.17 * ((i % 3) as f64), 0.0]).unwrap()).collect();
    let mol = Molecule::new(atoms, 0, 2);
    let json = r#"{"name":"audit-one-s","elements":{"H":[{"shell_type":"s","exponents":[0.8],"coefficients":[1.0]}]}}"#;
    let bs=BasisSet::from_json_str(json,&mol).unwrap();
    let options=IntegralOptions {screen_threshold:1e-11,use_screening:true,use_parallel:true,parallel_threshold:4};
    let sparse=compute_sparse_eri_with_options(&bs,&options);
    let dense=SparseSymmetricEri {n_ao:bs.n_ao,dense:Some(compute_eri_with_options(&bs,&options)),entries:Vec::new()};
    let n=bs.n_ao;
    let mut p=vec![0.0;n*n];let mut x=vec![0.0;n*n];
    for i in 0..n { for j in 0..n {p[i*n+j]=((i+j+3) as f64).sin()/(1.0+i.abs_diff(j) as f64);x[i*n+j]=((i*3+j*7+1) as f64).cos();}}
    for exx in [0.0,0.2,1.0] {
        let a=sparse.contract_rks(&p,exx);let b=dense.contract_rks(&p,exx);
        let err=a.iter().zip(b.iter()).map(|(a,b)|(a-b).abs()).fold(0.0,f64::max);
        println!("{{\"probe\":\"sparse_vs_dense\",\"n_ao\":{n},\"entries\":{},\"exx\":{exx},\"max_abs\":{err:.17e}}}",sparse.entries.len());
    }
    let a=sparse.contract_k_gen(&x);let b=dense.contract_k_gen(&x);
    let err=a.iter().zip(b.iter()).map(|(a,b)|(a-b).abs()).fold(0.0,f64::max);
    println!("{{\"probe\":\"sparse_general_K\",\"max_abs\":{err:.17e}}}");
    drop(dense);drop(sparse);

    let one=Molecule::new(vec![Atom::from_bohr("H",[0.0,0.0,0.0]).unwrap()],0,2);
    let grid=MolecularGrid::build(&one,GridLevel::Fine);
    let alpha=1e-4;
    let density_norm=(2.0*alpha/PI).powf(1.5);
    let mut q=0.0;
    for p in &grid.points {let r2=p.coords.iter().map(|x|x*x).sum::<f64>();q+=p.weight*density_norm*(-2.0*alpha*r2).exp();}
    println!("{{\"probe\":\"diffuse_grid_charge\",\"exponent\":{alpha},\"exact\":1.0,\"integrated\":{q:.17e}}}");

    let water=Molecule::new(vec![Atom::from_bohr("O",[0.04,-0.03,0.05]).unwrap(),Atom::from_bohr("H",[0.07,1.51,-0.04]).unwrap(),Atom::from_bohr("H",[1.21,-0.66,0.09]).unwrap()],0,1);
    let bs=BasisSet::from_name("sto-3g",&water).unwrap();
    let opts=OptOptions {reference:Reference::Rhf,solvent:Some("water".to_string()),..Default::default()};
    let (_,g)=scf_energy_gradient_with_options(&water,&bs,&opts).unwrap();
    let h=1e-4;
    let mut maxerr=0.0f64;
    for atom in 0..3 {for axis in 0..3 {
        let mut plus=water.clone();let mut minus=water.clone();plus.atoms[atom].position_bohr[axis]+=h;minus.atoms[atom].position_bohr[axis]-=h;
        let bp=BasisSet::from_name("sto-3g",&plus).unwrap();let bm=BasisSet::from_name("sto-3g",&minus).unwrap();
        let fd=(scf_energy_with_options(&plus,&bp,&opts).unwrap()-scf_energy_with_options(&minus,&bm,&opts).unwrap())/(2.0*h);
        maxerr=maxerr.max((fd-g[atom][axis]).abs());
    }}
    println!("{{\"probe\":\"solvent_optimizer_gradient_fd\",\"step\":{h},\"max_abs\":{maxerr:.17e}}}");
}
