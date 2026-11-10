use calchem_basis::BasisSet;
use calchem_int::{compute_eri, compute_kinetic, compute_nuclear, compute_overlap};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rhf;

fn main() {
    let h2 = Molecule::new(vec![
        Atom::from_bohr("H", [0.0, 0.0, 0.0]).unwrap(),
        Atom::from_bohr("H", [0.0, 0.0, 1.4]).unwrap(),
    ], 0, 1);
    let basis = BasisSet::sto_3g(&h2).unwrap();
    let t = compute_kinetic(&basis);
    let s = compute_overlap(&basis);
    println!("{{\"check\":\"H2_integrals\",\"T01\":{:.17},\"S01\":{:.17}}}", t.get(0,1), s.get(0,1));
    let water = Molecule::new(vec![
        Atom::from_bohr("O", [0.0, 0.0, 0.0]).unwrap(),
        Atom::from_bohr("H", [0.0, 1.432, 0.0]).unwrap(),
        Atom::from_bohr("H", [1.112, -0.716, 0.0]).unwrap(),
    ], 0, 1);
    let basis = BasisSet::sto_3g(&water).unwrap();
    let t = compute_kinetic(&basis);
    let v = compute_nuclear(&basis, &water);
    let eri = compute_eri(&basis);
    let n = basis.n_ao;
    for (label, opts) in [("default", ScfOptions::default()), ("loose", ScfOptions {
        conv_energy: 1e-5, conv_density: 1e-3, ..Default::default()
    })] {
        let wf = rhf(&water, &basis, &opts).unwrap();
        let mut energy = water.nuclear_repulsion_energy();
        let mut max_fock_diff: f64 = 0.0;
        for mu in 0..n { for nu in 0..n {
            let h = t.get(mu,nu) + v.get(mu,nu);
            let mut f = h;
            for lam in 0..n { for sig in 0..n {
                f += wf.density_alpha[lam*n+sig] * (eri.get(mu,nu,lam,sig) - 0.5*eri.get(mu,lam,nu,sig));
            }}
            energy += 0.5 * wf.density_alpha[mu*n+nu] * (h+f);
            max_fock_diff = max_fock_diff.max((f - wf.fock_alpha[mu*n+nu]).abs());
        }}
        println!("{{\"check\":\"water_final_state\",\"options\":\"{}\",\"iterations\":{},\"energy_recomputed_minus_reported\":{:.17e},\"max_F_of_returned_P_minus_returned_F\":{:.17e}}}",
            label, wf.iterations, energy-wf.total_energy_hartree, max_fock_diff);
    }
}
