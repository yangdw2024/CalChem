use calchem_basis::MoleculeCore;
use calchem_scf::ScfOutput;

pub fn compute_analytical_gradient(
    mol: &MoleculeCore,
    _scf: &ScfOutput,
) -> Vec<[f64; 3]> {
    let natom = mol.atoms.len();
    vec![[0.0; 3]; natom]
}
