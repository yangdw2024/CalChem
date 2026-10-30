use calchem_basis::MoleculeCore;
use calchem_tensor::PackedSymMatrix;

pub fn compute_sad_guess(_mol: &MoleculeCore, nbf: usize) -> PackedSymMatrix {
    PackedSymMatrix::zeros(nbf)
}
