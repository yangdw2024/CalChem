use calchem_scf::ScfOutput;
use calchem_tensor::PackedSymMatrix;

pub fn build_energy_weighted_density(scf: &ScfOutput) -> PackedSymMatrix {
    PackedSymMatrix::zeros(scf.density.dim)
}
