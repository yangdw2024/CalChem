// Resolution of Identity (RI-J / RI-K / RI-MP2)
use calchem_basis::BasisSetData;
use calchem_tensor::PackedSymMatrix;

pub struct RiEngine {
    pub n_aux: usize,
}

impl RiEngine {
    pub fn contract_j(&self, _b_tensor: &[f64], _dm: &PackedSymMatrix) -> PackedSymMatrix {
        PackedSymMatrix::zeros(_dm.dim)
    }
}
