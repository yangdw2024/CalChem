use calchem_basis::BasisSetData;
use calchem_tensor::PackedSymMatrix;
use crate::screening::SchwarzScreener;

pub fn build_jk_direct_parallel(
    basis: &BasisSetData,
    _screener: &SchwarzScreener,
    _dm: &PackedSymMatrix,
    _k_scale: f64,
) -> (PackedSymMatrix, PackedSymMatrix) {
    let nbf = basis.nbf;
    (PackedSymMatrix::zeros(nbf), PackedSymMatrix::zeros(nbf))
}
