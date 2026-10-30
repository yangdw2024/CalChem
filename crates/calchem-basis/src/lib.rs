//! 基组层。

pub struct BasisSet {
    pub nbf: usize,
}

impl BasisSet {
    pub fn load_sto3g(_atoms: &[calchem_model::Atom]) -> Self {
        // 占位：实际从 data/sto-3g.json 加载
        Self { nbf: 0 }
    }
}
