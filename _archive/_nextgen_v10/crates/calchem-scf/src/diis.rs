use calchem_tensor::PackedSymMatrix;

pub struct DiisManager {
    pub max_subspace: usize,
    pub err_history: Vec<PackedSymMatrix>,
    pub fock_history: Vec<PackedSymMatrix>,
}

impl DiisManager {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace,
            err_history: Vec::new(),
            fock_history: Vec::new(),
        }
    }
}
