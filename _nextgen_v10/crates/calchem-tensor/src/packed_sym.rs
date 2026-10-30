#[derive(Debug, Clone)]
pub struct PackedSymMatrix {
    pub dim: usize,
    pub data: Vec<f64>,
}

impl PackedSymMatrix {
    pub fn zeros(dim: usize) -> Self {
        let len = dim * (dim + 1) / 2;
        Self {
            dim,
            data: vec![0.0; len],
        }
    }

    #[inline(always)]
    pub fn idx(i: usize, j: usize) -> usize {
        let (r, c) = if i <= j { (i, j) } else { (j, i) };
        c * (c + 1) / 2 + r
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[Self::idx(i, j)]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        let idx = Self::idx(i, j);
        self.data[idx] = val;
    }

    #[inline(always)]
    pub fn add_assign(&mut self, i: usize, j: usize, val: f64) {
        let idx = Self::idx(i, j);
        self.data[idx] += val;
    }

    pub fn axpy(&mut self, alpha: f64, other: &Self) {
        assert_eq!(self.dim, other.dim);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * b;
        }
    }

    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.dim;
        let mut dense = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dense[i * n + j] = self.get(i, j);
            }
        }
        dense
    }
}
