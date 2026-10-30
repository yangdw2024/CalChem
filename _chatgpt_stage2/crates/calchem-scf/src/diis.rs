use calchem_tensor::{solve_linear_system, Matrix2D};

#[derive(Debug, Clone)]
pub struct DiisState {
    pub max_vecs: usize,
    pub fock_list: Vec<Matrix2D>,
    pub err_list: Vec<Matrix2D>,
}

impl DiisState {
    pub fn new(max_vecs: usize) -> Self {
        Self {
            max_vecs: max_vecs.max(2),
            fock_list: Vec::new(),
            err_list: Vec::new(),
        }
    }

    pub fn push_and_extrapolate(&mut self, fock: &Matrix2D, err: &Matrix2D) -> Matrix2D {
        self.fock_list.push(fock.clone());
        self.err_list.push(err.clone());
        while self.fock_list.len() > self.max_vecs {
            self.fock_list.remove(0);
            self.err_list.remove(0);
        }

        if self.fock_list.len() < 2 {
            return fock.clone();
        }

        // Try solving the Pulay B-matrix system; if near-singular, drop oldest vector(s)
        // and retry before falling back to the unextrapolated Fock matrix.
        let total = self.fock_list.len();
        for start in 0..(total - 1) {
            let f_slice = &self.fock_list[start..];
            let e_slice = &self.err_list[start..];
            if let Some(extrap) = Self::try_extrapolate_slice(f_slice, e_slice) {
                return extrap;
            }
        }

        fock.clone()
    }

    fn try_extrapolate_slice(focks: &[Matrix2D], errs: &[Matrix2D]) -> Option<Matrix2D> {
        let m = focks.len();
        if m < 2 {
            return None;
        }
        let dim = m + 1;
        let mut b = Matrix2D::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        for i in 0..m {
            b.set(i, m, -1.0);
            b.set(m, i, -1.0);
            for j in i..m {
                let dot: f64 = errs[i]
                    .data
                    .iter()
                    .zip(errs[j].data.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                b.set(i, j, dot);
                b.set(j, i, dot);
            }
        }
        b.set(m, m, 0.0);
        rhs[m] = -1.0;

        let coeffs = solve_linear_system(&b, &rhs).ok()?;
        let weight_sum: f64 = coeffs.iter().take(m).sum();
        if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1e-6 {
            return None;
        }

        let n = focks[0].rows;
        let mut out = Matrix2D::zeros(n, n);
        for (k, fk) in focks.iter().enumerate() {
            let ck = coeffs[k];
            for idx in 0..out.data.len() {
                out.data[idx] += ck * fk.data[idx];
            }
        }
        Some(out)
    }
}
