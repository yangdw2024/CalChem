use crate::eri::eri_shell_quartet;
use calchem_basis::BasisSet;

/// Computes the shell-pair upper bounds matrix `B_{ab} = max_(mu in a, nu in b) sqrt(max(0, (mu nu | mu nu)))`.
/// Returns a flattened `n_shells * n_shells` row-major vector.
pub fn shell_pair_bounds(basis: &BasisSet) -> Vec<f64> {
    let n = basis.shells.len();
    let mut bounds = vec![0.0; n * n];

    for i in 0..n {
        for j in i..n {
            let sh_i = &basis.shells[i];
            let sh_j = &basis.shells[j];
            let block = eri_shell_quartet(sh_i, sh_j, sh_i, sh_j);
            let ni = sh_i.n_functions();
            let nj = sh_j.n_functions();

            let mut max_val = 0.0f64;
            for ii in 0..ni {
                for jj in 0..nj {
                    // Index: ((ii * nj + jj) * ni + ii) * nj + jj
                    let idx = ((ii * nj + jj) * ni + ii) * nj + jj;
                    let val = block[idx].max(0.0);
                    if val > max_val {
                        max_val = val;
                    }
                }
            }

            let b_val = max_val.sqrt();
            bounds[i * n + j] = b_val;
            bounds[j * n + i] = b_val;
        }
    }

    bounds
}

/// Cauchy-Schwarz integral screening state for 4-center electron repulsion integrals.
#[derive(Debug, Clone)]
pub struct SchwarzScreening {
    pub n_shells: usize,
    pub bounds: Vec<f64>,
    pub threshold: f64,
}

impl SchwarzScreening {
    pub fn new(basis: &BasisSet, threshold: f64) -> Self {
        let n_shells = basis.shells.len();
        let bounds = shell_pair_bounds(basis);
        Self {
            n_shells,
            bounds,
            threshold,
        }
    }

    #[inline]
    pub fn bound(&self, a: usize, b: usize) -> f64 {
        self.bounds[a * self.n_shells + b]
    }

    #[inline]
    pub fn can_skip(&self, a: usize, b: usize, c: usize, d: usize) -> bool {
        let bab = self.bounds[a * self.n_shells + b];
        let bcd = self.bounds[c * self.n_shells + d];
        bab * bcd < self.threshold
    }

    /// Counts how many quartets in the provided slice would be skipped by this screening threshold.
    pub fn count_skipped(&self, quartets: &[[usize; 4]]) -> usize {
        quartets
            .iter()
            .filter(|&&[a, b, c, d]| self.can_skip(a, b, c, d))
            .count()
    }
}
