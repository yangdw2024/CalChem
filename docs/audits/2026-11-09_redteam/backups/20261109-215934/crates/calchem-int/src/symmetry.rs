/// Enumerates canonical shell quartets `[i, j, k, l]` satisfying `i <= j`, `k <= l`,
/// and composite pair index `ij <= kl`.
pub fn unique_shell_quartets(n_shells: usize) -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for i in 0..n_shells {
        for j in i..n_shells {
            let ij = i * n_shells + j;
            for k in 0..n_shells {
                for l in k..n_shells {
                    let kl = k * n_shells + l;
                    if ij <= kl {
                        out.push([i, j, k, l]);
                    }
                }
            }
        }
    }
    out
}

/// Returns all 8 real-AO permutation indices of `(mu nu | lam sig)`:
/// `(mu nu | lam sig) = (nu mu | lam sig) = (mu nu | sig lam) = (nu mu | sig lam)`
/// `= (lam sig | mu nu) = (sig lam | mu nu) = (lam sig | nu mu) = (sig lam | nu mu)`.
pub fn eri_permutations(mu: usize, nu: usize, lam: usize, sig: usize) -> [[usize; 4]; 8] {
    [
        [mu, nu, lam, sig],
        [nu, mu, lam, sig],
        [mu, nu, sig, lam],
        [nu, mu, sig, lam],
        [lam, sig, mu, nu],
        [sig, lam, mu, nu],
        [lam, sig, nu, mu],
        [sig, lam, nu, mu],
    ]
}
