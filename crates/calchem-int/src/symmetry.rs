/// Enumerates canonical shell quartets `[i, j, k, l]` satisfying `i <= j`, `k <= l`,
/// and composite pair index `ij <= kl`.
pub fn unique_shell_quartets(n_shells: usize) -> Vec<[usize; 4]> {
    shell_quartets(n_shells).collect()
}

/// Lazy canonical quartets: constant auxiliary storage.
pub fn shell_quartets(n_shells: usize) -> impl Iterator<Item = [usize; 4]> {
    (0..n_shells).flat_map(move |i| (i..n_shells).flat_map(move |j| {
        (i..n_shells).flat_map(move |k| {
            let start = if k == i { j } else { k };
            (start..n_shells).map(move |l| [i, j, k, l])
        })
    }))
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
