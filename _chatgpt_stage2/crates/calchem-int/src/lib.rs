pub mod boys;
pub mod eri;
pub mod one_electron;
pub mod primitive;
pub mod symmetry;

use calchem_basis::BasisSet;
use calchem_tensor::Tensor4D;

pub use boys::{boys, boys_batch, erf_approx};
pub use eri::{eri_primitive, eri_shell_quartet, EriPairData};
pub use one_electron::{
    compute_kinetic, compute_nuclear, compute_overlap, shell_kinetic_block, shell_nuclear_block,
    shell_overlap_block,
};
pub use primitive::{kinetic_primitive, nuclear_vrr, overlap_1d, overlap_primitive, GaussianPair};
pub use symmetry::{eri_permutations, unique_shell_quartets};

/// Backward-compatible alias for `erf_approx`.
pub fn erf(x: f64) -> f64 {
    erf_approx(x)
}

/// Evaluates the full 4-index AO electron repulsion integral tensor `(mu nu | lam sig)`
/// exploiting 8-fold shell quartet symmetry.
pub fn compute_eri(basis: &BasisSet) -> Tensor4D {
    let n = basis.n_ao;
    let mut eri = Tensor4D::zeros(n);
    let quartets = unique_shell_quartets(basis.shells.len());

    for [i, j, k, l] in quartets {
        let sh_a = &basis.shells[i];
        let sh_b = &basis.shells[j];
        let sh_c = &basis.shells[k];
        let sh_d = &basis.shells[l];
        let block = eri_shell_quartet(sh_a, sh_b, sh_c, sh_d);

        let na = sh_a.n_functions();
        let nb = sh_b.n_functions();
        let nc = sh_c.n_functions();
        let nd = sh_d.n_functions();

        for ia in 0..na {
            let mu = sh_a.ao_offset + ia;
            for ib in 0..nb {
                let nu = sh_b.ao_offset + ib;
                for ic in 0..nc {
                    let lam = sh_c.ao_offset + ic;
                    for id in 0..nd {
                        let sig = sh_d.ao_offset + id;
                        let val = block[((ia * nb + ib) * nc + ic) * nd + id];
                        for [p, q, r, s] in eri_permutations(mu, nu, lam, sig) {
                            eri.set(p, q, r, s, val);
                        }
                    }
                }
            }
        }
    }

    eri
}
