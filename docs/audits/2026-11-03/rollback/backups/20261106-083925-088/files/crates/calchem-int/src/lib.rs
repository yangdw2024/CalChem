pub mod boys;
pub mod eri;
pub mod one_electron;
pub mod primitive;
pub mod screening;
pub mod symmetry;

use calchem_basis::BasisSet;
use calchem_tensor::Tensor4D;
use rayon::prelude::*;

pub use boys::{boys, boys_batch, erf_approx};
pub use eri::{eri_primitive, eri_shell_quartet, EriPairData};
pub use one_electron::{
    compute_kinetic, compute_nuclear, compute_overlap, shell_kinetic_block, shell_nuclear_block,
    shell_overlap_block,
};
pub use primitive::{kinetic_primitive, nuclear_vrr, overlap_1d, overlap_primitive, GaussianPair};
pub use screening::{shell_pair_bounds, SchwarzScreening};
pub use symmetry::{eri_permutations, unique_shell_quartets};

/// Options controlling electron repulsion integral evaluation (screening, parallelism).
#[derive(Debug, Clone)]
pub struct IntegralOptions {
    pub screen_threshold: f64,
    pub use_screening: bool,
    pub use_parallel: bool,
    pub parallel_threshold: usize,
}

impl Default for IntegralOptions {
    fn default() -> Self {
        Self {
            screen_threshold: 1e-10,
            use_screening: true,
            use_parallel: true,
            parallel_threshold: 4,
        }
    }
}

/// Backward-compatible alias for `erf_approx`.
pub fn erf(x: f64) -> f64 {
    erf_approx(x)
}

#[inline]
fn scatter_quartet_block(
    sh_a: &calchem_basis::Shell,
    sh_b: &calchem_basis::Shell,
    sh_c: &calchem_basis::Shell,
    sh_d: &calchem_basis::Shell,
    block: &[f64],
    eri: &mut Tensor4D,
) {
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
                    if val.abs() > 1e-16 {
                        for [p, q, r, s] in eri_permutations(mu, nu, lam, sig) {
                            eri.set(p, q, r, s, val);
                        }
                    }
                }
            }
        }
    }
}

/// Evaluates the full 4-index AO electron repulsion integral tensor `(mu nu | lam sig)`
/// with options for Schwarz screening and Rayon multithreading.
pub fn compute_eri_with_options(basis: &BasisSet, opts: &IntegralOptions) -> Tensor4D {
    let n = basis.n_ao;
    let mut eri = Tensor4D::zeros(n);
    let quartets = unique_shell_quartets(basis.shells.len());

    let screening = if opts.use_screening {
        Some(SchwarzScreening::new(basis, opts.screen_threshold))
    } else {
        None
    };

    if opts.use_parallel && basis.shells.len() >= opts.parallel_threshold {
        let blocks: Vec<([usize; 4], Vec<f64>)> = quartets
            .into_par_iter()
            .filter_map(|[i, j, k, l]| {
                if let Some(sc) = &screening {
                    if sc.can_skip(i, j, k, l) {
                        return None;
                    }
                }
                let block = eri_shell_quartet(
                    &basis.shells[i],
                    &basis.shells[j],
                    &basis.shells[k],
                    &basis.shells[l],
                );
                Some(([i, j, k, l], block))
            })
            .collect();

        for ([i, j, k, l], block) in blocks {
            scatter_quartet_block(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
                &block,
                &mut eri,
            );
        }
    } else {
        for [i, j, k, l] in quartets {
            if let Some(sc) = &screening {
                if sc.can_skip(i, j, k, l) {
                    continue;
                }
            }
            let block = eri_shell_quartet(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
            );
            scatter_quartet_block(
                &basis.shells[i],
                &basis.shells[j],
                &basis.shells[k],
                &basis.shells[l],
                &block,
                &mut eri,
            );
        }
    }

    eri
}

/// Evaluates the full 4-index AO electron repulsion integral tensor `(mu nu | lam sig)`
/// using the default screening and parallel settings.
pub fn compute_eri(basis: &BasisSet) -> Tensor4D {
    compute_eri_with_options(basis, &IntegralOptions::default())
}
