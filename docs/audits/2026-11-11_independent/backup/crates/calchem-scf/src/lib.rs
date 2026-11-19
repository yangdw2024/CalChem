pub mod diis;
pub(crate) mod fock_builder;
pub mod rhf;
pub mod rks;
pub mod uhf;
pub mod uks;

pub use diis::DiisState;
pub use rhf::rhf;
pub use rks::{rks, rks_with_grid};
pub use uhf::{uhf, uhf_s_squared};
pub use uks::{uks, uks_with_grid};

/// Determines whether Direct SCF (on-the-fly contraction) should be used.
pub fn should_use_direct(
    _basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
) -> bool {
    match opts.direct {
        Some(d) => d,
        None => false,
    }
}

/// Conservative admission bound for Direct SCF (O(N_AO^2) working memory).
pub fn validate_direct_resources_and_options(
    basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
) -> Result<(), calchem_model::CalcError> {
    validate_direct_resources_for_method(basis, opts, false)
}

/// Detailed admission bound for Direct SCF taking method type (open-shell vs closed-shell),
/// state matrices, DIIS history vectors, and parallel accumulator scratch into account.
pub fn validate_direct_resources_for_method(
    basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
    is_open_shell: bool,
) -> Result<(), calchem_model::CalcError> {
    use calchem_model::CalcError;
    if opts.max_iter == 0
        || !opts.conv_energy.is_finite()
        || opts.conv_energy <= 0.0
        || !opts.conv_density.is_finite()
        || opts.conv_density <= 0.0
        || !opts.damping.is_finite()
        || !(0.0..1.0).contains(&opts.damping)
        || opts.diis_max_vecs > 64
    {
        return Err(CalcError::InvalidMolecule(
            "Invalid SCF iteration/tolerance/damping/DIIS options".into(),
        ));
    }
    let n = basis.n_ao as u128;
    if n > 65535 {
        return Err(CalcError::UnsupportedBasis(
            "AO count exceeds supported limit (65535)".into(),
        ));
    }

    let state_matrices = if is_open_shell { 18u128 } else { 10u128 };
    let diis_matrices = (opts.diis_max_vecs as u128) * (if is_open_shell { 4 } else { 2 });
    let threads = rayon::current_num_threads().max(1) as u128;
    let worker_channels = if is_open_shell { 2u128 } else { 1u128 };
    let accumulator_matrices = threads * worker_channels;

    let total_matrices = state_matrices + diis_matrices + accumulator_matrices;
    let estimated_bytes = total_matrices * n * n * 8;

    let budget_bytes: u128 = match std::env::var("CALCHEM_MAX_MEMORY_BYTES")
        .or_else(|_| std::env::var("CALCHEM_MAX_DIRECT_BYTES"))
    {
        Ok(v) => v.parse::<u128>().ok().filter(|v| *v > 0).ok_or_else(|| {
            CalcError::InvalidMolecule("CALCHEM_MAX_MEMORY_BYTES must be a positive integer".into())
        })?,
        Err(_) => 8 * 1024 * 1024 * 1024, // 8 GiB safe default for single-node workstation
    };

    if estimated_bytes > budget_bytes {
        let min_required_bytes = (state_matrices + diis_matrices + worker_channels) * n * n * 8;
        if min_required_bytes > budget_bytes {
            return Err(CalcError::UnsupportedBasis(format!(
                "Direct-SCF estimated memory requirement ({:.2} MB) exceeds budget ({:.2} MB). \
                 Even single-threaded execution requires {:.2} MB for N_AO={n}.",
                estimated_bytes as f64 / (1024.0 * 1024.0),
                budget_bytes as f64 / (1024.0 * 1024.0),
                min_required_bytes as f64 / (1024.0 * 1024.0),
            )));
        }
    }
    Ok(())
}

/// Conservative admission bound before stored ERIs are allocated. This is not a
/// runtime allocator quota; callers may choose a larger explicit E/D-host budget.
pub fn validate_resources_and_options(
    basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
) -> Result<(), calchem_model::CalcError> {
    use calchem_model::CalcError;
    if opts.direct == Some(true) {
        return validate_direct_resources_and_options(basis, opts);
    }
    if opts.max_iter == 0
        || !opts.conv_energy.is_finite()
        || opts.conv_energy <= 0.0
        || !opts.conv_density.is_finite()
        || opts.conv_density <= 0.0
        || !opts.damping.is_finite()
        || !(0.0..1.0).contains(&opts.damping)
        || opts.diis_max_vecs > 64
    {
        return Err(CalcError::InvalidMolecule(
            "Invalid SCF iteration/tolerance/damping/DIIS options".into(),
        ));
    }
    let n = basis.n_ao as u128;
    if n > 65535 {
        return Err(CalcError::UnsupportedBasis(
            "AO indices exceed the stored ERI index range".into(),
        ));
    }
    let pair = n * (n + 1) / 2;
    let shells = basis.shells.len() as u128;
    let sp = shells * (shells + 1) / 2;
    let estimated = if n <= 64 {
        n.pow(4) * 16
    } else {
        pair * (pair + 1) / 2 * 16 + sp * (sp + 1) / 2 * 32
    };
    let budget = match std::env::var("CALCHEM_MAX_ERI_BYTES") {
        Ok(v) => v.parse::<u128>().ok().filter(|v| *v > 0).ok_or_else(|| {
            CalcError::InvalidMolecule("CALCHEM_MAX_ERI_BYTES must be a positive integer".into())
        })?,
        Err(_) => 4 * 1024 * 1024 * 1024,
    };
    if estimated > budget {
        return Err(CalcError::UnsupportedBasis(format!(
            "Stored-ERI conservative memory estimate {estimated} bytes exceeds budget {budget}; reduce basis/system or explicitly set CALCHEM_MAX_ERI_BYTES. Direct/RI SCF is not implemented."
        )));
    }
    Ok(())
}
