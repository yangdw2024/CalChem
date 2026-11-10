pub mod diis;
pub mod rhf;
pub mod rks;
pub mod uhf;
pub mod uks;

pub use diis::DiisState;
pub use rhf::rhf;
pub use rks::{rks, rks_with_grid};
pub use uhf::{uhf, uhf_s_squared};
pub use uks::{uks, uks_with_grid};

/// Conservative admission bound before stored ERIs are allocated. This is not a
/// runtime allocator quota; callers may choose a larger explicit E/D-host budget.
pub fn validate_resources_and_options(
    basis: &calchem_basis::BasisSet,
    opts: &calchem_model::ScfOptions,
) -> Result<(), calchem_model::CalcError> {
    use calchem_model::CalcError;
    if opts.max_iter == 0 || !opts.conv_energy.is_finite() || opts.conv_energy <= 0.0
        || !opts.conv_density.is_finite() || opts.conv_density <= 0.0
        || !opts.damping.is_finite() || !(0.0..1.0).contains(&opts.damping)
        || opts.diis_max_vecs > 64 {
        return Err(CalcError::InvalidMolecule("Invalid SCF iteration/tolerance/damping/DIIS options".into()));
    }
    let n=basis.n_ao as u128;
    if n>65535 {return Err(CalcError::UnsupportedBasis("AO indices exceed the stored ERI index range".into()));}
    let pair=n*(n+1)/2;
    let shells=basis.shells.len() as u128;
    let sp=shells*(shells+1)/2;
    let estimated=if n<=64 {n.pow(4)*16} else {pair*(pair+1)/2*16+sp*(sp+1)/2*32};
    let budget=match std::env::var("CALCHEM_MAX_ERI_BYTES") {
        Ok(v)=>v.parse::<u128>().ok().filter(|v|*v>0)
            .ok_or_else(||CalcError::InvalidMolecule("CALCHEM_MAX_ERI_BYTES must be a positive integer".into()))?,
        Err(_)=>4*1024*1024*1024,
    };
    if estimated>budget {
        return Err(CalcError::UnsupportedBasis(format!(
            "Stored-ERI conservative memory estimate {estimated} bytes exceeds budget {budget}; reduce basis/system or explicitly set CALCHEM_MAX_ERI_BYTES. Direct/RI SCF is not implemented."
        )));
    }
    Ok(())
}
