use calchem_cuda::{cuda_rij_contract_coulomb, is_cuda_available};

#[test]
fn safe_wrapper_rejects_invalid_lengths_and_overflows_before_ffi() {
    let mut j = [7.0, 9.0];
    assert!(!cuda_rij_contract_coulomb(&[1.0], &[1.0], &mut j, 1, 2));
    assert!(!cuda_rij_contract_coulomb(&[], &[], &mut j, usize::MAX, 2));
    assert!(!cuda_rij_contract_coulomb(&[], &[], &mut j, 0, 2));
    assert_eq!(j, [7.0, 9.0]);
}

#[test]
fn gpu_result_or_unchanged_cpu_fallback_contract() {
    let available = is_cuda_available();
    let mut j = [7.0, 9.0];
    let success = cuda_rij_contract_coulomb(&[1.0, 2.0, 3.0, 4.0], &[0.5, 2.0], &mut j, 2, 2);
    println!("CUDA available={available}, kernel executed={success}, output={j:?}");
    if success {
        assert_eq!(j, [13.5, 18.0]);
    } else {
        assert_eq!(j, [7.0, 9.0]);
    }
}
