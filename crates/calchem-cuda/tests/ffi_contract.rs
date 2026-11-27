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

#[test]
fn consecutive_calls_observe_changed_middle_elements() {
    if !is_cuda_available() {
        return;
    }
    let mut b = [1.0, 2.0, 3.0, 4.0];
    let mut v = [0.0; 2];
    assert!(calchem_cuda::cuda_rij_contract_v(
        &b, &[1.0; 2], &mut v, 2, 2
    ));
    assert_eq!(v, [3.0, 7.0]);
    b[1] = 12.0;
    let mut j = [0.0; 2];
    assert!(cuda_rij_contract_coulomb(&b, &[1.0; 2], &mut j, 2, 2));
    assert_eq!(j, [4.0, 16.0]);

    let one = calchem_cuda::ResidentRijTensor::new(b.to_vec());
    b[2] = 13.0;
    let two = calchem_cuda::ResidentRijTensor::new(b.to_vec());
    for (tensor, expected) in [
        (&one, [13.0, 7.0]),
        (&two, [13.0, 17.0]),
        (&one, [13.0, 7.0]),
    ] {
        assert!(tensor.view().contract_v(&[1.0; 2], &mut v, 2, 2));
        assert_eq!(v, expected);
    }

    drop(one);
    drop(two);
    // Neither transient (tensor_id == 0) nor ResidentRijTensor allocations should remain
    // after dropping all tensors (only static lookup tables <= 256 KiB may persist).
    let after_drop = calchem_cuda::total_allocated_gpu_bytes();
    assert!(
        after_drop <= 256 * 1024,
        "RI-J tensor/scratch bytes leaked after drop: {after_drop}"
    );
}
