use calchem_basis::BasisSet;
use calchem_int::direct_fock::DirectFockEngine;
use calchem_model::{Atom, Molecule};
use std::time::Instant;

fn make_ne_cluster(n_atoms: usize) -> (Molecule, BasisSet) {
    let atoms: Vec<Atom> = (0..n_atoms)
        .map(|i| Atom::from_angstrom("Ne", [i as f64 * 3.2, 0.0, 0.0]).unwrap())
        .collect();
    let mol = Molecule::new(atoms, 0, 1);
    let bs = BasisSet::from_name("6-31g", &mol).unwrap();
    (mol, bs)
}

fn make_density(n: usize) -> Vec<f64> {
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        p[i * n + i] = 1.0 / (i as f64 + 1.0);
        for j in 0..i {
            let v = 0.05 / ((i + j) as f64 + 1.0);
            p[i * n + j] = v;
            p[j * n + i] = v;
        }
    }
    p
}

#[test]
fn test_bounded_memory_fock_thread_scaling_and_determinism() {
    // 8 Ne atoms in 6-31G -> 72 AOs, 32 shells, 528 shell pairs, 139,656 quartet candidates
    let (_mol, bs) = make_ne_cluster(8);
    let n = bs.n_ao;
    assert!(n >= 72);
    let p = make_density(n);
    let exx = 0.20; // B3LYP
    let thresh = 1e-10;

    let engine = DirectFockEngine::new(&bs, thresh);
    let single_buffer_bytes = n * n * 8;

    println!("\n=== Bounded Fock Accumulation Benchmark: N_AO = {n} ===");
    println!("Threads,AllocatedBuffers,BufferMemoryKB,WallTimeMs,MaxDiffVsSerial,ComputedQuartets,SkippedQuartets");

    // Reference calculation with 1 worker (serial)
    let t0 = Instant::now();
    let (g_serial, stats_serial) = engine.build_rks_bounded(&bs, &p, exx, thresh, 1);
    let elapsed_serial = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "1,1,{:.2},{:.2},0.000e0,{},{}",
        single_buffer_bytes as f64 / 1024.0,
        elapsed_serial,
        stats_serial.computed_quartets,
        stats_serial.skipped_quartets
    );

    // Test thread counts: 1, 4, 8, 16
    for &k_threads in &[4, 8, 16] {
        let t_start = Instant::now();
        let (g_par, stats_par) = engine.build_rks_bounded(&bs, &p, exx, thresh, k_threads);
        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let total_buffer_bytes = k_threads * single_buffer_bytes;

        let mut max_diff = 0.0f64;
        for i in 0..n * n {
            let d = (g_serial[i] - g_par[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }

        println!(
            "{},{},{:.2},{:.2},{:.3e},{},{}",
            k_threads,
            k_threads,
            total_buffer_bytes as f64 / 1024.0,
            elapsed_ms,
            max_diff,
            stats_par.computed_quartets,
            stats_par.skipped_quartets
        );

        // Verification of mathematical equivalence across thread counts
        assert!(
            max_diff < 1e-12,
            "Thread {k_threads} differed from serial by {max_diff:.3e} >= 1e-12"
        );
        assert_eq!(
            stats_serial.computed_quartets, stats_par.computed_quartets,
            "Computed quartet counts must be identical across thread counts"
        );
        assert_eq!(
            stats_serial.skipped_quartets, stats_par.skipped_quartets,
            "Skipped quartet counts must be identical across thread counts"
        );
    }
}
