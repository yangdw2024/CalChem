use std::time::Instant;
use calchem_basis::BasisSet;
use calchem_dft::{evaluate_vxc_rks, GridLevel, MolecularGrid, XcFunctional};
use calchem_int::{compute_overlap, DirectFockEngine, RiJEngine};
use calchem_model::{Atom, Molecule, ScfOptions};
use calchem_scf::rks;
use calchem_tensor::Matrix2D;

const ANG_TO_BOHR: f64 = 1.0 / 0.529177210903;

fn pm6_l8bo_60_atoms() -> Molecule {
    let syms = [
        "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "O", "O", "S",
        "S", "S", "C", "C", "S", "C", "C", "C", "C", "C", "H", "C", "C", "C", "C", "C",
        "H", "H", "C", "O", "C", "C", "N", "C", "N", "C", "C", "C", "C", "C", "H", "C",
        "H", "F", "F", "H", "H", "H", "H", "H", "H", "H", "H", "H",
    ];
    let coords_ang = [
        [-0.005689, -4.088235, -1.339131],
        [-1.745687, -4.608236, 0.600869],
        [-2.045690, -4.068235, 2.010869],
        [-1.645688, -2.798235, 2.480869],
        [-0.925687, -1.898236, 1.550869],
        [-0.315687, -2.528235, 0.370869],
        [-0.645685, -3.808236, -0.089131],
        [-2.095685, -2.488235, 3.740869],
        [0.564314, -1.838235, -0.509131],
        [-2.525686, -4.838236, 3.050869],
        [0.414317, -7.818236, -2.549131],
        [0.164313, -5.398235, -1.839131],
        [0.704314, -6.918236, -3.549131],
        [-2.345685, -5.528236, 0.120869],
        [-0.895685, -0.698235, 1.760869],
        [-0.045686, -6.988236, -1.089131],
        [0.654311, -2.638236, -2.049131],
        [-2.525686, -3.958236, 4.590869],
        [1.474314, -0.738235, -0.149131],
        [2.324313, -0.358235, -1.149131],
        [1.954314, -0.258235, 1.450869],
        [3.224314, 0.661765, -0.669131],
        [3.104315, 0.841765, 0.700869],
        [0.574313, -5.518236, -3.139131],
        [2.534312, 6.301766, 1.750869],
        [1.824313, 5.221766, 1.060869],
        [2.514315, 6.401766, 2.850869],
        [1.844317, 4.881764, -0.419131],
        [1.014315, 3.761765, -0.789131],
        [0.714312, 3.161765, -1.969131],
        [0.454310, 3.321764, 0.440869],
        [-0.395688, 2.221765, 0.340869],
        [1.204314, 3.661765, -2.879131],
        [-0.885686, 1.751765, 1.190869],
        [0.944316, 4.211764, 1.590869],
        [2.484313, 5.541764, -1.249131],
        [0.554316, 4.091765, 2.900869],
        [0.824313, 5.021765, 3.910869],
        [0.984313, 5.771765, 4.770869],
        [-0.355688, 3.101765, 3.340869],
        [-1.115682, 2.291765, 3.700869],
        [-0.695684, 1.601765, -0.929131],
        [-0.115686, 2.121764, -2.129131],
        [-1.615686, 0.561765, -1.059131],
        [-0.425687, 1.611765, -3.409131],
        [-1.885686, 0.021765, -2.279131],
        [-2.115690, 0.091765, -0.229131],
        [-1.295686, 0.531765, -3.489131],
        [0.074317, 2.041764, -4.341285],
        [-1.705686, -0.058235, -4.591292],
        [-2.755685, -0.968235, -2.399131],
        [-1.893463, -1.630538, 4.382399],
        [-3.085894, -5.767188, 2.944500],
        [0.225472, -8.864185, -2.790849],
        [0.838553, -7.197039, -4.594286],
        [2.194124, -0.568545, -2.210696],
        [3.993623, 1.191963, -1.230518],
        [3.798749, 1.558841, 1.138663],
        [0.685137, -4.712244, -3.864524],
        [3.126459, 7.163751, 1.443553],
    ];
    let atoms = syms
        .iter()
        .zip(coords_ang.iter())
        .map(|(s, c)| {
            Atom::from_bohr(s, [c[0] * ANG_TO_BOHR, c[1] * ANG_TO_BOHR, c[2] * ANG_TO_BOHR])
                .unwrap()
        })
        .collect();
    Molecule::new(atoms, 0, 1)
}

#[test]
#[ignore = "60-atom benchmark profiler; run explicitly with --ignored"]
fn profile_60atom_pm6_l8bo_cluster_stages() {
    let mol = pm6_l8bo_60_atoms();
    let basis = BasisSet::from_name("sto-3g", &mol).unwrap();
    let n_ao = basis.n_ao;
    println!("[60-ATOM] n_atoms={} n_ao={}", mol.atoms.len(), n_ao);

    let t0 = Instant::now();
    let mut rij = RiJEngine::new(&mol, &basis).unwrap();
    let t_rij_init = t0.elapsed().as_secs_f64();
    let cached_mb = rij
        .cached_b
        .as_ref()
        .map(|b| (b.len() * 8) as f64 / 1048576.0)
        .unwrap_or(0.0);
    println!(
        "[STAGE 1] RiJEngine::new: n_aux={} trunc={} cached_b={:.1}MB time={:.2}s",
        rij.aux_basis.n_aux, rij.metric_truncated_modes, cached_mb, t_rij_init
    );

    let p_dummy = Matrix2D::identity(n_ao);

    let t1 = Instant::now();
    let _j_cpu = rij.build_coulomb(&basis, &p_dummy);
    let t_j_cpu = t1.elapsed().as_secs_f64();

    rij.use_gpu = true;
    let t2 = Instant::now();
    let _j_gpu = rij.build_coulomb(&basis, &p_dummy);
    let t_j_gpu = t2.elapsed().as_secs_f64();
    println!(
        "[STAGE 2] build_coulomb (1 iter): cpu={:.3}s gpu={:.3}s (kernels={}, fallbacks={})",
        t_j_cpu,
        t_j_gpu,
        rij.gpu_kernel_invocations.load(std::sync::atomic::Ordering::Relaxed),
        rij.gpu_fallback_invocations.load(std::sync::atomic::Ordering::Relaxed),
    );

    let t3 = Instant::now();
    let df = DirectFockEngine::new(&basis, 1e-11);
    let t_df_init = t3.elapsed().as_secs_f64();
    let t4 = Instant::now();
    let _k = df.build_k(&basis, &p_dummy.data);
    let t_k_iter = t4.elapsed().as_secs_f64();
    println!(
        "[STAGE 3] DirectFock K (1 iter): init={:.3}s build_k={:.3}s",
        t_df_init, t_k_iter
    );

    let t5 = Instant::now();
    let grid = MolecularGrid::build_for_basis(&mol, &basis, GridLevel::Coarse);
    let t_grid_build = t5.elapsed().as_secs_f64();
    let t6 = Instant::now();
    let (_exc, _vxc) = evaluate_vxc_rks(&basis, &grid, XcFunctional::B3lyp, &p_dummy);
    let t_vxc_iter = t6.elapsed().as_secs_f64();
    println!(
        "[STAGE 4] DFT Coarse grid: build={:.3}s vxc(1 iter)={:.3}s",
        t_grid_build, t_vxc_iter
    );

    let _ = compute_overlap(&basis);

    // Full SCF run with RI-J + GPU
    let opts = ScfOptions {
        max_iter: 80,
        conv_energy: 1e-6,
        conv_density: 1e-5,
        density_fitting: true,
        use_gpu: true,
        ..Default::default()
    };
    let t_scf = Instant::now();
    let wf = rks(&mol, &basis, XcFunctional::B3lyp, GridLevel::Coarse, &opts).unwrap();
    let scf_s = t_scf.elapsed().as_secs_f64();
    println!(
        "[FULL SCF] converged={} iters={} E={:.10} kernels={} fallbacks={} walltime={:.2}s",
        wf.converged,
        wf.iterations,
        wf.total_energy_hartree,
        wf.diagnostics.gpu_kernel_invocations,
        wf.diagnostics.gpu_fallback_invocations,
        scf_s
    );
    assert!(wf.converged);
}
