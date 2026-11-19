import os
import time
import psutil
import calchem._core as c
from calchem.workflows.opv import get_real_opv_system

ANG_TO_BOHR = 1.0 / 0.529177210903
mol = get_real_opv_system("pm6_l8bo_cluster")
spec = mol.to_spec()
coords_bohr = [[x * ANG_TO_BOHR for x in row] for row in spec.coordinates_angstrom]

vram_before = c.get_cuda_memory_info()
t0 = time.perf_counter()
wf = c.run_single_point(
    spec.symbols,
    coords_bohr,
    0,
    1,
    "rks",
    "sto-3g",
    max_iter=80,
    conv_energy=1e-6,
    conv_density=1e-5,
    xc="b3lyp",
    grid_level="coarse",
    density_fitting=True,
    use_gpu=True,
)
dt = time.perf_counter() - t0
vram_after = c.get_cuda_memory_info()
rss_mb = psutil.Process(os.getpid()).memory_info().peak_wset / (1024.0 * 1024.0)

print(
    f"[60-ATOM STANDALONE] n_atoms={len(spec.symbols)} n_ao={wf.n_ao} "
    f"converged={wf.converged} iters={wf.iterations} E={wf.total_energy_hartree:.10f} "
    f"kernels={wf.gpu_kernel_invocations} fallbacks={wf.gpu_fallback_invocations} "
    f"walltime={dt:.2f}s peak_rss={rss_mb:.1f}MB "
    f"vram_before={vram_before} vram_after={vram_after}"
)
