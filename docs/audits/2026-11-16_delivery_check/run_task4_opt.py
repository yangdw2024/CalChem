import os
import time
import psutil
import calchem._core as c

ANG_TO_BOHR = 1.0 / 0.529177210903

# Slightly perturbed CH3CN (C-N stretched by +0.03 A, C-C compressed by -0.02 A)
sym = ["C", "C", "N", "H", "H", "H"]
coords_ang = [
    [0.0000, 0.0000, -1.1580],
    [0.0000, 0.0000, 0.2810],
    [0.0000, 0.0000, 1.4680],
    [0.0000, 1.0280, -1.5450],
    [0.8903, -0.5140, -1.5450],
    [-0.8903, -0.5140, -1.5450],
]
coords_bohr = [[x * ANG_TO_BOHR for x in r] for r in coords_ang]

for df in (False, True):
    t0 = time.perf_counter()
    res = c.run_optimize(
        sym,
        coords_bohr,
        basis="6-31g(d)",
        max_steps=40,
        charge=0,
        multiplicity=1,
        method="rks",
        xc="b3lyp",
        grid_level="medium",
        density_fitting=df,
        use_gpu=False,
    )
    dt = time.perf_counter() - t0
    rss = psutil.Process(os.getpid()).memory_info().peak_wset / (1024.0 * 1024.0)
    print(
        f"[CH3CN OPT df={df}] converged={res.converged} steps={res.steps} "
        f"E_final={res.energy:.10f} time={dt:.2f}s peak_rss={rss:.1f}MB"
    )
