import csv
import os
import time
from pathlib import Path

import numpy as np
import psutil
import calchem._core as c
from calchem.workflows.opv import get_real_opv_system

OUT_DIR = Path(r"E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_delivery_check")
ANG_TO_BOHR = 1.0 / 0.529177210903


def get_peak_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().peak_wset / (1024.0 * 1024.0)


# Reasonable fixed organic geometries (Angstrom -> Bohr)
MOLS = [
    (
        "CH3CN_N",
        ["C", "C", "N", "H", "H", "H"],
        [
            [0.0000, 0.0000, -1.1780],
            [0.0000, 0.0000, 0.2810],
            [0.0000, 0.0000, 1.4380],
            [0.0000, 1.0280, -1.5450],
            [0.8903, -0.5140, -1.5450],
            [-0.8903, -0.5140, -1.5450],
        ],
        "6-31g(d)",
        "fine",
        1e-9,
        1e-8,
        True,
    ),
    (
        "C2H3F_F",
        ["C", "C", "F", "H", "H", "H"],
        [
            [0.0000, 0.6650, 0.0000],
            [0.0000, -0.6650, 0.0000],
            [1.1800, 1.3200, 0.0000],
            [-0.9300, 1.2200, 0.0000],
            [0.9400, -1.2000, 0.0000],
            [-0.9400, -1.2000, 0.0000],
        ],
        "6-31g(d)",
        "fine",
        1e-9,
        1e-8,
        True,
    ),
    (
        "C4H4S_S",
        ["S", "C", "C", "C", "C", "H", "H", "H", "H"],
        [
            [0.0000, 0.0000, 1.1950],
            [0.0000, 1.2400, 0.0150],
            [0.0000, 0.7150, -1.2550],
            [0.0000, -0.7150, -1.2550],
            [0.0000, -1.2400, 0.0150],
            [0.0000, 2.2800, 0.3000],
            [0.0000, 1.3200, -2.1500],
            [0.0000, -1.3200, -2.1500],
            [0.0000, -2.2800, 0.3000],
        ],
        "6-31g(d)",
        "fine",
        1e-9,
        1e-8,
        True,
    ),
]

rows = []
for name, sym, coords_ang, basis, grid, ce, cd, do_grad in MOLS:
    coords_bohr = [[x * ANG_TO_BOHR for x in row] for row in coords_ang]
    t0 = time.perf_counter()
    w_ex = c.run_single_point(
        sym, coords_bohr, 0, 1, "rks", basis,
        max_iter=120, conv_energy=ce, conv_density=cd,
        xc="b3lyp", grid_level=grid, density_fitting=False, use_gpu=False,
    )
    g_ex = np.asarray(
        c.run_gradient(sym, coords_bohr, 0, 1, basis, "rks", "b3lyp", grid, False, False, ce, cd)
    ) if do_grad else None
    t_ex = time.perf_counter() - t0

    t1 = time.perf_counter()
    w_ri = c.run_single_point(
        sym, coords_bohr, 0, 1, "rks", basis,
        max_iter=120, conv_energy=ce, conv_density=cd,
        xc="b3lyp", grid_level=grid, density_fitting=True, use_gpu=False,
    )
    g_ri = np.asarray(
        c.run_gradient(sym, coords_bohr, 0, 1, basis, "rks", "b3lyp", grid, True, False, ce, cd)
    ) if do_grad else None
    t_ri = time.perf_counter() - t1

    de = w_ri.total_energy_hartree - w_ex.total_energy_hartree
    max_dg = float(np.max(np.abs(g_ri - g_ex))) if do_grad else float("nan")
    rms_dg = float(np.sqrt(np.mean((g_ri - g_ex) ** 2))) if do_grad else float("nan")
    rss_mb = get_peak_rss_mb()
    print(
        f"[{name}] n_atoms={len(sym)} n_ao={w_ex.n_ao} "
        f"E_exact={w_ex.total_energy_hartree:.10f} E_rij={w_ri.total_energy_hartree:.10f} "
        f"dE={de:+.3e} max_dGrad={max_dg:.3e} t_ex={t_ex:.2f}s t_ri={t_ri:.2f}s peak_rss={rss_mb:.1f}MB"
    )
    rows.append({
        "system": name,
        "n_atoms": len(sym),
        "n_ao": w_ex.n_ao,
        "basis": basis,
        "grid": grid,
        "e_exact_eh": f"{w_ex.total_energy_hartree:.10f}",
        "e_rij_eh": f"{w_ri.total_energy_hartree:.10f}",
        "delta_e_eh": f"{de:+.3e}",
        "max_abs_dgrad_eh_bohr": f"{max_dg:.3e}",
        "rms_dgrad_eh_bohr": f"{rms_dg:.3e}",
        "t_exact_s": f"{t_ex:.2f}",
        "t_rij_s": f"{t_ri:.2f}",
        "peak_rss_mb": f"{rss_mb:.1f}",
    })

# IC-2F exact Coulomb baseline (STO-3G, matching existing IC-2F benchmark)
ic2f = get_real_opv_system("y6_ic_ff")
ic_spec = ic2f.to_spec()
ic_bohr = [[x * ANG_TO_BOHR for x in row] for row in ic_spec.coordinates_angstrom]
t0 = time.perf_counter()
w_ic_ex = c.run_single_point(
    ic2f.symbols, ic_bohr, 0, 1, "rks", "sto-3g",
    max_iter=120, conv_energy=1e-7, conv_density=1e-6,
    xc="b3lyp", grid_level="coarse", density_fitting=False, use_gpu=False,
)
t_ic_ex = time.perf_counter() - t0
t1 = time.perf_counter()
w_ic_ri = c.run_single_point(
    ic2f.symbols, ic_bohr, 0, 1, "rks", "sto-3g",
    max_iter=120, conv_energy=1e-7, conv_density=1e-6,
    xc="b3lyp", grid_level="coarse", density_fitting=True, use_gpu=False,
)
t_ic_ri = time.perf_counter() - t1
de_ic = w_ic_ri.total_energy_hartree - w_ic_ex.total_energy_hartree
rss_mb = get_peak_rss_mb()
print(
    f"[IC-2F_STO3G] n_atoms={len(ic2f.symbols)} n_ao={w_ic_ex.n_ao} "
    f"E_exact={w_ic_ex.total_energy_hartree:.10f} E_rij={w_ic_ri.total_energy_hartree:.10f} "
    f"dE={de_ic:+.3e} t_ex={t_ic_ex:.2f}s t_ri={t_ic_ri:.2f}s peak_rss={rss_mb:.1f}MB"
)
rows.append({
    "system": "IC-2F_STO3G",
    "n_atoms": len(ic2f.symbols),
    "n_ao": w_ic_ex.n_ao,
    "basis": "sto-3g",
    "grid": "coarse",
    "e_exact_eh": f"{w_ic_ex.total_energy_hartree:.10f}",
    "e_rij_eh": f"{w_ic_ri.total_energy_hartree:.10f}",
    "delta_e_eh": f"{de_ic:+.3e}",
    "max_abs_dgrad_eh_bohr": "N/A",
    "rms_dgrad_eh_bohr": "N/A",
    "t_exact_s": f"{t_ic_ex:.2f}",
    "t_rij_s": f"{t_ic_ri:.2f}",
    "peak_rss_mb": f"{rss_mb:.1f}",
})

csv_path = OUT_DIR / "TASK3_ORGANIC_RIJ_BENCHMARK.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Saved {csv_path}")
