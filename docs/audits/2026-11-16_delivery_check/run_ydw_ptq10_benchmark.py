import os
import re
import subprocess
import time
from pathlib import Path

import psutil
import calchem._core as c

G16_EXE = Path(r"D:\G16W\g16.exe")
GJF_PATH = Path(r"E:\YDW\ChemDatas\ydw01\PTQ10\ydw-PTQ10.gjf")
WORK_DIR = Path(r"E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_delivery_check\ydw_ptq10_g16")
WORK_DIR.mkdir(parents=True, exist_ok=True)
ANG_TO_BOHR = 1.0 / 0.529177210903

syms = []
coords_ang = []
for line in GJF_PATH.read_text(errors="replace").splitlines():
    parts = line.split()
    if len(parts) == 4:
        try:
            xyz = [float(x) for x in parts[1:]]
            syms.append(parts[0])
            coords_ang.append(xyz)
        except ValueError:
            pass

print(f"Loaded ydw-PTQ10.gjf: n_atoms={len(syms)}")
coords_bohr = [[x * ANG_TO_BOHR for x in row] for row in coords_ang]

# 1. Run Gaussian 16 B3LYP/STO-3G
gjf = WORK_DIR / "input.gjf"
out = WORK_DIR / "input.out"
g16_lines = ["%nprocshared=4", "%mem=2GB", "#P B3LYP/STO-3G Int=Coarse", "", "ydw-PTQ10", "", "0 1"]
for s, xyz in zip(syms, coords_ang):
    g16_lines.append(f"{s:2s} {xyz[0]:15.10f} {xyz[1]:15.10f} {xyz[2]:15.10f}")
g16_lines.append("")
gjf.write_text("\n".join(g16_lines) + "\n", encoding="utf-8")

t_g0 = time.perf_counter()
proc = subprocess.run([str(G16_EXE), "input.gjf", "input.out"], cwd=str(WORK_DIR), capture_output=True, text=True)
t_g16 = time.perf_counter() - t_g0
txt = out.read_text(errors="replace") if out.exists() else ""
if proc.returncode != 0 or "Normal termination" not in txt:
    raise RuntimeError(f"G16 ydw-PTQ10 failed: {txt[-400:]}")
e_g16 = float(re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)", txt)[-1])
print(f"[G16 ydw-PTQ10 B3LYP/STO-3G] E={e_g16:.10f} Eh time={t_g16:.2f}s")

# 2. Run CalChem B3LYP/STO-3G Coarse RI-J + GPU
vram_before = c.get_cuda_memory_info()
t0 = time.perf_counter()
wf = c.run_single_point(
    syms,
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
    f"[CalChem ydw-PTQ10 B3LYP/STO-3G RI-J+GPU] n_ao={wf.n_ao} converged={wf.converged} "
    f"iters={wf.iterations} E={wf.total_energy_hartree:.10f} dE_vs_G16={wf.total_energy_hartree - e_g16:+.3e} "
    f"kernels={wf.gpu_kernel_invocations} fallbacks={wf.gpu_fallback_invocations} "
    f"time={dt:.2f}s peak_rss={rss_mb:.1f}MB vram_before={vram_before} vram_after={vram_after}"
)
