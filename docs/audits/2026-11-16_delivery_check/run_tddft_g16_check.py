import re
import subprocess
from pathlib import Path

import calchem._core as c

G16_EXE = Path(r"D:\G16W\g16.exe")
WORK_DIR = Path(r"E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_delivery_check\tddft_g16")
WORK_DIR.mkdir(parents=True, exist_ok=True)
BOHR_TO_ANG = 0.529177210903


def run_g16_tddft(name: str, sym: list[str], coords_bohr: list[list[float]]) -> list[tuple[float, float]]:
    job_dir = WORK_DIR / name
    job_dir.mkdir(parents=True, exist_ok=True)
    gjf = job_dir / "input.gjf"
    out = job_dir / "input.out"
    lines = ["%nprocshared=4", "%mem=1GB", "#P B3LYP/6-31G(d) TD=(NStates=3) Int=UltraFine", "", name, "", "0 1"]
    for s, xyz in zip(sym, coords_bohr):
        lines.append(f"{s:2s} {xyz[0]*BOHR_TO_ANG:15.10f} {xyz[1]*BOHR_TO_ANG:15.10f} {xyz[2]*BOHR_TO_ANG:15.10f}")
    lines.append("")
    gjf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc = subprocess.run([str(G16_EXE), "input.gjf", "input.out"], cwd=str(job_dir), capture_output=True, text=True)
    txt = out.read_text(errors="replace") if out.exists() else ""
    if proc.returncode != 0 or "Normal termination" not in txt:
        raise RuntimeError(f"G16 {name} failed")
    states = re.findall(r"Excited State\s+\d+:\s+Singlet-\S+\s+([\d.]+)\s+eV\s+[\d.]+\s+nm\s+f=([\d.]+)", txt)
    return [(float(e), float(f)) for e, f in states]


h2o_sym = ["O", "H", "H"]
h2o_bohr = [[0.0, 0.0, 0.0], [0.0, 1.432, 0.0], [1.112, -0.716, 0.0]]

c2h4_sym = ["C", "C", "H", "H", "H", "H"]
c2h4_ang = [
    [-0.67, 0.0, 0.0], [0.67, 0.0, 0.0],
    [-1.23, 0.93, 0.0], [-1.23, -0.93, 0.0],
    [1.23, 0.93, 0.0], [1.23, -0.93, 0.0],
]
c2h4_bohr = [[v / BOHR_TO_ANG for v in row] for row in c2h4_ang]

for name, sym, bohr in [("H2O", h2o_sym, h2o_bohr), ("C2H4", c2h4_sym, c2h4_bohr)]:
    g16_states = run_g16_tddft(name, sym, bohr)
    cc_td = c.run_tddft(sym, bohr, basis="6-31g(d)", method="rks", xc="b3lyp", grid_level="ultrafine", n_states=3, tda=False)
    print(f"=== {name} B3LYP/6-31G(d) Full-TDDFT (RPA) ===")
    for idx, (g, s) in enumerate(zip(g16_states, cc_td.states), 1):
        print(f"  S{idx}: G16={g[0]:.4f} eV (f={g[1]:.4f}) | CalChem={s.excitation_energy_ev:.4f} eV (f={s.oscillator_strength:.4f}) | dE={s.excitation_energy_ev - g[0]:+.4f} eV")
