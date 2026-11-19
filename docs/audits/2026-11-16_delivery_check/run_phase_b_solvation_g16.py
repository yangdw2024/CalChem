import re
import subprocess
from pathlib import Path

import calchem._core as c

G16_EXE = Path(r"D:\G16W\g16.exe")
WORK_DIR = Path(r"E:\YDW\Trae_Gaussian\docs\audits\2026-11-16_delivery_check\smd_ref\g16_runs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
BOHR_TO_ANG = 0.529177210903


def run_g16(name: str, route: str, atoms_bohr: list[tuple[str, list[float]]]) -> float:
    job_dir = WORK_DIR / name
    job_dir.mkdir(parents=True, exist_ok=True)
    gjf = job_dir / "input.gjf"
    out = job_dir / "input.out"
    lines = [
        "%nprocshared=4",
        "%mem=1GB",
        route,
        "",
        name,
        "",
        "0 1",
    ]
    for sym, xyz in atoms_bohr:
        x, y, z = [v * BOHR_TO_ANG for v in xyz]
        lines.append(f"{sym:2s} {x:15.10f} {y:15.10f} {z:15.10f}")
    lines.append("")
    gjf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    proc = subprocess.run([str(G16_EXE), "input.gjf", "input.out"], cwd=str(job_dir), capture_output=True, text=True)
    txt = out.read_text(errors="replace") if out.exists() else ""
    if proc.returncode != 0 or "Normal termination" not in txt:
        raise RuntimeError(f"G16 {name} failed: {txt[-400:]}")
    m = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)", txt)
    return float(m[-1])


h2o = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 1.432, 0.0]), ("H", [1.112, -0.716, 0.0])]
h2o_sym = [a[0] for a in h2o]
h2o_bohr = [a[1] for a in h2o]

ch3cn_ang = [
    ("C", [0.0000, 0.0000, -1.1780]),
    ("C", [0.0000, 0.0000, 0.2810]),
    ("N", [0.0000, 0.0000, 1.4380]),
    ("H", [0.0000, 1.0280, -1.5450]),
    ("H", [0.8903, -0.5140, -1.5450]),
    ("H", [-0.8903, -0.5140, -1.5450]),
]
ch3cn = [(s, [v / BOHR_TO_ANG for v in xyz]) for s, xyz in ch3cn_ang]
ch3cn_sym = [a[0] for a in ch3cn]
ch3cn_bohr = [a[1] for a in ch3cn]

# Gas baselines
g16_h2o_gas = run_g16("h2o_gas", "#P B3LYP/6-31G(d) Int=UltraFine SCF=Tight", h2o)
g16_h2o_ief = run_g16("h2o_ief_water", "#P B3LYP/6-31G(d) Int=UltraFine SCRF=(IEFPCM,Solvent=Water) SCF=Tight", h2o)
g16_h2o_smd = run_g16("h2o_smd_water", "#P B3LYP/6-31G(d) Int=UltraFine SCRF=(SMD,Solvent=Water) SCF=Tight", h2o)

g16_mecn_gas = run_g16("mecn_gas", "#P B3LYP/6-31G(d) Int=UltraFine SCF=Tight", ch3cn)
g16_mecn_ief_chl = run_g16("mecn_ief_chl", "#P B3LYP/6-31G(d) Int=UltraFine SCRF=(IEFPCM,Solvent=Chloroform) SCF=Tight", ch3cn)
g16_mecn_smd_chl = run_g16("mecn_smd_chl", "#P B3LYP/6-31G(d) Int=UltraFine SCRF=(SMD,Solvent=Chloroform) SCF=Tight", ch3cn)

KCAL = 627.509474063
print(f"G16 H2O  IEFPCM(Water)   dG={(g16_h2o_ief - g16_h2o_gas)*KCAL:+.3f} kcal/mol")
print(f"G16 H2O  SMD(Water)      dG={(g16_h2o_smd - g16_h2o_gas)*KCAL:+.3f} kcal/mol")
print(f"G16 MeCN IEFPCM(CHCl3)   dG={(g16_mecn_ief_chl - g16_mecn_gas)*KCAL:+.3f} kcal/mol")
print(f"G16 MeCN SMD(CHCl3)      dG={(g16_mecn_smd_chl - g16_mecn_gas)*KCAL:+.3f} kcal/mol")

cc_h2o_ief = c.run_solvation(h2o_sym, h2o_bohr, basis_name="6-31g(d)", solvent="water", solvent_model="iefpcm", method="rks", xc="b3lyp", scaling_factor=1.20, include_cds=False)
cc_h2o_smd = c.run_solvation(h2o_sym, h2o_bohr, basis_name="6-31g(d)", solvent="water", solvent_model="smd", method="rks", xc="b3lyp", scaling_factor=1.0, include_cds=True)
cc_h2o_ons = c.run_solvation(h2o_sym, h2o_bohr, basis_name="6-31g(d)", solvent="water", solvent_model="onsager", method="rks", xc="b3lyp", scaling_factor=1.20, include_cds=False)

cc_mecn_ief = c.run_solvation(ch3cn_sym, ch3cn_bohr, basis_name="6-31g(d)", solvent="chloroform", solvent_model="iefpcm", method="rks", xc="b3lyp", scaling_factor=1.20, include_cds=False)
cc_mecn_smd = c.run_solvation(ch3cn_sym, ch3cn_bohr, basis_name="6-31g(d)", solvent="chloroform", solvent_model="smd", method="rks", xc="b3lyp", scaling_factor=1.0, include_cds=True)
cc_mecn_rij_smd = c.run_solvation(ch3cn_sym, ch3cn_bohr, basis_name="6-31g(d)", solvent="chloroform", solvent_model="smd", method="rks", xc="b3lyp", scaling_factor=1.0, include_cds=True, density_fitting=True, use_gpu=False)

print(f"CalChem H2O  IEFPCM(Water)  dG_elec={cc_h2o_ief.delta_g_elec_kcal_mol:+.3f} kcal/mol")
print(f"CalChem H2O  SMD(Water)     dG_solv={cc_h2o_smd.delta_g_solv_kcal_mol:+.3f} (elec={cc_h2o_smd.delta_g_elec_kcal_mol:+.3f}, cds={cc_h2o_smd.delta_g_cds_kcal_mol:+.3f}) kcal/mol")
print(f"CalChem H2O  Onsager(Water) dG_elec={cc_h2o_ons.delta_g_elec_kcal_mol:+.3f} kcal/mol")
print(f"CalChem MeCN IEFPCM(CHCl3)  dG_elec={cc_mecn_ief.delta_g_elec_kcal_mol:+.3f} kcal/mol")
print(f"CalChem MeCN SMD(CHCl3)     dG_solv={cc_mecn_smd.delta_g_solv_kcal_mol:+.3f} (elec={cc_mecn_smd.delta_g_elec_kcal_mol:+.3f}, cds={cc_mecn_smd.delta_g_cds_kcal_mol:+.3f}) kcal/mol")
print(f"CalChem MeCN SMD(CHCl3) RIJ dG_solv={cc_mecn_rij_smd.delta_g_solv_kcal_mol:+.3f} kcal/mol")
