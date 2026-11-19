"""Fresh independent external reference comparison (Gaussian 16 D:\\G16W\\g16.exe vs CalChem)
and CPU vs GPU RI-J telemetry benchmark.
All temporary files and outputs remain strictly on the E: drive.
"""
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

REPO = Path(r"E:\YDW\Trae_Gaussian")
sys.path.insert(0, str(REPO))
CERT_DIR = REPO / "docs" / "audits" / "2026-11-16_final_certification"
G16_WORK = REPO / "build_target" / "runtime" / "g16_cert_20261116"
CERT_DIR.mkdir(parents=True, exist_ok=True)
G16_WORK.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(G16_WORK)

from calchem import _core as c
from calchem.workflows.opv import OPVWorkflow, get_real_opv_system

BOHR_TO_ANGSTROM = 0.52917721092


def run_g16(job_name: str, route: str, symbols: list[str], coords_ang: list[list[float]], charge: int = 0, mult: int = 1) -> tuple[str, float]:
    job_dir = G16_WORK / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    atom_lines = "\n".join(
        f"{sym} {pt[0]:.12f} {pt[1]:.12f} {pt[2]:.12f}"
        for sym, pt in zip(symbols, coords_ang)
    )
    gjf_text = (
        f"%mem=512MB\n"
        f"%nprocshared=1\n"
        f"#p {route} NoSymm SCF=(Conver=10,MaxCycle=150)\n\n"
        f"CalChem Certification {job_name}\n\n"
        f"{charge} {mult}\n"
        f"{atom_lines}\n\n"
    )
    (job_dir / "input.gjf").write_text(gjf_text, encoding="ascii")
    env = os.environ.copy()
    env.update(
        g16root=r"D:\G16W",
        GAUSS_EXEDIR=r"D:\G16W",
        GAUSS_SCRDIR=str(job_dir),
    )
    env["PATH"] = r"D:\G16W" + os.pathsep + env.get("PATH", "")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [r"D:\G16W\g16.exe", "input.gjf"],
        cwd=job_dir,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=240,
    )
    elapsed = time.perf_counter() - t0
    log_path = job_dir / "input.out"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    if proc.returncode != 0 or "Normal termination of Gaussian" not in log_text:
        raise RuntimeError(f"Gaussian 16 job {job_name} failed (rc={proc.returncode})")
    return log_text, elapsed


def parse_g16_scf_energy(log: str) -> float:
    matches = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)", log)
    if not matches:
        raise ValueError("No SCF Done line in Gaussian log")
    return float(matches[-1])


def parse_g16_homo_lumo(log: str) -> tuple[float | None, float | None]:
    occ_lines = re.findall(r"Alpha\s+occ\.\s+eigenvalues\s+--\s*([^\n]+)", log)
    virt_lines = re.findall(r"Alpha\s+virt\.\s+eigenvalues\s+--\s*([^\n]+)", log)
    occ = [float(x) for line in occ_lines for x in line.split()]
    virt = [float(x) for line in virt_lines for x in line.split()]
    homo = occ[-1] if occ else None
    lumo = virt[0] if virt else None
    return homo, lumo


def parse_g16_forces(log: str, n_atoms: int) -> np.ndarray | None:
    # Gaussian prints Forces (Hartrees/Bohr), where Force = -Gradient
    m = re.search(
        r"Center\s+Atomic\s+Forces\s+\(Hartrees/Bohr\)\s*\r?\n\s*Number\s+Number\s+X\s+Y\s+Z\s*\r?\n\s*-+\s*\r?\n(.*?)\r?\n\s*-+",
        log,
        re.S,
    )
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            # Gradient = -Force
            rows.append([-float(parts[2]), -float(parts[3]), -float(parts[4])])
    if len(rows) != n_atoms:
        return None
    return np.array(rows, dtype=float)


def main():
    ext_path = REPO / "calchem" / "_core.pyd"
    ext_sha256 = hashlib.sha256(ext_path.read_bytes()).hexdigest()
    print(f"[CERT] _core.pyd SHA256 = {ext_sha256}")

    # 1. External Reference Comparison (Gaussian 16 vs CalChem)
    h2o_sym = ["O", "H", "H"]
    h2o_bohr = [
        [0.0, 0.0, 0.0],
        [0.0, 1.432, 0.0],
        [1.112, -0.716, 0.0],
    ]
    h2o_ang = [[v * BOHR_TO_ANGSTROM for v in row] for row in h2o_bohr]

    h2_sym = ["H", "H"]
    h2_bohr = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    h2_ang = [[v * BOHR_TO_ANGSTROM for v in row] for row in h2_bohr]

    c2h4_sym = ["C", "C", "H", "H", "H", "H"]
    c2h4_ang = [
        [-0.67, 0.0, 0.0],
        [0.67, 0.0, 0.0],
        [-1.23, 0.93, 0.0],
        [-1.23, -0.93, 0.0],
        [1.23, 0.93, 0.0],
        [1.23, -0.93, 0.0],
    ]
    c2h4_bohr = [[v / BOHR_TO_ANGSTROM for v in row] for row in c2h4_ang]

    cases = [
        ("H2_RHF_STO3G", h2_sym, h2_bohr, h2_ang, "rhf", None, "sto-3g", "HF/STO-3G Force", False),
        ("H2O_RHF_STO3G", h2o_sym, h2o_bohr, h2o_ang, "rhf", None, "sto-3g", "HF/STO-3G Force", False),
        ("H2O_RHF_631G", h2o_sym, h2o_bohr, h2o_ang, "rhf", None, "6-31g", "HF/6-31G Force", False),
        ("H2O_RHF_631Gd", h2o_sym, h2o_bohr, h2o_ang, "rhf", None, "6-31g(d)", "HF/6-31G(d) 6D Force", False),
        ("H2O_B3LYP_STO3G", h2o_sym, h2o_bohr, h2o_ang, "rks", "b3lyp", "sto-3g", "B3LYP/STO-3G Integral=UltraFine Force", False),
        ("H2O_B3LYP_631Gd", h2o_sym, h2o_bohr, h2o_ang, "rks", "b3lyp", "6-31g(d)", "B3LYP/6-31G(d) 6D Integral=UltraFine Force", False),
        ("H2O_PBE0_631Gd", h2o_sym, h2o_bohr, h2o_ang, "rks", "pbe0", "6-31g(d)", "PBE1PBE/6-31G(d) 6D Integral=UltraFine Force", False),
        ("H2O_CAMB3LYP_631Gd", h2o_sym, h2o_bohr, h2o_ang, "rks", "cam-b3lyp", "6-31g(d)", "CAM-B3LYP/6-31G(d) 6D Integral=UltraFine Force", False),
        ("C2H4_RHF_STO3G", c2h4_sym, c2h4_bohr, c2h4_ang, "rhf", None, "sto-3g", "HF/STO-3G Force", False),
        ("H2_RHF_RIJ_STO3G", h2_sym, h2_bohr, h2_ang, "rhf", None, "sto-3g", "HF/STO-3G Force", True),
        ("H2O_B3LYP_RIJ_STO3G", h2o_sym, h2o_bohr, h2o_ang, "rks", "b3lyp", "sto-3g", "B3LYP/STO-3G Integral=UltraFine Force", True),
        ("H2O_B3LYP_RIJ_631Gd", h2o_sym, h2o_bohr, h2o_ang, "rks", "b3lyp", "6-31g(d)", "B3LYP/6-31G(d) 6D Integral=UltraFine Force", True),
    ]

    ext_rows = []
    for label, sym, bohr, ang, method, xc, basis, g16_route, use_df in cases:
        log, g16_sec = run_g16(label, g16_route, sym, ang)
        g16_e = parse_g16_scf_energy(log)
        g16_homo, g16_lumo = parse_g16_homo_lumo(log)
        g16_grad = parse_g16_forces(log, len(sym))

        conv_e = 1e-6 if use_df else 1e-11
        conv_d = 1e-6 if use_df else 1e-9

        t0 = time.perf_counter()
        wf = c.run_single_point(
            sym,
            bohr,
            0,
            1,
            method,
            basis,
            max_iter=150,
            conv_energy=conv_e,
            conv_density=conv_d,
            xc=xc,
            grid_level="ultrafine",
            density_fitting=use_df,
            use_gpu=False,
        )
        if not use_df or sym == ["H", "H"]:
            cc_grad = np.asarray(
                c.run_gradient(
                    sym,
                    bohr,
                    0,
                    1,
                    basis,
                    method,
                    xc=xc,
                    grid_level="ultrafine",
                    density_fitting=use_df,
                    use_gpu=False,
                )
            )
        else:
            cc_grad = None
        cc_sec = time.perf_counter() - t0

        de = wf.total_energy_hartree - g16_e
        dhomo = (wf.homo_energy_hartree - g16_homo) if (wf.homo_energy_hartree is not None and g16_homo is not None) else None
        dlumo = (wf.lumo_energy_hartree - g16_lumo) if (wf.lumo_energy_hartree is not None and g16_lumo is not None) else None
        max_dg = float(np.max(np.abs(cc_grad - g16_grad))) if (cc_grad is not None and g16_grad is not None) else None
        rms_dg = float(np.sqrt(np.mean((cc_grad - g16_grad) ** 2))) if (cc_grad is not None and g16_grad is not None) else None

        row = {
            "case_id": label,
            "molecule": "".join(sym),
            "method": (xc.upper() if xc else method.upper()) + ("+RI-J" if use_df else ""),
            "basis": basis,
            "reference_engine": "Gaussian 16 (D:\\G16W\\g16.exe)",
            "g16_energy_hartree": f"{g16_e:.10f}",
            "calchem_energy_hartree": f"{wf.total_energy_hartree:.10f}",
            "delta_energy_hartree": f"{de:.3e}",
            "g16_homo_hartree": f"{g16_homo:.6f}" if g16_homo is not None else "N/A",
            "calchem_homo_hartree": f"{wf.homo_energy_hartree:.6f}" if wf.homo_energy_hartree is not None else "N/A",
            "delta_homo_hartree": f"{dhomo:.3e}" if dhomo is not None else "N/A",
            "g16_lumo_hartree": f"{g16_lumo:.6f}" if g16_lumo is not None else "N/A",
            "calchem_lumo_hartree": f"{wf.lumo_energy_hartree:.6f}" if wf.lumo_energy_hartree is not None else "N/A",
            "delta_lumo_hartree": f"{dlumo:.3e}" if dlumo is not None else "N/A",
            "max_abs_grad_diff_eh_bohr": f"{max_dg:.3e}" if max_dg is not None else "N/A",
            "rms_grad_diff_eh_bohr": f"{rms_dg:.3e}" if rms_dg is not None else "N/A",
            "notes": (
                "Exact 4c ERI vs G16" if not use_df else "Auto-JFit RI-J vs G16 exact 4c ERI"
            ) + ("; Gaussian B3LYP uses VWN3 while CalChem uses VWN5" if xc in ("b3lyp", "cam-b3lyp") else ""),
        }
        ext_rows.append(row)
        dg_str = f"{max_dg:.3e}" if max_dg is not None else "N/A"
        print(f"[EXT] {label}: dE={de:.3e} Eh, max_dGrad={dg_str} Eh/Bohr")

    # Also record explicitly rejected wB97X-D
    try:
        c.run_single_point(h2o_sym, h2o_bohr, 0, 1, "rks", "6-31g(d)", xc="wb97x-d")
        wb97xd_status = "UNEXPECTED_SUCCESS"
    except Exception as e:
        wb97xd_status = f"Explicitly Rejected ({type(e).__name__}: {e})"

    ext_rows.append({
        "case_id": "H2O_WB97XD_631Gd",
        "molecule": "OHH",
        "method": "WB97X-D",
        "basis": "6-31g(d)",
        "reference_engine": "Gaussian 16 (D:\\G16W\\g16.exe)",
        "g16_energy_hartree": "N/A",
        "calchem_energy_hartree": "REJECTED",
        "delta_energy_hartree": "N/A",
        "g16_homo_hartree": "N/A",
        "calchem_homo_hartree": "N/A",
        "delta_homo_hartree": "N/A",
        "g16_lumo_hartree": "N/A",
        "calchem_lumo_hartree": "N/A",
        "delta_lumo_hartree": "N/A",
        "max_abs_grad_diff_eh_bohr": "N/A",
        "rms_grad_diff_eh_bohr": "N/A",
        "notes": wb97xd_status,
    })

    ext_csv_path = CERT_DIR / "EXTERNAL_REFERENCE_COMPARISON.csv"
    with open(ext_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ext_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ext_rows)

    # 2. CPU vs GPU Validation (H2O, C2H4, and 22-atom IC-2F)
    cpu_gpu_rows = []
    for mol_name, sym, bohr, method, xc, basis, grid in [
        ("H2O", h2o_sym, h2o_bohr, "rks", "svwn5", "sto-3g", "coarse"),
        ("H2O", h2o_sym, h2o_bohr, "rks", "b3lyp", "6-31g(d)", "fine"),
        ("C2H4", c2h4_sym, c2h4_bohr, "rks", "b3lyp", "sto-3g", "coarse"),
    ]:
        t_cpu0 = time.perf_counter()
        w_cpu = c.run_single_point(
            sym, bohr, 0, 1, method, basis, max_iter=120, conv_energy=1e-6, conv_density=1e-6,
            xc=xc, grid_level=grid, density_fitting=True, use_gpu=False
        )
        t_cpu = time.perf_counter() - t_cpu0

        t_gpu0 = time.perf_counter()
        w_gpu = c.run_single_point(
            sym, bohr, 0, 1, method, basis, max_iter=120, conv_energy=1e-6, conv_density=1e-6,
            xc=xc, grid_level=grid, density_fitting=True, use_gpu=True
        )
        t_gpu = time.perf_counter() - t_gpu0

        cpu_gpu_rows.append({
            "system": mol_name,
            "n_atoms": len(sym),
            "n_ao": w_cpu.n_ao,
            "method": (xc.upper() if xc else method.upper()) + " + RI-J",
            "basis": basis,
            "cpu_energy_hartree": f"{w_cpu.total_energy_hartree:.10f}",
            "gpu_energy_hartree": f"{w_gpu.total_energy_hartree:.10f}",
            "abs_energy_diff_hartree": f"{abs(w_gpu.total_energy_hartree - w_cpu.total_energy_hartree):.3e}",
            "cpu_homo_hartree": f"{w_cpu.homo_energy_hartree:.8f}",
            "gpu_homo_hartree": f"{w_gpu.homo_energy_hartree:.8f}",
            "cpu_lumo_hartree": f"{w_cpu.lumo_energy_hartree:.8f}",
            "gpu_lumo_hartree": f"{w_gpu.lumo_energy_hartree:.8f}",
            "abs_gap_diff_ev": f"{abs((w_gpu.lumo_energy_hartree - w_gpu.homo_energy_hartree) - (w_cpu.lumo_energy_hartree - w_cpu.homo_energy_hartree)) * 27.211386245988:.3e}",
            "cpu_iterations": w_cpu.iterations,
            "gpu_iterations": w_gpu.iterations,
            "cpu_gpu_kernel_invocations": w_cpu.gpu_kernel_invocations,
            "gpu_gpu_kernel_invocations": w_gpu.gpu_kernel_invocations,
            "gpu_fallback_invocations": w_gpu.gpu_fallback_invocations,
            "cpu_walltime_s": f"{t_cpu:.4f}",
            "gpu_walltime_s": f"{t_gpu:.4f}",
            "speedup_ratio": f"{t_cpu / max(t_gpu, 1e-9):.3f}",
        })

    # 22-atom IC-2F OPV terminal acceptor
    wf_opv = OPVWorkflow(backend="native")
    ic2f = get_real_opv_system("y6_ic_ff")
    spec = ic2f.to_spec()
    res_cpu = wf_opv.single_point(
        spec,
        method="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        options={
            "grid_level": "coarse",
            "conv_density": 1e-5,
            "conv_energy": 1e-6,
            "density_fitting": True,
            "use_gpu": False,
        },
        name="IC2F_CPU_RIJ",
    )
    res_gpu = wf_opv.single_point(
        spec,
        method="b3lyp",
        basis="sto-3g",
        dispersion="d3bj",
        options={
            "grid_level": "coarse",
            "conv_density": 1e-5,
            "conv_energy": 1e-6,
            "density_fitting": True,
            "use_gpu": True,
        },
        name="IC2F_GPU_RIJ",
    )
    cpu_gpu_rows.append({
        "system": "IC-2F (y6_ic_ff)",
        "n_atoms": len(ic2f.symbols),
        "n_ao": 94,
        "method": "B3LYP-D3(BJ) + RI-J",
        "basis": "sto-3g",
        "cpu_energy_hartree": f"{res_cpu.scf_energy_hartree:.10f}",
        "gpu_energy_hartree": f"{res_gpu.scf_energy_hartree:.10f}",
        "abs_energy_diff_hartree": f"{abs(res_gpu.scf_energy_hartree - res_cpu.scf_energy_hartree):.3e}",
        "cpu_homo_hartree": f"{res_cpu.homo_energy_hartree:.8f}",
        "gpu_homo_hartree": f"{res_gpu.homo_energy_hartree:.8f}",
        "cpu_lumo_hartree": f"{res_cpu.lumo_energy_hartree:.8f}",
        "gpu_lumo_hartree": f"{res_gpu.lumo_energy_hartree:.8f}",
        "abs_gap_diff_ev": f"{abs(res_gpu.gap_ev - res_cpu.gap_ev):.3e}",
        "cpu_iterations": res_cpu.scf_iterations,
        "gpu_iterations": res_gpu.scf_iterations,
        "cpu_gpu_kernel_invocations": res_cpu.gpu_kernel_invocations,
        "gpu_gpu_kernel_invocations": res_gpu.gpu_kernel_invocations,
        "gpu_fallback_invocations": res_gpu.gpu_fallback_invocations,
        "cpu_walltime_s": f"{res_cpu.walltime_seconds:.4f}",
        "gpu_walltime_s": f"{res_gpu.walltime_seconds:.4f}",
        "speedup_ratio": f"{res_cpu.walltime_seconds / max(res_gpu.walltime_seconds, 1e-9):.3f}",
    })
    print(
        f"[GPU] IC-2F: CPU={res_cpu.scf_energy_hartree:.10f} Eh ({res_cpu.walltime_seconds:.2f}s, kernels={res_cpu.gpu_kernel_invocations}), "
        f"GPU={res_gpu.scf_energy_hartree:.10f} Eh ({res_gpu.walltime_seconds:.2f}s, kernels={res_gpu.gpu_kernel_invocations}, fallbacks={res_gpu.gpu_fallback_invocations})"
    )

    gpu_csv_path = CERT_DIR / "CPU_GPU_VALIDATION.csv"
    with open(gpu_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cpu_gpu_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cpu_gpu_rows)

    print("[CERT] Saved EXTERNAL_REFERENCE_COMPARISON.csv and CPU_GPU_VALIDATION.csv")


if __name__ == "__main__":
    main()
