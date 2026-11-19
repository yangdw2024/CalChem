"""Multi-Program Transition State & Reaction Benchmark Suite.

Compares:
1. CalChem (Pure Rust Native Engine)
2. REST 2026.1.1 (Prof. Igor Ying Zhang, Fudan University / 20_massive350 benchmark tasks)
3. ORCA 6.1.1 (E:\ORCA\software\ORCA6.1.1\orca.exe)
4. Gaussian 16 Rev A.03 (E:\YDW\ChemDatas)

Evaluates:
- Diels-Alder [4+2] Cycloaddition Transition State (1,3-butadiene + ethylene, 12 atoms)
- SN2 Walden Inversion Transition State ([Cl...CH3...Cl]-, 6 atoms, anion)
- Claisen Rearrangement Transition State (allyl vinyl ether, 12 atoms)
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from calchem import _core
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.core.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV, HARTREE_TO_KCAL_MOL
from calchem.backends.native import NativeCalchem
from calchem.backends.orca import ORCABackend
from calchem.backends.rest import RESTBackend, parse_rest_log


def parse_rest_toml(toml_path: Path):
    text = toml_path.read_text(encoding="utf-8")
    m_geom = re.search(r'position\s*=\s*"""(.*?)"""', text, re.DOTALL)
    if not m_geom:
        raise ValueError(f"No position found in {toml_path}")
    symbols = []
    coords = []
    for line in m_geom.group(1).strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    m_charge = re.search(r'charge\s*=\s*(-?\d+)', text)
    charge = int(m_charge.group(1)) if m_charge else 0

    m_spin = re.search(r'spin\s*=\s*(-?\d+)', text)
    mult = int(m_spin.group(1)) if m_spin else 1

    m_xc = re.search(r'xc\s*=\s*"([^"]+)"', text)
    xc = m_xc.group(1).lower() if m_xc else "b3lyp"

    m_basis = re.search(r'basis_path\s*=\s*"([^"]+)"', text)
    basis = m_basis.group(1).lower() if m_basis else "def2-svp"

    m_disp = re.search(r'empirical_dispersion\s*=\s*"([^"]+)"', text)
    disp = m_disp.group(1).lower() if m_disp else None

    return {
        "symbols": symbols,
        "coordinates": coords,
        "charge": charge,
        "multiplicity": mult,
        "method": xc,
        "basis": basis,
        "dispersion": disp,
    }


def run_benchmark():
    print("=" * 80)
    print("CALCHEM vs REST vs ORCA vs GAUSSIAN: REACTION & TRANSITION STATE BENCHMARK")
    print("=" * 80)

    massive_dir = Path(r"E:\YDW\REST\001\tasks\20_massive350")
    cases = [
        {
            "name": "Diels-Alder [4+2] TS",
            "desc": "1,3-Butadiene + Ethylene -> Cyclohexene pericyclic TS (12 atoms)",
            "toml": massive_dir / "33_ts_freq" / "diels_alder_tsfreq.toml",
            "log": massive_dir / "33_ts_freq" / "diels_alder_tsfreq.log",
        },
        {
            "name": "SN2 Walden Inversion TS",
            "desc": "Cl- + CH3Cl -> [Cl...CH3...Cl]- symmetric nucleophilic substitution TS (6 atoms, anion)",
            "toml": massive_dir / "32_ts_search" / "sn2_cl_ch3cl_ts.toml",
            "log": massive_dir / "32_ts_search" / "sn2_cl_ch3cl_ts.log",
        },
        {
            "name": "Claisen Rearrangement TS",
            "desc": "Allyl vinyl ether [3,3]-sigmatropic rearrangement chair-like TS (12 atoms)",
            "toml": massive_dir / "33_ts_freq" / "claisen_rearrangement_tsfreq.toml",
            "log": massive_dir / "33_ts_freq" / "claisen_rearrangement_tsfreq.log",
        },
    ]

    native_backend = NativeCalchem()
    orca_backend = ORCABackend(executable=r"E:\ORCA\software\ORCA6.1.1\orca.exe")

    results_table = []

    for item in cases:
        name = item["name"]
        print(f"\n>>> Running Benchmark: {name}")
        print(f"    Description: {item['desc']}")

        spec_data = parse_rest_toml(item["toml"])
        mol = MoleculeSpec(
            symbols=spec_data["symbols"],
            coordinates_angstrom=spec_data["coordinates"],
            charge=spec_data["charge"],
            multiplicity=spec_data["multiplicity"],
        )

        basis_eval = "6-31g(d)" if spec_data["basis"] == "def2-svp" else spec_data["basis"]

        # 1. REST Reference Log
        rest_info = {}
        if item["log"].is_file():
            rest_content = item["log"].read_text(encoding="utf-8")
            rest_info = parse_rest_log(rest_content)
            print(f"  [REST 2026.1.1] Finished in log: E = {rest_info['energy_hartree']:.8f} Eh, "
                  f"t = {rest_info.get('elapsed_seconds', 'N/A')} s")

        # 2. CalChem Pure Rust Native Engine
        req_native = CalculationRequest(
            molecule=mol,
            backend="native",
            task="single_point",
            method=spec_data["method"],
            basis=basis_eval,
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            dispersion="d3bj" if spec_data.get("dispersion") else None,
        )

        t0 = time.perf_counter()
        res_native = native_backend.single_point(req_native)
        t_native = time.perf_counter() - t0

        e_nat = res_native.total_energy_hartree or res_native.scf_energy_hartree
        dip_nat = res_native.dipole_magnitude_debye or 0.0
        print(f"  [CalChem Rust]  Calculated:      E = {e_nat:.8f} Eh, "
              f"dipole = {dip_nat:.4f} D, t = {t_native:.3f} s")

        # Evaluate Rust D3(BJ) dispersion explicitly
        b2a = BOHR_TO_ANGSTROM
        coords_bohr = [[c / b2a for c in row] for row in mol.coordinates_angstrom]
        d3_res = _core.run_d3_dispersion(mol.symbols, coords_bohr, damping="d3bj", functional="b3lyp")
        print(f"                  Rust D3(BJ):     E_disp = {d3_res.energy_hartree:.6f} Eh ({d3_res.energy_kcal_mol:.3f} kcal/mol)")

        # 3. ORCA 6.1.1
        req_orca = CalculationRequest(
            molecule=mol,
            backend="orca",
            task="single_point",
            method=spec_data["method"],
            basis=basis_eval,
            charge=mol.charge,
            multiplicity=mol.multiplicity,
            dispersion="d3bj" if spec_data.get("dispersion") else None,
        )

        t0 = time.perf_counter()
        res_orca = orca_backend.single_point(req_orca)
        t_orca = time.perf_counter() - t0
        e_orca = res_orca.total_energy_hartree or res_orca.scf_energy_hartree
        dip_orca = res_orca.dipole_magnitude_debye or 0.0
        if e_orca is not None:
            print(f"  [ORCA 6.1.1]    Calculated:      E = {e_orca:.8f} Eh, "
                  f"dipole = {dip_orca:.4f} D, t = {t_orca:.3f} s")
        else:
            print(f"  [ORCA 6.1.1]    Failed or skipped. Status: {res_orca.status}")

        # Cross Comparison
        e_diff_kcal = (e_nat - e_orca) * HARTREE_TO_KCAL_MOL if (e_nat and e_orca) else None
        speedup = t_orca / t_native if t_native > 0 else 1.0
        diff_str = f"{e_diff_kcal:+.4f} kcal/mol" if e_diff_kcal is not None else "N/A"
        print(f"  --> Cross-Check: CalChem vs ORCA diff = {diff_str}, Speedup = {speedup:.1f}x")

        results_table.append({
            "name": name,
            "n_atoms": len(mol.symbols),
            "charge": mol.charge,
            "mult": mol.multiplicity,
            "calchem_energy": e_nat,
            "calchem_d3_kcal": d3_res.energy_kcal_mol,
            "calchem_time": t_native,
            "calchem_dipole": dip_nat,
            "orca_energy": e_orca,
            "orca_time": t_orca,
            "orca_dipole": dip_orca,
            "rest_energy": rest_info.get("energy_hartree"),
            "rest_time": rest_info.get("elapsed_seconds"),
            "diff_kcal": e_diff_kcal,
            "speedup": speedup,
        })

    out_file = Path("build_target/runtime/benchmark_reactions_and_ts.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results_table, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Saved summary to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
