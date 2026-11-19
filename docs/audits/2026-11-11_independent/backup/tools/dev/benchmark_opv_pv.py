#!/usr/bin/env python3
"""CalChem Industrial Benchmark Suite: Authentic OPV Photovoltaic Systems (PM6, Y6, BTP, IC-2F, MD Clusters).

All molecular structures are loaded directly from authoritative Gaussian 16 (.log / .gjf / .fchk)
research data in `E:\\YDW\\ChemDatas`:
1. `ydw-IC_FF.log` / `.fchk`: Optimized Y6 IC-2F acceptor terminal unit (22 atoms, C13H4F2N2O)
2. `ydw-BTP.log`: Optimized Y6 central BTP core unit (159 atoms, C60H90N4S5)
3. `ydw-BTP-4F.log`: Optimized full Y6 (BTP-4F) molecule (199 atoms, C86H94F4N8O2S5)
4. `ydw-PM6-opt01.gjf`: Full PM6 donor polymer repeat monomer (158 atoms, C68H78F2O2S8)
5. `00cluster_PM6-L8BO-2ac_rank1_donor-BDD1_Acceptor2-NC1.log` / `.gjf`:
   MD-extracted 60-atom interfacial donor-acceptor stacking cluster (32-atom donor-BDD1 + 28-atom Acceptor2-NC1)
"""

from collections import Counter
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from calchem import _core
from calchem.workflows.opv import (
    CHEMDATAS_REFERENCE_DATA,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    OPVMolecule,
    OPVWorkflow,
    get_pm6_fragment,
    get_real_opv_system,
    get_y6_fragment,
)


def _formula_str(symbols: list[str]) -> str:
    counts = Counter(symbols)
    order = ["C", "H", "F", "N", "O", "S"] + sorted(k for k in counts if k not in ("C", "H", "F", "N", "O", "S"))
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order if counts.get(el))


def run_benchmark():
    t_start = time.time()
    print("=" * 88)
    print("   CalChem Stage 4: Authentic OPV Photovoltaics Benchmark Suite (ChemDatas Ground Truth)")
    print("   Pure Rust Engine Acceleration: Sparse ERI / DFT / LRC / D3(BJ) / PCM / TDDFT")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)

    wf = OPVWorkflow(backend="native")

    # ---------------------------------------------------------
    # 1. Load & Verify Authentic ChemDatas Molecular Inventory
    # ---------------------------------------------------------
    print("\n[1/5] Authentic Gaussian 16 ChemDatas Molecular Inventory:")
    ic2f = get_real_opv_system("y6_ic_ff")
    btp_core = get_real_opv_system("y6_btp_core")
    y6_full = get_real_opv_system("y6_full")
    pm6_monomer = get_real_opv_system("pm6_monomer")
    md_cluster = get_real_opv_system("pm6_l8bo_cluster")
    donor_bdd1 = get_real_opv_system("donor_bdd1")
    acceptor_nc1 = get_real_opv_system("acceptor_nc1")

    for label, mol in [
        ("Y6 Terminal (IC-2F)", ic2f),
        ("Y6 Central Core (BTP)", btp_core),
        ("Y6 Full Molecule (BTP-4F)", y6_full),
        ("PM6 Full Repeat Monomer", pm6_monomer),
        ("PM6:L8-BO MD D-A Cluster", md_cluster),
        ("PM6 Cluster Donor (BDD1)", donor_bdd1),
        ("L8-BO Cluster Acceptor (NC1)", acceptor_nc1),
    ]:
        ref_e = mol.reference_data.get("scf_energy_hartree") if mol.reference_data else None
        ref_str = f" | G16 Ref E = {ref_e:14.6f} Eh" if ref_e is not None else ""
        print(f"  - {label:<28}: {len(mol.symbols):3d} atoms ({_formula_str(mol.symbols):<16}){ref_str}")

    # ---------------------------------------------------------
    # 2. Ground State DFT on Authentic IC-2F (22 atoms) & PM6 Donor-BDD1 (32 atoms)
    # ---------------------------------------------------------
    print("\n[2/5] Ground State DFT Benchmarks on Authentic Sub-Units (STO-3G Basis):")
    methods = [
        ("b3lyp", "B3LYP (Hybrid GGA)"),
        ("camb3lyp", "CAM-B3LYP (Range-Separated)"),
        ("wb97xd", "wB97X-D (Dispersion LRC)"),
    ]
    dft_results = {}

    for m_code, m_desc in methods:
        t0 = time.time()
        res_d = wf.single_point(donor_bdd1.to_spec(), method=m_code, basis="sto-3g", name=f"Donor_BDD1_{m_code}")
        res_a = wf.single_point(ic2f.to_spec(), method=m_code, basis="sto-3g", name=f"IC2F_{m_code}")
        dt = time.time() - t0

        dft_results[m_code] = {
            "desc": m_desc,
            "time_s": dt,
            "donor_bdd1_energy_eh": res_d.scf_energy_hartree,
            "donor_bdd1_homo_ev": res_d.homo_ev,
            "donor_bdd1_lumo_ev": res_d.lumo_ev,
            "donor_bdd1_gap_ev": res_d.gap_ev,
            "ic2f_energy_eh": res_a.scf_energy_hartree,
            "ic2f_homo_ev": res_a.homo_ev,
            "ic2f_lumo_ev": res_a.lumo_ev,
            "ic2f_gap_ev": res_a.gap_ev,
            "ic2f_g16_wb97xd_631gdp_ref_eh": CHEMDATAS_REFERENCE_DATA["y6_ic_ff"]["scf_energy_hartree"],
        }

        print(f"  * {m_desc:<30} (took {dt:5.2f}s):")
        print(
            f"      PM6 Donor-BDD1 (32 atoms) : E = {res_d.scf_energy_hartree:13.6f} Eh | "
            f"HOMO = {res_d.homo_ev:6.2f} eV | LUMO = {res_d.lumo_ev:6.2f} eV | Gap = {res_d.gap_ev:5.2f} eV"
        )
        print(
            f"      Y6 IC-2F       (22 atoms) : E = {res_a.scf_energy_hartree:13.6f} Eh | "
            f"HOMO = {res_a.homo_ev:6.2f} eV | LUMO = {res_a.lumo_ev:6.2f} eV | Gap = {res_a.gap_ev:5.2f} eV"
        )

    # ---------------------------------------------------------
    # 3. Authentic 60-Atom MD Interfacial Stacking Cluster (Donor-BDD1 + Acceptor2-NC1)
    # ---------------------------------------------------------
    print("\n[3/5] Authentic 60-Atom MD Interfacial Cluster Binding & D3(BJ) Dispersion:")
    t0 = time.time()
    sp_cluster = wf.single_point(
        md_cluster.to_spec(), method="camb3lyp", basis="sto-3g", dispersion="d3bj", name="MD_Cluster_60"
    )
    sp_donor = wf.single_point(
        donor_bdd1.to_spec(), method="camb3lyp", basis="sto-3g", dispersion="d3bj", name="Donor_BDD1_32"
    )
    sp_acc = wf.single_point(
        acceptor_nc1.to_spec(), method="camb3lyp", basis="sto-3g", dispersion="d3bj", name="Acceptor_NC1_28"
    )
    dt_cluster = time.time() - t0

    e_bind_tot_eh = sp_cluster.total_energy_hartree - sp_donor.total_energy_hartree - sp_acc.total_energy_hartree
    e_bind_tot_kcal = e_bind_tot_eh * HARTREE_TO_KCAL_MOL
    e_disp_inter_eh = (
        sp_cluster.dispersion_energy_hartree
        - sp_donor.dispersion_energy_hartree
        - sp_acc.dispersion_energy_hartree
    )
    e_disp_inter_kcal = e_disp_inter_eh * HARTREE_TO_KCAL_MOL

    print(f"  * 60-Atom MD Cluster (32-atom BDD1 + 28-atom NC1) in {dt_cluster:.2f}s:")
    print(f"      E_tot(Donor-BDD1, 32 atoms)    = {sp_donor.total_energy_hartree:16.8f} Eh")
    print(f"      E_tot(Acceptor2-NC1, 28 atoms) = {sp_acc.total_energy_hartree:16.8f} Eh")
    print(f"      E_tot(MD Cluster, 60 atoms)    = {sp_cluster.total_energy_hartree:16.8f} Eh (G16 6-31G(d) Ref: -3469.55737880 Eh)")
    print(f"      Delta E_bind (CAM-B3LYP-D3BJ)  = {e_bind_tot_eh:16.8f} Eh = {e_bind_tot_kcal:8.2f} kcal/mol")
    print(f"      Delta E_disp (D3(BJ) Intermol) = {e_disp_inter_eh:16.8f} Eh = {e_disp_inter_kcal:8.2f} kcal/mol")

    # ---------------------------------------------------------
    # 4. PCM Solvation Free Energies in Industry Solvents
    # ---------------------------------------------------------
    print("\n[4/5] Polarizable Continuum Solvation (C-PCM / SCRF) on Authentic Sub-Units:")
    solvents = [
        ("chlorobenzene", 5.62, "Processing Solvent"),
        ("chloroform", 4.71, "High-Performance Casting Solvent (ChemDatas TH Reference)"),
    ]
    solv_results = {}
    for solv_name, eps, desc in solvents:
        t0 = time.time()
        s_d = wf.solvation_free_energy(
            donor_bdd1.to_spec(), solvent=solv_name, method="b3lyp", basis="sto-3g", name=f"BDD1_{solv_name}"
        )
        s_a = wf.solvation_free_energy(
            ic2f.to_spec(), solvent=solv_name, method="b3lyp", basis="sto-3g", name=f"IC2F_{solv_name}"
        )
        dt = time.time() - t0

        solv_results[solv_name] = {
            "desc": desc,
            "epsilon": eps,
            "time_s": dt,
            "donor_bdd1_delta_g_solv_kcal": s_d.delta_g_solv_kcal_mol,
            "donor_bdd1_sasa_ang2": s_d.cavity_surface_area_angstrom2,
            "ic2f_delta_g_solv_kcal": s_a.delta_g_solv_kcal_mol,
            "ic2f_sasa_ang2": s_a.cavity_surface_area_angstrom2,
        }
        print(f"  * {solv_name.capitalize():<14} (eps={eps:4.2f}, {desc}) in {dt:.2f}s:")
        print(
            f"      PM6 Donor-BDD1 (32 atoms) : Delta_G_solv = {s_d.delta_g_solv_kcal_mol:8.2f} kcal/mol | "
            f"SASA = {s_d.cavity_surface_area_angstrom2:6.1f} A^2"
        )
        print(
            f"      Y6 IC-2F       (22 atoms) : Delta_G_solv = {s_a.delta_g_solv_kcal_mol:8.2f} kcal/mol | "
            f"SASA = {s_a.cavity_surface_area_angstrom2:6.1f} A^2"
        )

    # ---------------------------------------------------------
    # 5. TDDFT Linear Response on Authentic 22-Atom IC-2F Acceptor Terminal
    # ---------------------------------------------------------
    print("\n[5/5] TDDFT Linear Response Excited States on Authentic 22-Atom IC-2F:")
    t0 = time.time()
    abs_ic2f = wf.optical_absorption(
        ic2f.to_spec(),
        method="camb3lyp",
        basis="sto-3g",
        n_states=3,
        tda=True,
        compute_exciton_binding=False,
        name="IC2F_CAM-B3LYP_TDDFT",
    )
    dt_tddft = time.time() - t0
    print(f"  * Y6 IC-2F Optical Absorption (CAM-B3LYP/TDA, 22 atoms) in {dt_tddft:.2f}s:")
    print(
        f"      Optical Gap E_g^opt (S1) = {abs_ic2f.optical_gap_ev:5.2f} eV "
        f"({abs_ic2f.wavelength_max_nm:5.1f} nm, f = {abs_ic2f.oscillator_strength_max:.4f})"
    )
    print(f"      Kohn-Sham Gap            = {abs_ic2f.ks_gap_ev:5.2f} eV")
    if abs_ic2f.states:
        print("      Excited States:")
        for s in abs_ic2f.states:
            print(
                f"        State {s['label']}: {s['energy_ev']:5.2f} eV "
                f"({s['wavelength_nm']:5.1f} nm), f = {s['oscillator_strength']:.4f}"
            )

    total_time = time.time() - t_start
    print("\n" + "=" * 88)
    print(f"   Authentic ChemDatas Benchmark Completed Successfully in {total_time:.2f} seconds!")
    print("=" * 88)

    summary_data = {
        "benchmark": "OPV_Authentic_ChemDatas_Suite",
        "timestamp": datetime.now().isoformat(),
        "walltime_seconds": total_time,
        "chemdatas_inventory": {
            "y6_ic_ff": {"n_atoms": len(ic2f.symbols), "formula": _formula_str(ic2f.symbols), **CHEMDATAS_REFERENCE_DATA["y6_ic_ff"]},
            "y6_btp_core": {"n_atoms": len(btp_core.symbols), "formula": _formula_str(btp_core.symbols), **CHEMDATAS_REFERENCE_DATA["y6_btp_core"]},
            "y6_full": {"n_atoms": len(y6_full.symbols), "formula": _formula_str(y6_full.symbols), **CHEMDATAS_REFERENCE_DATA["y6_full"]},
            "pm6_monomer": {"n_atoms": len(pm6_monomer.symbols), "formula": _formula_str(pm6_monomer.symbols), **CHEMDATAS_REFERENCE_DATA["pm6_monomer"]},
            "pm6_l8bo_cluster": {"n_atoms": len(md_cluster.symbols), "formula": _formula_str(md_cluster.symbols), **CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]},
        },
        "dft_comparisons": dft_results,
        "md_cluster_60_atoms": {
            "e_donor_bdd1_eh": sp_donor.total_energy_hartree,
            "e_acceptor_nc1_eh": sp_acc.total_energy_hartree,
            "e_cluster_eh": sp_cluster.total_energy_hartree,
            "g16_ref_b3lyp_631gd_eh": CHEMDATAS_REFERENCE_DATA["pm6_l8bo_cluster"]["scf_energy_hartree"],
            "e_binding_kcal_mol": e_bind_tot_kcal,
            "e_dispersion_kcal_mol": e_disp_inter_kcal,
        },
        "solvation": solv_results,
        "ic2f_optical": {
            "optical_gap_ev": abs_ic2f.optical_gap_ev,
            "wavelength_max_nm": abs_ic2f.wavelength_max_nm,
            "oscillator_strength_max": abs_ic2f.oscillator_strength_max,
            "ks_gap_ev": abs_ic2f.ks_gap_ev,
        },
    }

    out_json = Path("build_target/runtime/benchmark_opv_pv.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Benchmark summary written to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
