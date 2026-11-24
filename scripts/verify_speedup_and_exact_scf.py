"""Verify the two root-cause fixes on Caffeine (8 threads CPU):
1. Direct exact-4c SCF convergence on Fine grid (was failing at 100 iters / 548s due to 1e-10 screening noise)
2. Analytical gradient speedup on Fine + def2-jfit-f (was 648s before density screening + single-pass VRR)
3. Caffeine 260-AO Direct full-LR TD-DFT vs Gaussian 16 reference (was impossible due to 9.9 GB stored-ERI limit)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

G16_REF_E = -680.431066858
KCAL = 627.509474063
EXE = Path(r"E:\YDW\Trae_Gaussian\build_target\release\calchem_cli.exe")
GEOM = Path(r"D:\YDW\Calc\qc_benchmark_runs\pilot\caffeine.xyz")
VERIFIER = Path(r"E:\YDW\Trae_Gaussian\scripts\verify_calchem_job.py")


def run_case(root: Path, name: str, task: str, grid: str, aux: str | None, extra: list[str] | None = None):
    out_dir = root / name
    cmd = [
        sys.executable, str(VERIFIER),
        "--exe", str(EXE), "--geometry", str(GEOM), "--output", str(out_dir),
        "--task", task, "--grid", grid, "--basis", "6-31g(d,p)", "--ao", "260",
        "--xc", "b3lyp", "--disp", "d3bj", "--threads", "8",
    ]
    if aux:
        cmd += ["--aux", aux]
    if extra:
        cmd += ["--"] + extra

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAIL {name} ({dt:.2f}s): {proc.stderr[-300:]}", flush=True)
        return {"name": name, "ok": False, "wall_sec": round(dt, 2), "error": proc.stderr[-300:]}
    res = json.loads((out_dir / "result.json").read_text(encoding="utf8"))
    e = res["energy_hartree"]
    de_kcal = (e - G16_REF_E) * KCAL
    row = {
        "name": name, "ok": True, "wall_sec": round(dt, 2),
        "scf_sec": round(res["timing_sec"]["scf_sec"], 2),
        "post_sec": round(res["timing_sec"]["post_scf_sec"], 2),
        "scf_iter": res["scf_iterations"],
        "energy_hartree": e,
        "delta_vs_g16_kcal": round(de_kcal, 4),
        "max_grad": res.get("max_gradient_atomic_units"),
        "rms_grad": res.get("rms_gradient_atomic_units"),
    }
    print(
        f"{name:22s} | wall={dt:6.2f}s (scf={row['scf_sec']:5.2f}s post={row['post_sec']:5.2f}s) "
        f"| iter={row['scf_iter']:2d} | E={e:.10f} | dE_G16={de_kcal:+.4f} kcal/mol | max_g={row['max_grad']}",
        flush=True,
    )
    return row


def main():
    root = Path(r"D:\YDW\Calc\qc_benchmark_runs\pilot") / f"calchem_postfix_20261123_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== Post-Fix Caffeine Verification (8 threads CPU) -> {root} ===", flush=True)
    results = []
    # 1. Analytical gradient speedup & accuracy check (compare vs previous 648.18s & max_g=2.5204e-2)
    results.append(run_case(root, "grad_fine_def2jfit", "gradient", "fine", "def2-jfit-f"))
    # 2. Exact-4c Fine SCF convergence check (previously failed at 100 iters / 548s!)
    results.append(run_case(root, "sp_fine_exact", "energy", "fine", None))
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"Saved post-fix summary to {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
