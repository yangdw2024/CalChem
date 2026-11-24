"""Run strictly 8-thread CPU Caffeine diagnostics (4-cell accuracy attribution, gradient, and short opt+resume)."""
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


def run_verified(root: Path, name: str, task: str, grid: str, aux: str | None, extra: list[str] | None = None):
    out_dir = root / name
    cmd = [
        sys.executable,
        str(VERIFIER),
        "--exe", str(EXE),
        "--geometry", str(GEOM),
        "--output", str(out_dir),
        "--task", task,
        "--grid", grid,
        "--basis", "6-31g(d,p)",
        "--ao", "260",
        "--xc", "b3lyp",
        "--disp", "d3bj",
        "--threads", "8",
    ]
    if aux:
        cmd += ["--aux", aux]
    if extra:
        cmd += ["--"] + extra

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAIL {name} (exit={proc.returncode}, {dt:.2f}s): {proc.stderr[-300:]}", flush=True)
        return {"name": name, "ok": False, "wall_sec": round(dt, 2), "error": proc.stderr[-300:]}

    res = json.loads((out_dir / "result.json").read_text(encoding="utf8"))
    e = res["energy_hartree"]
    de_kcal = (e - G16_REF_E) * KCAL
    row = {
        "name": name,
        "ok": True,
        "task": task,
        "grid": grid,
        "aux": aux or "exact-4c",
        "threads": res["threads"],
        "wall_sec": round(dt, 2),
        "scf_sec": round(res["timing_sec"]["scf_sec"], 2),
        "post_sec": round(res["timing_sec"]["post_scf_sec"], 2),
        "scf_iter": res["scf_iterations"],
        "energy_hartree": e,
        "disp_hartree": res.get("dispersion_energy_hartree"),
        "delta_vs_g16_kcal": round(de_kcal, 4),
        "max_grad": res.get("max_gradient_atomic_units"),
        "rms_grad": res.get("rms_gradient_atomic_units"),
        "homo_hartree": res["scf"].get("homo_hartree"),
        "lumo_hartree": res["scf"].get("lumo_hartree"),
    }
    print(
        f"{name:20s} | wall={dt:6.2f}s (scf={row['scf_sec']:5.2f}s post={row['post_sec']:5.2f}s) "
        f"| iter={row['scf_iter']:2d} | E={e:.10f} | dE_G16={de_kcal:+.4f} kcal/mol",
        flush=True,
    )
    return row


def main():
    root = Path(r"D:\YDW\Calc\qc_benchmark_runs\pilot") / f"calchem_diag_20261123_{int(time.time())}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"=== Caffeine 8-Thread CPU Diagnostic Suite -> {root} ===", flush=True)
    print(f"Gaussian 16 Reference (8 threads, B3LYP-D3BJ/6-31G(d,p), 260 AO): {G16_REF_E:.9f} Eh (31.5 s)", flush=True)
    print("-" * 110, flush=True)

    cases = [
        ("sp_coarse_autoj", "energy", "coarse", "auto-jfit", []),
        ("sp_coarse_exact", "energy", "coarse", None, []),
        ("sp_fine_autoj", "energy", "fine", "auto-jfit", []),
        ("sp_fine_def2jfit", "energy", "fine", "def2-jfit-f", []),
        ("sp_fine_exact", "energy", "fine", None, []),
        ("grad_fine_def2jfit", "gradient", "fine", "def2-jfit-f", []),
    ]

    summary = []
    for name, task, grid, aux, extra in cases:
        summary.append(run_verified(root, name, task, grid, aux, extra))

    # Short 2-step opt with checkpoint & 1-step resume (using fine + def2-jfit-f, strictly 8 threads)
    print("-" * 110, flush=True)
    print("Running short 2-step Caffeine Opt + Checkpoint verification (8 threads, fine + def2-jfit-f)...", flush=True)
    opt_dir = root / "opt_short_2steps"
    opt_dir.mkdir(parents=True, exist_ok=True)
    cp_file = opt_dir / "caffeine.checkpoint.json"
    out_file = opt_dir / "opt.json"
    cmd_opt = [
        str(EXE), str(GEOM),
        "--task", "opt", "--method", "rks", "--xc", "b3lyp",
        "--basis", "6-31g(d,p)", "--grid", "fine", "--aux-basis", "def2-jfit-f",
        "--disp", "d3bj", "--direct", "--threads", "8",
        "--opt-max-cycles", "2", "--checkpoint", str(cp_file),
        "--json", "--out", str(out_file),
    ]
    env = dict(**__import__("os").environ, RAYON_NUM_THREADS="8", CALCHEM_DISABLE_GPU="1")
    t0 = time.perf_counter()
    p = subprocess.run(cmd_opt, cwd=opt_dir, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    (opt_dir / "opt_stderr.log").write_text(p.stderr, encoding="utf8")
    cp_exists = cp_file.exists()
    cp_steps = []
    if cp_exists:
        cp_data = json.loads(cp_file.read_text(encoding="utf8"))
        cp_steps = cp_data["state"]["history"]
    print(f"Short 2-step opt finished in {dt:.2f}s (exit={p.returncode}, checkpoint_exists={cp_exists}, saved_steps={len(cp_steps)})", flush=True)
    for line in p.stderr.splitlines():
        if line.startswith("opt "):
            print("  " + line, flush=True)

    summary.append({
        "name": "opt_short_2steps",
        "exit_code": p.returncode,
        "wall_sec": round(dt, 2),
        "checkpoint_exists": cp_exists,
        "out_json_exists_on_limit": out_file.exists(),
        "saved_history": cp_steps,
    })

    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("=" * 110, flush=True)
    print(f"Saved complete diagnostic summary to {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
