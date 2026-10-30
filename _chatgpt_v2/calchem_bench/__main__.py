from __future__ import annotations
import argparse, json, pathlib, platform, statistics, sys, time, tracemalloc
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.grad.analytical import compute_hf_gradient
from calchem.grad.hf_grad import HFGradient

def run_benchmarks(manifest_path: str, output_path: str) -> int:
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    results = {
        "schema_version": "1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "numpy_version": np.__version__,
        },
        "cases": {},
    }
    for case in manifest["cases"]:
        cid = case["id"]
        mol = Molecule(case["symbols"], case["coordinates_bohr"], charge=case.get("charge", 0), multiplicity=case.get("multiplicity", 1), units="bohr")
        basis = case["basis"]
        warmup = case.get("warmup", 1)
        repeats = case.get("repeats", 3)
        for _ in range(warmup):
            c = RHF(mol, basis, energy_tol=1e-12, density_tol=1e-10, gradient_tol=1e-8)
            c.kernel()
        timings_ms = []
        tracemalloc.start()
        last_calc = None
        last_grad = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            calc = RHF(mol, basis, energy_tol=1e-12, density_tol=1e-10, gradient_tol=1e-8)
            calc.kernel()
            if case.get("compute_gradient", False):
                last_grad = compute_hf_gradient(calc)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            last_calc = calc
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        comm = last_calc.fock @ last_calc.dm @ last_calc._s - last_calc._s @ last_calc.dm @ last_calc.fock
        tr_ds = float(np.trace(last_calc.dm @ last_calc._s))
        grad_fd_err = None
        if last_grad is not None and case.get("verify_fd", False):
            g_fd = HFGradient(last_calc, numerical=True, step=1e-5).kernel()
            grad_fd_err = float(np.max(np.abs(last_grad - g_fd)))

        results["cases"][cid] = {
            "converged": bool(last_calc.converged),
            "iterations": int(last_calc._n_iter),
            "energy_eh": float(last_calc.energy),
            "trace_ds": tr_ds,
            "commutator_rms": float(np.sqrt(np.mean(comm ** 2))),
            "grad_norm_eh_bohr": float(np.linalg.norm(last_grad)) if last_grad is not None else None,
            "grad_max_fd_error_eh_bohr": grad_fd_err,
            "median_walltime_ms": float(statistics.median(timings_ms)),
            "iqr_walltime_ms": float(np.percentile(timings_ms, 75) - np.percentile(timings_ms, 25)),
            "peak_traced_memory_kb": float(peak_bytes / 1024.0),
        }
    out_file = pathlib.Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[calchem_bench] Saved benchmark results ({len(results['cases'])} cases) to {out_file}")
    return 0

def compare_benchmarks(result_path: str, ref_dir: str) -> int:
    actual = json.loads(pathlib.Path(result_path).read_text(encoding="utf-8"))
    ref_file = pathlib.Path(ref_dir) / "stage1_reference.json"
    reference = json.loads(ref_file.read_text(encoding="utf-8"))
    failed = 0
    print("=" * 82)
    print(f"{'Case ID':<20} | {'Energy (Eh)':<16} | {'|dE vs Ref|':<11} | {'Grad FD Err':<11} | {'Time(ms)':<8} | {'Status'}")
    print("-" * 82)
    for cid, ref_case in reference["cases"].items():
        act = actual["cases"][cid]
        de = abs(act["energy_eh"] - ref_case["energy_eh"])
        tol = ref_case.get("energy_tol_eh", 1e-7)
        g_err = act.get("grad_max_fd_error_eh_bohr")
        g_ok = True
        if ref_case.get("max_grad_fd_error") is not None:
            g_ok = (g_err is not None) and (g_err <= ref_case["max_grad_fd_error"])
        ok = act["converged"] and (de <= tol) and g_ok
        if not ok:
            failed += 1
        g_str = f"{g_err:.2e}" if g_err is not None else "N/A"
        print(f"{cid:<20} | {act['energy_eh']:<16.10f} | {de:<11.2e} | {g_str:<11} | {act['median_walltime_ms']:<8.2f} | {'PASS' if ok else 'FAIL'}")
    print("=" * 82)
    return 1 if failed > 0 else 0

def main() -> int:
    parser = argparse.ArgumentParser(prog="calchem_bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--output", required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("result_file")
    p_cmp.add_argument("--reference", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        return run_benchmarks(args.manifest, args.output)
    elif args.cmd == "compare":
        return compare_benchmarks(args.result_file, args.reference)
    return 1

if __name__ == "__main__":
    sys.exit(main())
