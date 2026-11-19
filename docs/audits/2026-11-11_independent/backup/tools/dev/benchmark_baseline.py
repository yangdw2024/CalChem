"""Comprehensive benchmark script for CalChem Rust core performance.

Measures:
1. RHF Single Point Calculation (Energy & Time)
2. Analytic Gradient Calculation (Gradient & Time)
3. BFGS Geometry Optimization (Steps & Time)
Across various basis sets (sto-3g, 6-31g) and molecules (H2, H2O, CH4).
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure calchem is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    import calchem._core as rust_core
except ImportError as e:
    print(f"Error: Could not import calchem._core: {e}")
    sys.exit(1)


@dataclass
class BenchmarkCase:
    name: str
    symbols: List[str]
    geometry_bohr: List[Tuple[f64, f64, f64]] if "f64" in globals() else List[Tuple[float, float, float]]
    basis: str
    charge: int = 0
    multiplicity: int = 1


BENCHMARK_CASES = [
    BenchmarkCase(
        name="H2",
        symbols=["H", "H"],
        geometry_bohr=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.4)],
        basis="sto-3g",
    ),
    BenchmarkCase(
        name="H2O",
        symbols=["O", "H", "H"],
        geometry_bohr=[
            (0.0, 0.0, 0.0),
            (0.0, 1.430429, -1.107157),
            (0.0, -1.430429, -1.107157),
        ],
        basis="sto-3g",
    ),
    BenchmarkCase(
        name="H2O",
        symbols=["O", "H", "H"],
        geometry_bohr=[
            (0.0, 0.0, 0.0),
            (0.0, 1.430429, -1.107157),
            (0.0, -1.430429, -1.107157),
        ],
        basis="6-31g",
    ),
    BenchmarkCase(
        name="CH4",
        symbols=["C", "H", "H", "H", "H"],
        geometry_bohr=[
            (0.0, 0.0, 0.0),
            (1.189, 1.189, 1.189),
            (-1.189, -1.189, 1.189),
            (-1.189, 1.189, -1.189),
            (1.189, -1.189, -1.189),
        ],
        basis="sto-3g",
    ),
    BenchmarkCase(
        name="CH4",
        symbols=["C", "H", "H", "H", "H"],
        geometry_bohr=[
            (0.0, 0.0, 0.0),
            (1.189, 1.189, 1.189),
            (-1.189, -1.189, 1.189),
            (-1.189, 1.189, -1.189),
            (1.189, -1.189, -1.189),
        ],
        basis="6-31g",
    ),
]


def run_benchmark(n_repeats: int = 3) -> Dict[str, dict]:
    results = {}
    print(f"=== CalChem Performance Benchmark (Rust Core) ===")
    print(f"Available Basis Sets: {rust_core.supported_basis_sets()}")
    print(f"CPU count: {os.cpu_count()}")
    print("-" * 75)

    for case in BENCHMARK_CASES:
        key = f"{case.name}/{case.basis}"
        print(f"\nEvaluating: {key} (N_atoms = {len(case.symbols)})")

        # 1. Single Point Timing
        # Warmup
        try:
            wf = rust_core.run_single_point(
                case.symbols,
                case.geometry_bohr,
                case.charge,
                case.multiplicity,
                "rhf",
                case.basis,
            )
        except Exception as e:
            print(f"  [FAIL] Single point failed: {e}")
            continue

        sp_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            wf = rust_core.run_single_point(
                case.symbols,
                case.geometry_bohr,
                case.charge,
                case.multiplicity,
                "rhf",
                case.basis,
            )
            t1 = time.perf_counter()
            sp_times.append((t1 - t0) * 1000.0)

        min_sp = min(sp_times)
        avg_sp = sum(sp_times) / len(sp_times)
        print(f"  Single Point:   Energy = {wf.total_energy_hartree:14.8f} Eh | Iter = {wf.iterations:2d} | Time = {min_sp:6.2f} ms (min), {avg_sp:6.2f} ms (avg)")

        # 2. Gradient Timing
        grad_times = []
        grad = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            grad = rust_core.run_gradient(
                case.symbols,
                case.geometry_bohr,
                case.charge,
                case.multiplicity,
                case.basis,
            )
            t1 = time.perf_counter()
            grad_times.append((t1 - t0) * 1000.0)

        min_grad = min(grad_times)
        avg_grad = sum(grad_times) / len(grad_times)
        # grad is a numpy array (N, 3)
        max_f = float(abs(grad).max()) if grad is not None else 0.0
        print(f"  Analytic Grad:  Max Force = {max_f:11.6f} Eh/Bohr | Time = {min_grad:6.2f} ms (min), {avg_grad:6.2f} ms (avg)")

        # 3. Geometry Optimization (sto-3g only for quick turnaround)
        opt_time_ms = None
        opt_steps = None
        if case.basis == "sto-3g":
            t0 = time.perf_counter()
            opt_res = rust_core.run_optimize(
                case.symbols,
                case.geometry_bohr,
                case.basis,
            )
            t1 = time.perf_counter()
            opt_time_ms = (t1 - t0) * 1000.0
            opt_steps = opt_res.steps
            print(f"  Geometry Opt:   Converged = {opt_res.converged} in {opt_steps} steps | E_final = {opt_res.energy:14.8f} Eh | Time = {opt_time_ms:6.2f} ms")

        results[key] = {
            "energy": wf.total_energy_hartree,
            "n_ao": wf.n_ao,
            "sp_time_ms": min_sp,
            "grad_time_ms": min_grad,
            "opt_time_ms": opt_time_ms,
            "opt_steps": opt_steps,
        }

    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_benchmark(n_repeats=n)
