#!/usr/bin/env python
"""
CalChem 基准测试运行脚本

用法:
    python run_bench.py --set small --method b3lyp --basis def2-svp
"""
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from calchem.mol.molecule import Molecule
from calchem.api import single_point


BENCH_SETS = {
    "small": ["water", "methane", "ethene", "benzene"],
    "medium": ["naphthalene", "anthracene", "porphine"],
    "large": ["c60", "dna_base_pair"],
}


def load_molecule(name: str, sets_dir: Path) -> Molecule:
    xyz_file = sets_dir / name / "mol.xyz"
    if xyz_file.exists():
        return Molecule.from_xyz(xyz_file.read_text())
    raise FileNotFoundError(f"Benchmark molecule not found: {name}")


def run_benchmark(mol: Molecule, method: str, basis: str) -> dict:
    start = time.perf_counter()
    result = single_point(mol, method=method, basis=basis)
    elapsed = time.perf_counter() - start

    return {
        "molecule": mol.formula,
        "natoms": mol.natoms,
        "method": method,
        "basis": basis,
        "energy": result.energy,
        "converged": result.converged,
        "scf_cycles": getattr(result, "scf_cycles", None),
        "time_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="CalChem Benchmark")
    parser.add_argument("--set", default="small", choices=BENCH_SETS.keys(),
                        help="Benchmark set")
    parser.add_argument("--method", default="B3LYP", help="Method")
    parser.add_argument("--basis", default="def2-SVP", help="Basis set")
    parser.add_argument("--output", default="bench_results.jsonl",
                        help="Output file")
    parser.add_argument("--sets-dir", default="sets",
                        help="Directory with benchmark sets")
    args = parser.parse_args()

    sets_dir = Path(args.sets_dir)
    results = []

    print(f"CalChem Benchmark: set={args.set}, method={args.method}, basis={args.basis}")
    print("=" * 60)

    for name in BENCH_SETS[args.set]:
        try:
            mol = load_molecule(name, sets_dir)
            print(f"\nRunning {name} ({mol.natoms} atoms)...")
            bench = run_benchmark(mol, args.method, args.basis)
            print(f"  Energy:  {bench['energy']:.6f} Hartree")
            print(f"  Time:    {bench['time_seconds']:.2f} s")
            print(f"  Converged: {bench['converged']}")
            results.append(bench)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"molecule": name, "error": str(e)})

    with open(args.output, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
