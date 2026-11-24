"""Run one CPU calibration in a fresh directory and validate the actual CLI output."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    binary = args.exe.resolve(strict=True)
    source = args.geometry.resolve(strict=True)
    model = list(args.options)
    if model[:1] == ["--"]:
        model = model[1:]
    locked = {"--json", "--out", "--threads", "--task", "--basis", "--method", "--xc", "--grid", "--gpu", "--rij", "--direct", "--stored", "--disp", "--aux-basis", "--charge", "--mult"}
    if any(arg in locked or not arg.startswith("--") and Path(arg).suffix.lower() in {".xyz",".gjf",".com"} for arg in model):
        raise ValueError("Conflicting overrides in additional task options")
    local = directory / ("geometry" + source.suffix.lower())
    local.write_bytes(source.read_bytes())
    if args.task == "freq":
        if not args.stationary_result:
            raise ValueError("Frequency calibration requires an optimization result")
        previous = json.loads(args.stationary_result.read_text(encoding="utf8"))
        if previous.get("task") != "opt" or not previous["converged"]:
            raise ValueError("Missing converged optimized geometry")
        atoms = previous["optimized_atoms"]
        if not atoms:
            raise ValueError("Empty optimized geometry")
        if (previous["basis"], previous["xc"], previous["grid"].lower(), previous["density_fitting"],
            previous["scf"]["diagnostics"]["jfit_basis"], previous["task_settings"]["options"]["dispersion"]) != (
                args.basis, args.xc, args.grid, bool(args.aux), "Def2UpToF" if args.aux == "def2-jfit-f" else "Auto", args.disp):
            raise ValueError("Stationary result used a different model")
        if previous.get("method") != "rks" or previous.get("solvation_energy_kcal_mol") is not None:
            raise ValueError("Stationary reference mismatch")
        if [a["symbol"] for a in atoms] != [line.split()[0] for line in local.read_text().splitlines()[2:] if line.strip()]:
            raise ValueError("Stationary result atom sequence differs from input")
        local = directory / "optimized.xyz"
        rows = ["{} {:.15g} {:.15g} {:.15g}".format(a["symbol"], a["x_angstrom"], a["y_angstrom"], a["z_angstrom"]) for a in atoms]
        local.write_text("\n".join([str(len(rows)), "converged geometry", *rows]) + "\n",encoding="utf8")
    out = directory / "result.json"
    cmd = [str(binary), str(local), "--task", args.task, "--threads", str(args.threads),
           "--method", "rks", "--xc", args.xc, "--basis", args.basis, "--grid", args.grid,
           "--direct", "--charge", "0", "--mult", "1", "--json", "--out", str(out)]
    if args.aux:
        cmd += ["--aux-basis", args.aux]
    if args.disp:
        cmd += ["--disp", args.disp]
    cmd += model
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TEMP=str(directory), TMP=str(directory),
               CALCHEM_MAX_MEMORY_BYTES=str(8 << 30), CALCHEM_MAX_ERI_BYTES=str(4 << 30), CALCHEM_DISABLE_GPU="1")
    manifest = {"command": cmd, "binary_sha256": digest(binary), "geometry_sha256": digest(local),
                "input_sha256": digest(source), "status": "running", "threads": args.threads}
    record = directory / "manifest.json"
    def save():
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf8")
        tmp.replace(record)
    save()
    try:
        begin = time.perf_counter()
        with (directory / "process.log").open("wb") as log:
            proc = subprocess.run(cmd, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest.update(exit_code=proc.returncode, elapsed_sec=time.perf_counter()-begin)
        proc.check_returncode()
        result = json.loads(out.read_text(encoding="utf8"))
        actual = (result["task"], result["method"], result["xc"], result["basis"], result["grid"].lower(),
                  result["n_ao"],result["threads"],result["density_fitting"],result["direct_scf"],result["gpu_requested"])
        expected = (args.task, "rks",args.xc,args.basis,args.grid,args.ao,args.threads,bool(args.aux),True,False)
        if actual != expected or not result["converged"] or result.get("status") != "converged" or not math.isfinite(result["energy_hartree"]):
            raise ValueError(f"Calculation contract mismatch: {actual!r} vs {expected!r}")
        if not result.get("initial_geometry_sha256") or not result.get("final_geometry_sha256"):
            raise ValueError("Missing geometry SHA256 hashes in result JSON")
        if (result.get("dispersion_energy_hartree") is not None) != bool(args.disp):
            raise ValueError("Dispersion setting mismatch")
        if args.task == "opt" and len(result["optimized_atoms"]) != result["n_atoms"]:
            raise ValueError("Incomplete optimized geometry")
        if args.task == "freq" and not result["frequencies_cm1"]:
            raise ValueError("Missing vibrational modes")
        if args.task == "tddft":
            states = result.get("excited_states") or []
            if not states:
                raise ValueError("Missing excited states")
            if any(s.get("residual_norm_hartree") is None or s["residual_norm_hartree"] > 1e-4 for s in states):
                raise ValueError("Unconverged excited state residual")
        manifest.update(status="verified", result_sha256=digest(out), energy_hartree=result["energy_hartree"])
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "geometry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--stationary-result",type=Path)
    parser.add_argument("--task",choices=["energy","gradient","opt","freq","tddft"],required=True)
    parser.add_argument("--grid",choices=["coarse","medium","fine","ultrafine"],required=True)
    parser.add_argument("--aux",choices=["auto-jfit","def2-jfit-f"])
    parser.add_argument("--basis",required=True)
    parser.add_argument("--ao",type=int,required=True)
    parser.add_argument("--xc",default="b3lyp")
    parser.add_argument("--disp",choices=["d3bj","d3zero"])
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("options",nargs=argparse.REMAINDER)
    run(parser.parse_args())
