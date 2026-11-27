"""Run one CPU or opt-in GPU calibration in a fresh directory and validate the actual CLI output."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_finite_num(val):
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(val)


def _is_non_bool_int(val):
    return not isinstance(val, bool) and isinstance(val, int)


def _parse_geometry_symbols(text, suffix):
    lines = text.splitlines()
    if suffix == ".xyz":
        return [line.split()[0] for line in lines[2:] if line.strip()]
    # Gaussian .gjf / .com format
    in_route = False
    passed_blank_after_route = False
    title_seen = False
    post_title_blank = False
    charge_mult_seen = False
    symbols = []
    for raw in lines:
        line = raw.strip()
        if not passed_blank_after_route:
            if line.startswith("#"):
                in_route = True
                continue
            if in_route and not line:
                passed_blank_after_route = True
            continue
        if not title_seen:
            title_seen = True
            if not line:
                post_title_blank = True
            continue
        if not post_title_blank:
            if not line:
                post_title_blank = True
            continue
        if not charge_mult_seen:
            if not line:
                continue
            charge_mult_seen = True
            continue
        if not line:
            if symbols:
                break
            continue
        if not symbols and line.startswith("@"):
            continue
        tok = line.split()[0].split("(")[0].strip()
        symbols.append(tok)
    return symbols


def _extract_cli_opt(model, flag, default):
    val = default
    i = 0
    while i < len(model):
        arg = model[i]
        if arg == flag and i + 1 < len(model):
            val = model[i + 1]
            i += 2
            continue
        if arg.startswith(flag + "="):
            val = arg.split("=", 1)[1]
        i += 1
    return val


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
    if args.allow_gpu_fallback and not args.gpu:
        raise ValueError("--allow-gpu-fallback requires --gpu")
    local = directory / ("geometry" + source.suffix.lower())
    local.write_bytes(source.read_bytes())
    if args.task == "freq":
        if not args.stationary_result:
            raise ValueError("Frequency calibration requires an optimization result")
        previous = json.loads(args.stationary_result.read_text(encoding="utf8"))
        if previous.get("task") != "opt" or previous.get("converged") is not True or previous.get("status") != "converged":
            raise ValueError("Missing converged optimized geometry")
        atoms = previous.get("optimized_atoms")
        if not atoms:
            raise ValueError("Empty optimized geometry")
        prev_grid = previous.get("grid")
        if not isinstance(prev_grid, str):
            raise ValueError(f"Stationary reference missing valid grid string: {prev_grid!r}")
        prev_sha = previous.get("initial_geometry_sha256")
        if not isinstance(prev_sha, str) or not _SHA256_RE.fullmatch(prev_sha):
            raise ValueError(f"Stationary reference invalid initial_geometry_sha256: {prev_sha!r}")
        prev_diag = ((previous.get("scf") or {}).get("diagnostics")) or {}
        prev_disp = ((previous.get("task_settings") or {}).get("options") or {}).get("dispersion")
        if (previous.get("basis"), previous.get("xc"), prev_grid.lower(), previous.get("density_fitting"),
            prev_diag.get("jfit_basis"), prev_disp) != (
                args.basis, args.xc, args.grid, bool(args.aux), "Def2UpToF" if args.aux == "def2-jfit-f" else "Auto", args.disp):
            raise ValueError("Stationary result used a different model")
        if previous.get("method") != "rks" or previous.get("solvation_energy_kcal_mol") is not None:
            raise ValueError("Stationary reference mismatch")
        if [a["symbol"] for a in atoms] != _parse_geometry_symbols(local.read_text(encoding="utf8"), source.suffix.lower()):
            raise ValueError("Stationary result atom sequence differs from input")
        local = directory / "optimized.xyz"
        rows = ["{} {:.15g} {:.15g} {:.15g}".format(a["symbol"], a["x_angstrom"], a["y_angstrom"], a["z_angstrom"]) for a in atoms]
        local.write_text("\n".join([str(len(rows)), "converged geometry", *rows]) + "\n", encoding="utf8")
    out = directory / "result.json"
    cmd = [str(binary), str(local), "--task", args.task, "--threads", str(args.threads),
           "--method", "rks", "--xc", args.xc, "--basis", args.basis, "--grid", args.grid,
           "--direct", "--charge", "0", "--mult", "1", "--json", "--out", str(out)]
    if args.aux:
        cmd += ["--aux-basis", args.aux]
    if args.disp:
        cmd += ["--disp", args.disp]
    if args.gpu:
        cmd.append("--gpu")
    cmd += model
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TEMP=str(directory), TMP=str(directory),
               CALCHEM_MAX_MEMORY_BYTES=str(8 << 30), CALCHEM_MAX_ERI_BYTES=str(4 << 30))
    if args.gpu:
        env.pop("CALCHEM_DISABLE_GPU", None)
    else:
        env["CALCHEM_DISABLE_GPU"] = "1"
    manifest = {"command": cmd, "binary_sha256": digest(binary), "geometry_sha256": digest(local),
                "input_sha256": digest(source), "status": "running", "threads": args.threads,
                "gpu_mode": bool(args.gpu), "allow_gpu_fallback": bool(args.allow_gpu_fallback)}
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
        energy = result.get("energy_hartree")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
            raise ValueError(f"Invalid energy_hartree: {energy!r}")
        actual = (result["task"], result["method"], result["xc"], result["basis"], (result.get("grid") or "").lower(),
                  result["n_ao"], result["threads"], result["density_fitting"], result["direct_scf"], result["gpu_requested"])
        expected = (args.task, "rks", args.xc, args.basis, args.grid, args.ao, args.threads, bool(args.aux), True, bool(args.gpu))
        scf_obj = result.get("scf") or {}
        if (
            actual != expected
            or not _is_non_bool_int(result.get("n_ao"))
            or not _is_non_bool_int(result.get("n_atoms"))
            or result["n_atoms"] <= 0
            or not _is_non_bool_int(result.get("threads"))
            or result.get("converged") is not True
            or result.get("status") != "converged"
            or scf_obj.get("converged") is not True
        ):
            raise ValueError(f"Calculation contract mismatch: {actual!r} vs {expected!r}")
        diag = scf_obj.get("diagnostics") or {}
        if args.aux:
            expected_jfit = "Def2UpToF" if args.aux == "def2-jfit-f" else "Auto"
            if diag.get("jfit_basis") != expected_jfit:
                raise ValueError(f"RI-J auxiliary basis mismatch: {diag.get('jfit_basis')!r} != {expected_jfit!r}")
        gpu_kernels = result.get("gpu_kernel_invocations")
        if gpu_kernels is None:
            gpu_kernels = diag.get("gpu_kernel_invocations")
        gpu_fallbacks = result.get("gpu_fallback_invocations")
        if gpu_fallbacks is None:
            gpu_fallbacks = diag.get("gpu_fallback_invocations")
        if args.gpu:
            if not _is_non_bool_int(gpu_kernels) or gpu_kernels < 0:
                raise ValueError(f"GPU verification failed: expected non-negative integer gpu_kernel_invocations, got {gpu_kernels!r}")
            if not _is_non_bool_int(gpu_fallbacks) or gpu_fallbacks < 0:
                raise ValueError(f"GPU verification failed: expected non-negative integer gpu_fallback_invocations, got {gpu_fallbacks!r}")
            if not args.allow_gpu_fallback:
                if gpu_kernels <= 0:
                    raise ValueError(f"GPU verification failed: expected gpu_kernel_invocations > 0, got {gpu_kernels!r}")
                if gpu_fallbacks != 0:
                    raise ValueError(f"GPU verification failed: gpu_fallback_invocations={gpu_fallbacks} != 0")
            elif gpu_kernels + gpu_fallbacks <= 0:
                raise ValueError(f"GPU verification failed: expected gpu_kernel_invocations + gpu_fallback_invocations > 0, got {gpu_kernels}+{gpu_fallbacks}")
        else:
            if gpu_kernels is not None and (not _is_non_bool_int(gpu_kernels) or gpu_kernels != 0):
                raise ValueError(f"CPU verification failed: unexpected gpu_kernel_invocations={gpu_kernels!r}")
            if gpu_fallbacks is not None and (not _is_non_bool_int(gpu_fallbacks) or gpu_fallbacks != 0):
                raise ValueError(f"CPU verification failed: unexpected gpu_fallback_invocations={gpu_fallbacks!r}")
        init_sha = result.get("initial_geometry_sha256")
        final_sha = result.get("final_geometry_sha256")
        if not isinstance(init_sha, str) or not _SHA256_RE.fullmatch(init_sha):
            raise ValueError(f"Invalid initial_geometry_sha256: {init_sha!r}")
        if not isinstance(final_sha, str) or not _SHA256_RE.fullmatch(final_sha):
            raise ValueError(f"Invalid final_geometry_sha256: {final_sha!r}")
        if args.task != "opt" and init_sha != final_sha:
            raise ValueError(f"Geometry SHA256 changed on non-opt task {args.task}: {init_sha} != {final_sha}")
        disp_e = result.get("dispersion_energy_hartree")
        if args.disp:
            if not _is_finite_num(disp_e):
                raise ValueError(f"Invalid dispersion_energy_hartree: {disp_e!r}")
        elif disp_e is not None:
            raise ValueError("Dispersion setting mismatch: unexpected dispersion_energy_hartree")
        ts = result.get("task_settings") or {}
        if args.task == "gradient":
            grad = result.get("gradient_atomic_units")
            if not isinstance(grad, list) or len(grad) != result["n_atoms"]:
                raise ValueError("Missing or mismatched gradient_atomic_units")
            if not all(isinstance(row, list) and len(row) == 3 and all(_is_finite_num(c) for c in row) for row in grad):
                raise ValueError("Non-finite or malformed gradient_atomic_units entries")
            max_g = result.get("max_gradient_atomic_units")
            rms_g = result.get("rms_gradient_atomic_units")
            if not _is_finite_num(max_g) or max_g < 0.0 or not _is_finite_num(rms_g) or rms_g < 0.0:
                raise ValueError(f"Invalid gradient norms: max={max_g!r}, rms={rms_g!r}")
            if args.gpu and result["n_atoms"] > 1:
                gk = ts.get("gradient_gpu_kernel_invocations")
                gfb = ts.get("gradient_gpu_fallback_invocations")
                if not _is_non_bool_int(gk) or not _is_non_bool_int(gfb):
                    raise ValueError(f"Missing gradient GPU telemetry: k={gk!r}, fb={gfb!r}")
                if not args.allow_gpu_fallback and (gk <= 0 or gfb != 0):
                    raise ValueError(f"Gradient GPU verification failed: kernels={gk}, fallbacks={gfb}")
        if args.task == "opt":
            opt_atoms = result.get("optimized_atoms")
            if not isinstance(opt_atoms, list) or len(opt_atoms) != result["n_atoms"]:
                raise ValueError("Incomplete optimized geometry")
            max_g = result.get("max_gradient_atomic_units")
            rms_g = result.get("rms_gradient_atomic_units")
            if not _is_finite_num(max_g) or max_g < 0.0 or not _is_finite_num(rms_g) or rms_g < 0.0:
                raise ValueError(f"Invalid optimization gradient norms: max={max_g!r}, rms={rms_g!r}")
        if args.task == "freq":
            freqs = result.get("frequencies_cm1")
            if not isinstance(freqs, list) or not all(_is_finite_num(f) for f in freqs):
                raise ValueError(f"Invalid vibrational modes: {freqs!r}")
            n_at = result["n_atoms"]
            if n_at >= 2:
                if not freqs or len(freqs) not in {3 * n_at - 6, 3 * n_at - 5}:
                    raise ValueError(f"Unexpected vibrational mode count {len(freqs)} for {n_at} atoms")
            elif len(freqs) != 0:
                raise ValueError(f"Expected 0 vibrational modes for single atom, got {len(freqs)}")
            zpe = result.get("zero_point_energy_hartree")
            gibbs = result.get("gibbs_free_energy_hartree")
            if not _is_finite_num(zpe) or not _is_finite_num(gibbs):
                raise ValueError(f"Non-finite thermochemistry: zpe={zpe!r}, gibbs={gibbs!r}")
        if args.task == "tddft":
            states = result.get("excited_states") or []
            if not states:
                raise ValueError("Missing excited states")
            nstates_raw = _extract_cli_opt(model, "--nstates", "3")
            spin_raw = _extract_cli_opt(model, "--spin", "singlet").lower()
            try:
                expected_nstates = int(nstates_raw)
            except ValueError as exc:
                raise ValueError(f"Invalid --nstates in options: {nstates_raw!r}") from exc
            if expected_nstates <= 0:
                raise ValueError(f"Expected positive --nstates, got {expected_nstates}")
            expected_states = expected_nstates * (2 if spin_raw == "both" else 1)
            if len(states) != expected_states:
                raise ValueError(f"Excited states count mismatch: {len(states)} != {expected_states}")
            if args.gpu and expected_nstates > 0:
                td_k = ts.get("tddft_gpu_kernel_invocations")
                td_fb = ts.get("tddft_gpu_fallback_invocations")
                if not _is_non_bool_int(td_k) or not _is_non_bool_int(td_fb):
                    raise ValueError(f"Missing TD-DFT GPU telemetry: k={td_k!r}, fb={td_fb!r}")
                if not args.allow_gpu_fallback:
                    if td_fb != 0:
                        raise ValueError(f"TD-DFT GPU fallback mismatch: {td_fb} != 0")
                    if (spin_raw != "triplet" or args.xc.lower() not in {"svwn5", "pbe", "blyp"}) and td_k <= 0:
                        raise ValueError(f"TD-DFT GPU kernels must be > 0, got {td_k}")
            for s in states:
                res_norm = s.get("residual_norm_hartree")
                omega_ev = s.get("energy_ev")
                wl_nm = s.get("wavelength_nm")
                osc = s.get("oscillator_strength")
                if not _is_finite_num(res_norm) or res_norm > 1e-4:
                    raise ValueError(f"Unconverged or non-finite excited state residual: {res_norm!r}")
                if not _is_finite_num(omega_ev) or omega_ev <= 0.0:
                    raise ValueError(f"Non-finite or non-positive excited state energy_ev: {omega_ev!r}")
                if not _is_finite_num(wl_nm):
                    raise ValueError(f"Non-finite excited state wavelength_nm: {wl_nm!r}")
                if not _is_finite_num(osc):
                    raise ValueError(f"Non-finite excited state oscillator_strength: {osc!r}")
        manifest.update(
            status="verified",
            result_sha256=digest(out),
            energy_hartree=result["energy_hartree"],
            gpu_requested=result["gpu_requested"],
            gpu_kernel_invocations=0 if gpu_kernels is None else gpu_kernels,
            gpu_fallback_invocations=0 if gpu_fallbacks is None else gpu_fallbacks,
        )
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
    parser.add_argument("--gpu",action="store_true",
                        help="Opt-in CUDA GPU execution and strict GPU telemetry verification")
    parser.add_argument("--allow-gpu-fallback",action="store_true",
                        help="Allow non-zero gpu_fallback_invocations when --gpu is enabled")
    parser.add_argument("options",nargs=argparse.REMAINDER)
    run(parser.parse_args())
