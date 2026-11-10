"""Read-only project audit; write evidence only beneath this audit directory."""
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def guard(event, args):
    if event == "open":
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)):
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            )
            if writing and not Path(path).resolve().is_relative_to(AUDIT):
                raise PermissionError(f"Audit write outside evidence directory blocked: {path}")
    elif event in ("os.mkdir", "os.remove", "os.rmdir"):
        if not Path(args[0]).resolve().is_relative_to(AUDIT):
            raise PermissionError(f"Audit mutation outside evidence directory blocked: {args[0]}")


sys.addaudithook(guard)
summary = {"root": str(ROOT), "audit": str(AUDIT), "checks": {}}


def check(name, action):
    started = time.perf_counter()
    try:
        value = action()
        summary["checks"][name] = {"executed": True, "value": value,
                                   "seconds": time.perf_counter() - started}
        print(name, json.dumps(value, ensure_ascii=False, default=str), flush=True)
    except Exception as exc:
        summary["checks"][name] = {"executed": False, "error": repr(exc),
                                   "traceback": traceback.format_exc()}
        print(name, repr(exc), flush=True)


def static_inventory():
    entries, failures = [], []
    for folder in ("calchem", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8-sig")
                tree = ast.parse(source, filename=str(path))
                compile(tree, str(path), "exec")
                funcs = [node.name for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name.startswith("test_")]
                entries.append({"path": str(path.relative_to(ROOT)), "lines": len(source.splitlines()),
                                "pytest_functions": funcs, "bytes": path.stat().st_size})
            except Exception as exc:
                failures.append({"path": str(path), "error": repr(exc)})
    (AUDIT / "evidence" / "python-inventory.json").write_text(
        json.dumps({"files": entries, "failures": failures}, indent=2), encoding="utf-8")
    return {"python_files": len(entries), "syntax_failures": failures,
            "test_files_with_top_level_test_functions": sum(bool(e["pytest_functions"]) for e in entries
                                                              if e["path"].startswith("tests"))}


def environment():
    deps = {}
    for name in ("numpy", "scipy", "pydantic", "pytest", "PyQt6", "ruff", "h5py", "PyYAML", "click", "rich"):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {"python": sys.version, "executable": sys.executable, "dependencies": deps,
            "environment": {key: os.environ.get(key) for key in (
                "TEMP", "TMP", "CALCHEM_DATA", "CALCHEM_RUN_DIR", "CALCHEM_CONFIG",
                "CARGO_HOME", "RUSTUP_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")}}


check("static_inventory", static_inventory)
check("environment", environment)

import numpy as np
import calchem
from calchem import _core
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.api.service import TaskService
import calchem.backends

check("import_and_extension", lambda: {
    "package": calchem.__file__, "package_version": calchem.__version__,
    "extension": _core.__file__, "extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
    "signature": str(inspect.signature(_core.run_single_point)),
    "exports": [name for name in dir(_core) if not name.startswith("_")]})

B2A = 0.529177210903
CASES = {
    "H2": (["H", "H"], [[0, 0, 0], [0, 0, 1.4]], -1.116714325063),
    "He": (["He"], [[0, 0, 0]], -2.807783957540),
    "Ne": (["Ne"], [[0, 0, 0]], -126.6045291715),
    "H2O": (["O", "H", "H"], [[0, 0, 0], [0, 1.432, 0], [1.112, -0.716, 0]], -74.69531911423618),
}
service = TaskService(data_root=str(AUDIT / "runtime" / "service"))


def request(name="H2", **kwargs):
    symbols, coords, _ = CASES[name]
    return CalculationRequest(task="single_point", backend="native", method="rhf", basis="sto-3g",
                              molecule=MoleculeSpec(symbols=symbols,
                                  coordinates_angstrom=(np.array(coords) * B2A).tolist()), **kwargs)


def baselines():
    out = {}
    from calchem.mol.molecule import Molecule
    from calchem.scf.hf import RHF
    for name, (symbols, coords, ref) in CASES.items():
        rust = _core.run_single_point(symbols, coords, 0, 1, "rhf", "sto-3g")
        with contextlib.redirect_stdout(io.StringIO()):
            py = RHF(Molecule(symbols=symbols, coordinates=coords, units="bohr"), basis="sto-3g")
            py.kernel()
        routed = service.run(request(name))
        out[name] = {"rust_energy": rust.total_energy_hartree, "python_energy": float(py.energy),
                     "rust_minus_reference": rust.total_energy_hartree - ref,
                     "rust_minus_python": rust.total_energy_hartree - float(py.energy),
                     "python_converged": bool(py.converged), "rust_converged": rust.converged,
                     "service_status": routed.status, "service_actual_method": routed.actual_method,
                     "service_orbitals": len(routed.orbitals), "service_homo": routed.homo_energy_hartree}
    return out


check("four_molecule_baselines", baselines)
check("631g_direct_bridge", lambda: {name: _core.run_single_point(symbols, coords, 0, 1, "rhf", "6-31g").total_energy_hartree
                                     for name, (symbols, coords, _) in CASES.items() if name in ("H2", "H2O")})


def ignored_options():
    direct = None
    try:
        _core.run_single_point(CASES["H2O"][0], CASES["H2O"][1], max_iter=1)
    except Exception as exc:
        direct = repr(exc)
    routed = service.run(request("H2O", options={"max_iter": 1}))
    return {"direct_max_iter_1_error": direct, "routed_status": routed.status,
            "routed_iterations": routed.scf_iterations, "routed_actual_method": routed.actual_method}


check("rust_option_forwarding", ignored_options)


def task_plumbing():
    from unittest.mock import patch
    from calchem.config.models import AppConfig
    from calchem.jobs.dispatcher import JobDispatcher
    from calchem.jobs.models import JobState
    from calchem.jobs.queue import JobQueue
    with patch.object(Path, "mkdir"):
        env_only = TaskService()
    job = service.submit(request())
    dispatcher = JobDispatcher(service)
    q = JobQueue()
    qj = q.submit("{}")
    q.get_next()
    q.cancel(qj.job_id)
    q.complete(qj.job_id, success=True)
    return {"environment_CALCHEM_DATA": os.environ["CALCHEM_DATA"],
            "default_service_data_root": str(env_only.data_root),
            "submitted_job_state": job.state, "dispatcher_uses_service_queue": dispatcher.queue is service.queue,
            "after_cancel_and_success_state": qj.state, "queue_running_id_after_completion": q.running_job_id,
            "default_config_data_root": AppConfig().data_root}


check("task_plumbing", task_plumbing)


def parsers_and_adapters():
    from unittest.mock import patch
    from types import SimpleNamespace
    from calchem.backends.gaussian import GaussianBackend, _parse_g16_log, _parse_frequencies
    from calchem.backends.orca import ORCABackend
    from calchem.config.models import AppConfig
    req = request()
    log = "SCF Done:  E(RHF) =  -1.100000 A.U.\nSCF Done:  E(RHF) =  -1.200000 A.U.\nFrequencies -- 100.0 200.0 300.0\nNormal termination"
    g = GaussianBackend(executable=sys.executable)
    freq_req = req.model_copy(update={"backend": "gaussian", "task": "frequency"})
    with patch("calchem.backends.gaussian.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=log, stderr="")):
        freq = g.frequency(freq_req)
    orca_error = None
    o = ORCABackend(executable=sys.executable)
    with patch("calchem.backends.orca.subprocess.run", return_value=SimpleNamespace(returncode=0,
            stdout="FINAL SINGLE POINT ENERGY -1.1\nSCF CONVERGED\nORCA TERMINATED NORMALLY", stderr="")):
        try:
            o.single_point(req.model_copy(update={"backend": "orca"}))
        except Exception as exc:
            orca_error = repr(exc)
    return {"two_SCF_lines_selected_energy": _parse_g16_log(log)["energy_hartree"],
            "expected_last_energy": -1.2, "standalone_frequency_parser": _parse_frequencies(log),
            "gaussian_frequency_result_status": freq.status, "gaussian_frequency_result_values": freq.frequencies_cm,
            "orca_result_error": orca_error,
            "note": "Synthetic parser/adapter diagnostics only; no external quantum program was launched."}


check("parser_adapter_diagnostics", parsers_and_adapters)


def gradient_opt():
    from calchem.grad.finite_difference import central_gradient
    from calchem.backends.native import NativeCalchem
    native = NativeCalchem()
    req = request()

    def energy(spec):
        return native.single_point(req.model_copy(update={"molecule": spec})).total_energy_hartree

    grad = central_gradient(energy, req.molecule, step_bohr=0.002)
    with contextlib.redirect_stdout(io.StringIO()):
        optimized = service.run(req.model_copy(update={"task": "optimize"}))
    c = np.asarray(optimized.optimized_coordinates_angstrom)
    return {"gradient_hartree_per_bohr": grad.tolist(), "gradient_total": grad.sum(axis=0).tolist(),
            "optimization_status": optimized.status, "energy": optimized.total_energy_hartree,
            "bond_angstrom": float(np.linalg.norm(c[1] - c[0])), "implementation": "Python finite differences + SciPy BFGS; Rust single-point energy route"}


check("h2_numeric_gradient_and_optimization", gradient_opt)

import pytest
pytest_exit = pytest.main([
    "tests/unit/test_request_validation.py", "tests/unit/test_molecule.py", "tests/unit/test_io.py",
    "tests/unit/test_core_rust.py", "-q", "--tb=short", "-p", "no:cacheprovider",
    "--log-file", str(AUDIT / "evidence" / "pytest.log"),
    "--basetemp", str(AUDIT / "runtime" / "pytest"),
])
summary["pytest_exit_code"] = int(pytest_exit)
(AUDIT / "evidence" / "checks.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("AUDIT_COMPLETE", flush=True)
