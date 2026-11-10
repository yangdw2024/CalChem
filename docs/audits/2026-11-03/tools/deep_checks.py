"""Additional bounded diagnostics; no external chemistry executables are run."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import calchem.backends
from calchem.backends.native import _build_calculator
from calchem.core.request import CalculationRequest, MoleculeSpec
from calchem.mol.molecule import Molecule
from calchem.api.service import TaskService

mol = Molecule(symbols=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 1.4]], units="bohr")
out = {"native_constructor_checks": {}}
for method in ("rhf", "uhf", "rohf", "rks", "uks", "mp2", "mp4", "ccsd", "ccsd(t)", "ccd", "casscf"):
    try:
        calc = _build_calculator(mol, method, "sto-3g", "pbe", {})
        out["native_constructor_checks"][method] = {"class": type(calc).__name__, "has_kernel": hasattr(calc, "kernel")}
    except Exception as exc:
        out["native_constructor_checks"][method] = {"error": repr(exc)}

service = TaskService(data_root=str(AUDIT / "runtime" / "deep-service"))
req = CalculationRequest(task="single_point", molecule=MoleculeSpec(symbols=["H", "H"],
    coordinates_angstrom=[[0, 0, 0], [0, 0, 1.4 * 0.529177210903]]), backend="native", method="rhf", basis="sto-3g")
from click.testing import CliRunner
from calchem.cli import cli
runner = CliRunner()
with patch("calchem.cli.TaskService", return_value=service):
    help_result = runner.invoke(cli, ["--help"])
    run_result = runner.invoke(cli, ["run", "--molecule", "H2", "--json"])
    try:
        parsed = json.loads(run_result.stdout)
        json_valid = True
    except Exception as exc:
        parsed = repr(exc)
        json_valid = False
out["cli"] = {"help": help_result.stdout, "run_exit": run_result.exit_code,
              "json_valid": json_valid, "result": parsed, "exception": repr(run_result.exception)}

# Confirm service persistence behavior and numerical failure classification.
saved = service.run_and_save(req.model_copy(update={"job_id": "audit_h2"}))
from calchem.core.result import Result
saved_path = service.run_root / "audit_h2" / "result.json"
loaded = Result.model_validate_json(saved_path.read_text())
out["result_persistence"] = {"path": str(saved_path), "energy_matches": loaded.total_energy_hartree == saved.total_energy_hartree,
    "result_job_id": loaded.job_id, "request_saved": (saved_path.parent / "request.json").exists(),
    "manifest_saved": (saved_path.parent / "manifest.json").exists()}

# A failing method should be described accurately, without launching expensive calculations.
out["service_mp2_error"] = None
try:
    service.run(req.model_copy(update={"method": "mp2"}))
except Exception as exc:
    out["service_mp2_error"] = repr(exc)

test_files = []
for path in sorted((ROOT / "tests").rglob("test_*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and n.name.startswith("Test")]
    module_calls = [n.lineno for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    test_files.append({"path": str(path.relative_to(ROOT)), "test_functions": functions,
                       "test_classes": classes, "top_level_call_lines": module_calls})
out["test_structure"] = {"test_named_files": len(test_files), "files_with_test_functions_or_classes": sum(bool(r["test_functions"] or r["test_classes"]) for r in test_files),
                         "files_without_test_functions_or_classes": sum(not r["test_functions"] and not r["test_classes"] for r in test_files)}
(AUDIT / "evidence" / "test-structure.json").write_text(json.dumps(test_files, indent=2), encoding="utf-8")

# Existing lint tool, no cache, no fixes. Output is evidence, not a source mutation.
lint = subprocess.run([r"D:\YDW\Trae_Gromacs\miniconda3\Scripts\ruff.exe", "check", "calchem", "--select", "F821,F822,F823", "--no-cache", "--output-format", "json"],
                      cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
(AUDIT / "evidence" / "ruff-critical.json").write_text(lint.stdout, encoding="utf-8")
lint_data = json.loads(lint.stdout)
out["critical_lint"] = {"exit_code": lint.returncode, "findings": len(lint_data),
    "files": sorted({str(Path(x["filename"]).relative_to(ROOT)) for x in lint_data})}

archives = []
import zipfile
for path in sorted(ROOT.glob("*.zip")):
    with zipfile.ZipFile(path) as archive:
        archives.append({"file": path.name, "entries": len(archive.infolist()),
            "python_reference_files": [n for n in archive.namelist() if "python_reference/" in n],
            "bad_crc_entry": archive.testzip()})
out["root_archive_integrity"] = archives
(AUDIT / "evidence" / "deep-checks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
for key, value in out.items():
    print(key, json.dumps(value, ensure_ascii=False), flush=True)
