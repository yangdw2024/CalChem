"""Regression tests for scripts/verify_calchem_job.py validation contracts."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_calchem_job as v


class TestVerifyCalchemJobAdversarial(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_verify_calchem_"))
        self.geom = self.tmp / "h2o.xyz"
        self.geom.write_text("3\nh2o\nO 0 0 0\nH 0 0.75 0.5\nH 0 -0.75 0.5\n", encoding="utf8")
        self.exe = self.tmp / "calchem_cli.exe"
        self.exe.write_bytes(b"MZ")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _base(self):
        return {
            "status": "converged",
            "converged": True,
            "task": "energy",
            "method": "rks",
            "xc": "b3lyp",
            "basis": "sto-3g",
            "grid": "Coarse",
            "n_ao": 7,
            "n_atoms": 3,
            "threads": 4,
            "density_fitting": False,
            "direct_scf": True,
            "gpu_requested": False,
            "energy_hartree": -76.0,
            "initial_geometry_sha256": "a" * 64,
            "final_geometry_sha256": "a" * 64,
            "gpu_kernel_invocations": 0,
            "gpu_fallback_invocations": 0,
            "scf": {"converged": True, "diagnostics": {"jfit_basis": "Auto"}},
        }

    def _run_mock(self, name, payload, **kw):
        out_dir = self.tmp / name

        def fake_sub(cmd, **kwargs):
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(payload), encoding="utf8")
            return SimpleNamespace(returncode=0, check_returncode=lambda: None)

        args = SimpleNamespace(
            exe=self.exe,
            geometry=kw.get("geometry", self.geom),
            output=out_dir,
            stationary_result=kw.get("stationary"),
            task=kw.get("task", "energy"),
            grid=kw.get("grid", "coarse"),
            aux=kw.get("aux"),
            basis="sto-3g",
            ao=kw.get("ao", 7),
            xc="b3lyp",
            disp=kw.get("disp"),
            threads=4,
            gpu=kw.get("gpu", False),
            allow_gpu_fallback=False,
            options=kw.get("options", []),
        )
        with patch.object(v.subprocess, "run", side_effect=fake_sub):
            v.run(args)

    def test_rejects_all_6_bypasses(self):
        base = self._base()
        with self.assertRaises(ValueError):
            self._run_mock("bad_sha", dict(base, initial_geometry_sha256="fake"))
        with self.assertRaises(ValueError):
            self._run_mock("bad_sha_mismatch", dict(base, final_geometry_sha256="b" * 64))
        with self.assertRaises(ValueError):
            self._run_mock("bad_scf", dict(base, scf={"converged": False, "diagnostics": {"jfit_basis": "Auto"}}))
        with self.assertRaises(ValueError):
            self._run_mock("bad_grad", dict(base, task="gradient", gradient_atomic_units=None), task="gradient")
        with self.assertRaises(ValueError):
            self._run_mock(
                "bad_gpu_grad",
                dict(
                    base,
                    task="gradient",
                    gpu_requested=True,
                    gpu_kernel_invocations=5,
                    gradient_atomic_units=[[0.0, 0.0, 0.0]] * 3,
                    max_gradient_atomic_units=0.0,
                    rms_gradient_atomic_units=0.0,
                    task_settings={"gradient_gpu_kernel_invocations": 0, "gradient_gpu_fallback_invocations": 0},
                ),
                task="gradient",
                gpu=True,
            )
        with self.assertRaises(ValueError):
            self._run_mock(
                "bad_td",
                dict(
                    base,
                    task="tddft",
                    task_settings={},
                    excited_states=[
                        {
                            "residual_norm_hartree": 1e-7,
                            "energy_ev": 5.0,
                            "wavelength_nm": None,
                            "oscillator_strength": 0.1,
                        }
                    ],
                ),
                task="tddft",
                options=["--nstates", "5", "--spin", "both"],
            )
        with self.assertRaises(ValueError):
            self._run_mock("bad_aux", dict(base, density_fitting=True), aux="def2-jfit-f")

    def test_freq_stationary_gjf_and_invalid_outputs_and_single_atom_gpu_grad(self):
        base = self._base()

        # Invalid freq modes / thermochemistry
        stat_bad_grid = self.tmp / "opt_bad_grid.json"
        opt_payload = dict(
            base,
            task="opt",
            final_geometry_sha256="b" * 64,
            optimized_atoms=[
                {"symbol": "O", "x_angstrom": 0.0, "y_angstrom": 0.0, "z_angstrom": 0.0},
                {"symbol": "H", "x_angstrom": 0.0, "y_angstrom": 0.75, "z_angstrom": 0.5},
                {"symbol": "H", "x_angstrom": 0.0, "y_angstrom": -0.75, "z_angstrom": 0.5},
            ],
            task_settings={"options": {"dispersion": None}},
            grid=None,
        )
        stat_bad_grid.write_text(json.dumps(opt_payload), encoding="utf8")
        with self.assertRaises(ValueError):
            self._run_mock(
                "freq_none_grid",
                dict(base, task="freq", frequencies_cm1=[1000.0, 2000.0, 3000.0], zero_point_energy_hartree=0.02, gibbs_free_energy_hartree=-75.9),
                task="freq",
                stationary=stat_bad_grid,
            )

        # Valid .gjf stationary reference + bad frequencies_cm1
        gjf = self.tmp / "h2o.gjf"
        gjf.write_text("#P B3LYP/STO-3G\n\ntitle\n\n0 1\nO 0 0 0\nH 0 0.75 0.5\nH 0 -0.75 0.5\n\n", encoding="utf8")
        stat_ok = self.tmp / "opt_ok.json"
        stat_ok.write_text(json.dumps(dict(opt_payload, grid="Coarse")), encoding="utf8")
        with self.assertRaises(ValueError):
            self._run_mock(
                "freq_bad_modes",
                dict(base, task="freq", frequencies_cm1=["bad", None], zero_point_energy_hartree=0.02, gibbs_free_energy_hartree=-75.9),
                task="freq",
                geometry=gjf,
                stationary=stat_ok,
            )

        # Single-atom GPU gradient must NOT false-fail on gradient_gpu_kernel_invocations == 0
        he_xyz = self.tmp / "he.xyz"
        he_xyz.write_text("1\nhe\nHe 0 0 0\n", encoding="utf8")
        self._run_mock(
            "he_gpu_grad",
            dict(
                base,
                task="gradient",
                n_ao=1,
                n_atoms=1,
                gpu_requested=True,
                gpu_kernel_invocations=2,
                gradient_atomic_units=[[0.0, 0.0, 0.0]],
                max_gradient_atomic_units=0.0,
                rms_gradient_atomic_units=0.0,
                task_settings={"gradient_gpu_kernel_invocations": 0, "gradient_gpu_fallback_invocations": 0},
            ),
            task="gradient",
            geometry=he_xyz,
            ao=1,
            gpu=True,
        )


if __name__ == "__main__":
    unittest.main()
