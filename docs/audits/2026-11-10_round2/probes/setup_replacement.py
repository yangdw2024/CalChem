"""Build the Rust extension from this source tree; never reuse a packaged binary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Distribution, Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class RustBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name != "calchem._core":
            return super().build_extension(ext)
        if os.environ.get("CARGO_BUILD_TARGET") or os.environ.get("PYO3_CROSS"):
            raise RuntimeError("Use the maturin release workflow for cross compilation and target tags")
        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable
        cargo = env.get("CALCHEM_CARGO", env.get("CARGO", "cargo"))
        result = subprocess.run(
            [cargo, "build", "--release", "--locked", "-p", "calchem-py",
             "--message-format=json-render-diagnostics"],
            cwd=ROOT, env=env, text=True, encoding="utf-8", stdout=subprocess.PIPE,
            check=True,
        )
        artifacts = []
        for line in result.stdout.splitlines():
            msg = json.loads(line)
            if (msg.get("reason") == "compiler-artifact"
                    and msg.get("target", {}).get("name") == "_core"
                    and "cdylib" in msg.get("target", {}).get("crate_types", [])):
                artifacts.extend(Path(p) for p in msg["filenames"]
                                 if Path(p).suffix in {".dll", ".so", ".dylib"})
        if len(artifacts) != 1 or not artifacts[0].is_file():
            raise RuntimeError(f"Cargo did not produce exactly one native extension: {artifacts}")
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifacts[0], destination)


setup(
    distclass=BinaryDistribution,
    ext_modules=[Extension("calchem._core", sources=[], py_limited_api=True)],
    cmdclass={"build_ext": RustBuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
