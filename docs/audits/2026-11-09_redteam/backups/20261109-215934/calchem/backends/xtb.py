"""xTB 后端（可选）。需要 xtb 可执行文件。"""

from __future__ import annotations

import shutil

from ..core.calculator import register_backend
from ..core.capabilities import BackendCapabilities
from .external import ExternalProgramBackend


@register_backend("xtb")
class XTBackend(ExternalProgramBackend):
    """xTB 半经验后端。"""

    program_name = "xtb"

    def __init__(self, executable: str = ""):
        if not executable:
            executable = shutil.which("xtb") or ""
        super().__init__(executable)

    def capabilities(self):
        caps = super().capabilities()
        caps.tasks = ["single_point", "optimization"]
        caps.methods = ["gfn1", "gfn2", "gfn-ff"]
        caps.basis_sets = ["gfn2"]
        return caps

    def single_point(self, request):
        raise NotImplementedError("xTB adapter: locate xtb binary and implement input writer")
