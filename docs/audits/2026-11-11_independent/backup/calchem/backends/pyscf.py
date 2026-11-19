"""PySCF 后端（可选 extra）。未安装 pyscf 时注册为 unavailable。"""

from __future__ import annotations

from ..core.calculator import register_backend
from ..core.capabilities import BackendCapabilities
from .external import ExternalProgramBackend


@register_backend("pyscf")
class PySCFBackend(ExternalProgramBackend):
    """PySCF 后端。需要 pip install pyscf。"""

    program_name = "pyscf"

    def __init__(self):
        super().__init__("")

    def capabilities(self):
        try:
            import pyscf  # noqa: F401
            available = True
            version = pyscf.__version__
        except ImportError:
            available = False
            version = None
        return BackendCapabilities(
            name="pyscf", version=version, available=available,
            requires_external_program=False,
            unavailable_reason=None if available else "pyscf not installed (pip install pyscf)",
            tasks=["single_point", "optimization"],
            methods=["rhf", "uhf", "rks", "uks", "mp2", "ccsd"],
            basis_sets=["sto-3g", "6-31g", "6-31g(d)", "def2-svp", "def2-tzvp"],
        )

    def single_point(self, request):
        raise NotImplementedError("PySCF adapter: install pyscf and implement binding")
