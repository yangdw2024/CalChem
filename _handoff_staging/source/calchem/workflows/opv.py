"""CalChem OPV（有机光伏）工作流。

为 OSC/OPV 研究提供标准化计算流水线：
1. 分子准备（电荷/多重度/几何）
2. DFT 优化（B3LYP/6-31G(d) 或 ωB97XD/def2-SVP）
3. 单点能（更高级方法/基组）
4. HOMO/LUMO 提取
5. 重组能计算（hole/electron）
6. 输出 OPV 关键参数

这是 dogfooding 工作流，不是产品边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from calchem.api.service import TaskService
from calchem.core.request import CalculationRequest, MoleculeSpec, ResourceLimits


@dataclass
class OPVMolecule:
    """OPV 分子（给体或受体）。"""
    name: str
    smiles_or_xyz: str
    charge: int = 0
    multiplicity: int = 1


@dataclass
class OPVWorkflowResult:
    """OPV 工作流结果。"""
    name: str
    hf_energy: Optional[float] = None
    scf_energy: Optional[float] = None
    homo_energy: Optional[float] = None
    lumo_energy: Optional[float] = None
    gap_ev: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class OPVWorkflow:
    """OPV 标准计算流水线。"""

    def __init__(self, service: TaskService = None, backend: str = "native"):
        self.service = service or TaskService()
        self.backend = backend

    def single_point(self, mol: MoleculeSpec, method: str = "rhf",
                     basis: str = "sto-3g") -> OPVWorkflowResult:
        """单点能计算。"""
        req = CalculationRequest(
            task="single_point", molecule=mol, backend=self.backend,
            method=method, basis=basis,
            resources=ResourceLimits(max_cpu_cores=8, max_memory_mb=4000, timeout_seconds=300),
        )
        result = self.service.run(req)
        return OPVWorkflowResult(
            name=f"{method}/{basis}",
            scf_energy=result.scf_energy_hartree,
            homo_energy=result.homo_energy_hartree if result.orbitals else None,
            lumo_energy=result.lumo_energy_hartree if result.orbitals else None,
            errors=result.errors,
        )
