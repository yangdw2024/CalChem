from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class SCFResult:
    """
    SCF 计算结果的结构化数据类。

    包含所有重要的 SCF 结果信息，便于后续分析和输出。
    """

    energy: float = 0.0
    converged: bool = False
    n_cycles: int = 0
    method: str = ""
    basis: str = ""

    e_1e: float = 0.0
    e_2e: float = 0.0
    e_nn: float = 0.0

    mo_energies: Optional[np.ndarray] = None
    mo_occupations: Optional[np.ndarray] = None
    mo_coefficients: Optional[np.ndarray] = None

    dipole_moment: Optional[np.ndarray] = None
    charges: Optional[np.ndarray] = None

    dm: Optional[Union[np.ndarray, tuple]] = None
    fock: Optional[Union[np.ndarray, tuple]] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"SCFResult(method={self.method}, basis={self.basis}, "
            f"E={self.energy:.8f} Eh, {status}, cycles={self.n_cycles})"
        )

    def summary(self) -> str:
        """生成结果摘要文本。"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {self.method}/{self.basis} SCF 结果")
        lines.append("=" * 60)
        lines.append(f"  总能量:          {self.energy:15.10f} Eh")
        lines.append(f"  单电子能量:      {self.e_1e:15.10f} Eh")
        lines.append(f"  双电子能量:      {self.e_2e:15.10f} Eh")
        lines.append(f"  核排斥能:        {self.e_nn:15.10f} Eh")
        lines.append(f"  收敛状态:        {'收敛 ✓' if self.converged else '未收敛 ✗'}")
        lines.append(f"  迭代次数:        {self.n_cycles}")

        if self.mo_energies is not None:
            lines.append("")
            lines.append("  分子轨道能量 (Eh):")
            if isinstance(self.mo_energies, tuple):
                lines.append("    Alpha:")
                for i, e in enumerate(self.mo_energies[0]):
                    occ = "occ" if self.mo_occupations is not None and i < len(self.mo_occupations[0]) and self.mo_occupations[0][i] > 0 else "vir"
                    lines.append(f"      {i+1:3d}. {e:12.6f}  {occ}")
                lines.append("    Beta:")
                for i, e in enumerate(self.mo_energies[1]):
                    occ = "occ" if self.mo_occupations is not None and i < len(self.mo_occupations[1]) and self.mo_occupations[1][i] > 0 else "vir"
                    lines.append(f"      {i+1:3d}. {e:12.6f}  {occ}")
            else:
                for i, e in enumerate(self.mo_energies):
                    occ = "occ" if self.mo_occupations is not None and self.mo_occupations[i] > 0 else "vir"
                    lines.append(f"    {i+1:3d}. {e:12.6f}  {occ}")

        if self.dipole_moment is not None:
            lines.append("")
            lines.append(f"  偶极矩 (a.u.):   [{self.dipole_moment[0]:.6f}, {self.dipole_moment[1]:.6f}, {self.dipole_moment[2]:.6f}]")
            lines.append(f"  |μ|:             {np.linalg.norm(self.dipole_moment):.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, tuple):
                d[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        return d
