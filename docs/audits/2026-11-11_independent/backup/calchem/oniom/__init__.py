"""
ONIOM (Our own N-layered Integrated molecular Orbital and Molecular mechanics)
分层计算方法。

ONIOM 将大分子分为多个层, 每一层使用不同精度的计算方法:
- Real (高): 高精度 QM 方法 (CCSD(T)/CBS)
- Model (中): 中等精度 QM 方法 (MP2/6-31G*)
- Submodel (低): 低精度方法 (HF/3-21G) 或 MM

总能量:
E_ONIOM = E_real(high) - E_model(high) + E_model(low)
       = E_high - E_high_model + E_low_model

适用场景:
- 酶催化机理研究
- 大分子局部反应
- 过渡金属配合物
- 溶剂化效应的分层处理

参考:
- Maseras & Morokuma, J. Comput. Chem. 16, 1170 (1995)
- Vreven & Morokuma, Theor. Chem. Acc. 109, 125 (2003)
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from ..mol.molecule import Molecule
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class Layer:
    """ONIOM 分层。"""

    def __init__(
        self,
        name: str,
        atoms: List[int],
        method: str,
        basis: Optional[str] = None,
        xc: str = "b3lyp",
        n_active: Optional[int] = None,
        n_active_elec: Optional[int] = None,
    ):
        self.name = name
        self.atoms = sorted(atoms)
        self.method = method.lower()
        self.basis = basis
        self.xc = xc
        self.n_active = n_active
        self.n_active_elec = n_active_elec

    def __repr__(self) -> str:
        return f"Layer({self.name}, {len(self.atoms)} atoms, {self.method}/{self.basis})"


class ONIOM:
    """
    ONIOM 分层计算。

    支持 2 层和 3 层 ONIOM。
    每一层使用不同的方法/基组组合, 对应不同精度的计算。

    Parameters
    ----------
    mol : Molecule
        完整分子
    layers : list of Layer
        分层定义, 从高精度到低精度

    Examples
    --------
    >>> mol = Molecule(['C', 'H', 'H', 'H', 'H'], coords)
    >>> layers = [
    >>>     Layer('high', atoms=[0, 1, 2], method='ccsd', basis='cc-pvdz'),
    >>>     Layer('low', atoms=[0, 1, 2, 3, 4], method='hf', basis='sto-3g'),
    >>> ]
    >>> oniom = ONIOM(mol, layers)
    >>> oniom.kernel()
    """

    def __init__(
        self,
        mol: Molecule,
        layers: List[Layer],
        charge: int = 0,
        multiplicity: int = 1,
        verbose: int = 1,
    ):
        self.mol = mol
        self.layers = layers
        self.charge = charge
        self.multiplicity = multiplicity
        self.verbose = verbose

        self.e_real_low: float = 0.0
        self.e_real_high: float = 0.0
        self.e_model_low: float = 0.0
        self.e_model_high: float = 0.0
        self.e_oniom: float = 0.0
        self.layer_energies: dict = {}

    def _print(self, msg: str, level: int = 1) -> None:
        if self.verbose >= level:
            print(msg, flush=True)

    def _get_submolecule(self, mol: Molecule, atom_indices: List[int]) -> Molecule:
        """提取子分子。"""
        symbols = [mol.symbols[i] for i in atom_indices]
        coords = mol.coordinates[atom_indices]
        return Molecule(
            symbols,
            coords,
            charge=self.charge,
            multiplicity=self.multiplicity,
            units=mol.units,
        )

    def _compute_layer_energy(self, layer: Layer) -> float:
        """计算单层能量。"""
        sub_mol = self._get_submolecule(self.mol, layer.atoms)

        from ..api.high_level import single_point

        if self.verbose >= 1:
            self._print(f"  计算层 {layer.name}: {len(layer.atoms)} 原子, "
                       f"{layer.method}/{layer.basis}", 1)

        result = single_point(
            sub_mol,
            method=layer.method,
            basis=layer.basis or "sto-3g",
            xc=layer.xc,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        return result.energy

    def kernel(self) -> float:
        """运行 ONIOM 计算。"""
        self._print("\n" + "=" * 60, 1)
        self._print("  ONIOM 分层计算", 1)
        self._print("=" * 60, 1)

        n_layers = len(self.layers)
        if n_layers < 2 or n_layers > 3:
            raise ValueError(f"ONIOM 支持 2-3 层, 当前 {n_layers} 层")

        for i, layer in enumerate(self.layers):
            self._print(f"\n  Layer {i+1}: {layer}", 1)

        if n_layers == 2:
            high = self.layers[0]
            low = self.layers[1]

            self.e_real_high = self._compute_layer_energy(high)
            self.e_model_high = self._compute_layer_energy(high)
            self.e_real_low = self._compute_layer_energy(low)
            self.e_model_low = self._compute_layer_energy(low)

            self.e_oniom = (
                self.e_real_low
                + self.e_model_high
                - self.e_real_high
            )
        elif n_layers == 3:
            high = self.layers[0]
            medium = self.layers[1]
            low = self.layers[2]

            self.e_real_low = self._compute_layer_energy(low)
            self.e_model_medium = self._compute_layer_energy(medium)
            self.e_model_low = self._compute_layer_energy(low)
            self.e_real_medium = self._compute_layer_energy(medium)
            self.e_model_high = self._compute_layer_energy(high)
            self.e_real_high = self._compute_layer_energy(high)

            self.e_oniom = (
                self.e_real_low
                + self.e_model_medium
                - self.e_model_low
                + self.e_model_high
                - self.e_model_medium
            )

        self._print(f"\n  ONIOM 总能量: {self.e_oniom:.10f} Eh", 1)
        self._print("=" * 60 + "\n", 1)

        return self.e_oniom
