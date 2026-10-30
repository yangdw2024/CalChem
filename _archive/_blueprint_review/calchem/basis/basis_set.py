"""
基组容器模块。

BasisSet 类负责：
- 从内置基组数据库加载基组
- 为分子构建基函数（每个原子上放置壳层）
- 管理笛卡尔/球谐函数形式
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from .gto import GTOShell
from .basis_data import get_basis_data, list_available_basis


class BasisSet:
    """
    基组容器类。

    负责管理基组数据，并为分子构建具体的基函数壳层。
    """

    def __init__(self, name: str, spherical: bool = False):
        """
        初始化基组。

        Parameters
        ----------
        name : str
            基组名称（不区分大小写）
        spherical : bool
            是否使用球谐函数形式，默认 False（笛卡尔函数）
        """
        self.name = name.lower()
        self.spherical = spherical

        # 从内置数据库加载基组数据
        self._basis_data = get_basis_data(self.name)

    def has_element(self, element: str) -> bool:
        """检查基组是否包含指定元素的数据。"""
        return element.capitalize() in self._basis_data

    def get_element_shells(self, element: str) -> List[dict]:
        """获取指定元素的壳层数据（原始字典形式）。"""
        element = element.capitalize()
        if not self.has_element(element):
            raise ValueError(f"基组 {self.name} 不包含元素 {element}")
        return self._basis_data[element]

    def build_molecule_basis(
        self,
        mol,
    ) -> Tuple[List[List[GTOShell]], int]:
        """
        为分子构建基组，返回每个原子的壳层列表和总基函数数。

        Parameters
        ----------
        mol : Molecule
            分子对象

        Returns
        -------
        shells_by_atom : List[List[GTOShell]]
            每个原子的壳层列表，shells_by_atom[i] 是第 i 个原子的壳层
        nbf : int
            总基函数数目
        """
        shells_by_atom = []
        nbf = 0

        for i in range(mol.natom):
            symbol = mol.symbols[i]
            origin = mol.coordinates[i]

            # 获取该元素的壳层数据
            shell_data = self.get_element_shells(symbol)

            # 创建 GTOShell 对象
            atom_shells = []
            for sh_data in shell_data:
                shell = GTOShell(
                    shell_type=sh_data['shell_type'],
                    exponents=sh_data['exponents'],
                    coefficients=sh_data['coefficients'],
                    origin=origin,
                )
                atom_shells.append(shell)

                if self.spherical:
                    nbf += shell.n_functions_spherical
                else:
                    nbf += shell.n_functions

            shells_by_atom.append(atom_shells)

        return shells_by_atom, nbf

    def get_all_shells(self, mol) -> List[GTOShell]:
        """获取分子的所有壳层（扁平列表）。"""
        shells_by_atom, _ = self.build_molecule_basis(mol)
        all_shells = []
        for atom_shells in shells_by_atom:
            all_shells.extend(atom_shells)
        return all_shells

    def shell_index_to_bf_range(self, all_shells: List[GTOShell], shell_idx: int) -> Tuple[int, int]:
        """
        获取第 shell_idx 个壳层对应的基函数索引范围 [start, end)。
        """
        start = 0
        for i in range(shell_idx):
            if self.spherical:
                start += all_shells[i].n_functions_spherical
            else:
                start += all_shells[i].n_functions
        if self.spherical:
            end = start + all_shells[shell_idx].n_functions_spherical
        else:
            end = start + all_shells[shell_idx].n_functions
        return start, end

    @property
    def elements(self) -> List[str]:
        """返回基组包含的元素列表。"""
        return sorted(self._basis_data.keys())

    def __repr__(self) -> str:
        n_elem = len(self._basis_data)
        return f"BasisSet(name='{self.name}', elements={n_elem}, spherical={self.spherical})"

    def __str__(self) -> str:
        return self.__repr__()
