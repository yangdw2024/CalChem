"""频率计算：数值 Hessian → 投影 → 振动频率。"""

from __future__ import annotations

import numpy as np

from ..core.request import MoleculeSpec
from ..core.periodic_table import get_atomic_mass


def numerical_hessian(
    gradient_fn,
    mol_spec: MoleculeSpec,
    step_bohr: float = 0.005,
) -> np.ndarray:
    """中心差分计算 Hessian（能量二阶导矩阵）。

    Args:
        gradient_fn: 输入 MoleculeSpec 返回梯度 (natom, 3) Eh/Bohr
        mol_spec: 分子
        step_bohr: 步长

    Returns:
        Hessian: shape (3*natom, 3*natom) Eh/Bohr^2
    """
    natom = len(mol_spec.symbols)
    ndim = 3 * natom
    hessian = np.zeros((ndim, ndim))
    b2a = 0.529177210903
    coords_bohr = np.array(mol_spec.coordinates_angstrom) / b2a

    for i in range(ndim):
        atom_i = i // 3
        comp_i = i % 3

        coords_plus = coords_bohr.copy()
        coords_plus[atom_i, comp_i] += step_bohr
        spec_plus = MoleculeSpec(
            symbols=mol_spec.symbols,
            coordinates_angstrom=coords_plus * b2a,
            charge=mol_spec.charge, multiplicity=mol_spec.multiplicity,
        )
        g_plus = gradient_fn(spec_plus).flatten()

        coords_minus = coords_bohr.copy()
        coords_minus[atom_i, comp_i] -= step_bohr
        spec_minus = MoleculeSpec(
            symbols=mol_spec.symbols,
            coordinates_angstrom=coords_minus * b2a,
            charge=mol_spec.charge, multiplicity=mol_spec.multiplicity,
        )
        g_minus = gradient_fn(spec_minus).flatten()

        hessian[i, :] = (g_plus - g_minus) / (2 * step_bohr)

    # 对称化
    hessian = 0.5 * (hessian + hessian.T)
    return hessian


def project_vibrational_modes(
    hessian: np.ndarray,
    mol_spec: MoleculeSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """投影平移/转动自由度并对角化质量加权 Hessian。

    Returns:
        frequencies_cm: 振动频率 (cm^-1)，虚频为负
        normal_modes: 归一化简正模式
    """
    natom = len(mol_spec.symbols)
    ndim = 3 * natom
    b2a = 0.529177210903

    # 质量对角矩阵
    masses = np.array([get_atomic_mass(s) for s in mol_spec.symbols])
    mass_matrix = np.repeat(masses, 3)  # (ndim,)
    sqrt_mass = np.sqrt(mass_matrix)

    # 质量加权 Hessian
    mw_hessian = hessian / sqrt_mass[:, None] / sqrt_mass[None, :]

    # 构建平移/转动投影矩阵
    coords_bohr = np.array(mol_spec.coordinates_angstrom) / b2a
    # 质心
    com = np.sum(coords_bohr * masses[:, None], axis=0) / np.sum(masses)
    coords_rel = coords_bohr - com

    # 6 个外部自由度（线性分子只有 5）
    # 判断是否线性
    linear = False
    if natom <= 2:
        linear = True
    else:
        # 检查三个原子是否共线
        v1 = coords_rel[1] - coords_rel[0]
        v2 = coords_rel[2] - coords_rel[0]
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < 1e-6:
            linear = True

    n_ext = 5 if linear else 6

    # 平移向量
    trans = np.zeros((3, ndim))
    for d in range(3):
        for a in range(natom):
            trans[d, 3*a + d] = 1.0

    # 转动向量（质心）
    rot = np.zeros((3, ndim))
    for d in range(3):
        for a in range(natom):
            rot[d, 3*a:(a+1)*3] = _cross_matrix(coords_rel[a])[d]

    external = np.vstack([trans, rot])[:n_ext]

    # 正交化外部向量
    external = np.linalg.qr(external.T)[0].T  # shape (n_ext, ndim)

    # 投影矩阵 P = I - Q(Q^T Q)^-1 Q^T
    # 由于 Q 已经是正交的，P = I - Q^T Q
    # 我们用 Q^T H Q 在内部子空间
    # 先把 Hessian 投影到内部子空间
    P = np.eye(ndim) - external.T @ external
    mw_proj = P @ mw_hessian @ P

    # 对角化
    eigvals, eigvecs = np.linalg.eigh(mw_proj)

    # 转换频率：omega = sqrt(lambda)，Eh/(Bohr^2 * amu) → cm^-1
    # 1 Eh = 27.2114 eV, 1 Bohr = 0.529177 Å, 1 amu = 1.6605e-27 kg
    # conversion factor: sqrt(1 Eh / (1 Bohr^2 * 1 amu)) in cm^-1
    # = sqrt(4.3598e-18 J / (2.8003e-21 m^2 * 1.6605e-27 kg)) / (2*pi*c) * 100
    # ≈ 5140.48 cm^-1 (common quantum chemistry constant)
    conv = 5140.4866

    # 只保留正特征值（投影后外部模式应该接近 0）
    frequencies = []
    modes = []
    for i in range(ndim):
        lam = eigvals[i]
        if lam < -1e-6:
            freq = -conv * np.sqrt(-lam)  # 虚频
            frequencies.append(freq)
            modes.append(eigvecs[:, i])
        elif lam > 1e-6:
            freq = conv * np.sqrt(lam)
            frequencies.append(freq)
            modes.append(eigvecs[:, i])
        # else: 近零，跳过（外部模式）

    return np.array(frequencies), np.array(modes)


def _cross_matrix(v: np.ndarray) -> np.ndarray:
    """返回向量 v 的叉乘矩阵。"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
