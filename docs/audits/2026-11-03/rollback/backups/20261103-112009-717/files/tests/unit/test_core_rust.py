"""Rust 核心扩展（calchem._core）回归测试。

覆盖：四分子 RHF/STO-3G 能量基线、解析梯度、几何优化、零拷贝网格。
所有数值均由真实积分与 SCF 迭代得到，无任何硬编码。
"""
import numpy as np
import pytest

calchem_core = pytest.importorskip("calchem._core")


def _run(z, c):
    return calchem_core.run_single_point(
        np.array(z, dtype=np.uint8),
        np.array(c, dtype=np.float64),
        0, 1, "rhf", "sto-3g",
    )


@pytest.mark.parametrize(
    "name,z,coords,ref,tol",
    [
        ("H2", [1, 1], [[0, 0, 0], [0, 0, 1.4]], -1.1167143251, 1e-8),
        ("He", [2], [[0, 0, 0]], -2.8077839575, 1e-6),
        ("Ne", [10], [[0, 0, 0]], -126.6045291715, 1e-6),
        ("H2O", [8, 1, 1],
         [[0, 0, 0], [0, 1.432, 0], [1.112, -0.716, 0]],
         -74.695319114, 1e-7),
    ],
)
def test_energy_baseline(name, z, coords, ref, tol):
    w = _run(z, coords)
    assert w.converged
    assert abs(w.total_energy_hartree - ref) < tol, (
        f"{name}: {w.total_energy_hartree} vs {ref}"
    )


def test_h2_orbital_energies():
    w = _run([1, 1], [[0, 0, 0], [0, 0, 1.4]])
    assert abs(w.homo_energy_hartree - (-0.5782029775)) < 1e-6
    assert abs(w.lumo_energy_hartree - 0.6702677683) < 1e-6


def test_h2o_dipole():
    w = _run([8, 1, 1], [[0, 0, 0], [0, 1.432, 0], [1.112, -0.716, 0]])
    d = w.dipole_debye
    assert abs(d[0] - 1.48296605) < 1e-5
    assert abs(d[1] - 0.83461342) < 1e-5
    assert abs(d[2]) < 1e-6
    assert abs(w.dipole_magnitude_debye - 1.7016956) < 1e-5


def test_h2_gradient():
    g = calchem_core.run_gradient(
        np.array([1, 1], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 0, 1.4]], dtype=np.float64),
        0, 1, "sto-3g",
    )
    # 两原子力大小相等、方向相反；H2 沿 z 时合力梯度 ±0.028454。
    assert abs(g[0, 2] - (-0.02845406)) < 1e-6
    assert abs(g[1, 2] - 0.02845406) < 1e-6
    assert np.allclose(g[0], -g[1])


def test_h2_optimization():
    o = calchem_core.run_optimize(
        np.array([1, 1], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 0, 1.4]], dtype=np.float64),
        "sto-3g", 100,
    )
    assert o.converged
    # 平衡键长（两原子都可移动，故键长为坐标差）。
    c = np.array(o.coords).reshape(2, 3)
    bond = np.linalg.norm(c[1] - c[0])
    assert abs(bond - 1.346138) < 1e-3
    assert abs(o.energy - (-1.1175058714)) < 1e-8


def test_orbital_cube_zero_copy():
    w = _run([1, 1], [[0, 0, 0], [0, 0, 1.4]])
    cube = calchem_core.eval_orbital_cube_zero_copy(
        w, 0, [-2, -2, -2], [0.4, 0.4, 0.4], [11, 11, 14]
    )
    assert cube.shape == (14, 11, 11)
    assert cube.dtype == np.float32
    # 成键轨道在分子上为正且有显著幅值。
    assert cube.max() > 0.3
