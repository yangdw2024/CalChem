"""Rust 核心扩展（calchem._core）回归测试。

覆盖：
- Stage 2 已完成能力：四分子 RHF/STO-3G 能量基线、HOMO/LUMO 轨道能。
- Stage 3 规划中能力（标记为 xfail）：偶极矩、解析梯度、几何优化、零拷贝网格。
"""
import numpy as np
import pytest

calchem_core = pytest.importorskip("calchem._core")

_Z_TO_SYMBOL = {1: "H", 2: "He", 8: "O", 10: "Ne"}


def _run(z, c, basis="sto-3g"):
    symbols = [_Z_TO_SYMBOL.get(x, str(x)) if isinstance(x, (int, np.integer)) else str(x) for x in z]
    coords = [[float(v) for v in row] for row in c]
    return calchem_core.run_single_point(
        symbols,
        coords,
        0, 1, "rhf", basis,
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
    assert w.homo_energy_hartree is not None
    assert w.lumo_energy_hartree is not None
    assert abs(w.homo_energy_hartree - (-0.5782029775)) < 1e-6
    assert abs(w.lumo_energy_hartree - 0.6702677683) < 1e-6


def test_h2o_dipole():
    w = _run([8, 1, 1], [[0, 0, 0], [0, 1.432, 0], [1.112, -0.716, 0]])
    d = getattr(w, "dipole_debye", None)
    assert d is not None
    assert abs(d[0] - 1.48296605) < 1e-5
    assert abs(d[1] - 0.83461342) < 1e-5
    assert abs(d[2]) < 1e-6
    assert abs(w.dipole_magnitude_debye - 1.7016956) < 1e-5


def test_h2_gradient():
    run_grad = getattr(calchem_core, "run_gradient", None)
    assert run_grad is not None, "run_gradient is not implemented in calchem._core yet"
    g = run_grad(
        np.array([1, 1], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 0, 1.4]], dtype=np.float64),
        0, 1, "sto-3g",
    )
    assert abs(g[0, 2] - (-0.02845406)) < 1e-6
    assert abs(g[1, 2] - 0.02845406) < 1e-6
    assert np.allclose(g[0], -g[1])


def test_h2_optimization():
    run_opt = getattr(calchem_core, "run_optimize", None)
    assert run_opt is not None, "run_optimize is not implemented in calchem._core yet"
    o = run_opt(
        np.array([1, 1], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 0, 1.4]], dtype=np.float64),
        "sto-3g", 100,
    )
    assert o.converged
    c = np.array(o.coords).reshape(2, 3)
    bond = np.linalg.norm(c[1] - c[0])
    assert abs(bond - 1.346138) < 1e-3
    assert abs(o.energy - (-1.1175058849)) < 1e-8


def test_orbital_cube_zero_copy():
    eval_cube = getattr(calchem_core, "eval_orbital_cube_zero_copy", None)
    assert eval_cube is not None, "eval_orbital_cube_zero_copy is not implemented in calchem._core yet"
    w = _run([1, 1], [[0, 0, 0], [0, 0, 1.4]])
    cube = eval_cube(w, 0, [-2, -2, -2], [0.4, 0.4, 0.4], [11, 11, 14])
    assert cube.shape == (14, 11, 11)
    assert cube.dtype == np.float32
    assert cube.max() > 0.3


def test_uhf_open_shell_single_point_gradient_and_optimize():
    # 1. H atom (1e doublet): exact STO-3G energy -0.466581850 Eh, <S^2> = 0.75
    w_h = calchem_core.run_single_point(["H"], [[0.0, 0.0, 0.0]], 0, 2, "uhf", "sto-3g")
    assert w_h.converged
    assert w_h.is_uhf is True
    assert (w_h.n_alpha, w_h.n_beta) == (1, 0)
    assert abs(w_h.total_energy_hartree - (-0.466581850)) < 1e-8
    assert abs(w_h.s_squared - 0.75) < 1e-12

    # 2. H2O+ radical cation (9e doublet): UHF gradient + alpha/beta orbital cubes
    coords = [[0.0, 0.0, 0.0], [0.0, 1.432, 0.0], [1.112, -0.716, 0.0]]
    w_cat = calchem_core.run_single_point(["O", "H", "H"], coords, 1, 2, "uhf", "sto-3g")
    assert w_cat.converged
    assert w_cat.is_uhf is True
    assert (w_cat.n_alpha, w_cat.n_beta) == (5, 4)
    assert 0.75 - 1e-10 <= w_cat.s_squared < 0.80
    cube_alpha = calchem_core.eval_orbital_cube_zero_copy(w_cat, 4, [-1, -1, -1], [0.5, 0.5, 0.5], [5, 5, 5])
    cube_beta = calchem_core.eval_orbital_cube_zero_copy(w_cat, w_cat.n_ao + 3, [-1, -1, -1], [0.5, 0.5, 0.5], [5, 5, 5])
    assert cube_alpha.shape == (5, 5, 5) and cube_beta.shape == (5, 5, 5)

    g_cat = calchem_core.run_gradient(["O", "H", "H"], coords, 1, 2, "sto-3g", "uhf")
    assert g_cat.shape == (3, 3)
    assert np.allclose(g_cat.sum(axis=0), 0.0, atol=1e-8)

    # 3. H2+ radical cation optimization: bond lengthens vs neutral H2 (1.346 Bohr)
    opt_h2p = calchem_core.run_optimize(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], "sto-3g", 100, 1, 2, "uhf")
    assert opt_h2p.converged
    c_opt = np.array(opt_h2p.coords).reshape(2, 3)
    r_eq = np.linalg.norm(c_opt[1] - c_opt[0])
    assert r_eq > 1.8

