"""
Unit tests for Pure Rust DFT-D3 / DFT-D3(BJ) Dispersion and PCM Solvation Gradients.
"""

import math
import numpy as np
import pytest

from calchem import _core as rust_core
from calchem.dft.d3 import compute_d3_energy, compute_d3_gradient


def test_rust_d3bj_dispersion():
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.22],
        [0.0, 1.43, -0.88],
        [0.0, -1.43, -0.88],
    ]
    res = rust_core.run_d3_dispersion(symbols, coords, functional="b3lyp", damping="bj")
    assert isinstance(res, rust_core.D3Result)
    assert res.energy_hartree < 0.0, f"D3 dispersion energy should be negative, got {res.energy_hartree}"
    assert res.energy_kcal_mol == res.energy_hartree * 627.509474

    grad = np.array(res.gradient)
    assert grad.shape == (3, 3)
    # Check translational invariance
    sum_g = np.sum(grad, axis=0)
    assert np.allclose(sum_g, 0.0, atol=1e-10), f"Translational invariance failed: {sum_g}"


def test_rust_d3zero_dispersion():
    symbols = ["C", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.19, 1.19, 1.19],
        [-1.19, -1.19, 1.19],
        [-1.19, 1.19, -1.19],
        [1.19, -1.19, -1.19],
    ]
    res = rust_core.run_d3_dispersion(symbols, coords, functional="pbe", damping="zero")
    assert res.energy_hartree < 0.0
    grad = np.array(res.gradient)
    assert grad.shape == (5, 3)
    sum_g = np.sum(grad, axis=0)
    assert np.allclose(sum_g, 0.0, atol=1e-10)


def test_python_d3_wrapper_integration():
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.22],
        [0.0, 1.43, -0.88],
        [0.0, -1.43, -0.88],
    ])
    e_d3 = compute_d3_energy(symbols, coords, functional="b3lyp", damping="bj")
    assert e_d3 < 0.0

    g_d3 = compute_d3_gradient(symbols, coords, functional="b3lyp", damping="bj")
    assert g_d3.shape == (3, 3)
    assert np.allclose(np.sum(g_d3, axis=0), 0.0, atol=1e-10)


def test_rust_pcm_solvation_gradient():
    symbols = ["O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.22],
        [0.0, 1.43, -0.88],
        [0.0, -1.43, -0.88],
    ]
    # Run gas SCF to get density
    wf = rust_core.run_single_point(
        symbols, coords,
        0, 1,
        "rhf", "sto-3g"
    )
    assert wf.converged
    density = np.array(wf.density).reshape((wf.n_ao, wf.n_ao))

    solv_grad = rust_core.run_solvation_gradient(
        symbols, coords,
        "sto-3g", density,
        solvent="water",
        _solvent_model="cpcm",
        charge=0,
        multiplicity=1,
    )
    grad_arr = np.array(solv_grad)
    assert grad_arr.shape == (3, 3)
    assert np.all(np.isfinite(grad_arr))
    # Forces should be physically bounded
    assert np.all(np.abs(grad_arr) < 1.0)


def test_rust_optimization_with_dispersion_and_solvent():
    symbols = ["H", "H"]
    # Compressed H2 at 1.0 Bohr (equilibrium is ~1.4 Bohr)
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # 1. Optimize with D3(BJ) dispersion
    res_d3 = rust_core.run_optimize(
        symbols, coords,
        basis="sto-3g",
        max_steps=20,
        charge=0,
        multiplicity=1,
        method="b3lyp",
        xc="b3lyp",
        dispersion="d3bj",
    )
    assert res_d3.converged
    # Final bond distance should relax towards equilibrium (~1.34-1.40 Bohr)
    opt_r = abs(res_d3.coords[5] - res_d3.coords[2])
    assert 1.25 < opt_r < 1.50, f"Optimized H2 bond length unexpected: {opt_r:.3f} Bohr"

    # 2. Optimize in water solvent
    res_solv = rust_core.run_optimize(
        symbols, coords,
        basis="sto-3g",
        max_steps=20,
        charge=0,
        multiplicity=1,
        method="rhf",
        solvent="water",
    )
    assert res_solv.converged
    opt_r_solv = abs(res_solv.coords[5] - res_solv.coords[2])
    assert 1.25 < opt_r_solv < 1.50, f"Solvent optimized H2 bond length unexpected: {opt_r_solv:.3f} Bohr"
