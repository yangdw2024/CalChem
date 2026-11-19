import pytest
from calchem.backends.native import NativeCalchem
from calchem.core.request import CalculationRequest, MoleculeSpec

def test_cuda_detection_and_toggle():
    from calchem import _core
    assert hasattr(_core, "is_cuda_available")
    assert hasattr(_core, "set_gpu_enabled")
    assert hasattr(_core, "get_cuda_memory_info")

    avail = _core.is_cuda_available()
    print(f"CUDA available: {avail}")
    if avail:
        free, total = _core.get_cuda_memory_info()
        assert total > 0
        assert free <= total

    # Test toggling
    _core.set_gpu_enabled(False)
    assert not _core.is_cuda_available()
    _core.set_gpu_enabled(True)


def test_native_backend_with_density_fitting_and_gpu():
    backend = NativeCalchem()
    h2o = MoleculeSpec(
        symbols=["O", "H", "H"],
        coordinates_angstrom=[
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
        charge=0,
        multiplicity=1,
    )

    # 1. Calculation with density fitting (RI-J) on CPU
    req_df = CalculationRequest(
        task="single_point",
        backend="native",
        molecule=h2o,
        method="b3lyp",
        basis="6-31g(d)",
        options={"density_fitting": True, "use_gpu": False, "conv_energy": 1e-6, "conv_density": 1e-6},
    )
    res_df = backend.single_point(req_df)
    assert res_df.scf_converged
    assert res_df.total_energy_hartree is not None
    assert abs(res_df.total_energy_hartree - (-76.38)) < 0.2

    # 2. Calculation with GPU flag enabled (should either run on GPU or gracefully fall back to CPU)
    req_gpu = CalculationRequest(
        task="single_point",
        backend="native",
        molecule=h2o,
        method="b3lyp",
        basis="6-31g(d)",
        options={"density_fitting": True, "use_gpu": True, "conv_energy": 1e-6, "conv_density": 1e-6},
    )
    res_gpu = backend.single_point(req_gpu)
    assert res_gpu.scf_converged
    assert res_gpu.total_energy_hartree is not None
    # GPU and CPU density fitting results must match within convergence tolerance
    assert abs(res_gpu.total_energy_hartree - res_df.total_energy_hartree) < 1e-5


def test_rij_provenance_blocks_inconsistent_exact_coulomb_response():
    from calchem import _core
    wf = _core.run_single_point(["H", "H"], [[0., 0., 0.], [0., 0., 1.4]],
        density_fitting=True)
    assert wf.density_fitting
    with pytest.raises(ValueError, match="RI-J response kernel"):
        _core.run_tddft_from_scf_py(wf, n_states=1)


def test_rij_analytical_gradient_and_geometry_optimization():
    from calchem import _core
    grad = _core.run_gradient(
        [1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.50]],
        0, 1, "6-31g(d)", "rhf",
        density_fitting=True,
    )
    assert grad.shape == (2, 3)
    assert abs(float(grad[0, 2] + grad[1, 2])) < 1e-10
    assert float(grad[1, 2]) > 0.0  # stretched bond pulls inward

    opt = _core.run_optimize(
        [1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.50]],
        "6-31g(d)",
        30,
        charge=0,
        multiplicity=1,
        method="rhf",
        density_fitting=True,
    )
    assert opt.converged
    r_eq = abs(opt.coords[5] - opt.coords[2])
    assert 1.35 < r_eq < 1.42

