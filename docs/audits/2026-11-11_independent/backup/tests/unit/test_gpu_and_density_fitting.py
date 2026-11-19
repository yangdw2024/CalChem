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
