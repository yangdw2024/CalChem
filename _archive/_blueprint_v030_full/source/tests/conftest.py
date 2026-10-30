"""
CalChem 测试配置和夹具
"""
import pytest
from pathlib import Path

TEST_ROOT = Path(__file__).parent
FIXTURES_DIR = TEST_ROOT / "fixtures"
GAUSSIAN_TEST_DIR = Path(r"D:\G16W\tests\gjf")


@pytest.fixture
def water_mol():
    """水分子测试夹具"""
    from calchem.mol.molecule import Molecule
    return Molecule(
        symbols=["O", "H", "H"],
        coords=[[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9572],
                [0.0, 0.9572, 0.0]],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture
def methane_mol():
    """甲烷分子测试夹具"""
    from calchem.mol.molecule import Molecule
    return Molecule(
        symbols=["C", "H", "H", "H", "H"],
        coords=[[0.0, 0.0, 0.0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629],
                [0.629, -0.629, -0.629]],
        charge=0,
        multiplicity=1,
    )


def pytest_collection_modifyitems(config, items):
    """标记需要 GPU 的测试"""
    for item in items:
        if "gpu" in item.keywords:
            try:
                import cupy
                cupy.cuda.Device(0).compute_capability
            except (ImportError, Exception):
                item.add_marker(pytest.mark.skip(reason="GPU/CuPy not available"))
