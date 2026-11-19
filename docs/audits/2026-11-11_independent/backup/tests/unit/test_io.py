import pytest
from pathlib import Path

from calchem.io.input import read_xyz, read_gjf


TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture
def water_xyz():
    return """3
water molecule
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.957200
H  0.000000  0.957200  0.000000
"""


def test_read_xyz_string(water_xyz):
    mol = read_xyz(water_xyz)
    assert mol.natoms == 3
    assert mol.symbols == ["O", "H", "H"]
    assert abs(mol.bond_length(0, 1) - 0.9572) < 1e-6


def test_read_gjf_simple():
    gjf = """#P B3LYP/6-31G* Opt

Water optimization

0 1
O
H 1 R
H 1 R 2 A

R = 0.96
A = 104.5
"""
    mol = read_gjf(gjf)
    assert mol.natoms == 3
    assert mol.charge == 0
    assert mol.multiplicity == 1


def test_read_gjf_route_section():
    from calchem.io.input import parse_gjf_route
    route = "#P B3LYP/6-31G* Opt Freq Test"
    info = parse_gjf_route(route)
    assert info["method"] == "B3LYP"
    assert info["basis"] == "6-31G*"
    assert "OPT" in info["jobs"]
    assert "FREQ" in info["jobs"]
