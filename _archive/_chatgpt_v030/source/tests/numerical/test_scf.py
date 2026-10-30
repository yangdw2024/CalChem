import pytest
from calchem.mol.molecule import Molecule


def test_hf_water_sto3g():
    pytest.importorskip("pyscf")

    from calchem.scf.hf import RHF

    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.9572],
                           [0.0, 0.9572, 0.0]],
                   charge=0, multiplicity=1)

    calc = RHF(mol, basis="sto-3g")
    result = calc.run()

    assert result.converged
    assert result.energy < -70.0  # 水分子 STO-3G 能量约 -75 Hartree
    assert result.energy > -80.0


def test_dft_water_b3lyp():
    pytest.importorskip("pyscf")

    from calchem.scf.dft import DFT

    mol = Molecule(symbols=["O", "H", "H"],
                   coords=[[0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.9572],
                           [0.0, 0.9572, 0.0]],
                   charge=0, multiplicity=1)

    calc = DFT(mol, xc="b3lyp", basis="sto-3g")
    result = calc.run()

    assert result.converged
    assert result.energy < -70.0
    assert result.energy > -80.0
