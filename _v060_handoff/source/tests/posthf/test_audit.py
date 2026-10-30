"""CalChem 全模块底层审计测试脚本。"""

import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF, UHF, ROHF
from calchem.scf.dft import RKS, UKS, ROKS
from calchem.post_hf.casscf import CASSCF
from calchem.post_hf.mp4 import MP4
from calchem.post_hf.ccsd_t import CCSDT


def test_casscf_h2():
    """测试 H2 CASSCF(2,2)/sto-3g。"""
    print("\n" + "="*60)
    print("TEST: H2 CASSCF(2,2)/sto-3g")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    mol = Molecule(['H', 'H'], coords)
    
    try:
        casscf = CASSCF(mol, 'sto-3g', n_active=2, n_active_elec=2, verbose=2)
        e = casscf.kernel()
        print(f"CASSCF 能量: {e:.10f} Eh")
        print(f"预期值: ~-1.1336 Eh (Gaussian参考)")
    except Exception as ex:
        print(f"ERROR: {ex}")


def test_mp4_h2o():
    """测试 H2O MP4/sto-3g。"""
    print("\n" + "="*60)
    print("TEST: H2O MP4/sto-3g")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    mol = Molecule(['O', 'H', 'H'], coords)
    
    try:
        rhf = RHF(mol, 'sto-3g', verbose=0)
        rhf.kernel()
        print(f"HF 能量: {rhf.e_tot:.10f} Eh")
        
        mp4 = MP4(rhf, verbose=2)
        e = mp4.kernel()
        print(f"MP4 总能量: {e:.10f} Eh")
        print(f"MP2 相关能: {mp4.e_mp2:.10f} Eh")
        print(f"MP3 相关能: {mp4.e_mp3:.10f} Eh")
        print(f"MP4 相关能: {mp4.e_mp4:.10f} Eh")
    except Exception as ex:
        print(f"ERROR: {ex}")


def test_ccsd_t_h2o():
    """测试 H2O CCSD(T)/sto-3g。"""
    print("\n" + "="*60)
    print("TEST: H2O CCSD(T)/sto-3g")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    mol = Molecule(['O', 'H', 'H'], coords)
    
    try:
        ccsd_t = CCSDT(mol, 'sto-3g', verbose=2)
        e = ccsd_t.kernel()
        print(f"CCSD(T) 能量: {e:.10f} Eh")
        print(f"CCSD 相关能: {ccsd_t.e_corr_ccsd:.10f} Eh")
        print(f"(T) 修正: {ccsd_t.e_corr_pt:.10f} Eh")
    except Exception as ex:
        print(f"ERROR: {ex}")


def test_rohf_ch3():
    """测试 CH3 自由基 ROHF/sto-3g。"""
    print("\n" + "="*60)
    print("TEST: CH3 ROHF/sto-3g (multiplicity=2)")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629], 
              [-0.796, 0.398, 0.398], [0.398, -0.796, 0.398]]
    mol = Molecule(['C', 'H', 'H', 'H'], coords, multiplicity=2)
    
    try:
        rohf = ROHF(mol, 'sto-3g', verbose=2)
        e = rohf.kernel()
        print(f"ROHF 能量: {e:.10f} Eh")
    except Exception as ex:
        print(f"ERROR: {ex}")


def test_roks_o2():
    """测试 O2 ROKS/B3LYP/sto-3g。"""
    print("\n" + "="*60)
    print("TEST: O2 ROKS/B3LYP/sto-3g (multiplicity=3)")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2075]]
    mol = Molecule(['O', 'O'], coords, multiplicity=3)
    
    try:
        roks = ROKS(mol, basis='sto-3g', xc='b3lyp', grid_level='coarse', verbose=2)
        e = roks.kernel()
        print(f"ROKS 能量: {e:.10f} Eh")
    except Exception as ex:
        print(f"ERROR: {ex}")


def test_metagga_h2o():
    """测试 H2O TPSS/sto-3g (meta-GGA)。"""
    print("\n" + "="*60)
    print("TEST: H2O TPSS/sto-3g (meta-GGA)")
    print("="*60)
    
    coords = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
    mol = Molecule(['O', 'H', 'H'], coords)
    
    try:
        rks = RKS(mol, basis='sto-3g', xc='tpss', grid_level='coarse', verbose=2)
        e = rks.kernel()
        print(f"TPSS 能量: {e:.10f} Eh")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {ex}")


if __name__ == "__main__":
    test_casscf_h2()
    test_mp4_h2o()
    test_ccsd_t_h2o()
    test_rohf_ch3()
    test_roks_o2()
    test_metagga_h2o()
