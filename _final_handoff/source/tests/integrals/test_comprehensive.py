"""
CalChem 双电子积分与 SCF 综合验证测试。

验证项目:
1. H2 (s 轨道) - Szabo 参考值
2. He 原子 (s 轨道)
3. Ne 原子 (s + p 轨道)
4. H2O (s + p 轨道, 多原子)
5. ERI 对称性
6. ERI 球对称性
"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.scf.hf import RHF
from calchem.integrals.two_electron import _ERIRecursionFixed, compute_eri
from calchem.basis.basis_set import BasisSet

print("=" * 70)
print("CalChem 双电子积分与 SCF 综合验证")
print("=" * 70)

# ===== 1. ERI 对称性验证 =====
print("\n1. ERI 对称性验证 (H2O)")
h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""").to_bohr()

basis = BasisSet("sto-3g")
eri = compute_eri(basis, h2o)
nbf = eri.shape[0]

max_diff = 0.0
for i in range(nbf):
    for j in range(nbf):
        for k in range(nbf):
            for l in range(nbf):
                v1 = eri[i, j, k, l]
                v2 = eri[j, i, k, l]
                v3 = eri[i, j, l, k]
                v4 = eri[k, l, i, j]
                d = max(abs(v1 - v2), abs(v1 - v3), abs(v1 - v4))
                if d > max_diff:
                    max_diff = d
print(f"   ERI 对称性 (ij|kl)=(ji|kl)=(ij|lk)=(kl|ij) 最大差值: {max_diff:.2e}")
print(f"   {'✓ 通过' if max_diff < 1e-12 else '✗ 失败'}")

# ===== 2. ERI 球对称性验证 =====
print("\n2. ERI 球对称性验证 (同中心 p 轨道)")
alpha = 1.0
origin = np.zeros(3)
rec = _ERIRecursionFixed(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

val_xx = rec.get(1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0)
val_yy = rec.get(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0)
val_zz = rec.get(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1)
print(f"   (px px|px px) = {val_xx:.10f}")
print(f"   (py py|py py) = {val_yy:.10f}")
print(f"   (pz pz|pz pz) = {val_zz:.10f}")
sph_ok = abs(val_xx - val_yy) < 1e-12 and abs(val_yy - val_zz) < 1e-12
print(f"   球对称 {'✓ 通过' if sph_ok else '✗ 失败'}")

# ===== 3. H2 STO-3G =====
print("\n3. H2 STO-3G RHF")
h2 = Molecule(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], units="bohr")
rhf = RHF(h2, basis="sto-3g", verbose=0)
e_h2 = rhf.kernel()
ref_h2 = -1.11671433
print(f"   能量: {e_h2:.8f} Eh")
print(f"   参考: {ref_h2:.8f} Eh (Szabo & Ostlund)")
print(f"   差值: {e_h2 - ref_h2:+.2e} Eh  {'✓' if abs(e_h2 - ref_h2) < 1e-5 else '✗'}")

# ===== 4. He STO-3G =====
print("\n4. He STO-3G RHF")
he = Molecule(["He"], [[0.0, 0.0, 0.0]], units="bohr")
rhf_he = RHF(he, basis="sto-3g", verbose=0)
e_he = rhf_he.kernel()
print(f"   能量: {e_he:.8f} Eh")
# He STO-3G 参考值约 -2.8077758 Eh (Szabo 练习)
print(f"   参考: -2.80778 Eh (Szabo 近似)")

# ===== 5. Ne STO-3G =====
print("\n5. Ne STO-3G RHF")
ne = Molecule(["Ne"], [[0.0, 0.0, 0.0]], units="bohr")
rhf_ne = RHF(ne, basis="sto-3g", verbose=0)
e_ne = rhf_ne.kernel()
print(f"   能量: {e_ne:.8f} Eh")
print(f"   参考: -126.60452917 Eh (标准 STO-3G 值)")
print(f"   差值: {e_ne - (-126.60452917):+.2e} Eh  {'✓' if abs(e_ne - (-126.60452917)) < 1e-5 else '✗'}")

# ===== 6. H2O STO-3G =====
print("\n6. H2O STO-3G RHF")
h2o = Molecule.from_xyz("""
3
H2O
O  0.000000  0.000000  0.119263
H  0.000000  0.763239 -0.477052
H  0.000000 -0.763239 -0.477052
""")
rhf_h2o = RHF(h2o, basis="sto-3g", verbose=0)
e_h2o = rhf_h2o.kernel()
print(f"   能量: {e_h2o:.8f} Eh")
print(f"   参考: -74.963212 Eh (STO-3G 标准几何)")
print(f"   差值: {e_h2o - (-74.963212):+.2e} Eh")

# ===== 总结 =====
print("\n" + "=" * 70)
print("验证总结:")
print(f"  H2  : {e_h2:.8f} Eh  (参考 {ref_h2:.8f}, 差 {abs(e_h2-ref_h2):.1e})")
print(f"  He  : {e_he:.8f} Eh")
print(f"  Ne  : {e_ne:.8f} Eh  (收敛 {rhf_ne.n_cycles} 步)")
print(f"  H2O : {e_h2o:.8f} Eh  (收敛 {rhf_h2o.n_cycles} 步)")
print(f"  ERI 对称性: {max_diff:.1e}  {'✓' if max_diff < 1e-12 else '✗'}")
print(f"  ERI 球对称: {'✓' if sph_ok else '✗'}")
print("=" * 70)
