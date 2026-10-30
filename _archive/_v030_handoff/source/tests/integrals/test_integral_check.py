import sys
sys.setrecursionlimit(10000)
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
import math

from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear

print("手动计算 O 1s 轨道积分 (单个原语)", flush=True)
alpha = 130.7093214  # O 1s 第一个指数
Z = 8.0

# 归一化常数: N = (2*alpha/pi)^(3/4)
N = (2 * alpha / math.pi) ** 0.75
print(f"N = {N:.6f}", flush=True)
print(f"N^2 = {N*N:.6f}", flush=True)

# 动能: <T> = 3/2 * alpha  (对于归一化的 1s 高斯)
# 等等，让我推导一下:
# T = -1/2 ∇²
# <T> = -1/2 ∫ φ ∇² φ dr
# 对于 1s 高斯: ∇² exp(-αr²) = (4α²r² - 6α) exp(-αr²)
# <T> = -1/2 * N² * ∫ exp(-αr²) * (4α²r² - 6α) exp(-αr²) * 4πr² dr
# = -1/2 * N² * 4π * ∫_0^∞ (4α²r² - 6α) exp(-2αr²) r² dr
# 
# ∫_0^∞ r^4 exp(-βr²) dr = (3/8) * sqrt(π) / β^(5/2)
# ∫_0^∞ r^2 exp(-βr²) dr = (1/4) * sqrt(π) / β^(3/2)
# 
# 所以:
# <T> = -2π N² [4α² * (3/8) sqrt(π) / (2α)^(5/2) - 6α * (1/4) sqrt(π) / (2α)^(3/2)]
#     = -2π N² sqrt(π) [ (12α²/8) / (2α)^(5/2) - (6α/4) / (2α)^(3/2) ]
#     = -2π^(3/2) N² [ (3α²/2) / (2α)^(5/2) - (3α/2) / (2α)^(3/2) ]

beta = 2 * alpha
t_4 = 3.0 / 8.0 * math.sqrt(math.pi) / beta**2.5
t_2 = 1.0 / 4.0 * math.sqrt(math.pi) / beta**1.5
kinetic = -0.5 * N*N * 4*math.pi * (4*alpha*alpha * t_4 - 6*alpha * t_2)
print(f"\n手动计算动能 (单个原语): {kinetic:.6f}", flush=True)

# 核吸引势能: V = -Z * ∫ φ² * (1/r) dr
# V = -Z * N² * 4π * ∫_0^∞ r² * exp(-2αr²) * (1/r) dr
#   = -Z * N² * 4π * ∫_0^∞ r * exp(-2αr²) dr
#   = -Z * N² * 4π * (1 / (4α))
#   = -Z * N² * π / α
nuclear = -Z * N*N * math.pi / alpha
print(f"手动计算核吸引 (单个原语): {nuclear:.6f}", flush=True)
print(f"手动计算 Hcore (单个原语): {kinetic + nuclear:.6f}", flush=True)

print(f"\n现在看实际计算的积分:", flush=True)
mol = Molecule(["O"], [[0,0,0]], units="bohr")
basis = BasisSet("sto-3g")
S = compute_overlap(basis, mol)
T = compute_kinetic(basis, mol)
V = compute_nuclear(basis, mol)
print(f"S[0,0] = {S[0,0]:.6f}", flush=True)
print(f"T[0,0] = {T[0,0]:.6f}", flush=True)
print(f"V[0,0] = {V[0,0]:.6f}", flush=True)
print(f"H[0,0] = {T[0,0] + V[0,0]:.6f}", flush=True)

print(f"\n基组的壳层:", flush=True)
all_shells = basis.get_all_shells(mol)
for i, sh in enumerate(all_shells):
    print(f"  Shell {i}: {sh.shell_type}, exponents = {sh.exponents}")
    print(f"          coeffs = {sh._contract_coeffs}")
