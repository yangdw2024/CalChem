"""
调试 Ne 原子的 SCF，看看问题出在哪里。
Ne 原子是单中心的，所有积分都是同中心的。
"""
import numpy as np
from calchem.mol.molecule import Molecule
from calchem.basis.basis_set import BasisSet
from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
from calchem.integrals.two_electron import compute_eri, shell_eri
from calchem.basis.gto import GTOShell

ne = Molecule(["Ne"], [[0.0, 0.0, 0.0]], units="bohr")
basis = BasisSet("sto-3g")

shells_by_atom, nbf = basis.build_molecule_basis(ne)
print(f"基函数数目: {nbf}")
print(f"壳层: {len(shells_by_atom[0])}")

all_shells = shells_by_atom[0]
for i, sh in enumerate(all_shells):
    print(f"  壳层 {i}: l={sh.angular}, n_func={sh.n_functions}, n_prim={sh.n_primitives}")

# 计算积分
S = compute_overlap(basis, ne)
T = compute_kinetic(basis, ne)
V = compute_nuclear(basis, ne)
H = T + V
ERI = compute_eri(basis, ne)

print("\n--- 单电子积分 ---")
print(f"S 矩阵迹: {np.trace(S):.6f} (预期: {nbf})")
print(f"T 矩阵迹: {np.trace(T):.6f}")
print(f"V 矩阵迹: {np.trace(V):.6f}")
print(f"H 矩阵迹: {np.trace(H):.6f}")

# 对于 Ne 原子，s 和 p 轨道应该是正交的
print(f"\nS[0,1] (s-p_x 重叠): {S[0,1]:.10f} (预期: 0)")

# 查看 ERI 的一些关键值
print("\n--- 双电子积分 (同中心) ---")
# (1s|1s|1s|1s) = (ss|ss)
print(f"(1s 1s | 1s 1s) = {ERI[0,0,0,0]:.10f}")

# (2s|2s|2s|2s) - 第二个 s 壳层
print(f"(2s 2s | 2s 2s) = {ERI[1,1,1,1]:.10f}")

# (2p_x|2p_x|2p_x|2p_x)
print(f"(2p_x 2p_x | 2p_x 2p_x) = {ERI[2,2,2,2]:.10f}")

# (2p_x|2p_x|2p_y|2p_y)
print(f"(2p_x 2p_x | 2p_y 2p_y) = {ERI[2,2,3,3]:.10f}")

# (2p_x|2p_y|2p_x|2p_y)
print(f"(2p_x 2p_y | 2p_x 2p_y) = {ERI[2,3,2,3]:.10f}")

# 验证球对称: Σ_i (2p_i 2p_i | 2s 2s)
sum_p_ss = 0.0
for i in range(3):
    sum_p_ss += ERI[2+i, 2+i, 1, 1]
print(f"\nΣ_i (2p_i 2p_i | 2s 2s) = {sum_p_ss:.10f}")

# 直接用原语积分验证
s_shell = all_shells[1]  # 2s
p_shell = all_shells[2]  # 2p

print(f"\n--- 直接壳层积分验证 ---")
ss_block = shell_eri(s_shell, s_shell, s_shell, s_shell)
print(f"(2s 2s | 2s 2s) = {ss_block[0,0,0,0]:.10f}")

sp_block = shell_eri(p_shell, p_shell, s_shell, s_shell)
print(f"(2p_x 2p_x | 2s 2s) = {sp_block[0,0,0,0]:.10f}")
print(f"(2p_y 2p_y | 2s 2s) = {sp_block[1,1,0,0]:.10f}")
print(f"(2p_z 2p_z | 2s 2s) = {sp_block[2,2,0,0]:.10f}")
sum_sp = sp_block[0,0,0,0] + sp_block[1,1,0,0] + sp_block[2,2,0,0]
print(f"Σ (p_i p_i | ss) = {sum_sp:.10f}")

pp_block = shell_eri(p_shell, p_shell, p_shell, p_shell)
print(f"\n(2p_x 2p_x | 2p_x 2p_x) = {pp_block[0,0,0,0]:.10f}")
print(f"(2p_x 2p_x | 2p_y 2p_y) = {pp_block[0,0,1,1]:.10f}")
print(f"(2p_x 2p_y | 2p_x 2p_y) = {pp_block[0,1,0,1]:.10f}")

# 计算球对称总和
sum_total = 0.0
for i in range(3):
    for j in range(3):
        sum_total += pp_block[i,i,j,j]
print(f"\nΣ_i Σ_j (p_i p_i | p_j p_j) = {sum_total:.10f}")

# 现在让我们手动计算参考值
# 用 STO-3G Ne 基组的指数和系数
print("\n--- 手动计算参考值 ---")
from calchem.integrals.two_electron import _ERIRecursionFixed, _gaussian_product
import math

# 取第一个原语来测试
alpha = s_shell.exponents[0]
print(f"使用 alpha = {alpha}")

origin = np.zeros(3)
rec = _ERIRecursionFixed(alpha, alpha, alpha, alpha, origin, origin, origin, origin)

# (ss|ss)
ssss = rec.get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
print(f"(ss|ss) = {ssss:.10f}")

# 解析值
zeta = 2 * alpha
ref = 2 * math.pi**2.5 / (zeta * zeta * math.sqrt(2*zeta))
print(f"解析值 = {ref:.10f}")
print(f"比值 = {ssss/ref:.10f}")

# (p_x p_x | ss)
pxpxss = rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
print(f"\n(p_x p_x | ss) = {pxpxss:.10f}")

# 用有限差分计算 -∂J/∂β1 / 3
h = 1e-5
J = lambda z1, z2: 2 * math.pi**2.5 / (z1 * z2 * math.sqrt(z1 + z2))
dJ_dz1 = -(J(zeta + h, zeta) - J(zeta - h, zeta)) / (2 * h)
print(f"(1/3)(-∂J/∂β1) = {dJ_dz1/3:.10f}")
print(f"比值 = {pxpxss / (dJ_dz1/3):.10f}")

# 等等！(p_x p_x | ss) 是 bra 侧两个中心 A 和 B 都是 s 吗？
# 不，在 get(la, ma, na, lb, mb, nb, ...) 中
# la,ma,na 是中心 A 的角动量
# lb,mb,nb 是中心 B 的角动量
# lc,mc,nc 是中心 C 的角动量
# ld,md,nd 是中心 D 的角动量
# 
# 所以 (p_x s | ss) = get(1,0,0, 0,0,0, 0,0,0, 0,0,0)
# 而 (p_x p_x | ss) 不是直接有的，因为 bra 侧只有一个中心有角动量
# 
# 等等，我混淆了。让我重新理解：
# (p_x p_x | ss) 在化学记号中表示两个 p 轨道在 bra 侧，
# 但它们是在同一个中心还是不同中心？
# 
# 对于同中心的情况，(p_x p_x | ss) 应该是：
# bra: 中心 A p_x + 中心 B p_x，其中 A=B
# ket: 中心 C s + 中心 D s，其中 C=D
# 
# 不对，化学记号 (ij|kl) 中 i,j,k,l 是基函数索引。
# 对于同中心的 p_x 和 s 轨道：
# (p_x p_x | s s) = 积分 φ_{p_x}(r1) φ_{p_x}(r1) 1/r12 φ_s(r2) φ_s(r2) dr1 dr2
# = ∫∫ [x² exp(-2α r1²)] 1/r12 [exp(-2α r2²)] dr1 dr2
# 
# 这对应的原语积分是：中心 A=p_x, 中心 B=s (不对，A 和 B 都在原点)...
# 等等，A=B=C=D=原点，那么：
# bra: φ_A(r) φ_B(r) = x^la exp(-α r²) * x^lb exp(-α r²) = x^{la+lb} exp(-2α r²)
# 所以如果 la=1, lb=1，那就是 x² exp(-2α r²)
# 
# 但是在 Obara-Saika 中，(a, b | c, d) 的意思是：
# bra: 中心 A 角动量 a + 中心 B 角动量 b
# ket: 中心 C 角动量 c + 中心 D 角动量 d
# 
# 当 A=B 时，这确实对应于 x^{a_x + b_x} exp(-(α_A + α_B) r²)
# 因为两个高斯函数在同中心相乘就是指数相加、角动量相加。
# 
# 所以对于同中心的 (p_x p_x | s s)，我们有：
# A = B = 原点，α_A = α_B = α
# C = D = 原点，α_C = α_D = α
# la=1, lb=1, lc=0, ld=0
# 也就是 rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
# 
# 这才是正确的！我之前搞错了。
# 我之前用的是 rec.get(1,0,0, 1,0,0, 0,0,0, 1,0,0) 之类的，不对。

print("\n" + "="*60)
print("重新验证：同中心积分的正确调用方式")
print("="*60)

# (s s | s s) = get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
val = rec.get(0,0,0, 0,0,0, 0,0,0, 0,0,0)
print(f"(s s | s s) = {val:.10f}  (正确，因为 A=B=C=D)")

# (p_x s | s s) = get(1,0,0, 0,0,0, 0,0,0, 0,0,0)
# 这对应 bra: x^1 * 1 = x，ket: 1*1 = 1
val_psss = rec.get(1,0,0, 0,0,0, 0,0,0, 0,0,0)
print(f"(p_x s | s s) = {val_psss:.10f}  (应该为 0，因为奇函数)")

# (p_x p_x | s s) = get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
# bra: x^1 * x^1 = x²，ket: 1*1 = 1
val_pxpxss = rec.get(1,0,0, 1,0,0, 0,0,0, 0,0,0)
print(f"(p_x p_x | s s) = {val_pxpxss:.10f}")
print(f"  解析值 (1/3)(-∂J/∂β) = {dJ_dz1/3:.10f}")
print(f"  比值 = {val_pxpxss / (dJ_dz1/3):.10f}")

# 验证球对称
sum_p2ss = 0.0
for i in range(3):
    li = [0,0,0]
    li[i] = 1
    val = rec.get(li[0],li[1],li[2], li[0],li[1],li[2], 0,0,0, 0,0,0)
    sum_p2ss += val
print(f"\nΣ_i (p_i p_i | s s) = {sum_p2ss:.10f}")
print(f"  -∂J/∂β = {dJ_dz1:.10f}")
print(f"  比值 = {sum_p2ss / dJ_dz1:.10f}")

# (p_x p_x | p_x p_x) = get(1,0,0, 1,0,0, 1,0,0, 1,0,0)
val_pxpxpxpx = rec.get(1,0,0, 1,0,0, 1,0,0, 1,0,0)
print(f"\n(p_x p_x | p_x p_x) = {val_pxpxpxpx:.10f}")

# 球对称总和
sum_pp = 0.0
for i in range(3):
    for j in range(3):
        li = [0,0,0]; li[i] = 1
        lj = [0,0,0]; lj[j] = 1
        val = rec.get(li[0],li[1],li[2], li[0],li[1],li[2], lj[0],lj[1],lj[2], lj[0],lj[1],lj[2])
        sum_pp += val
print(f"Σ_i Σ_j (p_i p_i | p_j p_j) = {sum_pp:.10f}")

h2 = 1e-4
d2J = (J(zeta+h2, zeta+h2) - J(zeta+h2, zeta-h2) - J(zeta-h2, zeta+h2) + J(zeta-h2, zeta-h2)) / (4*h2*h2)
print(f"  ∂²J/∂β² = {d2J:.10f}")
print(f"  比值 = {sum_pp / d2J:.10f}")

# 现在比较一下 shell_eri 的结果
print("\n" + "="*60)
print("与 shell_eri 的比较")
print("="*60)

# 用单 s 壳层测试
test_s = GTOShell("s", [alpha], [1.0], origin)
test_p = GTOShell("p", [alpha], [1.0], origin)

block_ss = shell_eri(test_s, test_s, test_s, test_s)
print(f"shell_eri (s s | s s) = {block_ss[0,0,0,0]:.10f}")

block_ps = shell_eri(test_p, test_s, test_s, test_s)  # (p s | s s)
print(f"shell_eri (p_x s | s s) = {block_ps[0,0,0,0]:.10f}")

block_pp_ss = shell_eri(test_p, test_p, test_s, test_s)  # (p p | s s)
print(f"shell_eri (p_x p_x | s s) = {block_pp_ss[0,0,0,0]:.10f}")
print(f"  对应 rec.get: {val_pxpxss:.10f}")
print(f"  相等吗? {abs(block_pp_ss[0,0,0,0] - val_pxpxss) < 1e-10}")
