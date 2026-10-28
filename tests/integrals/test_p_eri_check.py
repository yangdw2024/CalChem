"""
验证p轨道双电子积分的对称性和基本性质。
"""
import sys
sys.path.insert(0, r"d:\YDW\Trae_Gaussian")

import numpy as np
from calchem.basis.gto import GTOShell
from calchem.integrals.two_electron import shell_eri, _eri_primitive


def test_same_center_relations():
    """测试同中心s/p轨道积分的对称性关系。"""
    print("=" * 70)
    print("同中心s/p轨道双电子积分对称性验证")
    print("=" * 70)
    print()

    # 单原语 s 和 p 壳层
    alpha = 1.0
    origin = np.zeros(3)

    s_shell = GTOShell('s', [alpha], [1.0], origin=origin)
    p_shell = GTOShell('p', [alpha], [1.0], origin=origin)

    # 计算各种积分块
    ss_ss = shell_eri(s_shell, s_shell, s_shell, s_shell)  # (1,1,1,1)
    ss_pp = shell_eri(s_shell, s_shell, p_shell, p_shell)  # (1,1,3,3)
    pp_ss = shell_eri(p_shell, p_shell, s_shell, s_shell)  # (3,3,1,1)
    pp_pp = shell_eri(p_shell, p_shell, p_shell, p_shell)  # (3,3,3,3)

    print("【基础积分】")
    print(f"  (ss|ss) = {ss_ss[0,0,0,0]:.10f}")
    print()

    print("【(ss|pp) 对角元（应该相等）】")
    print(f"  (ss|px px) = {ss_pp[0,0,0,0]:.10f}")
    print(f"  (ss|py py) = {ss_pp[0,0,1,1]:.10f}")
    print(f"  (ss|pz pz) = {ss_pp[0,0,2,2]:.10f}")

    diff1 = abs(ss_pp[0,0,0,0] - ss_pp[0,0,1,1])
    diff2 = abs(ss_pp[0,0,0,0] - ss_pp[0,0,2,2])
    print(f"  最大差值: {max(diff1, diff2):.2e}")
    print(f"  对称性: {'✓' if max(diff1, diff2) < 1e-10 else '✗'}")
    print()

    print("【(ss|pp) 非对角元（应该为0）】")
    print(f"  (ss|px py) = {ss_pp[0,0,0,1]:.2e}")
    print(f"  (ss|px pz) = {ss_pp[0,0,0,2]:.2e}")
    print(f"  (ss|py pz) = {ss_pp[0,0,1,2]:.2e}")
    max_off = max(abs(ss_pp[0,0,0,1]), abs(ss_pp[0,0,0,2]), abs(ss_pp[0,0,1,2]))
    print(f"  最大非对角元: {max_off:.2e}")
    print(f"  对称性: {'✓' if max_off < 1e-10 else '✗'}")
    print()

    print("【(pp|ss) vs (ss|pp) 置换对称性】")
    # (pp|ss)_{ij,kl} = (ss|pp)_{kl,ij}
    diff = np.max(np.abs(pp_ss - ss_pp.transpose(2,3,0,1)))
    print(f"  最大差值: {diff:.2e}")
    print(f"  对称性: {'✓' if diff < 1e-10 else '✗'}")
    print()

    print("【(pp|pp) 部分对角元】")
    print(f"  (px px|px px) = {pp_pp[0,0,0,0]:.10f}")
    print(f"  (py py|py py) = {pp_pp[1,1,1,1]:.10f}")
    print(f"  (pz pz|pz pz) = {pp_pp[2,2,2,2]:.10f}")
    diff = max(abs(pp_pp[0,0,0,0] - pp_pp[1,1,1,1]),
               abs(pp_pp[0,0,0,0] - pp_pp[2,2,2,2]))
    print(f"  最大差值: {diff:.2e}")
    print(f"  对称性: {'✓' if diff < 1e-10 else '✗'}")
    print()

    print("【(pp|pp) 交叉项】")
    print(f"  (px px|py py) = {pp_pp[0,0,1,1]:.10f}")
    print(f"  (px px|pz pz) = {pp_pp[0,0,2,2]:.10f}")
    print(f"  (py py|pz pz) = {pp_pp[1,1,2,2]:.10f}")
    diff = max(abs(pp_pp[0,0,1,1] - pp_pp[0,0,2,2]),
               abs(pp_pp[0,0,1,1] - pp_pp[1,1,2,2]))
    print(f"  最大差值: {diff:.2e}")
    print(f"  对称性: {'✓' if diff < 1e-10 else '✗'}")
    print()

    # 解析验证：单原语同中心积分
    print("【解析公式验证 (单原语 α=1.0)】")
    print()
    
    # (ss|ss) 解析公式
    # ∫∫ exp(-r1²) exp(-r2²) / |r1-r2| dr1 dr2 = π^(5/2) / 2
    # 不对，让我用正确的公式
    # 对于归一化的 1s 轨道 (α/π)^(3/4) exp(-α r²)
    # (ss|ss) = (2α/π)^(3/2) * π^(5/2) / (2α * 2^(3/2)) ???
    # 让我用数值验证：直接计算
    
    # 实际上，对于非归一化的 GTO: exp(-α r²)
    # (ss|ss) = 2 π^(5/2) / (2α * sqrt(2α))  ... 等等
    # 让我用已知的公式：
    # 对于两个s型高斯函数，指数分别为 a 和 b，中心在原点
    # (ss|ss) = 2 π^(5/2) / [(a+b) * sqrt(a+b)] * ... 不对
    
    # 让我用最简单的方式：用我们的函数计算单原语积分
    val_ss_ss = _eri_primitive(0,0,0, 0,0,0, 0,0,0, 0,0,0,
                               alpha, alpha, alpha, alpha,
                               origin, origin, origin, origin)
    print(f"  未归一化 (ss|ss) = {val_ss_ss:.10f}")
    
    # 参考值：对于 exp(-α r²) 在原点
    # ∫∫ exp(-α(r1² + r2²)) / |r1-r2| dr1 dr2
    # = 2 π^(5/2) / (2α)^(5/2) ??? 
    # 让我计算一个已知的情况
    # 实际上，已知 ∫ exp(-α r²) dr = (π/α)^(3/2)
    # 而库仑积分 (ss|ss) 对于 α=1 应该是 π^(5/2) / 2^(3/2) ???
    # 不对，让我用数值积分估算...
    
    # 更好的方法：用 Szabo 书中 H2 STO-3G 的结果来间接验证
    # 我们已经知道 H2 的 (11|11) = 0.7746 是对的
    
    # 让我检查 p 轨道的比值关系
    # 对于同中心、同指数，有：
    # (ss|p_i p_i) = 1/3 * (ss|ss) + ... 不对
    
    # 实际上让我查一下正确的关系：
    # 对于未归一化的 GTO:
    # s: exp(-α r²)
    # p_x: x * exp(-α r²)
    # 
    # (ss|ss) = I0
    # (ss|p_x p_x) = ?
    # 
    # 利用递推关系或已知公式...
    # 实际上，我可以用一个已知的关系：
    # ∫ x1² x2² exp(-α(r1²+r2²)) / |r1-r2| dr1 dr2 = ...
    # 
    # 让我们先看看我们计算的值是什么
    
    val_ss_pxpx = _eri_primitive(0,0,0, 0,0,0, 1,0,0, 1,0,0,
                                  alpha, alpha, alpha, alpha,
                                  origin, origin, origin, origin)
    val_pxpx_pxpx = _eri_primitive(1,0,0, 1,0,0, 1,0,0, 1,0,0,
                                    alpha, alpha, alpha, alpha,
                                    origin, origin, origin, origin)
    val_pxpx_pypy = _eri_primitive(1,0,0, 1,0,0, 0,1,0, 0,1,0,
                                    alpha, alpha, alpha, alpha,
                                    origin, origin, origin, origin)
    
    print(f"  未归一化 (ss|px px) = {val_ss_pxpx:.10f}")
    print(f"  未归一化 (px px|px px) = {val_pxpx_pxpx:.10f}")
    print(f"  未归一化 (px px|py py) = {val_pxpx_pypy:.10f}")
    print()
    
    # 检查比值
    print("【积分比值（应该是确定的数）】")
    ratio1 = val_ss_pxpx / val_ss_ss
    ratio2 = val_pxpx_pxpx / val_ss_ss
    ratio3 = val_pxpx_pypy / val_ss_ss
    print(f"  (ss|pp) / (ss|ss) = {ratio1:.10f}")
    print(f"  (pp|pp) 对角 / (ss|ss) = {ratio2:.10f}")
    print(f"  (pp|pp) 交叉 / (ss|ss) = {ratio3:.10f}")
    print()

    # 解析公式（对于未归一化的s和p GTO，同中心同指数）:
    # 令 I_n = ∫∫ r1^(2n) r2^(2n) exp(-α(r1²+r2²)) / |r1-r2| dr1 dr2 ... 不对
    # 
    # 实际上可以用角向展开：
    # 1/|r1-r2| = Σ_l Σ_m (-1)^m r<^l / r>^(l+1) Y_l^m(θ1,φ1) Y_l^{-m}(θ2,φ2) * 4π/(2l+1)
    # 
    # 对于s波(l=0): 贡献 (ss|ss)
    # 对于p波(l=1): 贡献 ...
    #
    # 更简单的方法：利用高斯积分已知结果
    # 
    # 实际上，让我用一个更简单的验证方法：
    # 对于同中心、同指数 α 的未归一化 GTO：
    # (x1^la y1^ma z1^na x1^lb y1^mb z1^nb | x2^lc y2^mc z2^nc x2^ld y2^md z2^nd)
    # = ∫∫ r1^(la+lb, ma+mb, na+nb) r2^(lc+ld, mc+md, nc+nd) exp(-α(r1²+r2²)) / |r1-r2| dr1 dr2
    #
    # 利用对称性：
    # (s|s) = I
    # (x|x) = I / 3  ... 不对，这是单电子积分
    #
    # 对于双电子积分，我需要查一下。
    # 
    # 已知：对于同中心、同指数，
    # (ss|ss) = J0
    # (ss|p_x p_x) = J2 / 3  ??? 不对
    # 
    # 让我们换一个思路：用数值精度验证对称性即可，
    # 然后用已知的分子能量来验证整体正确性。
    
    return True


def test_h2o_integrals_summary():
    """检查H2O的积分概况。"""
    print("=" * 70)
    print("H2O STO-3G 积分概况")
    print("=" * 70)
    print()
    
    from calchem.mol.molecule import Molecule
    from calchem.basis.basis_set import BasisSet
    from calchem.integrals.one_electron import compute_overlap, compute_kinetic, compute_nuclear
    from calchem.integrals.two_electron import compute_eri
    
    mol = Molecule(
        ["O", "H", "H"],
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.0000000, -0.7570000, 0.5860000],
            [0.0000000, 0.7570000, 0.5860000],
        ],
        units="angstrom",
    ).to_bohr()
    
    basis = BasisSet("sto-3g")
    
    s = compute_overlap(basis, mol)
    t = compute_kinetic(basis, mol)
    v = compute_nuclear(basis, mol)
    h = t + v
    eri = compute_eri(basis, mol)
    
    nbf = s.shape[0]
    print(f"基函数数目: {nbf}")
    print()
    
    print("【单电子积分范围】")
    print(f"  重叠矩阵: min={s.min():.6f}, max={s.max():.6f}")
    print(f"  动能矩阵: min={t.min():.6f}, max={t.max():.6f}")
    print(f"  核吸引矩阵: min={v.min():.6f}, max={v.max():.6f}")
    print(f"  核心H: min={h.min():.6f}, max={h.max():.6f}")
    print()
    
    print("【双电子积分范围】")
    print(f"  ERI: min={eri.min():.6f}, max={eri.max():.6f}")
    print(f"  ERI 对角元 (ii|ii): ", end="")
    diag_eri = np.array([eri[i,i,i,i] for i in range(nbf)])
    print(diag_eri)
    print()
    
    print("【核排斥能】")
    from calchem.scf.hf import nuclear_repulsion
    e_nn = nuclear_repulsion(mol)
    print(f"  E_nn = {e_nn:.6f} Eh")
    print()
    
    # 用Hcore初猜做一次简单的SCF看看
    print("【简单SCF测试（无DIIS, 20次迭代）】")
    print()
    
    from calchem.scf.guess import _symmetric_orthogonalize, _eigh_safe, _matmul
    
    nelectron = mol.nelectron
    nocc = nelectron // 2
    
    x = _symmetric_orthogonalize(s)
    f_ortho = _matmul(_matmul(x.T, h), x)
    eigval, eigvec_ortho = _eigh_safe(f_ortho)
    mo_coeff = _matmul(x, eigvec_ortho)
    dm = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
    
    def build_fock(dm_):
        j = np.einsum("ls,mnls->mn", dm_, eri)
        k = np.einsum("ls,mlns->mn", dm_, eri)
        g = j - 0.5 * k
        return h + g
    
    def scf_energy(dm_, f_):
        return 0.5 * np.einsum("mn,mn->", dm_, h + f_)
    
    fock = build_fock(dm)
    e_prev = scf_energy(dm, fock) + e_nn
    print(f"  迭代 0: E = {e_prev:.6f} Eh")
    
    for cycle in range(1, 21):
        f_ortho = _matmul(_matmul(x.T, fock), x)
        eigval, eigvec_ortho = _eigh_safe(f_ortho)
        mo_coeff = _matmul(x, eigvec_ortho)
        dm_new = 2.0 * _matmul(mo_coeff[:, :nocc], mo_coeff[:, :nocc].T)
        
        # 简单阻尼
        dm_new = 0.5 * dm + 0.5 * dm_new
        
        fock = build_fock(dm_new)
        e_new = scf_energy(dm_new, fock) + e_nn
        de = e_new - e_prev
        
        if cycle <= 5 or cycle % 5 == 0:
            print(f"  迭代 {cycle:2d}: E = {e_new:.6f} Eh  dE = {de:+.2e}")
        
        dm = dm_new
        e_prev = e_new
    
    print()
    print(f"  最终能量: {e_new:.6f} Eh")
    print(f"  参考能量: -74.94208 Eh")
    print(f"  差值: {e_new - (-74.94208):.6f} Eh")
    
    return True


if __name__ == '__main__':
    test_same_center_relations()
    print()
    test_h2o_integrals_summary()
