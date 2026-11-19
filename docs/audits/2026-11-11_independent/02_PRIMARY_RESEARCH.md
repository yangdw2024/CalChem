# 理论依据核对（2026-11-11）

只采用原始论文与官方源码。检索目的为核对定义和组成，不依据搜索热度判定公式正确性。

1. Yanai, Tew, Handy, Chemical Physics Letters 393 (2004), CAM-B3LYP 原文：
   https://www.sciencedirect.com/science/article/pii/S0009261404008620
   精确交换为 alpha K_full + beta K_erf；alpha=.19, beta=.46, omega=.33。积分实现与 XC 半局域组成必须分别验证。
2. Libxc 官方 CAM 混合组成：
   https://gitlab.com/libxc/libxc/-/raw/devel/src/hyb_gga_xc_cam_b3lyp.c
   组成为 .35 B88 + .46 ITYH + .19 VWN + .81 LYP，Libxc 的长程/短程参数记号与本项目不同，不能直接按符号照搬。
3. Libxc 官方 ITYH 数学定义：
   https://gitlab.com/libxc/libxc/-/raw/devel/maple/gga_exc/gga_x_ityh.mpl
   k_GGA = sqrt(9 pi/(2 K_B88)) rho_spin^(1/3)。现有源码用 k_F，漏掉梯度链式项。本次独立按公式修正，用 70 位 mpmath 参考点及解析偏导/有限差分验证；未引入外部量化库依赖。
4. Chai, Head-Gordon, PCCP 10 (2008) 6615–6620，wB97X-D：
   https://pubs.rsc.org/en/content/articlelanding/2008/cp/b810189b/unauth
   官方 Libxc 数学组成：
   https://gitlab.com/libxc/libxc/-/raw/devel/maple/gga_exc/hyb_gga_xc_wb97.mpl
   需要衰减交换与拟合的同自旋/异自旋 B97 相关。现有简化三项交换多项式+PBE 相关不符合定义，亦无原始配套色散。本次保护性拒绝该标准方法，保留内部 legacy 实现供后续完整替换。wB97X 与 wB97X-D 不应互作物理等价别名。

erf ERI 自行复核：rho=pq/(p+q)，theta=omega/hypot(omega,sqrt(rho))，辅助积分为 theta^(2m+1) F_m(theta² rho R_PQ²)。一旦在辅助积分中纳入这些因子，VRR 系数不应重复缩放。以 Gaussian 中心微分恒等式 dI/dA=2 alpha I_(a+1)-a I_(a-1) 独立交叉检查至 f 角动量及导数所需更高阶辅助积分。

上述网页访问记录可追溯至本次会话。源码 devel 分支会变；本地探针和数值期望已随审计保存。公式/点值核对不构成 Gaussian/ORCA 分子能量认证。
