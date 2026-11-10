# CalChem 独立红队审计报告

审计日期：2026-11-09（Asia/Shanghai）。对象：E:\YDW\Trae_Gaussian 当前工作树。Git HEAD：c9fd62e；大量后续实现尚未提交，故本报告不等同于对该 commit 或已发布软件的认证。

**核心结论：这是具有真实积分、HF/部分 DFT、解析一阶导数和线性响应实现的研究原型；存在足以否决整体生产验收的模型冒名、导数不一致、状态误判和分发缺陷。没有发现按水、苯、IC-2F、PM6 或 Y6 的身份直接返回预设计算答案的证据；但已确认若干同名方法并不是其宣称的标准理论。**

本次不是仅阅读 README。执行了源码扫描、关键算法逐段审查、91 项 Rust 测试、96 项 Python 单元测试、独立数值反例、当前源码扩展重建，以及 Gaussian 16 / ORCA 新实跑。通过的自带测试与失败的独立审计可以同时成立。

## 1. 【总评与等级评定】

| 项目 | 评级 | 解释 |
|---|---|---|
| 代码架构 | **B** | crate 职责及 PyO3 边界基本合理，原生数值与外部适配器可区分；实际仍有旧 Python 数值路径、重复实现、参数和结果语义漂移，线性代数及分发未兑现高性能/工业化承诺。 |
| 量子化学理论还原度 | **60/100** | 对当前“整体宣称能力”的专家审计评分；不是能量精度百分比、测试覆盖率或统计置信度。HF、部分常规 DFT、ECP 和闭壳层响应有实证；LRC、D3、溶剂导数、IR 单位和 TS/IRC 认证存在实质错误。 |
| 本地开发可运行性 | **C** | 当前 Windows 环境可编译并通过选定门禁，真实 60 原子路径可运行；仍缺独立覆盖、资源限制及干净分发验证。 |
| 生产可用性与健壮性 | **D：不通过整体验收** | 错误模型能以正确名称输出，普通极小值可被标作 TS，含溶剂的能量/梯度不一致，正常构建路径可产生错误 wheel。 |

理论评分分解：积分/基组 18/20；SCF/DFT 13/20；色散 2/10；梯度/优化/TS/IRC 12/20；激发态/振动 10/15；溶剂化 5/15。评分针对功能整体，不能反推每个模块都只有 60% 正确。

### 1.1 范围、证据和保护

- 当前 workspace 为 13 个 crate：model、basis、tensor、int、dft、scf、tddft、vib、grad、prop、opt、solv、py。最后一个是 Python 绑定，不应算作额外独立物理算法。
- 全量模式扫描 353 个代码/配置文件、65,244 行，含测试；重点阅读物理算符、收缩、迭代、边界、调用链及基准来源。这不是对每一行作形式化证明。
- 376 个原文件已快照与 SHA-256 固定；不修改业务源码、不替换原 .pyd、不删除、不提交、不发布。审计新增文件位于本目录；运行产物位于 E:\YDW\Trae_Gaussian\build_target\runtime\redteam_20261109。
- Rust/Python 工具链位于 D:；新临时文件、缓存和输出位于 E:。未安装新软件或下载依赖。初期目录权限阻断已解决，详见操作记录。
- 当前源码重新构建的 release 扩展与原 calchem/_core.pyd 的 SHA-256 均为 `6596a745135a3d1c1f2d8c41915b1fa5fd815450f522dae204318b4f5da5f6a8`。

## 2. 【底层理论与算法逐项审查意见】

### 2.1 积分与基组：真实，有数值含金量，尚非通用工业积分包

**OS 递推确实存在。** [eri.rs:118](E:/YDW/Trae_Gaussian/crates/calchem-int/src/eri.rs:118) 的 VRR、ket/bra HRR、GaussianPair、Boys 和 contracted quartet 共同形成实际四中心 ERI；重叠、动能、核吸引也使用真实高斯积分递推。没有在 Rust 生产计算链发现调用 Gaussian/ORCA/PySCF 作为隐藏积分黑盒。

**Boys 在实际角动量范围内表现很好。** [boys.rs:115](E:/YDW/Trae_Gaussian/crates/calchem-int/src/boys.rs:115) 使用小 T 三项展开、中等 T 正项 Kummer 级数加向下递推、T>35 的 erf 形式加向上递推。直接编译当前 boys.rs，以 70 位 mpmath 不完全 gamma 定义独立核对：n=0～16、T=0～10^6 共 323 个样本，最大相对误差 1.14×10^-15、最大绝对误差 2.84×10^-16；11 个 erf 样本的最大绝对误差 8.64×10^-17。

限制：切换只按 T，不按 n/T 条件选择稳定方向。公开函数 n=128、T=35.000001 返回 **−3.90157077**，精确值约 **3.3585×10^-18**；Boys 必须为正。这是通用 API 的高阶失败，**不能误说当前 s/p/d/f 常用积分全部失效**，其所需低阶在本次探针中准确。

**ECP 是真实的半局域投影实现。** [ecp.rs:1](E:/YDW/Trae_Gaussian/crates/calchem-int/src/ecp.rs:1) 包括 local、角动量投影、实球谐、缩放球 Bessel、48 点径向 Gauss–Legendre 积分；不是简单把 Z 减小。basis 核电荷和核间排斥用 Z_eff，梯度通过升降角动量及平移恒等式处理。HBr/LANL2DZ 新实跑与 Gaussian 的 HF/B3LYP 能量差分别约 6.40×10^-7 / −4.18×10^-8 Eh，支持实现真实性。固定径向阶数、有限角阶及截断仍需更重元素/极端指数独立验证；不是全相对论或自旋轨道 ECP。

**稀疏 ERI 不是假名称。** [int/lib.rs:178](E:/YDW/Trae_Gaussian/crates/calchem-int/src/lib.rs:178) 的 SparseSymmetricEri 包含 8 重对称性和 stabilizer 权重，Schwarz 使用 sqrt((μν|μν)) 壳对上界。n_AO≤64 存 dense，较大体系存规范唯一项。本轮专门跨过分支边界，在 65 AO 的自定义单原始 s 基组比较 sparse/dense：纯 J、20%/100% K 及非对称 K 收缩，最大误差均 ≤1.0×10^-15。该探针两侧使用相同筛选阈值，证明的是压缩/收缩一致性，不是任意阈值下的全体系误差界。

**内存仍有危险路径。** dense 是 8N^4 字节；parallel dense 还先 collect 全部 quartet block。gradient 先生成未筛选 unique_shell_quartets，再筛除，最坏壳层数四次方内存。以 160 shells 为例，仅 canonical quartet 的 `[usize;4]` 列表约 2.65 GB（64 位，尚未含导数工作区）。稀疏列表也无最坏线性保证；AO 索引 `as u16` 无溢出拒绝。它不是 direct-SCF、RI/Cholesky 或有内存上限的分块流式引擎。

基组只嵌入 STO-3G、6-31G、6-31G(d)、6-31G(d,p)、LANL2DZ、LANL2DZdp；Cartesian s/p/d/f，不具通用球谐 5D/7F 转换与完整 diffuse/def2/cc 系列原生支持。不能把 Python capability 列表等同于 Rust 基组覆盖。

### 2.2 SCF 与 DFT：常规部分可信，长程泛函存在身份错误

**CDIIS 为真。** [diis.rs:45](E:/YDW/Trae_Gaussian/crates/calchem-scf/src/diis.rs:45) 解带约束 Pulay B 矩阵，接近奇异时缩小历史。误差使用 AO 非正交基中的 **FPS−SPF**，数学形式正确；没有发现 EDIIS 实现。普通 DIIS 不保证每次 SCF 能量下降，不能将非单调本身判为造假。变分原理约束最终所选波函数/泛函的变分问题，不等于 DIIS 轨迹单调。本次五个独立小分子样本无超过 10^-10 Eh 的上升步骤，但不能外推。

SCF 含 RHF/UHF/RKS/UKS、电子数/自旋和几何检查、对称正交化及末端收敛细化；真实未收敛的普通 SCF 会报错，Python 原生路由对此不静默切换引擎。这些是实际修复，旧审计“梯度未实现、测试编译失败”不再适用当前版本。

**ρ 和 ∇ρ 是真算的。** [vxc.rs:28](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/vxc.rs:28) 从 P 与 AO/AO 梯度生成自旋密度和 γ_aa、γ_ab、γ_bb，执行 XC 能量及矩阵收缩。VWN5/RPA、B88、LYP、PBE 的代码包含真实公式与解析一阶偏导；Gaussian 的水分子 SVWN5/BLYP/B3LYP 对齐在本次约 0.2 μEh，具有实证支持。

**网格真实，但范围有限。** Lebedev 26/50/110/194/302 点规则有全声明代数阶单项式单测；径向为 Murray–Handy–Laming q=2 的 Euler–Maclaurin 映射，配 Becke k=3 分区和解析权重响应，不应称成已实现所有 Treutler 方案。固定半径 **18 Bohr** 截断是硬限制：[grid.rs:437](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/grid.rs:437)。本轮独立积分归一化单中心密度 `(2α/π)^(3/2) exp(-2αr²)`，α=10^-4，精确电子数 1，Fine 网格仅得到 **0.00801168**。此例是 Rust 自定义基组/弥散密度压力探针，非当前六套内置基组全部出错；它否定“对任意弥散体系有可控精度”的说法。

**CAM-B3LYP 不成立。** [xc.rs:83](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/xc.rs:83) 固定全程 HF 交换为 **0.42**；SCF 只对普通 1/r12 ERI 乘这一系数。积分引擎没有相应 erf(μr12)/r12 和 erfc(μr12)/r12 的长短程算符。标准 CAM 需要短程 0.19、长程 0.65 及 μ=0.33 的非局域交换；半局域衰减不能补上缺失的算符。[CAM 原论文](https://doi.org/10.1016/j.cplett.2004.06.011)

**ωB97X-D 同样不成立。** [xc.rs:818](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/xc.rs:818) 使用三个简化 B97 多项式系数及 PBE 相关，HF 份额固定 0.22；没有标准完整范围分离交换，`run_single_point` 也不自动加入标准 wB97X-D 的色散项。`wb97x` 还被别名到 `wb97xd`。这些是模型身份错误，不是网格不够密或积分容差略宽造成的误差。

收敛限制也应说明：RKS n_AO≥40 末端细化最多两步，失败后仍以 commutator≤10^-6 放行；还取新旧 residual 的 min。与项目“严格 <10^-8”原则不一致，可能掩盖末端恶化；普通 RHF/UKS 等相关路径亦需逐一统一。重叠阈值以下方向被置零，但不建立明确降秩 MO 空间；`level_shift` 选项在结构中存在，未见 SCF 实际使用。没有普适 SCF 稳定性分析、断键/近简并、破缺对称态的系统验收。

### 2.3 D3：可微的自定义经验势，不是标准 Grimme D3

真实部分：有随距离变化的 fractional CN，BJ/zero damping 表达式及包含 dCN/dR 链式法则的解析梯度。梯度对自身能量的差分测试通过。

决定性问题：[d3.rs:589](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/d3.rs:589) 将单元素 C6 设为 **base×exp[−0.05(CN−CN_ref)]**；[d3.rs:283](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/d3.rs:283) 以调和平均组合 C6，并用 vdW 半径构造 C8。没有标准参考环境的 C6_AB(CN_A,CN_B) 数据与权重插值，也不是标准 r4/r2 数据路径。标准模型的参考结构和插值可直接在[维护者实现](https://dftd3.github.io/simple-dftd3/module/dftd3_model.html)核查。

同一水分子、B3LYP-D3(BJ) 的色散项：Gaussian **−0.0005738766 Eh**，ORCA **−0.000573877 Eh**，CalChem **−0.00115453735 Eh**。CalChem 约为标准项的 **2.012 倍**。这是对色散项本身的比较，不混入两程序 B3LYP 默认变体差异。

未知泛函在 Rust D3Model 中静默换用 PBE 参数，但 functional_name 保留原字符串；本轮 `not-a-functional` 仍返回能量。Python [d3.py:150](E:/YDW/Trae_Gaussian/calchem/dft/d3.py:150) 捕获所有异常后可走另一套简化实现，自定义参数也会改变算法路径，a1/a2 和 damping 的语义不能保证一致。标准 D3 的经验性本身不是问题；冒用标准名称且缺少数值等价验证才是问题。

### 2.4 解析导数、优化、TS、IRC：气相梯度强，混合路径和认证弱

**气相一阶导数是真解析。** [derivatives.rs:61](E:/YDW/Trae_Gaussian/crates/calchem-grad/src/derivatives.rs:61) 使用 ∂_Aχ_l=2αχ_(l+1)−lχ_(l−1)，对 overlap、kinetic、nuclear、ERI/ECP 求核导数；[grad/lib.rs:97](E:/YDW/Trae_Gaussian/crates/calchem-grad/src/lib.rs:97) 收缩 P、双电子密度组合、能量加权密度 W，包含 **−Tr(W S')** 的 AO Pulay 项。DFT 另加 AO Hessian、移动网格及 Becke 权重导数。

独立非对称水/水阳离子，RHF/STO-3G、RHF/6-31G(d,p)、B3LYP/STO-3G、PBE/STO-3G、UHF/STO-3G，在三种差分步长下有二阶收敛；h=10^-4 Bohr 时最大绝对差约 1.2～1.4×10^-8 Eh/Bohr，向量 L2 相对误差约 **1.8×10^-8**，满足此次 <10^-5 要求。总梯度约 10^-13，HF 旋转/平移/排列不变性近机器精度。DFT 旋转后的梯度最大差约 3.4×10^-6，体现有限角网格误差，不能宣称严格旋转不变。

**Rust 优化器是完整逆 Hessian BFGS，不是 L-BFGS-B。** [opt/lib.rs:344](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/lib.rs:344) 保存 3N×3N 矩阵，检查 s·y>10^-10，失败重置；有 Armijo 回溯及步长上限，没有 bound constraints 或 Wolfe 条件。主要路径合理，但回溯失败后的备用步骤只检查能量能算出来，不要求降低或满足 Armijo：[opt/lib.rs:474](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/lib.rs:474)。这会破坏接受准则。更新采用显式矩阵乘法，非大体系低存储优化。

**含溶剂优化梯度不自洽。** [opt/lib.rs:205](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/lib.rs:205) 无论原请求 HF/DFT/open shell，另算闭壳层 RHF 密度作溶剂修正；默认溶剂梯度是固定 AO 密度的差分，不是总能量随自洽 P(R) 的全导数。其能量表面积项与导数辅助函数的固定 0.005 系数亦不一致。独立 Rust 总能量中心差分与返回优化梯度的最大差 **8.5666×10^-3 Eh/Bohr**，比 <10^-5 目标大数百倍。带溶剂优化及相关频率/重组能应停止科学使用。

**TS 有真实 P-RFO 和 Bofill，但“认证”失败。** 模式重叠追踪、分区步进和 Bofill PSB/MS 插值是实际实现；[prfo.rs:326](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/prfo.rs:326) 只按力和步长判收敛，不要求恰好一个振动负模。本轮将普通优化后的 H₂ 极小值传入，返回 **converged=true**、所谓 imaginary_frequency_cm=**+3.78×10^-5**。此外直接将未质量加权、未剔除平转的 Cartesian Hessian 特征值乘频率常数，不构成物理虚频。必须在收敛后重算质量加权/投影 Hessian 并验证一阶鞍点。

**IRC 有 Gonzalez–Schlegel 超球面结构，但不健壮。** [irc.rs:177](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/irc.rs:177) 的 ds 固定，无自适应步长/失败重试；内层达到 max_gs_iter 未验证约束收敛，更新 q_trial 后可能使用更新前 last_eg，导致下一步坐标与保存能量/梯度不一致。[irc.rs:279](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/irc.rs:279) 在能量反弹时将 converged_to_minimum 标 true，即使梯度不小。当前不能把输出端点与能垒当成已验证反应物/产物极小值。

### 2.5 TDDFT/CIS、振动与热力学：真实响应，IR 单位存在确定错误

**TDDFT 不是轨道能差冒充。** [tddft/lib.rs:465](E:/YDW/Trae_Gaussian/crates/calchem-tddft/src/lib.rs:465) 构造 occupied–virtual 线性响应的 A、A±B 作用，包括 J、K、f_xc；[davidson.rs:255](E:/YDW/Trae_Gaussian/crates/calchem-tddft/src/davidson.rs:255) 用对称化的 (A−B)^(1/2)(A+B)(A−B)^(1/2) 求平方频率，这是与稳定 Casida 问题等价且更安全的形式，不必机械要求显式乘出非对称乘积。分块 Davidson、预条件、子空间重启、残差和 RPA 不稳定性拒绝均存在，小体系用 dense 快径。

跃迁偶极使用真实 AO r 积分及 sqrt(2)(X+Y)，振子强度 f=(2/3)ω|μ|²，自旋禁阻 triplet 设零正确。B3LYP/STO-3G 水分子三个 full-TDDFT 根和 Gaussian 在其输出精度内一致；CIS 在统一 **Full、无冻结核** 后亦一致，见后表。限制：闭壳层 singlet/triplet 为主；f_xc 的局域二阶偏导由 [fxc.rs:110](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/fxc.rs:110) 的四点差分获得，不是全解析二阶 XC。CAM/ωB97X-D 错误算符会传递到响应，不能据 B3LYP 成功推定 CT 态可靠。

**Hessian 为半数值二阶导数。** [hessian.rs:140](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/hessian.rs:140) 中心差分解析一阶梯度和偶极 APT，不是解析二阶 Hessian。质量加权、质心、平转向量的 Gram–Schmidt 与投影 PHP 基本标准，线性分子保留 3N−5。按零附近特征值删除模式而非严格在振动子空间求解，对极软模式还需压力验证。

同一优化水 HF/STO-3G：Gaussian 频率 2170.0559/4140.0126/4391.0684 cm^-1；CalChem 2170.05582/4140.01265/4391.06851 cm^-1，最大差约 1.2×10^-4 cm^-1。ZPE 亦匹配输出精度。

**IR 强度错了约 23.07 倍。** Gaussian 7.2374/44.2880/29.9732 km/mol；CalChem 0.313705/1.919661/1.299185。原因定位到 [masses.rs:22](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/masses.rs:22) 的 42.2561 与 [modes.rs:298](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/modes.rs:298) 的原子单位 dipole/Bohr 组合不匹配；该系数对应的单位转换没有正确接入。不能以频率准证明强度准。

量纲诊断也支持这个因子：原子单位偶极/Bohr 变成 Debye/Å 要乘 2.54174623/0.529177210903，其平方约 23.07；因此应先统一 APT 与质量加权坐标单位，再推导强度常数，不能仅调整回归参考值。

RRHO 基本公式、ZPE、E/H/G、Cv/Cp、理想气体电子熵可辨识。并非所有常数都达到统一 CODATA 精度：[thermo.rs:21](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/thermo.rs:21) 的 amu·Bohr²→kg·m² 为 4.6509653×10^-48，独立常数乘积约 4.64998298×10^-48，相对偏差 2.11×10^-4；主要影响转动熵。默认对称数为 1，不自动识别 H₂O 的 2，1 cm^-1 以下振动被排除，也无准 RRHO 低频修正。用于自由能必须明确温度、压力、同位素、对称数与低频处理。

### 2.6 连续介质：有真实静电求解，不等于完整 SMD/IEFPCM

[cavity.rs:123](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/cavity.rs:123) 使用 Bondi 球面和 Lebedev 面元；[potential.rs:29](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/potential.rs:29) 以真实量子 AO 势积分生成 V；[matrix.rs:23](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/matrix.rs:23) 构造 Coulomb 表面矩阵、Cholesky，并以 q=−fA^-1V 求电荷。RHF/RKS SCRF 和 post-SCF 两条路径确实存在。

限制很实质：硬 buried/unburied 面元筛选使空腔不平滑；PyO3 默认固定 50 点/原子，未暴露充分面元精度控制；矩阵显式求逆 O(N_s^3)，势张量 O(N_s N_AO²)，flatten 时有额外峰值；Cholesky 可加入较大 shift 但结果不报告该 shift 和原始 Aq+fV 残差；电荷归一化后不再保证解原始线性方程。

**SMD 名称不成立。** [smd.py:15](E:/YDW/Trae_Gaussian/calchem/solvent/smd.py:15) 仅调用 C-PCM 并启用 CDS，Rust CDS=宏观表面张力×总表面积；缺乏标准 SMD 的原子/局域环境表面张力与相应参数体系。IEFPCM 和 Onsager 也只是同一 conductor matrix 上换介电比例，并非完整对应形式。

**溶剂梯度默认是差分且模型参数被忽略。** [gradient.rs:200](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/gradient.rs:200) 路由到固定密度中心差分；辅助能量硬用 Cpcm 和 0.005 表面积系数。公开 analytical 函数的电子项使用核电荷类近似，不能认定为完整 AO 势导数。PyO3 的 `_solvent_model` 未使用，本轮 cpcm/cosmo 返回逐位相同梯度。

**溶剂化自由能参考不严谨。** [scrf.rs:295](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/scrf.rs:295) 将收敛溶液密度上的 gas-like 能量标为 gas_energy，未独立求气相极小 SCF，遗漏密度极化代价的参考差。SCRF 自身未收敛返回 converged=false 对象，Python native 会标 not_converged，这点没有伪装成功；但不符合项目“统一 Result::Err”原则，且末端 P/F/轨道一致性未像气相路径细化。

该参考问题也已数值复现：非对称水 RHF/STO-3G 的真正独立气相能为 −74.750008601108 Eh，返回 gas_energy=−74.749879893036 Eh，相差 1.28708×10^-4 Eh。返回 ΔG=0.683381 kcal/mol，而总溶液能减独立气相能为 0.764146 kcal/mol，差约 0.080766 kcal/mol。此检验是内部结果语义一致性检查，不是对标准 PCM 绝对溶剂化能的外部认证。

## 3. 【严苛挑刺与潜在缺陷揭露】

### 3.1 优先级与反例

P0=足以使关键科学方法身份失真，应停止以该名称交付；P1=可产生错误科学量/错误认证或正常分发失败；P2=能力/精度/鲁棒性缺口；P3=较小一致性问题。优先级不是对主观动机的判断。

| 级别 | 问题 | 本轮证据 | 影响 |
|---|---|---|---|
| **P0** | CAM-B3LYP、ωB97X-D 无标准范围分离算符 | 水同几何同 6D 基组分别偏 −1.31368 / +0.699154 Eh | 总能、轨道、CT 激发态及所有 OPV 衍生量的方法标签错误。 |
| **P1** | 标为 D3 的自定义 C6/C8 模型 | 水 BJ 项为两独立标准程序约 2.012 倍 | 簇结合能、构象排序、界面能和几何偏差。 |
| **P1** | 含溶剂优化能量/梯度不一致 | 中心差分最大差 8.57×10^-3 Eh/Bohr | 伪极小值、优化停滞、错误频率/重组能。 |
| **P1** | TS 仅力小就认证，虚频来自错误 Hessian 单位 | H₂ 极小值被返回 converged=true | 鞍点、反应路径及能垒可被错误标识。 |
| **P1** | IRC 内层未收敛/缓存几何不一致，反弹标到达极小值 | 源码状态与更新顺序可直接确认 | 错误端点与反应能/势垒；现有 HCN 下坡单测不能排除。 |
| **P1** | IR 单位系数错误 | 三个水模强度均小约 23.07 倍 | 强度、谱图和定量光谱比较失真。 |
| **P1** | SMD/溶剂能参考及导数与名称不符 | SMD=CPCM+γA；gas_energy 非独立气相解；cosmo 梯度参数无效 | 不可当标准 SMD 自由能或解析介质梯度。 |
| **P1** | 默认 wheel 构建链错误 | 干净源码生成无扩展 py3-none-any；含 .pyd 时仍标 any | 用户安装得不到内核或平台标签错误。 |
| **P2** | 固定半径网格不支持一般弥散密度精度 | 精确 N=1 算得 0.00801 | 阴离子/CT/自定义 diffuse 基组危险。 |
| **P2** | 大体系资源限制未真正约束核心分配与执行 | 四次方列表、显式 A^-1、稠密张量；FFI 多入口持有 GIL | OOM/进程中止、长任务不可及时取消；不能称工业高通量。 |
| **P2** | 末端 stationary tolerance、近线性相关及 option 落地不统一 | 接受 residual 10^-6；未降秩；level_shift 未用 | 梯度级收敛承诺过强。 |
| **P2** | 任意高 n 的 Boys 向上递推不稳定 | n=128,T≈35 得负值 | 公开函数能力边界未守卫；当前低阶范围已实证准确。 |

### 3.2 反作弊与真实性判定

1. 自有代码模式扫描未见 unsafe、static mut、Box::leak 或 mem::forget；Rust 生产代码未见目标分子的硬编码绝对能量或按其身份分支返回答案。water/benzene 命中主要为溶剂数据库，测试模块的参考值单独分类。
2. 水分子扰动、坐标平移/旋转/原子排列和独立 Gaussian 对齐支持真实从积分到 SCF 的依赖链。坐标扫描能量/梯度/偶极连续，某一方向在拉伸后梯度过零，符合势能曲线响应。
3. 固定基组系数、CODATA 常数、泛函参数和参考答案用于回归，本身不是作弊。CHEMDATAS_REFERENCE_DATA 是硬编码参考元数据，本轮确实复核了可用原始文件；未发现其替代 Rust SCF 输出。
4. 本轮不能证明不存在任何刻意混淆或所有未测路径都诚实，但已有确定证据足以说：**主要问题是错误模型/标签和验证设计，不能简化成“全是假数据”，也不能被“没有按分子硬返回”洗成全部科学正确。**
5. “只依赖元素+坐标+基组”并不完整：电荷、多重度、方法、ECP、介电常数等是必需输入。D3、SMD、Scharber 也含经验参数。OPV 的 Voc 与“ΔLUMO≥0.30 eV 可解离”是经验启发式，尤其不能视为非富勒烯受体的第一性原理普适判据。

### 3.3 工程安全与分发

Rust 自有代码未见 unsafe 滥用，Rayon 主要以局部 fold/reduce 聚合，无直接共享可变数组的证据，也未发现自有核心 Mutex 死锁模式；**未做第三方依赖 unsafe 审计、TSan、Miri、长期泄漏或跨平台运行验证**。Rust 无 GC，但大量 Vec/clone 分配和峰值内存仍存在，不能把“无 GC”说成“不会 OOM”。

Faer 只见依赖声明，当前 [tensor/lib.rs:100](E:/YDW/Trae_Gaussian/crates/calchem-tensor/src/lib.rs:100) 的 matmul 为三层标量循环，特征分解为 cyclic Jacobi，线性方程为自写消元；未见实际 faer:: 调用、自研 tensor pool、显式 SIMD/CUDA 路径。主要加速来自编译后的积分、筛选、对称压缩和部分 Rayon；不能据此声称全面 SIMD/GPU 或整体优于成熟软件。

PyO3 的 single_point/gradient/optimize/solvation 等长任务未统一 allow_threads；部分 TDDFT/振动/性质接口已释放 GIL，说明边界不一致。ResourceLimits 的 CPU/内存/timeout 未成为这些 in-process 核心计算的硬限额。[py/lib.rs:2983](E:/YDW/Trae_Gaussian/crates/calchem-py/src/lib.rs:2983) 对 NumPy density.as_slice().unwrap() 会在非连续数组时报 panic；结合 release panic=abort，存在整个宿主退出风险（本轮只静态确认，没有为触发崩溃执行该输入）。

绝对路径**没有彻底移除**：config/loader、OPV ChemDatas 默认、tools/dev 及 tests/conftest 仍有 C:/D:/E: 字面路径。一部分是可覆盖发现路径，不能统统叫运行必需；但 get_default_data_root 在 D: 目录不存在时回退 home/.calchem，仍可能写 C:，与本机用户规则不兼容。本轮用环境显式覆盖避免该路径。

分发有 abi3-py310 和多平台 Maturin workflow 的设计意图，但不是完整验收：

- pyproject 的 PEP 517 build-backend 仍为 setuptools，未接 Rust build hook。**实测两份独立暂存树**：fresh source 生成 `calchem_qc-0.2.0-py3-none-any.whl`，不含任何 Rust 扩展；保留 Windows .pyd 时，同名 any wheel 内含 _core.pyd，仍错误宣称 Purelib/全平台。
- CI 的 pip install -e 后 cargo build 只产出 target 库，没有把扩展安装到 package；本地脚本的复制步骤不在该 CI 里，干净 CI 的 `_core` 导入链缺失。
- release workflow 存在四目标矩阵，但无 wheel 安装后的 smoke/numerical tests，未验证 sdist 安装，没有真实发布/多平台结果证据。当前会审对象还包含大量未提交 crate。
- 无 npm/N-API/WASM 扩展生态实现，也无现成独立 calchem-cli 二进制 crate。这是未交付项，不是必须现在添加的无关功能。
- 根项目 0.2.0、Python __version__ 0.1.0、若干 backend_version 0.3.0 不统一；`all` extra 引用 calchem 而非 calchem-qc，需单独修正验证。

## 4. 【与 Gaussian 16 / ORCA 对标的公信力评价】

### 4.1 新实跑、同口径水分子能量

几何为 O(0,0,0.1173)，H(0,±0.757,−0.4692) Å；6-31G(d,p)，Cartesian 6D/10F，单重态。Gaussian UltraFine，CalChem Fine，均显式 SCF 收敛设置。各自数值网格不完全相同，差异记入结果，不假装排除所有网格效应。单位 ΔE=CalChem−Gaussian。

| 方法 | Gaussian / Eh | CalChem / Eh | ΔE / Eh | 判定 |
|---|---:|---:|---:|---|
| RHF | −76.0231361451 | −76.023136145065 | +3.49×10^-11 | 此例高度一致。 |
| SVWN5 | −75.8546696169 | −75.854669424822 | +1.92×10^-7 | 此例达到 μEh 目标。 |
| BLYP | −76.3982460302 | −76.398245798211 | +2.32×10^-7 | 此例达到 μEh 目标。 |
| PBE | −76.3327588760 | −76.332776670641 | −1.78×10^-5 | 高于本次 10^-5 Eh 严格上限，原因需进一步拆分，不能声称全部泛函已对齐。 |
| B3LYP | −76.4196224301 | −76.419622247190 | +1.83×10^-7 | 此例达到 μEh 目标。 |
| CAM-B3LYP | −76.3909806529 | −77.704658181171 | **−1.31367753** | 理论身份失败。 |
| ωB97X-D | −76.3978899787 | −75.698735931927 | **+0.69915405** | 理论身份失败。 |

补充：1 Eh≈627.509474 kcal/mol，10^-6 Eh≈0.00062751、10^-5 Eh≈0.00627509 kcal/mol；0.01 kcal/mol≈1.59×10^-5 Eh，二者不是严格同一阈值。

B3LYP 水偶极 z 分量 Gaussian −2.0298 D、CalChem −2.02982558 D；HOMO −0.29228 vs −0.292284145 Eh，LUMO 0.06755 vs 0.067554004 Eh，符合 Gaussian 打印精度。但 CAM 的偶极偏约 0.099 D；ωB97X-D 的 HOMO/LUMO 与标准值明显不符。**只有经核验的方法/结构范围可获此结论，不能概括为全部分子轨道一致。**

ORCA HF/STO-3G 与 CalChem ΔE≈3.97×10^-8 Eh；PBE/STO-3G 则有 7.75×10^-4 Eh 差异。Gaussian PBE/STO-3G 与 CalChem Fine 差约 −1.73×10^-5 Eh，表明不能把所有跨程序 PBE 差都归同一原因。ORCA 默认 B3LYP 的局域相关变体、色散和积分/网格条件须另外统一；本报告不把该默认总能量差当作 B3LYP 算法错误的证据。[ORCA 官方泛函说明](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html)

### 4.2 激发态、卤素 ECP 与光谱独立证据

| 项目 | Gaussian | CalChem | 解释 |
|---|---|---|---|
| CIS(Full)/STO-3G 水前三根 / eV | 13.1929,15.1487,16.7760 | 13.19287769,15.14874680,16.77600581 | 统一冻结核设置后符合打印精度。 |
| B3LYP full-TDDFT/STO-3G / eV | 11.6647,13.9716,14.5121 | 11.66470952,13.97164646,14.51207585 | 真实响应支持；相应 f 在打印精度内一致。 |
| HBr HF/LANL2DZ ΔE | — | +6.40×10^-7 Eh | 对卤化/ECP 路径的正面证据。 |
| HBr B3LYP/LANL2DZ ΔE | — | −4.18×10^-8 Eh | 同上；不能覆盖全部 Br/I 化学环境。 |
| 水 HF/STO-3G 频率最大差 | — | 约 1.2×10^-4 cm^-1 | 支持梯度差分 Hessian。 |
| 同一水 IR 强度 | 7.2374,44.2880,29.9732 | 0.313705,1.919661,1.299185 | 单位因子失败，约差 23.07 倍。 |

### 4.3 OPV 真实结构与实际执行结果

本轮从 ChemDatas 读取并保存文件 SHA-256，核对 IC-2F 22 原子、Donor-BDD1 32 原子、Acceptor-NC1 28 原子、界面簇 60 原子。完整 PM6(158)、BTP(159)、Y6(199) 只做来源/组成核验，没有假称运行了其全体系 SCF。

项目已有 benchmark 的 B3LYP/CAM/ωB97X-D 单点、60 原子簇及两单体的结合能、两种加工溶剂和 IC-2F TDA 路径实际运行，原始结果保存于 evidence/opv-reproduction.jsonl；不是直接照抄历史 benchmark_opv_pv.json。

另外直接调用 `OPVWorkflow.align_donor_acceptor`，重新执行两次 native SCF，耗时 81.73 s，双方 converged=true。B3LYP/STO-3G 下返回 ΔLUMO=−2.664079 eV、ΔHOMO=−1.565048 eV、Scharber estimated Voc≈5.076333 V、charge_transfer_feasible=false。原始证据为 evidence/opv-alignment.json。这些是当前片段、几何和启发式模型的实际输出，不能作为完整 PM6:Y6 器件的 Voc 或激子解离结论。

| 对象/配置 | 本次输出 |
|---|---:|
| IC-2F B3LYP/STO-3G | −870.541692477184 Eh；KS gap≈3.811285 eV |
| Donor-BDD1 B3LYP/STO-3G | −2378.695719032062 Eh；KS gap≈2.712255 eV |
| IC-2F 标为 CAM/STO-3G | −886.392360228882 Eh；gap≈5.789488 eV（模型身份不成立） |
| 60 原子簇 标为 CAM-D3BJ/STO-3G | −3451.369479363580 Eh；单点约 285.68 s |
| 簇−单体相互作用能 | −49.176801 kcal/mol（未作 counterpoise，不可认定真实标准 CAM-D3 结合能） |
| IC-2F 标为 CAM/TDA 的 S1 | 3.691513 eV；f≈4.93×10^-6；335.86 nm（模型身份不成立） |
| 同一 IC-2F 最强亮态 S2 | 4.727425 eV；f≈0.257218；262.27 nm；不能将该波长误配给 S1 |
| 氯苯 / 氯仿的 Donor-BDD1 ΔG | −4.799295 / −7.166008 kcal/mol（当前简化介质定义） |
| 氯苯 / 氯仿的 IC-2F ΔG | +9.086693 / +6.802315 kcal/mol（当前简化介质定义） |

这些数值**证明了路径可运行，不证明发表级正确性**。原始 Gaussian IC-2F 是 wB97X-D/6-31G(d,p)，60 原子簇参考是 B3LYP/6-31G(d) 并含 SMD 氯仿；不能与气相/不同方法的 STO-3G 结果用总能误差、相对百分比或“接近”做精度认证。D3/LRC 错误会直接污染此处结果，最小基组与 BSSE 也限制结合能解释。完整四点重组能、大分子结构优化、真实全 Y6/PM6 高阶基组与 CT 激发态认证并未在本轮完成。

### 4.4 基准套件的公信力与可使用范围

自带 Rust 91/91、Python unit 96/96 是当前可复现工程基线，**不是覆盖率**。部分新测试仅检查能量为负、gap 为正、频率在宽区间或自身有限差分；水 DFT 有 1.5 mEh 容差，OPV 有 10^-3 Eh 回归容差，远未覆盖宣称 μEh 的方法身份和单位。

完整 CI 不能称为全绿：按当前 CI 的 Ruff E/F/W/I/B 规则实跑，退出 1，共 848 项（以空白/长行、未使用导入、排序和未使用变量为主）；其中 F821/F822/F823 未见命中。这些风格/静态问题与上面的数值错误分开计，不以 lint 项数衡量物理正确性。

[benchmark_multi_program.py:111](E:/YDW/Trae_Gaussian/tools/dev/benchmark_multi_program.py:111) 的 Gaussian 参考能量、时间 1.80/14.60 秒是常量；参考能量可合理用于回归，本轮部分已独立证实，但用历史时间除当前 CalChem 时间不能给出可归因加速比。脚本还混排 REST 不同基组/色散/优化几何记录，以 bridge 文件存在打印集成 OK，并在缺少部分结果时照常打印 Completed Successfully。ORCA 共享 run 目录和追加既有 bench.out 的设计有陈旧输出污染风险。应改为同输入、同精度、明确失败断言、原始输出哈希、版本/硬件/线程/计时边界齐全的 benchmark。

**最终使用意见：**

- 允许定位为研究开发原型，并在本轮已核验的小分子 RHF、SVWN5/BLYP/B3LYP、有限 ECP、气相解析梯度、闭壳层 CIS/B3LYP TDDFT 范围内做受控复现与开发；新化学空间仍需独立验证。
- **不批准作为已验证通用量化引擎、标准 CAM/ωB97X-D/D3/SMD 实现，或独立支撑论文主结论和工业 OPV 高通量筛选。** 不能仅凭全绿测试、60 原子能跑或单个水分子对齐背书全部能力。
- 下一轮必须先修方法身份与溶剂能量/导数，再修 TS/IRC、IR 和发布链；随后建立同输入外部 golden 数据与 CI 硬断言，再谈性能和大体系推广。本轮只审计，不擅自改算法。

### 4.5 明确未验证的范围

未运行新的 REST/PySCF 参考（当前解释器无 PySCF，未安装）；未做 Linux/macOS 实机、多平台 wheel 安装/发布、GPU、显式 SIMD 性能对照、持续内存泄漏/竞态检测、所有 ECP/自旋/异常输入的形式化证明。未将旧审计或旧 benchmark 的结果充作当前实测。执行命令、数值汇总和复现边界见 04_TEST_RESULTS.md；所有环境/探针失误已在 05_OPERATIONS_AND_LESSONS.md 单独列明。
