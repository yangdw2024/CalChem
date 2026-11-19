# CalChem 量子化学计算引擎：全功能能力矩阵 (Functional Capability Matrix)

**版本基准**：CalChem Master Engineering Mission v1.0 认证版  
**更新日期**：2026-11-11  
**硬件执行环境**：AMD Ryzen 7 9700X (8 核 16 线程, 3.8~5.5 GHz), 16 GB DDR5 内存, NVIDIA GeForce RTX 5060 Ti (8 GB VRAM, 5.9 GB 计算安全上限), NVCC 13.3.73, Windows 11 x64  
**底层语言**：100% 纯 Rust 核心算子 + C/CUDA 原生扩展 (`calchem-cuda`) + PyO3 Python 原生绑定 (`calchem._core`)

---

## 一、 电子结构与波函数方法支持矩阵 (Electronic Structure & Wavefunction)

| 计算方法 | 形式体系 | 积分计算模式 (Integral Engine) | 求解加速器 | 解析核梯度 (Gradients) | 几何优化 / 频率 | 状态 | 规范与误差界 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **RHF** | 闭壳层 Hartree-Fock | Direct SCF / Stored ERI / RI-J | DIIS (自适应阻尼) | 原生支持 | 原生支持 | **完全就绪** | 能量误差 $\le 10^{-10}\text{ Eh}$ 对齐 Gaussian 16 |
| **UHF** | 开壳层非限制性 HF | Direct SCF / Stored ERI | DIIS (自适应阻尼) | 原生支持 | 原生支持 | **完全就绪** | 自旋污染分析 $\langle S^2 \rangle$ 原生输出 |
| **ROHF** | 开壳层限制性 HF | — | — | — | — | **显式保护** | 严格阻断静默错误，推荐原生 UHF |
| **MP2** | 二阶二体微扰理论 | 轨道变换两电子积分 | 快速半变换 | 原生有限差分 | 原生支持 | **完全就绪** | 与 Gaussian 16 `MP2` 相关能误差 $\le 10^{-8}\text{ Eh}$ |
| **CCSD** | 单双激发耦合簇 | $T_1/T_2$ 振幅振荡迭代 | DIIS 振幅加速 | 有限差分 | 原生支持 | **完全就绪** | 水分子微扰对齐权威基准 |
| **CCSD(T)** | 包含三激发非迭代校正 | $(T)$ 微扰解析张量收缩 | 局部内存流 | 有限差分 | 原生支持 | **完全就绪** | 达到量化“黄金标准”精度 |

---

## 二、 密度泛函理论 (DFT) 泛函支持矩阵

CalChem 原生实现了分子三维贝克重叠网格（Becke Partitioning Quadrature）与 Treutler-Ahlrichs 径向格点，支持多种格点密度（`Coarse`, `Medium`, `Fine`, `Ultrafine`）。

| 泛函类型 | 泛函标识 (Aliases) | 近似阶次 | 交换相关组成 | 交换积分核心 | 激发态 (TD-DFT) | 解析梯度 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **LDA** | `SVWN`, `SVWN5`, `LDA` | 局域密度 (LDA) | Slater 交换 + VWN5 相关 | 无精确交换 (Pure DFT) | 支持 | 原生解析 |
| **GGA** | `BLYP` | 广义梯度 (GGA) | Becke88 交换 + LYP 相关 | 无精确交换 (Pure DFT) | 支持 | 原生解析 |
| **GGA** | `PBE`, `PBEPBE` | 广义梯度 (GGA) | PBE 交换 + PBE 相关 | 无精确交换 (Pure DFT) | 支持 | 原生解析 |
| **Hybrid GGA** | `B3LYP`, `B3LYP5` | 杂化泛函 (20% HF) | Becke88 + 20% Exact HF + LYP + VWN | 四中心 Direct / RI-J | 原生解析 | 原生解析 |
| **Hybrid GGA** | `PBE0`, `PBE1PBE` | 杂化泛函 (25% HF) | PBE + 25% Exact HF + PBE 相关 | 四中心 Direct / RI-J | 原生解析 | 原生解析 |
| **Range-Separated** | `CAM-B3LYP` | 范围分离杂化泛函 | 短程 $\alpha=0.19$，长程 $\alpha+\beta=0.65$，$\omega=0.33$ | 衰减核 $\text{erf}(\omega r_{12})/r_{12}$ | 原生解析 | 原生解析 |
| **Range-Separated** | `wB97X-D`, `wB97X` | 范围分离杂化泛函 | 短程精确交换 + 长程 100% HF ($\omega=0.20$) | 衰减核 $\text{erf}(\omega r_{12})/r_{12}$ | 原生解析 | 原生解析 |

---

## 三、 双电子积分与加速引擎架构矩阵 (Integral Engines)

| 积分引擎 | 算法原理 | 存储复杂度 | 时间复杂度 | 硬件加速 | 最大承载基函数量级 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stored ERI** | 预计算并内存缓存全部 $(\mu\nu\|\lambda\sigma)$ | $O(N_{\text{ao}}^4)$ | $O(1)$ 每轮 SCF | CPU AVX2 | 适用 $N_{\text{ao}} \le 200$，小分子极速计算 |
| **Direct SCF** | Schwarz $L_1$ 筛选 + Rayon 任务窃取动态缩并 | $O(N_{\text{worker}} \times N_{\text{ao}}^2)$ | $O(N_{\text{ao}}^2) \sim O(N_{\text{ao}}^3)$ | CPU Rayon 多线程 | 突破 6000+ $N_{\text{ao}}$ 内存墙，内存消耗 $\le 5\text{ GB}$ |
| **RI-J Density Fitting** | 辅助基组分解 $(P\|Q)^{-1}$，三中心拟合库仑 | $O(N_{\text{aux}} \times N_{\text{ao}})$ | $O(N_{\text{aux}} N_{\text{ao}}^2) \approx O(N^3)$ | CPU / GPU 原生 | 纯库仑构建提速 5~15 倍，大体系极速响应 |
| **CUDA Acceleration** | 双精度原生核函数，异步流式调度与显存守卫 | 受限 GPU 显存缓冲 | $O(N)$ 并行映射 | NVIDIA RTX 5060 Ti | 显存安全上限 5.0 GB，无缝 CPU 回退 |

---

## 四、 激发态与物性光谱支持矩阵 (Spectroscopy & Properties)

| 功能模块 | 计算物理理论 | 求解器与算法 | 核心输出物理量 | 认证基准 |
| :--- | :--- | :--- | :--- | :--- |
| **TD-DFT (TDA)** | Tamm-Dancoff 近似 | Davidson 对角化 / 预处理共轭迭代 | 激发能、跃迁偶极矩、振子强度 | 水分子、甲醛、乙烯基准对齐 |
| **TD-DFT (Full)** | 包含去激发项的 RPA 方程 | 非厄米辛对称 Davidson 求解器 | 单重态激发态光谱、吸收截面 | UV-Vis 吸收光谱、跃迁轨道贡献 |
| **光伏激子分析** | 激子结合能评估 | $E_b = E_{\text{opt}} - (E_{\text{HOMO}} - E_{\text{LUMO}})$ | 激子结合能 $E_b$、CT 电荷转移态能级 | IC-2F、Y6 光伏受体电荷分离能力分析 |
| **静电势 (ESP)** | 电子密度与核库仑场卷积 | 零拷贝内存流式网格计算 | ESP 表面映射图、极性分子偶极矩 | C-F、C=O 强极性基团电荷分布分析 |
| **布居数分析** | Mulliken 与 Lowdin 正交布居 | 密度矩阵与重叠积分正交投影 | 原生原子净电荷、自旋多重度分布 | 自由基阳离子自旋密度分布流 |

---

## 五、 结构优化与反应动力学矩阵 (Optimization & Kinetics)

| 计算功能 | 物理实现 | 收敛判定判据 | 特殊边界保护 |
| :--- | :--- | :--- | :--- |
| **平衡几何优化 (Opt)** | L-BFGS / 拟牛顿法 (BFGS) | 最大受力 $\le 10^{-4}\text{ a.u.}$，位移 $\le 10^{-3}\text{ Bohr}$ | 支持笛卡尔与冗余内坐标，自适应步长控制 |
| **过渡态搜索 (TS Search)** | 鞍点优化 (Partitioned Rational Function) | 沿负曲率方向逆向爬坡，力收敛 $\le 3 \times 10^{-4}$ | 严格要求且验证单虚频振动模 |
| **内禀反应坐标 (IRC)** | Gonzalez-Schlegel 变步长阻尼弧长投影 | 沿过渡态虚频前后双向追踪能量极小点 | 正反向反应能垒 $\Delta E^\ddagger$、反应热 $\Delta H_{rxn}$ 闭环 |
| **振动频率与热力学 (Freq)** | 准简谐二阶 Hessian 投影质心与转动模 | 零点能 (ZPE)、焓变 $H$、自由能 $G$、定容热容 $C_v$ | 自动识别线性分子 (5 自由度) 与非线性分子 (6 自由度) |

---

## 六、 溶剂化模型与色散校正矩阵 (Solvation & Dispersion)

| 模型类别 | 具体实现 | 介电参数与参数化 | 作用机制 | 解析梯度支持 |
| :--- | :--- | :--- | :--- | :---: |
| **C-PCM** | 导体极化连续介质模型 | 溶剂介电常数 $\epsilon$（水、氯仿、氯苯等） | 边界电荷密度诱导极化 | 原生支持 |
| **SMD** | 基于连续介质的全溶剂自由能模型 | 包含空穴形成、色散与结构重排非静电项 | 准确预测溶剂化自由能 $\Delta G_{\text{solv}}$ | 原生支持 |
| **DFT-D3(Zero)** | Grimme D3 零阻尼色散校正 | 经验色散参数 $s_6, s_8, r_{0,ab}$ | 修正中长程范德华相互作用 | 原生支持 |
| **DFT-D3(BJ)** | Becke-Johnson 有理阻尼色散校正 | 消除短程非物理发散，平滑阻尼过渡 | 精确捕捉有机光伏大分子 $\pi$-$\pi$ 堆积 | 原生支持 |

---

## 七、 基组与有效核势 (Basis Sets & ECP)

* **全电子高斯基组 (Pople & Dunning)**：
  * 极小基：`STO-3G`
  * 劈裂价层基组：`3-21G`, `6-31G`, `6-31G(d)`, `6-31G(d,p)`, `6-311G(d)`, `6-311G(d,p)`
  * 极化弥散基组：`6-31+G(d)`, `6-311++G(d,p)`
  * 相关一致基组：`cc-pVDZ`, `cc-pVTZ`
* **有效核势 (ECP / Pseudopotentials)**：
  * `LANL2DZ`：支持重主族元素及过渡金属（Br, I, Cu, Ag, Au 等），消除相对论深核电子计算负担。
* **辅助密度拟合基组 (Auxiliary Fitting Basis)**：
  * 内置自动辅助基组生成算法（根据轨道基函数角动量与指数动态生成 $l_{\text{aux}} \le 2 l_{\text{orb}}$ 拟合高斯球）。
