# 一手资料调研与判据

检索日期：2026-11-09（Asia/Shanghai）。以原论文、算法维护者代码/文档、软件官方手册及 NIST 为依据；检索结果中的转载、聚合页、论坛和非官方镜像不作为结论依据。经典理论按原始发表版本核对，工程分发按当前官方文档核对。没有把点赞、星数或“近期收录”当成数值正确性的证据。

| 资料 | 本次用途与核对结论 |
|---|---|
| Yanai, Tew, Handy, 2004, [CAM-B3LYP 原论文](https://doi.org/10.1016/j.cplett.2004.06.011) | CAM 需要对电子间相互作用作距离分离，短程/长程 HF 交换份额分别为 0.19/0.65，分离参数 0.33；不能用全程常数 0.42 替代。 |
| [DIRAC 官方 CAM 文档](https://dirac-fork.readthedocs.io/en/latest/tutorials/dft/cam.html) | 核对 CAM 算符分解和参数，避免将半局域衰减项当成长程 HF 积分的替代。 |
| [ORCA 6.1 官方 DFT 手册](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html) | 核对全局杂化、范围分离、泛函变体。ORCA 与 Gaussian 的 B3LYP 局域相关默认变体可能不同，不能直接把默认 B3LYP 能量差归咎某一内核。 |
| [ORCA 官方数值积分说明](https://orca-manual.mpi-muelheim.mpg.de/contents/essentialelements/numericalintegration.html) | 网格大小/剪枝/最终能量与导数设置是独立的精度控制条件，必须记录；不同网格不能据此宣称绝对相同算法或精度。 |
| [Simple-DFT-D3 模型数据结构与算法](https://dftd3.github.io/simple-dftd3/module/dftd3_model.html) | 标准实现存储参考 C6、参考配位数、r4/r2 与多个参考环境，包含参考权重。CalChem 单原子基准 C6 的指数缩放及调和平均不等价。 |
| [Grimme 课题组 DFT-D3 页面](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/) | 核对标准 D3 软件及参考体系数据来源。没有安装或运行新下载的 D3。D3 数值对照来自本机 Gaussian 16 与 ORCA 新实跑。 |
| [PySCF 官方梯度 API](https://pyscf.org/pyscf_api_docs/pyscf.grad.html) | 核对 AO 基组响应、网格响应等导数项的概念边界。没有安装或运行 PySCF。 |
| [PySCF 官方溶剂模型说明](https://pyscf.org/user/solvent.html) | 介质响应与基态 SCF 的耦合应单独验证；仅有带溶剂名称的表面积能量不能建立 SMD 等价性。 |
| [NIST CODATA 常数](https://physics.nist.gov/cuu/Constants/)及[能量换算](https://physics.nist.gov/cuu/Constants/energy.html) | 使用一致单位口径：1 Eh 约 627.509474 kcal/mol；频率与惯量常数各自检查。本次以安装的 SciPy 常数作额外独立数值核算，未混称为严格 CODATA 2022 全常数复核。 |
| [Maturin 官方项目布局](https://www.maturin.rs/project_layout.html)及[官方教程](https://www.maturin.rs/tutorial.html?highlight=stable) | abi3 和 mixed project 需要正确构建/安装流程。配置声明不能代替实际 wheel 元数据及干净环境导入验证。 |

## 本次调研转化成的独立检验

1. 使用相同水分子几何、6-31G(d,p)、Cartesian 6D/10F，对 Gaussian 16 实跑 HF、SVWN5、BLYP、PBE、B3LYP、CAM-B3LYP、wB97X-D。
2. 对 CIS 先发现 Gaussian 默认冻结一个 O 1s 轨道，随后显式 CIS(Full) 统一活性空间；不将默认冻结核差异判成 CalChem 算法错误。
3. 在相同 HBr 几何下验证 LANL2DZ 的 HF 与 B3LYP ECP 能量。
4. 比较标准 Gaussian/ORCA D3(BJ) 色散项本身，不把 ORCA 默认 B3LYP 变体与 CalChem B3LYP 总能量差混入 D3 误差。
5. 在同一 HF/STO-3G 水分子结构上分别比较频率、IR 强度、ZPE，避免“频率准确”掩盖强度单位错误。

原始输入/完整输出：E:\YDW\Trae_Gaussian\build_target\runtime\redteam_20261109\external_alignment、external_followup、external_extra。
