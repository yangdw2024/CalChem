# REST 架构与 CalChem / Gaussian 性能复核

## 4 线程实测

两个算例使用相同坐标和 STO-3G。Gaussian 原始输出为 4 线程、2 GB、Int=Coarse，显示 RMS 密度阈值 1e-8、最大密度阈值 1e-6、能量阈值 1e-6 Eh。CalChem 本轮使用 4 线程、--conv-d 1e-8、--conv-e 1e-6、--max-iter 128。

| 作业 | Gaussian 原作业墙钟及 SCF 步数 | CalChem RI-J + GPU 墙钟及 SCF 步数 | CalChem 进程峰值工作集 |
|---|---:|---:|---:|
| PTQ10 B3LYP | 16.81 s，14 步 | 39.15 s，26 步 | 1.07 GiB |
| BTP-4F PBE | 107.10 s，20 步 | 136.58 s，53 步 | 3.40 GiB |

两个新作业均收敛，GPU 回退次数为 0。PTQ10 CPU RI-J 在相同 CalChem 设置下为 39.84 s、26 步、总能量 −1849.309081084224 Eh，GPU 为 −1849.309081084231 Eh。CPU 和 GPU 的这次墙钟差别不足 1 秒，属于单次观察。GPU 加速 RI-J 收缩，PTQ10 的精确交换在 CPU 计算。

两种 Coarse 网格的节点及剪枝规则尚未对齐。Gaussian 库仑路径与 CalChem 自动 RI-J 的辅助基和近似不同。CalChem 增加交换子判据和固定点检查，最大密度变化判据也没有直接等价对应。Gaussian 2 GB 是配置的内存参数，CalChem 峰值工作集是进程观察值且使用 GPU。阈值数值相同也不保证指标计算定义一致。结果反映线程及部分阈值对齐后的性能，完整物理定义、收敛条件、内存和 GPU 资源仍有差异。

Gaussian 输出在 `docs/audits/2026-11-16_delivery_check/ydw_ptq10_g16/input.out` 和 `docs/audits/2026-11-16_delivery_check/ydw_btp4f_g16/input.out`。Gaussian 墙钟为上轮记录，输出自身 elapsed 为 16.5 和 106.8 秒。新运行命令、输入与二进制哈希在本目录 JSON 文件。每组为单次测量。

## 分段开销

| 4 线程 CalChem 分段 | PTQ10 B3LYP | BTP-4F PBE |
|---|---:|---:|
| Fock 构建器初始化 | 4.44 s | 51.82 s |
| 主 SCF 双电子 | 26.05 s | 7.09 s |
| 主 SCF 交换相关矩阵 | 6.97 s | 55.72 s |
| 最终自洽检查 | 0.62 s | 6.64 s |

PTQ10 含精确交换，K 使用 CPU 四中心积分。BTP-4F 的主要开销是 RI-J 初始化与交换相关矩阵。`vxc.rs` 当前逐点求 AO、密度及矩阵贡献，内部包含活跃 AO 成对循环，可评估分块矩阵收缩和 AO 稀疏处理。具体收益需要 profiler 和修改前后计时验证。

SCF 在 AO 数至少 40 时对前 10 次迭代使用 0.3 阻尼，并在误差较大时继续阻尼。两程序初猜、误差定义、DIIS、格点及 Fock 算法均可能影响迭代次数，本轮无法单独归因于阻尼。应在相同最终物理 Fock、密度和能量准确度要求下，对初猜、阻尼和 DIIS 进行同程序 A/B。

## 本机 REST 源码

`E:/YDW/REST/knowledge_base/rest_source/Cargo.toml` 声明 REST 2026.1.1，依赖 rest_libcint、libxc、OpenBLAS 及张量运算库。`rest_doc/source_zh/user/input.md` 说明默认使用 RI，辅助基默认 def2-universal-JKFIT。`rest_doc/source_zh/contributor/ri-jk/` 包含 incore、direct RI-JK。具体运行功能由版本与配置决定。

成熟积分内核、泛函库、BLAS 批量线性代数与 RI-JK 可共同降低计算成本。CalChem 当前采用自动 RI-J 与精确四中心 K，编程语言本身不能决定运行速度。CalChem 项目要求独立量子化学核心，引入外部库需要考虑该目标与许可证。REST 的分块布局和调度可以独立实现。

本机 `rest_source/src/dft/fxc_matvec_performance_analysis.md` 报告单项 fxc matvec 优化，release 下 LDA 为 1.70 倍、GGA 为 2.09 倍，属于 REST 内部算子比较。没有找到针对两个体系的三方同条件原始性能输出，本次没有运行 REST。

## 后续优化与验收顺序

1. 对齐基准定义，记录版本、CPU 和 BLAS 线程、资源限制、初猜、实际网格节点、辅助基、筛选阈值及残差。用网格收敛序列和固定网格精确 J / RI-J A/B，检查总能量、相对能量和密度误差。重复运行取中位数。
2. 分别剖析 BTP-4F 的辅助基生成、二中心度量、分解、三中心张量及 GPU 传输，针对热点批处理或分块。验收固定密度 J 矩阵、总能量、近相关基保护及梯度。
3. 对 RKS 格点评估分块、AO 筛选、重复 AO 值复用和矩阵乘，约束内存。验证 XC 能、Vxc、解析梯度及密网格结果。
4. 对初猜、阻尼、DIIS 和最终残差做 A/B，记录逐步能量、密度及交换子，覆盖易不收敛体系。
5. 对杂化泛函评估四中心交换积分的壳对分批、向量化及 GPU ERI 批处理。RI-K 需要 JKFIT 辅助基和单独的误差验证。先验收固定密度 K 与完整 SCF，再比较端到端墙钟。

本轮重新计时并诊断源码，没有改动引擎。此前 16 线程 CalChem 与 4 线程 Gaussian 的速度比无法解释同资源下的算法优劣。
