# CalChem 新一轮科研认证独立复核（2026-11-16）

## 总结

这是一次有实质进展的交付：当前二进制哈希匹配，Rust/Python 全量回归日志完整；Gaussian 16 的原始输入和正常结束的输出存在，GPU RI-J 调用计数已经贯通 IC-2F 作业。结论应限定为**若干小体系基态方法的外部数值对照与单机 GPU 正确性**。名称中的“最终科研认证”超过现有证据；大体系与 RI-J 高精度仍未通过。

## 已核实的原始证据

- `calchem/_core.pyd` 和 `build_target/release/_core.dll` 的 SHA-256 都是 `37F96B1A58C64F86DA8ED0A997BDCA2386A04635602E070D5FAA151E32A16F8C`。
- 本轮归档 `cargo_full.exit.txt=0`，逐个测试组求和为 **140 passed、0 failed、0 ignored**；`pytest_full.exit.txt=0`，末行 **120 passed / 995.97 s**；`cuda_full.log` 合计 **5 passed**。这些是当前交付的实测回归，不再引用上一版 139/119。
- 外部比较脚本中的 12 个 Gaussian 数值作业均在 `build_target/runtime/g16_cert_20261116/<case>/input.gjf` 和 `input.out` 留存，并有 `Normal termination of Gaussian`；当前目录共 13 份正常结束的输出，其中一份是清单外的 H2O_RHF_RIJ_STO3G。CSV 有 **12 条数值行 + 1 条 wB97X-D 拒绝行**，报告“11 个算例”计数不符。
- 当前 IC-2F CSV 记录 CPU kernel=0、GPU kernel=26、GPU fallback=0；源码在 GPU FFI 成功返回时递增计数，并经 ScfDiagnostics/PyO3/NativeCalchem/OPVResult 透传。该证据比仅传 `use_gpu=True` 更强，可支持本机本次作业的 GPU J 缩并确实成功。
- 效率测试明确给出 CPU 126.7951 s、GPU 127.6530 s、比值 0.993；**没有端到端提速**。该单次顺序计时还缺重复、误差带和统一 GPU/CPU 热启动条件，不能据此量化稳定性能差。

## 需要纠正的科学口径

1. **Gaussian 与 CalChem 的 DFT 格点没有设成同一规则。** Gaussian 输入使用 `Integral=UltraFine`；CalChem 的 `GridLevel::from_name("ultrafine")` 实际映射 `Fine`，自身径向 55/75/90（Fine 时另有至少 150 径向壳层）及最高 302 点 Lebedev 角向规则。两边积分网格并非同一节点/权重/剪枝。因此“同几何、同基组、同网格”中前两项可从脚本核实，第三项不成立。DFT 能量/梯度差可能混合网格与方法实现差，不能单独归因于泛函。
2. **B3LYP 注释写反。** CSV 的 B3LYP/CAM 注释称“Gaussian B3LYP uses VWN3 while CalChem uses VWN5”；当前 `xc.rs` 中 CalChem `B3lyp` 用 `VWN_RPA_PARAMS`，`B3lyp5` 才用 `VWN5_PARAMS`，`CamB3lyp` 用 VWN5。应按每个方法分别标注，不要把 B3LYP 当成 B3LYP5。
3. **DFT 对照仍有可见差异。** PBE0/6-31G(d) 水分子总能差 `-1.517e-5 Eh`、最大核梯度分量差 `1.131e-5 Eh/Bohr`；B3LYP/CAM 梯度差约 `3.57e-6` 至 `4.50e-6 Eh/Bohr`。这些是有价值的基准，不是“所有泛函高精度完全一致”。应先对齐网格或做双方网格收敛外推，再查余差。
4. **自动 RI-J 辅助基误差是明显短板。** 在同一 CalChem B3LYP/6-31G(d) 水分子，RI-J 与精确四中心库仑能量差 `-0.0081003068 Eh`（约 5.08 kcal/mol）；相对 Gaussian 精确库仑也是 `-0.008101 Eh`。H2O/STO-3G RI-J 差 `-4.915e-4 Eh`。因此这套 auto-jfit 不能按化学精度或定量 OPV 结果使用，哪怕 CPU/GPU 两条 RI 路径彼此吻合。
5. **轨道能比较精度受原始输出限制。** CSV 中 Gaussian HOMO/LUMO 仅保留 6 位小数，不能从几微 Hartree 的差值推导更高精度的轨道认证。

## 工程整改核查

- **DIIS：主要缺陷已闭环。** RHF/RKS/UHF/UKS 与 SCRF 调用 `try_push_and_extrapolate` 并映射 I/O 错误；有活实例只读文件故障注入、SCF 入口坏目录异常传递和 Drop 清理测试。此前“SCF 吞掉磁盘错误”不再适用于当前源码。该测试证明被注入的路径；磁盘满、进程异常退出、并发/长期任务和文件系统异常仍需单独压力验证。
- **RI-J 秩截断：边界保护已闭环。** 测试用两份真实重复 s 辅助壳层触发截断，并验证梯度拒绝。它证明不会在此边界返回错误梯度；不意味着截断辅助基上的解析梯度已经实现。
- **GPU：本机小体系核函数及 IC-2F 调用遥测已证实。** 仅 J 缩并上 GPU，三中心张量生成、精确交换 K、DFT 网格和其他瓶颈仍在 CPU。16 GB/726 原子与 GPU 加速目标未由这些数据证明。
- **复现脚本尚需加固。** `reproduce_certification.ps1` 逐项启动子 PowerShell/Cargo/Python，但没有逐步检查 `$LASTEXITCODE`；在 Windows PowerShell 下 `$ErrorActionPreference="Stop"` 不保证外部进程非零退出时脚本停止，末尾可能仍显示 completed successfully。还没有把源码快照哈希与 12 个 Gaussian 原始输出哈希写入机器可核验 manifest。

## 实际成熟度

对 RHF 小体系，所列同几何/基组外部对照很有说服力：最大总能差约 `5.53e-8 Eh`，梯度最大分量差约 `1.51e-8 Eh/Bohr`。DFT 小体系已达到可继续精化的阶段，但“同网格”前提未满足。RI-J 的自动辅助基在水/6-31G(d) 上出现 8.1 mEh 偏差，是定量使用的阻断项。GPU 算子正确、遥测真实，却没有端到端速度收益。BTP/Y6/PM6/726 原子仍无原生完整 SCF 收敛和峰值内存实测。当前应定位为**研究开发级，部分小体系方法得到独立外部基准支持**；不能宣称通用科研认证或工业级 OPV 生产引擎。

## 本次操作

只读当前源码、CSV、测试日志与 Gaussian 原始输入输出；新建本审查文档。未修改业务源码，未重跑已有完整测试，未安装或清理文件。
