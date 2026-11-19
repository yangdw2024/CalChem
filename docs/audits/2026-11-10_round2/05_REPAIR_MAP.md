# 修复映射与完整替换代码入口

所有路径指向最终磁盘版本；行号从实际源码自动提取。每项完整替换代码在 [06_DROPIN_CODE.md](06_DROPIN_CODE.md)，配套数据/文件在 replacements.zip。回归失败的早期日志另存，不替代下表最终证据。

| 缺陷 | 目标文件与精确入口 | Replace 的实际内容与数学原因 | 回归证据 |
|---|---|---|---|
| IR 单位与响应精度 | [crates/calchem-vib/src/masses.rs:27](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/masses.rs:27) | 完整单位系数正确保留；Hessian/APT 自动收紧 SCF 以控制差分噪声 | [acceptance-delivery.jsonl](evidence/acceptance-delivery.jsonl) |
| 软模式与 Eckart 子空间 | [crates/calchem-vib/src/modes.rs:282](E:/YDW/Trae_Gaussian/crates/calchem-vib/src/modes.rs:282) | 在刚体运动正交补中对角化，避免按最小绝对本征值误删真实软模式 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| 新鲜 Hessian TS 认证 | [crates/calchem-opt/src/prfo.rs:337](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/prfo.rs:337) | 终点重新计算 Hessian；恰好一个分辨负模式；失败传播，无后门认证 | [ecp-ts-delivery.jsonl](evidence/ecp-ts-delivery.jsonl) |
| IRC 终点与步长 | [crates/calchem-opt/src/irc.rs:204](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/irc.rs:204) | 终点 E/g 重新评估；未满足约束减半步长；未完成分支不产生最终势垒 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| IRC 起点与模式 | [crates/calchem-opt/src/irc.rs:372](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/irc.rs:372) | 起点驻点/负模式认证及启动向量投影检查；H2 极小值拒绝 | [pytest-delivery.log](evidence/pytest-delivery.log) |
| IRC PyO3 字段丢失 | [crates/calchem-py/src/lib.rs:255](E:/YDW/Trae_Gaussian/crates/calchem-py/src/lib.rs:255) | 显式导出完成标志、可空的端点势垒/反应能，同步 stub 和 Python 消费端 | [bridge-contracts.log](evidence/bridge-contracts.log) |
| Boys 高阶/高 T | [crates/calchem-int/src/boys.rs:109](E:/YDW/Trae_Gaussian/crates/calchem-int/src/boys.rs:109) | 正项 Kummer、补偿和、适用阶数递推；移除高 T 无条件无穷上限替代 | [boys-certified.jsonl](evidence/boys-certified.jsonl) |
| 标准 D3 C6/CN | [crates/calchem-dft/src/d3.rs:67](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/d3.rs:67) | 原子对参考态插值与 r4/r2 C8；完整 CN 链式解析导数；标准参数 | [acceptance-delivery.jsonl](evidence/acceptance-delivery.jsonl) |
| Python D3 回退与参数 | [calchem/dft/d3.py:11](E:/YDW/Trae_Gaussian/calchem/dft/d3.py:11) | 取消近似 Python 数值回退；未知模型和未支持自定义参数明确失败 | [pytest-delivery.log](evidence/pytest-delivery.log) |
| 独立气相失败吞异常 | [crates/calchem-solv/src/scrf.rs:320](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/scrf.rs:320) | 独立未极化气相 SCF 成功才返回 ΔG，错误直接传播 | [acceptance-delivery.jsonl](evidence/acceptance-delivery.jsonl) |
| 溶剂总导数口径 | [crates/calchem-opt/src/lib.rs:169](E:/YDW/Trae_Gaussian/crates/calchem-opt/src/lib.rs:169) | 重新收敛完整 SCRF 总能量做中心差分；CDS 与能量使用相同定义 | [core-closed-loop.log](evidence/core-closed-loop.log) |
| 伪解析 PCM 梯度 | [crates/calchem-solv/src/gradient.rs:91](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/gradient.rs:91) | 拒绝未实现解析能力；保留明确标注的 fixed-density partial FD | [02_ACCEPTANCE_AND_REPAIR_REPORT.md](02_ACCEPTANCE_AND_REPAIR_REPORT.md) |
| 开壳层后修正冒充 SCRF | [crates/calchem-solv/src/scrf.rs:590](E:/YDW/Trae_Gaussian/crates/calchem-solv/src/scrf.rs:590) | 阻断仅气相 UHF/UKS 加后修正的自洽身份；方法与 XC 冲突也拒绝 | [pytest-delivery.log](evidence/pytest-delivery.log) |
| 网格半径截断 | [crates/calchem-dft/src/grid.rs:426](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/grid.rs:426) | 不再固定 rmax；按最小基组指数增加径向节点并用于 SCF/梯度 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| DIIS 小残差奇异 | [crates/calchem-scf/src/diis.rs:87](E:/YDW/Trae_Gaussian/crates/calchem-scf/src/diis.rs:87) | 统一缩放 B 子块，系数尺度不变；约束行保持原样 | [diis-regression.log](evidence/diis-regression.log) |
| 最终 SCF 假平稳 | [crates/calchem-scf/src/uhf.rs:272](E:/YDW/Trae_Gaussian/crates/calchem-scf/src/uhf.rs:272) | 堆叠 alpha/beta polishing；最终实际残差决定接受，不靠旧残差最小值 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| 壳层四元组预分配 | [crates/calchem-int/src/symmetry.rs:8](E:/YDW/Trae_Gaussian/crates/calchem-int/src/symmetry.rs:8) | 惰性常数辅助存储枚举，Rayon 分发不预存完整四元组空间 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| 索引/容量/非法迭代 | [crates/calchem-scf/src/lib.rs:15](E:/YDW/Trae_Gaussian/crates/calchem-scf/src/lib.rs:15) | 分配/建网格前检查保守储存界与 u16 索引、容差和迭代预算 | [cargo-delivery.log](evidence/cargo-delivery.log) |
| LRC 方法身份 | [crates/calchem-dft/src/xc.rs:79](E:/YDW/Trae_Gaussian/crates/calchem-dft/src/xc.rs:79) | 正式名称与 Rust 旧 enum SCF 路径均拒绝，直到实现 range-separated ERI | [pytest-delivery.log](evidence/pytest-delivery.log) |
| wheel 旧二进制复用 | [setup.py:22](E:/YDW/Trae_Gaussian/setup.py:22) | 每次构建调用 Cargo JSON；只复制真实 cdylib；标准 Extension/abi3 标签 | [package-delivery-contracts.jsonl](evidence/package-delivery-contracts.jsonl) |

## 替换方式

1. 使用 final_manifest.json 核对当前基线；保留已有修改。
2. 在新的 D/E 候选目录应用完整文件，不使用 reset/clean。
3. D3 必须同时放置 d3.rs、d3_data.rs、data/d3/c6.bin、原始数据和许可证；Python wheel 也需要随附 calchem/data/d3。
4. 重新构建 PyO3 扩展，再按 03_RESULTS_AND_REPRODUCTION.md 复验；不要假设正在运行的 Python 会热重载新二进制。

跨模块修复涉及 imports、Result 错误契约与调用者；完整文件比只粘贴单个公式更能避免漏改。changes.patch 提供文本审阅，二进制文件通过 replacements.zip 和 SHA256 校验。
