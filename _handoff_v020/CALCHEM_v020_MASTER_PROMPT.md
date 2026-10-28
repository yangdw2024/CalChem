# CalChem v0.2.0 → v1.0 下一步规划提示词

你是 CalChem 的架构师。这是一个统一量子化学计算平台，不是 Gaussian 克隆：
- 自带原生 Python 引擎（RHF/UHF/DFT/MP2/CCSD/CCSD(T)）
- 通过适配层接入 Gaussian 16 / ORCA / PySCF / xTB
- PyQt6 GUI + Click CLI + Python API
- Windows 11，AMD Ryzen 7 9700X，15GB RAM，严禁并行重计算

## 当前状态

v0.1.0 已完成并打包（PyInstaller 238MB one-folder）。三后端 H2O/STO-3G RHF 交叉验证到 1e-8 Eh。
v0.2.0 P0 已部分修复（见 handoff/PROGRESS_v020.md）：
- 去掉了 native 后端 energy=0/converged=True 的假值默认
- ORCA 偶极不再伪造 z 矢量
- 新建了 JobDispatcher（单并发后台线程消费队列）
- CLI --json 模式 stdout 干净可解析

但 P0 还有未完成项：Gaussian 硬编码路径、GUI 未接入队列、external 超时未清理、Request 输入验证缺失、tests 分层缺失。

## 你的任务

基于附件中的：
1. 现有源码（source/calchem/）
2. v0.1.0 数值基线（BASELINE.md）
3. v0.1.0 验收报告（RELEASE_ACCEPTANCE_v010.md）
4. v0.2.0 修复进展（handoff/PROGRESS_v020.md）

请给出：

1. **剩余 P0 项的优先级排序和文件级修复计划**（哪些必须先做，哪些可以并行）
2. **v0.3.0（后端配置与 cclib 集成）的具体模块任务**——精确到文件名、函数签名、验收标准
3. **v0.4.0（几何优化）的最小可行实现路径**——native 梯度用有限差分起步，还是等解析梯度？Gaussian/ORCA 优化如何接入？
4. **v0.5.0（频率）的范围**——哪些方法可以开放频率，Hessian 怎么算
5. **风险评估**：哪些功能最容易产生"假通过"（看起来跑了但科学上错了），如何防范
6. **v1.0 发布的最小能力集**——哪些是必须的，哪些可以推迟到 v1.1+

要求：
- 每个任务给出：文件路径、改动内容、验收方法、预计复杂度
- 不要给宏观建议，要给可直接执行的文件级计划
- 明确标注哪些可以在 Windows 上串行完成，哪些需要外部许可
- 不要假设 handoff zip 里有原生计算内核（scf/、integrals/、post_hf/）——本地仓库有但没打包进来
