# CalChem v0.3.0 → v1.0 完整开发规划

你是 CalChem 的首席架构师。这是统一量子化学计算平台，不是 Gaussian 克隆：
- 原生 Python 引擎：RHF/UHF/DFT/MP2/CCSD/CCSD(T)
- 适配层：Gaussian 16 / ORCA / PySCF / xTB
- PyQt6 GUI + Click CLI + Python API
- Windows 11，AMD Ryzen 7 9700X，15GB RAM，严禁并行重计算

## 当前状态（v0.2.0 已完成）

v0.2.0 全部 P0 修复完成：
- 无假值（energy/converged 严格验证）
- JobDispatcher 单并发队列
- external 超时进程树清理
- 配置系统（自动检测 + 文件 + 环境变量）
- CLI --json 干净
- 22 unit tests pass

附件包含**完整源码**（含原生计算内核 scf/、integrals/、post_hf/、dft/、basis/）。

## 你的任务

请给出 v0.3.0 到 v1.0 的**文件级开发计划**，包括：

### 1. v0.3.0 后端稳定
- cclib 集成：哪些字段用 cclib，哪些保留手写 parser？
- PySCF/xTB 真实适配的最小实现
- 跨进程锁的具体实现方案（文件锁 + PID）

### 2. v0.4.0 几何优化
- native 梯度：有限差分起步还是解析梯度？
- BFGS 优化器的具体实现
- Gaussian/ORCA Opt 任务如何接入

### 3. v0.5.0 频率
- Hessian 计算路径
- 平移/转动投影
- 频率/IR 开放范围

### 4. v0.6.0 DFT 质量
- 格点收敛验证
- 新基组验证
- 新泛函开放条件

### 5. v0.7.0 GUI 完整
- GUI 接入 TaskService
- 项目/队列/结果 tabs

### 6. v0.8.0 OPV dogfooding
- 工作流依赖图
- 重组能计算

### 7. v0.9.0-v1.0 发布
- CI、打包、验收

要求：
- 每个任务给出：文件路径、函数签名、改动内容、验收标准
- 标注依赖关系（哪些串行、哪些可并行）
- 标注"假通过"风险点
- 给出 v1.0 最小能力集
