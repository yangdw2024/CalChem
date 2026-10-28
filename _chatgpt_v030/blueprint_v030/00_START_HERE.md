# CalChem v0.3.0 → v1.0 完整产品蓝图包

本包 = 本次上传的 v0.2.0 完整源码快照（原样保留在 `source/`）、原始 handoff 与 v0.3.0 主提示词、以及 `blueprint_v030/` 中的新蓝图。**这不是已实现的 v1.0 程序，也不包含 Windows EXE。**

## 阅读与执行顺序

1. `blueprint_v030/01_AUDIT.md`：先认识本次源码中仍存在的真实风险。
2. `blueprint_v030/02_PRODUCT_ARCHITECTURE.md`：完整产品形态、技术栈、接口、模块与边界。
3. `blueprint_v030/03_FILE_LEVEL_TASKS.md`：按依赖顺序实施的文件级任务。
4. `blueprint_v030/04_UI_SPEC.md`：PyQt6 页面、交互、状态与视觉规范。
5. `blueprint_v030/05_INTEGRATION_PACKAGING.md`：外部程序、REST-agent、GROMACS、Windows 构建。
6. `blueprint_v030/06_ACCEPTANCE_GATES.md`：逐阶段和最终验收，任何假通过均为失败。
7. `blueprint_v030/07_LOCAL_AGENT_DIRECTIVE.md`：可复制给本地 agent 的执行命令。
8. `blueprint_v030/08_REFERENCES.md`：官方资料与关键决策依据。
9. `blueprint_v030/09_FULL_PRODUCT_HORIZON.md`：v1.0 之后的方法平台、性能、agent 和开放生态扩展边界。
10. `blueprint_v030/TASK_TRACKER.csv`：本地 agent 填写每项任务状态、commit 与验收证据。

## 对本地 agent 的一条总指令

**持续完成全部已授权的本地实现、测试、文档和构建工作；阶段验收通过后自动进入下一阶段。用户只审最终质量，不在普通决策、阶段切换或测试失败时停下来询问。** 遇到失败先定位修复，绝不通过降低阈值、删除测试、伪造数据或将 stub 标记 PASS 解决。确实需要用户凭据、商业许可、不可逆删除、对外发布或外发私人数据时，只阻塞依赖那一步，继续完成不受影响的任务，并在最终报告中给出具体证据。

## 产品发布定义

v1.0 是已验收的 Windows 本地量子化学工作台：Python API + CLI + PyQt6 GUI + 可追溯项目 + 串行任务 + 已实际验收的后端能力 + one-folder 发布包。Native/Gaussian/ORCA 的单点是基础；优化、频率、PySCF/xTB、OPV/GROMACS 分别按能力表、真实执行和验收状态开放。REST 量化程序缺少真实协议时不得冒充可用；REST-agent 是上层调用方。原生 GPU、自动后端择优、JAX、周期体系与未验证新泛函不列入 v1.0 承诺。
