# CalChem 最终产品验收表

使用规则：每项标记 `PASS / FAIL / BLOCKED / NOT APPLICABLE`，并附运行命令、环境、输出文件路径或截图。不得用“看起来正常”代替证据；不得把 mock/fixture 的通过写成真实后端通过。

## A. 核心与科学正确性

- [ ] 原生 H2/STO-3G RHF 基线复现；绝对能量偏差满足既定参考阈值。
- [ ] He、Ne、H2O、CH3 及回归清单中所有可执行条目均记录结果。
- [ ] H2O RHF 冲突的参考值有相同输入与设置依据；README、测试和基线文档一致。
- [ ] `calchem/tests/` 和顶层 `tests/` 均存在；scratch/debug 不误进默认测试收集。
- [ ] API facade 不改原生 SCF/积分调用参数与顺序；legacy API 可用或有经过记录的迁移方案。

## B. 核心契约与数据质量

- [ ] Request、Result、配置均 schema-versioned。
- [ ] Result JSON round-trip，无 numpy scalar/ndarray 序列化失败。
- [ ] 单位对坐标、能量、轨道、梯度、偶极、频率明确且经过转换验证。
- [ ] 不可得量为 null；不支持的任务明确错误；未收敛结果有状态/错误/原始文件。
- [ ] 后端版本、CalChem 版本、输入 snapshot、耗时、raw file path 可追溯。
- [ ] 能力表来自 backend 代码，不是手写 UI 宣传文本。

## C. 后端真实可用性

- [ ] Native：Python API、CLI、GUI 三入口执行同一请求并返回同一 Result 契约。
- [ ] Gaussian：真实程序执行单点；检查 gjf、log、normal termination、SCF 收敛、最终能量和退出状态。
- [ ] ORCA：真实程序执行单点；检查 inp、out、正常结束、SCF 收敛、最终能量和退出状态。
- [ ] PySCF：optional extra 未安装时 core 可导入；安装后能力表/实际支持吻合。
- [ ] xTB：binary 缺失与真实执行成功分别验证；解析器只暴露确有的数据。
- [ ] REST：仅当真实 REST 规范和程序可访问时执行端到端；否则清楚标注 blocked/unavailable，不允许模拟成功。
- [ ] 至少一个真实相同水分子任务依次运行 native/Gaussian/ORCA；记录方法、基组、坐标、数值误差及计算设置差异。

## D. Job / 项目与 UI

- [ ] 同一时刻最多一个 heavy job running；其他任务只排队。
- [ ] Job 状态只按合法状态机转换；终止、超时、失败、SCF 未收敛区分。
- [ ] 任务输入/日志/Result/manifest 在唯一目录；应用重开后可查看历史结果。
- [ ] 3D 分子、原子表、设置表、任务队列、结果 tabs、日志入口均能使用。
- [ ] 大计算期间 GUI 不冻结；widget 更新只发生于 GUI 主线程。
- [ ] 关闭时运行任务提示真实状态；取消仅表示 request，不能伪报立即终止。
- [ ] 无 backend、无可选字段、无项目、解析失败、无效几何都有明确空/错误状态。
- [ ] 分子导入/导出、任务设置与结果展示不生成未经用户请求的额外计算。

## E. CLI、打包与 Windows

- [ ] `calchem --version` / `calchem backends` / `calchem run` 在干净环境可执行。
- [ ] JSON CLI stdout 可直接解析；日志不混入 JSON stdout。
- [ ] one-folder ZIP 在 Windows 11 干净用户账户解压运行。
- [ ] Python、build work、app、data、run、cache 均使用配置的非 C 盘位置。
- [ ] 路径包含空格和中文时仍能打开项目并生成输入/日志。
- [ ] 缺少 Gaussian/ORCA 时 core 和 native 模式照常启动，状态提示可理解。
- [ ] PyQt6 只有一个 Qt binding 被收集；启动窗口、Matplotlib canvas 与图标正常。
- [ ] ZIP 含 README、版本号、license notices、示例和 SHA-256 文件。
- [ ] Gaussian/ORCA/xTB 可执行程序和许可证文件没有被打包进 CalChem。

## F. 发布与维护

- [ ] Windows CI 通过纯 Python、native 快速基线、parser fixture、lint/type gates。
- [ ] 完整数值回归串行完成并保留输出。
- [ ] 已知功能、限制、依赖、安装、外部程序配置和常见错误有文档。
- [ ] 所有宣传文本和 About 页面与 backend capability registry 一致。
- [ ] `RELEASE_ACCEPTANCE.md` 对每条给出状态与证据；没有假绿、未标注的阻塞或遗留占位功能。

## 最终报告格式

```text
版本：
交付 ZIP：
Windows 构建环境：
数值回归：PASS/FAIL + 证据路径
后端：逐个列真实可用/未验收/未配置
GUI 流程：PASS/FAIL + 证据路径
CLI/API：PASS/FAIL + 命令
打包 smoke：PASS/FAIL + 环境
已知限制：
阻塞项：
```
