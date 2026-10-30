# 分阶段及最终验收矩阵

每项记录 `PASS/FAIL/BLOCKED/NOT_APPLICABLE`、命令或操作、预期与实得、原始输入/输出路径、git commit、软件版本、运行日期、机器。验收程序必须以非零退出码表达 FAIL。不得以“创建了文件”“导入了模块”“运行进程 5 秒”“stub 注册了名称”代替实际功能。

## 基线门（v0.3 开工前）

- 完整源码快照、版本控制和构建元数据齐全；若旧 repo 没有 pyproject/CI，先创建并记录。
- H2、He、Ne、H2O、CH3 的实际输入、基组、方法、参考值/来源固定。`1e-8 Eh` 仅对已经验证的相同计算条件应用；其它方法先建立独立阈值。
- `test_regression.py` 每项失败会导致 pytest 失败；`acceptance.py` 恒真条件清除，FAIL 导致非零退出。
- v0.2.0 的“22 tests pass”在完整环境重跑，附真实 log；本次沙盒只静态审查，未运行 pytest。

## v0.3 后端和调度门

- CLI、GUI、Python API 两个独立进程同时提交，机器级文件锁让 heavy 进程峰值为 1；一个进程被杀后锁释放、另一个继续。
- `TaskService.submit` 返回的两个排队 job 均真实执行且持久化；任何直接 `run()` 无法绕过锁。
- 外部 timeout 结束进程树并保留 stdout/stderr；失败、未收敛、成功三个状态分离。
- Gaussian/ORCA 同一水分子 HF 单点真实顺序执行；请求、实际 method、输入、log、退出码、Result 互相对应。
- MP2/CCSD(T) parser 取各自最终能，不读首个 SCF Done 冒充；错误日志无最终能时返回 failed。
- PySCF/xTB 在安装可用时实际单点，缺依赖时显示 disabled；REST 无真实协议时 disabled。

## v0.4 优化门

- Native RHF/H2O 数值梯度与经独立验证的梯度逐坐标比较；步长收敛、Å/bohr 单位正确。
- 扰动水分子使用 native BFGS 收敛，最终梯度与坐标落在预定参考范围；达到最大迭代则 not_converged。
- Gaussian/ORCA Opt 各运行一次，最后几何与能量、日志中的优化收敛标志一致。

## v0.5 频率门

- Native H2/H2O 的 Hessian 对称；线性与非线性投影支数正确；负模式保留虚频。
- Gaussian/ORCA 的 H2O Freq 真实输出解析，频率、振动模式、IR（有才展示）与结果字段一致。
- 同位素和单位边界用例有数值检验；旧热化学公式未经重新验证时 UI 不展示自由能。

## v0.6 DFT/基组门

- H2O、开壳层、小有机分子在多档网格能量/梯度收敛，有日志/表格与独立程序同设置对照。
- 6-31G(d)、def2-SVP 与扩散基组有逐元素覆盖、函数数、积分对称性和小分子参考；缺基组元素明确拒绝。
- 新泛函只有 E_xc、V_xc、spin、grid 和梯度/独立参考通过才注册 native；否则 GUI 只显示已验证外部后端能力。

## v0.7 GUI/项目门

- 新建/导入/编辑/保存/关闭/重开项目，所有 job 与输入、结果、日志保持一致。
- 成功 Result 显示 method 最终总能；失败/无能量 Result 不抛格式异常；post-HF 总能与 SCF 能分别标注。
- 队列面板显示等待/运行/成功/失败/取消请求；运行中窗口可交互，关闭行为真实可解释。
- 设置页修改 executable/data root 后后端实际使用新配置；离线且无外部程序时仍可运行 native。
- 结果 tabs 有真实轨道、几何、收敛、频率数据才绘制；空值使用文字空状态。

## v0.8 OPV/GROMACS 门

- workflow 前一步失败后下游 blocked；每一步有不同 request/result，几何 hash 可追踪。
- λ_h/λ_e 四点能量的电荷、spin、几何与方法一致；用已知数字组合正算反算验证公式。
- 真实 GROMACS `grompp` 读入拓扑；力场参数缺失时导出器拒绝输出，不把 RESP 当全部拓扑。
- 一批分子串行执行且可断点恢复；中型例子记录资源和实际结果，不拿 H2O 代替 OPV 成功。

## v0.9→v1 发布门

- CI Windows 实际触发并完成，版本约束、Ruff/mypy/快测均通过；heavy 与 commercial software tests 单独串行。
- 两个 one-folder EXE 在无开发 conda 的 Windows 新用户账户运行，CLI stdout 是纯 JSON；中文/空格路径、只读目录、缺后端、无网络条件有结果。
- 发布 ZIP 完整性/哈希通过，明确版本和软件许可；外部程序、凭据、个人日志未包含。
- PyQt6/Qt 和其余 bundled 依赖的实际分发许可证、通知和所需材料完成核对；若要对外公开发布且许可证条件未满足，此项 FAIL，先修正许可/技术方案。
- 所有宣称的 backend/method/task 每条都有实际验收证据；未通过项不在产品能力表中。
- 最终 `RELEASE_ACCEPTANCE.md` 与版本、README、GUI About、CLI help 一致。

## 最终报告模板

| 模块 | 状态 | 命令/操作与输入 | 实得/阈值 | 原始证据 | 版本/限制 |
|---|---|---|---|---|---|
| 数值内核 | | | | | |
| 后端各方法 | | | | | |
| JobQueue/跨进程锁 | | | | | |
| 优化/频率/DFT | | | | | |
| GUI/项目 | | | | | |
| OPV/GROMACS | | | | | |
| CI/Windows ZIP | | | | | |

只有该表完整、必需项 PASS、发布包实际构建并进行干净环境验收，才向用户宣称 v1.0 产品完成。外部协议无法取得的 REST 项可写 BLOCKED 并从 v1 能力声明移除。
