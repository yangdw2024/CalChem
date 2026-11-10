# 本次操作记录

日期：2026-11-03，Asia/Shanghai。按实际执行顺序记录，检查输出保存在 evidence/。所有业务问题均为审计发现，本轮未实施业务修复。

| 顺序 | 操作 | 结果 / 证据 |
|---|---|---|
| 1 | 核对 cwd、父目录 AGENTS、顶层结构；快速检索相关历史记忆索引 | 本机实际路径为 E:\YDW\Trae_Gaussian；未发现本地额外 AGENTS；记忆索引无相关条目，没有用旧记忆代替当前证据 |
| 2 | 保存 Git 分支、状态、完整提交时间线与文件清单 | master/c9fd62e；初始未跟踪 Stage 3 设计；git-history/status-before/tracked-files/file-inventory |
| 3 | 写检查规划、起始状态及执行边界；保存当前核心源码哈希 | 00_START_STATE.md、01_PLAN.md；source-hashes-before.csv、design-hash-before.csv |
| 4 | 完整阅读关键运行链、Rust 内核、当前设计和主要历史验收；检索其余模块的接口/占位/副作用 | 识别统一入口与旧 API 双链、未实现 crate、配置/队列断点；历史与当前状态分开记录 |
| 5 | 查本地 Python、Rust、Ruff 位置，重定向临时与缓存目录 | 工具均来自 D 盘；TEMP/TMP、NUMBA/MPL/CuPy 缓存、测试目录和 Rust target 放在 E 盘审计 runtime |
| 6 | 对 244 个当前 Python 文件 AST parse + 内存 compile | 无语法错误，未使用会生成 pyc 的 py_compile；python-inventory.json |
| 7 | 对四个分子分别运行 Rust、Python RHF 和统一服务；直接验证两个 6-31G 案例 | 收敛并保存数值；checks.json 和 python-checks.log |
| 8 | 有界复现选项转发、队列、配置、取消和适配器解析问题 | Gaussian/ORCA 用合成日志和 mock，不启动真实外部程序；默认服务目录诊断屏蔽 mkdir |
| 9 | 执行 H₂ 有限差分与服务优化 | 优化成功；SciPy ftol 警告保留；没有声称解析梯度已实现 |
| 10 | 首轮 pytest 被审计写入防护拦截空设备日志；保留初次结果后修正验证工具 | python-checks-attempt1.log / checks-attempt1.json；仅调整审计脚本的日志路径 |
| 11 | 重跑选定现有 Python 测试 | 19 passed、9 failed，真实 pytest 状态保存在 checks.json；诊断程序本身正常结束不代表测试全绿 |
| 12 | 隔离 target 中运行 cargo test --workspace --offline --locked | 测试编译因 .15f 错误阻断；rust-tests.log / rust-exit.txt；没有下载依赖或修改 Cargo.lock |
| 13 | 分别执行 library、basis/SCF、Boys、one-electron 和 ERI symmetry 检查 | 3 library、4 basis、6 SCF 通过；Boys 与 one-electron 各一项失败；各分组日志记录真实结果 |
| 14 | 构造阶段审查多种 native 方法，测试受控 CLI JSON 和 Result 保存，检查测试结构 | deep-checks.json、test-structure.json；没有启动大分子或耗时高级计算 |
| 15 | 运行本地 Ruff F821/F822/F823，只读、不缓存、不修复 | 五处未定义名称；ruff-critical.json |
| 16 | 检查根目录 Stage 2 ZIP 的条目和 CRC | 文件无损坏但参考包不全；没有解压覆盖项目 |
| 17 | 独立数值积分核对 Boys 固定参考；用已构建 rlib 编译审计探针 | 多个参考值不准确；Rust 返回波函数的一致性误差被量化；独立 s/s 公式与 Rust T01 相同 |
| 18 | 备份 21 个关键文件、核对 SHA256、执行恢复 WhatIf | backup-manifest-path.txt 指向真实清单；没有实际恢复覆盖 |
| 19 | 整理状态、工作流、问题优先级、执行规则和经验；对大体积 runtime 做局部 Git 忽略 | 所有文档集中本审计目录；未修改根 .gitignore |
| 20 | 最终核对 Git、前后源码哈希、Stage 3 原稿、备份与新工具语法 | 结果记录在 integrity-after.json 和 git-status-after.txt；没有提交/推送 |

## 没有执行的动作及原因

- 没有直接跑全目录 pytest：部分历史文件收集时就运行 GUI/计算，且不构成有效断言；先通过 AST 和内容检查界定风险。
- 没有运行 tests/acceptance.py：它会调用真实外部程序、写独立数据目录，而且包含恒真验收项，不能无审查当成最终验收。
- 没有安装或升级软件：D 盘现有工具足够；Rust 使用 --offline，缺失依赖时应记录阻塞而非自动联网安装。
- 没有启动 GUI 或已打包 EXE：本轮以代码/数值/接口审计为主，不把文件存在当运行通过。
- 没有联网：本次结论来自本地现状，没有外部资料的新旧时效判断需求；未来方法研究或技术选型另建调研记录。
- 没有改动旧实现、设计、归档，也没有 Git 自动提交。备份不构成修复完成证明。

## 可重放检查入口

在项目根执行 `tools/run_checks.ps1`（本目录下的工具）可复查 Python 检查；`run_rust_checks.ps1` 执行全 workspace，`run_rust_subsets.ps1` 执行部分分组。所有脚本的默认运行产物都指向本审计 runtime。详细命令和脚本本体一起保留，注意重跑会更新本目录检查日志，重要证据需先备份。
