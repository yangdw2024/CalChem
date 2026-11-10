# 操作记录与经验

日期：2026-11-09（Asia/Shanghai）。所有操作为本轮审计，不代表已经修复业务缺陷。

## 操作记录

1. 读取用户规则、历史审计摘要、当前 Git 状态/提交时间线、项目架构原则、开发脚本与两个 CI workflow；定位实际工作树。
2. 创建本轮计划、状态、回滚规则和目录。使用 PowerShell 固定 376 个原文件 SHA-256 并建立 rollback/source_snapshot.zip，保留已跟踪和未跟踪实现。
3. 读取/检索 13 个 crate、Python 原生路由、OPV、适配器、分发、参考和测试逻辑；全量模式扫描 353 个代码/配置文件、65,244 行（含测试）。此数字是扫描规模，不是人工逐行证明或测试覆盖率。
4. 用 probes/env.ps1 设置 D: Rust/Python 工具链、E: target、TEMP/TMP/数据/缓存和 4 个 Rayon 线程。Cargo 离线且 --locked；没有安装软件、联网下载依赖、提交、发布或启动 GUI。
5. cargo test --workspace --locked：退出 0，91 个测试通过、0 失败、0 忽略；13 个 crate 的测试/文档测试入口均执行（部分入口 0 个测试）。
6. cargo build --release --locked -p calchem-py：退出 0。复制构建扩展到独立运行目录，未替换项目 calchem/_core.pyd。重建扩展与原扩展 SHA-256 相同，证明本次 Python 实测二进制与当前源码构建一致。
7. Python 正常 JIT 环境的 tests/unit：96 通过，0 失败，420.81 秒。未将 unit 子集称作整个 Python 历史 tests/ 全量通过。
8. 编译并执行同一 boys.rs 的独立探针，以 mpmath 70 位精度不完全 gamma 定义核对 399 个 Boys 样本和 11 个 erf 样本。
9. 独立梯度、坐标扫描、旋转/平移/原子排列、极小值冒充 TS、溶剂模型参数、未收敛传播探针全部保存原始 JSONL。
10. 编译独立 Rust 探针，验证 >64 AO 真稀疏分支与 dense Fock/K、一电子弥散密度网格积分及含溶剂优化梯度。
11. 本机 Gaussian 16/ORCA 在独立 E: 目录实跑，留存输入、输出、退出码和正常终止标记。未实际运行 REST 或 PySCF。
12. 使用当前安装的 setuptools 在两份独立 E: 暂存源码树构建 wheel，检查 zip 内容及 WHEEL tag；未安装或上传 wheel。
13. 对真实 ChemDatas 22/32/60 原子 OPV 输入执行与项目 benchmark 对应的单点、簇结合能、溶剂与 TDA 流程；另记录 158/159/199 原子完整体系的结构来源，不对这些完整大体系运行 SCF。
14. 对快照 376 个归档文件逐一复核哈希，ZIP CRC 通过；在独立 E: 暂存目录抽取 Cargo.toml 与 XC 核心文件演练恢复，均与起点一致，未写回业务目录。外部原始输入/输出 51 个文件另归档，CRC 通过；报告绝对本地文件链接全部可解析。
15. 最后补测当前 CI 的 Ruff E/F/W/I/B：848 项，exit 1，未见 F821/F822/F823；这不改变 Rust/Python 数值单元测试通过的事实，也不等于完整 CI 通过。
16. 补充实际调用 OPVWorkflow.align_donor_acceptor，重新运行两次 native 单点，81.73 秒成功；独立保存结果，不使用缓存或参考数值替代 SCF。

## 环境与探针错误及处理

- Python 对新建 docs/audits 子目录的 mkdir/write_text 返回 PermissionError，而 PowerShell 写入正常。未更改 ACL、关闭防护或改动业务源码。
- 初次 pytest 常规捕获因 TEMP 写入失败，tempfile 还枚举了系统默认候选目录；均失败，未有成功 C: 写入证据。后续显式固定 tempfile.tempdir，避免继续回退到 C:。
- 曾使用 NUMBA_DISABLE_JIT=1、-s 运行受控诊断；发现项目既有 build_target/runtime 可正常写入后，终止仅本轮诊断 Python（PID 经命令行核对），保留日志，恢复正常 JIT 并重跑成功。该中断诊断不计入通过总数。
- Boys 探针首次相对路径多写两层，rustc 报找不到文件；核对真实路径后修正，仅修改审计新增探针。
- 频率对齐探针首次给 run_optimize 传入其未暴露的 max_force 等参数，报 TypeError；检查实际 PyO3 签名后移除，原日志另存 external-followup-first.log，重跑完成。此错误属于审计脚本，不算项目算法失败。
- 早期接口探索中对 run_scrf/run_frequencies/ScfWavefunction 的名称猜测得到 AttributeError，随后依据导出列表与签名使用 run_solvation/run_frequency/PyScfWavefunction；不列为项目缺陷。
- 个别 rg 参数与 PowerShell 参数混用的查询已改为显式文件路径/正确 rg 参数。检索失误不作为“代码不存在”的证据。

## 可复用经验

- 先验证运行目录实际可写，再启动缓存、pytest 和外部程序；TEMP 环境变量本身不保证其目录可写。
- 审计必须验证 Python 实际加载的扩展哈希，否则会把旧 .pyd 的行为误归当前 Rust 源码。
- 标准模型正确性要靠外部独立数值/理论对照；同一错误能量函数的梯度也可以完美通过有限差分。
- 比较前统一 Cartesian/spherical、泛函变体、是否冻结核、色散和溶剂、几何、收敛/网格及单位。
- 稀疏 ERI 需跨过实现的 64 AO 分支边界测试；默认小分子全绿不能验证该分支。
- 对“收敛”“一阶鞍点”“到达极小值”“溶剂化自由能”等语义也必须做反例测试。
- 运行外部 benchmark 时，不沿用共享 bench.inp/bench.out，不拼接旧输出，不以打印 Successfully 代替数值断言。

## 回滚原则

本次不修改业务源码和现有扩展；如哈希变化，首先排查并行用户编辑，不自动恢复。需要恢复时从快照提取到独立 E: 暂存目录后逐文件审核。不得把整个已含用户改动的工作树 reset/clean，不清理旧审计、历史 benchmark 或原始 ChemDatas。
