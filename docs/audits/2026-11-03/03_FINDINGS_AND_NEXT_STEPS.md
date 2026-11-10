# 问题清单与后续执行规划

基于 2026-11-03 的当前源码和本次实测。这里的 P0 表示“继续扩展能力前应先解决的可靠性问题”，不等同于生产事故评级。本次只建立清单，没有修改这些业务实现。

## 1. 应优先修复的问题

| 编号 | 优先级 | 问题与用户影响 | 代码证据 / 实测证据 |
|---|---|---|---|
| Q01 | P0 | Gaussian 优化日志使用第一次 SCF 能量，可能把初始能量当作最终结果；Post-HF 总能量也不能用 SCF 值替代 | `calchem/backends/gaussian.py:73` 使用 re.search；合成两轮 -1.1/-1.2 输出选出 -1.1；checks.json |
| Q02 | P0 | Gaussian 频率局部解析成功，Result 却返回 null 且 status=succeeded，调用者看不到频率 | `gaussian.py:302` 读取 parsed.get("frequencies") 而不是已计算的 frequencies；checks.json |
| Q03 | P0 | ORCA 在构造 Result 时引用局部作用域不存在的 dipole_mag，正常计算输出仍可能报 NameError | `calchem/backends/orca.py:190`；合成正常输出已复现；Ruff 同时确认 |
| Q04 | P0 | 原生能力表与构造接口不符，很多已列出方法不能实际创建计算对象 | native.py 的 _build_calculator；ROHF 缺类，RKS/UKS 的 HF 父类参数不兼容，MP2/MP4/CCSD 构造签名不匹配，CCSDT 类名错误，CASSCF 引用已不存在的辅助函数；deep-checks.json |
| Q05 | P0 | Rust 路径忽略请求计算选项；失败又静默回退 Python，使执行路径和约束不透明 | `native.py:152` 未转发 request.options；`native.py:182` catch-all 返回 None；直接 max_iter=1 报未收敛，但服务同请求选项仍运行 9 次并成功 |
| Q06 | P0 | 当前测试无法提供全绿验收：Rust 测试编译错误、错误参考值、Python 测试接口漂移 | `eri_reference.rs:54`、`boys.rs`、`one_electron.rs`、`test_core_rust.py`；详见日志和独立参考复核 |
| Q07 | P1 | 数据目录/外部路径配置没有连通默认入口；环境变量被默认值遮蔽，且存在 C 盘 fallback | `service.py:28`、`config/loader.py`、CLI/GUI TaskService()；checks.json 证实 CALCHEM_DATA 被遮蔽 |
| Q08 | P1 | 两个独立队列导致 service.submit 任务没有消费者；取消、恢复、并发语义不完整 | service.py:94、dispatcher.py:29；实测 queue identity 不同，取消完成后状态不终结 |
| Q09 | P1 | Result 与文件关联不完整，缺 request/manifest，单独 ProjectStore 未接入主链 | service.py:104；实测 job_id=null、request_saved=false、manifest_saved=false |
| Q10 | P1 | Rust 单点结果未映射轨道/HOMO/LUMO，OPV 工作流相关字段为空 | native.py 的 _try_rust 返回构造器；四分子服务结果 orbitals=[]、homo=null |
| Q11 | P1 | SCF 返回的密度、Fock、能量未做到严格同一最终自洽点，影响后续解析梯度 | `crates/calchem-scf/src/rhf.rs:194` 起；探针默认 F 差约 1.24e-10，放宽容差后约 8.28e-6 |
| Q12 | P1 | 版本、入口说明和发布验收相互矛盾，难以判断产物对应哪一份源码 | package/CLI 0.1.0，pyproject 和 crates 0.2.0，native capability 0.3.0，Git 阶段标记 v0.6/v1.0；README CLI 示例不符 |
| Q13 | P1 | 当前 CI 未覆盖 Rust、分支不匹配、无 remote，Python 全目录收集还有导入副作用 | `.github/workflows/ci.yml`、test-structure.json；本次不声称线上 CI 已运行 |
| Q14 | P2 | 大体系、GPU、OPV 卤化研究能力距离产品描述仍有明显缺口 | Rust N⁴ ERI、无梯度/DFT/UHF；GPU build 为 NotImplementedError；Rust 无 Br/I 内嵌基组；OPV 仅单点封装 |

Q01/Q02/Q03 的适配器验证使用 monkeypatch 替换 subprocess，未启动外部程序。它们证明当前输入输出链存在缺陷，不代表外部软件本身有问题。

## 2. 测试失败的准确归因

### Python 扩展测试

本次选择 28 个现有 pytest 用例：19 通过、9 失败。test_core_rust.py 的公共助手传入 uint8 原子序数数组，当前桥接实际要求元素字符串列表；另外测试调用尚不存在的 run_gradient/run_optimize。不能通过跳过整个扩展测试来宣称修复，应把 Stage 2 当前契约的回归与 Stage 3 未来能力的验收拆清楚，未来功能明确列为未实现。

### Rust 编译错误

`eri_reference.rs:54` 的 `{got:.15f}` 和 `{expected:.15f}` 使用了不适用于 Rust 的格式写法。仅是测试编译问题，尚未跑到该测试的积分断言。可以修正显示格式，但不应改动算法以绕过编译错误。

### Rust 固定参考值问题

独立证据见 `boys-reference-quadrature.json`、`kinetic-reference.json` 和 `inspect-integrals.jsonl`：

| 项目 | 现有测试参考 | 独立计算 | 结论 |
|---|---:|---:|---|
| F₀(25) | 0.1772453850905516 | 0.17724538509027907 | 与测试 1e-14 阈值不一致；Rust 约 0.17724538509027923 |
| F₄(25) | 0.0002555074603299322 | 0.0000029777221487496273 | 固定参考明显不准确；本次 Rust 测试已在前一项终止，未实际执行此断言 |
| F₀(29.34071349) | 0.16360991294575364 | 0.16360990987453258 | 固定参考不准确；不能作为当前精度验收金值 |
| H₂ T₀₁ / STO-3G | 0.236454655845882 | 0.23645465597967397 | 用当前基组数据的 s/s 闭式公式与 Rust 完全同值；现有参考差约 1.34e-10，而阈值为 1e-12 |

Boys 独立计算使用 SciPy 自适应积分计算定义积分，记录估计误差；动能独立计算采用闭式高斯 s/s 公式，并使用当前 JSON 的系数与归一化。未扩大阈值、未替换参考值、未改算法。后续应保存参考生成方法、单位、几何和数据来源，必要时再用高精度算法或外部独立程序确认。

历史 Python 水能量与 Rust 相差约 6.29e-8 Eh，不能继续用“所有后端一定相差 1e-8”作为无条件通行标准；应先统一 Boys 的参考生成口径并保留旧值溯源。

## 3. 需要进一步核验的静态风险

以下不是本次已经执行完整端到端复现的全部缺陷，后续需针对性验证：

- native 的有限能量检查用 e==e 排除 NaN，没有显式排除无穷值；部分 Post-HF 的 energy/e_tot/e_corr 契约也需统一。
- native.optimize 的失败能量被替换成 1e10，可能让优化器继续处理失败单点；它只检查是否有能量，没有强制要求单点成功收敛，且不向子请求传递完整选项。
- 旧高层数值梯度重建扰动 Molecule 时没有明确保留原分子的 charge/multiplicity。涉及离子重组能时尤其需要实测修复。
- 旧频率 API 和新 numeric_frequency 工具并行存在；投影、质量加权、线性分子、虚频处理和单位转换尚未做本次科学验收。
- GUI 失败结果 None 能量格式化、参数变化后 Pydantic 对象直接赋值绕过重建验证、原子表编辑不回写计算几何、线程/worker 生命周期需补针对性检查。
- GUI worker 的取消只设置标志；先保存结果再改 cancelled，可能出现显示状态与磁盘不一致。
- external.py 有超时清理辅助函数，但 Gaussian/ORCA 当前直接 subprocess.run，未调用该辅助函数；不能据辅助函数存在就认定所有子进程都能清理。
- global_lock 注释与实现不符，且空/损坏锁文件清理存在竞争窗口；跨进程/多目录并发没有本次压力验证。
- tests/acceptance.py 中部分检查恒真，如 dipole 为 None 或不为 None、schema 判断 or True；历史 PASS 不能替代有效断言。
- Python CCSD 源码和回归说明明确保留“简化版”，不能默认当作已验证的完整 CCSD；D3、部分基组和 TDDFT 模块也有近似/简化说明。

## 4. 建议的依赖顺序和验收条件

### 第一步：先恢复可信验收基础

保护未跟踪 Stage 3 设计文件，备份关键文件；修正 Rust 测试格式、固定参考与 Python 当前桥接测试契约。将 import 时执行的历史脚本与真正 pytest 用例隔离；把 return bool 的回归改成有效 assert。接入 Rust CI，并明确当前主分支策略。

验收：当前支持的能量/积分/错误路径测试可执行；每个失败都有可复现原因；不能通过大规模 skip、删断言或随意放宽阈值获得全绿。

### 第二步：统一服务与结果真实性

先修 Gaussian 最终能量、频率字段和 ORCA 结果构造；再修正 native 方法构造契约与能力表。把 Rust 选项显式映射并记录 actual_method、fallback_reason；将未收敛和真正程序异常区分。统一 CLI/GUI/API 的配置装载、任务 ID、请求与结果持久化，并确保全部输出默认在 D/E 盘。

验收：同一请求从不同入口得到相同任务/单位/后端信息；运行选项确实生效；请求、最终结果和日志可以互相定位；外部计算须用同几何、同方法和同基组重新做真实回归。修 adapter 的合成测试通过后，仍需真实软件验收。

### 第三步：进入 Stage 3 自研 Rust 内核

先处理最终 SCF 状态一致性，再做积分中心导数、核排斥导数和 Pulay 项，随后才接 BFGS。梯度先与独立中心差分逐坐标比较，再做平移/转动不变性检查。优化参考必须使用同理论级别，不把实验键长当 RHF/STO-3G 的严格金值。

验收建议沿用设计中的能量 1e-6 Eh、梯度 1e-4 Eh/Bohr 作为起点，但以已验证参考和 SCF 精度为前提；优化检查最大力、RMS 力、位移和真实能量下降，拒绝把失败 SCF 的哨兵能量当有效能量。

Stage 3 文档中“6-31G 没有验证”“矩阵乘需要改 i,k,j”两点已落后于当前代码：已有对应能量测试，matmul 也已经 i,k,j。应保留设计原稿，在实施规划中标注差异，避免重复开发。

### 第四步：以测量决定性能工作

在能量/梯度/优化通过后，分别增加筛选、并行与数据布局改进，每次保留串行未筛选基线。先测 N⁴ 内存和 Fock 瓶颈，记录机器、build profile、线程、预热、样本和误差。已有 `faer`/`rayon` 依赖不代表主路径已经使用它们。

验收：性能报告同时给数值误差和用时/内存；小分子串并行阈值由测量决定；能量 Schwarz 阈值不能直接当作梯度误差保证。本次没有对 <10 ms 或“超过 Gaussian”给出结论。

### 第五步：再扩展到卤化 OPV 研究

先补目标元素/基组覆盖，尤其 Br/I；再建立闭壳层/开壳层离子、DFT/色散、溶剂、轨道和激发态的独立基准。只有这些路径可靠后，才完成优化、频率、四点重组能、HOMO/LUMO 与 GROMACS 参数交接的实际端到端流水线。

验收：选取研究中有代表性的含 F/Cl/Br/I 小模型分级验证，然后逐步扩大到 NFA；记录理论级别、几何、基组来源和对照程序，不能用四个小体系 RHF 结果替代 OPV 应用验收。

## 5. 本轮交付边界

本轮审计目标已完成。上述缺陷仍保留在业务源码中，供下一轮按优先级修复；没有把“发现问题”写成“已经修复”，也没有因测试失败而删除文件或覆盖用户现有设计。
