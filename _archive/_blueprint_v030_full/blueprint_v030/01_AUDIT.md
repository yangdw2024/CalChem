# v0.2.0 源码审查与证据边界

## 审查范围

本次附件 ZIP 含完整 `calchem/` 原生计算目录、测试目录、`calchem.spec` 和 handoff 报告；但**没有 `pyproject.toml`、requirements/lock 文件、`.github/workflows/` 或 Windows 发布包**。不能从此 ZIP 验证可编辑安装、CI 或干净机器打包。源码中的 Python 文件已做静态语法解析；未运行数值回归或 Windows/Gaussian/ORCA。当前沙盒的 Python 没有 pytest，因此报告的“22 tests pass”只能记为作者本机报告，不能记作本次复验通过。

## 高优先级发现

| 严重性 | 证据位置 | 发现 | v0.3.0 必做 |
|---|---|---|---|
| P0 | `source/calchem/api/service.py`、`jobs/dispatcher.py`、`gui/worker.py` | `TaskService.run()` 直接调用后端，`submit()` 只写自己的 `JobQueue`；新 `JobDispatcher` 没被这些入口使用。`run_now()` 的锁仅锁本实例。GUI 仍调用 `start_calc_thread(service.run)`。跨进程和跨实例的单并发尚无保证。 | 唯一调度入口 + 跨进程锁；CLI/GUI/API 实际执行路径验证 |
| P0 | `source/tests/posthf/test_regression.py` | 多个 `test_*` 函数失败只 `return False`，没有 assert/raise；pytest 会将正常返回视为成功。部分阈值宽于交接文档宣称。 | 改成断言、精确 fixture、每项阈值来源；保留原始参考和几何 |
| P0 | `source/tests/acceptance.py` | `B2` 条件为 `x is None or x is not None`，`B3` 为 `... or True`；脚本末尾只打印 FAIL，未以失败状态退出。CLI/spec 存在只验证文件存在。 | 可执行验收断言与非零退出；外部程序真实日志证据 |
| P0 | `source/calchem/backends/gaussian.py`、`orca.py` | 仍用专有 `subprocess.run()`，未调用 `external.py` 新增的超时进程树清理；成功状态未要求 return code 为 0 且 SCF 收敛。 | 统一进程执行器与成功判定，保留原始 stdout/stderr |
| P0 | `source/calchem/backends/gaussian.py` | 首个 `SCF Done` 作为 `total_energy_hartree`，MP2/CCSD(T) 可能被误标成 HF/KS 能量；匹配首个而非目标步骤。 | 按方法取最终能，SCF 参考能另存；解析失败不能 success |
| P0 | `source/calchem/backends/orca.py` | `wb97xd` 静默映射 `wB97X-D4`，方法定义不同；`FINAL SINGLE POINT ENERGY` 只取首个匹配。 | 精确方法别名、真实方法写入 Result；多阶段日志选正确终值 |
| P1 | `source/calchem/backends/pyscf.py`、`xtb.py` | 依赖/binary 存在时能力表可能显示 available，实际 `single_point()` 仍抛 `NotImplementedError`；任务表还写了不在 Request Literal 中的 `optimization`。 | 未验收能力禁用；真正实现最小单点并进行真实执行 |
| P1 | `source/calchem/config/loader.py`、`api/service.py` | loader 实现没有被 TaskService 默认调用；默认用户配置在 `Path.home()`，可能位于禁止安装/写入的 C 盘；文件配置只合并少数字段，异常被静默忽略。 | 配置统一加载、字段合并、路径选择/写入检查、错误诊断 |
| P1 | `source/calchem/vib/frequency.py`、`grad/hf_grad.py`、`grad/dft_grad.py` | 频率模块引用未定义 `self.multiplicity`；新分子硬写 `units='bohr'`；负 Hessian 特征值裁为零；频率与热化学公式/零模处理未验证。梯度差分同样硬写 bohr。 | 单位修复、数值梯度基线、投影、虚频、有量纲换算；旧热化学暂禁用 |
| P1 | `source/calchem/gui/main_window.py`、`gui/worker.py` | GUI 单点按钮未接 JobDispatcher；失败/无能量结果用 `{energy:.10f}` 会报错，结果页显示 `scf_energy_hartree` 为总能；取消只是完成后改状态。 | 统一 JobService、Result 状态视图、真实取消语义 |
| P1 | `source/calchem/core/request.py` | `_VALID_ELEMENTS` 至 Og，但 `_ATOMIC_NUMBER` 只到 Pa；较重元素可能被计为零电子。 | 单一完整原子序数表，元素支持须与 native basis 能力分开 |
| P2 | `source/calchem/io/project.py` | 项目文件非原子写入，manifest 版本硬编码 0.1.0，绝对运行路径和原始日志可能不在项目内；请求和结果未构成崩溃恢复流程。 | 相对路径、原子写入、迁移、恢复、版本来源 |
| P2 | `source/calchem/gpu/fock.py` | GPU Fock 明确抛 `NotImplementedError`。 | UI/文档显示未实现，不计入 v1 能力 |

## 静态核验记录

- 附件 ZIP 有原生 `scf/`、`integrals/`、`post_hf/`、`dft/`、`basis/`；本次较上次 v0.1 handoff 更完整。
- ZIP 中 `tests/unit/` 的 `test_*` 函数数目与报告“22 tests”数量相符，**数量相符不等于本次执行通过**。
- ZIP 中 Python 源文件静态语法解析无错误；这不证明运行时导入、数值正确或打包成功。
- v0.1 报告列 H2O native/G16/ORCA 能量按小数精确反算，分别相差 4.232×10⁻⁹、9.373×10⁻⁹ Eh；只有同一几何、方法、基组及日志可追溯时，这种比较才成立。

## 必须保留的科学边界

STO-3G 的特定小分子 RHF 数值基线不能扩展声称所有方法/基组/分子都达到相同精度。`test_regression.py` 对 CCSD 的注释称“简化版、与 MP2 一致”，因此在独立参考值验证之前，不得把该路径标成成熟 CCSD 实现。DFT、梯度、频率、RESP、GROMACS、GPU 都需要独立方法级验收。
