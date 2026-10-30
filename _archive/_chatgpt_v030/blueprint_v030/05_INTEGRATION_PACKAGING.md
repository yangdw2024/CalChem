# 集成、项目数据、Windows 打包与交付

## 1. 统一调用链

所有入口创建同一个 `CalculationRequest`：CLI/GUI/Python/REST-agent → `TaskService` → JobStore/Dispatcher → machine-wide heavy lock → `Calculator` → `Result` → `.calchem` 项目目录。同步 API `run` 可等待结果，但也必须经锁和相同持久化。`submit` 返回 job ID 并由真正消费队列的 dispatcher 执行。任何 backend 不得从 GUI 读取状态，GUI 不得直接实例化 RHF 或解析 Gaussian log。

配置优先级：调用时显式参数 > 项目覆盖 > 用户配置 > 自动检测 > 安全默认值。`load_config()` 必须由 TaskService/CLI/GUI 默认调用。用户配置位置默认 `CALCHEM_HOME=D:\CalChemData` 下，首次启动未设置时让用户选择非 C 盘，不能以 `Path.home()` 静默写 C 盘。运行目录、临时目录、Gaussian scratch 都按该数据根目录派生。外部程序路径、版本检测结果和最后检测时间显示于 doctor/UI。

## 2. 外部 backend 实施边界

| 后端 | 输入 | 运行 | Result 证据 | 出错策略 |
|---|---|---|---|---|
| Native | `MoleculeSpec` 转旧 `Molecule`，只调用已验证方法 | 内存计算，但受全局 heavy 锁 | 原生版本、输入 hash、SCF/最终能/收敛/轨道（若有） | SCF 无结果或未收敛保留失败 Result；旧数值核心不因 API 重构被改动 |
| Gaussian 16 | gjf、job 独立目录、配置 `%mem/%nprocshared` | 绝对路径参数列表，`GAUSS_SCRDIR` 在数据盘 | 原始 gjf/log/stdout/stderr、正常终止、SCF/后 HF 目标总能 | 不同方法终值分别解析；错误链路不标成功 |
| ORCA 6.1.1 | inp + xyz，资源换算，独立目录 | 绝对路径参数列表 | stdout/out、正常终止、SCF、最终能、优化/频率字段 | 真实 ORCA 6.1.1 fixture 必需；不以 cclib 对旧版支持推断全部字段 |
| PySCF | Python `gto` 和实际已验收的 SCF/post-HF 对象 | in-process，可选依赖 | 基组/xc/收敛、MO alpha-beta、能量与版本 | 未安装时 disabled；不继承 ExternalProgramBackend |
| xTB | XYZ + 参数，GFN2-xTB | 本机 binary、独立目录 | stdout/stderr/终值/版本 | 未安装 disabled；不允许 `basis='gfn2'` 被误当 GTO 基组 |
| REST | 真实 REST 量化程序 schema | endpoint 或本机接口按协议实现 | 协议版本/请求/响应脱敏记录 | 无协议/凭据/样例时 disabled，不猜测 endpoint |

cclib 建议用于具有明确字段映射且真实 fixture 通过的 atomcoords、moenergies、mosyms、vibfreqs、vibdisps 等；“正常结束、SCF 收敛、最后一个目标方法总能、优化是否完成”保留专用安全检查。先在同一份 Gaussian 16 A.03/ORCA 6.1.1 输出上双 parser 比对；缺字段可回退旧 parser，但回退必须明示来源并有 fixture。cclib 当前官方版本表将 Gaussian 16 列为测试版本，ORCA 列到 5.0；6.1.1 由项目自行验收。

## 3. REST-agent 和 RESTBackend 明确分离

- REST-agent/Qwen3-8B 是上层 L3：它产生完整、可审阅的 `CalculationRequest`，通过 `TaskService.submit` 获取 job ID，轮询/订阅状态、读取 Result。agent 不得直接读外部程序 log 猜测能量，也不得绕开 capability/lock。
- RESTBackend 是 L1：仅当作者的 REST 量子计算程序提供真实协议时，按版本实现输入转换、认证、状态/结果解析。若 REST 是本地 CLI 而非 HTTP，adapter 应使用真实 CLI，不因名称为 REST 就假定 HTTP。
- agent 工具只暴露 `list_backends`, `validate_request`, `submit_job`, `get_job`, `get_result`, `cancel_job` 等有界函数；token 权限最小化，所有 agent 提交的 method/basis/charge/spin 经过同样校验。
- 首版没有自动“推荐最快后端”；后端由用户或明确 workflow 指定，失败时不自动换另一个方法产生看似等价的结果。

## 4. 可复现项目格式

```text
my-study.calchem/
  project.json                         # 项目 schema、名称、相对资源索引
  molecules/<id>.xyz                  # 输入文件副本
  jobs/<job-id>/request.json          # 最终生效配置、几何、method/basis
  jobs/<job-id>/state.json            # 状态机、时间、parent_job_id
  jobs/<job-id>/manifest.json         # 程序/版本/输入 hash/退出码/相对路径
  jobs/<job-id>/result.json           # Result v2
  jobs/<job-id>/input.gjf|input.inp|input.xyz
  jobs/<job-id>/stdout.txt; stderr.txt; program.log
  jobs/<job-id>/artifacts/*           # chk/hess/cube 等按可用性选择保留
```

持久化前写同目录临时文件，flush/fsync 后原子替换；日志以 append 方式保存。ProjectStore 以相对路径记录 job 文件，打开项目时验证不越出项目目录。移动项目后能打开历史结果；原始日志保留路径。启动时 `running` 且对应进程不存在的 job 标为 `interrupted`，不能自动当成功或静默重跑。重试生成新 job ID 并关联 parent ID。

## 5. OPV 与 GROMACS

工作流的每一步生成独立 Result：中性优化 `M0`、离子优化 `M+`/`M-`、所需垂直单点、HOMO/LUMO、偶极。重组能的每项使用明确几何、电荷与多重度；λ_h/λ_e 由四点能量相减计算，并将全部原始能量与符号定义保存。缺少任何项、方法改变或几何标识不一致，停止计算该指标。

GROMACS `.itp` 需要力场 atom types、非键和键合参数等，RESP 只给部分电荷；出口必须验证参数完整性。导出后用本机 `gmx grompp` 读真实 topology + coordinate，出错时保留输入和 stdout/stderr。不能用“写出 `.itp` 文件”替代读入验收。对中型 OPV 例子记录 RAM、耗时与失败率，超过预算时建议外部后端但不自动改方法。

## 6. Windows 发布构建

1. 从真实 Git commit 的干净 checkout 建立 `D:\CalChemBuild\venv`，固定 Python 版本和依赖约束，构建日志记录 commit、锁文件 hash、PyInstaller 版本。
2. 构建两个 one-folder 入口：`CalChem.exe` GUI（无控制台）与 `calchem-cli.exe` CLI（有控制台和 stdout JSON）。两者独立目录，使用同一版本化 `calchem` 源和同一 `D:\CalChemData` 配置/锁位置。不要把 `console=False` GUI EXE 当作可用于管道的 CLI。
3. 两个 spec 都只收集必需资源：基组/数据文件、Qt plugin、Matplotlib 资源和明确的 backend 导入；不要无差别 `datas=[('calchem','calchem')]` 或无界 `collect_submodules('calchem')`，以免夹带源码、测试、可选巨型依赖。
4. 单一 Qt binding 为 PyQt6；构建环境检查 PySide6/PyQt5 等不会被意外收集。GUI 先导入 PyQt6，再导入 Matplotlib QtAgg。
5. 构建 work/dist、pip cache、TEMP/TMP、Gaussian scratch、项目数据都设在 D/E 盘；记录实际落点。Windows 系统自身的不可控临时写入不作不存在的承诺，但构建和应用管理的路径不可指向 C 盘。
6. 发布 ZIP 结构：`CalChem-v1.0.0-win64/GUI/CalChem.exe+_internal/`、`CLI/calchem-cli.exe+_internal/`、`README_FIRST_RUN.md`、`LICENSES/`、`VERSION.json`、`SHA256SUMS.txt`、`examples/`。Gaussian/ORCA/xTB 安装本体、license 文件、REST token、个人路径/日志不得打包。PyQt6 为 GPLv3/商业双许可，发布前核对所选许可与整包分发义务，不靠 ZIP 能启动就当许可证已满足。
7. 干净 Windows 用户环境解压到有空格和中文的非 C 盘路径，启动 GUI，做 native 单点、打开/保存项目、查看结果；CLI `--json` 由其他程序解析。删除开发 conda PATH 后再测试。
8. 更新为替换整个 one-folder 目录，数据在外部 `D:\CalChemData` 保留；升级前检查项目 schema 迁移与回滚。v1.0 先交便携 ZIP，安装器若新增须另做安装/卸载测试。

PyInstaller 官方说明 one-folder 是默认可分发模式，Windows 包需要在 Windows 上构建；本机仅 Linux 沙盒无法替代 Windows release gate。

## 7. CI 与本机门槛

- GitHub Actions Windows runner：`pytest tests/unit tests/parser tests/numerical` 中的轻量、确定性用例；Ruff、公开 API mypy、安装和 CLI JSON smoke。Linux 可跑跨平台逻辑但不替代 Windows UI gate。
- `tests/external`、完整 `tests/posthf`、中型 OPV、Windows GUI、`gmx grompp`：在有程序许可的本机顺序执行，输出机器可读日志。不能 `pytest -n auto` 对 heavy 套件并行。
- 退出码 gate：数字比对必须 `assert` 或抛异常；失败时 CI/PowerShell 返回非零。允许 BLOCKED 的原因只限不可获取外部程序/协议/许可等，并在最终报告逐项标注。

## 8. 官方依据

- [cclib Overview](https://cclib.github.io/) 和 [Parsed data](https://cclib.github.io/data.html)。
- [PyInstaller Operating mode](https://pyinstaller.org/en/stable/operating-mode.html)、[Usage](https://pyinstaller.org/en/stable/usage.html)、[Qt hook configuration](https://pyinstaller.org/en/stable/hooks-config.html)。
- [GROMACS topology file formats](https://manual.gromacs.org/2025.2/reference-manual/topologies/topology-file-formats.html)。
- [PyQt6 licensing](https://riverbankcomputing.com/software/pyqt/intro)。
