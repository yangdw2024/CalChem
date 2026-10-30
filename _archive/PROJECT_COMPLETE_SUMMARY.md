# CalChem 项目完整总结文档

> 版本：v0.6.0 | 日期：2026-10-29 | 作者：阳德武（延边大学化学研究生，OSC 方向）

---

## 一、项目身份

**CalChem 不是 Gaussian 克隆，是统一量子化学计算平台。**

- 自带原生 Python 计算引擎（RHF/UHF/DFT/MP2/CCSD/CCSD(T)）
- 通过适配层接入 Gaussian 16 / ORCA / PySCF / xTB
- 前端工作台为 MolForge（PyQt6 + VTK，2D 编辑器 + 3D 渲染 + AI 助手）
- 目标：在 UI、底层框架、方法精度三个层面超越 Gaussian/ORCA

**定位演进：**
- ~~Gaussian 克隆~~ → ~~OPV 专用工具~~ → ~~替代 Gaussian~~ → **统一量子化学平台 + 自研高性能引擎**

---

## 二、当前版本 v0.6.0 能力清单

### 已实现并验证

| 模块 | 状态 | 数值验证 |
|------|------|----------|
| RHF/STO-3G | 完成 | H2 = -1.116714325063 Eh（三后端交叉验证） |
| UHF | 完成 | H2O 开壳层测试通过 |
| DFT (LDA/PBE/B3LYP) | 完成 | 网格积分已实现，精度未系统验证 |
| MP2 | 完成 | H2O MP2 能量与参考值对齐 |
| CCSD/CCSD(T) | 完成 | H2/STO-3G CCSD(T) 能量对齐 |
| Gaussian 16 后端 | 完成 | SP/Opt/Freq 全通，H2 opt = 1.346 Bohr |
| ORCA 后端 | 完成 | SP 跑通（H2O 差 9.4e-9 vs native） |
| 跨进程全局锁 | 完成 | O_EXCL 文件锁 + PID 检测 + 陈旧锁回收 |
| 任务调度 | 完成 | JobDispatcher 单并发队列 + TaskService |
| 配置系统 | 完成 | 自动检测 + 文件配置 + 环境变量 |
| 几何优化（native） | 框架完成 | BFGS + 有限差分梯度（精度受限） |
| 频率计算（native） | 框架完成 | 数值 Hessian + 模式投影（精度受限） |
| 计算面板（MolForge） | 完成 | Tab 0：CalChem 运行 + cube 加载 |
| 22 unit tests | 通过 | 数值基线不破坏 |

### 已知致命缺陷（必须底层重构）

| 缺陷 | 影响 | 根因 |
|------|------|------|
| 没有解析梯度 | 优化/频率是玩具，慢且噪声大 | 纯 Python，手写 AD 不现实 |
| Python 性能 | 比 PySCF 慢 10-100 倍 | 解释执行，无 BLAS/LAPACK 优化 |
| 基组极少 | 只能跑 STO-3G~6-31G(d) | 基组数据未集成 |
| 没有溶剂模型 | 不能做溶液相计算 | PCM/SMD 未实现 |
| 没有 TDDFT | OSC 激发态不能算 | TDDFT 未实现 |
| 没有并行 | 多核用不上 | GIL + 无线程设计 |
| cube 生成是 stub | 轨道渲染靠 Gaussian | 波函数网格积分未实现 |

---

## 三、数值基线（不可破坏）

```
H2/RHF/STO-3G (1.4 Bohr):    -1.116714325063 Eh
He/RHF/STO-3G:              -2.8077839575  Eh
H2O/RHF/STO-3G (Szabo):     -74.6953191771 Eh

Szabo 几何 (Bohr):
  O(0, 0, 0)
  H(0, 1.432, 0)
  H(1.112, -0.716, 0)

b2a = 0.529177210903
```

**三后端交叉验证（H2O/STO-3G Szabo 几何）：**
- Native Python:   -74.695319177132 Eh
- Gaussian 16 A.03: -74.695319172900 Eh（差 4.2e-9）
- ORCA 6.1.1:      -74.695319167759 Eh（差 9.4e-9）

---

## 四、项目架构

```
E:\YDW\Trae_Gaussian\
├── calchem/                    # 核心计算引擎
│   ├── core/                   # 数据契约（Pydantic v2）
│   │   ├── request.py          # CalculationRequest / MoleculeSpec
│   │   ├── result.py           # Result（含 actual_method）
│   │   ├── periodic_table.py   # 120 种元素统一表
│   │   └── units.py            # 单位换算
│   ├── scf/                    # SCF 主循环
│   │   ├── hf.py               # RHF/UHF
│   │   ├── dft.py              # DFT SCF
│   │   ├── diis.py             # DIIS 加速
│   │   └── guess.py            # 初始猜测
│   ├── integrals/              # 积分引擎
│   │   ├── one_electron.py     # 重叠/动能/核吸引
│   │   ├── two_electron.py     # 双电子 ERI
│   │   └── direct_scf.py       # 直接 SCF
│   ├── basis/                  # 基组系统
│   │   ├── gto.py              # 高斯型轨道
│   │   ├── basis_set.py        # 基组定义
│   │   └── basis_data.py       # 基组数值
│   ├── dft/                    # DFT 模块
│   │   ├── functionals.py      # LDA/PBE/B3LYP
│   │   ├── grid.py            #  Lebedev 网格
│   │   └── fock.py             # DFT Fock 构建
│   ├── post_hf/               # 后 HF 方法
│   │   ├── mp2.py
│   │   ├── ccsd.py
│   │   └── ccsd_t.py
│   ├── grad/                   # 梯度
│   │   └── finite_difference.py # 中心差分（临时）
│   ├── opt/                    # 几何优化
│   │   └── numeric_optimizer.py # scipy BFGS
│   ├── vib/                    # 频率
│   │   └── numeric_frequency.py
│   ├── backends/               # 外部后端适配
│   │   ├── native.py           # 原生引擎
│   │   ├── gaussian.py         # Gaussian 16
│   │   ├── orca.py             # ORCA 6
│   │   ├── external.py         # 通用子进程管理
│   │   └── parsers/            # 输出解析
│   ├── config/                 # 配置
│   │   ├── loader.py
│   │   └── models.py
│   ├── jobs/                   # 调度
│   │   ├── dispatcher.py
│   │   ├── global_lock.py
│   │   └── queue.py
│   └── api/
│       └── service.py         # TaskService.run()
├── tests/unit/                 # 22 个单元测试
├── docs/BASELINE.md            # 数值基线
└── pyproject.toml              # v0.2.0
```

**MolForge 前端（独立项目 E:\YDW\Trae_Chemdraw\MolForge\）：**
```
molforge/
├── domain/                    # MoleculeDocument / 拓扑 / 坐标
├── application/
│   ├── command_bus.py         # 命令总线 + 撤销重做
│   ├── selection.py           # 选择服务
│   ├── jobs.py                # 任务中心
│   ├── calculators/           # 计算引擎注册表
│   │   ├── base.py            # CalculationEngine 抽象
│   │   ├── calchem_engine.py  # ← 新增：CalChem in-process 适配
│   │   ├── gaussian_engine.py
│   │   ├── orca_engine.py
│   │   └── rest_engine.py
│   └── ai/                    # DeepSeek 助手
├── rendering/
│   ├── vtk_backend.py         # VTK 3D（PBR/SSAO/轨道等值面）
│   └── orbitals.py            # Cube 文件解析
└── ui/
    ├── window.py              # 主窗口
    ├── editor2d_widget.py     # 2D 化学编辑器
    ├── render3d_widget.py     # 3D 组件
    └── qc_result_dialog.py    # 量化计算工作台（Tab 0 = CalChem）
```

---

## 五、开发历史（按版本）

### v0.1.0 — 基础框架
- Git 初始化、数值基线
- core/ 数据契约
- NativeCalchem 后端
- Gaussian 16 后端跑通
- ORCA 后端修复跑通
- TaskService + JobQueue + ProjectStore
- PyQt6 GUI（3D 分子视图）
- PyInstaller one-folder 打包（238MB）

### v0.2.0 — P0 科学正确性修复（6 个致命 bug）
1. native 后端 `getattr(calc, "converged", True)` 假值 → 严格非零非 NaN 检查
2. ORCA 偶极 magnitude 被伪造成 z 轴矢量 → 只存标量 magnitude
3. JobQueue 只有 submit/get_next 没有 dispatcher → 新增 JobDispatcher
4. CLI `--json` 模式 Rich 输出污染 stdout → _suppress_stdout 重定向
5. 后端硬编码路径 → 从 AppConfig 读取
6. 跨进程并发锁缺失 → O_EXCL 文件锁

### v0.3.0 — 调度与配置
- global_lock.py 跨进程 heavy 锁
- config/loader.py 自动检测已知路径
- external.py Popen + taskkill /T 进程树清理
- parsers/base.py 共享解析工具
- periodic_table.py 统一元素表

### v0.4.0 — 几何优化框架
- finite_difference.py 中心差分梯度
- numeric_optimizer.py scipy BFGS
- Native optimize() 端到端跑通
- H2 从 2.0 Bohr 优化到 1.67 Bohr（精度受有限差分限制）

### v0.5.0 — 频率框架
- numeric_frequency.py 数值 Hessian
- 平移/转动投影 + 质量加权对角化
- H2 频率 = 5028 cm⁻¹（框架正确，数值精度有限）

### v0.6.0 — 后端任务扩展
- Result.actual_method 字段
- Gaussian optimize() + frequency()（Opt/Freq 路由关键字）
- Gaussian 几何解析器（Standard orientation 逐行提取）
- Gaussian H2 opt = 1.346 Bohr（正确）
- MolForge 集成：calchem_engine.py 适配器 + 计算面板 Tab 0

---

## 六、硬件与环境

```
CPU:    AMD Ryzen 7 9700X (8核16线程, Zen 5)
RAM:    15.1 GB
OS:     Windows 11
Python: D:\YDW\Trae_Gromacs\miniconda3\python.exe (base, numpy/scipy/pydantic)
Python GUI: D:\YDW\Trae_Gromacs\miniconda3\envs\molforge-webengine\python.exe (PyQt6/VTK)
Gaussian 16: D:\G16W\g16.exe (Rev A.03)
ORCA 6.1.1: E:\ORCA\software\ORCA6.1.1\orca.exe
```

**硬约束：**
- 严禁并行跑多个 heavy 计算任务（必须串行）
- C 盘禁止安装任何软件
- Gaussian mem=12GB, nprocshared=8

---

## 七、已验证做不通的事

| 事项 | 原因 |
|------|------|
| base conda 环境 PyQt6 DLL load failed | QtCore 缺失，必须用 molforge-webengine 环境 |
| Gaussian 输出是 .out 不是 .log | gjf 用单 % 不是双 %% |
| ORCA 6.x 输出主要到 stdout | .out 文件不一定存在 |
| 有限差分梯度精度 | 步长 0.002 Bohr 时 SCF 噪声主导，优化到 1.67 Bohr 而非 1.39 |

---

## 八、战略决策记录

1. **不做 Gaussian 克隆** → 统一平台，多后端适配
2. **不做 OPV 专用** → OPV 是 dogfooding 领域，不是边界
3. **不做 Python 性能优化** → 承认语言瓶颈，选择底层重构
4. **前端用 MolForge** → 不自己写 GUI，直接集成成熟工作台
5. **native 引擎保持教学级精度** → 快速验证理论，不追求生产性能

---

## 九、下一步战略方向

当前 Python 版本作为**数值正确性参考**保留。底层重构需要：

1. **语言选型**：Rust / C++20 / Julia / 混合架构
2. **解析梯度**：HF/DFT 解析一/二/三阶导数
3. **GPU 加速**：Fock 构建和 RI 变换
4. **基组库**：至少 def2-SVP/TZVP/cc-pVDZ
5. **TDDFT**：OSC 激发态计算
6. **PCM 溶剂模型**
7. **并行化**：多线程 + 分布式

详细调研提示词见 `REBUILD_RESEARCH_PROMPT.md`。

---

## 十、项目时间线

| 时间 | 事件 |
|------|------|
| 2026-10-27 | 项目启动，v0.1.0 三后端跑通 |
| 2026-10-28 | v0.2.0 P0 修复，v0.3.0 调度配置 |
| 2026-10-28 | v0.4.0 几何优化框架，v0.5.0 频率框架 |
| 2026-10-29 | v0.6.0 Gaussian Opt/Freq + MolForge 集成 |
| 2026-10-29 | 决定底层重构，启动 Rust/C++ 架构调研 |
