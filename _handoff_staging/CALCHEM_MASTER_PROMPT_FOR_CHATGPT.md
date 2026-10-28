# CalChem v0.1.0 → v0.2.0 架构演进主提示词

你是 CalChem 的架构师。CalChem 是一个统一量子化学计算平台，不是 Gaussian 克隆。

## 项目身份

- **名称**: CalChem
- **定位**: 统一量子化学工作台（Python API + CLI + PyQt6 GUI）
- **核心**: 自带原生 Python 量化引擎（RHF/UHF/DFT/MP2/CCSD/CCSD(T)），同时通过适配层接入 Gaussian 16/ORCA/PySCF/xTB/REST 作为可替换后端
- **长期目标**: 从底层方法框架上超越现有量子化学程序（不仅仅是 UI）
- **dogfooding 领域**: OPV/OSC（有机光伏），但产品边界不限于此

## 当前状态（v0.1.0 已完成）

### 已验证数值（H2O/STO-3G RHF Szabo 几何）
| 后端 | 能量 (Eh) | 与 native 差 |
|------|-----------|-------------|
| Native | -74.695319177132 | 基准 |
| Gaussian 16 A.03 | -74.695319172900 | 4.2e-9 |
| ORCA 6.1.1 | -74.695319167759 | 9.4e-9 |

### 代码结构
```
calchem/
├── core/           # Pydantic v2 数据契约
│   ├── request.py      # CalculationRequest, MoleculeSpec, ResourceLimits
│   ├── result.py       # Result, OrbitalInfo, JobStatus
│   ├── calculator.py   # Calculator ABC + 显式注册表
│   ├── capabilities.py # BackendCapabilities
│   ├── errors.py       # CalChemError 继承体系
│   └── units.py       # 换算常数
├── backends/
│   ├── native.py      # NativeCalchem (直接调 scf/hf, dft, post_hf)
│   ├── gaussian.py     # Gaussian 16 适配 (gjf→g16→log解析)
│   ├── orca.py        # ORCA 6.1.1 适配 (inp+xyz→orca→stdout解析)
│   ├── pyscf.py       # 注册 stub (未装)
│   ├── xtb.py         # 注册 stub (无 binary)
│   ├── rest.py        # 注册 stub (无 endpoint)
│   └── external.py   # 外部程序基类
├── jobs/
│   ├── models.py      # Job 状态机
│   └── queue.py       # 单并发 JobQueue
├── api/
│   └── service.py    # TaskService 统一入口
├── io/
│   └── project.py     # .calchem 项目目录格式
├── config/
│   └── models.py      # AppConfig Pydantic
├── gui/
│   ├── main_window.py # PyQt6 主窗口
│   └── worker.py     # QThread worker
├── workflows/
│   └── opv.py         # OPV dogfooding 骨架
├── cli.py            # Click+Rich CLI
└── __main__.py       # python -m calchem
```

### 技术栈
- Python 3.10+, Pydantic v2, NumPy/SciPy
- CLI: Click + Rich
- GUI: PyQt6 + Matplotlib 3D
- 打包: PyInstaller one-folder (已产出 238MB)
- 数据: JSON, 非 C 盘路径

### 硬件约束
- AMD Ryzen 7 9700X (8核16线程), 15.1GB RAM
- Gaussian 16: D:\G16W (mem=12GB, nprocshared=8, 串行)
- ORCA 6.1.1: E:\ORCA\software\ORCA6.1.1
- C 盘禁止装软件

## 你需要规划的事（v0.2.0 → v1.0）

### 1. 计算内核增强
- 几何优化（解析/数值梯度）
- 频率计算（Hessian, 红外光谱）
- 更多泛函（ωB97X-D, M06-2X, etc.）
- DFT 数值积分（ Lebedev 网格, 交换相关）
- 极化/扩散基组支持
- 对称性识别

### 2. 后端扩展
- PySCF 真实绑定（替代 stub）
- xTB 真实调用
- REST API 客户端（对接 Qwen3-8B agent）
- cclib 集成替代手写 regex parser

### 3. GUI 完整度
- 项目打开/保存（.calchem 格式已在 io/）
- 作业队列面板（当前只有日志）
- 结果 tabs（摘要/轨道/几何/频率/日志）
- 收敛曲线图表
- 轨道能级图（HOMO/LUMO）
- 设置/诊断页

### 4. 工作流
- OPV 完整流水线：优化→单点→HOMO/LUMO→偶极→重组能
- GROMACS topology 导出
- 批量分子计算

### 5. 工程化
- pytest 测试覆盖（当前只有 acceptance.py）
- Ruff/mypy 类型检查
- 版本锁定 requirements
- CI (GitHub Actions Windows runner)

## 输出要求

请给出：
1. v0.2.0 → v1.0 的阶段划分（每个阶段的目标、产出、验收标准）
2. 每个模块的具体任务拆解（文件级）
3. 关键技术决策建议
4. 优先级排序
5. 风险评估
