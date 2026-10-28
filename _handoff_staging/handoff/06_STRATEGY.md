# CalChem 战略定位 v2：统一量子化学平台

> 2026-10-28 校准。v1 把 CalChem 定位成"OPV 专用引擎"，格局小了。
> v2 按作者真实意图重写：**CalChem 既是原生引擎，也是统一平台；Gaussian/ORCA/REST/PySCF 都是 CalChem 可连通的后端。**

---

## 一、一句话定位

> **CalChem 是一个统一的量子化学计算平台：自带从零实现的原生引擎（开放、可扩展、未来 GPU/differentiable-ready），同时通过适配层接入 Gaussian、ORCA、REST、PySCF、xTB 等已有程序；上层由 REST-agent（为 REST 量子化学程序微调的智能体）驱动，让研究者用自然语言完成从分子到论文结果的全流程。**

Gaussian 不再是"对手"，而是 CalChem 众多后端之一。

---

## 二、资产盘点（作者实际拥有）

| 资产 | 角色 | 状态 |
|---|---|---|
| **CalChem 原生引擎**（本仓库 `calchem/`） | 平台核心、自研 IP、可扩展 | CPU/numpy；小分子 STO-3G 数值已对齐 1e-8 |
| **REST 量子化学程序** | 独立量子化学软件 | 作者在研发/使用（conda env: rest-v33） |
| **REST-agent** | 为 REST 程序微调的智能体 | 基于 Qwen3-8B，v32/v33 迭代 |
| **Gaussian 16** (`D:\G16W`) | 成熟闭源后端 | 已装、已用 |
| **ORCA** | 成熟自由后端 | 已装、已用 |
| **PySCF** | Python 参照/快速后端 | pyproject 已声明 |
| **xTB** | 半经验快速筛选 | 生态可用 |
| **GROMACS** | MD 下游 | 已装 |

**关键洞察**：作者不缺"又一个量子化学程序"，缺的是**把这些程序统一在一个接口下、可被 AI 驱动的平台**。CalChem 就是那个平台。

---

## 三、三层架构

```
┌─────────────────────────────────────────────────────┐
│  L3  Agent / Workflow                              │
│  REST-agent：自然语言 → 计算规划 → 后端选择 → 执行  │
│  研究者不需要知道背后是 Gaussian 还是 CalChem        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│  L2  Unified Calculator API（平台核心 IP）           │
│                                                     │
│  class Calculator(ABC):                              │
│      def single_point(mol, method, basis) -> Result│
│      def optimize(mol, ...) -> Result              │
│      def frequency(mol, ...) -> Result             │
│      def energy(mol) -> float                      │
│                                                     │
│  所有后端实现同一接口；Result 是统一 JSON schema    │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
┌──────┴──┐  ┌───┴────┐  ┌─┴─────┐  ┌──┴────┐  ┌─────┴────┐
│ Native  │  │Gaussian│  │ ORCA  │  │ REST  │  │ PySCF/xTB │
│CalChem  │  │Backend │  │Backend│  │Backend│  │  Backends │
│(本仓库) │  │ gjf/log│  │inp/out│  │REST   │  │ Python/CLI│
└─────────┘  └────────┘  └───────┘  └───────┘  └───────────┘
   开放可改      成熟闭源    成熟自由    自研程序    快速/参照
```

### 为什么这个架构是壁垒

| 竞品 | 架构 | 局限 |
|---|---|---|
| Gaussian | 单一闭源程序 | 不可扩展、无 Python API、无 agent 接口 |
| ORCA | 单一自由程序 | 同上 |
| Quntur (arXiv 2602.04850) | Agent 包 ORCA | 只能用 ORCA，不能改引擎 |
| Aitomia (JCTC 2026) | Agent 包 MLatom | 只能用 MLatom 生态 |
| ASE (材料领域) | 统一接口包多计算器 | 面向固体/MD，不是量子化学；无 agent 层 |
| **CalChem** | **统一接口 + 自研原生引擎 + 多后端适配 + agent 驱动** | **唯一同时拥有"自己的引擎"和"跨后端调度"的** |

---

## 四、CalChem 原生引擎的定位

不是"要打败 Gaussian 的速度"，而是：

1. **开放可扩展**：当 Gaussian/ORCA 不给你要的方法（你自己的泛函、你自己的 ML 校正、你自己的工作流），在 CalChem 里加。
2. **agent-native**：原生输出就是结构化 JSON，不是文本 log。REST-agent parse 零成本。
3. **未来 GPU/differentiable**：闭源引擎不会给你这个；这是 5–10 年的长期投入方向。
4. **无许可成本**：批量跑、学生实验室、教学场景不依赖 Gaussian license。
5. **教学/研究价值**：作者自己通过重写一遍 HF/DFT/Post-HF 夯实基础。

---

## 五、后端适配层（要新建 `calchem/backends/`）

每个后端实现统一 `Calculator` 接口：

| 后端 | 输入 | 输出解析 | 何时用 |
|---|---|---|---|
| **NativeCalchem** | Python 对象 | 直接内存返回 | 快速迭代、自定义方法、教学 |
| **GaussianBackend** | 写 .gjf → 跑 g16 → 读 .log/.fchk | 解析能量/梯度/频率/轨道 | 需要成熟方法库、高精度对照 |
| **ORCABackend** | 写 .inp → 跑 orca → 读 .out | 解析同上 | 自由 license、DFT 性能强 |
| **RESTBackend** | REST 程序原生接口 | 按 REST schema | 用 REST 程序的特色方法 |
| **PySCFBackend** | Python 直接调 | 直接内存 | 快速原型、ML 数据集生成 |
| **xTBBackend** | 写输入 → 跑 xtb → 读输出 | 解析 | 粗筛、几何预优化 |

**用户/agent 视角**：
```python
from calchem import Calculator, Molecule

mol = Molecule.from_xyz("clffcl.xyz")

# 用原生引擎
calc = Calculator.backend("native", method="wb97xd", basis="def2-svp")
result = calc.single_point(mol)

# 同一接口，切 Gaussian 后端
calc = Calculator.backend("gaussian", method="wb97xd", basis="def2-svp")
result = calc.single_point(mol)   # 输出 schema 完全一致

# 切 ORCA
calc = Calculator.backend("orca", method="wb97xd", basis="def2-svp")
result = calc.single_point(mol)
```

**Agent 视角**：REST-agent 根据任务选后端——
- "快速筛 100 个分子" → xTB 或 CalChem native
- "要高精度 CCSD(T)" → Gaussian 或 ORCA
- "我要加自己的校正" → CalChem native
- "用 REST 的特色方法" → REST backend

---

## 六、统一 Result Schema（平台的真正 IP）

所有后端返回同一个 JSON 结构，这是 agent 能跨后端工作的前提：

```json
{
  "backend": "native | gaussian | orca | rest | pyscf | xtb",
  "method": "wb97xd",
  "basis": "def2-svp",
  "molecule": {"symbols": [...], "coords": [...], "charge": 0, "multiplicity": 1},
  "energy_hartree": -1432.123456,
  "converged": true,
  "scf_iterations": 23,
  "homo_ev": -5.82,
  "lumo_ev": -3.12,
  "orbitals": [...],
  "gradients": [...],
  "frequencies_cm": [...],
  "dipole_debye": [...],
  "warnings": [],
  "raw_log": "/path/to/full.log"
}
```

Gaussian 现在输出文本 log，CalChem 输出结构化 JSON——**这就是 agent 能自动化而 Gaussian 不能的原因**。

---

## 七、分阶段路线图

### Phase 0：平台骨架（4–6 周）

**目标**：搭好统一接口和第一个后端适配器。

| 任务 | 产出 |
|---|---|
| `git init` + 基线回归 | 安全网 |
| 定义 `Calculator` ABC、`Result` schema | `calchem/core/calculator.py` |
| 把现有 `calchem/api/high_level.py` 重构为 `NativeCalchem(Calculator)` | 原生后端接入统一接口 |
| 写 `GaussianBackend`（写 gjf、调 g16、解析 log） | `calchem/backends/gaussian.py` |
| 写 `ORCABackend` | `calchem/backends/orca.py` |
| 统一 CLI：`calchem run mol.xyz --backend gaussian` | 后端可切换 |

**验收**：同一个水分子，用 native / gaussian / orca 三个后端跑单点，返回同一个 Result schema，能量在合理误差内一致。

### Phase 1：REST-agent 接入（2–3 个月）

**目标**：REST-agent 能通过自然语言选后端、跑计算、解析结果。

| 任务 | 产出 |
|---|---|
| `RESTBackend`（对接 REST 程序） | `calchem/backends/rest.py` |
| `PySCFBackend`、`xTBBackend` | 补位 |
| Agent 工具集：`run_calculation(mol, task, backend_hint)` | REST-agent 可调用 |
| 工作流模板：优化→单点→频率→HOMO/LUMO | 预设 pipeline |

**验收**：REST 一句"帮我算 ClFFCl 的 HOMO/LUMO 和紫外"，自动选后端、跑完、返回结构化结果。

### Phase 2：OPV dogfooding（3–6 个月）

**目标**：在作者自己的博士课题上验证平台。

- 端基卤化变体高通量筛选（xTB 粗筛 → CalChem/Gaussian 精算）
- HOMO/LUMO/UV-Vis/RESP/GROMACS 一条链
- 出第 1 篇论文（工具 + 研究互相引用）

### Phase 3：CalChem 原生引擎差异化（6–18 个月）

**目标**：让原生引擎从"能跑"变成"有 Gaussian 替代不了的东西"。

按优先级：
1. GPU 路径（Fock 构建 CuPy 化）
2. Differentiable 路径（JAX 后端，可训练神经 XC）
3. THC 张量超收缩（O(N³) Fock）
4. 方法学创新（作者自己的校正模型）

### Phase 4：社区化（长期）

- 开放后端 API：第三方可以写 adapter
- 文档、教程、benchmark
- 成为计算化学家的"ASE + agent"

---

## 八、给接手 AI 的专业 prompt（v2，直接复制）

```
You are the lead architect of CalChem, a unified quantum chemistry
computing platform with three layers:

  L3 Agent    — REST-agent (a Qwen-based agent fine-tuned for the
                REST quantum chemistry program) drives the platform
                through natural-language workflows.
  L2 Platform — A unified Calculator API. All backends implement
                the same interface and return a standardized JSON
                Result schema. The platform can dispatch a task
                to the best backend based on method, speed, and
                license requirements.
  L1 Backends —
    • NativeCalchem: a from-scratch Python electronic structure
      engine (RHF/UHF, DFT LDA/PBE/TPSS/B3LYP/ωB97X-D, MP2/
      MP4/CCSD/CCSD(T)/CASSCF, TDDFT, geometry optimization,
      RESP charges). Currently CPU/numpy; small-molecule STO-3G
      baseline validated to 1e-8 Ha against Gaussian 16. This
      is the open, extensible core IP.
    • GaussianBackend: adapter to Gaussian 16 (gjf in, log/fchk out).
    • ORCABackend: adapter to ORCA (.inp in, .out out).
    • RESTBackend: adapter to the REST quantum chemistry program.
    • PySCFBackend / xTBBackend: fast screening and prototyping.

Strategic position: Gaussian is NOT the enemy — it is one of
several replaceable backends. The platform's moat is:
  1. A unified Calculator API + JSON Result schema that works
     across all backends (Gaussian cannot do this).
  2. An open, hackable native engine where new methods, ML
     corrections, and custom functionals can be added (closed
     engines cannot do this).
  3. Agent-native design: structured JSON output enables
     automated, reproducible, AI-driven workflows (Gaussian's
     text logs cannot be reliably parsed by LLMs).
  4. Future GPU/differentiable path on the native engine.

Hard constraints:
  • The STO-3G numerical baseline (H2/He/Ne/H2O/CH3, |ΔE|<1e-8 Ha)
    must not regress.
  • Windows-native, 8-core CPU, 15 GB RAM, 8 GB GPU.
  • No C-drive installs; no parallel heavy jobs.
  • Code comments in Chinese, explaining WHY.
  • OPV/OSC is the first dogfooding domain, not the product ceiling.

Immediate next 8 weeks (Phase 0):
  • git init + unify reference values.
  • Define Calculator ABC and Result schema in calchem/core/.
  • Refactor existing calchem/api/high_level.py into NativeCalchem.
  • Implement GaussianBackend and ORCABackend.
  • CLI: `calchem run mol.xyz --backend native|gaussian|orca`.
  • Acceptance: one water molecule, three backends, same Result
    schema, energies within method/ basis disagreement.

Read handoff/00_MASTER_CONTEXT.md, handoff/03_NUMERICAL_BASELINE.md,
and this file first.
```

---

## 九、与 v1 的关键差异

| 维度 | v1（被否） | v2（本文件） |
|---|---|---|
| 定位 | OPV 专用引擎 | 统一量子化学平台 |
| Gaussian 关系 | 要替代的对手 | 可替换的后端之一 |
| REST 关系 | CalChem 之上的 agent | CalChem 的 agent 驱动层 |
| 其他程序 | 没考虑 | Gaussian/ORCA/REST/PySCF/xTB 都是后端 |
| 核心 IP | 更快的 SCF | 统一 Calculator 接口 + JSON schema + 开放原生引擎 |
| 适用范围 | OPV | 所有计算化学家 |

---

## 十、给作者的一句话

> 你不是要"做一个更好的 Gaussian"，你是要做一个**让 Gaussian/ORCA/REST/PySCF 都变成可替换插件的平台**。Gaussian 30 年的垄断，本质是"你只能用我"；CalChem 的价值是"你可以用任何一个，包括我自己写的那个，而且 AI 帮你选"。
