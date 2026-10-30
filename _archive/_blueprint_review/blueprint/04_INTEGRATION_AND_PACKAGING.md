# CalChem 集成、数据目录与 Windows 打包

## 1. 入口集成

GUI、CLI、Python API 必须共用 `TaskService.submit(request)` / `TaskService.run(request)`。只有 `TaskService` 能验证请求、检查能力、创建 run directory、调用 backend、规范异常并持久化 Result。

### Python API

```python
from calchem.core import CalculationRequest
from calchem.api.service import TaskService
from calchem.mol import Molecule

mol = Molecule.from_xyz_file(r"D:\projects\water.xyz")
request = CalculationRequest(
    task="single_point",
    backend="native",
    method="rhf",
    basis="sto-3g",
    molecule=mol,
)
result = TaskService().run(request)
print(result.model_dump_json(indent=2))
```

此调用示例为接口目标；若相应模块尚未实现，agent 不得把本示例当作已能运行的当前代码。

### CLI

```powershell
calchem backends --json
calchem doctor --output D:\CalChemData\diagnostics.json
calchem run D:\projects\water.xyz --task single_point --backend native --method rhf --basis sto-3g --output D:\projects\water.result.json
calchem run D:\projects\water.xyz --task single_point --backend gaussian --method b3lyp --basis 6-31g(d) --output D:\projects\water.g16.result.json
calchem job show <job-id>
calchem project validate D:\projects\water.calchem\project.json
```

CLI JSON 模式的 stdout 只能输出 JSON；日志写 stderr 或日志文件，避免污染管道。

## 2. GUI 与引擎集成

- GUI 是当前独立目录 `D:\YDW\CalChem\gui` 的延续，封装为 PyQt6 app 包。
- GUI 通过公开 Python API 调用 TaskService；不直接调用 `scf.hf.RHF`，不解析 Gaussian/ORCA 文本。
- 计算通过单一 Qt worker thread 执行；GUI 主线程只更新 widgets。
- UI 和核心间传递 Pydantic Request/Result 或已校验的普通数据对象；禁止跨线程传递 QWidget、numpy mutable references 或进程句柄。
- engine 与 GUI 的依赖边界通过公开版本号和 `CALCHEM_RESULT_SCHEMA_VERSION` 检查；版本不兼容时在启动时给出具体错误。
- GUI 崩溃不应损坏已写盘的 project/result；计算仍运行与否由 job supervisor 策略决定，并在关闭提示中说明。

## 3. 可复现项目格式

建议文件夹扩展名 `.calchem`，本质是目录：

```text
water.calchem/
  project.json            # schema、项目 id/name、默认 molecule、任务列表
  molecules/water.xyz
  jobs/<job-id>/request.json
  jobs/<job-id>/result.json
  jobs/<job-id>/manifest.json
  jobs/<job-id>/input.gjf | input.inp
  jobs/<job-id>/stdout.txt
  jobs/<job-id>/stderr.txt
  jobs/<job-id>/program.log
  jobs/<job-id>/artifacts/...
```

`manifest.json` 至少记：CalChem/schema/backend 版本、程序版本、完整命令数组（脱敏）、method/basis/xc、分子快照、输入文件 hash、开始时间、结束时间、耗时、退出码、状态、raw files 相对路径。用户设置路径和秘密不得写入 manifest。

默认数据路径 `D:\CalChemData`，应用安装/便携路径 `D:\CalChem`。支持 `CALCHEM_HOME` 和 GUI 设置覆盖。每个任务的 cwd 在该 project/jobs 下创建唯一目录；禁止覆盖已有任务目录。

## 4. 配置与后端发现

配置优先级：CLI 参数 > 当前 project 设置 > 用户配置 > 默认值。

全局配置包含：

- `data_root`、`run_root`、`cache_root`、`log_root`。
- Gaussian executable、ORCA executable、xTB executable；PySCF 是否可导入。
- `default_backend`、`default_method`、`default_basis`、默认 XC。
- `max_heavy_jobs=1`、`max_cpu_cores=8`、`memory_budget_mb`、backend timeout。
- 输出日志级别、保留策略、是否打开结束通知。

后端发现是显式诊断：定位 binary、取得版本、执行最小 `--version` 或等效检查。不能以文件存在就判断程序工作正常；不运行 heavy smoke calculation 来探测。配置中没有路径时，UI 显示未配置并提供手动选择。

## 5. Python 依赖和安装分组

基础运行：numpy、scipy、h5py、PyYAML、click、rich、pydantic。  
开发：pytest、pytest-cov、ruff、mypy、pre-commit。  
`external` extra：cclib。  
`pyscf` extra：pyscf。  
`symmetry` extra：spglib。  
GUI extra/独立环境：PyQt6、Matplotlib、pyqtgraph（按现有 GUI 导入盘点保留实际需要项）。  
GPU extra：保持可选且标注未验证，不能影响 CPU-only 安装。

逐项检查 `mendeleev`、MDAnalysis、tables 等既有声明。仅用到可选功能的移到 optional extras；没有源码引用且无经确认的动态导入时才移除。发布依赖在 Windows 干净环境中安装验证，禁止 `pip install` 到 C 盘环境。

## 6. Windows 构建策略

1. 在 Windows 11 构建机 `D:\CalChemBuild\venv` 创建干净 venv；固定 Python patch 版本和 requirements lock。
2. 先构建 CLI，再构建 GUI one-folder。GUI 用 PyInstaller spec file 显式收集 `calchem.data`、基组参数、Qt plugins、字体资源和 Matplotlib data。
3. 构建目录 `D:\CalChemBuild\work`，dist `D:\CalChemBuild\dist`，不可写入仓库 `C:\` 临时目录以外的系统目录。把 `%TEMP%` / PyInstaller work path 配到 D 盘。
4. PyInstaller 默认 one-folder；第一版交付 ZIP 解压后运行。one-file 会启动时临时展开，且不利于诊断，除非后续需求明确并验证自定义 temp 位置，否则不做。
5. 构建后在干净 Windows 用户账户测试，不依赖开发机 PATH、编辑器、conda base、已安装的开发依赖。
6. 发布目录包含：`CalChem.exe`、必要 Python/Qt runtime、`README_FIRST_RUN.md`、`LICENSES/`、版本文件和示例项目。不要包含 Gaussian/ORCA/xTB 安装文件或作者绝对路径。
7. 用 SHA-256 manifest 记录发布 ZIP 校验和；版本号与 `calchem.__version__`、GUI About、CLI `--version` 一致。
8. 路径测试必须覆盖空格、中文、长路径和只读目录。对 Gaussian/ORCA 路径采用 `Path` 对象和参数列表，不拼接命令字符串。

PyInstaller 官方说明默认生成 one-folder 目录，建议先用 one-folder 调试；若目标包含其他操作系统，需在相应平台分别构建。Windows 产品的发布构建必须在 Windows 环境完成。

## 7. CI 与验收环境

- GitHub Actions 在 Windows runner 执行 Python unit/API/parser tests；不要求 runner 安装 Gaussian/ORCA license。
- Linux 可跑纯 Python 层与接口测试，GUI/external tests skip 并注明原因。
- 禁止 `pytest-xdist` 执行 heavy tests；一般测试默认也不启用 `-n auto`。
- Pull/commit gate：format/lint、公开 API type check、unit、Result schema、input writer、parser fixtures、快速 numerical baseline。
- Nightly/手动 gate：完整基线与安装了软件的外部集成测试串行执行。
- release gate：打包 app 在目标 Windows 干净环境启动并执行一个轻量 native job，外部后端单独逐个验证。

## 8. 更新与诊断

第一版手动下载 ZIP 替换目录，不做自动联网更新。GUI 可导出脱敏诊断 JSON，字段包括 CalChem/Python/OS 版本、可用后端/版本探测状态、数据目录可写状态、最近错误 code。诊断包发送前必须让用户清楚看到文件内容与目标；默认只保存在本机。
