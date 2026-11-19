# CalChem 极速与确定性复现指南 (Reproducibility & Verification Guide)

本指南提供在标准 Windows 11 / Linux 环境下 100% 独立复现 **CalChem Master Engineering Mission v1.0** 全套内核编译、全栈单元回归以及真实有机光伏（OPV）大分子基准计算的标准化操作流程。

---

## 一、 系统与环境准备 (Prerequisites)

### 1. 硬件基准与推荐配置
* **CPU**：AMD Ryzen 7 9700X (8 核 / 16 线程) 或更高规格 x86_64 处理器
* **内存**：物理内存 $\ge 16\text{ GB}$ DDR5
* **显卡 (可选)**：NVIDIA GeForce RTX 5060 Ti / RTX 40 系 / 30 系，计算能力 $\ge 7.5$（搭载 CUDA Toolkit 12.x 或 13.x），计算安全显存上限为 5.0 GB
* **操作系统**：Windows 11 x64 或 Ubuntu 22.04+ LTS

### 2. 软件工具链
* **Rust 编译器**：`rustc 1.84+` 或稳定版
* **Python 环境**：Python 3.11 ~ 3.13（推荐使用 Miniconda/Anaconda 环境）
* **CUDA 驱动与编译器**：NVIDIA CUDA 12.0+ / 13.0+，`nvcc` 位于 PATH 中（如需 GPU 加速）

---

## 二、 核心引擎一键编译与部署 (Build Core Extension)

CalChem 核心采用纯 Rust 编写并通过 PyO3 导出为原生 Python 扩展库。

### 1. 编译 release 版本 Python 原生动态库
在 PowerShell 或终端中执行编译脚本：
```powershell
# 编译并生成 calchem\_core.pyd 并拷贝至源码目录
powershell -File tools/dev/build_ext.ps1
```
* **输出文件**：`calchem\_core.pyd`
* **底层包含模块**：`calchem-model`, `calchem-tensor`, `calchem-basis`, `calchem-int`, `calchem-dft`, `calchem-scf`, `calchem-grad`, `calchem-opt`, `calchem-solv`, `calchem-tddft`, `calchem-vib`, `calchem-cuda`, `calchem-py`。

---

## 三、 Rust 核心测试套件全量验证 (Rust Cargo Verification)

验证所有 12 个 Rust Crates 的底层数值物理逻辑与单元测试：
```powershell
powershell -File scripts/cargo_test.ps1
```
* **验证标准**：所有 12 个 crates 测试 100% 通过（0 失败，0 告警中断）。
* **核心覆盖**：
  * 两电子积分与 Obara-Saika 衰减积分递推精度验证；
  * RI-J 密度拟合度量对称正定性与 $(P|\mu\nu)$ 解析积分；
  * 范围分离泛函与 CAM-B3LYP / $\omega$B97X-D 交换核；
  * 极化连续介质模型 (C-PCM/SMD) 表面电荷响应；
  * 解析核梯度与中心有限差分误差界（$\le 10^{-6}\text{ a.u.}$）。

---

## 四、 Python 全功能回归套件运行 (Python Pytest Suite)

运行全套 Python 顶层接口与端到端回归测试：
```powershell
powershell -File scripts/run_pytest.ps1 tests/unit
```
* **执行时间**：约 6~7 分钟（涵盖多分子几何优化、过渡态追踪、二阶频率计算与激发态光谱）。
* **通过指标**：114 / 114 个单元测试全绿通过（100% Passed）。

### 针对单独关键模块的专项测试指令：
```powershell
# 1. 验证真实 OPV 分子体系与化学真实数据对齐 (IC-2F, BTP, Y6, PM6)
powershell -File scripts/run_pytest.ps1 tests/unit/test_p6_opv_molecular_validation.py

# 2. 验证 GPU 加速与 RI-J 库仑密度拟合
powershell -File scripts/run_pytest.ps1 tests/unit/test_gpu_and_density_fitting.py

# 3. 验证范围分离泛函 CAM-B3LYP 与 wB97X-D
powershell -File scripts/run_pytest.ps1 tests/unit/test_opv_cam_b3lyp_and_dimer.py

# 4. 验证解析梯度、溶剂化模型与 TD-DFT 激发态
powershell -File scripts/run_pytest.ps1 tests/unit/test_p5_gradients_solvation_tddft.py
```

---

## 五、 真实有机光伏（OPV）大分子验证复现

运行对标真实 Gaussian 16 记录（`E:\YDW\ChemDatas\Log`）的 OPV 计算：
```powershell
$env:PYTHONPATH = "E:\YDW\Trae_Gaussian"
python -c "
from calchem.workflows.opv import OPVWorkflow, CHEMDATAS_REFERENCE_DATA, get_real_opv_system
wf = OPVWorkflow(backend='native')
res = wf.run_system_benchmark('IC-2F', method='b3lyp', basis='sto-3g', include_d3=True)
print(f'IC-2F Energy: {res[\"scf_energy_hartree\"]:.6f} Eh, HOMO: {res[\"homo_ev\"]:.2f} eV, LUMO: {res[\"lumo_ev\"]:.2f} eV')
"
```
* **预期输出**：
  * 计算在 ~14 秒内收敛；
  * 能量稳定在约 `-892.418 Eh`；
  * 前线轨道能级（HOMO/LUMO）与真实 Gaussian 16 计算记录（`ydw-IC-2F-opt.log`）精准对齐。

---

## 六、 常用环境变量与安全控制开关

| 环境变量 | 默认值 | 作用说明 |
| :--- | :---: | :--- |
| `CALCHEM_DISABLE_GPU=1` | 0 | 显式强制禁用 CUDA GPU 加速，回退至纯 CPU 多线程计算 |
| `CALCHEM_DISABLE_LOCK=1` | 0 | 在单机单任务评测或无写入冲突时禁用重型作业全局锁 |
| `RAYON_NUM_THREADS=8` | 自动匹配物理核 | 限制底层多线程并行 Worker 数目，控制峰值内存消耗 |
| `CALCHEM_MAX_GPU_MEM_MB=5000` | 5000 | 设定 GPU 显存分配硬上限，防止大分子突发 OOM 崩溃 |
