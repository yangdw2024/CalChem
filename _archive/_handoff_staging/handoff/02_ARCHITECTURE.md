# CalChem — Architecture Map

> 每个模块干什么、依赖什么、成熟度如何。

## 顶层包结构

```
calchem/
├── mol/          # 分子表示
├── io/           # 输入输出 (xyz/gjf/gro/mol2/hdf5/checkpoint)
├── basis/        # 基组库
├── integrals/   # 一/双电子积分、RI、屏蔽、direct SCF
├── scf/          # SCF 主循环 (HF/DFT/DIIS/guess/results)
├── dft/          # 泛函、网格、Vxc、Fock、D3
├── post_hf/     # MP2/MP4/CCD/CCSD/CCSD(T)/CASSCF
├── grad/         # 解析梯度 (HF/DFT/analytical)
├── opt/          # BFGS / Berny / IRC / TS search
├── vib/          # 频率、Hessian
├── excited/      # CIS / TDDFT / spectra
├── props/        # 电荷、多极矩、cube、布居
├── solvent/      # PCM / SMD
├── md/           # GROMACS 写出、力场拟合
├── oniom/        # 杂化 QM/QM
├── gpu/          # ⚠️ 空壳 (CuPy 占位)
├── api/          # 高层 Python API (single_point/optimize/freq)
├── cli/          # click 命令行 (calchem run/opt/info/batch)
├── data/         # 默认配置、溶剂参数
├── utils/        # 配置、常量、单位、日志、异常、加速
└── tests/        # 包自带的测试 (与顶层 tests/ 分离)
```

## 关键文件（按重要性）

| 文件 | 行数量级 | 作用 | 成熟度 |
|---|---|---|---|
| `calchem/basis/basis_data.py` | 67 KB | 基组参数表（手录） | ✅ |
| `calchem/integrals/one_electron.py` | 30 KB | S/T/V 矩阵 | ✅ |
| `calchem/integrals/two_electron.py` | 23 KB | ERI / VRR 递推 | ✅ |
| `calchem/integrals/ri.py` | 17 KB | 密度拟合 | ✅ |
| `calchem/scf/hf.py` | 22 KB | RHF/UHF SCF 主循环 | ✅ |
| `calchem/scf/dft.py` | 15 KB | RKS/UKS SCF | ✅ |
| `calchem/scf/diis.py` | 6 KB | DIIS 收敛加速 | ✅ |
| `calchem/dft/functionals.py` | 35 KB | LDA/PBE/TPSS/B3LYP/Vxc | ✅ |
| `calchem/dft/grid.py` | 19 KB | Lebedev 球网格 | ✅ |
| `calchem/post_hf/casscf.py` | 20 KB | CASSCF | ✅ |
| `calchem/post_hf/mp4.py` | 16 KB | MP4 | ✅ |
| `calchem/post_hf/ccsd.py` | 10 KB | CCSD | ✅ |
| `calchem/post_hf/ccsd_t.py` | 3 KB | (T) 修正 | ✅ |
| `calchem/api/high_level.py` | 17 KB | `single_point()` 等 | ✅ |
| `calchem/cli/main.py` | 10 KB | `calchem` 命令 | ✅ |
| `calchem/grad/analytical.py` | 13 KB | 解析梯度 | 🟡 |
| `calchem/opt/ts_search.py` | 11 KB | 过渡态搜索 | 🟡 |
| `calchem/excited/tddft_native.py` | 13 KB | TDDFT | 🟡 |
| `calchem/gpu/fock.py` | **0.9 KB** | ⚠️ 空壳 | ❌ |

## 调用链（一次单点能计算）

```
用户入口:
  calchem.cli.main:cli           (命令行)
  或 calchem.api.high_level:single_point()
        │
        ▼
  calchem.io.input:parse         (读 xyz/gjf)
        │
        ▼
  calchem.mol.molecule:Molecule
        │
        ▼
  calchem.basis.basis_set:BasisSet
        │
        ▼
  ┌─ calchem.integrals.one_electron   → H_core (T + V_N)
  └─ calchem.integrals.two_electron   → ERI (pq|rs)
        │
        ▼
  calchem.scf.guess:make_guess  (H_core 对角化得 P0)
        │
        ▼
  ┌─ HF:    calchem.scf.hf:RHF/UHF.kernel
  └─ DFT:   calchem.scf.dft:RKS/UKS.kernel
              │
              ├─ calchem.dft.grid:LebedevGrid    (DFT only)
              ├─ calchem.dft.functionals:Vxc     (DFT only)
              └─ calchem.scf.diis:DIIS           (收敛加速)
        │
        ▼
  calchem.scf.results:SCFResults
        │
        ▼
  (可选) calchem.post_hf.mp2 / ccsd / ccsd_t / casscf
  (可选) calchem.grad.analytical → calchem.opt.bfgs
  (可选) calchem.vib.frequency
  (可选) calchem.props.charges / multipole / cube
        │
        ▼
  calchem.io.output  → JSON / 文本
```

## 测试布局（2026-10-28 整理后）

```
tests/
├── conftest.py              # 夹具（water_mol, methane_mol, GPU skip）
├── test_io.py / test_molecule.py / test_scf.py   # 原有正式测试
├── unit/        (4)  # import、原子、基函数归一化
├── integrals/  (14)  # ERI、1e、对称性
├── scf/        (26) # H2/H2O/DIIS/收敛
├── dft/        (21)  # 网格/LDA/PBE/B3LYP/Vxc
├── posthf/     (3)  # MP/CCSD(T)/CASSCF + 主回归
├── gui/        (2)   # PyQt6 冒烟
├── debug/     (15)  # 历史调试脚本（不进 CI）
└── scratch/    (21)  # 一次性分析稿 + numpy 原型（不进 CI）
```

**注意**：`calchem/tests/`（包自带）和顶层 `tests/` 是两套，不要合并。

## GUI 在哪？

**不在这个压缩包里。** GUI 在 `D:\YDW\CalChem\gui\`（独立目录），本仓库的 `gui_upgrade/` 只是 GUI 的增量补丁（main_window.py 39KB + result_viewer.py 37KB + 安装说明.md）。打包脚本 `build_calchem_exe.py` 硬编码引用了这两个路径。
