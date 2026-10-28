# CalChem 量子化学软件 - 打包说明

## 项目概述
CalChem 是一款自主研发的量子化学计算软件，包含完整的 GUI 界面和计算内核。

## 系统要求
- Windows 10/11 64位
- Python 3.9+ (仅打包时需要)
- Visual C++ Redistributable (运行时需要)

## 文件结构

```
D:\YDW\CalChem\gui\           # GUI 源码
├── main.py                   # 程序入口
├── build_exe.py              # 打包脚本
├── requirements.txt          # 依赖清单
├── windows/
│   └── main_window.py        # 主窗口
├── widgets/                  # UI组件
│   ├── molecule_viewer.py    # 3D分子可视化
│   ├── molecule_editor.py    # 分子编辑器
│   ├── parameter_panel.py    # 参数面板
│   ├── result_viewer.py      # 结果展示
│   ├── job_monitor.py        # 作业监控
│   └── convergence_plot.py   # 收敛曲线
├── core/                     # 核心层
│   ├── project.py            # 数据模型
│   ├── calchem_bridge.py     # 内核桥接
│   └── worker.py             # 计算线程
└── styles/
    └── style.qss             # 样式表

D:\YDW\Trae_Gaussian\calchem\  # 计算内核
├── scf/                      # SCF模块
├── post_hf/                  # Post-HF方法
├── dft/                      # DFT模块
├── integrals/                # 积分计算
├── basis/                    # 基组系统
└── ...
```

## 快速启动

```bash
cd D:\YDW\CalChem\gui
python main.py
```

## 打包成 EXE

### 方法一：使用 PyInstaller (推荐)

```bash
# 1. 安装 PyInstaller
pip install pyinstaller

# 2. 运行打包脚本
cd D:\YDW\Trae_Gaussian
python build_calchem_exe.py

# 3. 输出位置
# D:\YDW\CalChem\gui\dist\CalChem.exe
```

### 方法二：手动打包

```bash
cd D:\YDW\CalChem\gui

pyinstaller --name CalChem --onefile --windowed --noconfirm \
    --add-data "D:\YDW\Trae_Gaussian;calchem" \
    --add-data "styles;styles" \
    --hidden-import numpy --hidden-import scipy \
    --hidden-import matplotlib --hidden-import PyQt6 \
    main.py
```

## 依赖安装

```bash
pip install PyQt6 matplotlib numpy scipy pyqtgraph PyOpenGL
pip install pyinstaller  # 打包时需要
```

## 支持的计算方法

| 类型 | 方法 |
|------|------|
| HF | RHF, UHF, ROHF |
| DFT | LDA, PBE, B3LYP, TPSS, BLYP, PBE0 |
| Post-HF | MP2, MP4, CCSD, CCSD(T), CCD, CASSCF |
| 激发态 | CIS |

## 支持的基组
sto-3g, 3-21g, 6-31g, 6-31g(d), cc-pvdz, cc-pvtz, def2-svp, def2-tzvp, d95

## 已验证测试结果

| 测试项 | 结果 |
|--------|------|
| RHF/sto-3g H2O (Szabo几何) | -74.6953 Eh ✓ |
| PBE/sto-3g H2O | -75.5322 Eh ✓ |
| B3LYP/sto-3g H2O | -75.5566 Eh ✓ |
| CCSD(T)/sto-3g H2O | -74.9988 Eh ✓ |
| MP4/sto-3g H2O | -75.0180 Eh ✓ |
| ROHF/sto-3g CH3 | -40.5539 Eh ✓ |
| CASSCF(2,2)/sto-3g H2 | -1.1373 Eh ✓ |

## 性能优化

已实施的优化：
- ✓ ERI转换从O(N^8)优化到O(N^5)
- ✓ CCSD振幅更新向量化
- ✓ Fock矩阵einsum优化
- ✓ DFT网格数值稳定性增强

## 常见问题

**Q: 打包后运行报错找不到模块？**
A: 确保 `--add-data` 参数正确指定了 calchem 内核路径。

**Q: exe 文件太大？**
A: 使用 `--exclude-module` 排除不必要的模块，如 `tkinter`, `matplotlib.tests`。

**Q: 运行时闪退？**
A: 检查 Visual C++ Redistributable 是否已安装。

## 联系方式
CalChem 开发团队