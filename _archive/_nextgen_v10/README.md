# CalChem v1.0 下一代高性能量子化学计算平台 (Rust + C99 + Python 混合架构)

## 项目定位
CalChem 致力于打造在计算速度、收敛稳定性、桌面交互体验三个维度超越传统量子化学软件（Gaussian 16, ORCA 6.1, PySCF, REST）的新一代一体化计算平台。

- **核心计算引擎**：基于 Rust 2021 Edition + `faer` AVX-512 线性代数 + `rayon` 共享内存并行
- **底层积分与泛函**：静态链接 `libcint` C99 SIMD 算子与 `libxc` 600+ 交换关联泛函
- **前端工作台**：MolForge (PyQt6 + VTK 3D 视口) 零拷贝直连
- **部署模式**：Windows 11 One-Folder 单目录免安装独立部署（零 C 盘依赖）

## 目录结构
```
calchem_v1/
├── docs/                               # 完整架构蓝图与数值基准文档
│   ├── ARCHITECTURE_BLUEPRINT.md       # 12 章节完整架构白皮书 (65KB)
│   └── BASELINE.md                     # 严格数值基准阈值 (与 Gaussian/ORCA/PySCF 对齐)
├── Cargo.toml                          # Rust 工作区根配置 (优化 Zen 5 AVX-512)
├── pyproject.toml                      # Maturin 构建配置
├── crates/                             # 8 个核心 Rust 计算子模块
│   ├── calchem-tensor/                 # Layer 1: 内存水位守卫 + 对称打包矩阵 + faer AVX-512
│   ├── calchem-basis/                  # Layer 3: BSE 基组解析 + 球谐/笛卡尔变换 + 壳层聚类
│   ├── calchem-int/                    # Layer 2: 降序 Schwarz 筛选 + 无锁 Direct JK + In-Core RI
│   ├── calchem-scf/                    # Layer 4: SAD 猜想 + EDIIS+CDIIS+SOSCF 状态机 + Libxc 网格
│   ├── calchem-grad/                   # Layer 5: Pulay 能量加权密度矩阵 + 解析梯度
│   ├── calchem-opt/                    # Layer 10: TRIC 平移-转动-内坐标 + RFO 几何优化
│   ├── calchem-prop/                   # Layer 8: AVX-512 极速轨道网格 (<15ms) + 布居分析
│   └── calchem-py/                     # PyO3 桥接层 (释放 GIL + 零拷贝 NumPy 传递)
└── calchem/                            # Python 兼容适配层
    ├── __init__.py
    ├── _core.pyi                       # Rust 导出接口类型存根
    ├── backends/native.py              # 改造为优先调用 Rust 内核的双轨后端
    └── molforge_bridge.py              # MolForge VTK 毫秒级轨道渲染直连接口
```

## 快速构建与测试 (Windows 11)
详细环境配置与构建说明见 `docs/WINDOWS_BUILD_GUIDE.md`。
