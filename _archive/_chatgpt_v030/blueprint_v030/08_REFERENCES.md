# 官方技术依据与核对事项

以下资料只用于设计决策。实施时应记录实际使用的软件版本、样例日志、参数和校验结果，不把文档中的一般能力当作本机后端已验收。

| 资料 | 用在蓝图的决策 |
|---|---|
| [cclib Overview](https://cclib.github.io/) | 官方列出 Gaussian 16、ORCA 4.1/4.2/5.0 等系统测试版本；本项目 ORCA 6.1.1 需自己建 fixture |
| [cclib Parsed data](https://cclib.github.io/data.html) | 按解析器逐字段确认坐标、轨道、振动等支持情况，缺字段不伪造 |
| [cclib Changelog](https://cclib.github.io/changelog.html) | 新版 xTB parser 与 ORCA 解析改进，但仍要与本机输出对照 |
| [PySCF Geometry optimization](https://pyscf.org/user/geomopt.html) | PySCF 优化接口可能依赖 geomeTRIC 或 PyBerny，属于 optional extra |
| [PySCF DFT](https://pyscf.org/user/dft.html) | 区分泛函名称、Libxc/XCFun 来源、RKS/UKS 及网格设置 |
| [ORCA 6.1 Geometry optimization](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html) | 外部 Opt 能力、梯度模式与收敛检查 |
| [ORCA 6.1 Vibrational frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html) | AnFreq/NumFreq、平移转动投影、虚频与热化学的限制 |
| [PyInstaller Operating mode](https://pyinstaller.org/en/stable/operating-mode.html) | one-folder 默认、干净目标平台构建、one-file 临时解包行为 |
| [PyInstaller Usage](https://pyinstaller.org/en/stable/usage.html) | Windows 构建目录和 spec 参数 |
| [PyInstaller Qt hook configuration](https://pyinstaller.org/en/stable/hooks-config.html) | 避免一个发布包同时收集多个 Qt binding |
| [Qt QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html) | GUI 主线程和 worker 线程亲和性原则，PyQt6 使用 Qt 同一线程模型 |
| [GROMACS topology format](https://manual.gromacs.org/2025.2/reference-manual/topologies/topology-file-formats.html) | `.top/.itp` 与 force field 参数完整性，不能只有电荷就声称可用 |
| [PyQt6 introduction and licensing](https://riverbankcomputing.com/software/pyqt/intro) | PyQt6 的 GPLv3/商业双许可；发布前做实际分发许可核对 |

来源日期：2026-10-28。本蓝图没有对尚未提供的 REST 量化程序接口作任何假设。
