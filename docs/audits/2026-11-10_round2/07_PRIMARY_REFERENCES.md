# 一手资料、使用范围与出处

检索日期：2026-11-10。资料用于确认公式、标准数据结构和构建契约；实际通过/失败结论来自本轮源码和本机实测，不用网页声称代替运行证据。

| 一手资料 | 本轮用途 |
|---|---|
| [Simple DFT-D3 模型](https://dftd3.github.io/simple-dftd3/module/dftd3_model.html) | 参考 C6、参考 CN、r4/r2、原子对半径与参考权重结构；证明只修平均 C6 或 s8 不足以实现标准 D3 |
| [原始参考态 C6/CN 表](https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/reference.f90) | 无损转换完整 7×7×5356 C6 表；原文件与 hash 保留 |
| [r4/r2 数据](https://raw.githubusercontent.com/dftd3/simple-dftd3/main/src/dftd3/data/r4r2.f90) | C8/C6 比例，替代经验范德华半径拼接 |
| [标准 D3 参数表](https://raw.githubusercontent.com/dftd3/simple-dftd3/main/assets/parameters.toml) | B3LYP/BLYP/PBE/PBE0/HF 等 BJ/zero 参数，发现并修复多个不正确 s8 |
| [mctc-lib 共价半径](https://raw.githubusercontent.com/grimme-lab/mctc-lib/main/src/mctc/data/covrad.f90) | Pyykkö/Atsumi 共价半径及 D3 的 4/3 尺度 |
| [setuptools 扩展模块构建](https://setuptools.pypa.io/en/latest/userguide/ext_modules.html) | 使用 Extension/build_ext 的正式二进制构建契约，而非只修改 purelib 标志 |
| [Maturin 混合项目布局](https://www.maturin.rs/project_layout.html) | 跨平台发布路径的边界；本轮没有下载/安装 maturin 或声称远程 CI 已验证 |
| [PySCF SCF 方法说明](https://pyscf.org/user/scf.html) | SCF 方程、初猜与稳定性检查分离；收敛不是全局电子基态证明 |

上游 GitHub commit API 当时 rate-limited，因此没有伪造下载文件的 commit id。所有实际下载字节以 `crates/calchem-dft/data/d3/source_sha256.json` 识别，并附相同原始源表和 COPYING/COPYING.LESSER。这比引用可变化的 main URL 而不保存内容更可复核。

数据许可证按各原文件保留：s-dftd3 表与转换数据 LGPL-3.0-or-later；mctc-lib 半径 Apache-2.0。CalChem 的 Rust 计算没有调用外部 D3 引擎；表是标准经验模型的参考输入。对 Grimme 模型的科学归因和数据许可并未因“纯 Rust 实现”而消失。

Gaussian 16 和 ORCA 是实际安装的软件，不通过网上能量表拼装本轮结果。第二轮新鲜外部数值主要来自 Gaussian 16；不将上一轮 ORCA、REST 或 PySCF 结果冒充本轮新运行。本轮未安装 PySCF。原始 Gaussian 输入/输出在 evidence/external_raw_round2.zip。
