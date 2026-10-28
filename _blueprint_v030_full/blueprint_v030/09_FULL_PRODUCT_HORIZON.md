# v1.0 之后的完整产品技术蓝图

本文件描述最终平台方向，帮助当前模块留好扩展边界。**它不是 v1.0 的已实现能力清单，也不要求本地 agent 为赶当前发布而跳过 v1 验收。** 每一项在未来真正启用前都必须有自己的数值/性能/端到端证据。

## A. 方法开发平台（v1.1+）

- `calchem/methods/registry.py`：方法描述包括理论定义、spin/基组/梯度/频率支持、程序版本、验证数据、参考文献和性能范围。
- `calchem/dft/xc_registry.py`：泛函组件分别提供能量密度、密度/梯度/动能密度导数、exact exchange、色散及自旋处理。先验证数值导数，再进入 SCF，随后梯度/频率。
- `calchem/grad/` 和 `calchem/vib/` 提供同一个自检框架：解析与有限差分梯度，Hessian 对称性，平移/转动零模，实验室分子的参考表。
- 对 ωB97X-D、M06-2X、双杂化等每个方法建独立 golden fixtures 与基组/grid 约定，不通过就只允许外部 backend 运行。
- 高阶 Post-HF 的 CCSD、CCSD(T)、CASSCF 均须独立参考对照；“和 MP2 一致”的简化 CCSD 不得宣称成熟 CCSD。

## B. 高性能与规模（v1.2+）

- `integrals/` 先做 profile 与 memory budget、RI/Schwarz 屏蔽 correctness，再考虑 C++/Rust 加速。任何替换都以同一基线与扩展分子矩阵对照。
- GPU 路径先设计设备后端抽象（CPU/GPU array backend），在 8 GB 显存条件下按内存块运行；CuPy 仅作为实验性 optional extra，Fock/ERI/DFT grid 分开验收，不能凭一个 GPU Fock stub 宣称全流程加速。
- JAX/differentiable 路径是独立研究分支，需决定自定义积分导数、数值稳定和编译缓存策略，不进入 Windows v1 默认环境。
- 大分子使用计划预算、渐进 RI、checkpoint/restart 和内存/磁盘配额。必须在有许可、可分享的中型真实体系上记录结果与失败率。

## C. 科研工作流与 agent（v1.3+）

- `workflows/engine.py` 保留有向无环图与状态日志；后续增加 NEB/激发态/电荷拟合/批量筛选，但每一步都有原始 Result 和明确依赖。
- REST-agent 仅通过固定工具接口提出计算、查看能力、读取结果；所有方法替换、资源预算和数据外发由用户配置与 TaskService 验证。agent 的自然语言建议不是科学验收依据。
- `RESTBackend` 根据真实 REST 程序的版本化协议开发；若它是本地应用，使用实际 CLI/文件接口。agent 与该后端可以同时存在，职责不混同。
- OPV 首先要求重组能、轨道/偶极、可选激发态；RESP→GROMACS 只有力场参数齐全并通过 `grompp` 后才作为可运行链路。

## D. 开放生态与扩展点（v2.0+）

- 第三方 backend 插件使用版本化入口点和 capability contract；插件在隔离环境检查其 schema、timeout、error semantics 与资源限制。
- 项目格式迁移器保持读取旧版本；Result schema 的重大含义变化必须升大版本，并有单向迁移或标记无法无损转换。
- Linux/macOS 如有用户需求，在各目标系统单独构建与验收；Windows 仍是主平台。远程调度/云环境需要认证、配额、数据边界和许可证方案，在这些条件满足前不自动开放。

## 各长期模块的准入条件

1. 理论/公式与来源完整，方法名无模糊别名。
2. 单元、数值导数、参考分子、边界情况和资源预算均有记录。
3. Request/Result/CLI/GUI/文档的能力声明一致。
4. 无外部程序/依赖时 core 不受影响；失败保留原始日志。
5. 发布包在目标环境真实验收。任何一项缺失，能力保持 experimental/disabled。
