# CalChem 当前进展审查（2026-11-16）

## 结论

相对 11 月 12 日，三项针对性加固已进入源码并通过当前版本的 Rust 回归；22 原子 IC-2F 的 CPU 与开启 GPU 选项的 RI-J 双路径计算也在本机通过。当前仍是研究开发级软件。测试名称和交付叙述中有几处证据外推，不能因此宣称大分子生产能力或多泛函 Gaussian 高精度对齐。

## 本轮独立实测

| 对象 | 结果 | 原始证据 |
|---|---|---|
| Rust workspace | 139 passed，0 failed，68 个测试组，exit 0 | `cargo-current.log`、`cargo-current.exit.txt` |
| OPV/GPU/RI-J Python 专项 | 8 passed，0 failed，592.81 s，exit 0 | `pytest-targeted.log`、`pytest-targeted.exit.txt` |
| CUDA 算子 | device available=true，7.34 GB free/8.55 GB total；GPU 与 CPU 缩并最大绝对差 1.679e-15；3 passed，exit 0 | `cuda-current.log`、`cuda-current.exit.txt` |
| 构建身份 | `_core.pyd` 与 `build_target/release/_core.dll` SHA-256 同为 `105262A8CE0D5FB59530926D5FEBAFCA068CBEEE4562AEFFCE45EF3B043F67E0` | 本次 Get-FileHash |

Rust 和 Python 专项均是现有源码与已安装二进制的当前运行结果。Python 全量 tests/unit 在本次改动后尚未独立重跑；11 月 12 日的 119 项结果属于上一版扩展。

## 四项整改逐项判断

1. **CUDA 生命周期：明显改善。** `CudaDriverApi` 增加单次初始化尝试、失败清理与析构时 primary context release / FreeLibrary；当前 GPU 算子实测成功。粘贴所说“PTX JIT 失败时立即释放主上下文”不完全对应代码：模块加载失败在缩并函数中返回错误，设备上下文仍由全局 Driver API 持有至进程结束。尚无该失败路径的注入测试。
2. **磁盘 DIIS：避免 RAM 历史膨胀，但错误仍被隐藏。** `try_push_and_extrapolate` 会返回 I/O 错误并清空 RAM 列表，方向正确；SCF 四种求解器实际调用的仍是非 Result 包装 `push_and_extrapolate`，它在 I/O 错误时返回原始 Fock，没有把错误交给 SCF 调用者。新增测试只独立验证一个无效目录会报错，随后以有效默认目录运行 `new_disk`，没有注入当前 DIIS 实例的磁盘读写故障，所以“故障传播到 SCF”未被证明。`CALCHEM_DATA` 分支直接使用该目录，硬编码 E 路径不存在时仍退回系统 temp；“任何运行方式绝不写 C”不成立。
3. **RI-J 截断梯度：保护性拒绝已实现。** 度量求逆现在记录截断模态数，梯度入口在计数非零时返回错误。源码截断阈值是 `1e-12 * lambda_max`，粘贴中的 `1e-10` 不符。新增测试手动把计数设成 1，只证明拦截分支；还需用真实近线性相关辅助基组验证计数产生、SCF 结果和梯度拒绝贯通。
4. **OPV 双路径与外部对照：计算确实运行，证据范围有限。** 22 原子 IC-2F 的两个 RI-J SCF 在本次 Python 专项通过，测试要求总能差小于 1e-7 Eh、能隙差小于 1e-5 eV。但测试未检查 CUDA kernel 调用成功标志；若 `build_coulomb` 自动回退 CPU，依旧可能通过。当前 CUDA 算子专项独立证明 GPU 能执行，并未记录该 IC-2F 作业每次 J 构建是否使用 GPU。粘贴中的两组具体能量未写入测试日志，本次只能证实断言通过。

## “外部基准交叉验证”的准确范围

`test_p6_external_reference_cross_validation` 对 CHCl3 的核排斥能使用测试代码直接做 Z_i Z_j/r_ij 求和，再与硬编码的 Gaussian 日志数比较；这没有调用 CalChem 的 CHCl3 积分/SCF。H2O STO-3G RHF 对照一个硬编码能量，UHF 与 RHF 互比；6-31G(d) B3LYP/PBE0/CAM-B3LYP 只检查均落在 -76.50 到 -76.10 Eh 的宽区间，并没有逐一对照同几何、同基组、同格点的 Gaussian 总能。D3(BJ) 仅断言能量为负。故“多泛函高精度外部交叉认证”不成立。

## 大体系与物理边界

- BTP/Y6/PM6 测试读取结构和保存的参考数据；p4D18 测试读取 Gaussian 日志并计算模型内存数字，不执行该体系原生 SCF。6224 AO 单张 f64 矩阵约 295.55 MiB，16 张约 4.62 GiB；这不覆盖整个进程的 Fock、DIIS、对角化、网格和 RI 度量峰值。726–2000 原子/16 GB 目标仍未验收。
- 标准 wB97X-D、SMD、RI-J TDDFT 响应、范围分离 SCRF 等上一轮未实现边界未因本次四项修复而自动补齐。
- GPU 算子双精度正确性和单台机器可用性已实测；大分子端到端加速比、显存峰值与跨驱动回退仍缺证明。

## 当前定位

工程进展可评为 **研究 Alpha 后期**：小体系主链和关键回归稳定，GPU RI-J 核可运行，三项高风险边界有实质修复。科学可靠性仍需按具体方法、基组和分子逐项认证；大体系工业生产准入仍未达到。建议下一关先补真实 I/O 故障与秩截断端到端测试、把 GPU 使用状态写入 SCF 诊断，再做同输入外部能量/梯度基准和可控内存压力测试。

## 操作记录

本次未改业务源码、未清理工作区、未安装依赖。只在 E 盘本目录新增计划、审查报告和原始测试日志。旧 11 月 12 日报告为历史快照，本报告以当前 11 月 16 日源码及本次运行结果为准。
