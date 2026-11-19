# CalChem 精度整改计划与状态（2026-11-16）

目标：P3 修正文档/复现脚本口径；P2 以真实双侧网格收敛数据定位 PBE0 能量差；P1 修复 auto-jfit RI-J 辅助基误差，以同几何同方法的精确四中心库仑为基准，H2O/B3LYP/6-31G(d) 误差目标 <0.1 mEh。不得把测试条件差异或拒绝功能算成通过。

规则：先盘点当前工作树和计算入口，重要文件修改前在 E 盘本目录保存原副本及 SHA-256；不覆盖未提交用户工作，不 reset/clean；原始失败和成功日志均保留。新增实验缓存全部在 E 盘。网格与辅助基变动须检查数值收敛及其他体系回归。

## 执行结果（额度收尾快照）

- P3 已完成：PowerShell 解析无语法错误；外部对照 CSV 可读，13 行含 12 个数值结果和 1 个拒绝记录；报告已改为承认网格不同，并修正 CalChem B3LYP 的 VWN_RPA 说明。复现脚本对每个子进程检查 `$LASTEXITCODE`。
- P1 在目标水分子上完成首轮修复与定向复测：`auto-jfit` 指数比改为 2.2，并扩展角动量覆盖；`H2O/B3LYP/6-31G(d)` 相对同程序精确 4c-ERI 的偏差为 `-1.24e-5 Eh`，小于 0.1 mEh；147 个辅助函数，度量矩阵截断模为 0。三次独立运行差值约 `-1.21e-5` 至 `-1.24e-5 Eh`。RI-J 定向 SCF 既有测试 2/2 通过。
- 加入 Lebedev N590 与 `GridLevel::UltraFine`（200 径向壳、590 点角网格、流式构造）；N590 球面单项式测试通过。
- P2 尚未完成双侧 PBE0 网格收敛扫描，因此原 `1.52e-5 Eh` 差异仍不能归因。新 UltraFine 是 CalChem 的密网格选项，不宣称与 Gaussian `Int(Grid=200974)` 节点/径向权重完全一致。
- 当前结果仅证明一个分子/方法/基组的 RI-J 精度；更大的辅助基带来成本上升。尚未完成其他元素/基组、解析梯度截断边界、GPU 与大体系内存复核，也未运行全量 Rust/Python 回归或重建 PyO3 扩展。

## 本轮定向验证

- `cargo test -p calchem-dft test_lebedev_monomials_and_exp_benchmark --offline -- --nocapture`：通过。
- `cargo test -p calchem-scf --test rij_precision_probe --offline -- --nocapture`：通过，输出 `n_ao=19 n_aux=147`、`metric_truncated_modes=0`、`delta=-0.0000123863 Eh`。
- `cargo test -p calchem-scf --test density_fitting_scf --offline`：2/2 通过。
- 复现脚本 PowerShell Parser：0 个语法错误；CSV 导入：13 行。

状态：P3 已完成；P1 单案例目标通过、尚待跨体系验证；P2 已提供更密格点能力但收敛扫描未执行；全量交付验收未执行。
