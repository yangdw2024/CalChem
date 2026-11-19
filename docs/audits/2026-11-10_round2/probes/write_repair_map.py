from pathlib import Path
R=Path(r'E:\YDW\Trae_Gaussian');O=R/'build_target/runtime/round2_20261110/final_documents'
rows=[
('IR 单位与响应精度','crates/calchem-vib/src/masses.rs','pub const IR_INTENSITY','完整单位系数正确保留；Hessian/APT 自动收紧 SCF 以控制差分噪声','acceptance-delivery.jsonl'),
('软模式与 Eckart 子空间','crates/calchem-vib/src/modes.rs','// Diagonalize only the orthogonal complement','在刚体运动正交补中对角化，避免按最小绝对本征值误删真实软模式','cargo-delivery.log'),
('新鲜 Hessian TS 认证','crates/calchem-opt/src/prfo.rs','let exact_hessian','终点重新计算 Hessian；恰好一个分辨负模式；失败传播，无后门认证','ecp-ts-delivery.jsonl'),
('IRC 终点与步长','crates/calchem-opt/src/irc.rs','let mut accepted','终点 E/g 重新评估；未满足约束减半步长；未完成分支不产生最终势垒','cargo-delivery.log'),
('IRC 起点与模式','crates/calchem-opt/src/irc.rs','let g_ts_norm','起点驻点/负模式认证及启动向量投影检查；H2 极小值拒绝','pytest-delivery.log'),
('IRC PyO3 字段丢失','crates/calchem-py/src/lib.rs','pub struct PyIrcBranch','显式导出完成标志、可空的端点势垒/反应能，同步 stub 和 Python 消费端','bridge-contracts.log'),
('Boys 高阶/高 T','crates/calchem-int/src/boys.rs','pub fn boys_batch','正项 Kummer、补偿和、适用阶数递推；移除高 T 无条件无穷上限替代','boys-certified.jsonl'),
('标准 D3 C6/CN','crates/calchem-dft/src/d3.rs','fn weights','原子对参考态插值与 r4/r2 C8；完整 CN 链式解析导数；标准参数','acceptance-delivery.jsonl'),
('Python D3 回退与参数','calchem/dft/d3.py','def _evaluate','取消近似 Python 数值回退；未知模型和未支持自定义参数明确失败','pytest-delivery.log'),
('独立气相失败吞异常','crates/calchem-solv/src/scrf.rs','let gas_energy = calchem_scf::rhf','独立未极化气相 SCF 成功才返回 ΔG，错误直接传播','acceptance-delivery.jsonl'),
('溶剂总导数口径','crates/calchem-opt/src/lib.rs','pub fn scf_energy_gradient_with_options','重新收敛完整 SCRF 总能量做中心差分；CDS 与能量使用相同定义','core-closed-loop.log'),
('伪解析 PCM 梯度','crates/calchem-solv/src/gradient.rs','pub fn compute_pcm_gradient_analytic','拒绝未实现解析能力；保留明确标注的 fixed-density partial FD','02_ACCEPTANCE_AND_REPAIR_REPORT.md'),
('开壳层后修正冒充 SCRF','crates/calchem-solv/src/scrf.rs','pub fn run_scrf(','阻断仅气相 UHF/UKS 加后修正的自洽身份；方法与 XC 冲突也拒绝','pytest-delivery.log'),
('网格半径截断','crates/calchem-dft/src/grid.rs','pub fn build_for_basis','不再固定 rmax；按最小基组指数增加径向节点并用于 SCF/梯度','cargo-delivery.log'),
('DIIS 小残差奇异','crates/calchem-scf/src/diis.rs','let mut scale','统一缩放 B 子块，系数尺度不变；约束行保持原样','diis-regression.log'),
('最终 SCF 假平稳','crates/calchem-scf/src/uhf.rs','let fp_comm_tol','堆叠 alpha/beta polishing；最终实际残差决定接受，不靠旧残差最小值','cargo-delivery.log'),
('壳层四元组预分配','crates/calchem-int/src/symmetry.rs','pub fn shell_quartets','惰性常数辅助存储枚举，Rayon 分发不预存完整四元组空间','cargo-delivery.log'),
('索引/容量/非法迭代','crates/calchem-scf/src/lib.rs','pub fn validate_resources','分配/建网格前检查保守储存界与 u16 索引、容差和迭代预算','cargo-delivery.log'),
('LRC 方法身份','crates/calchem-dft/src/xc.rs','pub fn validate_native','正式名称与 Rust 旧 enum SCF 路径均拒绝，直到实现 range-separated ERI','pytest-delivery.log'),
('wheel 旧二进制复用','setup.py','class RustBuildExt','每次构建调用 Cargo JSON；只复制真实 cdylib；标准 Extension/abi3 标签','package-delivery-contracts.jsonl'),
]
out=['# 修复映射与完整替换代码入口\n\n所有路径指向最终磁盘版本；行号从实际源码自动提取。每项完整替换代码在 [06_DROPIN_CODE.md](06_DROPIN_CODE.md)，配套数据/文件在 replacements.zip。回归失败的早期日志另存，不替代下表最终证据。\n\n| 缺陷 | 目标文件与精确入口 | Replace 的实际内容与数学原因 | 回归证据 |\n|---|---|---|---|\n']
for title,file,needle,fix,proof in rows:
    lines=(R/file).read_text(encoding='utf-8-sig').splitlines();line=next(i for i,s in enumerate(lines,1) if needle in s)
    proofpath=proof if proof.endswith('.md') else 'evidence/'+proof
    out.append(f'| {title} | [{file}:{line}]({R.as_posix()}/{file}:{line}) | {fix} | [{proof}]({proofpath}) |\n')
out.append('\n## 替换方式\n\n1. 使用 final_manifest.json 核对当前基线；保留已有修改。\n2. 在新的 D/E 候选目录应用完整文件，不使用 reset/clean。\n3. D3 必须同时放置 d3.rs、d3_data.rs、data/d3/c6.bin、原始数据和许可证；Python wheel 也需要随附 calchem/data/d3。\n4. 重新构建 PyO3 扩展，再按 03_RESULTS_AND_REPRODUCTION.md 复验；不要假设正在运行的 Python 会热重载新二进制。\n\n跨模块修复涉及 imports、Result 错误契约与调用者；完整文件比只粘贴单个公式更能避免漏改。changes.patch 提供文本审阅，二进制文件通过 replacements.zip 和 SHA256 校验。\n')
(O/'05_REPAIR_MAP.md').write_text(''.join(out),encoding='utf-8')
