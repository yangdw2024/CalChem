# 操作记录、修复经验与回滚

日期：2026-11-11（Asia/Shanghai）。以下按依赖顺序记录；精确执行输出见 evidence。

1. 读取工程 git 状态，发现大量既有修改；保存 status 和 tracked diff，未使用 reset/clean。
2. 完整读取五份 master_mission 资产、MISSION_TRACKER、开发环境/构建/测试脚本。建立本目录的计划、证据、探针和备份分类；按相对路径复制原始 crates、calchem、scripts、tools/dev、tests/unit 和交付文档到 backup。
3. 先修测试入口：无参 cargo 脚本此前没有执行测试，且会恢复 C: TEMP。修正后运行全 Rust 基线，123 tests passed。所有本次工具缓存、temp、target、Python 数据都使用 D:/E:，没有下载安装依赖。
4. 沿源码核查 Fock 内存、ERI/梯度、RI 度量与缓存、XC、SCF、CUDA、TDDFT、SCRF 和原生绑定。查原始论文与 Libxc 官方定义，调研记录保存在 02_PRIMARY_RESEARCH。
5. 修复 CUDA FFI 边界、C++ 错误检查/生命周期和索引；禁止旧 prebuilt 掩盖编译失败；补齐 use_gpu/disable/预算语义；CRT 使用 /MD。新增无效切片与成功/回退契约测试。
6. 修复 omega=0/极值、CAM ITYH 链式项、直接积分筛选、Vxc 缓冲数和能量求和、RI 归一化谱截断/缓存构建/辅助指数范围。
7. 修复 SCF/Davidson 的假成功路径；增加重建物理 Fock 后的 canonicalization 残差复核。统一到 RHF/RKS/UHF/UKS。
8. 标准 wB97X-D、范围分离/Direct/RI SCRF、SMD 的实现不完整，选择明确拒绝，避免以别的公式返回标准方法的结果。保留 legacy 低层代码以便完整替换；更改相关测试的预期为正确拒绝。此处是能力边界纠正，不是实现这些方法。
9. 新增波函数 RI-J 来源字段，串联 Rust/Serde/PyO3/pyi。拦截将 RI-J 波函数送入精确库仑导数或响应的请求；新增四种梯度及 Python TDDFT 来源验证。
10. 全量扩展重建、Rust/Python 测试和用户指定 OPV 专项；每轮独立日志与退出码保存。并行测试仅用于本任务的独立工具进程，没有派生子代理。
11. 新增 IC-2F 真正 Direct SCF 进程实测：输出方法/基组/格点/AO/能量/耗时/Windows 峰值工作集和 Gaussian 元数据；不混淆 loader 测试与大分子计算。
12. 在历史交付文档顶部加撤回/更正提示；CSV 所有旧行标为 UNVERIFIED_HISTORICAL，原始数值不删除。生成最终问题评级、可复现脚本、差异、行号和哈希记录。

## 本次遇到的错误和处理

- 初次尝试用开发 Python 直接写源码返回 PermissionError；文件不是只读，ACL 允许写。未修改权限或绕过保护，随后仅用 apply_patch 编辑源码。Python 数值探针只读取源码/输出证据。
- PowerShell 7 将透传 cargo `-p` 解释为脚本公共参数，出现歧义；改用显式 `-CargoArgs @('test',...,'-p',...)`。默认无参入口仍可按用户命令运行。
- 初次改变 Vxc 分组后，原 1e-9 有限差分断言出现约 1.5e-9 误差。原因是能量求和顺序改变与消差；通过补偿求和解决，没有放松测试阈值。
- 真正启用 RI-J 的 LDA 测试后出现约 1.092e-3 Eh 偏差，原测试阈值 1e-3。调查自动辅助指数截断/粗间隔并修复，保留原精度要求。不能通过关回 density_fitting 或改阈值交付绿色。
- 添加最终 SCF 阈值后，乙烯 PBE 暴露“DIIS 已小但额外纯 canonicalization 又放大误差”。修改为在最终纯步骤后重新验证并继续 polishing，专项 3 tests passed；没有恢复任意有限残差即成功的逻辑。
- 中间 Python 回归有 3 个失败：一项 IC-2F 溶剂流程触发同类 SCF 驻点问题，另两项仍期待不完整 wB97X-D 成功。前者通过同源 SCF 修复验证，后两项改为保护性拒绝契约；最终结果另见 summary。
- 中间 pytest 对项目 .pytest_cache 写入被拒绝，随后将 cache_dir 显式定向到 E: runtime。日志中的测试失败和 cache 警告分开判定。
- 关闭 Davidson 的假成功路径后，全量 Rust 回归暴露了本征分解器人为相对精度下限：请求 1e-10，实际停在约 6.127e-10。没有放宽测试或恢复假成功，而是把 Jacobi 下限改为机器舍入尺度，并用独立本征残差测试约束。TDDFT 专项 4 项通过后再次重建、全量回归。
- 证据脚本直接写 audit JSON 也受开发 Python 的写入限制；改为只从 Python 输出结构化文本，由 PowerShell 的原生文件写入保存，未修改 ACL。
- 所有中间失败日志保留，不覆盖成“从未失败”。新的源码改动发生后，重新构建和回归，避免引用旧 pyd/旧测试结果。
- 最后补齐 UHF/UKS dense stored-ERI 独立内存准入，防止误用 RKS 稀疏估算。等待在用 Python 进程退出后再替换 pyd，最后重新执行完整脚本，而非仅引用补丁前的测试结果。

## 最终闭环

Release 构建成功；14 crates 全量 134 passed；Python 全量 118 passed（514.13 s）；OPV 专项 3 passed（14.70 s）。最终 IC-2F Direct SCF 22 步收敛、94 AO、−870.5416758786621 Eh，333.24 s、进程峰值 131,301,376 bytes。数值测量期间其他测试并行，因此不作为独占性能比较。

最终 pyd/DLL 哈希一致；源文件与二进制 manifest 无漂移。CUDA availability=false、kernel executed=false，不认证 GPU 精度。没有运行 726/2000 原子完整 SCF。22 项发现、能力拒绝范围、现有编译告警与工业准入未通过的结论全部保留在最终报告。

## 回滚约束

backup 的内容是本次任务开始时的工作文件，包含此前开发团队尚未提交的改动，并不等于 Git HEAD。回滚只复制本次对应文件，禁止全仓库清理。新的 density_fitting 字段改变 Rust 公共结构字面量 API；外部 Rust 调用方需要补齐该字段，Serde 旧数据缺字段时默认 false，旧的拟合波函数不应据此获得新来源认证。Python 新扩展有对应只读字段，回滚或分发时须同步源码、pyi 与 pyd。

原 pyd 由 build_ext.ps1 自动备份在 build_target/runtime/pyd-backup。该脚本现对复制失败中止并检查构建 DLL 与安装 pyd 哈希，避免 DLL 被占用时误认为安装成功。

最终无失败仅指本次已执行的测试集合；未实现方法和未执行硬件/大体系路径的状态必须继续保留在报告中。
