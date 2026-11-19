# 操作、恢复与经验

## 操作边界

1. 按任务授权审计并直接修复。先保存 378 个源码/配置/原扩展文件的 ZIP 与 SHA256；保留用户未提交内容。
2. 所有计划、报告、数据、缓存、编译与恢复候选在 E 盘；工具链使用 D 盘。C 盘只读。本轮未安装软件依赖。
3. 原修复版、过渡版本、最终版本分别记录。原始日志和失败探针保留，只有最终 exit 0 的验收进入汇总。
4. 最终源码变更和完整文件导出在本目录；当前部署扩展哈希由 final_manifest.json 和实际加载探针共同确认。
5. 未执行 Git commit、push、发布 wheel/npm，未删除用户文件，未 reset/clean，未终止用户的开发/服务进程。

## 时间线

- 08:33 起重新盘点，保存原修复版快照，读取新源码并启动旧反例复测。
- 复现 D3 多分子误差、弥散电子积分缺失、溶剂总梯度不一致、TS/IRC/打包残留问题；实施第一组修复。
- Python 进程写源码路径遇 PermissionError；未改 ACL，改用 E 盘已存在运行区生成候选文件，再由 PowerShell 定点复制。
- 开发进程占用旧 `_core.pyd` 时保留其进程，用隔离二进制加载核验。最初带 after 名称但仍加载旧扩展的日志不作为新内核证据。
- 额度暂停期间出现外部提交 `2724136`，以及不同扩展。14:29 恢复后保存 `resume_1429/resume-source.zip` 与已部署扩展，复查新分支；没有把外部提交当作本助手工作。
- 响应精度测试触发乙烯 IR 门槛；通过收紧真实 SCF 平稳性和响应容差解决，未放宽阈值。
- 更严 UHF 6-31G(d) 回归失败后，定位 DIIS 误差子块与约束行尺度失衡；统一缩放子块、堆叠 spin polishing 后原测试通过。
- Rust 已有的 IRC 状态字段没有完整暴露给 PyO3，出现 AttributeError。补齐结果/分支字段和 `.pyi`，新增类型契约测试，先定点检查再跑最终全量。
- 冻结业务源码、只格式化已修改 Rust 文件，独立 target 下重编译；最终 Cargo 102、Python 104 通过，独立数值与 wheel/sdist 验收通过。

精确时间、原始命令、哈希及返回码在 evidence/ 中；以上不是把所有并发进程归因于本助手。暂停时其他 pytest、maturin 安装和开发进程均未由本轮创建或干预。

收尾读取 Git 时还观察到外部提交 `1f90ec0`（14:46，npm wrapper 与部分审计改动）；本助手没有执行该提交。最终再次对 manifest 的 406 个文件逐一计算 SHA256，全部与已验收版本一致，见 `evidence/last-source-check.json`。因此该 Git 状态变化没有使本轮数值验收对象失效；未将提交标题当作源码或运行证明。

## 失败与经验

- **先检查绑定契约**：Rust result 中有字段，不代表 Python 也有；必须同时检查 pyclass getter、映射闭包、stub 和后端消费端。现已用低成本 introspection 回归锁定。
- **冻结源码再执行全量测试**：早期一边运行 workspace、一边编译/编辑导致 doctest 使用旧 rlib 的假故障。最终以独立 target 和冻结源码的全量结果为准。
- **不要按 mtime 手工选 rlib**：不同源目录构建的同名 crate 可以有不同 crate identity，手工链接探针曾混入不同版本。最终使用独立 Cargo manifest，让 Cargo 解析一致依赖图。
- **区分 F-contiguous 和真正 strided**：最初探针把 Fortran-contiguous 对称密度误当“不连续”；这不是内核错误。更正为实际有 stride 的视图后，边界明确返回 ValueError。
- **IR 响应检查不只看 SCF 能量**：小位移会放大密度误差；通过步长扫描和 SCF 残差诊断后再修算法，不能直接增大 IR 容差。
- **外部数据的修订要可追溯**：GitHub commit API 被限流，留下失败记录；因此未编造上游 commit，保存实际下载文件与 SHA256、对应原始源表和许可证。
- **终止只针对本轮已识别进程**：被替换的本轮 pytest 在父进程/命令核对后停止，记录于 pytest-superseded.txt；未算作通过。未终止 AstrBot、MolForge 或用户开发进程。

## 恢复机制

原始修复版：`rollback/source_snapshot.zip` + `evidence/source_before.json`。暂停后外部提交状态：`build_target/runtime/round2_20261110/resume_1429/resume-source.zip`。

恢复演练已将原快照解压到**新的** `build_target/runtime/round2_20261110/restore_round2_candidate/`，378 个文件逐一验证 SHA256，全部一致，未覆盖工作区。该演练证明文件恢复完整性，不是对旧算法重新做科学认证。

需要回滚时先选择正确基线，解压到新的 D/E 候选目录，核对 manifest，再在候选目录构建、复测和人工比对新增工作。禁止将旧 ZIP 直接覆盖当前项目；本轮没有提供会误删未跟踪文件的自动回滚命令。

`changes.patch` 是相对本轮最初修复版的文本审阅差分；二进制与新标准表由 replacements.zip 携带。该时窗包括暂停期间外部提交，不应把整份 patch 不经审查套到不同分支。final_manifest.json 记录 60 个发生变化/新增的文件，原始文件缺失数为零。用户 npm/ 未被移动或清理。
