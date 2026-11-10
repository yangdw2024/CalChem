# 后续执行规则与回滚机制

这些规则落实本次用户要求，适用于本项目后续工作。它们不修改系统或其它项目的配置。

## 执行前后必须记录什么

1. 开始前核对本机实际路径、Git HEAD/分支/差异、未跟踪文件和最近状态记录，先写任务规划及范围。旧归档、设计和当前运行代码分别确认。
2. 只在 D/E 盘创建文件、缓存、临时目录、虚拟环境和软件；C 盘仅可读。软件默认安装路径、Path.home()、TEMP/TMP 都要核对，不能只检查主输出目录。
3. 修改重要文件前，先保存原文件、SHA256 和差异。工作区原有变更/未跟踪文件属于用户内容，不覆盖、不自动提交。
4. 先用最小范围复现，再改一类问题；科学计算修改先核对单位、理论级别、参考几何、基组和收敛条件。
5. 验证失败时记录命令、真实错误和归因；多次相同失败或严重修复必须补充经验记录，不能无依据反复重试。
6. 本地查找优先 rg；注意 Windows 不会总替 rg 展开通配符。清单、长日志保存为证据，分批完整阅读关键文件，避免被输出截断误导。
7. 如任务需要联网调研，优先相关、可靠、及时的一手资料；记录发布日期/版本、访问日期与来源，收藏/点赞只作辅助，并写调研文档。本次没有联网，不虚构资料调研。
8. 收尾记录改动、检查结果、未验证范围、回滚入口，核对文件哈希和 Git 差异。不要把模块存在、程序退出零、空测试集等同功能验收。

## 已建立的关键文件备份

本次复制了 21 个重要文件到本目录 `rollback/backups/<batch>/files/`，每个文件都校验原件和副本 SHA256 一致。包括 Cargo 配置、README、Python/外部适配关键入口、Rust SCF/失败测试、CI 和原有未跟踪 Stage 3 设计。

实际 manifest 路径见 `evidence/backup-manifest-path.txt`。当前源码没有被修改，因此这些备份不是被自动应用的补丁。

备份工具用法（在项目根运行，先明确列出要保护的文件）：

```powershell
& .\docs\audits\2026-11-03\tools\backup_important.ps1 -Paths @(
    'calchem\backends\native.py',
    'crates\calchem-scf\src\rhf.rs'
)
```

工具仅接受本仓库中的现有明确文件，不遍历删除目录，不备份 .git 或审计自身；备份批次不会覆盖现有批次。建议每项后续修复单独建批次。

## 恢复步骤

先预览：

```powershell
$manifest = Get-Content -LiteralPath 'E:\YDW\Trae_Gaussian\docs\audits\2026-11-03\evidence\backup-manifest-path.txt'
& 'E:\YDW\Trae_Gaussian\docs\audits\2026-11-03\tools\restore_backup.ps1' -ManifestPath $manifest -WhatIf
```

确认具体要恢复的批次后，对同一命令去掉 `-WhatIf` 才执行恢复。恢复前会校验备份哈希和路径边界，并将当前目标文件另存到 `rollback/before-restore/`，再恢复及校验；不会删除未列入 manifest 的文件。本次只运行了预览，**没有实际覆盖恢复原件**。该机制保护明确备份文件，不承诺整个开发环境或数据库快照恢复。

## 撤回本次审计文件

源码未改，撤回只需将整个审计目录移到 E 盘隔离区，保留全部文档和编译产物：

```powershell
& 'E:\YDW\Trae_Gaussian\docs\audits\2026-11-03\tools\quarantine_audit.ps1' -WhatIf
```

去掉 `-WhatIf` 时才移动到 `E:\YDW\_CalChemAuditArchive\2026-11-03-<timestamp>`。脚本在移动前验证源路径和目标路径，采用 PowerShell 原生命令，不跨 shell 拼接删除。本次未实际移动。

## 目录与磁盘规则

- 正文、证据、工具、备份按子目录集中归类，根目录不散落新报告。
- `runtime/` 是本地生成物，已在审计目录的 .gitignore 中排除，不能整体加入 Git。独立 Rust 检查产生约 1.04 GB 调试构建文件，这是编译验证的本地产物，不是交付源码。
- 现有 `build_target/`、`_archive/` 和 `_chatgpt_stage2/` 没有移动、清理或删除。彻底废弃文件日后可清理，但须先确认不再被引用并记录清单，不能仅凭名字判定可删。
- 禁止使用 `git reset --hard`、`git clean -fd` 或无范围的递归删除来回滚本任务。
