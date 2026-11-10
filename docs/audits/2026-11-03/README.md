# CalChem 本地项目审计 · 2026-11-03

建议先读 [状态与工作流](02_STATUS_AND_WORKFLOW.md)，再读 [问题和后续规划](03_FINDINGS_AND_NEXT_STEPS.md)。本次是状态审计，不包含业务修复。

| 文件 | 用途 |
|---|---|
| [00_START_STATE.md](00_START_STATE.md) | 执行前的状态和保护边界 |
| [01_PLAN.md](01_PLAN.md) | 检查规划和验收要求 |
| [02_STATUS_AND_WORKFLOW.md](02_STATUS_AND_WORKFLOW.md) | 实际架构、时间线、能力边界和实测结果 |
| [03_FINDINGS_AND_NEXT_STEPS.md](03_FINDINGS_AND_NEXT_STEPS.md) | 缺陷证据、优先级、建议修复顺序 |
| [04_RULES_AND_ROLLBACK.md](04_RULES_AND_ROLLBACK.md) | 后续执行规则、备份和恢复方法 |
| [05_OPERATIONS.md](05_OPERATIONS.md) | 本次操作与验证边界 |
| [06_LESSONS.md](06_LESSONS.md) | 排错记录与避免重复错误的方法 |
| evidence/ | Git、文件清单、哈希、检查日志和结构化结果 |
| rollback/ | 21 个关键文件的备份及 SHA256 清单 |
| tools/ | 可重放检查、备份、恢复和隔离脚本 |
| runtime/ | 本地测试、缓存和编译产物；已通过本目录 .gitignore 排除 Git 跟踪 |

审计日期按 Asia/Shanghai 记录。所有主动创建的文件均在 E 盘；现有 Python、Rust 和 Ruff 工具来自 D 盘。没有安装、下载、提交、推送、删除或修改业务文件。报告中的外部适配器诊断使用合成日志，不等于真实 Gaussian/ORCA 验收。
