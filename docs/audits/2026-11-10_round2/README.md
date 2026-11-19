# CalChem 第二轮验收与修复 — 2026-11-10

结论：**原修复版并未八项全部闭环；本轮已直接修复确认的实现缺陷并重新编译。** 架构 B，理论还原度 73/100，生产健壮性 C；限定研究用途，整体工业/高通量验收仍不通过。

- [完整审计与八项验收表](02_ACCEPTANCE_AND_REPAIR_REPORT.md)
- [最终实测及复现](03_RESULTS_AND_REPRODUCTION.md)：Rust 102、Python 104 通过，独立探针和 Windows wheel/sdist 通过
- [修复文件、精确行号与回归映射](05_REPAIR_MAP.md)
- [完整 Drop-in 生产代码](06_DROPIN_CODE.md)、[替换文件与数据包](replacements.zip)、[文本差分](changes.patch)
- [计划/状态/规则](01_PLAN_STATUS_RULES.md)、[操作与经验及回滚机制](04_OPERATIONS_AND_LESSONS.md)、[一手资料](07_PRIMARY_REFERENCES.md)

仍需明确：完整解析溶剂梯度/平滑 CDS、标准 SMD、原生 LRC、开壳层 SCRF、通用 SCF 稳定性分析及 Linux/macOS 安装认证尚未完成。错误身份/未实现能力已被显式阻断，不能把“拒绝错误方法”当作实现了该方法。

最终部署扩展 SHA256：`4e68564103ad866fc14ad71fe057d6860d9b827913e07ef38f5816b70c1c9bbd`。

原始 378 文件已备份并在新 E 盘目录完成逐文件 SHA256 恢复演练；无原文件缺失。暂停期间外部新增提交也已保留。本轮未执行 Git 提交、发布或清理用户文件。Windows wheel/sdist 在 artifacts/，尚未发布。
