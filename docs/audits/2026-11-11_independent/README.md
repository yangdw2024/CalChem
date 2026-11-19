# CalChem 独立审计入口

日期：2026-11-11。**当前定位为研究开发引擎，未通过原工业级/千原子认证主张。**

- [最终物理与工程报告](FINAL_AUDIT_REPORT.md)：问题、修复、能力边界、评级和复现。
- [修改文件与精确行号](04_CHANGE_MAP.md)：与任务开始时的 backup 对比，含格式化变更。
- [计划、状态和规则](01_PLAN_STATUS_RULES.md)。
- [原始论文与官方定义调研](02_PRIMARY_RESEARCH.md)。
- [操作、失败修复经验与回滚](03_OPERATIONS_AND_LESSONS.md)。
- `evidence/acceptance-summary.json`：最终退出码、测试数、真实 OPV 测量及二进制一致性。
- `evidence/independent-changes.patch`、`changes.json`、`source-binary-sha256.json`：本次变更和哈希。
- `backup/`：任务开始时的相对路径备份，保留原有未提交改动。
- `probes/`：独立公式、实际 Direct SCF、差异与验收汇总脚本。

日志中保留了中间失败，不用早期/旧二进制通过结果替代最终验收。含 GPU 设备不可用时提前 return 的测试不能认证 GPU 内核精度。
