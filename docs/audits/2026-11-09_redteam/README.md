# CalChem 2026-11-09 独立红队审计

结论：**真实研究型量化原型，整体生产验收不通过。** 架构 B，理论还原度 60/100，生产健壮性 D。Rust 91 测试和 Python 96 单元测试通过；独立黄金标准及反例揭示 LRC 方法身份、D3、溶剂导数、TS/IRC、IR 单位和 wheel 分发问题。

阅读顺序：

1. [完整审计报告](03_REDTEAM_REPORT.md)：四大交付章节与六维穿透结果。
2. [实测及复现](04_TEST_RESULTS.md)：测试计数、误差、外部实跑、OPV、包构建。
3. [计划与回滚规则](01_PLAN_AND_RULES.md)、[状态与工作流](02_STATUS_AND_WORKFLOW.md)。
4. [操作与经验](05_OPERATIONS_AND_LESSONS.md)、[一手资料调研](06_EXTERNAL_RESEARCH.md)。

证据保存在 evidence/，独立探针在 probes/，原文件快照在 rollback/source_snapshot.zip。运行产物在 E:\YDW\Trae_Gaussian\build_target\runtime\redteam_20261109；外部原始输入/输出另归档 evidence/external_raw.zip。业务源码、原扩展与用户未提交修改保持原样，详见 evidence/integrity-after.json。
