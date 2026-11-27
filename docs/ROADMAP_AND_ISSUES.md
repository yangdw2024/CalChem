# CalChem 全量问题台账与分阶段推进总规划

> **生效日期**：2026-11-26（`v0.3.0` Phase 3 + Tier-5 加固发布版）  
> **强制前置约束**：本规划下的一切研发、测试、基准对比与汇报，必须无条件遵守仓库根目录最高执行铁律 [AGENTS.md](file:///E:/YDW/Trae_Gaussian/AGENTS.md)。

---

## 1. 当前真实基线与能力边界（客观盘点）

### 1.1 版本与交付状态
| 状态层级 | 当前真实状态 | 说明 |
|---|---|---|
| **历史基线 (`v0.2.2`, Git `9f57bd7`)** | **2026-11-19 归档版** | 已被 `v0.2.3` / `v0.3.0` 替代。 |
| **封板基线 (`v0.2.3`, Git `27929de`)** | **2026-11-24 已发布** | 完成 `OPEN-01/02/03` 物理闭环与 `calchem-py` 剥离；冻结二进制 `C00AB91F...` 备份于 `D:\YDW\Calc\qc_benchmark_runs\pilot\v023_final_C00AB91F\calchem_cli_C00AB91F.exe`；GitHub Release `456555083`（`windows-latest` CI 成功，`ubuntu-latest` CI 失败）与 `@yangdw2024/calchem@0.2.3` 已发布。 |
| **当前发布版 (`v0.3.0`, Phase 3 + Tier-5)** | **2026-11-26 终验通过，发布中/已封板** | 完成 **OPEN-04**（Pure CPU SCF + TD-DFT 批量收缩，以及 CUDA `sm_120` GPU F-shell RI-J 3c / Direct-K / TD-DFT 响应 / 解析梯度全链路卸载）与 3 轮 Tier-5 白盒红队对抗加固；`cargo test --workspace` **`230 passed, 0 failed, 1 ignored`**（72 suites）、`cargo clippy` **exit 0**、`scripts/test_verify_calchem_job.py` **2/2 passed**。最终二进制 `build_target/release/calchem_cli.exe`（`22,292,480` B，SHA256 `C9F553BB0F96D183F34BD1E4A79563D0FAF750C33EFC3B6B9426DD908F1FD653`；Cubin SHA256 `5A7DBBFC0908E9A40DCD2BFE5DAA34CD9C98F2EEFCA919B7A1E379A4BC30F533`）已完成 `D:\YDW\Calc\qc_benchmark_runs\phase3_m6\final_benchmarks\` 全量实测与独立 `teamwork_victory_auditor` 终验（`VICTORY CONFIRMED`）。`crates/*/Cargo.toml` 保留 `0.2.2` 以保持与 `C9F553BB...` 源码一致。 |

### 1.2 真实长板 vs 真实短板（不夸大、不贬低）
- **✅ 真实长板（源码与 `C9F553BB...` 实测已验证）**：
  1. **Pure CPU 8T 已追平/小幅反超 Caffeine G16 8T**：Direct4c 单点 `31.15 s`（`+0.0079 kcal/mol`，`8.16x` v0.2.3 / `1.01x` G16）、`auto-jfit` 单点 `26.77 s`（`-0.0859 kcal/mol`，`3.54x` v0.2.3 / `1.18x` G16）、5 根 Full-LR TD-DFT `174.53 s`（`max|Δω|=0.000233 eV`，`29.33x` v0.2.3 `5119.84 s` / `1.10x` G16 `192.2 s`）；同时修复 `v0.2.3` `wB97X-D` 能量漏计色散（`IC2F` 从 `+9.18` 降至 `-0.0366 kcal/mol`）。
  2. **CUDA GPU `sm_120` 覆盖 F-shell / Direct-K / TD-DFT / 解析梯度且全程零回退**：74 项 `含 GPU` E2E 与 3 项独立 GPU 运行 `gpu_fallback_invocations == 0`；Direct4c CPU 与 GPU 一致性 `<= 1.89e-11 Eh` / `<= 5.11e-10 Eh/Bohr`。`含 GPU` 实测：Caffeine SP `13.45–14.02 s`、Caffeine LR-TDDFT `96.43 s`（`1.81x` CPU / `53.09x` v0.2.3）、`IC2F` SP `15.61–16.14 s`、`IC2F` LR-TDDFT `105.50 s`（`2.72x` CPU）、`pm6_l8bo_60` B3LYP `292.75 s`（`6.87x` CPU `2011.78 s`）。
  3. **质量与契约防线**：`cargo test` `230/0/1`、`clippy` 0 错、`npm/index.js` 严格 fail-fast `CALCHEM_BIN`、[verify_calchem_job.py](file:///E:/YDW/Trae_Gaussian/scripts/verify_calchem_job.py) 支持 `--gpu` 严检与防绕过单测。
- **⚠️ 真实短板（必须继续解决，不隐藏）**：
  1. **E2E 与梯度精度/速度未过门禁项**：CPU (`83/92`) 与 GPU (`65/74`) 均一致存在 **7 项失败 + 2 项已知超标**：
     - 3 项 `PBE/auto-jfit`（`C4H4S` `-0.1307`、`Caffeine` `-0.1820`、`pm6_l8bo_60` `-2.1029 kcal/mol`）；
     - `C2H3F` Full-LR TD-DFT (`nstates=3`) 低激发态根序差异（`max|Δω| = 0.74776 eV`）；
     - `Caffeine` / `IC2F` `B3LYP-D3BJ/6-31G(d,p)` `Fine` 格点解析梯度相对 G16 默认 `UltraFine` 偏差 `1.39e-4–2.22e-4 Eh/Bohr`（Direct4c 与 RI-J 均一致出现），且总耗时仍慢于 G16 8T（Direct4c 梯度 CPU `261.90 s` / GPU `48.52 s` vs G16 `40.2 s`；`IC2F` CPU `403.77 s` / GPU `62.08 s` vs G16 `43.9 s`）。
  2. **其余慢于 G16 8T 项**：`IC2F wB97X-D RI-J`（CPU `251.50 s` / GPU `61.50 s` vs G16 `55.9 s`）、`IC2F LR-TDDFT` 纯 CPU（`287.82 s` vs G16 `176.0 s`）、`IC2F Direct4c SP` 纯 CPU（`57.16 s` vs G16 `43.9 s`）、`C4H4S Direct4c Grad` 纯 CPU（`9.49 s` vs G16 `5.7 s`）。
  3. **优化器仍为笛卡尔 BFGS**：缺少冗余内坐标（DLC）优化器（`OPEN-05`）。
  4. **大体系阶梯尚未完整**：`pm6_l8bo_60`（`B3LYP`）与 `D18`（`STO-3G`）仅验证 SCF 收敛，缺 G16 基准能及 `72/94/199/726` 原子极化基组梯度/频率验证（`OPEN-07/08`）。

---

## 2. 全部已知问题总台账

### 2.1 已修复并实测闭环项（`v0.2.3` + `v0.3.0`）
- **P0-01 ~ P0-05, P1-01 ~ P1-04**：见 `v0.2.3` 归档验收记录（已全部通过）。
- **OPEN-01 / OPEN-02 / OPEN-03**：`v0.2.3` 完成单哈希精度闭环，`v0.3.0` 进一步将 Caffeine LR-TDDFT 5 根从 `5119.84 s` 降至 `174.53 s`（纯 CPU）/ `96.43 s`（含 GPU）。
- **OPEN-04（Phase 3 CPU + GPU 攻坚）**：✅ 已交付 `v0.3.0`（`C9F553BB...`、`230/0/1`、`clippy=0`、CPU `83/92` & GPU `65/74` `fb=0`），并将遗留的梯度网格偏差/耗时、PBE RI-J 偏差与 `C2H3F` TD-DFT 根序拆分至下方 **OPEN-09 / OPEN-10** 跟进。
- **OPEN-06（文档纠偏与 `v0.2.3` / `v0.3.0` 发布对齐）**：✅ 已对齐并随 `v0.3.0` 更新发布。

### 2.2 尚未闭合的待办问题（按优先级排序）

| 编号 | 优先级 | 问题描述 | 根因 / 现状 | 关单验收标准 |
|---|---|---|---|---|
| **OPEN-09** | **P0** | Caffeine/IC2F `Fine` 格点梯度偏差（`1.39e-4–2.22e-4 Eh/Bohr`）与解析梯度速度慢于 G16；`C2H3F` TD-DFT 根序差异 | `Fine` vs G16 `UltraFine` 格点敏感性及解析梯度阶段仍慢于 G16 8T；`C2H3F` `nstates=3` 与 G16 排序差异 | 核明格点收敛及解析梯度热点提速；确认 `C2H3F` 根序与 `UltraFine` 梯度与 G16 对齐至 `< 1e-4 Eh/Bohr` |
| **OPEN-10** | **P1** | `PBE/auto-jfit`（`-0.13` 至 `-2.10 kcal/mol`）与 `STO-3G` RI-J（`-4.08 kcal/mol`）误差 | AutoAux 在 `PBE` / `STO-3G` 下库仑密度拟合残差偏大 | 极化基组 `PBE/auto-jfit` 收口至 `<= 0.1 kcal/mol`（或明确规则边界） |
| **OPEN-05** | **P1** | 优化器为笛卡尔 BFGS，缺少冗余内坐标（DLC） | Rust `calchem-opt` 当前仅实现笛卡尔 BFGS/P-RFO | 在 `calchem-opt` 实现真实 DLC + 信赖域 RFO；咖啡因完整 Opt 收敛步数实测下降且驻点 Freq 0 虚频 |
| **OPEN-07** | **P2** | `ubuntu-latest` CI 测试失败与第三方四引擎对照（ORCA/REST/G16 5D-6D）未闭环 | `v0.2.3` GitHub Actions `45367616846` Linux 权限相关 `disk_diis` 单测在容器权限模型下未拒绝写目录 | 修复 Linux `disk_diis` 测试并使 `ubuntu-latest` CI 全绿；恢复 Pueue（2×4 核上限）并补完四引擎对照表 |
| **OPEN-08** | **P2** | 中大体系（72 / 94 / 199 / 726 原子）极化基组与梯度/频率缺实测 | 当前仅 `pm6_l8bo_60` / `D18` 单点 SCF 收敛（无 G16 对照） | 按 `24 -> 72 -> 94 -> 199 -> 726` 阶梯实测，落盘完整收敛 JSON + 峰值 RSS/VRAM |

---

## 3. 分阶段推进路线图（硬门禁驱动，不达标不进下一阶段）

```mermaid
flowchart LR
    P1["Phase 1: 精度与正确性收口\n(OPEN-01 / 02 / 03) ✅"]
    P2["Phase 2: 文档纠偏与 v0.2.3 发布\n(OPEN-06) ✅"]
    P3["Phase 3: CPU/GPU 攻坚与 v0.3.0 发布\n(OPEN-04 + Tier-5) ✅"]
    P4["Phase 4: 梯度/PBE/TD-DFT 遗留闭环 + DLC 优化器\n(OPEN-09 / 10 / 05)"]
    P5["Phase 5: Linux CI + 四引擎对齐与大体系验证\n(OPEN-07 / 08)"]
    P1 --> P2 --> P3 --> P4 --> P5
```

### 已完成阶段
- **Phase 1（`OPEN-01 / 02 / 03`）**：✅ 冻结二进制 `C00AB91F...` 单哈希验收闭环。
- **Phase 2（`OPEN-06`）**：✅ `v0.2.3` 提交、打 tag、`windows-latest` CI 与 GitHub Release / `npm@0.2.3` 发布完成。
- **Phase 3（`OPEN-04` + Tier-5 加固 + `v0.3.0`）**：✅ 完成纯 CPU（DirectFock 筛选缓存、增量 ΔP、SAD 初猜、TD-DFT 批量收缩）与 CUDA GPU（F-shell `L<=3` RI-J 3c、杂化 `Direct-K`、TD-DFT 响应收缩、解析梯度）全链路开发、3 轮 Tier-5 对抗加固（`230/0/1`、`clippy=0`）、92 CPU / 74 GPU E2E 复测及独立 `VICTORY CONFIRMED` 终验，并推进 `v0.3.0` GitHub/npm 发布。

### 后续待办阶段
- **Phase 4（`OPEN-09 / OPEN-10 / OPEN-05`）**：攻关 `Fine`/`UltraFine` 解析梯度精度与提速、`PBE/auto-jfit` 精度、`C2H3F` LR-TDDFT 根序，以及 `calchem-opt` 冗余内坐标（DLC）+ 咖啡因完整 `Opt -> Freq` 零虚频闭环。
- **Phase 5（`OPEN-07 / OPEN-08`）**：修复 `ubuntu-latest` `disk_diis` CI、补完四引擎同条件对照、完成 `72 -> 94 -> 199 -> 726` 原子极化基组阶梯实测。
