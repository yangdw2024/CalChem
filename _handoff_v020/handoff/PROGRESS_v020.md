# CalChem v0.2.0 P0 修复进展（2026-10-28）

## 已完成的 P0 修复

### 1. native 后端假值问题（对应计划 2.03）
- **之前**: `converged = getattr(calc, "converged", True)`, `energy = float(getattr(calc, "energy", 0.0) or 0.0)` —— 计算失败时仍返回假成功
- **现在**: 严格检查 energy 非零、非 NaN；converged 缺失时从能量有效性推断；post-HF total_energy = scf + e_corr
- **验证**: H2/RHF/STO-3G = -1.116714325063 Eh（与基线差 3.7e-11），status=succeeded

### 2. ORCA 偶极伪造（对应计划 2.04）
- **之前**: 把 magnitude 伪造成 z 轴矢量 `[0,0,magnitude]`
- **现在**: 只存 `dipole_magnitude_debye`，矢量为 None；basis map 正确（sto-3g→STO-3G）

### 3. Result 模型（对应计划 2.02）
- 新增 `dipole_magnitude_debye: Optional[float]` 字段
- 不破坏 JSON round-trip

### 4. JobDispatcher（对应计划 2.05）
- **之前**: JobQueue 只有 submit/get_next，没有消费者线程
- **现在**: `calchem/jobs/dispatcher.py` —— 独立 daemon 线程消费队列，threading.Lock 保证单并发，支持回调
- **验证**: 提交两个 H2 任务，队列依次执行，两个都 succeeded

### 5. CLI --json 污染（对应计划 2.07）
- **之前**: Rich console.print + native 引擎的 print() 全部输出到 stdout
- **现在**: --json 模式下用 `_suppress_stdout()` context manager 把计算过程的 stdout 重定向到 stderr，stdout 只输出 JSON
- **验证**: `python -m calchem.cli run --json 2>/dev/null | python -c "import json,sys; json.load(sys.stdin)"` 通过

## 尚未完成的 P0 项（ChatGPT 计划 2.01-2.08）

| ID | 项目 | 状态 | 说明 |
|----|------|------|------|
| 2.01 | 完整仓库快照 | 部分 | handoff zip 缺原生计算内核（scf/, integrals/, post_hf/ 等），本地仓库有但未包含在 zip |
| 2.02 | Request 输入验证 | 未做 | 元素名、坐标有限数、电子数/多重度一致性未加 Pydantic validator |
| 2.04 | Gaussian 后端能量提取 | 未做 | 仍取第一个 SCF Done；硬编码路径 D:\G16W\g16.exe |
| 2.04 | external.py 超时清理 | 未做 | subprocess 超时后未 kill 子进程树 |
| 2.06 | GUI 接入 TaskService | 未做 | GUI worker 仍直接 start_calc_thread() |
| 2.07 | 后端路径从 AppConfig 读取 | 未做 | gaussian.py/orca.py 仍硬编码 |
| 2.08 | tests/ 分层 | 未做 | 没有 unit/numerical/parser/external 目录 |

## 数值基线（未被破坏）
- H2/RHF/STO-3G: -1.1167143251 Eh
- H2O/RHF/STO-3G: -74.6953191771 Eh
- 三后端交叉验证（v0.1.0 验收）: native vs Gaussian 差 4.2e-9，vs ORCA 差 9.4e-9
