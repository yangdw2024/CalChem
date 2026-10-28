# CalChem v0.2.0 完整验收报告

日期：2026-10-28

## 已完成（v0.2.0 全部 P0）

### 科学正确性
- [x] native 后端：移除 energy=0/converged=True 假值，严格验证
- [x] ORCA 偶极：只存 magnitude，不伪造矢量
- [x] Result 模型：新增 dipole_magnitude_debye
- [x] Request 验证：元素名、有限坐标、电子数/多重度一致性
- [x] Post-HF 总能：total = SCF + correlation

### 作业调度
- [x] JobDispatcher：后台线程消费队列，单并发
- [x] 两个排队任务依次执行验证

### 外部执行
- [x] external.py：Popen + 超时 taskkill /T 进程树清理
- [x] Gaussian/ORCA 路径不硬编码，从 AppConfig 注入

### 配置系统
- [x] config/loader.py：自动检测 + 文件 + 环境变量
- [x] 优先级：CLI > project > user > auto-detect > default

### CLI
- [x] --json 模式 stdout 纯 JSON，可 json.loads()

### 测试
- [x] tests/unit: 22 tests pass
- [x] tests/ 分层：unit/numerical/parser/external/gui

## 数值基线（未被破坏）
- H2/RHF/STO-3G: -1.1167143251 Eh
- He/RHF/STO-3G: -2.8077839575 Eh
- H2O/RHF/STO-3G: -74.6953191771 Eh

## 仍未实现（v0.3+）
- 几何优化 / 频率
- cclib 集成
- PySCF/xTB 真实适配
- GUI 接入 TaskService
- 跨进程锁
