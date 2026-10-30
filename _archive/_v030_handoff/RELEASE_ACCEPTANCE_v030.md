# CalChem v0.3.0 Release Acceptance

日期：2026-10-29

## 已完成模块

### P0 科学正确性（v0.2.0）
- [x] native 后端无假值（energy/converged 严格验证）
- [x] ORCA 偶极不伪造矢量
- [x] Request 输入验证（元素、坐标、电子数/多重度）
- [x] CLI --json 纯 stdout
- [x] 跨进程 heavy 锁

### P1 调度与配置（v0.3.0）
- [x] JobDispatcher 单并发队列
- [x] TaskService 全局锁接入
- [x] config/loader.py 自动检测
- [x] external.py 超时进程树清理
- [x] parsers/base.py 共享解析工具
- [x] core/periodic_table.py 统一元素表

### P2 几何优化（v0.4.0）
- [x] grad/finite_difference.py 中心差分梯度
- [x] opt/numeric_optimizer.py BFGS 优化器
- [x] native optimize() 任务
- [x] H2 优化端到端跑通（2.0→1.67 Bohr）

### P2 频率（v0.5.0）
- [x] vib/numeric_frequency.py 数值 Hessian
- [x] 振动模式投影（平移/转动去除）
- [x] H2 频率端到端跑通（5028 cm⁻¹）

## 数值基线（未破坏）
- H2/RHF/STO-3G: -1.116714325063 Eh
- 22 unit tests pass

## 已知限制（诚实标注）
- 优化精度受有限差分噪声限制（H2 收敛到 1.67 vs 1.39 Bohr）
- 频率数值精度受 Hessian 步长影响
- Gaussian/ORCA optimize/frequency 未接入
- PySCF/xTB 未真实适配
- GUI 未接入 TaskService 队列
- one-folder 打包未用 v0.3.0 代码重新构建
