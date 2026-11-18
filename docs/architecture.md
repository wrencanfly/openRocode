# openRocode 架构设计与路线规划

## 1. 核心定位与设计哲学

`openRocode`（CLI 命令：`rocode`）旨在成为具身智能领域的 **“Claude Code / OpenCode”**——一个**单二进制、低延迟、强安全隔离的终端原生机器人编排运行时（Robotics Agent Harness）**。

它不重造底层物理引擎或策略训练框架，而是专门弥合**高层推理模型（System 2 VLM）**、**高频学习策略（System 1 VLA / RL）**与**底层硬件控制（System 0 Controller）**之间的工程鸿沟：

- **Rust 掌控生命线，Python 专注跑模型**：核心状态机、多频异步抢占、心跳看门狗（Watchdog）、物理安全护栏（Safety Arbiter）、黑匣子录放与终端 TUI 由 **Rust** 实现。即便 Python VLA 进程发生 CUDA OOM、崩溃或远端 VLM 请求卡死，Rust S0 仲裁器依然按固定高频节拍存活并瞬时切入 `SafeHold`。
- **契约驱动的异构策略编排**：将解析控制原语与冻结的 VLA 策略统一抽象为 `SKILL.toml`（物理前置条件、看门狗时限、安全围栏）+ `SKILL.md`（S2 语义说明与恢复指引）。
- **控制面与数据面分离**：上层仅流转轻量的 `FrameRef { stream_id, seq, timestamp_ns, slot_id }`；v0.1 采用轻量本地 IPC 快速验证闭环，v0.3 无缝切换至 **POSIX 共享内存无锁环形缓冲区（SHM Ring Buffer + Seqlock + Python DLPack 零拷贝）**与跨机 **Zenoh** 总线。
- **可复现与可调试优先**：v0.1 内置支持故障注入的 `MockRobot` 与时间轴黑匣子回放引擎，无需真机即可验证抢占、看门狗拦截、失败切片上报与人工遥操接管（Teleop Takeover）。

---

## 2. 分层架构（S2 - S1 - S0）

1. **S2 Orchestrator (`rocode-vlm` + `rocode-context`)**：0.2–2 Hz 异步推理层。接收经蒸馏的关键帧快照与失败前 3 秒黑匣子切片（`FailureTrace`），通过结构化调用向 S1 下发技能目标。
2. **S1 Skill Supervisor (`rocode-core` + `rocode-transport`)**：10–50 Hz 技能生命周期监督器。负责入场前置条件校验（Precheck）、进程健康度监控、有限次数局部重试（Bounded Local Retry）及毫秒级异步抢占。
3. **S0 Safety Arbiter (`rocode-safety` + `rocode-adapters`)**：200–1000 Hz 纯 Rust 确定性安全阀门。执行关节速度/单步跳变裁剪、心跳超时自动锁止（`SafeHold`）、操作员遥操接管多路复用（`TeleopTakeover`）以及硬件急停（`EStop`）。

---

## 3. 分阶段路线图（Roadmap）

### Phase 1 (`v0.1`)：纯抽象运行时、安全沙盒与黑匣子底座
- 跑通 `S2 -> S1 -> S0 -> MockRobot` 异步状态机与故障注入闭环。
- 打通 Rust 主进程与 Python VLA Worker 子进程的本地 IPC 协议及 80ms 超时看门狗自动 `SafeHold`。
- 完成黑匣子录制（`rocode-recorder`）、确定性回放命令（`rocode replay`）及初版 `Ratatui` 交互终端。

### Phase 2 (`v0.2`)：首个仿真具身闭环与接管采数
- 接入 `MuJoCo` Adapter 与 `python/openrocode_sdk` 的 `openpi` / `lerobot` VLA 包装器。
- 实现 `Teleop Takeover -> Dataset` 闭环：人工接管救场片段自动导出为 LeRobot / MCAP 标准格式。

### Phase 3 (`v0.3`)：轻量高速零拷贝数据流
- 在 `rocode-transport` 落地共享内存零拷贝环形缓冲区（SHM Ring Buffer + Python Buffer/DLPack），实现亚百微秒级多路图像/点云流分发。
- 提供可选的 `Zenoh` 分布式传输后端（机载边缘端 S0 <-> 工作站 S1 VLA <-> 云端/本地 S2 VLM）。

### Phase 4 (`v0.4`)：真机部署与生态桥接
- 提供主流机械臂与人形真机 Adapter（Piper / xArm / Unitree G1）及非侵入式 ROS2 Bridge。
- 加入策略交接前的入场分布检测（OOD Precheck）与自动过渡站位规划（Staging Planner）。
