# openRocode 架构设计与路线规划（LLM-First 修订版）

## 1. 核心定位：为 LLM 打造的具身操控底座（LLM-First, Zero-VLA Out of the Box）

`openRocode`（CLI 命令：`rocode`）的核心定位是**让多模态大模型（LLM/VLM）直接、安全、精细地操控物理机器人**，而不是做成依附于某一代 VLA/WAM 范式的专用调度胶水：

- **开箱零 VLA 依赖（LLM as Primary Driver）**：开发者只需一个前沿大模型 API（或本地 VLM）加一台机器人/仿真环境，即可通过内置的 **L1 直接空间原子原语**与 **L2 即时轨迹/约束合成**完成绝大多数空间移动与操作任务，把大模型在开放词汇理解、空间几何推理和代码生成上的能力彻底释放出来。
- **Rust 守护 500Hz 物理边界（Deterministic Physical Sandbox）**：大模型只有 0.5–2 Hz 的推理频率，且物理世界没有 `git reset`。Rust 实现的 **S0 Safety Arbiter** 负责在 LLM 思考间隙执行高频平滑插值、接触力阈值即停（Guarded Move）、关节跳变限幅、心跳超时锁止（`SafeHold`）以及操作员毫秒级接管。
- **连续策略按需外挂（L3 Policy Plugins as Co-processors）**：面对双足动态平衡（50Hz RL）或叠衣服/盲插（30Hz VLA）等超越符号几何带宽的接触任务，系统允许像在 Shell 里调外部命令一样挂载高频策略子进程，由 Rust 看门狗统一督导与抢占。
- **控制面与数据面解耦**：上层仅传递轻量 `FrameRef` 句柄；v0.1 采用轻量本地 IPC，v0.3 无缝升级为 **POSIX 共享内存无锁环形缓冲区（SHM Ring Buffer + Seqlock + Python DLPack 零拷贝）**与跨机 **Zenoh** 总线。

---

## 2. 三层操控能力谱系（L1 / L2 / L3）

| 层级 | `SkillKind` | 执行机制 | 典型技能与用途 |
|---|---|---|---|
| **L1：直接空间原子操控** *(内置标准库，主力)* | `direct_primitive` | LLM 输出目标点/位姿增量/力控条件，Rust 内置插值器 + IK + 力矩守卫直接闭环执行 | `move_ee`（含触碰即停）、`set_gripper`（含空抓检测）、`point_to_3d`、`navigate_xy` |
| **L2：即时轨迹/约束合成** *(零训练泛化)* | `synthesized_trajectory` | LLM 现场生成参数化几何轨迹或姿态约束（如保持杯口水平、圆弧擦拭），跟踪器高频执行 | 擦拭桌面、倾倒液体、绕障曲线、使用工具拨弄物体 |
| **L3：外部连续策略协处理器** *(按需选配插件)* | `policy_plugin` | 通过隔离进程运行高频 RL / VLA / WAM，受 50–80ms 心跳看门狗与前置条件约束 | 人形全身步态控制、柔性织物操作、毫米级精密插拔 |

---

## 3. 分阶段路线图（Roadmap）

### Phase 1 (`v0.1`)：LLM 直接操控标准原语库 + S0 安全沙盒 + 黑匣子底座
- 完善内置 **L1 直接空间原语**（`move_ee`、`set_gripper` 等）契约、前置条件检查与接触力守卫逻辑。
- 跑通 `LLM Planner -> S1 Supervisor -> S0 Safety Arbiter (500Hz) -> MockRobot` 异步抢占闭环与故障注入测试。
- 完成黑匣子录制（`rocode-recorder`）、时间轴回放（`rocode replay`）、外部进程 IPC 看门狗及初版 `Ratatui` 交互终端。

### Phase 2 (`v0.2`)：MuJoCo 零 VLA 闭环验证 + L2 轨迹合成 + 遥操接管
- 接入 `MuJoCo` Adapter（桌面机械臂 / 人形仿真），验证**纯 LLM + L1/L2 原语零训练跑通长程操作任务**。
- 打通操作员一键遥操接管（`Teleop Takeover`）与可选 L3 Policy Plugin（如 `openpi` / `lerobot`）挂载验证。

### Phase 3 (`v0.3`)：轻量高速零拷贝数据流
- 在 `rocode-transport` 落地共享内存零拷贝环形缓冲区（SHM Ring Buffer + Seqlock），实现亚百微秒级多路相机/深度/点云流分发。
- 提供可选的 `Zenoh` 跨机传输后端。

### Phase 4 (`v0.4`)：真机部署与生态桥接
- 提供主流机械臂与人形真机 Adapter（SO-100 / Piper / xArm / Unitree G1）及非侵入式 ROS2 / MCP 桥接。
