# openRocode (`rocode`)

**LLM-First Terminal Robotics CLI & Embodied Execution Harness.**

`openRocode` gives frontier Multimodal LLMs (`Gemini`, `GPT`, `Claude`, open-weights VLMs) a **safe, real-time physical body** out of the box—**with zero VLA or policy training required**—while still letting you plug in high-frequency RL/VLA/WAM co-processors whenever contact-rich tasks demand them.

## Core Philosophy: Built for LLMs, Grounded in Physics

Just as `Claude Code` and `OpenCode` are built for LLMs (giving them deterministic file/shell primitives rather than forcing them to emit raw machine code), `openRocode` treats the **frontier LLM/VLM as the primary driver** and equips it with a three-tier control hierarchy guarded by a deterministic **Rust S0 Safety Arbiter**:

| Tier | Mode | Who Drives | Typical Tasks |
|---|---|---|---|
| **L1: Direct Spatial Primitives** *(Built-in, Zero VLA)* | `direct_primitive` | LLM directly outputs 2D/3D points, 6D pose deltas, guarded contact moves, and gripper apertures | 80% of everyday rigid-body pick/place, pushing, button pressing, drawer opening, active perception |
| **L2: On-the-fly Trajectory Synthesis** *(Zero VLA)* | `synthesized_trajectory` | LLM synthesizes parameterized curves or spatial constraints tracked in real time | Wiping, pouring, circular motions, obstacle-clearing paths, novel tool use |
| **L3: Policy Co-processors** *(Optional Plugins)* | `policy_plugin` | LLM invokes & supervises external 30–100 Hz RL / VLA / WAM workers (`openpi`, `lerobot`, locomotion policies) | Bipedal balance/walking, deformable cloth folding, sub-millimeter blind insertion |

## Systems Pillars

1. **Multi-Rate Preemptive Rust Runtime**: Decouples ~0.5–2 Hz LLM reasoning from 100–1000 Hz trajectory tracking, contact guarding, and policy supervision—never blocking hardware safety on HTTP latency or Python's GIL.
2. **Deterministic S0 Safety Arbiter**: Enforces step-delta & velocity clamps, contact-force stops, heartbeat watchdogs (`SafeHold` on worker stall/OOM), and instant human teleop takeover muxing.
3. **Control / Tensor Plane Separation**: Passes lightweight `FrameRef` handles on the control plane while evolving the tensor plane from local IPC (`v0.1`) to zero-copy POSIX shared-memory ring buffers and Zenoh (`v0.3`).
4. **Flight Recorder & Time-Travel Replay**: Built-in `MockRobot` fault injection and episode recording for deterministic offline debugging before touching real hardware.

## Workspace Layout

```text
openRocode/
├── crates/
│   ├── rocode-proto/       # Wire schemas: FrameRef, SkillKind (L1/L2/L3), SkillSpec, SafetyEnvelope, FailureTrace
│   ├── rocode-safety/      # Deterministic S0 Safety Arbiter (clamps, watchdog, teleop mux, E-Stop)
│   ├── rocode-transport/   # Dual-plane IPC traits (v0.1 UDS/IPC -> v0.3 SHM Zero-Copy RingBuffer)
│   ├── rocode-recorder/    # Flight Recorder & deterministic replay engine
│   ├── rocode-context/     # Multimodal context distiller & pre-failure window slicer
│   ├── rocode-adapters/    # EmbodimentAdapter trait + MockRobot with fault injection
│   ├── rocode-vlm/         # LLM/VLM orchestrator & tool/MCP bridge
│   ├── rocode-core/        # Preemption state machine & skill supervisor
│   └── rocode-cli/         # `rocode` CLI binary & TUI entrypoint
├── python/
│   └── openrocode_sdk/     # Optional Python SDK for L2/L3 perception & policy plugins
├── skills/                 # Built-in L1 direct primitives (`move_ee`, `set_gripper`) & L3 plugin examples
└── docs/
    └── architecture.md     # Full architecture design & v0.1-v0.4 roadmap
```

## Quickstart

```bash
# Validate environment and skill contracts
cargo run -p rocode-cli -- check

# Inspect L1 direct primitives and optional L3 policy plugins
cargo run -p rocode-cli -- skills

# Run a 500Hz mock closed-loop smoke simulation with synthetic effort-spike fault injection
cargo run -p rocode-cli -- run --adapter mock --steps 20 --fault-spike-at 8

# Run workspace unit tests
cargo test
```

See [`docs/architecture.md`](docs/architecture.md) for the complete design blueprint and milestone roadmap.
