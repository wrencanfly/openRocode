# openRocode (`rocode`)

**LLM-First Terminal Robotics CLI & Embodied Execution Harness.**

`openRocode` gives frontier Multimodal LLMs (`OpenAI`, `Anthropic`, `Gemini`, open-weights VLMs) a **safe, real-time physical body** out of the box—**with zero VLA or policy training required**—while still letting you plug in high-frequency RL/VLA/WAM co-processors whenever contact-rich tasks demand them.

## Core Philosophy: Built for LLMs, Grounded in Physics

`openRocode` treats the **frontier LLM/VLM as the primary driver** and equips it with a three-tier control hierarchy guarded by a deterministic **500 Hz Rust S0 Safety Arbiter**:

| Tier | Mode | Who Drives | Built-in / Example Skills |
|---|---|---|---|
| **L1: Direct Spatial Primitives** *(Built-in, Zero VLA)* | `direct_primitive` | LLM directly outputs 3D targets, guarded contact thresholds, and gripper apertures | `move_ee` (min-jerk + IK + guarded contact force stop), `set_gripper` (empty-grasp detection), `home_arm` |
| **L2: On-the-fly Trajectory Synthesis** *(Zero VLA)* | `synthesized_trajectory` | LLM synthesizes parametric curves or multi-waypoint paths tracked at 200 Hz | `trace_path` (`circle` / `waypoints` for wiping, stirring, pouring, obstacle clearance) |
| **L3: Policy Co-processors** *(Optional Isolated Plugins)* | `policy_plugin` | LLM invokes external 30–100 Hz RL / VLA / WAM subprocesses over UDS under an 80 ms Watchdog | `pick_object` (`python/examples/mock_policy_worker.py`) |

## Quickstart

```bash
# 1. Launch the interactive Ratatui robotics cockpit (60 FPS UI + 500 Hz S0 loop + 2D workspace radar)
#    Uses OPENAI_API_KEY / ANTHROPIC_API_KEY automatically when set, or built-in spatial planner offline.
cargo run -p rocode-cli -- tui

# 2. Inspect registered L1 / L2 / L3 skill contracts
cargo run -p rocode-cli -- skills

# 3. Run a full headless tabletop episode with mid-air grasp slip injection & autonomous re-grasp recovery
cargo run -p rocode-cli -- run "Put the red mug on the coaster and wipe the spill" \
  --fault-slip-at 800 \
  --record recordings/slip_recovery.rlog

# 4. Time-travel inspect the recorded episode & captured blackbox failure trace
cargo run -p rocode-cli -- replay recordings/slip_recovery.rlog --at-ms 1550 --failures

# 5. Run all unit & end-to-end closed-loop tests (including Python UDS worker + 80ms Watchdog SafeHold)
cargo test
```

### Interactive TUI Slash Commands (`rocode tui`)

- Type any goal (e.g., `put red mug on coaster and wipe spill`, `probe table contact`, `run l3 plugin`) and press `Enter`
- `/takeover` — Instant **Operator Teleop Override** (`W/S`: ±X, `A/D`: ±Y, `Q/E`: ±Z, `Space`: Gripper, `Esc`: Release)
- `/hold` / `/resume` — Trigger or clear S0 `SAFE_HOLD`
- `/inject slip` | `/inject spike` — Inject live grasp-slip or 95 Nm effort-spike faults mid-execution
- `/estop` / `/reset` — Emergency Stop (de-energize motors) or reset tabletop world
- `/save [path]` — Persist `.rlog` flight blackbox to disk

## Workspace Layout

```text
openRocode/
├── crates/
│   ├── rocode-proto/       # Wire schemas: FrameRef, SkillKind (L1/L2/L3), SkillSpec, FailureTrace, UDS protocol
│   ├── rocode-safety/      # Deterministic 500Hz S0 Safety Arbiter (joint/EE clamps, watchdog, teleop mux, E-Stop)
│   ├── rocode-transport/   # Tensor ring buffer + Unix Domain Socket policy subprocess supervisor
│   ├── rocode-recorder/    # Persistent `.rlog` NDJSON Flight Recorder & time-travel replay queries
│   ├── rocode-context/     # Multimodal context ring buffer & pre-failure blackbox window slicer
│   ├── rocode-adapters/    # EmbodimentAdapter + 6-DoF tabletop micro-world (IK, contact spring, grasp physics)
│   ├── rocode-vlm/         # Live HTTP LLM planner (OpenAI / Anthropic) + offline reactive SpatialDemoPlanner
│   ├── rocode-core/        # S2-S1-S0 state machine + L1/L2 trajectory executors + mid-motion slip monitor
│   └── rocode-cli/         # `rocode` CLI (`tui`, `run`, `replay`, `skills`, `check`) + Ratatui cockpit
├── python/
│   ├── openrocode_sdk/     # UDS `PolicyWorker` base class for optional L3 RL/VLA plugins
│   └── examples/           # `mock_policy_worker.py` (with `--stall-at-step` watchdog test hook)
├── skills/                 # L1 (`move_ee`, `set_gripper`, `home_arm`), L2 (`trace_path`), L3 (`pick_object`)
└── docs/
    └── architecture.md     # Full architecture design & v0.1-v0.4 roadmap
```
