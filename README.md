# openRocode (`rocode`)

**Terminal-native Robotics CLI & S2–S1–S0 Agent Harness.**

`openRocode` bridges frontier Vision-Language Models (**System 2**), high-frequency learned policies / VLAs (**System 1**, such as `openpi` and `lerobot`), and deterministic hardware controllers (**System 0**) inside a single-binary, safety-isolated Rust runtime.

## Why openRocode?

Software coding agents (`Claude Code`, `OpenCode`) proved that pairing foundation models with a disciplined execution harness unlocks order-of-magnitude reliability gains. Physical robots require a domain-native harness built around five systems primitives that software CLIs lack:

1. **Multi-Rate Preemptive Runtime**: Coordinates ~0.5 Hz VLM reasoning, 10–50 Hz VLA action chunks, and 200–1000 Hz deterministic safety loops without blocking on Python's GIL or network stalls.
2. **Deterministic S0 Safety Arbiter**: Enforces joint velocity/step-delta clamps, heartbeat watchdogs (`SafeHold` on VLA stall/OOM), and instant human teleop takeover muxing.
3. **Typed Skill Contracts (`SKILL.toml` + `SKILL.md`)**: Pairs natural-language agent instructions with machine-checked physical preconditions, safety envelopes, and bounded retry rules.
4. **Control / Tensor Plane Separation**: Passes lightweight `FrameRef` descriptors on the control plane while evolving the tensor plane from local IPC (`v0.1`) to zero-copy POSIX shared-memory ring buffers and Zenoh (`v0.3`).
5. **Flight Recorder & Time-Travel Replay**: Built-in `MockRobot` fault injection and episode recording for deterministic offline debugging before touching real hardware.

## Workspace Layout

```text
openRocode/
├── crates/
│   ├── rocode-proto/       # Wire schemas: FrameRef, ActionChunk, SkillSpec, SafetyEnvelope, FailureTrace
│   ├── rocode-safety/      # Deterministic S0 Safety Arbiter (clamps, watchdog, teleop mux, E-Stop)
│   ├── rocode-transport/   # Dual-plane IPC traits (v0.1 UDS/IPC -> v0.3 SHM Zero-Copy RingBuffer)
│   ├── rocode-recorder/    # Flight Recorder & deterministic replay engine
│   ├── rocode-context/     # Multimodal context distiller & pre-failure window slicer
│   ├── rocode-adapters/    # EmbodimentAdapter trait + MockRobot with fault injection
│   ├── rocode-vlm/         # S2 VLM orchestrator & tool/MCP bridge
│   ├── rocode-core/        # S2-S1-S0 preemption state machine & skill supervisor
│   └── rocode-cli/         # `rocode` CLI binary & TUI entrypoint
├── python/
│   └── openrocode_sdk/     # Thin Python worker SDK for wrapping VLA / perception processes
├── skills/                 # Declarative skill contracts (SKILL.toml + SKILL.md)
└── docs/
    └── architecture.md     # Full architecture design & v0.1-v0.4 roadmap
```

## Quickstart

```bash
# Check environment and validate workspace contracts
cargo run -p rocode-cli -- check

# Inspect registered skill contracts in ./skills
cargo run -p rocode-cli -- skills

# Run a mock closed-loop smoke simulation with optional fault injection
cargo run -p rocode-cli -- run --adapter mock --steps 20

# Run workspace unit tests (safety arbiter, contracts, state transitions)
cargo test
```

See [`docs/architecture.md`](docs/architecture.md) for the complete design blueprint and milestone roadmap.
