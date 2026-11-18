//! Core wire contracts, skill specifications, and telemetry schemas for openRocode.

use serde::{Deserialize, Serialize};

/// Lightweight descriptor passed across the control plane while bulk pixels/point-clouds
/// reside in the tensor plane (v0.1 IPC buffer -> v0.3 zero-copy SHM ring buffer).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRef {
    pub stream_id: String,
    pub seq: u64,
    pub timestamp_ns: u64,
    pub slot_id: u32,
}

/// Grounded object tracked in the workspace frame for spatial context distillation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: String,
    pub label: String,
    pub xyz_m: [f64; 3],
    pub graspable: bool,
    pub held: bool,
}

/// Proprioceptive & spatial robot snapshot sampled at S0 frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobotState {
    pub timestamp_ns: u64,
    pub joint_positions: Vec<f64>,
    pub joint_velocities: Vec<f64>,
    pub joint_efforts: Vec<f64>,
    pub ee_xyz_m: [f64; 3],
    pub ee_rpy_rad: [f64; 3],
    pub ee_contact_force_n: f64,
    pub gripper_open_ratio: f64,
    pub gripper_contact_detected: bool,
    pub held_object_id: Option<String>,
    pub scene_objects: Vec<SceneObject>,
    pub frames: Vec<FrameRef>,
}

/// Target command emitted by S1 (or Teleop) and deterministically filtered by S0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointCommand {
    pub timestamp_ns: u64,
    pub target_positions: Vec<f64>,
    pub target_ee_xyz_m: Option<[f64; 3]>,
    pub gripper_open_ratio: f64,
}

/// Deterministic safety bounds enforced at S0 before any command reaches hardware.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// Maximum allowed joint jump (radians) in a single S0 control tick.
    pub max_step_delta: f64,
    /// Maximum allowed task-space EE jump (meters) in a single S0 control tick.
    pub max_ee_step_m: f64,
    /// Maximum joint effort magnitude before triggering SafeHold.
    pub max_effort: f64,
    /// Maximum allowed silence (ms) from active S1 skill before Watchdog triggers SafeHold.
    pub heartbeat_timeout_ms: u64,
}

impl Default for SafetyEnvelope {
    fn default() -> Self {
        Self {
            max_step_delta: 0.05,
            max_ee_step_m: 0.004,
            max_effort: 40.0,
            heartbeat_timeout_ms: 80,
        }
    }
}

/// Execution tier of a registered skill in the LLM-first control hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind {
    /// L1: Built-in deterministic spatial/motor primitive driven directly by the LLM (zero VLA dependency).
    DirectPrimitive,
    /// L2: On-the-fly parameterized trajectory or spatial constraint synthesized by the LLM.
    SynthesizedTrajectory,
    /// L3: Optional external high-frequency policy plugin (RL gait, VLA, or WAM co-processor).
    PolicyPlugin,
}

/// Machine-checked physical preconditions declared in `SKILL.toml`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillPreconditions {
    pub require_gripper_open: Option<bool>,
    pub max_base_distance_m: Option<f64>,
}

/// Optional shell command configuration for external L3 policy workers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PolicyWorkerConfig {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// Machine-readable skill contract loaded from `SKILL.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    pub kind: SkillKind,
    pub rate_hz: u32,
    pub heartbeat_timeout_ms: u64,
    pub timeout_sec: f64,
    pub max_retries: u32,
    #[serde(default)]
    pub preconditions: SkillPreconditions,
    #[serde(default)]
    pub worker: Option<PolicyWorkerConfig>,
}

impl SkillSpec {
    pub fn from_toml_str(raw: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(raw)
    }
}

/// Structured skill call emitted by the System 2 LLM driver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillCall {
    pub thought: String,
    pub skill_name: String,
    pub arguments: serde_json::Value,
}

/// Structured failure codes surfaced from S0/S1 back to the S2 LLM driver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    PreconditionRefused,
    InvalidArguments,
    HeartbeatTimeout,
    SafetyEnvelopeViolation,
    GraspSlip,
    ExecutionTimeout,
    WorkerCrashed,
    PreemptedByOperator,
}

/// Distilled pre-failure blackbox trace packaged for S2 diagnosis and re-planning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureTrace {
    pub skill_name: String,
    pub code: FailureCode,
    pub message: String,
    pub window_frames: Vec<FrameRef>,
    pub final_state: RobotState,
}

/// Final outcome of executing a single skill invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SkillOutcome {
    Completed {
        summary: String,
        ticks_elapsed: u64,
        final_state: RobotState,
    },
    Failed(FailureTrace),
}

/// Wire request sent from Rust S1 Supervisor to an isolated L3 Python Policy Worker over UDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerRequest {
    Init {
        skill_name: String,
        arguments: serde_json::Value,
    },
    Step {
        state: RobotState,
    },
    Shutdown,
}

/// Wire response returned by an L3 Python Policy Worker on each step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WorkerResponse {
    Ready,
    Command {
        command: JointCommand,
    },
    Completed {
        summary: String,
    },
    Failed {
        reason: String,
    },
}
