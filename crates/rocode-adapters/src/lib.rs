//! EmbodimentAdapter trait and built-in `MockRobotAdapter` with fault injection.

use rocode_proto::{FrameRef, JointCommand, RobotState};

pub trait EmbodimentAdapter: Send {
    fn name(&self) -> &str;
    fn read_state(&mut self) -> RobotState;
    fn apply_command(&mut self, cmd: &JointCommand);
}

/// Pure-Rust synthetic 6-DoF arm + gripper embodiment with deterministic clock
/// and optional fault injection for v0.1 offline testing.
pub struct MockRobotAdapter {
    now_ns: u64,
    dt_ns: u64,
    seq: u64,
    positions: Vec<f64>,
    efforts: Vec<f64>,
    gripper_open_ratio: f64,
    fault_effort_spike_at_step: Option<u64>,
}

impl MockRobotAdapter {
    pub fn new(dof: usize, dt_ns: u64) -> Self {
        Self {
            now_ns: 0,
            dt_ns,
            seq: 0,
            positions: vec![0.0; dof],
            efforts: vec![0.5; dof],
            gripper_open_ratio: 1.0,
            fault_effort_spike_at_step: None,
        }
    }

    pub fn with_effort_spike_at_step(mut self, step: u64) -> Self {
        self.fault_effort_spike_at_step = Some(step);
        self
    }
}

impl EmbodimentAdapter for MockRobotAdapter {
    fn name(&self) -> &str {
        "mock-6dof"
    }

    fn read_state(&mut self) -> RobotState {
        self.seq += 1;
        self.now_ns += self.dt_ns;
        let mut efforts = self.efforts.clone();
        if self.fault_effort_spike_at_step == Some(self.seq) && !efforts.is_empty() {
            efforts[0] = 99.0;
        }
        RobotState {
            timestamp_ns: self.now_ns,
            joint_positions: self.positions.clone(),
            joint_velocities: vec![0.0; self.positions.len()],
            joint_efforts: efforts,
            gripper_open_ratio: self.gripper_open_ratio,
            frames: vec![FrameRef {
                stream_id: "wrist_rgb".to_string(),
                seq: self.seq,
                timestamp_ns: self.now_ns,
                slot_id: (self.seq % 64) as u32,
            }],
        }
    }

    fn apply_command(&mut self, cmd: &JointCommand) {
        self.positions = cmd.target_positions.clone();
        self.gripper_open_ratio = cmd.gripper_open_ratio;
    }
}
