//! Deterministic S0 Safety Arbiter: enforces joint & task-space step-delta clamps,
//! heartbeat watchdogs, teleop takeover muxing, and emergency stop invariants.

use rocode_proto::{JointCommand, RobotState, SafetyEnvelope};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbiterMode {
    Idle,
    Autonomous,
    SafeHold,
    TeleopTakeover,
    EStop,
}

#[derive(Debug, Clone)]
pub struct SafetyArbiter {
    envelope: SafetyEnvelope,
    mode: ArbiterMode,
    hold_positions: Option<Vec<f64>>,
    hold_ee_xyz: Option<[f64; 3]>,
    last_s1_heartbeat_ns: Option<u64>,
}

impl SafetyArbiter {
    pub fn new(envelope: SafetyEnvelope) -> Self {
        Self {
            envelope,
            mode: ArbiterMode::Idle,
            hold_positions: None,
            hold_ee_xyz: None,
            last_s1_heartbeat_ns: None,
        }
    }

    pub fn envelope(&self) -> &SafetyEnvelope {
        &self.envelope
    }

    pub fn mode(&self) -> ArbiterMode {
        self.mode
    }

    pub fn last_heartbeat_age_ms(&self, now_ns: u64) -> Option<u64> {
        self.last_s1_heartbeat_ns
            .map(|last| now_ns.saturating_sub(last) / 1_000_000)
    }

    pub fn engage_autonomous(&mut self, now_ns: u64) {
        if self.mode != ArbiterMode::EStop && self.mode != ArbiterMode::TeleopTakeover {
            self.mode = ArbiterMode::Autonomous;
            self.last_s1_heartbeat_ns = Some(now_ns);
        }
    }

    pub fn finish_autonomous_to_idle(&mut self, state: &RobotState) {
        if self.mode == ArbiterMode::Autonomous {
            self.mode = ArbiterMode::Idle;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    pub fn record_s1_heartbeat(&mut self, now_ns: u64) {
        self.last_s1_heartbeat_ns = Some(now_ns);
    }

    pub fn trigger_safe_hold(&mut self, state: &RobotState) {
        if self.mode != ArbiterMode::EStop {
            self.mode = ArbiterMode::SafeHold;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    pub fn clear_safe_hold_to_idle(&mut self, state: &RobotState) {
        if self.mode == ArbiterMode::SafeHold {
            self.mode = ArbiterMode::Idle;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    pub fn engage_teleop(&mut self, state: &RobotState) {
        if self.mode != ArbiterMode::EStop {
            self.mode = ArbiterMode::TeleopTakeover;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    pub fn release_teleop_to_idle(&mut self, state: &RobotState) {
        if self.mode == ArbiterMode::TeleopTakeover {
            self.mode = ArbiterMode::Idle;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    pub fn trigger_estop(&mut self) {
        self.mode = ArbiterMode::EStop;
    }

    pub fn reset_estop_to_idle(&mut self, state: &RobotState) {
        if self.mode == ArbiterMode::EStop {
            self.mode = ArbiterMode::Idle;
            self.hold_positions = Some(state.joint_positions.clone());
            self.hold_ee_xyz = Some(state.ee_xyz_m);
        }
    }

    /// Filter an incoming command at the S0 tick rate. Never panics; always returns
    /// a bounded command or `None` when motors must be de-energized (`EStop`).
    pub fn filter_command(
        &mut self,
        now_ns: u64,
        state: &RobotState,
        s1_cmd: Option<&JointCommand>,
        teleop_cmd: Option<&JointCommand>,
    ) -> Option<JointCommand> {
        if self.mode == ArbiterMode::EStop {
            return None;
        }

        // Effort spike check triggers immediate SafeHold in Autonomous mode.
        if self.mode == ArbiterMode::Autonomous
            && state
                .joint_efforts
                .iter()
                .any(|e| e.abs() > self.envelope.max_effort)
        {
            self.trigger_safe_hold(state);
        }

        // Watchdog heartbeat check for S1 policy stalls / crashes.
        if self.mode == ArbiterMode::Autonomous {
            let timeout_ns = self.envelope.heartbeat_timeout_ms * 1_000_000;
            match self.last_s1_heartbeat_ns {
                Some(last_ns) if now_ns.saturating_sub(last_ns) <= timeout_ns => {}
                _ => self.trigger_safe_hold(state),
            }
        }

        match self.mode {
            ArbiterMode::EStop => None,
            ArbiterMode::TeleopTakeover => match teleop_cmd {
                Some(cmd) => {
                    let clamped = self.clamp_step(now_ns, state, cmd);
                    self.hold_positions = Some(clamped.target_positions.clone());
                    self.hold_ee_xyz = clamped.target_ee_xyz_m;
                    Some(clamped)
                }
                None => Some(self.hold_command(now_ns, state)),
            },
            ArbiterMode::Autonomous => match s1_cmd {
                Some(cmd) => Some(self.clamp_step(now_ns, state, cmd)),
                None => Some(self.hold_command(now_ns, state)),
            },
            ArbiterMode::SafeHold | ArbiterMode::Idle => Some(self.hold_command(now_ns, state)),
        }
    }

    fn hold_command(&mut self, now_ns: u64, state: &RobotState) -> JointCommand {
        let pos = self
            .hold_positions
            .get_or_insert_with(|| state.joint_positions.clone())
            .clone();
        let ee = *self.hold_ee_xyz.get_or_insert(state.ee_xyz_m);
        JointCommand {
            timestamp_ns: now_ns,
            target_positions: pos,
            target_ee_xyz_m: Some(ee),
            gripper_open_ratio: state.gripper_open_ratio,
        }
    }

    fn clamp_step(&self, now_ns: u64, state: &RobotState, cmd: &JointCommand) -> JointCommand {
        let max_d = self.envelope.max_step_delta;
        let clamped_joints = if cmd.target_positions.len() == state.joint_positions.len() {
            state
                .joint_positions
                .iter()
                .zip(cmd.target_positions.iter())
                .map(|(&cur, &target)| cur + (target - cur).clamp(-max_d, max_d))
                .collect()
        } else {
            state.joint_positions.clone()
        };

        let clamped_ee = cmd.target_ee_xyz_m.map(|target_xyz| {
            let dx = target_xyz[0] - state.ee_xyz_m[0];
            let dy = target_xyz[1] - state.ee_xyz_m[1];
            let dz = target_xyz[2] - state.ee_xyz_m[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > self.envelope.max_ee_step_m && dist > 1e-12 {
                let scale = self.envelope.max_ee_step_m / dist;
                [
                    state.ee_xyz_m[0] + dx * scale,
                    state.ee_xyz_m[1] + dy * scale,
                    state.ee_xyz_m[2] + dz * scale,
                ]
            } else {
                target_xyz
            }
        });

        JointCommand {
            timestamp_ns: now_ns,
            target_positions: clamped_joints,
            target_ee_xyz_m: clamped_ee,
            gripper_open_ratio: cmd.gripper_open_ratio.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_state() -> RobotState {
        RobotState {
            timestamp_ns: 0,
            joint_positions: vec![0.0, 0.0],
            joint_velocities: vec![0.0, 0.0],
            joint_efforts: vec![1.0, 1.0],
            ee_xyz_m: [0.30, 0.0, 0.25],
            ee_rpy_rad: [0.0, 0.0, 0.0],
            ee_contact_force_n: 0.0,
            gripper_open_ratio: 1.0,
            gripper_contact_detected: false,
            held_object_id: None,
            scene_objects: vec![],
            frames: vec![],
        }
    }

    #[test]
    fn clamps_ood_joint_and_ee_jumps() {
        let mut arbiter = SafetyArbiter::new(SafetyEnvelope::default());
        let state = dummy_state();
        arbiter.engage_autonomous(0);

        let wild_cmd = JointCommand {
            timestamp_ns: 2_000_000,
            target_positions: vec![1.5, -2.0],
            target_ee_xyz_m: Some([1.30, 0.0, 0.25]),
            gripper_open_ratio: 0.5,
        };
        let safe = arbiter
            .filter_command(2_000_000, &state, Some(&wild_cmd), None)
            .expect("expected clamped command");

        assert!((safe.target_positions[0] - 0.05).abs() < 1e-9);
        assert!((safe.target_positions[1] + 0.05).abs() < 1e-9);
        let ee = safe.target_ee_xyz_m.unwrap();
        assert!((ee[0] - 0.304).abs() < 1e-9);
    }

    #[test]
    fn watchdog_triggers_safe_hold_on_stalled_worker() {
        let mut arbiter = SafetyArbiter::new(SafetyEnvelope::default());
        let state = dummy_state();
        arbiter.engage_autonomous(0);

        // 100ms elapsed without S1 heartbeat (> 80ms budget)
        let _ = arbiter.filter_command(100_000_000, &state, None, None);
        assert_eq!(arbiter.mode(), ArbiterMode::SafeHold);
    }

    #[test]
    fn teleop_overrides_autonomous_and_recovers_from_safe_hold() {
        let mut arbiter = SafetyArbiter::new(SafetyEnvelope::default());
        let state = dummy_state();
        arbiter.trigger_safe_hold(&state);
        assert_eq!(arbiter.mode(), ArbiterMode::SafeHold);

        arbiter.engage_teleop(&state);
        assert_eq!(arbiter.mode(), ArbiterMode::TeleopTakeover);

        let teleop_cmd = JointCommand {
            timestamp_ns: 4_000_000,
            target_positions: vec![0.02, -0.01],
            target_ee_xyz_m: Some([0.302, 0.0, 0.25]),
            gripper_open_ratio: 0.0,
        };
        let out = arbiter
            .filter_command(4_000_000, &state, None, Some(&teleop_cmd))
            .unwrap();
        assert_eq!(out.gripper_open_ratio, 0.0);
    }
}
