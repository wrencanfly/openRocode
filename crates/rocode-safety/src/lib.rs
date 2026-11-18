//! Deterministic S0 Safety Arbiter: enforces step-delta clamps, heartbeat watchdogs,
//! teleop takeover muxing, and emergency stop invariants.

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
    last_s1_heartbeat_ns: Option<u64>,
}

impl SafetyArbiter {
    pub fn new(envelope: SafetyEnvelope) -> Self {
        Self {
            envelope,
            mode: ArbiterMode::Idle,
            hold_positions: None,
            last_s1_heartbeat_ns: None,
        }
    }

    pub fn mode(&self) -> ArbiterMode {
        self.mode
    }

    pub fn engage_autonomous(&mut self, now_ns: u64) {
        if self.mode != ArbiterMode::EStop {
            self.mode = ArbiterMode::Autonomous;
            self.last_s1_heartbeat_ns = Some(now_ns);
        }
    }

    pub fn record_s1_heartbeat(&mut self, now_ns: u64) {
        self.last_s1_heartbeat_ns = Some(now_ns);
    }

    pub fn trigger_safe_hold(&mut self, state: &RobotState) {
        if self.mode != ArbiterMode::EStop {
            self.mode = ArbiterMode::SafeHold;
            self.hold_positions = Some(state.joint_positions.clone());
        }
    }

    pub fn engage_teleop(&mut self) {
        if self.mode != ArbiterMode::EStop {
            self.mode = ArbiterMode::TeleopTakeover;
        }
    }

    pub fn release_teleop_to_idle(&mut self, state: &RobotState) {
        if self.mode == ArbiterMode::TeleopTakeover {
            self.mode = ArbiterMode::Idle;
            self.hold_positions = Some(state.joint_positions.clone());
        }
    }

    pub fn trigger_estop(&mut self) {
        self.mode = ArbiterMode::EStop;
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
            ArbiterMode::TeleopTakeover => {
                let raw = teleop_cmd.cloned().unwrap_or_else(|| JointCommand {
                    timestamp_ns: now_ns,
                    target_positions: state.joint_positions.clone(),
                    gripper_open_ratio: state.gripper_open_ratio,
                });
                Some(self.clamp_step(now_ns, state, &raw))
            }
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
        JointCommand {
            timestamp_ns: now_ns,
            target_positions: pos,
            gripper_open_ratio: state.gripper_open_ratio,
        }
    }

    fn clamp_step(&self, now_ns: u64, state: &RobotState, cmd: &JointCommand) -> JointCommand {
        let max_d = self.envelope.max_step_delta;
        let clamped = state
            .joint_positions
            .iter()
            .zip(cmd.target_positions.iter())
            .map(|(&cur, &target)| cur + (target - cur).clamp(-max_d, max_d))
            .collect();
        JointCommand {
            timestamp_ns: now_ns,
            target_positions: clamped,
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
            gripper_open_ratio: 1.0,
            frames: vec![],
        }
    }

    #[test]
    fn clamps_ood_action_jump() {
        let mut arbiter = SafetyArbiter::new(SafetyEnvelope::default());
        let state = dummy_state();
        arbiter.engage_autonomous(0);

        let wild_cmd = JointCommand {
            timestamp_ns: 2_000_000,
            target_positions: vec![1.5, -2.0],
            gripper_open_ratio: 0.5,
        };
        let safe = arbiter
            .filter_command(2_000_000, &state, Some(&wild_cmd), None)
            .expect("expected clamped command");

        assert!((safe.target_positions[0] - 0.05).abs() < 1e-9);
        assert!((safe.target_positions[1] + 0.05).abs() < 1e-9);
    }

    #[test]
    fn watchdog_triggers_safe_hold_on_stalled_vla() {
        let mut arbiter = SafetyArbiter::new(SafetyEnvelope {
            max_step_delta: 0.05,
            max_effort: 40.0,
            heartbeat_timeout_ms: 80,
        });
        let state = dummy_state();
        arbiter.engage_autonomous(0);

        // 100ms elapsed without S1 heartbeat (> 80ms budget)
        let _ = arbiter.filter_command(100_000_000, &state, None, None);
        assert_eq!(arbiter.mode(), ArbiterMode::SafeHold);
    }
}
