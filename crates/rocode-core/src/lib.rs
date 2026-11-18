//! S2-S1-S0 Preemption State Machine and Harness Supervisor.

use rocode_adapters::EmbodimentAdapter;
use rocode_context::ContextRingBuffer;
use rocode_proto::{FailureCode, JointCommand, RobotState, SafetyEnvelope, SkillSpec};
use rocode_recorder::{FlightRecorder, RecordedEvent};
use rocode_safety::{ArbiterMode, SafetyArbiter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePhase {
    Idle,
    Prechecking,
    Staging,
    Executing,
    LocalRetry,
    SafeHold,
    TeleopTakeover,
    EStop,
}

pub struct HarnessSupervisor<A: EmbodimentAdapter> {
    adapter: A,
    arbiter: SafetyArbiter,
    context: ContextRingBuffer,
    recorder: FlightRecorder,
    phase: RuntimePhase,
}

impl<A: EmbodimentAdapter> HarnessSupervisor<A> {
    pub fn new(adapter: A, envelope: SafetyEnvelope, failure_window_ns: u64) -> Self {
        Self {
            adapter,
            arbiter: SafetyArbiter::new(envelope),
            context: ContextRingBuffer::new(failure_window_ns),
            recorder: FlightRecorder::new(),
            phase: RuntimePhase::Idle,
        }
    }

    pub fn phase(&self) -> RuntimePhase {
        self.phase
    }

    pub fn recorder(&self) -> &FlightRecorder {
        &self.recorder
    }

    pub fn check_preconditions(&mut self, spec: &SkillSpec, state: &RobotState) -> Result<(), FailureCode> {
        self.phase = RuntimePhase::Prechecking;
        if let Some(true) = spec.preconditions.require_gripper_open
            && state.gripper_open_ratio < 0.8
        {
            self.phase = RuntimePhase::Idle;
            return Err(FailureCode::PreconditionRefused);
        }
        Ok(())
    }

    /// Execute one deterministic S0 control tick while monitoring S1 health.
    pub fn step_tick(
        &mut self,
        active_skill: Option<&str>,
        s1_cmd: Option<&JointCommand>,
        s1_heartbeat_fresh: bool,
    ) -> RobotState {
        let state = self.adapter.read_state();
        for frame in &state.frames {
            self.context.push_frame(frame.clone());
        }
        self.recorder.record(RecordedEvent::StateSample(state.clone()));

        if self.phase == RuntimePhase::Idle && active_skill.is_some() {
            self.phase = RuntimePhase::Executing;
            self.arbiter.engage_autonomous(state.timestamp_ns);
        }

        if s1_heartbeat_fresh {
            self.arbiter.record_s1_heartbeat(state.timestamp_ns);
        }

        if let Some(safe_cmd) = self.arbiter.filter_command(state.timestamp_ns, &state, s1_cmd, None) {
            self.adapter.apply_command(&safe_cmd);
            self.recorder.record(RecordedEvent::CommandIssued(safe_cmd));
        }

        if self.arbiter.mode() == ArbiterMode::SafeHold && self.phase == RuntimePhase::Executing {
            self.phase = RuntimePhase::SafeHold;
            let trace = self.context.build_failure_trace(
                active_skill.unwrap_or("unknown"),
                FailureCode::SafetyEnvelopeViolation,
                "S0 Safety Arbiter engaged SafeHold (effort spike or heartbeat timeout)",
                state.clone(),
            );
            self.recorder.record(RecordedEvent::FailureCaptured(trace));
        }

        state
    }
}
