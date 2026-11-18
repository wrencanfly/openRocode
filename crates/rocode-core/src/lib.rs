//! S2-S1-S0 Preemption State Machine and Harness Supervisor.

pub mod primitives;

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use primitives::{ActiveSkillRunner, StepStatus};
use rocode_adapters::EmbodimentAdapter;
use rocode_context::ContextRingBuffer;
use rocode_proto::{
    FailureCode, FailureTrace, JointCommand, RobotState, SafetyEnvelope, SkillCall, SkillOutcome,
    SkillSpec,
};
use rocode_recorder::{FlightRecorder, RecordedEvent};
use rocode_safety::{ArbiterMode, SafetyArbiter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePhase {
    Idle,
    Prechecking,
    Executing,
    SafeHold,
    TeleopTakeover,
    EStop,
}

struct RunningSkill {
    spec: SkillSpec,
    call: SkillCall,
    runner: ActiveSkillRunner,
    ticks_elapsed: u64,
    started_holding_object: bool,
}

pub struct HarnessSupervisor<A: EmbodimentAdapter> {
    workdir: PathBuf,
    adapter: A,
    arbiter: SafetyArbiter,
    context: ContextRingBuffer,
    recorder: FlightRecorder,
    skills: HashMap<String, SkillSpec>,
    phase: RuntimePhase,
    tick_counter: u64,
    last_state: RobotState,
    active_skill: Option<RunningSkill>,
}

impl<A: EmbodimentAdapter> HarnessSupervisor<A> {
    pub fn new(
        workdir: impl Into<PathBuf>,
        mut adapter: A,
        envelope: SafetyEnvelope,
        failure_window_ns: u64,
        recorder_stride: u64,
    ) -> Self {
        let initial_state = adapter.read_state();
        let mut recorder = FlightRecorder::new_with_stride(recorder_stride);
        recorder.record(RecordedEvent::EpisodeHeader {
            adapter: adapter.name().to_string(),
            goal: String::new(),
            envelope: envelope.clone(),
        });
        Self {
            workdir: workdir.into(),
            adapter,
            arbiter: SafetyArbiter::new(envelope),
            context: ContextRingBuffer::new(failure_window_ns),
            recorder,
            skills: HashMap::new(),
            phase: RuntimePhase::Idle,
            tick_counter: 0,
            last_state: initial_state,
            active_skill: None,
        }
    }

    pub fn register_skill(&mut self, spec: SkillSpec) {
        self.skills.insert(spec.name.clone(), spec);
    }

    pub fn load_skills_from_dir(&mut self, dir: &Path) -> usize {
        let mut count = 0;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let manifest = entry.path().join("SKILL.toml");
                if manifest.is_file()
                    && let Ok(raw) = fs::read_to_string(&manifest)
                    && let Ok(spec) = SkillSpec::from_toml_str(&raw)
                {
                    self.register_skill(spec);
                    count += 1;
                }
            }
        }
        count
    }

    pub fn skills(&self) -> &HashMap<String, SkillSpec> {
        &self.skills
    }

    pub fn phase(&self) -> RuntimePhase {
        self.phase
    }

    pub fn arbiter(&self) -> &SafetyArbiter {
        &self.arbiter
    }

    pub fn last_state(&self) -> &RobotState {
        &self.last_state
    }

    pub fn active_skill_name(&self) -> Option<&str> {
        self.active_skill.as_ref().map(|s| s.spec.name.as_str())
    }

    pub fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    pub fn recorder(&self) -> &FlightRecorder {
        &self.recorder
    }

    pub fn set_episode_goal(&mut self, goal: &str) {
        self.recorder.record(RecordedEvent::EpisodeHeader {
            adapter: self.adapter.name().to_string(),
            goal: goal.to_string(),
            envelope: self.arbiter.envelope().clone(),
        });
    }

    pub fn check_preconditions(&self, spec: &SkillSpec, state: &RobotState) -> Result<(), String> {
        if let Some(true) = spec.preconditions.require_gripper_open
            && state.gripper_open_ratio < 0.70
        {
            return Err(format!(
                "precondition require_gripper_open failed (open_ratio={:.2})",
                state.gripper_open_ratio
            ));
        }
        if let Some(max_r) = spec.preconditions.max_base_distance_m {
            let r = (state.ee_xyz_m[0].powi(2) + state.ee_xyz_m[1].powi(2)).sqrt();
            if r > max_r {
                return Err(format!(
                    "EE radial reach {:.2}m exceeds max_base_distance_m {:.2}m",
                    r, max_r
                ));
            }
        }
        Ok(())
    }

    /// Start executing an LLM skill call asynchronously across subsequent `tick_once` steps.
    pub fn start_skill(&mut self, call: SkillCall) -> Result<(), FailureTrace> {
        if matches!(
            self.phase,
            RuntimePhase::SafeHold | RuntimePhase::TeleopTakeover | RuntimePhase::EStop
        ) {
            return Err(self.capture_failure(
                &call.skill_name,
                FailureCode::PreconditionRefused,
                format!("cannot start skill while harness is in {:?}", self.phase),
            ));
        }

        self.recorder.record(RecordedEvent::LlmDecision {
            timestamp_ns: self.last_state.timestamp_ns,
            call: call.clone(),
        });

        let Some(spec) = self.skills.get(&call.skill_name).cloned() else {
            return Err(self.capture_failure(
                &call.skill_name,
                FailureCode::InvalidArguments,
                format!("unknown skill '{}'", call.skill_name),
            ));
        };

        self.phase = RuntimePhase::Prechecking;
        if let Err(msg) = self.check_preconditions(&spec, &self.last_state) {
            self.phase = RuntimePhase::Idle;
            return Err(self.capture_failure(&spec.name, FailureCode::PreconditionRefused, msg));
        }

        let runner = ActiveSkillRunner::instantiate(
            &self.workdir,
            &spec,
            &call,
            &self.last_state,
            self.adapter.dt_ns(),
        )
        .map_err(|(code, msg)| {
            self.phase = RuntimePhase::Idle;
            self.capture_failure(&spec.name, code, msg)
        })?;

        let started_holding_object = self.last_state.gripper_contact_detected
            && self.last_state.gripper_open_ratio < 0.35
            && spec.name != "set_gripper";
        self.phase = RuntimePhase::Executing;
        self.arbiter.engage_autonomous(self.last_state.timestamp_ns);
        self.active_skill = Some(RunningSkill {
            spec,
            call,
            runner,
            ticks_elapsed: 0,
            started_holding_object,
        });
        Ok(())
    }

    /// Advance the S0/S1 loop by one deterministic tick. Returns `Some(SkillOutcome)` whenever
    /// an active skill completes, fails, or is preempted.
    pub fn tick_once(&mut self, teleop_cmd: Option<&JointCommand>) -> Option<SkillOutcome> {
        self.tick_counter += 1;
        let state = self.adapter.read_state();
        for frame in &state.frames {
            self.context.push_frame(frame.clone());
        }

        let mut s1_cmd: Option<JointCommand> = None;
        let mut s1_heartbeat = false;
        let mut finished_outcome: Option<SkillOutcome> = None;

        if let Some(mut active) = self.active_skill.take() {
            active.ticks_elapsed += 1;
            if active.started_holding_object && !state.gripper_contact_detected {
                self.arbiter.finish_autonomous_to_idle(&state);
                self.phase = RuntimePhase::Idle;
                let trace = self.context.build_failure_trace(
                    active.spec.name,
                    FailureCode::GraspSlip,
                    format!(
                        "Held object slipped mid-motion at [{:.3}, {:.3}, {:.3}]",
                        state.ee_xyz_m[0], state.ee_xyz_m[1], state.ee_xyz_m[2]
                    ),
                    state.clone(),
                );
                self.recorder.record(RecordedEvent::FailureCaptured {
                    timestamp_ns: state.timestamp_ns,
                    trace: trace.clone(),
                });
                finished_outcome = Some(SkillOutcome::Failed(trace));
            } else {
                match active.runner.step(&state) {
                    StepStatus::Running {
                        cmd,
                        heartbeat_fresh,
                    } => {
                        s1_cmd = Some(cmd);
                        s1_heartbeat = heartbeat_fresh;
                        self.active_skill = Some(active);
                    }
                    StepStatus::Completed { summary } => {
                        self.arbiter.finish_autonomous_to_idle(&state);
                        self.phase = RuntimePhase::Idle;
                        self.recorder.record(RecordedEvent::SkillCompleted {
                            timestamp_ns: state.timestamp_ns,
                            skill_name: active.spec.name.clone(),
                            summary: summary.clone(),
                            ticks_elapsed: active.ticks_elapsed,
                        });
                        finished_outcome = Some(SkillOutcome::Completed {
                            summary,
                            ticks_elapsed: active.ticks_elapsed,
                            final_state: state.clone(),
                        });
                    }
                    StepStatus::Failed { code, message } => {
                        if matches!(
                            code,
                            FailureCode::HeartbeatTimeout
                                | FailureCode::SafetyEnvelopeViolation
                                | FailureCode::WorkerCrashed
                        ) {
                            self.arbiter.trigger_safe_hold(&state);
                            self.phase = RuntimePhase::SafeHold;
                        } else {
                            self.arbiter.finish_autonomous_to_idle(&state);
                            self.phase = RuntimePhase::Idle;
                        }
                        let trace = self.context.build_failure_trace(
                            active.spec.name,
                            code,
                            message,
                            state.clone(),
                        );
                        self.recorder.record(RecordedEvent::FailureCaptured {
                            timestamp_ns: state.timestamp_ns,
                            trace: trace.clone(),
                        });
                        finished_outcome = Some(SkillOutcome::Failed(trace));
                    }
                }
            }
        }

        if s1_heartbeat {
            self.arbiter.record_s1_heartbeat(state.timestamp_ns);
        }

        let filtered =
            self.arbiter
                .filter_command(state.timestamp_ns, &state, s1_cmd.as_ref(), teleop_cmd);
        if let Some(ref safe_cmd) = filtered {
            self.adapter.apply_command(safe_cmd);
        }

        // Check if S0 Safety Arbiter autonomously intercepted an active skill into SafeHold
        if self.arbiter.mode() == ArbiterMode::SafeHold && self.phase == RuntimePhase::Executing {
            self.phase = RuntimePhase::SafeHold;
            if let Some(aborted) = self.active_skill.take() {
                let _ = &aborted.call;
                let trace = self.context.build_failure_trace(
                    aborted.spec.name,
                    FailureCode::SafetyEnvelopeViolation,
                    "S0 Safety Arbiter intercepted execution into SafeHold (effort spike or missed watchdog)",
                    state.clone(),
                );
                self.recorder.record(RecordedEvent::FailureCaptured {
                    timestamp_ns: state.timestamp_ns,
                    trace: trace.clone(),
                });
                finished_outcome = Some(SkillOutcome::Failed(trace));
            }
        }

        self.recorder.record(RecordedEvent::Tick {
            tick: self.tick_counter,
            mode: self.arbiter.mode(),
            state: state.clone(),
            command: filtered,
        });

        self.last_state = state;
        finished_outcome
    }

    /// Blocking convenience helper that runs `call` (with bounded local retries on `GraspSlip`)
    /// to completion. Ideal for CLI batch runs and integration tests.
    pub fn execute_skill_blocking(&mut self, call: SkillCall) -> SkillOutcome {
        let max_retries = self
            .skills
            .get(&call.skill_name)
            .map(|s| s.max_retries)
            .unwrap_or(0);

        let mut attempt = 0;
        loop {
            if let Err(trace) = self.start_skill(call.clone()) {
                return SkillOutcome::Failed(trace);
            }
            let outcome = loop {
                if let Some(done) = self.tick_once(None) {
                    break done;
                }
            };
            match &outcome {
                SkillOutcome::Failed(trace)
                    if trace.code == FailureCode::GraspSlip
                        && call.skill_name == "set_gripper"
                        && attempt < max_retries =>
                {
                    attempt += 1;
                    continue;
                }
                _ => return outcome,
            }
        }
    }

    pub fn operator_hold(&mut self) -> Option<SkillOutcome> {
        let state = self.last_state.clone();
        self.arbiter.trigger_safe_hold(&state);
        self.phase = RuntimePhase::SafeHold;
        self.recorder.record(RecordedEvent::OperatorNote {
            timestamp_ns: state.timestamp_ns,
            action: "operator_hold".into(),
        });
        self.preempt_active_skill("Preempted by operator /hold")
    }

    pub fn operator_resume_idle(&mut self) {
        let state = self.last_state.clone();
        self.arbiter.clear_safe_hold_to_idle(&state);
        if self.phase == RuntimePhase::SafeHold {
            self.phase = RuntimePhase::Idle;
        }
        self.recorder.record(RecordedEvent::OperatorNote {
            timestamp_ns: state.timestamp_ns,
            action: "operator_resume_idle".into(),
        });
    }

    pub fn operator_engage_teleop(&mut self) -> Option<SkillOutcome> {
        let state = self.last_state.clone();
        self.arbiter.engage_teleop(&state);
        self.phase = RuntimePhase::TeleopTakeover;
        self.recorder.record(RecordedEvent::OperatorNote {
            timestamp_ns: state.timestamp_ns,
            action: "teleop_engaged".into(),
        });
        self.preempt_active_skill("Preempted by operator teleop takeover")
    }

    pub fn operator_release_teleop(&mut self) {
        let state = self.last_state.clone();
        self.arbiter.release_teleop_to_idle(&state);
        if self.phase == RuntimePhase::TeleopTakeover {
            self.phase = RuntimePhase::Idle;
        }
        self.recorder.record(RecordedEvent::OperatorNote {
            timestamp_ns: state.timestamp_ns,
            action: "teleop_released".into(),
        });
    }

    pub fn operator_estop(&mut self) -> Option<SkillOutcome> {
        self.arbiter.trigger_estop();
        self.phase = RuntimePhase::EStop;
        self.recorder.record(RecordedEvent::OperatorNote {
            timestamp_ns: self.last_state.timestamp_ns,
            action: "estop_triggered".into(),
        });
        self.preempt_active_skill("Emergency Stop triggered")
    }

    pub fn operator_reset_world(&mut self) {
        let _ = self.active_skill.take();
        self.adapter.reset_world();
        let state = self.adapter.read_state();
        self.arbiter.reset_estop_to_idle(&state);
        self.arbiter.clear_safe_hold_to_idle(&state);
        self.arbiter.release_teleop_to_idle(&state);
        self.phase = RuntimePhase::Idle;
        self.last_state = state;
    }

    fn preempt_active_skill(&mut self, reason: &str) -> Option<SkillOutcome> {
        let aborted = self.active_skill.take()?;
        let trace = self.context.build_failure_trace(
            aborted.spec.name,
            FailureCode::PreemptedByOperator,
            reason,
            self.last_state.clone(),
        );
        self.recorder.record(RecordedEvent::FailureCaptured {
            timestamp_ns: self.last_state.timestamp_ns,
            trace: trace.clone(),
        });
        Some(SkillOutcome::Failed(trace))
    }

    fn capture_failure(
        &mut self,
        skill_name: &str,
        code: FailureCode,
        message: String,
    ) -> FailureTrace {
        let trace = self.context.build_failure_trace(
            skill_name,
            code,
            message,
            self.last_state.clone(),
        );
        self.recorder.record(RecordedEvent::FailureCaptured {
            timestamp_ns: self.last_state.timestamp_ns,
            trace: trace.clone(),
        });
        trace
    }
}
