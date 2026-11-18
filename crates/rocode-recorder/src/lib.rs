//! Blackbox Flight Recorder & Deterministic Time-Travel Replay Engine (`.rlog` NDJSON).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use rocode_proto::{FailureTrace, JointCommand, RobotState, SafetyEnvelope, SkillCall};
use rocode_safety::ArbiterMode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("recorder I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("recorder JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RecordedEvent {
    EpisodeHeader {
        adapter: String,
        goal: String,
        envelope: SafetyEnvelope,
    },
    LlmDecision {
        timestamp_ns: u64,
        call: SkillCall,
    },
    SkillCompleted {
        timestamp_ns: u64,
        skill_name: String,
        summary: String,
        ticks_elapsed: u64,
    },
    Tick {
        tick: u64,
        mode: ArbiterMode,
        state: RobotState,
        command: Option<JointCommand>,
    },
    FailureCaptured {
        timestamp_ns: u64,
        trace: FailureTrace,
    },
    OperatorNote {
        timestamp_ns: u64,
        action: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct EpisodeSummary {
    pub adapter: String,
    pub goal: String,
    pub total_events: usize,
    pub total_ticks: u64,
    pub duration_ms: u64,
    pub llm_decisions: usize,
    pub skills_completed: usize,
    pub failures_captured: usize,
    pub safe_hold_ticks: u64,
    pub teleop_ticks: u64,
}

#[derive(Debug, Default, Clone)]
pub struct FlightRecorder {
    events: Vec<RecordedEvent>,
    tick_stride: u64,
}

impl FlightRecorder {
    /// `tick_stride = 1` records every S0 tick; larger strides decimate high-rate ticks while
    /// keeping every LLM decision, skill completion, and failure event at full fidelity.
    pub fn new_with_stride(tick_stride: u64) -> Self {
        Self {
            events: Vec::new(),
            tick_stride: tick_stride.max(1),
        }
    }

    pub fn new() -> Self {
        Self::new_with_stride(1)
    }

    pub fn record(&mut self, event: RecordedEvent) {
        if let RecordedEvent::Tick {
            tick,
            mode: ArbiterMode::Autonomous | ArbiterMode::Idle,
            ..
        } = &event
            && tick % self.tick_stride != 0
        {
            return;
        }
        self.events.push(event);
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), RecorderError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for ev in &self.events {
            serde_json::to_writer(&mut writer, ev)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn load_from_path(path: &Path) -> Result<Self, RecorderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                events.push(serde_json::from_str(trimmed)?);
            }
        }
        Ok(Self {
            events,
            tick_stride: 1,
        })
    }

    pub fn summary(&self) -> EpisodeSummary {
        let mut s = EpisodeSummary {
            total_events: self.events.len(),
            ..Default::default()
        };
        let mut max_ts_ns = 0u64;
        for ev in &self.events {
            match ev {
                RecordedEvent::EpisodeHeader { adapter, goal, .. } => {
                    s.adapter = adapter.clone();
                    s.goal = goal.clone();
                }
                RecordedEvent::LlmDecision { timestamp_ns, .. } => {
                    s.llm_decisions += 1;
                    max_ts_ns = max_ts_ns.max(*timestamp_ns);
                }
                RecordedEvent::SkillCompleted { timestamp_ns, .. } => {
                    s.skills_completed += 1;
                    max_ts_ns = max_ts_ns.max(*timestamp_ns);
                }
                RecordedEvent::Tick {
                    tick, mode, state, ..
                } => {
                    s.total_ticks = s.total_ticks.max(*tick);
                    max_ts_ns = max_ts_ns.max(state.timestamp_ns);
                    if *mode == ArbiterMode::SafeHold {
                        s.safe_hold_ticks += 1;
                    } else if *mode == ArbiterMode::TeleopTakeover {
                        s.teleop_ticks += 1;
                    }
                }
                RecordedEvent::FailureCaptured { timestamp_ns, .. } => {
                    s.failures_captured += 1;
                    max_ts_ns = max_ts_ns.max(*timestamp_ns);
                }
                RecordedEvent::OperatorNote { timestamp_ns, .. } => {
                    max_ts_ns = max_ts_ns.max(*timestamp_ns);
                }
            }
        }
        s.duration_ms = max_ts_ns / 1_000_000;
        s
    }

    /// Find the closest recorded tick snapshot at or before `target_ms`.
    pub fn tick_at_ms(&self, target_ms: u64) -> Option<(u64, ArbiterMode, &RobotState)> {
        let target_ns = target_ms * 1_000_000;
        let mut best = None;
        for ev in &self.events {
            if let RecordedEvent::Tick {
                tick, mode, state, ..
            } = ev
                && state.timestamp_ns <= target_ns
            {
                best = Some((*tick, *mode, state));
            }
        }
        best
    }

    pub fn failures(&self) -> Vec<&FailureTrace> {
        self.events
            .iter()
            .filter_map(|ev| match ev {
                RecordedEvent::FailureCaptured { trace, .. } => Some(trace),
                _ => None,
            })
            .collect()
    }
}
