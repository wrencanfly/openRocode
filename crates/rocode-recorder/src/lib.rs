//! Flight Recorder & Deterministic Time-Travel Replay Log.

use rocode_proto::{FailureTrace, JointCommand, RobotState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordedEvent {
    StateSample(RobotState),
    CommandIssued(JointCommand),
    FailureCaptured(FailureTrace),
}

#[derive(Debug, Default)]
pub struct FlightRecorder {
    events: Vec<RecordedEvent>,
}

impl FlightRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, event: RecordedEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }
}
