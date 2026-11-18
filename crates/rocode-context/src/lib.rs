//! Multimodal Context Distiller: maintains a bounded keyframe ring buffer and extracts
//! pre-failure blackbox windows for System 2 diagnosis.

use std::collections::VecDeque;

use rocode_proto::{FailureCode, FailureTrace, FrameRef, RobotState};

pub struct ContextRingBuffer {
    window_ns: u64,
    frames: VecDeque<FrameRef>,
}

impl ContextRingBuffer {
    pub fn new(window_ns: u64) -> Self {
        Self {
            window_ns,
            frames: VecDeque::new(),
        }
    }

    pub fn push_frame(&mut self, frame: FrameRef) {
        let cutoff = frame.timestamp_ns.saturating_sub(self.window_ns);
        self.frames.push_back(frame);
        while let Some(front) = self.frames.front() {
            if front.timestamp_ns < cutoff {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn build_failure_trace(
        &self,
        skill_name: impl Into<String>,
        code: FailureCode,
        message: impl Into<String>,
        final_state: RobotState,
    ) -> FailureTrace {
        FailureTrace {
            skill_name: skill_name.into(),
            code,
            message: message.into(),
            window_frames: self.frames.iter().cloned().collect(),
            final_state,
        }
    }
}
