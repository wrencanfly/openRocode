//! System 2 VLM Orchestrator traits and structured Skill Call payloads.

use rocode_proto::{FailureTrace, RobotState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInvocation {
    pub thought: String,
    pub skill_name: String,
    pub arguments: serde_json::Value,
}

pub trait System2Planner: Send + Sync {
    fn decide_next_skill(
        &self,
        goal: &str,
        state: &RobotState,
        last_failure: Option<&FailureTrace>,
    ) -> Option<SkillInvocation>;
}
