//! System 2 LLM/VLM Orchestrator: supports live HTTP Frontier LLMs (OpenAI / Anthropic / Gemini)
//! as well as a reactive zero-key spatial planner (`SpatialDemoPlanner`) for offline closed-loop testing.

use std::env;

use rocode_proto::{FailureTrace, RobotState, SkillCall, SkillSpec};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStep {
    Invoke(SkillCall),
    Done { summary: String },
}

pub trait System2Planner: Send + Sync {
    fn name(&self) -> &str;
    fn decide_next(
        &self,
        goal: &str,
        skills: &[SkillSpec],
        state: &RobotState,
        history: &[String],
        last_failure: Option<&FailureTrace>,
    ) -> Result<PlanStep, String>;
}

/// Reactive zero-API-key spatial planner that inspects live 3D object states, contact forces,
/// and `FailureTrace` feedback to execute tabletop goals using L1 & L2 primitives.
#[derive(Debug, Default)]
pub struct SpatialDemoPlanner;

impl System2Planner for SpatialDemoPlanner {
    fn name(&self) -> &str {
        "spatial-demo-planner (offline)"
    }

    fn decide_next(
        &self,
        goal: &str,
        _skills: &[SkillSpec],
        state: &RobotState,
        history: &[String],
        last_failure: Option<&FailureTrace>,
    ) -> Result<PlanStep, String> {
        let g = goal.to_lowercase();

        // Goal pattern A: Guarded surface touch / probing
        if g.contains("touch") || g.contains("probe") || g.contains("contact") {
            if history.iter().any(|h| h.contains("Guarded contact detected")) {
                return Ok(PlanStep::Invoke(SkillCall {
                    thought: "Surface height measured via guarded contact; retracting to safe home.".into(),
                    skill_name: "home_arm".into(),
                    arguments: json!({}),
                }));
            }
            if history.len() >= 2 {
                return Ok(PlanStep::Done {
                    summary: "Completed guarded surface contact probe and returned home.".into(),
                });
            }
            return Ok(PlanStep::Invoke(SkillCall {
                thought: "Descending along -Z with stop_on_contact_n=5.0N to detect tabletop.".into(),
                skill_name: "move_ee".into(),
                arguments: json!({
                    "target_xyz_m": [state.ee_xyz_m[0], state.ee_xyz_m[1], -0.01],
                    "stop_on_contact_n": 5.0
                }),
            }));
        }

        // Goal pattern B: L3 policy plugin test
        if g.contains("plugin") || g.contains("l3") {
            if state.held_object_id.as_deref() == Some("red_mug") {
                return Ok(PlanStep::Done {
                    summary: "L3 policy plugin grasped and lifted red_mug.".into(),
                });
            }
            return Ok(PlanStep::Invoke(SkillCall {
                thought: "Delegating grasp to external L3 policy_plugin worker over UDS.".into(),
                skill_name: "pick_object".into(),
                arguments: json!({ "target_id": "red_mug" }),
            }));
        }

        let want_mug = g.contains("mug") || g.contains("cup") || g.contains("tidy") || g.is_empty();
        let want_wipe = g.contains("wipe") || g.contains("spill") || g.contains("clean") || g.contains("tidy");

        let mug = state.scene_objects.iter().find(|o| o.id == "red_mug");
        let coaster = state.scene_objects.iter().find(|o| o.id == "coaster");
        let spill = state.scene_objects.iter().find(|o| o.id == "spill_patch");

        // 1. Place mug onto coaster if desired and not yet on coaster
        if want_mug
            && let (Some(mug_obj), Some(coaster_obj)) = (mug, coaster)
        {
            let mug_xy_dist_to_coaster = ((mug_obj.xyz_m[0] - coaster_obj.xyz_m[0]).powi(2)
                + (mug_obj.xyz_m[1] - coaster_obj.xyz_m[1]).powi(2))
            .sqrt();
            let mug_placed = !mug_obj.held && mug_xy_dist_to_coaster < 0.04;

            if !mug_placed {
                if state.held_object_id.as_deref() != Some("red_mug") {
                    // Ensure gripper is open before approaching mug
                    if state.gripper_open_ratio < 0.7 {
                        return Ok(PlanStep::Invoke(SkillCall {
                            thought: format!(
                                "Re-opening gripper before re-approaching red_mug (last_failure={:?}).",
                                last_failure.map(|f| &f.code)
                            ),
                            skill_name: "set_gripper".into(),
                            arguments: json!({ "open_ratio": 1.0 }),
                        }));
                    }
                    let dist_to_mug = dist3(state.ee_xyz_m, mug_obj.xyz_m);
                    if dist_to_mug > 0.025 {
                        return Ok(PlanStep::Invoke(SkillCall {
                            thought: format!(
                                "Moving EE directly to red_mug at [{:.2}, {:.2}, {:.2}] via L1 move_ee.",
                                mug_obj.xyz_m[0], mug_obj.xyz_m[1], mug_obj.xyz_m[2]
                            ),
                            skill_name: "move_ee".into(),
                            arguments: json!({ "target_xyz_m": mug_obj.xyz_m }),
                        }));
                    }
                    return Ok(PlanStep::Invoke(SkillCall {
                        thought: "Aligned with red_mug; closing gripper with expect_object=true.".into(),
                        skill_name: "set_gripper".into(),
                        arguments: json!({ "open_ratio": 0.0, "expect_object": true }),
                    }));
                } else {
                    // Holding red_mug -> lift if low, move above coaster, lower, and release
                    let target_above = [coaster_obj.xyz_m[0], coaster_obj.xyz_m[1], 0.16];
                    let target_place = [coaster_obj.xyz_m[0], coaster_obj.xyz_m[1], 0.05];
                    let xy_err = ((state.ee_xyz_m[0] - coaster_obj.xyz_m[0]).powi(2)
                        + (state.ee_xyz_m[1] - coaster_obj.xyz_m[1]).powi(2))
                    .sqrt();
                    if xy_err > 0.025 {
                        if state.ee_xyz_m[2] < 0.14 {
                            return Ok(PlanStep::Invoke(SkillCall {
                                thought: "Lifting held red_mug to safe transit height z=0.18m.".into(),
                                skill_name: "move_ee".into(),
                                arguments: json!({
                                    "target_xyz_m": [state.ee_xyz_m[0], state.ee_xyz_m[1], 0.18]
                                }),
                            }));
                        }
                        return Ok(PlanStep::Invoke(SkillCall {
                            thought: "Transporting red_mug above coaster.".into(),
                            skill_name: "move_ee".into(),
                            arguments: json!({ "target_xyz_m": target_above }),
                        }));
                    }
                    if (state.ee_xyz_m[2] - target_place[2]).abs() > 0.02 {
                        return Ok(PlanStep::Invoke(SkillCall {
                            thought: "Lowering red_mug gently onto coaster.".into(),
                            skill_name: "move_ee".into(),
                            arguments: json!({ "target_xyz_m": target_place }),
                        }));
                    }
                    return Ok(PlanStep::Invoke(SkillCall {
                        thought: "Releasing red_mug on coaster.".into(),
                        skill_name: "set_gripper".into(),
                        arguments: json!({ "open_ratio": 1.0 }),
                    }));
                }
            }
        }

        // 2. Wipe spill using L2 synthesized circular trajectory if requested
        if want_wipe
            && let Some(spill_obj) = spill
            && !spill_obj.label.contains("wiped")
        {
            return Ok(PlanStep::Invoke(SkillCall {
                thought: format!(
                    "Synthesizing L2 circular wiping trajectory over spill at [{:.2}, {:.2}, 0.005].",
                    spill_obj.xyz_m[0], spill_obj.xyz_m[1]
                ),
                skill_name: "trace_path".into(),
                arguments: json!({
                    "shape": "circle",
                    "center_xyz_m": [spill_obj.xyz_m[0], spill_obj.xyz_m[1], 0.005],
                    "radius_m": 0.035,
                    "cycles": 2.5,
                    "duration_sec": 0.8
                }),
            }));
        }

        // 3. Return home before finishing
        if dist3(state.ee_xyz_m, [0.30, 0.00, 0.28]) > 0.03 {
            return Ok(PlanStep::Invoke(SkillCall {
                thought: "Task objectives satisfied; retracting manipulator to safe home pose.".into(),
                skill_name: "home_arm".into(),
                arguments: json!({}),
            }));
        }

        Ok(PlanStep::Done {
            summary: "Completed tabletop task via direct LLM L1/L2 primitives.".into(),
        })
    }
}

/// Live HTTP Frontier LLM Planner supporting OpenAI-compatible (`/v1/chat/completions`)
/// and Anthropic (`/v1/messages`) endpoints via environment keys.
pub struct ApiLlmPlanner {
    provider: ApiProvider,
    model: String,
    api_key: String,
    base_url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    OpenAiCompatible,
    Anthropic,
}

impl ApiLlmPlanner {
    pub fn from_env(model_override: Option<&str>) -> Result<Self, String> {
        if let Ok(key) = env::var("OPENAI_API_KEY")
            && !key.trim().is_empty()
        {
            let base_url = env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
            let model = model_override
                .map(ToOwned::to_owned)
                .or_else(|| env::var("ROCODE_MODEL").ok())
                .unwrap_or_else(|| "gpt-4.1".to_string());
            return Ok(Self {
                provider: ApiProvider::OpenAiCompatible,
                model,
                api_key: key,
                base_url,
            });
        }
        if let Ok(key) = env::var("ANTHROPIC_API_KEY")
            && !key.trim().is_empty()
        {
            let base_url = env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string());
            let model = model_override
                .map(ToOwned::to_owned)
                .or_else(|| env::var("ROCODE_MODEL").ok())
                .unwrap_or_else(|| "claude-sonnet-4-5".to_string());
            return Ok(Self {
                provider: ApiProvider::Anthropic,
                model,
                api_key: key,
                base_url,
            });
        }
        Err("Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set in environment".into())
    }
}

impl System2Planner for ApiLlmPlanner {
    fn name(&self) -> &str {
        &self.model
    }

    fn decide_next(
        &self,
        goal: &str,
        skills: &[SkillSpec],
        state: &RobotState,
        history: &[String],
        last_failure: Option<&FailureTrace>,
    ) -> Result<PlanStep, String> {
        let skill_catalog: Vec<serde_json::Value> = skills
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "kind": s.kind,
                    "description": s.description,
                })
            })
            .collect();

        let system_prompt = "You are openRocode, an LLM-first embodied robot controller. \
Control the robot directly using L1 spatial primitives (`move_ee`, `set_gripper`, `home_arm`), \
L2 synthesized trajectories (`trace_path`), or optional L3 plugins. \
Respond ONLY with a single JSON object matching either:\n\
{\"done\": false, \"thought\": \"...\", \"skill_name\": \"...\", \"arguments\": {...}}\n\
or\n\
{\"done\": true, \"summary\": \"...\"}";

        let user_payload = json!({
            "goal": goal,
            "available_skills": skill_catalog,
            "robot_state": {
                "ee_xyz_m": state.ee_xyz_m,
                "ee_contact_force_n": state.ee_contact_force_n,
                "gripper_open_ratio": state.gripper_open_ratio,
                "gripper_contact_detected": state.gripper_contact_detected,
                "held_object_id": state.held_object_id,
                "scene_objects": state.scene_objects,
            },
            "recent_history": history,
            "last_failure": last_failure,
        });

        let raw_text = match self.provider {
            ApiProvider::OpenAiCompatible => {
                let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
                let body = json!({
                    "model": self.model,
                    "response_format": { "type": "json_object" },
                    "messages": [
                        { "role": "system", "content": system_prompt },
                        { "role": "user", "content": user_payload.to_string() }
                    ]
                });
                let mut resp = ureq::post(&url)
                    .header("Authorization", &format!("Bearer {}", self.api_key))
                    .send_json(&body)
                    .map_err(|e| format!("LLM HTTP error: {e}"))?;
                let val: serde_json::Value = resp
                    .body_mut()
                    .read_json()
                    .map_err(|e| format!("LLM JSON decode error: {e}"))?;
                val["choices"][0]["message"]["content"]
                    .as_str()
                    .ok_or_else(|| format!("missing choices[0].message.content in {val}"))?
                    .to_string()
            }
            ApiProvider::Anthropic => {
                let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
                let body = json!({
                    "model": self.model,
                    "max_tokens": 512,
                    "system": system_prompt,
                    "messages": [
                        { "role": "user", "content": user_payload.to_string() }
                    ]
                });
                let mut resp = ureq::post(&url)
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .send_json(&body)
                    .map_err(|e| format!("Anthropic HTTP error: {e}"))?;
                let val: serde_json::Value = resp
                    .body_mut()
                    .read_json()
                    .map_err(|e| format!("Anthropic JSON decode error: {e}"))?;
                val["content"][0]["text"]
                    .as_str()
                    .ok_or_else(|| format!("missing content[0].text in {val}"))?
                    .to_string()
            }
        };

        parse_llm_json(&raw_text)
    }
}

fn parse_llm_json(raw: &str) -> Result<PlanStep, String> {
    let trimmed = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let val: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("invalid JSON from LLM '{trimmed}': {e}"))?;
    if val.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
        let summary = val
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("Task completed")
            .to_string();
        Ok(PlanStep::Done { summary })
    } else {
        let thought = val
            .get("thought")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let skill_name = val
            .get("skill_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("LLM JSON missing skill_name: {val}"))?
            .to_string();
        let arguments = val.get("arguments").cloned().unwrap_or_else(|| json!({}));
        Ok(PlanStep::Invoke(SkillCall {
            thought,
            skill_name,
            arguments,
        }))
    }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
