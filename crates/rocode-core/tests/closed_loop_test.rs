use std::path::PathBuf;

use rocode_adapters::MockRobotAdapter;
use rocode_core::{HarnessSupervisor, RuntimePhase};
use rocode_proto::{
    FailureCode, PolicyWorkerConfig, SafetyEnvelope, SkillCall, SkillKind, SkillOutcome,
    SkillPreconditions, SkillSpec,
};
use rocode_vlm::{PlanStep, SpatialDemoPlanner, System2Planner};
use serde_json::json;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn l1_and_l2_zero_vla_full_tabletop_cleanup_and_slip_recovery() {
    let root = repo_root();
    // Inject a mid-air grasp slip at tick 800 (while transporting the held mug!)
    let robot = MockRobotAdapter::new(6, 2_000_000).with_slip_at_step(800);
    let mut sup = HarnessSupervisor::new(&root, robot, SafetyEnvelope::default(), 3_000_000_000, 5);
    let loaded = sup.load_skills_from_dir(&root.join("skills"));
    assert!(loaded >= 4);

    let planner = SpatialDemoPlanner;
    let goal = "Put the red mug on the coaster and wipe the spill";
    let mut history = Vec::new();
    let mut last_failure = None;
    let mut finished = false;

    for _ in 0..16 {
        let skills_vec: Vec<SkillSpec> = sup.skills().values().cloned().collect();
        match planner
            .decide_next(
                goal,
                &skills_vec,
                sup.last_state(),
                &history,
                last_failure.as_ref(),
            )
            .unwrap()
        {
            PlanStep::Done { .. } => {
                finished = true;
                break;
            }
            PlanStep::Invoke(call) => match sup.execute_skill_blocking(call) {
                SkillOutcome::Completed { summary, .. } => {
                    history.push(summary);
                    last_failure = None;
                }
                SkillOutcome::Failed(trace) => {
                    history.push(format!("FAILED {:?}", trace.code));
                    last_failure = Some(trace);
                }
            },
        }
    }

    assert!(finished, "planner did not complete full cleanup after slip recovery");
    let final_state = sup.last_state();
    let mug = final_state
        .scene_objects
        .iter()
        .find(|o| o.id == "red_mug")
        .unwrap();
    let coaster = final_state
        .scene_objects
        .iter()
        .find(|o| o.id == "coaster")
        .unwrap();
    let xy_dist =
        ((mug.xyz_m[0] - coaster.xyz_m[0]).powi(2) + (mug.xyz_m[1] - coaster.xyz_m[1]).powi(2))
            .sqrt();
    assert!(xy_dist < 0.04, "red_mug was not placed onto coaster: {mug:?}");

    let spill = final_state
        .scene_objects
        .iter()
        .find(|o| o.id == "spill_patch")
        .unwrap();
    assert!(
        spill.label.contains("wiped"),
        "spill_patch was not wiped by L2 trace_path: {spill:?}"
    );
}

#[test]
fn l1_guarded_move_stops_cleanly_on_surface_contact() {
    let root = repo_root();
    let robot = MockRobotAdapter::new(6, 2_000_000);
    let mut sup = HarnessSupervisor::new(&root, robot, SafetyEnvelope::default(), 3_000_000_000, 1);
    sup.load_skills_from_dir(&root.join("skills"));

    let outcome = sup.execute_skill_blocking(SkillCall {
        thought: "Probe table height".into(),
        skill_name: "move_ee".into(),
        arguments: json!({
            "target_xyz_m": [0.35, 0.0, -0.012],
            "stop_on_contact_n": 5.0
        }),
    });

    match outcome {
        SkillOutcome::Completed {
            summary,
            final_state,
            ..
        } => {
            assert!(summary.contains("Guarded contact detected"));
            assert!(final_state.ee_contact_force_n >= 5.0);
            assert_eq!(sup.phase(), RuntimePhase::Idle);
        }
        other => panic!("expected guarded contact completion, got {other:?}"),
    }
}

#[test]
fn l3_python_worker_succeeds_and_stalled_worker_triggers_safe_hold() {
    let root = repo_root();

    // 1. Healthy Python L3 worker over Unix Domain Socket
    {
        let robot = MockRobotAdapter::new(6, 2_000_000);
        let mut sup =
            HarnessSupervisor::new(&root, robot, SafetyEnvelope::default(), 3_000_000_000, 5);
        sup.load_skills_from_dir(&root.join("skills"));

        let outcome = sup.execute_skill_blocking(SkillCall {
            thought: "Run external L3 grasp policy".into(),
            skill_name: "pick_object".into(),
            arguments: json!({ "target_id": "red_mug" }),
        });

        match outcome {
            SkillOutcome::Completed { final_state, .. } => {
                assert_eq!(final_state.held_object_id.as_deref(), Some("red_mug"));
            }
            other => panic!("expected healthy L3 worker to succeed, got {other:?}"),
        }
    }

    // 2. Stalling Python L3 worker (--stall-at-step 3 sleeps 500ms > 80ms watchdog)
    {
        let robot = MockRobotAdapter::new(6, 2_000_000);
        let mut sup =
            HarnessSupervisor::new(&root, robot, SafetyEnvelope::default(), 3_000_000_000, 5);
        sup.register_skill(SkillSpec {
            name: "stalling_policy".into(),
            description: "Simulates a hung VLA worker".into(),
            kind: SkillKind::PolicyPlugin,
            rate_hz: 50,
            heartbeat_timeout_ms: 80,
            timeout_sec: 5.0,
            max_retries: 0,
            preconditions: SkillPreconditions::default(),
            worker: Some(PolicyWorkerConfig {
                program: "python3".into(),
                args: vec![
                    "python/examples/mock_policy_worker.py".into(),
                    "--stall-at-step".into(),
                    "3".into(),
                ],
            }),
        });

        let outcome = sup.execute_skill_blocking(SkillCall {
            thought: "Test watchdog intercept".into(),
            skill_name: "stalling_policy".into(),
            arguments: json!({ "target_id": "red_mug" }),
        });

        match outcome {
            SkillOutcome::Failed(trace) => {
                assert_eq!(trace.code, FailureCode::HeartbeatTimeout);
                assert_eq!(sup.phase(), RuntimePhase::SafeHold);
            }
            other => panic!("expected HeartbeatTimeout -> SafeHold, got {other:?}"),
        }
    }
}
