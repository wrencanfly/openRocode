//! Built-in L1 Direct Spatial Primitives, L2 Trajectory Synthesizers, and L3 Policy Worker runners.

use std::f64::consts::TAU;
use std::path::Path;

use rocode_proto::{
    FailureCode, JointCommand, RobotState, SkillCall, SkillKind, SkillSpec, WorkerResponse,
};
use rocode_transport::{TransportError, UdsPolicySubprocess};

#[inline]
pub fn min_jerk(tau: f64) -> f64 {
    let t = tau.clamp(0.0, 1.0);
    10.0 * t.powi(3) - 15.0 * t.powi(4) + 6.0 * t.powi(5)
}

#[inline]
fn lerp3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * s,
        a[1] + (b[1] - a[1]) * s,
        a[2] + (b[2] - a[2]) * s,
    ]
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub enum StepStatus {
    Running {
        cmd: JointCommand,
        heartbeat_fresh: bool,
    },
    Completed {
        summary: String,
    },
    Failed {
        code: FailureCode,
        message: String,
    },
}

pub enum ActiveSkillRunner {
    MoveEe {
        start_xyz: [f64; 3],
        target_xyz: [f64; 3],
        gripper: f64,
        stop_on_contact_n: Option<f64>,
        total_ticks: u64,
        elapsed_ticks: u64,
    },
    SetGripper {
        hold_xyz: [f64; 3],
        start_open: f64,
        target_open: f64,
        expect_object: bool,
        total_ticks: u64,
        elapsed_ticks: u64,
    },
    TraceCircle {
        entry_start_xyz: [f64; 3],
        center_xyz: [f64; 3],
        radius_m: f64,
        cycles: f64,
        gripper: f64,
        approach_ticks: u64,
        trace_ticks: u64,
        elapsed_ticks: u64,
    },
    TraceWaypoints {
        waypoints: Vec<[f64; 3]>,
        gripper: f64,
        total_ticks: u64,
        elapsed_ticks: u64,
    },
    PolicyPlugin {
        worker: UdsPolicySubprocess,
        decimation: u64,
        elapsed_ticks: u64,
        max_ticks: u64,
        last_cmd: Option<JointCommand>,
    },
}

impl ActiveSkillRunner {
    pub fn instantiate(
        workdir: &Path,
        spec: &SkillSpec,
        call: &SkillCall,
        state: &RobotState,
        dt_ns: u64,
    ) -> Result<Self, (FailureCode, String)> {
        let dt_s = (dt_ns as f64) * 1e-9;
        match spec.kind {
            SkillKind::DirectPrimitive => match spec.name.as_str() {
                "move_ee" => {
                    let raw_xyz = parse_vec3(&call.arguments, "target_xyz_m").ok_or_else(|| {
                        (
                            FailureCode::InvalidArguments,
                            "move_ee requires target_xyz_m: [x, y, z]".into(),
                        )
                    })?;
                    let relative = call
                        .arguments
                        .get("relative")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let target_xyz = if relative {
                        [
                            state.ee_xyz_m[0] + raw_xyz[0],
                            state.ee_xyz_m[1] + raw_xyz[1],
                            state.ee_xyz_m[2] + raw_xyz[2],
                        ]
                    } else {
                        raw_xyz
                    };
                    let stop_on_contact_n = call
                        .arguments
                        .get("stop_on_contact_n")
                        .and_then(|v| v.as_f64());
                    let dist = dist3(state.ee_xyz_m, target_xyz);
                    let dur_s = call
                        .arguments
                        .get("duration_sec")
                        .and_then(|v| v.as_f64())
                        .unwrap_or_else(|| (dist / 0.35).clamp(0.18, spec.timeout_sec));
                    let total_ticks = ((dur_s / dt_s).ceil() as u64).max(5);
                    Ok(Self::MoveEe {
                        start_xyz: state.ee_xyz_m,
                        target_xyz,
                        gripper: state.gripper_open_ratio,
                        stop_on_contact_n,
                        total_ticks,
                        elapsed_ticks: 0,
                    })
                }
                "set_gripper" => {
                    let target_open = call
                        .arguments
                        .get("open_ratio")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| {
                            (
                                FailureCode::InvalidArguments,
                                "set_gripper requires open_ratio in [0.0, 1.0]".into(),
                            )
                        })?
                        .clamp(0.0, 1.0);
                    let expect_object = call
                        .arguments
                        .get("expect_object")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let total_ticks = ((0.20 / dt_s).ceil() as u64).max(5);
                    Ok(Self::SetGripper {
                        hold_xyz: state.ee_xyz_m,
                        start_open: state.gripper_open_ratio,
                        target_open,
                        expect_object,
                        total_ticks,
                        elapsed_ticks: 0,
                    })
                }
                "home_arm" => {
                    let home = [0.30, 0.00, 0.28];
                    let dist = dist3(state.ee_xyz_m, home);
                    let total_ticks = (((dist / 0.35).clamp(0.20, 2.5)) / dt_s).ceil() as u64;
                    Ok(Self::MoveEe {
                        start_xyz: state.ee_xyz_m,
                        target_xyz: home,
                        gripper: state.gripper_open_ratio,
                        stop_on_contact_n: None,
                        total_ticks: total_ticks.max(5),
                        elapsed_ticks: 0,
                    })
                }
                other => Err((
                    FailureCode::InvalidArguments,
                    format!("unknown L1 direct_primitive '{other}'"),
                )),
            },
            SkillKind::SynthesizedTrajectory => {
                let shape = call
                    .arguments
                    .get("shape")
                    .and_then(|v| v.as_str())
                    .unwrap_or("circle");
                let dur_s = call
                    .arguments
                    .get("duration_sec")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.2)
                    .clamp(0.2, spec.timeout_sec);
                match shape {
                    "circle" => {
                        let center_xyz =
                            parse_vec3(&call.arguments, "center_xyz_m").ok_or_else(|| {
                                (
                                    FailureCode::InvalidArguments,
                                    "trace_path(circle) requires center_xyz_m".into(),
                                )
                            })?;
                        let radius_m = call
                            .arguments
                            .get("radius_m")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.04)
                            .clamp(0.005, 0.20);
                        let cycles = call
                            .arguments
                            .get("cycles")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(2.0)
                            .max(0.5);
                        let rim_start = [center_xyz[0] + radius_m, center_xyz[1], center_xyz[2]];
                        let approach_s = (dist3(state.ee_xyz_m, rim_start) / 0.35).clamp(0.15, 1.5);
                        Ok(Self::TraceCircle {
                            entry_start_xyz: state.ee_xyz_m,
                            center_xyz,
                            radius_m,
                            cycles,
                            gripper: state.gripper_open_ratio,
                            approach_ticks: ((approach_s / dt_s).ceil() as u64).max(5),
                            trace_ticks: ((dur_s / dt_s).ceil() as u64).max(10),
                            elapsed_ticks: 0,
                        })
                    }
                    "waypoints" => {
                        let pts_val = call
                            .arguments
                            .get("points_xyz_m")
                            .and_then(|v| v.as_array())
                            .ok_or_else(|| {
                                (
                                    FailureCode::InvalidArguments,
                                    "trace_path(waypoints) requires points_xyz_m array".into(),
                                )
                            })?;
                        let mut waypoints = vec![state.ee_xyz_m];
                        for item in pts_val {
                            if let Some(arr) = item.as_array()
                                && arr.len() == 3
                                && let (Some(x), Some(y), Some(z)) =
                                    (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64())
                            {
                                waypoints.push([x, y, z]);
                            }
                        }
                        if waypoints.len() < 2 {
                            return Err((
                                FailureCode::InvalidArguments,
                                "points_xyz_m must contain at least one valid [x,y,z]".into(),
                            ));
                        }
                        Ok(Self::TraceWaypoints {
                            waypoints,
                            gripper: state.gripper_open_ratio,
                            total_ticks: ((dur_s / dt_s).ceil() as u64).max(10),
                            elapsed_ticks: 0,
                        })
                    }
                    other => Err((
                        FailureCode::InvalidArguments,
                        format!("unsupported trace_path shape '{other}'"),
                    )),
                }
            }
            SkillKind::PolicyPlugin => {
                let worker_cfg = spec.worker.as_ref().ok_or_else(|| {
                    (
                        FailureCode::InvalidArguments,
                        format!("policy_plugin '{}' missing [worker] config", spec.name),
                    )
                })?;
                let worker = UdsPolicySubprocess::spawn(
                    workdir,
                    worker_cfg,
                    &spec.name,
                    &call.arguments,
                    spec.heartbeat_timeout_ms,
                    1500,
                )
                .map_err(map_transport_err)?;
                let s0_hz = (1_000_000_000u64 / dt_ns.max(1)).max(1);
                let decimation = (s0_hz / (spec.rate_hz.max(1) as u64)).max(1);
                let max_ticks = ((spec.timeout_sec / dt_s).ceil() as u64).max(10);
                Ok(Self::PolicyPlugin {
                    worker,
                    decimation,
                    elapsed_ticks: 0,
                    max_ticks,
                    last_cmd: None,
                })
            }
        }
    }

    pub fn step(&mut self, state: &RobotState) -> StepStatus {
        match self {
            Self::MoveEe {
                start_xyz,
                target_xyz,
                gripper,
                stop_on_contact_n,
                total_ticks,
                elapsed_ticks,
            } => {
                if let Some(thresh) = *stop_on_contact_n
                    && *elapsed_ticks > 2
                    && state.ee_contact_force_n >= thresh
                {
                    return StepStatus::Completed {
                        summary: format!(
                            "Guarded contact detected ({:.1} N >= {:.1} N) at xyz=[{:.3}, {:.3}, {:.3}]",
                            state.ee_contact_force_n,
                            thresh,
                            state.ee_xyz_m[0],
                            state.ee_xyz_m[1],
                            state.ee_xyz_m[2]
                        ),
                    };
                }
                *elapsed_ticks += 1;
                if *elapsed_ticks > *total_ticks {
                    return StepStatus::Completed {
                        summary: format!(
                            "Moved EE to [{:.3}, {:.3}, {:.3}]",
                            state.ee_xyz_m[0], state.ee_xyz_m[1], state.ee_xyz_m[2]
                        ),
                    };
                }
                let s = min_jerk((*elapsed_ticks as f64) / (*total_ticks as f64));
                let xyz = lerp3(*start_xyz, *target_xyz, s);
                StepStatus::Running {
                    cmd: JointCommand {
                        timestamp_ns: state.timestamp_ns,
                        target_positions: state.joint_positions.clone(),
                        target_ee_xyz_m: Some(xyz),
                        gripper_open_ratio: *gripper,
                    },
                    heartbeat_fresh: true,
                }
            }
            Self::SetGripper {
                hold_xyz,
                start_open,
                target_open,
                expect_object,
                total_ticks,
                elapsed_ticks,
            } => {
                *elapsed_ticks += 1;
                if *elapsed_ticks > *total_ticks {
                    if *expect_object && *target_open < 0.35 && !state.gripper_contact_detected {
                        return StepStatus::Failed {
                            code: FailureCode::GraspSlip,
                            message: format!(
                                "Gripper closed to {:.2} at [{:.3}, {:.3}, {:.3}] without object contact",
                                state.gripper_open_ratio,
                                state.ee_xyz_m[0],
                                state.ee_xyz_m[1],
                                state.ee_xyz_m[2]
                            ),
                        };
                    }
                    let holding = state.held_object_id.as_deref().unwrap_or("nothing");
                    return StepStatus::Completed {
                        summary: format!(
                            "Gripper at {:.0}% open, holding {holding}",
                            state.gripper_open_ratio * 100.0
                        ),
                    };
                }
                let s = min_jerk((*elapsed_ticks as f64) / (*total_ticks as f64));
                let aperture = *start_open + (*target_open - *start_open) * s;
                StepStatus::Running {
                    cmd: JointCommand {
                        timestamp_ns: state.timestamp_ns,
                        target_positions: state.joint_positions.clone(),
                        target_ee_xyz_m: Some(*hold_xyz),
                        gripper_open_ratio: aperture,
                    },
                    heartbeat_fresh: true,
                }
            }
            Self::TraceCircle {
                entry_start_xyz,
                center_xyz,
                radius_m,
                cycles,
                gripper,
                approach_ticks,
                trace_ticks,
                elapsed_ticks,
            } => {
                *elapsed_ticks += 1;
                let rim_start = [center_xyz[0] + *radius_m, center_xyz[1], center_xyz[2]];
                if *elapsed_ticks <= *approach_ticks {
                    let s = min_jerk((*elapsed_ticks as f64) / (*approach_ticks as f64));
                    let xyz = lerp3(*entry_start_xyz, rim_start, s);
                    return StepStatus::Running {
                        cmd: JointCommand {
                            timestamp_ns: state.timestamp_ns,
                            target_positions: state.joint_positions.clone(),
                            target_ee_xyz_m: Some(xyz),
                            gripper_open_ratio: *gripper,
                        },
                        heartbeat_fresh: true,
                    };
                }
                let t_trace = *elapsed_ticks - *approach_ticks;
                if t_trace > *trace_ticks {
                    return StepStatus::Completed {
                        summary: format!(
                            "Traced {:.1} circular cycles (r={:.3}m) around [{:.2}, {:.2}, {:.2}]",
                            cycles, radius_m, center_xyz[0], center_xyz[1], center_xyz[2]
                        ),
                    };
                }
                let phase = min_jerk((t_trace as f64) / (*trace_ticks as f64)) * (*cycles) * TAU;
                let xyz = [
                    center_xyz[0] + *radius_m * phase.cos(),
                    center_xyz[1] + *radius_m * phase.sin(),
                    center_xyz[2],
                ];
                StepStatus::Running {
                    cmd: JointCommand {
                        timestamp_ns: state.timestamp_ns,
                        target_positions: state.joint_positions.clone(),
                        target_ee_xyz_m: Some(xyz),
                        gripper_open_ratio: *gripper,
                    },
                    heartbeat_fresh: true,
                }
            }
            Self::TraceWaypoints {
                waypoints,
                gripper,
                total_ticks,
                elapsed_ticks,
            } => {
                *elapsed_ticks += 1;
                if *elapsed_ticks > *total_ticks {
                    return StepStatus::Completed {
                        summary: format!("Completed {}-waypoint synthesized path", waypoints.len()),
                    };
                }
                let segs = (waypoints.len() - 1) as f64;
                let u = min_jerk((*elapsed_ticks as f64) / (*total_ticks as f64)) * segs;
                let idx = (u.floor() as usize).min(waypoints.len() - 2);
                let frac = u - (idx as f64);
                let xyz = lerp3(waypoints[idx], waypoints[idx + 1], frac);
                StepStatus::Running {
                    cmd: JointCommand {
                        timestamp_ns: state.timestamp_ns,
                        target_positions: state.joint_positions.clone(),
                        target_ee_xyz_m: Some(xyz),
                        gripper_open_ratio: *gripper,
                    },
                    heartbeat_fresh: true,
                }
            }
            Self::PolicyPlugin {
                worker,
                decimation,
                elapsed_ticks,
                max_ticks,
                last_cmd,
            } => {
                *elapsed_ticks += 1;
                if *elapsed_ticks > *max_ticks {
                    return StepStatus::Failed {
                        code: FailureCode::ExecutionTimeout,
                        message: "L3 policy worker exceeded skill timeout_sec".into(),
                    };
                }
                if (*elapsed_ticks - 1) % *decimation == 0 {
                    match worker.step(state) {
                        Ok(WorkerResponse::Command { command }) => {
                            *last_cmd = Some(command.clone());
                            StepStatus::Running {
                                cmd: command,
                                heartbeat_fresh: true,
                            }
                        }
                        Ok(WorkerResponse::Completed { summary }) => {
                            StepStatus::Completed { summary }
                        }
                        Ok(WorkerResponse::Failed { reason }) => StepStatus::Failed {
                            code: FailureCode::WorkerCrashed,
                            message: reason,
                        },
                        Ok(WorkerResponse::Ready) => StepStatus::Failed {
                            code: FailureCode::WorkerCrashed,
                            message: "unexpected Ready frame during step loop".into(),
                        },
                        Err(err) => {
                            let (code, message) = map_transport_err(err);
                            StepStatus::Failed { code, message }
                        }
                    }
                } else if let Some(cmd) = last_cmd {
                    StepStatus::Running {
                        cmd: cmd.clone(),
                        heartbeat_fresh: false,
                    }
                } else {
                    StepStatus::Running {
                        cmd: JointCommand {
                            timestamp_ns: state.timestamp_ns,
                            target_positions: state.joint_positions.clone(),
                            target_ee_xyz_m: Some(state.ee_xyz_m),
                            gripper_open_ratio: state.gripper_open_ratio,
                        },
                        heartbeat_fresh: false,
                    }
                }
            }
        }
    }
}

fn parse_vec3(args: &serde_json::Value, key: &str) -> Option<[f64; 3]> {
    let arr = args.get(key)?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

fn map_transport_err(err: TransportError) -> (FailureCode, String) {
    match err {
        TransportError::HeartbeatTimeout(ms) => (
            FailureCode::HeartbeatTimeout,
            format!("L3 policy worker missed {ms}ms heartbeat deadline"),
        ),
        other => (FailureCode::WorkerCrashed, other.to_string()),
    }
}
