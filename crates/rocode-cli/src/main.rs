mod tui;

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use rocode_adapters::MockRobotAdapter;
use rocode_core::{HarnessSupervisor, RuntimePhase};
use rocode_proto::{FailureTrace, SafetyEnvelope, SkillOutcome, SkillSpec};
use rocode_recorder::{FlightRecorder, RecordedEvent};
use rocode_vlm::{ApiLlmPlanner, PlanStep, SpatialDemoPlanner, System2Planner};

#[derive(Parser)]
#[command(
    name = "rocode",
    version,
    about = "openRocode: LLM-First Terminal Robotics CLI & S2-S1-S0 Execution Harness"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PlannerChoice {
    Auto,
    Api,
    Demo,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the interactive Ratatui robotics cockpit (default in TTY)
    Tui {
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = PlannerChoice::Auto)]
        planner: PlannerChoice,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "recordings/latest_tui.rlog")]
        record: PathBuf,
    },
    /// Validate workspace environment and registered skill contracts
    Check {
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
    },
    /// List registered L1/L2/L3 skill contracts (`SKILL.toml`)
    Skills {
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
    },
    /// Execute a natural-language task headlessly and write a `.rlog` flight recording
    Run {
        /// Natural-language task instruction for the LLM driver
        #[arg(default_value = "Put the red mug on the coaster and wipe the spill")]
        goal: String,
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = PlannerChoice::Auto)]
        planner: PlannerChoice,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "recordings/latest_run.rlog")]
        record: PathBuf,
        /// Optional S0 tick index to inject a 95 Nm effort spike fault
        #[arg(long)]
        fault_spike_at: Option<u64>,
        /// Optional S0 tick index to inject a grasp slip fault
        #[arg(long)]
        fault_slip_at: Option<u64>,
    },
    /// Time-travel inspect a recorded `.rlog` flight episode
    Replay {
        /// Path to `.rlog` flight log
        path: PathBuf,
        /// Inspect exact robot state & safety mode at target timestamp (ms)
        #[arg(long)]
        at_ms: Option<u64>,
        /// Print all captured blackbox failure traces
        #[arg(long)]
        failures: bool,
    },
}

fn build_planner(choice: PlannerChoice, model: Option<&str>) -> Arc<dyn System2Planner> {
    match choice {
        PlannerChoice::Demo => Arc::new(SpatialDemoPlanner),
        PlannerChoice::Api => match ApiLlmPlanner::from_env(model) {
            Ok(p) => Arc::new(p),
            Err(err) => {
                eprintln!("[rocode] error: {err}");
                std::process::exit(2);
            }
        },
        PlannerChoice::Auto => match ApiLlmPlanner::from_env(model) {
            Ok(p) => Arc::new(p),
            Err(_) => Arc::new(SpatialDemoPlanner),
        },
    }
}

fn load_skills(dir: &Path) -> Vec<(PathBuf, SkillSpec)> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let manifest = entry.path().join("SKILL.toml");
        if manifest.is_file()
            && let Ok(raw) = fs::read_to_string(&manifest)
            && let Ok(spec) = SkillSpec::from_toml_str(&raw)
        {
            found.push((manifest, spec));
        }
    }
    found.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    found
}

fn main() {
    let cli = Cli::parse();
    let workdir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let cmd = cli.command.unwrap_or_else(|| {
        if std::io::stdout().is_terminal() {
            Commands::Tui {
                skills_dir: PathBuf::from("skills"),
                planner: PlannerChoice::Auto,
                model: None,
                record: PathBuf::from("recordings/latest_tui.rlog"),
            }
        } else {
            Commands::Check {
                skills_dir: PathBuf::from("skills"),
            }
        }
    });

    match cmd {
        Commands::Tui {
            skills_dir,
            planner,
            model,
            record,
        } => {
            let p = build_planner(planner, model.as_deref());
            if let Err(err) = tui::run_tui(&workdir, &skills_dir, p, record) {
                eprintln!("[rocode tui] error: {err}");
                std::process::exit(1);
            }
        }
        Commands::Check { skills_dir } => {
            let skills = load_skills(&skills_dir);
            println!(
                "[rocode check] OK | s0_arbiter=ready (500Hz) | skills_loaded={} (from {})",
                skills.len(),
                skills_dir.display()
            );
        }
        Commands::Skills { skills_dir } => {
            let skills = load_skills(&skills_dir);
            if skills.is_empty() {
                println!("No skill contracts found in {}", skills_dir.display());
                return;
            }
            for (path, spec) in skills {
                println!(
                    "- {:<16} kind={:<22?} rate={:>3}Hz watchdog={:>2}ms retries={} ({})",
                    spec.name,
                    spec.kind,
                    spec.rate_hz,
                    spec.heartbeat_timeout_ms,
                    spec.max_retries,
                    path.display()
                );
            }
        }
        Commands::Run {
            goal,
            skills_dir,
            planner,
            model,
            record,
            fault_spike_at,
            fault_slip_at,
        } => {
            let p = build_planner(planner, model.as_deref());
            let mut robot = MockRobotAdapter::new(6, 2_000_000);
            if let Some(step) = fault_spike_at {
                robot = robot.with_effort_spike_at_step(step);
            }
            if let Some(step) = fault_slip_at {
                robot = robot.with_slip_at_step(step);
            }
            let mut supervisor =
                HarnessSupervisor::new(&workdir, robot, SafetyEnvelope::default(), 3_000_000_000, 5);
            supervisor.load_skills_from_dir(&skills_dir);
            supervisor.set_episode_goal(&goal);

            println!(
                "[rocode run] planner=\"{}\" | goal=\"{}\"",
                p.name(),
                goal
            );

            let mut history: Vec<String> = Vec::new();
            let mut last_failure: Option<FailureTrace> = None;

            for turn in 1..=16 {
                let skills_vec: Vec<SkillSpec> = supervisor.skills().values().cloned().collect();
                let step = match p.decide_next(
                    &goal,
                    &skills_vec,
                    supervisor.last_state(),
                    &history,
                    last_failure.as_ref(),
                ) {
                    Ok(s) => s,
                    Err(err) => {
                        eprintln!("  [turn {turn:02}] S2 error: {err}");
                        break;
                    }
                };

                match step {
                    PlanStep::Done { summary } => {
                        println!("  [turn {turn:02}] DONE -> {summary}");
                        break;
                    }
                    PlanStep::Invoke(call) => {
                        println!(
                            "  [turn {turn:02}] S2 -> {}({}) | {}",
                            call.skill_name, call.arguments, call.thought
                        );
                        match supervisor.execute_skill_blocking(call) {
                            SkillOutcome::Completed {
                                summary,
                                ticks_elapsed,
                                ..
                            } => {
                                println!(
                                    "            S1 OK ({ticks_elapsed} ticks) -> {summary}"
                                );
                                history.push(summary);
                                last_failure = None;
                            }
                            SkillOutcome::Failed(trace) => {
                                println!(
                                    "            S0/S1 INTERCEPT [{:?}] -> {} (trace_frames={})",
                                    trace.code,
                                    trace.message,
                                    trace.window_frames.len()
                                );
                                history.push(format!("FAILED {:?}: {}", trace.code, trace.message));
                                last_failure = Some(trace);
                                if supervisor.phase() == RuntimePhase::SafeHold {
                                    println!(
                                        "            [SAFE_HOLD] Halting autonomous execution for operator review."
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if let Err(err) = supervisor.recorder().save_to_path(&record) {
                eprintln!("[rocode run] warning: failed to save flight log: {err}");
            } else {
                let s = supervisor.recorder().summary();
                println!(
                    "[rocode run] flight log saved -> {} (duration={}ms, ticks={}, skills_ok={}, failures={})",
                    record.display(),
                    s.duration_ms,
                    s.total_ticks,
                    s.skills_completed,
                    s.failures_captured
                );
            }
        }
        Commands::Replay {
            path,
            at_ms,
            failures,
        } => {
            let rec = match FlightRecorder::load_from_path(&path) {
                Ok(r) => r,
                Err(err) => {
                    eprintln!("[rocode replay] failed to open {}: {err}", path.display());
                    std::process::exit(1);
                }
            };
            let s = rec.summary();
            println!("=== Episode Replay: {} ===", path.display());
            println!(
                "Adapter: {} | Goal: \"{}\"",
                s.adapter, s.goal
            );
            println!(
                "Duration: {} ms | Total Ticks: {} | LLM Turns: {} | Skills OK: {} | Failures: {} | SafeHold Ticks: {}",
                s.duration_ms,
                s.total_ticks,
                s.llm_decisions,
                s.skills_completed,
                s.failures_captured,
                s.safe_hold_ticks
            );

            println!("\n--- High-Level Timeline ---");
            for ev in rec.events() {
                match ev {
                    RecordedEvent::LlmDecision { timestamp_ns, call } => {
                        println!(
                            "[{:>6} ms] S2 CALL  {}({}) — {}",
                            timestamp_ns / 1_000_000,
                            call.skill_name,
                            call.arguments,
                            call.thought
                        );
                    }
                    RecordedEvent::SkillCompleted {
                        timestamp_ns,
                        skill_name,
                        summary,
                        ..
                    } => {
                        println!(
                            "[{:>6} ms] S1 OK    {skill_name} -> {summary}",
                            timestamp_ns / 1_000_000
                        );
                    }
                    RecordedEvent::FailureCaptured {
                        timestamp_ns,
                        trace,
                    } => {
                        println!(
                            "[{:>6} ms] FAILURE  {} [{:?}] -> {} (window_frames={})",
                            timestamp_ns / 1_000_000,
                            trace.skill_name,
                            trace.code,
                            trace.message,
                            trace.window_frames.len()
                        );
                    }
                    RecordedEvent::OperatorNote {
                        timestamp_ns,
                        action,
                    } => {
                        println!(
                            "[{:>6} ms] OPERATOR {action}",
                            timestamp_ns / 1_000_000
                        );
                    }
                    _ => {}
                }
            }

            if let Some(target_ms) = at_ms {
                println!("\n--- Time-Travel Snapshot @ {target_ms} ms ---");
                if let Some((tick, mode, st)) = rec.tick_at_ms(target_ms) {
                    println!(
                        "Tick #{tick} (t={} ms) | Mode: {mode:?} | Held: {:?}",
                        st.timestamp_ns / 1_000_000,
                        st.held_object_id
                    );
                    println!(
                        "EE XYZ: [{:+.4}, {:+.4}, {:+.4}] m | Contact Force: {:.2} N | Gripper: {:.2}",
                        st.ee_xyz_m[0], st.ee_xyz_m[1], st.ee_xyz_m[2], st.ee_contact_force_n, st.gripper_open_ratio
                    );
                    println!("Joints (rad): {:?}", st.joint_positions);
                    println!("Efforts (Nm): {:?}", st.joint_efforts);
                } else {
                    println!("No tick recorded at or before {target_ms} ms.");
                }
            }

            if failures {
                println!("\n--- Blackbox Failure Traces ---");
                for (idx, f) in rec.failures().iter().enumerate() {
                    println!(
                        "#{idx} skill={} code={:?} msg=\"{}\" final_ee={:?} frames_captured={}",
                        f.skill_name,
                        f.code,
                        f.message,
                        f.final_state.ee_xyz_m,
                        f.window_frames.len()
                    );
                }
            }
        }
    }
}
