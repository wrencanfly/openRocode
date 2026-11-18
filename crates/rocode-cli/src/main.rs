use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use rocode_adapters::MockRobotAdapter;
use rocode_core::{HarnessSupervisor, RuntimePhase};
use rocode_proto::{JointCommand, SafetyEnvelope, SkillSpec};

#[derive(Parser)]
#[command(
    name = "rocode",
    version,
    about = "openRocode: Terminal-native Robotics CLI & S2-S1-S0 Agent Harness"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate workspace environment and skill contracts
    Check {
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
    },
    /// List registered skill contracts (SKILL.toml)
    Skills {
        #[arg(long, default_value = "skills")]
        skills_dir: PathBuf,
    },
    /// Run the S2-S1-S0 harness loop against an embodiment adapter
    Run {
        #[arg(long, default_value = "mock")]
        adapter: String,
        #[arg(long, default_value_t = 20)]
        steps: u64,
        /// Optional step index at which to inject a synthetic effort spike fault
        #[arg(long)]
        fault_spike_at: Option<u64>,
    },
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
    match cli.command {
        Commands::Check { skills_dir } => {
            let skills = load_skills(&skills_dir);
            println!(
                "[rocode check] OK | safety_arbiter=ready | skills_loaded={} (from {})",
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
                    "- {:<16} kind={:?} rate={}Hz watchdog={}ms retries={} ({})",
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
            adapter,
            steps,
            fault_spike_at,
        } => {
            if adapter != "mock" {
                eprintln!("Unsupported adapter '{adapter}' in v0.1 bootstrap (supported: mock)");
                std::process::exit(2);
            }
            let mut robot = MockRobotAdapter::new(6, 2_000_000); // 500 Hz (2ms tick)
            if let Some(step) = fault_spike_at {
                robot = robot.with_effort_spike_at_step(step);
            }
            let mut supervisor =
                HarnessSupervisor::new(robot, SafetyEnvelope::default(), 3_000_000_000);

            println!(
                "[rocode run] adapter=mock-6dof rate=500Hz steps={steps} fault_spike_at={fault_spike_at:?}"
            );
            for step in 1..=steps {
                let cmd = JointCommand {
                    timestamp_ns: step * 2_000_000,
                    target_positions: vec![0.01 * step as f64; 6],
                    gripper_open_ratio: 1.0,
                };
                let state = supervisor.step_tick(Some("pick_object"), Some(&cmd), true);
                if supervisor.phase() == RuntimePhase::SafeHold {
                    println!(
                        "  [step {step:02}] SAFETY INTERCEPT -> SafeHold engaged at t={}ms (events_logged={})",
                        state.timestamp_ns / 1_000_000,
                        supervisor.recorder().events().len()
                    );
                    break;
                }
            }
            println!(
                "[rocode run] finished | final_phase={:?} | recorded_events={}",
                supervisor.phase(),
                supervisor.recorder().events().len()
            );
        }
    }
}
