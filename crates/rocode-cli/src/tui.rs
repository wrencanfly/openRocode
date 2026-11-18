//! Ratatui Interactive Terminal Cockpit for openRocode.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
};
use rocode_adapters::{EmbodimentAdapter, MockRobotAdapter};
use rocode_core::{HarnessSupervisor, RuntimePhase};
use rocode_proto::{FailureTrace, JointCommand, RobotState, SafetyEnvelope, SkillOutcome, SkillSpec};
use rocode_safety::ArbiterMode;
use rocode_vlm::{PlanStep, System2Planner};

#[derive(Clone)]
struct LogEntry {
    tag: &'static str,
    color: Color,
    text: String,
}

pub fn run_tui(
    workdir: &Path,
    skills_dir: &Path,
    planner: Arc<dyn System2Planner>,
    record_out: PathBuf,
) -> io::Result<()> {
    let robot = MockRobotAdapter::new(6, 2_000_000); // 500 Hz (2ms tick)
    let mut supervisor =
        HarnessSupervisor::new(workdir, robot, SafetyEnvelope::default(), 3_000_000_000, 10);
    let loaded = supervisor.load_skills_from_dir(skills_dir);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = tui_loop(&mut terminal, &mut supervisor, planner, loaded, &record_out);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    supervisor: &mut HarnessSupervisor<MockRobotAdapter>,
    planner: Arc<dyn System2Planner>,
    loaded_skills: usize,
    record_out: &Path,
) -> io::Result<()> {
    let mut logs: Vec<LogEntry> = vec![
        LogEntry {
            tag: "SYS",
            color: Color::Cyan,
            text: format!(
                "openRocode v0.1 ready | S0=500Hz | skills={loaded_skills} | planner={}",
                planner.name()
            ),
        },
        LogEntry {
            tag: "TIP",
            color: Color::DarkGray,
            text: "Type a task (e.g. 'put red mug on coaster and wipe spill' or 'probe table contact') or /takeover /hold /inject spike /estop /save /quit".into(),
        },
    ];

    let mut input = String::new();
    let mut active_goal: Option<String> = None;
    let mut history: Vec<String> = Vec::new();
    let mut last_failure: Option<FailureTrace> = None;
    let mut s2_thinking = false;
    let mut plan_step_count = 0usize;

    let (plan_tx, plan_rx) = mpsc::channel::<Result<PlanStep, String>>();

    let mut teleop_target_ee = supervisor.last_state().ee_xyz_m;
    let mut teleop_gripper = supervisor.last_state().gripper_open_ratio;
    let mut pending_teleop_step = false;

    let frame_budget = Duration::from_millis(16); // ~60 FPS UI, 8 x 2ms S0 ticks per frame

    loop {
        let frame_start = Instant::now();

        // 1. Poll keyboard events without blocking the 500Hz S0 loop
        while event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    let _ = supervisor.recorder().save_to_path(record_out);
                    return Ok(());
                }

                if supervisor.phase() == RuntimePhase::TeleopTakeover {
                    let step = 0.012;
                    match key.code {
                        KeyCode::Esc => {
                            supervisor.operator_release_teleop();
                            logs.push(LogEntry {
                                tag: "OP",
                                color: Color::Magenta,
                                text: "Released teleop takeover -> returned to IDLE holding pose.".into(),
                            });
                        }
                        KeyCode::Char('w') | KeyCode::Char('W') => {
                            teleop_target_ee[0] = (teleop_target_ee[0] + step).clamp(0.14, 0.58);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char('s') | KeyCode::Char('S') => {
                            teleop_target_ee[0] = (teleop_target_ee[0] - step).clamp(0.14, 0.58);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char('a') | KeyCode::Char('A') => {
                            teleop_target_ee[1] = (teleop_target_ee[1] + step).clamp(-0.40, 0.40);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char('d') | KeyCode::Char('D') => {
                            teleop_target_ee[1] = (teleop_target_ee[1] - step).clamp(-0.40, 0.40);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char('q') | KeyCode::Char('Q') => {
                            teleop_target_ee[2] = (teleop_target_ee[2] + step).clamp(0.0, 0.48);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char('e') | KeyCode::Char('E') => {
                            teleop_target_ee[2] = (teleop_target_ee[2] - step).clamp(0.0, 0.48);
                            pending_teleop_step = true;
                        }
                        KeyCode::Char(' ') => {
                            teleop_gripper = if teleop_gripper > 0.5 { 0.0 } else { 1.0 };
                            pending_teleop_step = true;
                        }
                        _ => {}
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    KeyCode::Char(c) => {
                        input.push(c);
                    }
                    KeyCode::Enter => {
                        let line = input.trim().to_string();
                        input.clear();
                        if line.is_empty() {
                            continue;
                        }
                        if line.starts_with('/') {
                            if handle_slash_command(
                                &line,
                                supervisor,
                                &mut logs,
                                &mut active_goal,
                                &mut teleop_target_ee,
                                &mut teleop_gripper,
                                record_out,
                            ) {
                                return Ok(());
                            }
                        } else {
                            logs.push(LogEntry {
                                tag: "USER",
                                color: Color::White,
                                text: line.clone(),
                            });
                            supervisor.set_episode_goal(&line);
                            history.clear();
                            last_failure = None;
                            plan_step_count = 0;
                            active_goal = Some(line);
                        }
                    }
                    _ => {}
                }
            }
        }

        // 2. Check if background S2 LLM planner returned a decision
        if s2_thinking && let Ok(plan_res) = plan_rx.try_recv() {
            s2_thinking = false;
            match plan_res {
                Ok(PlanStep::Done { summary }) => {
                    logs.push(LogEntry {
                        tag: "DONE",
                        color: Color::Green,
                        text: summary,
                    });
                    active_goal = None;
                }
                Ok(PlanStep::Invoke(call)) => {
                    plan_step_count += 1;
                    logs.push(LogEntry {
                        tag: "S2",
                        color: Color::LightBlue,
                        text: format!(
                            "#{} {}({}) — {}",
                            plan_step_count, call.skill_name, call.arguments, call.thought
                        ),
                    });
                    if let Err(trace) = supervisor.start_skill(call) {
                        logs.push(LogEntry {
                            tag: "REFUSED",
                            color: Color::Yellow,
                            text: format!("{:?}: {}", trace.code, trace.message),
                        });
                        last_failure = Some(trace);
                    }
                }
                Err(err) => {
                    logs.push(LogEntry {
                        tag: "ERR",
                        color: Color::Red,
                        text: format!("S2 planner error: {err}"),
                    });
                    active_goal = None;
                }
            }
        }

        // 3. Spawn background S2 decision when goal is active and robot is Idle
        if let Some(ref goal) = active_goal
            && !s2_thinking
            && supervisor.phase() == RuntimePhase::Idle
        {
            if plan_step_count >= 16 {
                logs.push(LogEntry {
                    tag: "WARN",
                    color: Color::Yellow,
                    text: "Reached 16-step safety cap for single goal; stopping.".into(),
                });
                active_goal = None;
            } else {
                s2_thinking = true;
                let tx = plan_tx.clone();
                let planner_clone = Arc::clone(&planner);
                let goal_clone = goal.clone();
                let skills_vec: Vec<SkillSpec> = supervisor.skills().values().cloned().collect();
                let state_snap = supervisor.last_state().clone();
                let hist_snap = history.clone();
                let fail_snap = last_failure.clone();
                thread::spawn(move || {
                    let res = planner_clone.decide_next(
                        &goal_clone,
                        &skills_vec,
                        &state_snap,
                        &hist_snap,
                        fail_snap.as_ref(),
                    );
                    let _ = tx.send(res);
                });
            }
        }

        // 4. Advance 8 deterministic 500Hz S0/S1 ticks (16ms simulated & wall-clock slice)
        for _ in 0..8 {
            let teleop_cmd = if supervisor.phase() == RuntimePhase::TeleopTakeover && pending_teleop_step {
                Some(JointCommand {
                    timestamp_ns: supervisor.last_state().timestamp_ns,
                    target_positions: supervisor.last_state().joint_positions.clone(),
                    target_ee_xyz_m: Some(teleop_target_ee),
                    gripper_open_ratio: teleop_gripper,
                })
            } else {
                None
            };

            if let Some(outcome) = supervisor.tick_once(teleop_cmd.as_ref()) {
                match outcome {
                    SkillOutcome::Completed { summary, .. } => {
                        logs.push(LogEntry {
                            tag: "S1 OK",
                            color: Color::Green,
                            text: summary.clone(),
                        });
                        history.push(summary);
                        last_failure = None;
                    }
                    SkillOutcome::Failed(trace) => {
                        logs.push(LogEntry {
                            tag: "S0/S1",
                            color: Color::Red,
                            text: format!(
                                "[{:?}] {} (frames_in_trace={})",
                                trace.code,
                                trace.message,
                                trace.window_frames.len()
                            ),
                        });
                        history.push(format!("FAILED {:?}: {}", trace.code, trace.message));
                        if supervisor.phase() == RuntimePhase::SafeHold {
                            active_goal = None;
                            logs.push(LogEntry {
                                tag: "HOLD",
                                color: Color::Yellow,
                                text: "Harness locked in SAFE_HOLD. Use /resume or /takeover after inspecting workspace.".into(),
                            });
                        }
                        last_failure = Some(trace);
                    }
                }
            }
        }
        if supervisor.phase() == RuntimePhase::TeleopTakeover
            && dist3(supervisor.last_state().ee_xyz_m, teleop_target_ee) < 0.002
        {
            pending_teleop_step = false;
        }

        // 5. Render TUI cockpit
        terminal.draw(|f| {
            draw_cockpit(
                f,
                supervisor,
                &logs,
                &input,
                s2_thinking,
                active_goal.as_deref(),
            );
        })?;

        let elapsed = frame_start.elapsed();
        if elapsed < frame_budget {
            thread::sleep(frame_budget - elapsed);
        }
    }
}

fn handle_slash_command(
    cmd: &str,
    supervisor: &mut HarnessSupervisor<MockRobotAdapter>,
    logs: &mut Vec<LogEntry>,
    active_goal: &mut Option<String>,
    teleop_target_ee: &mut [f64; 3],
    teleop_gripper: &mut f64,
    record_out: &Path,
) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts[0] {
        "/quit" | "/exit" => {
            let _ = supervisor.recorder().save_to_path(record_out);
            return true;
        }
        "/skills" => {
            let mut names: Vec<_> = supervisor.skills().values().collect();
            names.sort_by(|a, b| a.name.cmp(&b.name));
            for s in names {
                logs.push(LogEntry {
                    tag: "SKILL",
                    color: Color::Cyan,
                    text: format!("{} [{:?}] rate={}Hz — {}", s.name, s.kind, s.rate_hz, s.description),
                });
            }
        }
        "/hold" => {
            *active_goal = None;
            let _ = supervisor.operator_hold();
            logs.push(LogEntry {
                tag: "OP",
                color: Color::Yellow,
                text: "Operator engaged SAFE_HOLD.".into(),
            });
        }
        "/resume" => {
            supervisor.operator_resume_idle();
            logs.push(LogEntry {
                tag: "OP",
                color: Color::Cyan,
                text: "Cleared SAFE_HOLD -> IDLE.".into(),
            });
        }
        "/takeover" => {
            *active_goal = None;
            *teleop_target_ee = supervisor.last_state().ee_xyz_m;
            *teleop_gripper = supervisor.last_state().gripper_open_ratio;
            let _ = supervisor.operator_engage_teleop();
            logs.push(LogEntry {
                tag: "OP",
                color: Color::Magenta,
                text: "TELEOP TAKEOVER active: use W/S (±X), A/D (±Y), Q/E (±Z), Space (Gripper), Esc (Release).".into(),
            });
        }
        "/estop" => {
            *active_goal = None;
            let _ = supervisor.operator_estop();
            logs.push(LogEntry {
                tag: "ESTOP",
                color: Color::Red,
                text: "EMERGENCY STOP engaged — motors de-energized! Use /reset to clear.".into(),
            });
        }
        "/reset" => {
            *active_goal = None;
            supervisor.operator_reset_world();
            logs.push(LogEntry {
                tag: "OP",
                color: Color::Cyan,
                text: "Workspace & safety arbiter reset to initial tabletop state.".into(),
            });
        }
        "/inject" => {
            let kind = parts.get(1).copied().unwrap_or("spike");
            if kind == "slip" {
                supervisor.adapter_mut().inject_grasp_slip();
                logs.push(LogEntry {
                    tag: "FAULT",
                    color: Color::Yellow,
                    text: "Injected synthetic GRASP SLIP fault!".into(),
                });
            } else {
                supervisor.adapter_mut().inject_effort_spike();
                logs.push(LogEntry {
                    tag: "FAULT",
                    color: Color::Red,
                    text: "Injected 95 Nm EFFORT SPIKE fault -> watch S0 intercept!".into(),
                });
            }
        }
        "/save" => {
            let path = parts
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| record_out.to_path_buf());
            match supervisor.recorder().save_to_path(&path) {
                Ok(()) => logs.push(LogEntry {
                    tag: "REC",
                    color: Color::Green,
                    text: format!(
                        "Saved flight recording ({} events) to {}",
                        supervisor.recorder().events().len(),
                        path.display()
                    ),
                }),
                Err(e) => logs.push(LogEntry {
                    tag: "ERR",
                    color: Color::Red,
                    text: format!("Failed to save recording: {e}"),
                }),
            }
        }
        other => logs.push(LogEntry {
            tag: "WARN",
            color: Color::Yellow,
            text: format!(
                "Unknown command '{other}'. Available: /skills /hold /resume /takeover /estop /inject [spike|slip] /reset /save /quit"
            ),
        }),
    }
    false
}

fn draw_cockpit(
    f: &mut ratatui::Frame,
    supervisor: &HarnessSupervisor<MockRobotAdapter>,
    logs: &[LogEntry],
    input: &str,
    s2_thinking: bool,
    active_goal: Option<&str>,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(f.area());

    let mode = supervisor.arbiter().mode();
    let (mode_label, mode_color) = match mode {
        ArbiterMode::Idle => (" IDLE ", Color::Cyan),
        ArbiterMode::Autonomous => (" EXECUTING ", Color::Green),
        ArbiterMode::SafeHold => (" SAFE_HOLD ", Color::Yellow),
        ArbiterMode::TeleopTakeover => (" TELEOP_TAKEOVER ", Color::Magenta),
        ArbiterMode::EStop => (" ESTOP ", Color::Red),
    };

    let state = supervisor.last_state();
    let active_skill = supervisor.active_skill_name().unwrap_or("-");
    let s2_badge = if s2_thinking { "THINKING..." } else { "READY" };
    let header_line = Line::from(vec![
        Span::styled(
            mode_label,
            Style::default()
                .fg(Color::Black)
                .bg(mode_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  S0: 500Hz  |  S1 Skill: {active_skill}  |  S2: {s2_badge}  |  Held: {}  |  Events: {}",
            state.held_object_id.as_deref().unwrap_or("none"),
            supervisor.recorder().events().len()
        )),
    ]);
    let header = Paragraph::new(header_line).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" openRocode — LLM-First Robotics Harness Cockpit "),
    );
    f.render_widget(header, chunks[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3)])
        .split(body[0]);

    // Left top: Agent & Execution Stream
    let visible_height = left[0].height.saturating_sub(2) as usize;
    let start_idx = logs.len().saturating_sub(visible_height.max(1));
    let items: Vec<ListItem> = logs[start_idx..]
        .iter()
        .map(|entry| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("[{:^7}] ", entry.tag),
                    Style::default().fg(entry.color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(&entry.text),
            ]))
        })
        .collect();
    let log_title = match active_goal {
        Some(g) => format!(" Execution Timeline — Active Goal: \"{g}\" "),
        None => " Execution Timeline ".to_string(),
    };
    f.render_widget(
        List::new(items).block(Block::default().borders(Borders::ALL).title(log_title)),
        left[0],
    );

    // Left bottom: Prompt / Teleop Controls
    let prompt_widget = if supervisor.phase() == RuntimePhase::TeleopTakeover {
        Paragraph::new(" [TELEOP] W/S: ±X  |  A/D: ±Y  |  Q/E: ±Z  |  Space: Gripper  |  Esc: Release")
            .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL).title(" Operator Teleop Override "))
    } else {
        Paragraph::new(format!("> {input}█"))
            .block(Block::default().borders(Borders::ALL).title(" Goal / Slash Command "))
    };
    f.render_widget(prompt_widget, left[1]);

    // Right column: Telemetry + 2D Tabletop Radar
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(8)])
        .split(body[1]);

    let max_eff = supervisor.arbiter().envelope().max_effort;
    let mut telem_lines = vec![
        Line::from(format!(
            "EE Pose (m) : [{:+.3}, {:+.3}, {:+.3}]",
            state.ee_xyz_m[0], state.ee_xyz_m[1], state.ee_xyz_m[2]
        )),
        Line::from(format!(
            "Contact (N) : {:5.1} N   Gripper: {:3.0}% ({})",
            state.ee_contact_force_n,
            state.gripper_open_ratio * 100.0,
            if state.gripper_contact_detected {
                "CONTACT"
            } else {
                "open/empty"
            }
        )),
        Line::from("─".repeat(38)),
    ];
    for (idx, (&q, &eff)) in state
        .joint_positions
        .iter()
        .zip(state.joint_efforts.iter())
        .enumerate()
    {
        let ratio = (eff.abs() / max_eff).clamp(0.0, 1.0);
        let filled = (ratio * 10.0).round() as usize;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled));
        let bar_color = if ratio > 0.85 {
            Color::Red
        } else if ratio > 0.5 {
            Color::Yellow
        } else {
            Color::Green
        };
        telem_lines.push(Line::from(vec![
            Span::raw(format!("J{idx} {:+5.2}r [", q)),
            Span::styled(bar, Style::default().fg(bar_color)),
            Span::raw(format!("] {:4.1}/{:.0}Nm", eff.abs(), max_eff)),
        ]));
    }
    f.render_widget(
        Paragraph::new(telem_lines)
            .block(Block::default().borders(Borders::ALL).title(" S0 Telemetry & Effort Envelope "))
            .wrap(Wrap { trim: false }),
        right[0],
    );

    let radar_text = render_workspace_radar(state, right[1].width.saturating_sub(4) as usize, right[1].height.saturating_sub(3) as usize);
    f.render_widget(
        Paragraph::new(radar_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Top-Down Workspace (◉Base ✦EE M=Mug C=Coaster ~=Spill) "),
        ),
        right[1],
    );
}

fn render_workspace_radar(state: &RobotState, width: usize, height: usize) -> Vec<Line<'static>> {
    let w = width.clamp(16, 48);
    let h = height.clamp(6, 18);
    let mut grid = vec![vec!['·'; w]; h];

    let map_xy = |x: f64, y: f64| -> (usize, usize) {
        // x in [0.0, 0.60] -> row (top is far +x, bottom is base x=0)
        // y in [0.40, -0.40] -> col (left is +y, right is -y)
        let rx = (1.0 - (x / 0.60).clamp(0.0, 1.0)) * ((h - 1) as f64);
        let cy = ((0.40 - y) / 0.80).clamp(0.0, 1.0) * ((w - 1) as f64);
        (rx.round() as usize, cy.round() as usize)
    };

    let (br, bc) = map_xy(0.0, 0.0);
    grid[br][bc] = '◉';

    for obj in &state.scene_objects {
        let (r, c) = map_xy(obj.xyz_m[0], obj.xyz_m[1]);
        grid[r][c] = match obj.id.as_str() {
            "red_mug" => 'M',
            "coaster" => 'C',
            "spill_patch" => {
                if obj.label.contains("wiped") {
                    '✓'
                } else {
                    '~'
                }
            }
            _ => '□',
        };
    }

    let (er, ec) = map_xy(state.ee_xyz_m[0], state.ee_xyz_m[1]);
    grid[er][ec] = '✦';

    grid.into_iter()
        .map(|row| Line::from(row.into_iter().collect::<String>()))
        .collect()
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
