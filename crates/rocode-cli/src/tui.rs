//! Terminal cockpit for openRocode.
//!
//! The screen has three parts: a transcript of what you asked and what the arm did,
//! a two-view drawing of the arm and the table, and a prompt. All state lives in [`App`]
//! so the whole thing can be driven and rendered off-screen in tests.

use std::cell::Cell;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Paragraph,
        canvas::{Canvas, Circle, Context, Line as Stroke, Rectangle},
    },
};
use rocode_adapters::{EmbodimentAdapter, MockRobotAdapter};
use rocode_core::{HarnessSupervisor, RuntimePhase};
use rocode_proto::{
    FailureCode, FailureTrace, JointCommand, RobotState, SafetyEnvelope, SceneObject, SkillCall,
    SkillKind, SkillOutcome, SkillSpec,
};
use rocode_safety::ArbiterMode;
use rocode_vlm::{PlanStep, System2Planner};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const DT_NS: u64 = 2_000_000; // 500 Hz control tick
const TICKS_PER_FRAME: usize = 8; // 8 x 2 ms = one 16 ms UI frame
const MAX_STEPS_PER_GOAL: usize = 16;
const ACCENT: Color = Color::Cyan;
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const EXAMPLE_GOALS: [&str; 3] = [
    "put the red mug on the coaster and wipe the spill",
    "probe the table until contact",
    "grab the mug with the plugin policy",
];

/// Slash commands: (what you type, what it does, whether it takes an argument).
const COMMANDS: [(&str, &str, bool); 12] = [
    ("/takeover", "drive the arm by hand", false),
    ("/hold", "pause and hold position", false),
    ("/resume", "clear a hold", false),
    ("/estop", "cut motor power", false),
    ("/reset", "put the arm and the table back to the start", false),
    ("/inject slip", "make the gripper drop what it holds", false),
    ("/inject spike", "fake a 95 N·m torque spike", false),
    ("/skills", "list the motions the planner can call", false),
    ("/save", "save the flight recording (optional path)", true),
    ("/clear", "clear the transcript", false),
    ("/help", "show this list", false),
    ("/quit", "save the recording and exit", false),
];

#[derive(Clone, Copy, PartialEq)]
enum Tone {
    Info,
    Warn,
    Danger,
}

struct StepResult {
    ok: bool,
    /// Short lead-in: a duration on success, the kind of failure otherwise.
    label: String,
    text: String,
}

enum Entry {
    Goal(String),
    Step {
        n: usize,
        skill: String,
        what: String,
        thought: String,
        started_ns: u64,
        result: Option<StepResult>,
    },
    Note {
        tone: Tone,
        text: String,
    },
    Table(Vec<(String, String)>),
    Done(String),
}

type PlanMessage = (u64, Result<PlanStep, String>);

pub struct App {
    sup: HarnessSupervisor<MockRobotAdapter>,
    planner: Arc<dyn System2Planner>,
    offline: bool,
    record_out: PathBuf,

    entries: Vec<Entry>,
    input: String,
    past_inputs: Vec<String>,
    past_idx: Option<usize>,
    example_idx: usize,
    scroll_up: usize,
    max_scroll: Cell<usize>,

    goal: Option<String>,
    history: Vec<String>,
    last_failure: Option<FailureTrace>,
    thinking: bool,
    plan_gen: u64,
    step_count: usize,
    plan_tx: mpsc::Sender<PlanMessage>,
    plan_rx: mpsc::Receiver<PlanMessage>,

    teleop_target: [f64; 3],
    teleop_gripper: f64,
    teleop_pending: bool,

    frame: u64,
    quit: bool,
}

pub fn run_tui(
    workdir: &Path,
    skills_dir: &Path,
    planner: Arc<dyn System2Planner>,
    offline: bool,
    record_out: PathBuf,
) -> io::Result<()> {
    let mut app = App::new(workdir, skills_dir, planner, offline, record_out);

    let mut terminal = ratatui::init();
    let frame_budget = Duration::from_millis(16);
    let res = (|| -> io::Result<()> {
        loop {
            let frame_start = Instant::now();
            while event::poll(Duration::ZERO)? {
                if let Event::Key(key) = event::read()?
                    && key.kind == KeyEventKind::Press
                {
                    app.on_key(key);
                }
            }
            if app.quit {
                return Ok(());
            }
            app.advance();
            terminal.draw(|f| app.draw(f))?;
            if let Some(rest) = frame_budget.checked_sub(frame_start.elapsed()) {
                thread::sleep(rest);
            }
        }
    })();
    ratatui::restore();

    match app.sup.recorder().save_to_path(&app.record_out) {
        Ok(()) => println!("Flight recording saved to {}", app.record_out.display()),
        Err(err) => eprintln!("Could not save the flight recording: {err}"),
    }
    res
}

impl App {
    pub fn new(
        workdir: &Path,
        skills_dir: &Path,
        planner: Arc<dyn System2Planner>,
        offline: bool,
        record_out: PathBuf,
    ) -> Self {
        let robot = MockRobotAdapter::new(6, DT_NS);
        let mut sup =
            HarnessSupervisor::new(workdir, robot, SafetyEnvelope::default(), 3_000_000_000, 10);
        let loaded = sup.load_skills_from_dir(skills_dir);
        let (plan_tx, plan_rx) = mpsc::channel();
        let teleop_target = sup.last_state().ee_xyz_m;
        let teleop_gripper = sup.last_state().gripper_open_ratio;

        let mut app = Self {
            sup,
            planner,
            offline,
            record_out,
            entries: Vec::new(),
            input: String::new(),
            past_inputs: Vec::new(),
            past_idx: None,
            example_idx: 0,
            scroll_up: 0,
            max_scroll: Cell::new(0),
            goal: None,
            history: Vec::new(),
            last_failure: None,
            thinking: false,
            plan_gen: 0,
            step_count: 0,
            plan_tx,
            plan_rx,
            teleop_target,
            teleop_gripper,
            teleop_pending: false,
            frame: 0,
            quit: false,
        };
        if loaded == 0 {
            app.note(
                Tone::Warn,
                format!(
                    "No skills found in {}. The planner has nothing to call; run from the repo root or pass --skills-dir.",
                    skills_dir.display()
                ),
            );
        }
        app
    }

    // ---- input ---------------------------------------------------------------------------

    pub fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && key.code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        if self.sup.phase() == RuntimePhase::TeleopTakeover {
            self.on_teleop_key(key.code);
            return;
        }
        match key.code {
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.input.pop();
                self.past_idx = None;
            }
            KeyCode::Esc => {
                self.input.clear();
                self.past_idx = None;
            }
            KeyCode::Tab => self.complete(),
            KeyCode::Up => self.recall(-1),
            KeyCode::Down => self.recall(1),
            KeyCode::PageUp => {
                self.scroll_up = (self.scroll_up + 8).min(self.max_scroll.get());
            }
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(8),
            KeyCode::Char('u') if ctrl => self.input.clear(),
            KeyCode::Char(c) if !ctrl => {
                self.input.push(c);
                self.past_idx = None;
            }
            _ => {}
        }
    }

    fn on_teleop_key(&mut self, code: KeyCode) {
        const STEP: f64 = 0.012;
        let t = &mut self.teleop_target;
        match code {
            KeyCode::Esc => {
                self.sup.operator_release_teleop();
                self.note(Tone::Info, "Control handed back. The arm is holding its pose.");
                return;
            }
            KeyCode::Char('w' | 'W') => t[0] = (t[0] + STEP).clamp(0.14, 0.58),
            KeyCode::Char('s' | 'S') => t[0] = (t[0] - STEP).clamp(0.14, 0.58),
            KeyCode::Char('a' | 'A') => t[1] = (t[1] + STEP).clamp(-0.40, 0.40),
            KeyCode::Char('d' | 'D') => t[1] = (t[1] - STEP).clamp(-0.40, 0.40),
            KeyCode::Char('q' | 'Q') => t[2] = (t[2] + STEP).clamp(0.0, 0.48),
            KeyCode::Char('e' | 'E') => t[2] = (t[2] - STEP).clamp(0.0, 0.48),
            KeyCode::Char(' ') => {
                self.teleop_gripper = if self.teleop_gripper > 0.5 { 0.0 } else { 1.0 };
            }
            _ => return,
        }
        self.teleop_pending = true;
    }

    fn recall(&mut self, dir: i32) {
        if self.past_inputs.is_empty() {
            return;
        }
        let last = self.past_inputs.len() - 1;
        let next = match (self.past_idx, dir) {
            (None, -1) => Some(last),
            (Some(i), -1) => Some(i.saturating_sub(1)),
            (Some(i), 1) if i < last => Some(i + 1),
            _ => None,
        };
        self.past_idx = next;
        self.input = next.map(|i| self.past_inputs[i].clone()).unwrap_or_default();
    }

    /// Tab: finish a slash command, or cycle through the example goals.
    fn complete(&mut self) {
        if self.input.starts_with('/') {
            if let Some((cmd, _, takes_arg)) = self.matching_commands().first() {
                self.input = if *takes_arg { format!("{cmd} ") } else { (*cmd).to_string() };
            }
        } else if self.input.is_empty() || EXAMPLE_GOALS.contains(&self.input.as_str()) {
            self.input = EXAMPLE_GOALS[self.example_idx % EXAMPLE_GOALS.len()].to_string();
            self.example_idx += 1;
        }
    }

    fn matching_commands(&self) -> Vec<(&'static str, &'static str, bool)> {
        if !self.input.starts_with('/') || self.sup.phase() == RuntimePhase::TeleopTakeover {
            return Vec::new();
        }
        let typed = self.input.trim_end();
        COMMANDS
            .iter()
            .filter(|(cmd, _, _)| cmd.starts_with(typed))
            .copied()
            .collect()
    }

    fn submit(&mut self) {
        let line = self.input.trim().to_string();
        self.input.clear();
        self.past_idx = None;
        if line.is_empty() {
            return;
        }
        if self.past_inputs.last() != Some(&line) {
            self.past_inputs.push(line.clone());
        }
        self.scroll_up = 0;
        if line.starts_with('/') {
            self.run_command(&line);
        } else {
            self.start_goal(line);
        }
    }

    fn start_goal(&mut self, goal: String) {
        match self.sup.phase() {
            RuntimePhase::SafeHold => {
                self.note(
                    Tone::Warn,
                    "The arm is in a safety hold. /resume before giving it a new goal.",
                );
                return;
            }
            RuntimePhase::EStop => {
                self.note(
                    Tone::Warn,
                    "Motor power is cut. /reset before giving it a new goal.",
                );
                return;
            }
            _ => {}
        }
        self.cancel_goal();
        self.entries.push(Entry::Goal(goal.clone()));
        self.sup.set_episode_goal(&goal);
        self.history.clear();
        self.last_failure = None;
        self.step_count = 0;
        self.goal = Some(goal);
    }

    /// Drop the active goal and ignore any planner answer still in flight for it.
    fn cancel_goal(&mut self) {
        self.goal = None;
        self.plan_gen += 1;
        self.thinking = false;
    }

    fn run_command(&mut self, line: &str) {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or_default();
        let arg = parts.next();
        match cmd {
            "/quit" | "/exit" => self.quit = true,
            "/help" => {
                let rows = COMMANDS
                    .iter()
                    .map(|(c, what, _)| ((*c).to_string(), (*what).to_string()))
                    .collect();
                self.entries.push(Entry::Table(rows));
            }
            "/skills" => {
                let mut specs: Vec<&SkillSpec> = self.sup.skills().values().collect();
                specs.sort_by(|a, b| a.name.cmp(&b.name));
                let rows = specs
                    .into_iter()
                    .map(|s| {
                        let kind = match s.kind {
                            SkillKind::DirectPrimitive => "built-in motion",
                            SkillKind::SynthesizedTrajectory => "generated path",
                            SkillKind::PolicyPlugin => "plugin policy",
                        };
                        (s.name.clone(), format!("{kind}, {} Hz. {}", s.rate_hz, s.description))
                    })
                    .collect();
                self.entries.push(Entry::Table(rows));
            }
            "/clear" => self.entries.clear(),
            "/hold" => {
                self.cancel_goal();
                let outcome = self.sup.operator_hold();
                self.absorb_interrupt(outcome);
                self.note(Tone::Warn, "Holding position. /resume to continue.");
            }
            "/resume" => match self.sup.phase() {
                RuntimePhase::SafeHold => {
                    self.sup.operator_resume_idle();
                    self.note(Tone::Info, "Hold cleared. Ready for a goal.");
                }
                RuntimePhase::EStop => {
                    self.note(Tone::Warn, "/resume cannot clear an emergency stop. Use /reset.");
                }
                _ => self.note(Tone::Info, "Nothing to resume; the arm is not on hold."),
            },
            "/takeover" => {
                if self.sup.phase() == RuntimePhase::EStop {
                    self.note(Tone::Warn, "Motor power is cut. /reset first.");
                    return;
                }
                self.cancel_goal();
                self.teleop_target = self.sup.last_state().ee_xyz_m;
                self.teleop_gripper = self.sup.last_state().gripper_open_ratio;
                self.teleop_pending = false;
                let outcome = self.sup.operator_engage_teleop();
                self.absorb_interrupt(outcome);
                self.note(
                    Tone::Info,
                    "You are driving. Your moves still pass the safety gate. Esc hands control back.",
                );
            }
            "/estop" => {
                self.cancel_goal();
                let outcome = self.sup.operator_estop();
                self.absorb_interrupt(outcome);
                self.note(Tone::Danger, "Emergency stop. Motor power is cut. /reset to clear.");
            }
            "/reset" => {
                self.cancel_goal();
                self.finish_step(false, "interrupted", "Reset by operator.");
                self.sup.operator_reset_world();
                self.note(Tone::Info, "The arm and the table are back at the starting state.");
            }
            "/inject" => match arg {
                Some("slip") => {
                    if self.sup.last_state().held_object_id.is_none() {
                        self.note(
                            Tone::Info,
                            "Nothing is in the gripper, so there is nothing to drop.",
                        );
                    } else {
                        self.sup.adapter_mut().inject_grasp_slip();
                        self.note(Tone::Warn, "Fault injected: the gripper lost its grip.");
                    }
                }
                Some("spike") => {
                    self.sup.adapter_mut().inject_effort_spike();
                    if self.sup.phase() == RuntimePhase::Executing {
                        self.note(Tone::Warn, "Fault injected: 95 N·m torque spike on joint 0.");
                    } else {
                        self.note(
                            Tone::Info,
                            "Fault injected, but the safety gate only reacts to torque spikes while a skill is running.",
                        );
                    }
                }
                _ => self.note(Tone::Warn, "Usage: /inject slip or /inject spike."),
            },
            "/save" => {
                let path = arg.map(PathBuf::from).unwrap_or_else(|| self.record_out.clone());
                match self.sup.recorder().save_to_path(&path) {
                    Ok(()) => {
                        let n = self.sup.recorder().events().len();
                        self.note(
                            Tone::Info,
                            format!("Saved {n} recorded events to {}.", path.display()),
                        );
                    }
                    Err(err) => self.note(Tone::Danger, format!("Could not save: {err}")),
                }
            }
            other => self.note(
                Tone::Warn,
                format!("Unknown command {other}. Type / to see the list."),
            ),
        }
    }

    // ---- simulation ----------------------------------------------------------------------

    /// One UI frame of work: collect a planner answer, ask for the next one, run the control loop.
    pub fn advance(&mut self) {
        self.frame += 1;

        while let Ok((generation, res)) = self.plan_rx.try_recv() {
            if generation != self.plan_gen {
                continue; // answer to a goal that has since been cancelled or replaced
            }
            self.thinking = false;
            match res {
                Ok(PlanStep::Done { summary }) => {
                    self.entries.push(Entry::Done(summary));
                    self.cancel_goal();
                }
                Ok(PlanStep::Invoke(call)) => self.begin_step(call),
                Err(err) => {
                    self.note(Tone::Danger, format!("The planner gave up: {err}"));
                    self.cancel_goal();
                }
            }
        }

        if let Some(goal) = self.goal.clone()
            && !self.thinking
            && self.sup.phase() == RuntimePhase::Idle
        {
            if self.step_count >= MAX_STEPS_PER_GOAL {
                self.note(
                    Tone::Warn,
                    format!("Stopped after {MAX_STEPS_PER_GOAL} steps without finishing the goal."),
                );
                self.cancel_goal();
            } else {
                self.thinking = true;
                let tx = self.plan_tx.clone();
                let planner = Arc::clone(&self.planner);
                let generation = self.plan_gen;
                let skills: Vec<SkillSpec> = self.sup.skills().values().cloned().collect();
                let state = self.sup.last_state().clone();
                let history = self.history.clone();
                let failure = self.last_failure.clone();
                thread::spawn(move || {
                    let res =
                        planner.decide_next(&goal, &skills, &state, &history, failure.as_ref());
                    let _ = tx.send((generation, res));
                });
            }
        }

        for _ in 0..TICKS_PER_FRAME {
            let teleop_cmd = (self.sup.phase() == RuntimePhase::TeleopTakeover
                && self.teleop_pending)
                .then(|| JointCommand {
                    timestamp_ns: self.sup.last_state().timestamp_ns,
                    target_positions: self.sup.last_state().joint_positions.clone(),
                    target_ee_xyz_m: Some(self.teleop_target),
                    gripper_open_ratio: self.teleop_gripper,
                });
            if let Some(outcome) = self.sup.tick_once(teleop_cmd.as_ref()) {
                self.absorb_outcome(outcome);
            }
        }
        if self.sup.phase() == RuntimePhase::TeleopTakeover
            && dist3(self.sup.last_state().ee_xyz_m, self.teleop_target) < 0.002
            && (self.sup.last_state().gripper_open_ratio - self.teleop_gripper).abs() < 1e-6
        {
            self.teleop_pending = false;
        }
    }

    fn begin_step(&mut self, call: SkillCall) {
        self.step_count += 1;
        self.entries.push(Entry::Step {
            n: self.step_count,
            skill: call.skill_name.clone(),
            what: describe_call(&call),
            thought: call.thought.clone(),
            started_ns: self.sup.last_state().timestamp_ns,
            result: None,
        });
        if let Err(trace) = self.sup.start_skill(call) {
            self.finish_step(false, failure_label(&trace.code), &trace.message);
            self.history
                .push(format!("FAILED {:?}: {}", trace.code, trace.message));
            self.last_failure = Some(trace);
        }
    }

    fn absorb_outcome(&mut self, outcome: SkillOutcome) {
        match outcome {
            SkillOutcome::Completed {
                summary,
                ticks_elapsed,
                ..
            } => {
                let secs = ticks_elapsed as f64 * DT_NS as f64 * 1e-9;
                self.finish_step(true, &format!("{secs:.2} s"), &summary);
                self.history.push(summary);
                self.last_failure = None;
            }
            SkillOutcome::Failed(trace) => {
                self.finish_step(false, failure_label(&trace.code), &trace.message);
                self.history
                    .push(format!("FAILED {:?}: {}", trace.code, trace.message));
                if self.sup.phase() == RuntimePhase::SafeHold {
                    self.cancel_goal();
                    self.note(
                        Tone::Warn,
                        "The safety gate is holding the arm in place. /resume to continue, or /takeover to drive by hand.",
                    );
                }
                self.last_failure = Some(trace);
            }
        }
    }

    /// A skill cut short by the operator comes back as a failure trace; show it on its step.
    fn absorb_interrupt(&mut self, outcome: Option<SkillOutcome>) {
        if let Some(SkillOutcome::Failed(trace)) = outcome {
            self.finish_step(false, failure_label(&trace.code), &trace.message);
        }
    }

    fn finish_step(&mut self, ok: bool, label: &str, text: &str) {
        let open = self.entries.iter_mut().rev().find_map(|e| match e {
            Entry::Step { result, .. } if result.is_none() => Some(result),
            _ => None,
        });
        if let Some(slot) = open {
            *slot = Some(StepResult {
                ok,
                label: label.to_string(),
                text: text.to_string(),
            });
        }
    }

    fn note(&mut self, tone: Tone, text: impl Into<String>) {
        self.entries.push(Entry::Note {
            tone,
            text: text.into(),
        });
    }

    // ---- drawing -------------------------------------------------------------------------

    pub fn draw(&self, f: &mut Frame) {
        let area = f.area();
        let palette = self.matching_commands();
        let footer_h = 3 + palette.len() as u16;
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(footer_h),
        ])
        .areas(area);

        self.draw_header(f, header);

        if body.width >= 92 && body.height >= 14 {
            let side_w = (body.width * 2 / 5).clamp(40, 60);
            let [left, rule, right] = Layout::horizontal([
                Constraint::Min(30),
                Constraint::Length(3),
                Constraint::Length(side_w),
            ])
            .areas(body);
            self.draw_transcript(f, left);
            let bar: Vec<Line> = (0..rule.height).map(|_| Line::styled(" │", dim())).collect();
            f.render_widget(Paragraph::new(bar), rule);
            self.draw_arm_panel(f, right);
        } else {
            self.draw_transcript(f, body);
        }

        self.draw_footer(f, footer, &palette);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let mode = self.sup.arbiter().mode();
        let lamp = match mode {
            ArbiterMode::Idle => Span::styled(
                " Idle ",
                Style::new().fg(Color::DarkGray).add_modifier(Modifier::REVERSED),
            ),
            ArbiterMode::Autonomous => lamp_span(" Running ", Color::Green),
            ArbiterMode::SafeHold => lamp_span(" Safety hold ", Color::Yellow),
            ArbiterMode::TeleopTakeover => lamp_span(" Hand control ", Color::Magenta),
            ArbiterMode::EStop => lamp_span(" Emergency stop ", Color::Red),
        };

        let mut title = vec![
            Span::styled(" openRocode", bold()),
            Span::raw("  "),
            lamp,
        ];
        let planner = if self.offline {
            "scripted demo planner".to_string()
        } else {
            format!("planner {}", self.planner.name())
        };
        let context = format!("{planner}    simulated arm ");
        let used: usize = title.iter().map(|s| s.content.width()).sum();
        let room = (area.width as usize).saturating_sub(used);
        if context.width() + 2 <= room {
            title.push(Span::raw(" ".repeat(room - context.width())));
            title.push(Span::styled(context, dim()));
        }

        // Who is in control right now, as a path from you to the motors.
        #[derive(PartialEq)]
        enum Stage {
            You,
            Planner,
            Skill,
            Gate,
        }
        let spin = SPINNER[(self.frame / 5) as usize % SPINNER.len()];
        let (active, color, detail) = match mode {
            ArbiterMode::EStop => (Stage::Gate, Color::Red, ", power cut".to_string()),
            ArbiterMode::SafeHold => (Stage::Gate, Color::Yellow, ", holding".to_string()),
            ArbiterMode::TeleopTakeover => {
                (Stage::You, Color::Magenta, ", driving by hand".to_string())
            }
            ArbiterMode::Autonomous => (
                Stage::Skill,
                Color::Green,
                format!(" {}", self.sup.active_skill_name().unwrap_or("")),
            ),
            ArbiterMode::Idle if self.goal.is_some() => {
                (Stage::Planner, ACCENT, format!(" {spin} deciding"))
            }
            ArbiterMode::Idle => (Stage::You, ACCENT, ", your turn".to_string()),
        };
        let bypassed = mode == ArbiterMode::TeleopTakeover;
        let mut path = vec![Span::raw(" ")];
        let stages = [
            (Some(Stage::You), "you"),
            (Some(Stage::Planner), "planner"),
            (Some(Stage::Skill), "skill"),
            (Some(Stage::Gate), "safety gate"),
            (None, "arm"),
        ];
        for (i, (stage, label)) in stages.into_iter().enumerate() {
            if i > 0 {
                path.push(Span::styled("  ▸  ", dim()));
            }
            if stage.as_ref() == Some(&active) {
                path.push(Span::styled(
                    label,
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ));
                path.push(Span::styled(detail.clone(), Style::new().fg(color)));
            } else if bypassed && matches!(stage, Some(Stage::Planner | Stage::Skill)) {
                path.push(Span::styled(label, dim().add_modifier(Modifier::CROSSED_OUT)));
            } else {
                path.push(Span::styled(label, dim()));
            }
        }

        let rule = Line::styled("─".repeat(area.width as usize), dim());
        f.render_widget(
            Paragraph::new(vec![Line::from(title), Line::from(path), rule]),
            area,
        );
    }

    fn draw_transcript(&self, f: &mut Frame, area: Rect) {
        let width = (area.width as usize).saturating_sub(1);
        let lines = if self.entries.is_empty() {
            self.welcome_lines(width)
        } else {
            self.transcript_lines(width)
        };
        let height = area.height as usize;
        let max_scroll = lines.len().saturating_sub(height);
        self.max_scroll.set(max_scroll);
        let scroll = self.scroll_up.min(max_scroll);
        let start = lines.len().saturating_sub(height + scroll);
        let mut visible: Vec<Line> = lines.into_iter().skip(start).take(height).collect();
        if scroll > 0
            && let Some(last) = visible.last_mut()
        {
            *last = Line::styled(
                format!(" ↓ {scroll} newer lines, PgDn to follow"),
                Style::new().fg(ACCENT),
            );
        }
        f.render_widget(Paragraph::new(visible), area);
    }

    fn welcome_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut out = vec![Line::default()];
        push_wrapped(
            &mut out,
            2,
            "Tell the arm what to do, in plain language.",
            bold(),
            width,
        );
        out.push(Line::default());
        push_wrapped(
            &mut out,
            2,
            "A planner turns your goal into one motion at a time. Every command then passes a \
             safety gate, 500 times a second, before it reaches the arm. Each step shows up here \
             with its result.",
            Style::new(),
            width,
        );
        out.push(Line::default());
        push_wrapped(&mut out, 2, "Goals to try (Tab fills one in):", dim(), width);
        for goal in EXAMPLE_GOALS {
            push_wrapped(&mut out, 4, goal, Style::new().fg(ACCENT), width);
        }
        out.push(Line::default());
        if self.offline {
            push_wrapped(
                &mut out,
                2,
                "No API key is set, so a scripted stand-in is playing the planner. It only \
                 understands the three goals above. Set ANTHROPIC_API_KEY or OPENAI_API_KEY and \
                 restart to put a real model in charge.",
                Style::new().fg(Color::Yellow),
                width,
            );
        } else {
            push_wrapped(
                &mut out,
                2,
                &format!(
                    "{} is planning. It sees object positions and the gripper state, not camera images.",
                    self.planner.name()
                ),
                dim(),
                width,
            );
        }
        out
    }

    fn transcript_lines(&self, width: usize) -> Vec<Line<'static>> {
        let now_ns = self.sup.last_state().timestamp_ns;
        let spin = SPINNER[(self.frame / 5) as usize % SPINNER.len()];
        let mut out: Vec<Line<'static>> = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            match entry {
                Entry::Goal(text) => {
                    if i > 0 {
                        out.push(Line::default());
                    }
                    for (j, part) in wrap(text, width.saturating_sub(3)).into_iter().enumerate() {
                        let lead = if j == 0 {
                            Span::styled(" ❯ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
                        } else {
                            Span::raw("   ")
                        };
                        out.push(Line::from(vec![lead, Span::styled(part, bold())]));
                    }
                    out.push(Line::default());
                }
                Entry::Step {
                    n,
                    skill,
                    what,
                    thought,
                    started_ns,
                    result,
                } => {
                    let head = if what.is_empty() {
                        skill.clone()
                    } else {
                        format!("{skill} {what}")
                    };
                    for (j, part) in wrap(&head, width.saturating_sub(5)).into_iter().enumerate() {
                        if j == 0 {
                            let cut = skill.len().min(part.len());
                            out.push(Line::from(vec![
                                Span::styled(format!("{n:>3}  "), dim()),
                                Span::styled(part[..cut].to_string(), bold()),
                                Span::raw(part[cut..].to_string()),
                            ]));
                        } else {
                            out.push(Line::from(format!("     {part}")));
                        }
                    }
                    if !thought.is_empty() {
                        push_wrapped(&mut out, 5, thought, dim(), width);
                    }
                    match result {
                        Some(r) => {
                            let (mark, color) = if r.ok {
                                ("✓", Color::Green)
                            } else {
                                ("✗", Color::Red)
                            };
                            let lead = format!("{mark} {}  ", r.label);
                            let room = width.saturating_sub(5 + lead.width());
                            for (j, part) in wrap(&r.text, room).into_iter().enumerate() {
                                let first = if j == 0 {
                                    Span::styled(format!("     {lead}"), Style::new().fg(color))
                                } else {
                                    Span::raw(" ".repeat(5 + lead.width()))
                                };
                                out.push(Line::from(vec![first, Span::raw(part)]));
                            }
                        }
                        None => {
                            let secs = now_ns.saturating_sub(*started_ns) as f64 * 1e-9;
                            out.push(Line::styled(
                                format!("     {spin} running, {secs:.2} s"),
                                Style::new().fg(Color::Green),
                            ));
                        }
                    }
                }
                Entry::Note { tone, text } => {
                    let style = match tone {
                        Tone::Info => dim(),
                        Tone::Warn => Style::new().fg(Color::Yellow),
                        Tone::Danger => Style::new().fg(Color::Red).add_modifier(Modifier::BOLD),
                    };
                    push_wrapped(&mut out, 5, text, style, width);
                }
                Entry::Table(rows) => {
                    let key_w = rows.iter().map(|(k, _)| k.width()).max().unwrap_or(0);
                    for (key, value) in rows {
                        let room = width.saturating_sub(5 + key_w + 2);
                        for (j, part) in wrap(value, room).into_iter().enumerate() {
                            let lead = if j == 0 {
                                format!("     {key:<key_w$}  ")
                            } else {
                                " ".repeat(5 + key_w + 2)
                            };
                            out.push(Line::from(vec![
                                Span::styled(lead, bold()),
                                Span::styled(part, dim()),
                            ]));
                        }
                    }
                }
                Entry::Done(summary) => {
                    out.push(Line::default());
                    let lead = "✓ Done  ";
                    let room = width.saturating_sub(5 + lead.width());
                    for (j, part) in wrap(summary, room).into_iter().enumerate() {
                        let first = if j == 0 {
                            Span::styled(
                                format!("     {lead}"),
                                Style::new().fg(Color::Green).add_modifier(Modifier::BOLD),
                            )
                        } else {
                            Span::raw(" ".repeat(5 + lead.width()))
                        };
                        out.push(Line::from(vec![first, Span::raw(part)]));
                    }
                }
            }
        }

        if self.thinking && self.goal.is_some() {
            out.push(Line::styled(
                format!("     {spin} planner is deciding the next step"),
                Style::new().fg(ACCENT),
            ));
        }
        out
    }

    fn draw_arm_panel(&self, f: &mut Frame, area: Rect) {
        let readout_h = 5;
        let views = area.height.saturating_sub(readout_h);
        let two_views = views >= 22;
        let top_h = if two_views { views * 11 / 20 } else { views };
        let [top, side, readout] = Layout::vertical([
            Constraint::Length(top_h),
            Constraint::Length(views - top_h),
            Constraint::Length(readout_h),
        ])
        .areas(area);

        let state = self.sup.last_state();
        self.draw_view(f, top, "From above", |canvas_area| top_view(state, canvas_area));
        if two_views {
            self.draw_view(f, side, "From the side", |canvas_area| side_view(state, canvas_area));
        }
        f.render_widget(Paragraph::new(self.readout_lines(state)), readout);
    }

    fn draw_view<'a, W, B>(&self, f: &mut Frame, area: Rect, label: &'static str, build: B)
    where
        W: ratatui::widgets::Widget + 'a,
        B: FnOnce(Rect) -> W,
    {
        if area.height < 3 {
            return;
        }
        let [caption, canvas] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(2)]).areas(area);
        f.render_widget(Paragraph::new(Line::styled(label, dim())), caption);
        f.render_widget(build(canvas), canvas);
    }

    fn readout_lines(&self, state: &RobotState) -> Vec<Line<'static>> {
        let label = |text: &str| Span::styled(format!("{text:<9}"), dim());

        let cells = (state.gripper_open_ratio * 10.0).round() as usize;
        let grip_text = match (&state.held_object_id, state.gripper_open_ratio) {
            (Some(id), _) => format!("holding {}", id.replace('_', " ")),
            (None, r) if r >= 0.95 => "open".to_string(),
            (None, r) if r <= 0.05 => "closed, empty".to_string(),
            (None, r) => format!("{:.0}% open", r * 100.0),
        };

        let contact = if state.ee_contact_force_n > 0.0 {
            Span::styled(
                format!("{:.1} N, touching the table", state.ee_contact_force_n),
                Style::new().fg(Color::Yellow),
            )
        } else {
            Span::raw("none")
        };

        let limit = self.sup.arbiter().envelope().max_effort;
        let peak = state
            .joint_efforts
            .iter()
            .fold(0.0_f64, |m, e| m.max(e.abs()));
        let bars: String = state
            .joint_efforts
            .iter()
            .map(|e| {
                let level = ((e.abs() / limit).clamp(0.0, 1.0) * 7.0).round() as usize;
                ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level]
            })
            .collect();
        let effort_color = match peak / limit {
            r if r > 0.85 => Color::Red,
            r if r > 0.5 => Color::Yellow,
            _ => Color::Green,
        };

        vec![
            Line::default(),
            Line::from(vec![
                label("hand"),
                Span::raw(format!(
                    "x {:+.3}  y {:+.3}  z {:+.3} m",
                    state.ee_xyz_m[0], state.ee_xyz_m[1], state.ee_xyz_m[2]
                )),
            ]),
            Line::from(vec![
                label("gripper"),
                Span::styled("█".repeat(cells), Style::new().fg(ACCENT)),
                Span::styled("░".repeat(10 - cells.min(10)), dim()),
                Span::raw(format!("  {grip_text}")),
            ]),
            Line::from(vec![label("contact"), contact]),
            Line::from(vec![
                label("torque"),
                Span::styled(bars, Style::new().fg(effort_color)),
                Span::raw(format!("  peak {peak:.1} of {limit:.0} N·m allowed")),
            ]),
        ]
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect, palette: &[(&str, &str, bool)]) {
        let phase = self.sup.phase();
        let width = area.width as usize;
        let mut lines = vec![Line::styled("─".repeat(width), dim())];

        for (i, (cmd, what, _)) in palette.iter().enumerate() {
            let style = if i == 0 { Style::new().fg(ACCENT) } else { Style::new() };
            lines.push(Line::from(vec![
                Span::styled(format!("   {cmd:<15}"), style),
                Span::styled((*what).to_string(), dim()),
            ]));
        }

        let teleop = phase == RuntimePhase::TeleopTakeover;
        if teleop {
            lines.push(Line::styled(
                " You are driving the arm.",
                Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ));
        } else if self.input.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(" ❯ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("Tell the arm what to do, or type / for commands", dim()),
            ]));
        } else {
            let shown = tail_fit(&self.input, width.saturating_sub(4));
            lines.push(Line::from(vec![
                Span::styled(" ❯ ", Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
                Span::raw(shown.to_string()),
            ]));
        }

        let hint = match phase {
            RuntimePhase::TeleopTakeover => {
                "W/S forward, back    A/D left, right    Q/E up, down    Space gripper    Esc hand back control"
            }
            RuntimePhase::EStop => "/reset restores power and starts over",
            RuntimePhase::SafeHold => "/resume continue    /takeover drive by hand    /reset start over",
            _ if !palette.is_empty() => "Tab complete    Enter run    Esc cancel",
            _ if self.goal.is_some() || phase == RuntimePhase::Executing => {
                "/hold pause    /takeover drive by hand    /estop cut power"
            }
            _ if width < 78 => "Enter send    Tab example    / commands    Ctrl-C quit",
            _ => "Enter send    Tab example    / commands    PgUp/PgDn scroll    Ctrl-C quit",
        };
        lines.push(Line::styled(format!(" {hint}"), dim()));
        f.render_widget(Paragraph::new(lines), area);

        if !teleop {
            let input_row = area.y + 1 + palette.len() as u16;
            let shown = tail_fit(&self.input, width.saturating_sub(4));
            let x = area.x + 3 + shown.width() as u16;
            f.set_cursor_position((x.min(area.right().saturating_sub(1)), input_row));
        }
    }
}

// ---- arm drawings ----------------------------------------------------------------------------

/// Canvas bounds that show at least `span` around `center` without distorting distances.
/// A braille dot is half a cell wide and a quarter of a cell tall, which is close to square.
fn fit_bounds(area: Rect, center: (f64, f64), span: (f64, f64)) -> ([f64; 2], [f64; 2]) {
    let dots_x = (area.width as f64 * 2.0).max(2.0);
    let dots_y = (area.height as f64 * 4.0).max(4.0);
    let scale = (span.0 / dots_x).max(span.1 / dots_y);
    let (hx, hy) = (scale * dots_x / 2.0, scale * dots_y / 2.0);
    ([center.0 - hx, center.0 + hx], [center.1 - hy, center.1 + hy])
}

fn object_color(obj: &SceneObject) -> Color {
    if obj.id.contains("mug") {
        Color::Red
    } else if obj.id.contains("coaster") {
        Color::Yellow
    } else if obj.id.contains("spill") {
        Color::Blue
    } else {
        Color::Reset
    }
}

fn is_wiped(obj: &SceneObject) -> bool {
    obj.id.contains("spill") && obj.label.contains("wiped")
}

fn object_name(obj: &SceneObject) -> String {
    if is_wiped(obj) {
        "✓ wiped clean".to_string()
    } else {
        obj.id.replace('_', " ")
    }
}

fn segment(ctx: &mut Context, a: (f64, f64), b: (f64, f64), color: Color) {
    ctx.draw(&Stroke::new(a.0, a.1, b.0, b.1, color));
}

/// Looking straight down. The arm's base is at the bottom, forward (+x) is up, +y is left.
fn top_view(state: &RobotState, area: Rect) -> impl ratatui::widgets::Widget + '_ {
    let (xb, yb) = fit_bounds(area, (0.0, 0.30), (1.00, 0.74));
    let cell_w = (xb[1] - xb[0]) / f64::from(area.width.max(1));
    let cell_h = (yb[1] - yb[0]) / f64::from(area.height.max(1));
    let at = |x: f64, y: f64| (-y, x);

    Canvas::default()
        .marker(Marker::Braille)
        .x_bounds(xb)
        .y_bounds(yb)
        .paint(move |ctx| {
            // Reachable area of the table.
            ctx.draw(&Rectangle::new(-0.45, 0.12, 0.90, 0.50, Color::DarkGray));
            ctx.layer();

            for obj in &state.scene_objects {
                let (cx, cy) = at(obj.xyz_m[0], obj.xyz_m[1]);
                let color = object_color(obj);
                if obj.id.contains("mug") {
                    ctx.draw(&Circle::new(cx, cy, 0.032, color));
                } else if obj.id.contains("coaster") {
                    ctx.draw(&Rectangle::new(cx - 0.05, cy - 0.05, 0.10, 0.10, color));
                } else if obj.id.contains("spill") {
                    if !is_wiped(obj) {
                        ctx.draw(&Circle::new(cx, cy, 0.036, color));
                        ctx.draw(&Circle::new(cx + 0.022, cy - 0.012, 0.022, color));
                        ctx.draw(&Circle::new(cx - 0.012, cy + 0.016, 0.014, color));
                    }
                } else {
                    ctx.draw(&Rectangle::new(cx - 0.02, cy - 0.02, 0.04, 0.04, color));
                }
            }
            ctx.layer();

            // Arm: base, upper arm, forearm, and the two gripper jaws.
            let az = state.joint_positions.first().copied().unwrap_or(0.0);
            let elbow = MockRobotAdapter::elbow_xyz(&state.joint_positions);
            let hand = state.ee_xyz_m;
            ctx.draw(&Circle::new(0.0, 0.0, 0.030, Color::Reset));
            segment(ctx, at(0.0, 0.0), at(elbow[0], elbow[1]), Color::Reset);
            segment(ctx, at(elbow[0], elbow[1]), at(hand[0], hand[1]), Color::Reset);
            let (ux, uy) = (az.cos(), az.sin());
            let (px, py) = (-az.sin(), az.cos());
            let gap = 0.012 + 0.024 * state.gripper_open_ratio;
            for side in [-1.0, 1.0] {
                let (jx, jy) = (hand[0] + side * gap * px, hand[1] + side * gap * py);
                segment(
                    ctx,
                    at(jx - 0.014 * ux, jy - 0.014 * uy),
                    at(jx + 0.026 * ux, jy + 0.026 * uy),
                    ACCENT,
                );
            }
            ctx.layer();

            let mut placed: Vec<(f64, f64, f64)> = Vec::new();
            for obj in &state.scene_objects {
                let (cx, cy) = at(obj.xyz_m[0], obj.xyz_m[1]);
                let style = if is_wiped(obj) {
                    Style::new().fg(Color::Green)
                } else {
                    Style::new().fg(object_color(obj))
                };
                let name = object_name(obj);
                let width = name.width() as f64 * cell_w;
                let reach = if obj.id.contains("mug") { 0.04 } else { 0.065 };
                // Labels grow outward: leftward for things left of the arm, rightward otherwise.
                let x = if cx < -0.05 {
                    cx - reach - width - cell_w
                } else {
                    cx + reach + cell_w
                };
                // Two things in the same spot (the mug on the coaster) get stacked labels.
                let mut y = cy;
                while placed.iter().any(|&(x0, x1, py): &(f64, f64, f64)| {
                    (y - py).abs() < cell_h * 0.9 && x < x1 + cell_w && x + width > x0 - cell_w
                }) {
                    y += cell_h;
                }
                placed.push((x, x + width, y));
                ctx.print(x, y, Span::styled(name, style));
            }
            ctx.print(0.05, -0.005, Span::styled("base", dim()));
        })
}

/// Looking from the side, in the vertical plane the arm is pointing along:
/// reach away from the base runs left to right, height runs up.
fn side_view(state: &RobotState, area: Rect) -> impl ratatui::widgets::Widget + '_ {
    let span = (0.86, 0.56);
    let (xb, yb) = fit_bounds(area, (0.0, 0.24), span);
    // Keep the base near the left edge however wide the panel is.
    let half = (xb[1] - xb[0]) / 2.0;
    let xb = [-0.12, -0.12 + 2.0 * half];

    Canvas::default()
        .marker(Marker::Braille)
        .x_bounds(xb)
        .y_bounds(yb)
        .paint(move |ctx| {
            segment(ctx, (xb[0], 0.0), (xb[1], 0.0), Color::DarkGray);
            ctx.layer();

            let az = state.joint_positions.first().copied().unwrap_or(0.0);
            let hand = state.ee_xyz_m;
            let reach = hand[0].hypot(hand[1]);

            // Only things in the slice the arm is pointing through belong in this view.
            for obj in &state.scene_objects {
                let along = az.cos() * obj.xyz_m[0] + az.sin() * obj.xyz_m[1];
                let aside = -az.sin() * obj.xyz_m[0] + az.cos() * obj.xyz_m[1];
                if aside.abs() > 0.07 || along < 0.0 || is_wiped(obj) {
                    continue;
                }
                let z = obj.xyz_m[2];
                let color = object_color(obj);
                if obj.id.contains("mug") {
                    ctx.draw(&Rectangle::new(along - 0.03, z - 0.04, 0.06, 0.08, color));
                } else if obj.id.contains("coaster") {
                    ctx.draw(&Rectangle::new(along - 0.05, 0.0, 0.10, 0.012, color));
                } else if obj.id.contains("spill") {
                    segment(ctx, (along - 0.04, 0.006), (along + 0.04, 0.006), color);
                } else {
                    ctx.draw(&Rectangle::new(along - 0.02, z - 0.02, 0.04, 0.04, color));
                }
            }
            ctx.layer();

            let shoulder = MockRobotAdapter::shoulder_xyz();
            let elbow = MockRobotAdapter::elbow_xyz(&state.joint_positions);
            let elbow_reach = elbow[0].hypot(elbow[1]);
            let wrist = (reach, hand[2] + 0.045);
            ctx.draw(&Rectangle::new(-0.035, 0.0, 0.07, shoulder[2], Color::Reset));
            segment(ctx, (0.0, shoulder[2]), (elbow_reach, elbow[2]), Color::Reset);
            segment(ctx, (elbow_reach, elbow[2]), wrist, Color::Reset);
            let gap = 0.012 + 0.024 * state.gripper_open_ratio;
            segment(ctx, (reach - gap, wrist.1), (reach + gap, wrist.1), ACCENT);
            for side in [-1.0, 1.0] {
                segment(
                    ctx,
                    (reach + side * gap, wrist.1),
                    (reach + side * gap, hand[2]),
                    ACCENT,
                );
            }
            ctx.layer();

            ctx.print(xb[1] - 0.16, 0.035, Span::styled("table", dim()));
            if state.ee_contact_force_n > 0.0 {
                ctx.print(
                    reach + 0.07,
                    hand[2] + 0.05,
                    Span::styled(
                        format!("contact {:.0} N", state.ee_contact_force_n),
                        Style::new().fg(Color::Yellow),
                    ),
                );
            }
        })
}

// ---- small helpers ---------------------------------------------------------------------------

fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

fn bold() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

fn lamp_span(text: &'static str, color: Color) -> Span<'static> {
    Span::styled(
        text,
        Style::new().fg(Color::Black).bg(color).add_modifier(Modifier::BOLD),
    )
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn failure_label(code: &FailureCode) -> &'static str {
    match code {
        FailureCode::PreconditionRefused => "refused",
        FailureCode::InvalidArguments => "bad arguments",
        FailureCode::HeartbeatTimeout => "watchdog timeout",
        FailureCode::SafetyEnvelopeViolation => "stopped by the safety gate",
        FailureCode::GraspSlip => "lost its grip",
        FailureCode::ExecutionTimeout => "timed out",
        FailureCode::WorkerCrashed => "plugin crashed",
        FailureCode::PreemptedByOperator => "interrupted",
    }
}

fn point_text(value: Option<&serde_json::Value>) -> Option<String> {
    let arr = value?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some(format!(
        "({:.2}, {:.2}, {:.2})",
        arr[0].as_f64()?,
        arr[1].as_f64()?,
        arr[2].as_f64()?
    ))
}

/// A skill call's arguments in words, e.g. `to (0.42, 0.16, 0.04) m` for `move_ee`.
fn describe_call(call: &SkillCall) -> String {
    let args = &call.arguments;
    let num = |key: &str| args.get(key).and_then(|v| v.as_f64());
    let flag = |key: &str| args.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
    let raw = || match args.as_object() {
        Some(map) if map.is_empty() => String::new(),
        _ => args.to_string(),
    };

    match call.skill_name.as_str() {
        "move_ee" => match point_text(args.get("target_xyz_m")) {
            Some(point) => {
                let how = if flag("relative") { "by" } else { "to" };
                let guard = num("stop_on_contact_n")
                    .map(|n| format!(", stopping at {n:.0} N of contact"))
                    .unwrap_or_default();
                format!("{how} {point} m{guard}")
            }
            None => raw(),
        },
        "set_gripper" => match num("open_ratio") {
            Some(r) if r <= 0.05 && flag("expect_object") => "close, expecting to grip something".into(),
            Some(r) if r <= 0.05 => "close".into(),
            Some(r) if r >= 0.95 => "open".into(),
            Some(r) => format!("to {:.0}% open", r * 100.0),
            None => raw(),
        },
        "trace_path" => {
            let shape = args.get("shape").and_then(|v| v.as_str()).unwrap_or("circle");
            if shape == "waypoints" {
                let n = args
                    .get("points_xyz_m")
                    .and_then(|v| v.as_array())
                    .map_or(0, |a| a.len());
                format!("through {n} waypoints")
            } else {
                match point_text(args.get("center_xyz_m")) {
                    Some(center) => format!(
                        "circle around {center} m, radius {:.1} cm, {} laps",
                        num("radius_m").unwrap_or(0.04) * 100.0,
                        num("cycles").unwrap_or(2.0)
                    ),
                    None => raw(),
                }
            }
        }
        _ => raw(),
    }
}

/// Word-wrap to a display width. Counts wide (CJK) characters as two columns and breaks
/// inside a word only when the word alone is wider than the line.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0;
    for word in text.split(' ') {
        let w = word.width();
        let sep = usize::from(!cur.is_empty());
        if cur_w + sep + w <= width {
            if sep == 1 {
                cur.push(' ');
            }
            cur.push_str(word);
            cur_w += sep + w;
            continue;
        }
        if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            cur_w = 0;
        }
        for ch in word.chars() {
            let cw = ch.width().unwrap_or(0);
            if cur_w + cw > width {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            cur.push(ch);
            cur_w += cw;
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

fn push_wrapped(out: &mut Vec<Line<'static>>, indent: usize, text: &str, style: Style, width: usize) {
    for part in wrap(text, width.saturating_sub(indent)) {
        out.push(Line::styled(format!("{}{part}", " ".repeat(indent)), style));
    }
}

/// The longest suffix of `text` that fits in `width` columns, so the end of a long input stays visible.
fn tail_fit(text: &str, width: usize) -> &str {
    let mut used = 0;
    let mut start = text.len();
    for (i, ch) in text.char_indices().rev() {
        used += ch.width().unwrap_or(0);
        if used > width {
            break;
        }
        start = i;
    }
    &text[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
    use rocode_vlm::SpatialDemoPlanner;
    use serde_json::json;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn demo_app() -> App {
        let root = repo_root();
        App::new(
            &root,
            &root.join("skills"),
            Arc::new(SpatialDemoPlanner),
            true,
            std::env::temp_dir().join("rocode_tui_test.rlog"),
        )
    }

    fn type_line(app: &mut App, text: &str) {
        for ch in text.chars() {
            app.on_key(KeyEvent::from(KeyCode::Char(ch)));
        }
        app.on_key(KeyEvent::from(KeyCode::Enter));
    }

    fn screen(app: &App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Step the app until `done` holds, giving the planner thread time to answer.
    fn run_until(app: &mut App, mut done: impl FnMut(&App) -> bool) {
        for _ in 0..4000 {
            app.advance();
            if done(app) {
                return;
            }
            if app.thinking {
                thread::sleep(Duration::from_millis(1));
            }
        }
        panic!("app did not reach the expected state:\n{}", screen(app, 120, 40));
    }

    #[test]
    fn wrap_respects_width_and_wide_characters() {
        assert_eq!(wrap("move the arm home", 8), vec!["move the", "arm home"]);
        assert_eq!(wrap("把红色杯子放到杯垫上", 8), vec!["把红色杯", "子放到杯", "垫上"]);
        assert_eq!(wrap("", 20), vec![""]);
        assert_eq!(tail_fit("abcdef", 3), "def");
    }

    #[test]
    fn skill_calls_read_as_plain_language() {
        let call = |skill: &str, arguments| SkillCall {
            thought: String::new(),
            skill_name: skill.into(),
            arguments,
        };
        assert_eq!(
            describe_call(&call("move_ee", json!({"target_xyz_m": [0.42, 0.16, 0.04]}))),
            "to (0.42, 0.16, 0.04) m"
        );
        assert_eq!(
            describe_call(&call(
                "move_ee",
                json!({"target_xyz_m": [0.3, 0.0, -0.01], "stop_on_contact_n": 5.0})
            )),
            "to (0.30, 0.00, -0.01) m, stopping at 5 N of contact"
        );
        assert_eq!(
            describe_call(&call("set_gripper", json!({"open_ratio": 0.0, "expect_object": true}))),
            "close, expecting to grip something"
        );
        assert_eq!(describe_call(&call("home_arm", json!({}))), "");
        assert_eq!(describe_call(&call("custom", json!({"a": 1}))), r#"{"a":1}"#);
    }

    #[test]
    fn welcome_screen_explains_itself_at_any_size() {
        let app = demo_app();
        let wide = screen(&app, 140, 42);
        println!("{wide}");
        assert!(wide.contains("Tell the arm what to do"));
        assert!(wide.contains("scripted stand-in"));
        assert!(wide.contains("From above") && wide.contains("From the side"));

        let narrow = screen(&app, 70, 20);
        println!("{narrow}");
        assert!(narrow.contains("Tell the arm what to do"));
        assert!(!narrow.contains("From above"));
    }

    #[test]
    fn a_goal_runs_to_done_and_survives_a_dropped_mug() {
        let mut app = demo_app();
        type_line(&mut app, EXAMPLE_GOALS[0]);

        run_until(&mut app, |a| a.sup.last_state().held_object_id.is_some() && a.step_count >= 4);
        println!("{}\n", screen(&app, 140, 42));
        type_line(&mut app, "/inject slip");

        run_until(&mut app, |a| a.goal.is_none());
        let text = screen(&app, 140, 60);
        println!("{text}");
        assert!(text.contains("lost its grip"), "the slip should show on its step");
        assert!(text.contains("✓ Done"));
        assert!(text.contains("wiped clean"));
    }

    #[test]
    fn every_example_goal_finishes_and_unknown_goals_are_declined() {
        for goal in EXAMPLE_GOALS {
            let mut app = demo_app();
            type_line(&mut app, goal);
            run_until(&mut app, |a| a.goal.is_none());
            let text = screen(&app, 140, 60);
            assert!(text.contains("✓ Done"), "'{goal}' did not finish:\n{text}");
            assert!(app.step_count < 8 || goal.contains("mug on"), "'{goal}' took {} steps", app.step_count);
        }

        let mut app = demo_app();
        type_line(&mut app, "dance");
        run_until(&mut app, |a| a.goal.is_none());
        let text = screen(&app, 140, 42);
        println!("{text}");
        assert!(text.contains("The planner gave up"));
        assert!(!text.contains("✓ Done"));
    }

    #[test]
    fn hold_estop_and_takeover_gate_new_goals() {
        let mut app = demo_app();
        type_line(&mut app, "/hold");
        type_line(&mut app, "wipe the spill");
        assert!(app.goal.is_none(), "a held arm must not accept a goal");
        type_line(&mut app, "/resume");
        assert_eq!(app.sup.phase(), RuntimePhase::Idle);

        type_line(&mut app, "/takeover");
        assert_eq!(app.sup.phase(), RuntimePhase::TeleopTakeover);
        let start = app.sup.last_state().ee_xyz_m;
        for _ in 0..5 {
            app.on_key(KeyEvent::from(KeyCode::Char('w')));
        }
        run_until(&mut app, |a| !a.teleop_pending);
        assert!(app.sup.last_state().ee_xyz_m[0] > start[0] + 0.05);
        println!("{}\n", screen(&app, 140, 42));
        app.on_key(KeyEvent::from(KeyCode::Esc));

        type_line(&mut app, "/estop");
        type_line(&mut app, "wipe the spill");
        assert!(app.goal.is_none());
        let text = screen(&app, 140, 42);
        println!("{text}");
        assert!(text.contains("Emergency stop"));
    }

    #[test]
    fn slash_palette_filters_and_completes() {
        let mut app = demo_app();
        for ch in "/in".chars() {
            app.on_key(KeyEvent::from(KeyCode::Char(ch)));
        }
        let text = screen(&app, 140, 42);
        println!("{text}");
        assert!(text.contains("/inject slip") && text.contains("/inject spike"));
        assert!(!text.contains("/takeover  "));
        app.on_key(KeyEvent::from(KeyCode::Tab));
        assert_eq!(app.input, "/inject slip");
    }
}
