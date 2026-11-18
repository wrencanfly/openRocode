//! EmbodimentAdapter trait and interactive 6-DoF `MockTabletopAdapter` micro-world
//! featuring analytical 3-link arm IK, surface contact forces, object grasping, and fault injection.

use rocode_proto::{FrameRef, JointCommand, RobotState, SceneObject};

pub trait EmbodimentAdapter: Send {
    fn name(&self) -> &str;
    fn dt_ns(&self) -> u64;
    fn read_state(&mut self) -> RobotState;
    fn apply_command(&mut self, cmd: &JointCommand);
    fn inject_effort_spike(&mut self) {}
    fn inject_grasp_slip(&mut self) {}
    fn reset_world(&mut self) {}
}

/// Interactive 6-DoF tabletop micro-world with analytic arm kinematics, table contact spring
/// (`z = 0.0`), graspable objects (`red_mug`, `coaster`, `spill_patch`), and fault injection.
pub struct MockRobotAdapter {
    now_ns: u64,
    dt_ns: u64,
    seq: u64,
    joints: [f64; 6],
    velocities: [f64; 6],
    efforts: [f64; 6],
    ee_xyz_m: [f64; 3],
    gripper_open_ratio: f64,
    held_object_id: Option<String>,
    objects: Vec<SceneObject>,
    fault_effort_spike_at_step: Option<u64>,
    fault_slip_at_step: Option<u64>,
    pending_effort_spike: bool,
    pending_grasp_slip: bool,
    spill_wiped_passes: u32,
}

impl MockRobotAdapter {
    pub const HOME_XYZ: [f64; 3] = [0.30, 0.00, 0.28];
    const L1: f64 = 0.34;
    const L2: f64 = 0.32;
    const BASE_Z: f64 = 0.18;

    pub fn new(dof: usize, dt_ns: u64) -> Self {
        let _ = dof;
        let ee_xyz_m = Self::HOME_XYZ;
        let joints = Self::solve_ik(ee_xyz_m);
        Self {
            now_ns: 0,
            dt_ns,
            seq: 0,
            joints,
            velocities: [0.0; 6],
            efforts: [1.2, 3.5, 2.1, 0.4, 0.3, 0.2],
            ee_xyz_m,
            gripper_open_ratio: 1.0,
            held_object_id: None,
            objects: Self::default_objects(),
            fault_effort_spike_at_step: None,
            fault_slip_at_step: None,
            pending_effort_spike: false,
            pending_grasp_slip: false,
            spill_wiped_passes: 0,
        }
    }

    fn default_objects() -> Vec<SceneObject> {
        vec![
            SceneObject {
                id: "red_mug".to_string(),
                label: "Red ceramic coffee mug".to_string(),
                xyz_m: [0.42, 0.16, 0.04],
                graspable: true,
                held: false,
            },
            SceneObject {
                id: "coaster".to_string(),
                label: "Cork coaster target zone".to_string(),
                xyz_m: [0.36, -0.18, 0.01],
                graspable: false,
                held: false,
            },
            SceneObject {
                id: "spill_patch".to_string(),
                label: "Coffee spill on tabletop".to_string(),
                xyz_m: [0.46, -0.02, 0.00],
                graspable: false,
                held: false,
            },
        ]
    }

    pub fn with_effort_spike_at_step(mut self, step: u64) -> Self {
        self.fault_effort_spike_at_step = Some(step);
        self
    }

    pub fn with_slip_at_step(mut self, step: u64) -> Self {
        self.fault_slip_at_step = Some(step);
        self
    }

    /// Analytical 3-link spherical-wrist IK mapping workspace XYZ to 6 joint angles.
    pub fn solve_ik(xyz: [f64; 3]) -> [f64; 6] {
        let x = xyz[0].max(0.08);
        let y = xyz[1];
        let z_rel = xyz[2] - Self::BASE_Z;
        let q0 = y.atan2(x);
        let r = (x * x + y * y).sqrt();
        let d_sq = (r * r + z_rel * z_rel).clamp(0.01, (Self::L1 + Self::L2 - 0.005).powi(2));
        let cos_q2 = ((d_sq - Self::L1 * Self::L1 - Self::L2 * Self::L2)
            / (2.0 * Self::L1 * Self::L2))
            .clamp(-1.0, 1.0);
        let q2 = -cos_q2.acos(); // elbow-up configuration
        let alpha = z_rel.atan2(r);
        let beta = ((-Self::L2 * q2.sin()).atan2(Self::L1 + Self::L2 * q2.cos())).clamp(-1.5, 1.5);
        let q1 = alpha + beta;
        let q3 = -(q1 + q2) - std::f64::consts::FRAC_PI_2;
        [q0, q1, q2, 0.0, q3, 0.0]
    }

    pub fn forward_kinematics(q: [f64; 6]) -> [f64; 3] {
        let r = Self::L1 * q[1].cos() + Self::L2 * (q[1] + q[2]).cos();
        let z = Self::BASE_Z + Self::L1 * q[1].sin() + Self::L2 * (q[1] + q[2]).sin();
        [r * q[0].cos(), r * q[0].sin(), z]
    }

    /// Where the shoulder joint sits above the table, for drawing the arm.
    pub fn shoulder_xyz() -> [f64; 3] {
        [0.0, 0.0, Self::BASE_Z]
    }

    /// Where the elbow is for the given joint angles, for drawing the arm.
    pub fn elbow_xyz(q: &[f64]) -> [f64; 3] {
        let (q0, q1) = (
            q.first().copied().unwrap_or(0.0),
            q.get(1).copied().unwrap_or(0.0),
        );
        let r = Self::L1 * q1.cos();
        [r * q0.cos(), r * q0.sin(), Self::BASE_Z + Self::L1 * q1.sin()]
    }

    fn compute_contact_force(&self) -> f64 {
        let table_penetration = 0.006 - self.ee_xyz_m[2];
        if table_penetration > 0.0 {
            (2.0 + table_penetration * 1600.0).min(80.0)
        } else {
            0.0
        }
    }

    fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
}

impl EmbodimentAdapter for MockRobotAdapter {
    fn name(&self) -> &str {
        "mock-tabletop-6dof"
    }

    fn dt_ns(&self) -> u64 {
        self.dt_ns
    }

    fn inject_effort_spike(&mut self) {
        self.pending_effort_spike = true;
    }

    fn inject_grasp_slip(&mut self) {
        self.pending_grasp_slip = true;
    }

    fn reset_world(&mut self) {
        self.ee_xyz_m = Self::HOME_XYZ;
        self.joints = Self::solve_ik(self.ee_xyz_m);
        self.velocities = [0.0; 6];
        self.gripper_open_ratio = 1.0;
        self.held_object_id = None;
        self.objects = Self::default_objects();
        self.pending_effort_spike = false;
        self.pending_grasp_slip = false;
        self.spill_wiped_passes = 0;
    }

    fn read_state(&mut self) -> RobotState {
        self.seq += 1;
        self.now_ns += self.dt_ns;

        if self.fault_slip_at_step == Some(self.seq) || self.pending_grasp_slip {
            self.pending_grasp_slip = false;
            if let Some(held_id) = self.held_object_id.take() {
                for obj in &mut self.objects {
                    if obj.id == held_id {
                        obj.held = false;
                        // Drops back to tabletop slightly displaced!
                        obj.xyz_m = [self.ee_xyz_m[0] + 0.03, self.ee_xyz_m[1] - 0.02, 0.04];
                    }
                }
            }
        }

        let contact_force = self.compute_contact_force();
        let mut efforts = self.efforts.to_vec();
        efforts[1] += contact_force * 0.25;
        if self.fault_effort_spike_at_step == Some(self.seq) || self.pending_effort_spike {
            self.pending_effort_spike = false;
            efforts[0] = 95.0;
        }

        RobotState {
            timestamp_ns: self.now_ns,
            joint_positions: self.joints.to_vec(),
            joint_velocities: self.velocities.to_vec(),
            joint_efforts: efforts,
            ee_xyz_m: self.ee_xyz_m,
            ee_rpy_rad: [std::f64::consts::PI, 0.0, 0.0],
            ee_contact_force_n: contact_force,
            gripper_open_ratio: self.gripper_open_ratio,
            gripper_contact_detected: self.held_object_id.is_some(),
            held_object_id: self.held_object_id.clone(),
            scene_objects: self.objects.clone(),
            frames: vec![
                FrameRef {
                    stream_id: "head_rgb".to_string(),
                    seq: self.seq,
                    timestamp_ns: self.now_ns,
                    slot_id: (self.seq % 64) as u32,
                },
                FrameRef {
                    stream_id: "wrist_rgb".to_string(),
                    seq: self.seq,
                    timestamp_ns: self.now_ns,
                    slot_id: ((self.seq + 32) % 64) as u32,
                },
            ],
        }
    }

    fn apply_command(&mut self, cmd: &JointCommand) {
        let dt_s = (self.dt_ns as f64) * 1e-9;
        let prev_joints = self.joints;

        if let Some(target_xyz) = cmd.target_ee_xyz_m {
            let clamped_xyz = [
                target_xyz[0].clamp(0.12, 0.62),
                target_xyz[1].clamp(-0.45, 0.45),
                target_xyz[2].clamp(-0.015, 0.55),
            ];
            self.ee_xyz_m = clamped_xyz;
            self.joints = Self::solve_ik(clamped_xyz);
        } else if cmd.target_positions.len() == 6 {
            self.joints.copy_from_slice(&cmd.target_positions);
            self.ee_xyz_m = Self::forward_kinematics(self.joints);
        }

        for i in 0..6 {
            self.velocities[i] = (self.joints[i] - prev_joints[i]) / dt_s.max(1e-6);
        }

        self.gripper_open_ratio = cmd.gripper_open_ratio.clamp(0.0, 1.0);

        // Update object grasp / release mechanics
        if self.gripper_open_ratio < 0.35 && self.held_object_id.is_none() {
            for obj in &mut self.objects {
                if obj.graspable && Self::dist3(self.ee_xyz_m, obj.xyz_m) <= 0.065 {
                    obj.held = true;
                    self.held_object_id = Some(obj.id.clone());
                    break;
                }
            }
        } else if self.gripper_open_ratio > 0.65
            && let Some(released_id) = self.held_object_id.take()
        {
            for obj in &mut self.objects {
                if obj.id == released_id {
                    obj.held = false;
                    obj.xyz_m = [self.ee_xyz_m[0], self.ee_xyz_m[1], self.ee_xyz_m[2].max(0.04)];
                }
            }
        }

        // Move held object with end-effector
        if let Some(ref held_id) = self.held_object_id {
            for obj in &mut self.objects {
                if &obj.id == held_id {
                    obj.xyz_m = self.ee_xyz_m;
                }
            }
        }

        // Track spill wiping progress when EE rubs near spill_patch close to z=0
        if self.ee_xyz_m[2] <= 0.02 {
            let mut wiped = false;
            for obj in &mut self.objects {
                if obj.id == "spill_patch" && Self::dist3(self.ee_xyz_m, obj.xyz_m) <= 0.08 {
                    self.spill_wiped_passes += 1;
                    if self.spill_wiped_passes >= 60 {
                        wiped = true;
                    }
                }
            }
            if wiped {
                for obj in &mut self.objects {
                    if obj.id == "spill_patch" {
                        obj.label = "Clean tabletop (spill wiped)".to_string();
                    }
                }
            }
        }
    }
}
