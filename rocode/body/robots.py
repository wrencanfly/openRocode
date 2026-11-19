"""Robots that can stand at the tabletop: which model, where it is mounted, and how its arms are driven.

    gripper   a floating parallel gripper (the default): no arm, so no reach limits or joint limits
    panda     a Franka Emika Panda arm mounted behind the table, facing the front camera
    aloha     two ViperX 300 arms (the ALOHA setup), one at each end of the table
    leap      a floating 16-joint LEAP hand, palm down

Arm models come from MuJoCo Menagerie through the `robot_descriptions` package, which downloads them on
first use (`pip install 'openrocode[robots]'`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .base import ArmInfo, Workspace
from .mj import ArmDriver, JointArm, MocapArm, geoms_of_bodies, subtree_bodies

TABLE_WORKSPACE = Workspace(lo=(-0.40, -0.30, 0.0), hi=(0.40, 0.30, 0.40), table_z=0.0)


@dataclass(frozen=True)
class Robot:
    name: str
    description: str                  # what the model is told it controls
    workspace: Workspace
    arms: tuple[ArmInfo, ...]
    home_tcp: dict = field(default_factory=dict)   # arm -> fingertip point after reset
    cameras: dict = field(default_factory=dict)    # name -> (position, look-at point, fovy); empty: the scene's own


def look_at(pos, target) -> list[float]:
    """MuJoCo `xyaxes` for a camera at pos looking at target, with the image's up toward world +z."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return [*right, *np.cross(right, f)]


ROBOTS: dict[str, Robot] = {
    "gripper": Robot("gripper", "a parallel gripper that moves freely above the table", TABLE_WORKSPACE,
                     (ArmInfo("hand", "gripper", 0.09),), {"hand": (0.0, -0.05, 0.18)}),
    "panda": Robot("panda", "a Franka Emika Panda arm, mounted behind the table and facing the front camera",
                   Workspace(lo=(-0.38, -0.28, 0.0), hi=(0.38, 0.15, 0.40), table_z=0.0),
                   (ArmInfo("arm", "gripper", 0.08),), {"arm": (0.0, -0.02, 0.22)},
                   {"front": ((0, -0.80, 0.68), (0, -0.02, 0.06), 50), "side": ((0.70, -0.12, 0.48), (0, 0.0, 0.05), 52)}),
    "aloha": Robot("aloha", "two ViperX arms (ALOHA): `left` at the left end of the table and `right` at the right "
                            "end, as seen from the front camera",
                   Workspace(lo=(-0.32, -0.22, 0.0), hi=(0.32, 0.22, 0.35), table_z=0.0),
                   (ArmInfo("left", "gripper", 0.055), ArmInfo("right", "gripper", 0.055)),
                   {"left": (-0.25, -0.02, 0.20), "right": (0.25, -0.02, 0.20)},
                   {"front": ((0, -0.72, 0.56), (0, 0.0, 0.05), 55), "top": ((0, -0.22, 0.92), (0, 0.02, 0.0), 55)}),
    "leap": Robot("leap", "a four-fingered LEAP hand, palm down, that moves freely above the table. Its marker is the "
                          "point the fingers close around; grip close curls them into a fist around what is under "
                          "that point, and the open hand is about 20 cm across, so mind neighbouring objects",
                  TABLE_WORKSPACE, (ArmInfo("hand", "hand", 0.10),), {"hand": (0.0, -0.05, 0.15)}),
}


def get(name: str) -> Robot:
    if name not in ROBOTS:
        raise ValueError(f"robot {name!r} is not one of {', '.join(ROBOTS)}")
    return ROBOTS[name]


# ---- building ------------------------------------------------------------------------------------


def compile_with_robot(world_xml: str, robot: str) -> mujoco.MjModel:
    """The world (table, objects, cameras; no robot) with the named robot attached."""
    spec = mujoco.MjSpec.from_string(world_xml)
    # Arm actuators are stiff and heavily damped; explicit Euler at 2 ms makes light links oscillate.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    cams = get(robot).cameras
    if cams:   # replace the scene's cameras (before attaching, so the robot's own wrist cameras stay)
        for cam in list(spec.cameras):
            _delete(spec, cam)
        for name, (pos, target, fovy) in cams.items():
            cam = spec.worldbody.add_camera(name=name, pos=list(pos), fovy=fovy)
            cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
            cam.alt.xyaxes = look_at(pos, target)
    {"panda": _attach_panda, "aloha": _attach_aloha, "leap": _attach_leap}[robot](spec)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach")
        return spec.compile()


def _menagerie(module: str) -> str:
    try:
        import importlib

        return importlib.import_module(f"robot_descriptions.{module}").MJCF_PATH
    except ImportError as err:
        raise RuntimeError("this robot's model comes from MuJoCo Menagerie: pip install 'openrocode[robots]'") from err


def _delete(spec, element) -> None:
    """MuJoCo 3.3 deletes through the element, 4.0 through the spec."""
    if hasattr(spec, "delete"):
        spec.delete(element)
    else:
        element.delete()


def _pedestal(spec, x: float, y: float, half: tuple[float, float]) -> None:
    spec.worldbody.add_geom(name=f"pedestal_{len(spec.geoms)}", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[half[0], half[1], 0.375], pos=[x, y, -0.375], rgba=[0.25, 0.26, 0.28, 1])


def _gravcomp(body) -> None:
    body.gravcomp = 1
    for child in body.bodies:
        _gravcomp(child)


def _attach_panda(spec) -> None:
    child = mujoco.MjSpec.from_file(_menagerie("panda_mj_description"))
    child.body("hand").add_site(name="tcp", pos=[0, 0, 0.105], size=[0.003] * 3, rgba=[0, 0, 0, 0])
    for key in list(child.keys):
        _delete(child, key)
    base = child.worldbody.first_body()
    _gravcomp(base)
    frame = spec.worldbody.add_frame(pos=[0, 0.42, 0], quat=[np.cos(-np.pi / 4), 0, 0, np.sin(-np.pi / 4)])
    frame.attach_body(base, "panda/", "")
    _pedestal(spec, 0, 0.42, (0.10, 0.08))


def _attach_aloha(spec) -> None:
    child = mujoco.MjSpec.from_file(os.path.join(os.path.dirname(_menagerie("aloha_mj_description")), "aloha.xml"))
    for key in list(child.keys):
        _delete(child, key)
    for body in child.worldbody.bodies:
        _gravcomp(body)
    spec.attach(child, frame=spec.worldbody.add_frame(), prefix="aloha/")
    for x in (-0.469, 0.469):
        _pedestal(spec, x, -0.019, (0.06, 0.06))


def _attach_leap(spec) -> None:
    child = mujoco.MjSpec.from_file(_menagerie("leap_hand_mj_description"))
    for key in list(child.keys):
        _delete(child, key)
    palm = child.worldbody.first_body()
    palm.pos = [0, 0, 0]   # so the weld to the mocap target is a pure rotation (the model's built-in half turn)
    palm.add_freejoint()
    palm.add_site(name="tcp", pos=list(LEAP_TCP_IN_PALM), size=[0.003] * 3, rgba=[0, 0, 0, 0])
    target = spec.worldbody.add_body(name="leap_target", mocap=True, pos=[0, 0, 0.3])
    del target
    frame = spec.worldbody.add_frame(pos=[0, 0, 0.3])
    frame.attach_body(palm, "leap/", "")
    weld = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="leap_target", name2="leap/palm",
                             objtype=mujoco.mjtObj.mjOBJ_BODY, solref=[0.01, 1])
    # anchor 0, palm at the target's origin, turned by the model's built-in half turn about x, torque scale 1
    data = np.zeros(len(weld.data))
    data[6:11] = [0, 1, 0, 0, 1]
    weld.data = data


# ---- arms ----------------------------------------------------------------------------------------

# palm frame: between the thumb and fingertips of the closed fist, 1.5 cm above them, so that lowering to the table
# leaves the closing fingers just clear of it (like the gripper's fingertip point)
LEAP_TCP_IN_PALM = (-0.038, -0.003, -0.119)
LEAP_PALM_QUAT = (0.0, 1.0, 0.0, 0.0)        # palm down
LEAP_RATE = 0.008                            # rad per physics step that finger targets may change (4 rad/s)


def make_arms(robot: str, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, ArmDriver]:
    r = get(robot)
    if robot == "gripper":
        a = r.arms[0]
        return {a.name: MocapArm(a, model, data, "tcp", ([model.geom("finger_left").id], [model.geom("finger_right").id]),
                                 [model.geom(n).id for n in ("palm", "finger_left", "finger_right")], "target", 0.075,
                                 ("finger_left", "finger_right"), 0.5, ("finger_left", "finger_right"),
                                 r.home_tcp[a.name])}
    if robot == "panda":
        a = r.arms[0]
        act8 = model.actuator("panda/actuator8").id
        f1, f2 = (model.jnt_qposadr[model.joint(f"panda/finger_joint{i}").id] for i in (1, 2))
        grip = (lambda d, w: d.ctrl.__setitem__(act8, w / 0.08 * 255),
                lambda d: d.qpos[f1] + d.qpos[f2])
        hand = subtree_bodies(model, "panda/hand")
        return {a.name: JointArm(a, model, data, "panda/tcp",
                                 (geoms_of_bodies(model, ["panda/left_finger"]), geoms_of_bodies(model, ["panda/right_finger"])),
                                 geoms_of_bodies(model, hand), [f"panda/joint{i}" for i in range(1, 8)],
                                 [f"panda/actuator{i}" for i in range(1, 8)],
                                 [0, 0.2, 0, -2.2, 0, 2.4, 0.785], grip, finger_axis="y", down_axis="z",
                                 yaw_offset=np.pi / 2)}
    if robot == "aloha":
        arms = {}
        for a in r.arms:
            s = f"aloha/{a.name}"
            act = model.actuator(f"{s}/gripper").id
            q1, q2 = (model.jnt_qposadr[model.joint(f"{s}/{j}").id] for j in ("left_finger", "right_finger"))
            grip = (lambda d, w, act=act: d.ctrl.__setitem__(act, float(np.clip((w + 0.0166) / 2, 0.002, 0.037))),
                    lambda d, q1=q1, q2=q2: max(0.0, d.qpos[q1] + d.qpos[q2] - 0.0166))
            joints = [f"{s}/{j}" for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
            hand = subtree_bodies(model, f"{s}/gripper_link")
            arms[a.name] = JointArm(a, model, data, f"{s}/gripper",
                                    (geoms_of_bodies(model, [f"{s}/left_finger_link"]),
                                     geoms_of_bodies(model, [f"{s}/right_finger_link"])),
                                    geoms_of_bodies(model, hand), joints, joints, [0, -0.6, 0.9, 0, 0.9, 0], grip,
                                    finger_axis="y", down_axis="x", orient_weight=0.25,
                                    yaw_offset=np.pi / 2 if a.name == "left" else -np.pi / 2)
        return arms
    if robot == "leap":
        a = r.arms[0]
        acts = [model.actuator(i).name for i in range(model.nu) if model.actuator(i).name.startswith("leap/")]
        joints = [model.joint(model.actuator_trnid[model.actuator(n).id, 0]).name for n in acts]
        hand = subtree_bodies(model, "leap/palm")
        # a power grasp holds between the fingers on one side and the thumb or the palm on the other
        thumb = [b for b in hand if b.split("/")[-1].startswith("th_") or b == "leap/palm"]
        fingers = [b for b in hand if b.split("/")[-1].split("_")[0] in ("if", "mf", "rf")]
        return {a.name: LeapHand(a, model, data, "leap/tcp", (geoms_of_bodies(model, thumb), geoms_of_bodies(model, fingers)),
                                 geoms_of_bodies(model, hand), "leap_target", acts, joints, r.home_tcp[a.name])}
    raise ValueError(robot)


class LeapHand(MocapArm):
    """A floating LEAP hand. `opening` maps onto one grasp synergy, from flat (open) to a power grasp (closed)."""

    def __init__(self, info, model, data, tcp_site, sides, hand_geoms, mocap_body, actuators, joints, home_tcp):
        ArmDriver.__init__(self, info, model, data, tcp_site, sides, hand_geoms)
        self.mocap = model.body(mocap_body).mocapid[0]
        palm, tcp = model.body("leap/palm").id, model.site(tcp_site).id
        self.offset = None
        self.act = [model.actuator(n).id for n in actuators]
        self.qadr = [model.jnt_qposadr[model.joint(j).id] for j in joints]
        self.names = [j.split("/")[-1] for j in joints]
        self.home_tcp = np.asarray(home_tcp, dtype=float)
        self.base_quat = np.asarray(LEAP_PALM_QUAT, dtype=float)
        self.tcp_local = np.asarray(LEAP_TCP_IN_PALM, dtype=float)
        self._palm, self._tcp_site = palm, tcp
        lo, hi = model.actuator_ctrlrange[self.act, 0], model.actuator_ctrlrange[self.act, 1]
        self.open_pose = np.clip(np.zeros(len(self.act)), lo, hi)
        self.closed_pose = np.clip(np.array([LEAP_CLOSED.get(n, 0.0) for n in self.names]), lo, hi)
        self.empty_gap = self._gap_when_closed_on_nothing()
        self.grip_steps = 900      # the fingers are slow and their targets are rate-limited

    def _tcp_offset(self) -> np.ndarray:
        """Palm origin minus TCP, in world. The palm itself is only rotated by yaw (the weld to the mocap target
        carries the model's built-in half turn), and its fingers curl toward world -z: palm down."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return -np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.asarray(self.tcp_local)

    def commanded_tcp(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap] - self._tcp_offset()

    def set_target(self, tcp, yaw: float | None = None) -> None:
        if yaw is not None:
            self.yaw = yaw
        self.data.mocap_pos[self.mocap] = np.asarray(tcp) + self._tcp_offset()
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)]), self.base_quat)
        self.data.mocap_quat[self.mocap] = q

    def set_opening(self, opening_m: float) -> None:
        self.opening_cmd = float(np.clip(opening_m, 0.0, self.info.max_opening_m))
        s = 1.0 - self.opening_cmd / self.info.max_opening_m
        self.ctrl_goal = self.open_pose + s * (self.closed_pose - self.open_pose)

    def before_step(self) -> None:
        """Move the finger targets toward the goal at a limited rate, so a release does not flick things away."""
        cur = self.data.ctrl[self.act]
        self.data.ctrl[self.act] = cur + np.clip(self.ctrl_goal - cur, -LEAP_RATE, LEAP_RATE)

    def opening(self) -> float:
        """How much wider the grasp is than a fist closed on nothing: about the width of what it holds. Open, it
        is the thumb-to-fingertip gap."""
        gap = self._gap(self.data) - (self.empty_gap if self.opening_cmd < self.info.max_opening_m else 0)
        return float(np.clip(gap, 0.0, self.info.max_opening_m))

    def _gap(self, d) -> float:
        tips = {n: d.geom_xpos[self.model.geom(f"leap/{n}_tip").id] for n in ("th", "if", "mf")}
        return float(min(np.linalg.norm(tips["th"] - tips[f]) for f in ("if", "mf")))

    def _gap_when_closed_on_nothing(self) -> float:
        s = mujoco.MjData(self.model)
        s.qpos[:] = self.data.qpos
        s.qpos[self.qadr] = self.closed_pose
        mujoco.mj_kinematics(self.model, s)
        return self._gap(s)

    def home(self) -> None:
        self.yaw = 0.0
        self.set_target(self.home_tcp)
        self.ctrl_goal = self.open_pose.copy()
        self.data.ctrl[self.act] = self.open_pose
        palm = self.data.mocap_pos[self.mocap]
        jnt = self.model.body("leap/palm").jntadr[0]
        qa = self.model.jnt_qposadr[jnt]
        self.data.qpos[qa: qa + 3] = palm
        self.data.qpos[qa + 3: qa + 7] = self.data.mocap_quat[self.mocap]
        self.set_opening(self.info.max_opening_m)


# Joint targets (radians) for the closed grasp; joints not listed stay at zero. Found by a grid search that
# lifts a 4 cm cube, a cube turned 30° and 45°, a 5 x 3 cm box, a 5 cm cylinder, a 5 cm ball and a mug.
LEAP_CLOSED: dict[str, float] = {
    **{f"{f}_{j}": v for f in ("if", "mf", "rf") for j, v in (("mcp", 1.3), ("pip", 0.5), ("dip", 0.5))},
    "th_cmc": 1.4, "th_axl": 0.3, "th_mcp": 0.2, "th_ipl": 0.3,
}
