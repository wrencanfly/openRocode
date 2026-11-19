"""MuJoCo tabletop body: a table, two cameras, the objects of a Scene, and a robot.

The default robot is a floating parallel gripper: a free body welded to a mocap target, so motion is
commanded in task space without an arm model, and reachability and joint limits are not modelled. A scene
can name another robot instead (see rocode/body/robots.py): a Franka Panda, a dual-arm ALOHA, or a LEAP hand.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..scene import DEFAULT_SCENE, Scene, SceneObject
from . import robots
from .base import BodyInfo, Workspace
from .mj import JointArm, MujocoBody

GRIPPER_XML = """    <body name="target" mocap="true" pos="0 0 0.30"/>
    <body name="gripper" pos="0 0 0.30">
      <freejoint/>
      <geom name="palm" type="box" size="0.055 0.014 0.015" rgba="0.18 0.18 0.20 1" mass="0.3"/>
      <geom name="wrist" type="cylinder" pos="0 0 0.04" size="0.02 0.025" rgba="0.55 0.56 0.60 1"
            mass="0.05" contype="0" conaffinity="0"/>
      <site name="tcp" pos="0 0 -0.075" size="0.003" rgba="0 0 0 0"/>
      <body name="finger_left" pos="0 0 -0.015">
        <joint name="finger_left" type="slide" axis="-1 0 0" range="0 0.045" damping="2"/>
        <geom name="finger_left" type="box" pos="-0.006 0 -0.035" size="0.006 0.013 0.035"
              rgba="0.30 0.30 0.33 1" friction="2.0 0.05 0.005" mass="0.03"/>
      </body>
      <body name="finger_right" pos="0 0 -0.015">
        <joint name="finger_right" type="slide" axis="1 0 0" range="0 0.045" damping="2"/>
        <geom name="finger_right" type="box" pos="0.006 0 -0.035" size="0.006 0.013 0.035"
              rgba="0.30 0.30 0.33 1" friction="2.0 0.05 0.005" mass="0.03"/>
      </body>
    </body>

"""

GRIPPER_TAIL = """  <equality>
    <weld body1="target" body2="gripper" solref="0.01 1"/>
  </equality>
  <actuator>
    <position name="finger_left" joint="finger_left" kp="400" ctrlrange="0 0.045" forcerange="-25 25"/>
    <position name="finger_right" joint="finger_right" kp="400" ctrlrange="0 0.045" forcerange="-25 25"/>
  </actuator>
"""

SCENE_XML = """
<mujoco model="rocode_tabletop">
  <option timestep="0.002" cone="elliptic" impratio="10" noslip_iterations="3"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.4 0.4 0.4"/>
  </visual>
  <asset>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.78 0.70 0.58" width="64" height="64"/>
    <material name="table" texture="wood"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.35 0.36 0.38" rgb2="0.30 0.31 0.33"
             width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="6 6"/>
  </asset>
  <default>
    <geom condim="4" friction="1.0 0.02 0.002" solref="0.004 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.5 1.4" dir="-0.2 0.35 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="2 2 0.1" material="floor"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.45 0.35 0.02" material="table"/>

{objects}
{robot}    <camera name="front" pos="0 -0.62 0.50" xyaxes="1 0 0 0 0.628 0.778" fovy="52"/>
    <camera name="side" pos="0.66 0.05 0.42" xyaxes="-0.075 0.997 0 -0.535 -0.040 0.844" fovy="52"/>
  </worldbody>
{robot_tail}</mujoco>
"""

TCP_BELOW_PALM_M = 0.075
MAX_OPENING_M = 0.09
HOME_TCP = np.array([0.0, -0.05, 0.18])
GAP_M = 0.01   # loose objects start at least this far apart


def _f(*vals: float) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def object_xml(o: SceneObject, radians: bool = False) -> str:
    """MJCF for one scene object: a geom on the table if fixed, otherwise a free body."""
    sx, sy, sz = (v / 100 for v in o.size_cm)
    x, y = (v / 100 for v in o.at_cm)
    rgba = _f(*o.color, 1)
    kind = {"box": "box", "sphere": "sphere"}.get(o.shape, "cylinder")
    size = {"box": _f(sx / 2, sy / 2, sz / 2), "sphere": _f(sx / 2)}.get(o.shape, _f(sx / 2, sz / 2))
    if o.fixed:
        return (f'    <geom name="{o.name}" type="{kind}" pos="{_f(x, y, sz / 2)}" size="{size}" rgba="{rgba}"'
                f'{_euler(o, radians)}/>\n')
    out = (f'    <body name="{o.name}" pos="{_f(x, y, sz / 2)}"{_euler(o, radians)}>\n      <freejoint/>\n'
           f'      <geom name="{o.name}" type="{kind}" size="{size}" rgba="{rgba}" mass="{_f(o.mass_kg)}"/>\n')
    if o.shape == "mug":   # a handle on the +x side, sized for the default 6 x 9 cm mug and scaled with height
        out += (f'      <geom name="{o.name}_handle" type="box" pos="{_f(sx / 2 + 0.006, 0, sz * (0.005 / 0.09))}" '
                f'size="{_f(0.008, 0.006, sz * (0.02 / 0.09))}"\n            rgba="{rgba}" mass="0.01"/>\n')
    return out + "    </body>\n"


def _euler(o: SceneObject, radians: bool = False) -> str:
    if not o.yaw_deg:
        return ""
    return f' euler="0 0 {np.deg2rad(o.yaw_deg) if radians else o.yaw_deg:.6g}"'


def scene_xml(scene: Scene, robot: bool = True) -> str:
    """The MJCF for a scene; with robot=False, without the floating gripper (another robot is attached later).
    That version is written in radians: MuJoCo 3.3.0 converts an attached robot's joint ranges with the parent's
    angle unit, so a parent in degrees would shrink them about 57 times."""
    objects = "".join(object_xml(o, radians=not robot) for o in (*scene.fixed, *scene.movable))
    xml = (SCENE_XML.replace("{objects}", objects).replace("{robot}", GRIPPER_XML if robot else "")
           .replace("{robot_tail}", GRIPPER_TAIL if robot else ""))
    if not robot:
        xml = xml.replace('<mujoco model="rocode_tabletop">', '<mujoco model="rocode_tabletop">\n  <compiler angle="radian"/>')
    return xml


class MujocoTabletop(MujocoBody):
    """Implements `rocode.body.base.Body`: a table, the objects of a Scene, two cameras and one of the robots in
    rocode/body/robots.py (the scene's `robot`, a floating gripper by default)."""

    def __init__(self, width: int = 640, height: int = 480, seed: int | None = None, scene: Scene | None = None):
        self.scene = scene or DEFAULT_SCENE
        self.robot = robots.get(self.scene.robot)
        if self.robot.name == "gripper":
            self.model = mujoco.MjModel.from_xml_string(scene_xml(self.scene))
        else:
            self.model = robots.compile_with_robot(scene_xml(self.scene, robot=False), self.robot.name)
        self.data = mujoco.MjData(self.model)
        self._init_render(width, height)
        self.arms = robots.make_arms(self.robot.name, self.model, self.data)
        self._static_geoms = {self.model.geom(n).id for n in ("floor", "table", *(o.name for o in self.scene.fixed))}
        self._static_geoms |= {g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("pedestal")}
        self._info = BodyInfo(
            name="mujoco-tabletop" if self.robot.name == "gripper" else f"mujoco-tabletop-{self.robot.name}",
            cameras=tuple(self.robot.cameras) or ("front", "side"),
            has_depth=True,
            workspace=self.robot.workspace,
            max_opening_m=self.robot.arms[0].max_opening_m,
            arms=self.robot.arms,
            robot=self.robot.description,
        )
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for o, (x, y) in zip(self.scene.movable, self._layout(seed), strict=True):
            qadr = self.model.jnt_qposadr[self.model.body(o.name).jntadr[0]]
            self.data.qpos[qadr : qadr + 2] = [x, y]
        if self.robot.name == "gripper":
            self._reset_gripper()
            return
        for name, arm in self.arms.items():
            arm.home()
            if isinstance(arm, JointArm):
                q, _ = arm.solve(self.robot.home_tcp[name], 0.0)
                arm.home_q = q
                arm.home()
                arm.set_target(self.robot.home_tcp[name], 0.0)
        mujoco.mj_forward(self.model, self.data)
        self._settle(300)

    def _reset_gripper(self) -> None:
        """The original floating-gripper reset, kept step for step so old seeds replay exactly."""
        arm = self.arms["hand"]
        arm.yaw = 0.0
        arm.opening_cmd = MAX_OPENING_M
        self.data.ctrl[:] = MAX_OPENING_M / 2
        palm = HOME_TCP + [0, 0, TCP_BELOW_PALM_M]
        self.data.mocap_pos[arm.mocap] = palm
        gq = self.model.jnt_qposadr[self.model.body("gripper").jntadr[0]]
        self.data.qpos[gq : gq + 3] = palm
        for name in ("finger_left", "finger_right"):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = MAX_OPENING_M / 2
        mujoco.mj_forward(self.model, self.data)
        self._settle(200)

    def ground_truth(self) -> dict:
        """objects: every object's centre (metres, world). tilt_deg: how far each loose object's up axis leans
        from vertical. held: the object in the (first) hand, or None; held_by: each arm's. tcp: the first
        arm's fingertip centre."""
        objects, tilt = {}, {}
        for o in self.scene.objects:
            if o.fixed:
                objects[o.name] = self.data.geom(o.name).xpos.copy()
            else:
                objects[o.name] = self.data.body(o.name).xpos.copy()
                up_z = float(self.data.body(o.name).xmat[8])
                tilt[o.name] = round(float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))), 2)
        held_by = self._held_all()
        first = next(iter(self.arms.values()))
        out = {"objects": objects, "tilt_deg": tilt, "held": next((h for h in held_by.values() if h), None),
               "tcp": first.tcp()}
        if len(self.arms) > 1:
            out["held_by"] = held_by
        return out

    # ---- internals -----------------------------------------------------------------------

    def _layout(self, seed: int | None) -> list[tuple[float, float]]:
        """Starting x, y (metres) of each loose object. A seed moves each one within its jitter, redrawing
        until no two overlap or leave the reachable area."""
        home = [(o.at_cm[0] / 100, o.at_cm[1] / 100) for o in self.scene.movable]
        if seed is None:
            return home
        rng = np.random.default_rng(seed)
        ws = self._info.workspace
        for _ in range(200):
            xy = []
            for o, (x, y) in zip(self.scene.movable, home, strict=True):
                jx, jy = o.jitter_cm[0] / 100, o.jitter_cm[1] / 100
                xy.append((x + rng.uniform(-jx, jx), y + rng.uniform(-jy, jy)))
            if self._layout_ok(xy, ws):
                return xy
        raise ValueError(f"seed {seed}: could not place the objects without overlap; spread them out or "
                         "reduce their jitter_cm")

    def _layout_ok(self, xy, ws: Workspace) -> bool:
        objs = self.scene.movable
        for i, (o, (x, y)) in enumerate(zip(objs, xy, strict=True)):
            r = o.reach_radius_cm / 100
            if not (ws.lo[0] <= x <= ws.hi[0] and ws.lo[1] <= y <= ws.hi[1]):
                return False
            for p, (px, py) in zip(objs[:i], xy[:i], strict=True):
                if np.hypot(x - px, y - py) < r + p.reach_radius_cm / 100 + GAP_M:
                    return False
        return True
