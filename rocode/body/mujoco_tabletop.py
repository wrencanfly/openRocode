"""MuJoCo tabletop body: a floating parallel gripper, a mug, a block and a coaster, two cameras.

The gripper is a free body welded to a mocap target, so motion is commanded in task space without an
arm model. Reachability and joint limits of a real arm are therefore not modelled.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .base import BodyInfo, CameraModel, CancelToken, Guard, Observation, Proprio, Receipt, Workspace

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

    <geom name="coaster" type="cylinder" pos="-0.14 -0.08 0.003" size="0.05 0.003" rgba="0.93 0.74 0.20 1"/>

    <body name="mug" pos="0.10 0.10 0.045">
      <freejoint/>
      <geom name="mug" type="cylinder" size="0.03 0.045" rgba="0.80 0.15 0.12 1" mass="0.15"/>
      <geom name="mug_handle" type="box" pos="0.036 0 0.005" size="0.008 0.006 0.02"
            rgba="0.80 0.15 0.12 1" mass="0.01"/>
    </body>

    <body name="block" pos="0.12 -0.12 0.02" euler="0 0 25">
      <freejoint/>
      <geom name="block" type="box" size="0.02 0.02 0.02" rgba="0.15 0.35 0.80 1" mass="0.05"/>
    </body>

    <body name="target" mocap="true" pos="0 0 0.30"/>
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

    <camera name="front" pos="0 -0.62 0.50" xyaxes="1 0 0 0 0.628 0.778" fovy="52"/>
    <camera name="side" pos="0.66 0.05 0.42" xyaxes="-0.075 0.997 0 -0.535 -0.040 0.844" fovy="52"/>
  </worldbody>
  <equality>
    <weld body1="target" body2="gripper" solref="0.01 1"/>
  </equality>
  <actuator>
    <position name="finger_left" joint="finger_left" kp="400" ctrlrange="0 0.045" forcerange="-25 25"/>
    <position name="finger_right" joint="finger_right" kp="400" ctrlrange="0 0.045" forcerange="-25 25"/>
  </actuator>
</mujoco>
"""

TCP_BELOW_PALM_M = 0.075
MAX_OPENING_M = 0.09
HOME_TCP = np.array([0.0, -0.05, 0.18])
OBJECTS = ("mug", "block")
INITIAL_XY = {"mug": (0.10, 0.10), "block": (0.12, -0.12)}


class MujocoTabletop:
    """Implements `rocode.body.base.Body`."""

    def __init__(self, width: int = 640, height: int = 480, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self._rgb = mujoco.Renderer(self.model, height, width)
        self._depth = mujoco.Renderer(self.model, height, width)
        self._depth.enable_depth_rendering()
        self._target = self.model.body("target").mocapid[0]
        self._gripper_geoms = {self.model.geom(n).id for n in ("palm", "finger_left", "finger_right")}
        self._finger_geoms = (self.model.geom("finger_left").id, self.model.geom("finger_right").id)
        self._static_geoms = {self.model.geom(n).id for n in ("floor", "table", "coaster")}
        self._info = BodyInfo(
            name="mujoco-tabletop",
            cameras=("front", "side"),
            has_depth=True,
            workspace=Workspace(lo=(-0.40, -0.30, 0.0), hi=(0.40, 0.30, 0.40), table_z=0.0),
            max_opening_m=MAX_OPENING_M,
        )
        self.reset(seed)

    # ---- Body ----------------------------------------------------------------------------

    def info(self) -> BodyInfo:
        return self._info

    def reset(self, seed: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed) if seed is not None else None
        for name, (x0, y0) in INITIAL_XY.items():
            jitter = (rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03)) if rng else (0.0, 0.0)
            qadr = self.model.jnt_qposadr[self.model.body(name).jntadr[0]]
            self.data.qpos[qadr : qadr + 2] = [x0 + jitter[0], y0 + jitter[1]]
        self._yaw = 0.0
        self._opening_cmd = MAX_OPENING_M
        self.data.ctrl[:] = MAX_OPENING_M / 2
        palm = HOME_TCP + [0, 0, TCP_BELOW_PALM_M]
        self.data.mocap_pos[self._target] = palm
        gq = self.model.jnt_qposadr[self.model.body("gripper").jntadr[0]]
        self.data.qpos[gq : gq + 3] = palm
        for name in ("finger_left", "finger_right"):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = MAX_OPENING_M / 2
        mujoco.mj_forward(self.model, self.data)
        self._settle(200)

    def observe(self) -> Observation:
        rgb, depth, cams = {}, {}, {}
        for name in self._info.cameras:
            self._rgb.update_scene(self.data, camera=name)
            self._depth.update_scene(self.data, camera=name)
            rgb[name] = self._rgb.render().copy()
            depth[name] = self._depth.render().copy()
            cams[name] = self._camera(name)
        return Observation(t=float(self.data.time), rgb=rgb, depth=depth, cameras=cams, proprio=self._proprio())

    def move(self, goal_xyz, guard: Guard, cancel: CancelToken, max_s: float = 6.0) -> Receipt:
        goal = np.asarray(goal_xyz, dtype=float)
        if not self._info.workspace.contains(goal):
            return Receipt("none", "refused", True, detail="goal is outside the workspace")
        start = self._commanded_tcp()
        dist = float(np.linalg.norm(goal - start))
        direction = (goal - start) / dist if dist > 1e-9 else np.zeros(3)
        dt = self.model.opt.timestep
        n = max(1, int(np.ceil(dist / (guard.speed_mps * dt))))
        baseline = self._external_force()  # tare, like a wrist F/T sensor
        peak = 0.0
        for k in range(1, min(n, int(max_s / dt)) + 1):
            if cancel.cancelled:
                self.stop()
                return Receipt("partial", "cancelled", True, peak, cancel.reason)
            s = k / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            self._set_target(start + (goal - start) * s)
            mujoco.mj_step(self.model, self.data)
            if guard.max_force_n is not None:
                f = self._external_force() - baseline
                peak = max(peak, f)
                if f > guard.max_force_n:
                    self._set_target(self._tcp() - direction * 0.004)  # back off what we hit
                    self._settle(80)
                    return Receipt("partial", "contact", True, peak)
        self._settle(150)
        reached = float(np.linalg.norm(self._tcp() - goal)) < 0.01
        return Receipt("full" if reached else "partial", "reached" if reached else "blocked", True, peak)

    def rotate(self, yaw: float, cancel: CancelToken) -> Receipt:
        start, n = self._yaw, 250
        for k in range(1, n + 1):
            if cancel.cancelled:
                self.stop()
                return Receipt("partial", "cancelled", True, detail=cancel.reason)
            self._yaw = start + (yaw - start) * k / n
            self._set_target(self._commanded_tcp())
            mujoco.mj_step(self.model, self.data)
        self._settle(100)
        return Receipt("full", "reached", True)

    def grip(self, opening_m: float, cancel: CancelToken) -> Receipt:
        self._opening_cmd = float(np.clip(opening_m, 0.0, MAX_OPENING_M))
        self.data.ctrl[:] = self._opening_cmd / 2
        for _ in range(400):
            if cancel.cancelled:
                return Receipt("partial", "cancelled", True, detail=cancel.reason)
            mujoco.mj_step(self.model, self.data)
        return Receipt("full", "reached", True)

    def stop(self) -> None:
        """Hold where we are. Safe to call from another thread between physics steps."""
        self._set_target(self._tcp())

    def ground_truth(self) -> dict:
        return {
            "objects": {n: self.data.body(n).xpos.copy() for n in OBJECTS},
            "held": self._held_body(),
            "tcp": self._tcp(),
        }

    # ---- internals -----------------------------------------------------------------------

    def _camera(self, name: str) -> CameraModel:
        cam = self.model.camera(name)
        fovy = np.deg2rad(self.model.cam_fovy[cam.id])
        return CameraModel(name, self.width, self.height, (self.height / 2) / np.tan(fovy / 2),
                           self.data.cam_xpos[cam.id].copy(), self.data.cam_xmat[cam.id].reshape(3, 3).copy())

    def _proprio(self) -> Proprio:
        q = sum(self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]]
                for n in ("finger_left", "finger_right"))
        return Proprio(self._tcp(), self._yaw, float(q), self._opening_cmd, self._external_force())

    def _tcp(self) -> np.ndarray:
        return self.data.site("tcp").xpos.copy()

    def _commanded_tcp(self) -> np.ndarray:
        return self.data.mocap_pos[self._target] - [0, 0, TCP_BELOW_PALM_M]

    def _set_target(self, tcp) -> None:
        self.data.mocap_pos[self._target] = np.asarray(tcp) + [0, 0, TCP_BELOW_PALM_M]
        self.data.mocap_quat[self._target] = [np.cos(self._yaw / 2), 0, 0, np.sin(self._yaw / 2)]

    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def _held_geom(self) -> int | None:
        touching: list[set[int]] = [set(), set()]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for side, finger in enumerate(self._finger_geoms):
                if finger in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == finger else c.geom1
                    if other not in self._gripper_geoms and other not in self._static_geoms:
                        touching[side].add(other)
        both = touching[0] & touching[1]
        return min(both) if both else None

    def _held_body(self) -> str | None:
        g = self._held_geom()
        return None if g is None else self.model.body(self.model.geom_bodyid[g]).name

    def _external_force(self) -> float:
        held = self._held_geom()
        carried = set(self._gripper_geoms)
        if held is not None:
            body = self.model.geom_bodyid[held]
            carried |= {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body}
        total, f6 = 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in carried) != (c.geom2 in carried):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                total += abs(f6[0])
        return total
