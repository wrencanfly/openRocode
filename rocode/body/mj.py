"""Shared machinery for bodies simulated in MuJoCo: cameras, guarded motion, grasp and force sensing.

A `MujocoBody` owns a model and data (built by openRocode, or a benchmark's own) and one `ArmDriver` per arm.
A driver turns a task-space command (fingertip point, yaw about vertical, finger opening) into actuator
commands: a mocap weld for a floating hand, inverse kinematics for a jointed arm, or a benchmark's action
vector. The body runs the motion loops, watches contact force and writes receipts, the same way for all of
them, so every robot honours the Body contract identically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import mujoco
import numpy as np

from .base import (
    ArmInfo,
    ArmState,
    BodyInfo,
    CameraModel,
    CancelToken,
    Guard,
    Observation,
    Proprio,
    Receipt,
)

REACHED_M = 0.01


class ArmDriver:
    """One arm in a MuJoCo model. Subclasses implement how a command reaches the actuators."""

    def __init__(self, info: ArmInfo, model: mujoco.MjModel, data: mujoco.MjData, tcp_site: str,
                 sides: tuple[Iterable[int], Iterable[int]], hand_geoms: Iterable[int]):
        self.info, self.model, self.data = info, model, data
        self.site = model.site(tcp_site).id
        self.sides = (set(sides[0]), set(sides[1]))   # geoms that oppose each other when grasping
        self.hand_geoms = set(hand_geoms)             # everything that is part of the hand
        self.yaw = 0.0
        self.opening_cmd = info.max_opening_m
        self.grip_steps = 400          # physics steps a grip command runs for

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    # -- what subclasses provide -------------------------------------------------------------

    def commanded_tcp(self) -> np.ndarray:
        raise NotImplementedError

    def set_target(self, tcp, yaw: float | None = None) -> None:
        """Command the fingertip point (world metres) and, if given, the yaw."""
        raise NotImplementedError

    def set_opening(self, opening_m: float) -> None:
        raise NotImplementedError

    def opening(self) -> float:
        """Measured gap between the opposing sides."""
        raise NotImplementedError

    def before_step(self) -> None:
        """Called before every physics step, e.g. to refresh inverse kinematics."""

    def reachable(self, tcp) -> bool:
        """Whether the hand can get its fingertips to this point, pointing down."""
        return True

    def home(self) -> None:
        """Put the arm in its starting pose (qpos written directly; the body settles afterwards)."""


class MocapArm(ArmDriver):
    """A hand welded to a mocap body: commands are exact poses, with no arm to limit them."""

    def __init__(self, info: ArmInfo, model, data, tcp_site: str, sides, hand_geoms, mocap_body: str,
                 tcp_below_m: float, finger_actuators: Iterable[str], finger_scale: float, finger_joints: Iterable[str],
                 home_tcp, base_quat=(1.0, 0.0, 0.0, 0.0)):
        super().__init__(info, model, data, tcp_site, sides, hand_geoms)
        self.mocap = model.body(mocap_body).mocapid[0]
        self.offset = np.array([0.0, 0.0, tcp_below_m])
        self.fingers = [model.actuator(a).id for a in finger_actuators]
        self.finger_scale = finger_scale
        self.finger_qadr = [model.jnt_qposadr[model.joint(j).id] for j in finger_joints]
        self.home_tcp = np.asarray(home_tcp, dtype=float)
        self.base_quat = np.asarray(base_quat, dtype=float)

    def commanded_tcp(self) -> np.ndarray:
        return self.data.mocap_pos[self.mocap] - self.offset

    def set_target(self, tcp, yaw: float | None = None) -> None:
        if yaw is not None:
            self.yaw = yaw
        self.data.mocap_pos[self.mocap] = np.asarray(tcp) + self.offset
        q = np.array([np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)])
        if self.base_quat[0] != 1.0:
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, self.base_quat)
            q = out
        self.data.mocap_quat[self.mocap] = q

    def set_opening(self, opening_m: float) -> None:
        self.opening_cmd = float(np.clip(opening_m, 0.0, self.info.max_opening_m))
        for a in self.fingers:
            self.data.ctrl[a] = self.opening_cmd * self.finger_scale

    def opening(self) -> float:
        return float(sum(self.data.qpos[a] for a in self.finger_qadr))


class JointArm(ArmDriver):
    """A jointed arm with position-controlled joints. Each physics step runs one damped-least-squares IK step on a
    kinematic copy of the model toward the commanded fingertip pose (pointing straight down, rotated by yaw),
    and the joint actuators track the result."""

    def __init__(self, info: ArmInfo, model, data, tcp_site: str, sides, hand_geoms, joints: Iterable[str],
                 actuators: Iterable[str], home_q, grip, finger_axis: str = "y", down_axis: str = "z",
                 down_sign: float = 1.0, orient_weight: float = 0.35, yaw_offset: float = 0.0):
        super().__init__(info, model, data, tcp_site, sides, hand_geoms)
        self.joints = [model.joint(j).id for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.lo, self.hi = model.jnt_range[self.joints, 0], model.jnt_range[self.joints, 1]
        self.act = [model.actuator(a).id for a in actuators]
        self.home_q = np.asarray(home_q, dtype=float)
        self.grip = grip                      # (set_opening(data, m), measure(data) -> m)
        axes = {"x": 0, "y": 1, "z": 2}
        self.finger_col, self.down_col, self.down_sign = axes[finger_axis], axes[down_axis], down_sign
        self.yaw_offset = yaw_offset          # so that yaw 0 is a comfortable wrist angle for this arm
        self.w = orient_weight
        self.scratch = mujoco.MjData(model)
        self.q_des = self.home_q.copy()
        self.target = None
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def home(self) -> None:
        self.q_des = self.home_q.copy()
        self.data.qpos[self.qadr] = self.q_des
        self.data.ctrl[self.act] = self.q_des
        self.target = None
        self.set_opening(self.info.max_opening_m)

    def commanded_tcp(self) -> np.ndarray:
        if self.target is None:
            self.target = self._fk(self.q_des)
        return self.target.copy()

    def set_target(self, tcp, yaw: float | None = None) -> None:
        if yaw is not None:
            self.yaw = yaw
        self.target = np.asarray(tcp, dtype=float).copy()

    def set_opening(self, opening_m: float) -> None:
        self.opening_cmd = float(np.clip(opening_m, 0.0, self.info.max_opening_m))
        self.grip[0](self.data, self.opening_cmd)

    def opening(self) -> float:
        return float(self.grip[1](self.data))

    def desired_rotation(self) -> np.ndarray:
        """Hand frame columns: the axis the fingers close along is world x rotated by yaw, and the approach axis
        (wrist toward fingertips, times approach_sign) points straight down."""
        c, s = np.cos(self.yaw + self.yaw_offset), np.sin(self.yaw + self.yaw_offset)
        cols: list = [None, None, None]
        cols[self.finger_col] = np.array([c, s, 0.0])
        cols[self.down_col] = np.array([0.0, 0.0, -self.down_sign])
        third = 3 - self.finger_col - self.down_col
        cols[third] = np.cross(cols[(third + 1) % 3], cols[(third + 2) % 3])   # keep it right-handed
        return np.column_stack(cols)

    def before_step(self) -> None:
        if self.target is None:
            return
        self.q_des = self._ik_step(self.q_des, self.target, self.desired_rotation())
        self.data.ctrl[self.act] = self.q_des

    def reachable(self, tcp) -> bool:
        """Tried from the current posture, then from home: IK from far away can stall in a local minimum."""
        return any(self.solve(tcp, self.yaw, iters=150, q0=q0)[1] < REACHED_M / 2 for q0 in (self.q_des, self.home_q))

    def solve(self, tcp, yaw: float, iters: int = 200, q0=None) -> tuple[np.ndarray, float]:
        """Joint angles for a pose and the remaining position error, without moving anything."""
        old = self.yaw
        self.yaw = yaw
        rot = self.desired_rotation()
        self.yaw = old
        q = self.home_q.copy() if q0 is None else np.asarray(q0, dtype=float)
        for _ in range(iters):
            q = self._ik_step(q, np.asarray(tcp, dtype=float), rot)
        return q, float(np.linalg.norm(self._fk(q) - tcp))

    # -- kinematics on the scratch copy -------------------------------------------------------

    def _fk(self, q) -> np.ndarray:
        d = self.scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site].copy()

    def _ik_step(self, q, target, rot_des) -> np.ndarray:
        m, d = self.model, self.scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pos_err = target - d.site_xpos[self.site]
        r = d.site_xmat[self.site].reshape(3, 3)
        rot_err = 0.5 * sum(np.cross(r[:, i], rot_des[:, i]) for i in range(3))
        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site)
        jac = np.vstack([self._jacp[:, self.dadr], self.w * self._jacr[:, self.dadr]])
        err = np.concatenate([np.clip(pos_err, -0.02, 0.02), self.w * np.clip(rot_err, -0.2, 0.2)])
        lam = 0.02
        jjt = jac @ jac.T + lam**2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)
        # pull gently toward the home posture inside the null space, to stay away from joint limits
        pinv = jac.T @ np.linalg.solve(jjt, np.eye(6))
        dq += (np.eye(len(q)) - pinv @ jac) @ (0.02 * (self.home_q - q))
        return np.clip(q + dq, self.lo, self.hi)


class MujocoBody:
    """Implements `rocode.body.base.Body` on a MuJoCo model. Subclasses set up `arms`, `_info`, the static geoms
    and the objects ground truth reports, and may override `_step` and `reset`."""

    model: mujoco.MjModel
    data: mujoco.MjData
    arms: dict[str, ArmDriver]
    _info: BodyInfo
    _static_geoms: set[int]
    substeps: int = 1          # physics steps per `_step`: a benchmark's env.step may advance several
    camera_alias: Mapping[str, str] = MappingProxyType({})   # name shown -> the camera's name in the model

    def _ticks(self, physics_steps: int) -> int:
        """How many `_step` calls cover this many physics steps."""
        return physics_steps if self.substeps == 1 else max(1, round(physics_steps / self.substeps))

    def _init_render(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        g = self.model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, width), max(g.offheight, height)
        self._rgb = mujoco.Renderer(self.model, height, width)
        self._depth = mujoco.Renderer(self.model, height, width)
        self._depth.enable_depth_rendering()

    # ---- Body ----------------------------------------------------------------------------

    def info(self) -> BodyInfo:
        return self._info

    def close(self) -> None:
        """Free the renderers now, on this thread. Left to the garbage collector, a renderer can be freed on
        whatever thread happens to collect it (a dashboard request, say), and on macOS that deadlocks."""
        for name in ("_rgb", "_depth"):
            renderer = getattr(self, name, None)
            if renderer is not None:
                renderer.close()
                setattr(self, name, None)

    def observe(self) -> Observation:
        rgb, depth, cams = {}, {}, {}
        for name in self._info.cameras:
            internal = self.camera_alias.get(name, name)
            self._rgb.update_scene(self.data, camera=internal)
            self._depth.update_scene(self.data, camera=internal)
            rgb[name] = self._rgb.render().copy()
            depth[name] = self._depth.render().copy()
            cams[name] = self._camera(name)
        return Observation(t=float(self.data.time), rgb=rgb, depth=depth, cameras=cams, proprio=self._proprio())

    def move(self, goal_xyz, guard: Guard, cancel: CancelToken, arm: str | None = None, max_s: float = 6.0) -> Receipt:
        a = self._arm(arm)
        goal = np.asarray(goal_xyz, dtype=float)
        if not self._info.workspace.contains(goal):
            return Receipt("none", "refused", True, detail="goal is outside the workspace")
        if not a.reachable(goal):
            who = f"the {a.info.name} arm" if len(self.arms) > 1 else "the arm"
            return Receipt("none", "refused", True, detail=f"{who} cannot reach that point")
        start = a.commanded_tcp()
        dist = float(np.linalg.norm(goal - start))
        direction = (goal - start) / dist if dist > 1e-9 else np.zeros(3)
        dt = self.model.opt.timestep * self.substeps
        n = max(1, int(np.ceil(dist / (guard.speed_mps * dt))))
        baseline = self._external_force(a)  # tare, like a wrist F/T sensor
        peak = 0.0
        for k in range(1, min(n, int(max_s / dt)) + 1):
            if cancel.cancelled:
                self.stop()
                return Receipt("partial", "cancelled", True, peak, cancel.reason)
            s = k / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            a.set_target(start + (goal - start) * s)
            self._step()
            if guard.max_force_n is not None:
                f = self._external_force(a) - baseline
                peak = max(peak, f)
                if f > guard.max_force_n:
                    touched = self._touching(a)
                    a.set_target(a.tcp() - direction * 0.004)  # back off what we hit
                    self._settle(80)
                    # names are for traces and debugging; tool results never show them to the model
                    return Receipt("partial", "contact", True, peak, extra={"touched": touched})
        self._settle(150)
        reached = float(np.linalg.norm(a.tcp() - goal)) < REACHED_M
        return Receipt("full" if reached else "partial", "reached" if reached else "blocked", True, peak)

    def rotate(self, yaw: float, cancel: CancelToken, arm: str | None = None) -> Receipt:
        a = self._arm(arm)
        if not a.info.turns:
            return Receipt("none", "refused", True, detail="this hand's wrist does not turn")
        start, n = a.yaw, self._ticks(250)
        for k in range(1, n + 1):
            if cancel.cancelled:
                self.stop()
                return Receipt("partial", "cancelled", True, detail=cancel.reason)
            a.set_target(a.commanded_tcp(), start + (yaw - start) * k / n)
            self._step()
        self._settle(100)
        return Receipt("full", "reached", True)

    def grip(self, opening_m: float, cancel: CancelToken, arm: str | None = None) -> Receipt:
        a = self._arm(arm)
        a.set_opening(opening_m)
        for _ in range(self._ticks(a.grip_steps)):
            if cancel.cancelled:
                return Receipt("partial", "cancelled", True, detail=cancel.reason)
            self._step()
        return Receipt("full", "reached", True)

    def drive(self, forward_m: float, left_m: float, turn_rad: float, cancel: CancelToken) -> Receipt:
        return Receipt("none", "refused", True, detail="this robot has no mobile base")

    def stop(self) -> None:
        """Hold where we are. Safe to call from another thread between physics steps."""
        for a in self.arms.values():
            a.set_target(a.tcp())

    # ---- shared internals ----------------------------------------------------------------

    def _arm(self, name: str | None) -> ArmDriver:
        if name is None:
            return next(iter(self.arms.values()))
        if name not in self.arms:
            raise KeyError(f"no arm called {name!r}; this robot has {', '.join(self.arms)}")
        return self.arms[name]

    def _step(self) -> None:
        for a in self.arms.values():
            a.before_step()
        mujoco.mj_step(self.model, self.data)

    def _settle(self, steps: int) -> None:
        for _ in range(self._ticks(steps)):
            self._step()

    def _camera(self, name: str) -> CameraModel:
        cam = self.model.camera(self.camera_alias.get(name, name))
        fovy = np.deg2rad(self.model.cam_fovy[cam.id])
        return CameraModel(name, self.width, self.height, (self.height / 2) / np.tan(fovy / 2),
                           self.data.cam_xpos[cam.id].copy(), self.data.cam_xmat[cam.id].reshape(3, 3).copy())

    def _proprio(self) -> Proprio:
        states = {name: ArmState(a.tcp(), a.yaw, self._opening(a), a.opening_cmd, self._external_force(a))
                  for name, a in self.arms.items()}
        return Proprio.of(states, self._base_pose())

    def _opening(self, a: ArmDriver) -> float:
        """A gripper's measured gap. A closed hand's fingers rarely meet, so its width only counts when touch
        on both sides of the grasp says it is holding something (tactile sensing; no object identity)."""
        if a.info.hand == "hand" and a.opening_cmd < a.info.max_opening_m:
            return a.opening() if self._held_geom(a) is not None else 0.0
        return a.opening()

    def _base_pose(self) -> tuple[float, float, float] | None:
        return None

    def _held_geom(self, a: ArmDriver) -> int | None:
        touching: list[set[int]] = [set(), set()]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for side, geoms in enumerate(a.sides):
                if c.geom1 in geoms or c.geom2 in geoms:
                    other = c.geom2 if c.geom1 in geoms else c.geom1
                    if other not in a.hand_geoms and other not in self._static_geoms:
                        touching[side].add(other)
        both = touching[0] & touching[1]
        if not both:   # a hand can hold an object with different geoms of it on each side
            bodies = [{self.model.geom_bodyid[g] for g in t} for t in touching]
            common = bodies[0] & bodies[1]
            if common:
                body = min(common)
                return min(g for g in touching[0] if self.model.geom_bodyid[g] == body)
        return min(both) if both else None

    def _held_body(self, a: ArmDriver) -> str | None:
        g = self._held_geom(a)
        return None if g is None else self.model.body(self.model.geom_bodyid[g]).name

    def _external_force(self, a: ArmDriver) -> float:
        held = self._held_geom(a)
        carried = set(a.hand_geoms)
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

    def _touching(self, a: ArmDriver) -> list[str]:
        """Bodies outside this hand (and what it holds) that it is pressing on."""
        held = self._held_geom(a)
        carried = set(a.hand_geoms)
        if held is not None:
            body = self.model.geom_bodyid[held]
            carried |= {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body}
        out = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in carried) != (c.geom2 in carried):
                other = c.geom2 if c.geom1 in carried else c.geom1
                out.add(self.model.body(self.model.geom_bodyid[other]).name or self.model.geom(other).name)
        return sorted(out)

    def _held_all(self) -> dict[str, str | None]:
        return {name: self._held_body(a) for name, a in self.arms.items()}


def place_camera(model: mujoco.MjModel, name: str, pos, target, fovy: float | None = None) -> None:
    """Move an existing fixed camera of a compiled model to look from pos at target, image up toward world +z."""
    cam = model.camera(name).id
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rot = np.column_stack([right, up, -f])            # camera x right, y up, z backward
    world_from_parent = model.body(model.cam_bodyid[cam]).id
    if world_from_parent != 0:
        raise ValueError(f"camera {name} is attached to a moving body")
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    model.cam_pos[cam], model.cam_quat[cam] = pos, q
    model.cam_mode[cam] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    if fovy is not None:
        model.cam_fovy[cam] = fovy


def geoms_of_bodies(model: mujoco.MjModel, bodies: Iterable[str], colliding_only: bool = True) -> set[int]:
    """Geoms attached to the named bodies (and, with colliding_only, only those that take part in contacts)."""
    ids = {model.body(b).id for b in bodies}
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in ids
            and (not colliding_only or model.geom_contype[g] or model.geom_conaffinity[g])}


def subtree_bodies(model: mujoco.MjModel, root: str) -> list[str]:
    rid = model.body(root).id
    out = []
    for b in range(model.nbody):
        p = b
        while p not in (0, rid):
            p = model.body_parentid[p]
        if p == rid:
            out.append(model.body(b).name)
    return out
