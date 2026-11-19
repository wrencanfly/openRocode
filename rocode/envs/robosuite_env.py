"""robosuite (1.5) environments as an openRocode body.

The environment keeps its own scene, randomisation, controller and success check. Each control step, the arm's
commanded fingertip point and yaw become an operational-space action (the controller's own delta mode), so
robosuite's physics and `_check_success()` run exactly as they would for any other policy. Pictures come from
the environment's cameras, rendered with depth and calibration for the pointing tools.

Needs `pip install 'openrocode[robosuite]'` (robosuite pins MuJoCo below 3.4).
"""

from __future__ import annotations

import numpy as np

from . import EnvTask

REQUIRES = ("robosuite",)

TASKS: dict[str, EnvTask] = {
    "lift": EnvTask("Lift", "Pick up the red cube and lift it well above the table."),
    "stack": EnvTask("Stack", "Put the red cube on top of the green cube."),
    "pick_place_can": EnvTask("PickPlaceCan", "Move the red can from the light bin into the compartment of the dark "
                                              "bin that shows a grey outline of a can."),
    "pick_place_milk": EnvTask("PickPlaceMilk", "Move the milk carton from the light bin into the compartment of "
                                                "the dark bin that shows a grey outline of a milk carton."),
    "nut_assembly_square": EnvTask("NutAssemblySquare", "Pick up the brown square nut by its handle and lower it "
                                                        "over the brown square peg so that the peg goes through its "
                                                        "hole."),
}
# Not yet: Door, Wipe and ToolHang need the wrist to pitch, and the pointing tools keep the hand pointing down.

CAMERA_PREFERENCE = ("agentview", "sideview", "frontview", "birdview")
ROBOT = "Panda"


def make_body(task: EnvTask, seed: int | None = None, width: int = 640, height: int = 480,
              depth: bool = True):
    return RobosuiteBody(task.name, seed, width, height, depth)


def _body_class():
    """Defined lazily so that listing tasks does not import MuJoCo or robosuite."""
    import mujoco

    from ..body.base import ArmInfo, BodyInfo, Workspace
    from ..body.mj import ArmDriver, MujocoBody, geoms_of_bodies, subtree_bodies

    class RobosuiteArm(ArmDriver):
        """Holds the commanded pose; the body turns it into an action each control step."""

        def __init__(self, info, model, data, tcp_site, sides, hand_geoms):
            super().__init__(info, model, data, tcp_site, sides, hand_geoms)
            self.target = None
            self.rot0 = data.site_xmat[self.site].reshape(3, 3).copy()   # pointing down, as robosuite starts

        def commanded_tcp(self):
            return self.tcp() if self.target is None else self.target.copy()

        def set_target(self, tcp, yaw=None):
            if yaw is not None:
                self.yaw = yaw
            self.target = np.asarray(tcp, dtype=float).copy()

        def set_opening(self, opening_m):
            self.opening_cmd = float(np.clip(opening_m, 0.0, self.info.max_opening_m))

        def opening(self):
            """The measured gap between the fingers: the nearest pair of opposing finger geoms."""
            return max(0.0, min(float(mujoco.mj_geomDistance(self.model, self.data, a, b, 0.2, None))
                                for a in self.sides[0] for b in self.sides[1]))

        def action(self, base_rot: np.ndarray) -> np.ndarray:
            target = self.commanded_tcp()
            pos = base_rot.T @ (target - self.tcp())
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            want = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.rot0
            err = want @ self.data.site_xmat[self.site].reshape(3, 3).T
            q, rotvec = np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(q, err.flatten())
            mujoco.mju_quat2Vel(rotvec, q, 1.0)
            rot = base_rot.T @ rotvec
            grip = 1.0 - 2.0 * self.opening_cmd / self.info.max_opening_m      # robosuite: -1 open, +1 closed
            return np.concatenate([np.clip(pos / 0.05, -1, 1), np.clip(rot / 0.5, -1, 1), [grip]])

    class RobosuiteBody(MujocoBody):
        """Implements `rocode.body.base.Body` on a robosuite environment."""

        def __init__(self, env_name: str, seed: int | None, width: int, height: int, depth: bool = True):
            import robosuite as suite

            self.env_name, self._size, self._depth_on = env_name, (width, height), depth
            # robosuite draws every random placement from one generator made from this seed
            self.env = suite.make(env_name, robots=ROBOT, has_renderer=False, has_offscreen_renderer=False,
                                  use_camera_obs=False, ignore_done=True, horizon=10**9, control_freq=20,
                                  hard_reset=False, seed=0 if seed is None else seed)
            self.env.reset()
            self.model, self.data = self.env.sim.model._model, self.env.sim.data._data
            self.substeps = max(1, round(self.env.control_timestep / self.env.model_timestep))
            robot = self.env.robots[0]
            gripper = robot.gripper["right"]
            site = gripper.important_sites["grip_site"]
            names = lambda key: [self.model.geom(n).id for n in gripper.important_geoms[key]]
            hand = geoms_of_bodies(self.model, subtree_bodies(self.model, gripper.root_body))
            arm = ArmInfo("arm", "gripper", 0.08)
            self.arms = {"arm": RobosuiteArm(arm, self.model, self.data, site, (names("left_finger"),
                                                                                names("right_finger")), hand)}
            self._base = robot.robot_model.root_body
            self._static_geoms = {g for g in range(self.model.ngeom)
                                  if self.model.body_weldid[self.model.geom_bodyid[g]] == 0}
            cams = [self.model.camera(i).name for i in range(self.model.ncam)]
            self._info = BodyInfo(
                name=f"robosuite-{env_name}",
                cameras=tuple(c for c in CAMERA_PREFERENCE if c in cams)[:2],
                has_depth=depth,
                workspace=self._workspace(),
                max_opening_m=arm.max_opening_m,
                arms=(arm,),
                robot=f"a Franka Panda arm with a parallel gripper (robosuite {env_name})",
            )
            self._init_render(width, height, depth)

        def _workspace(self) -> Workspace:
            env = self.env
            if getattr(env, "table_full_size", None) is not None and getattr(env, "table_offset", None) is not None:
                (cx, cy, top), (sx, sy, _) = env.table_offset, env.table_full_size
                return Workspace((cx - sx / 2, cy - sy / 2, top), (cx + sx / 2, cy + sy / 2, top + 0.45), top)
            if hasattr(env, "bin1_pos") and hasattr(env, "bin2_pos"):
                lo = np.minimum(env.bin1_pos, env.bin2_pos) - [0.2, 0.25, 0]
                hi = np.maximum(env.bin1_pos, env.bin2_pos) + [0.2, 0.25, 0]
                return Workspace((lo[0], lo[1], lo[2]), (hi[0], hi[1], lo[2] + 0.45), float(lo[2]))
            tcp = self.arms["arm"].tcp()
            return Workspace((tcp[0] - 0.3, tcp[1] - 0.4, tcp[2] - 0.3), (tcp[0] + 0.3, tcp[1] + 0.4, tcp[2] + 0.3),
                             float(tcp[2] - 0.3))

        def reset(self, seed: int | None = None) -> None:
            """A fresh environment made with this seed, so a seed always gives the same layout."""
            self.close()
            self.__init__(self.env_name, seed, *self._size, self._depth_on)

        def _step(self) -> None:
            base_rot = self.data.body(self._base).xmat.reshape(3, 3)
            self.env.step(self.arms["arm"].action(base_rot))

        def ground_truth(self) -> dict:
            objects, tilt = {}, {}
            for j in range(self.model.njnt):
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    b = self.model.jnt_bodyid[j]
                    name = self.model.body(b).name.removesuffix("_main")
                    objects[name] = self.data.xpos[b].copy()
                    tilt[name] = round(float(np.degrees(np.arccos(np.clip(self.data.xmat[b][8], -1, 1)))), 2)
            held = self._held_body(self.arms["arm"])
            return {"objects": objects, "tilt_deg": tilt, "held": held and held.removesuffix("_main"),
                    "tcp": self.arms["arm"].tcp(), "success": bool(self.env._check_success())}

        def close(self) -> None:
            super().close()
            env, self.env = getattr(self, "env", None), None
            if env is not None:
                env.close()

    return RobosuiteBody


def RobosuiteBody(env_name: str, seed: int | None = None, width: int = 640, height: int = 480, depth: bool = True):
    return _body_class()(env_name, seed, width, height, depth)
