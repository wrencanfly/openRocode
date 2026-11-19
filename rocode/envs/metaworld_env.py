"""Meta-World (v3) tasks as an openRocode body.

Meta-World's Sawyer hand is a mocap-driven gripper that always points down and cannot turn. openRocode drives
the mocap target and the finger actuators directly, step by step, instead of through `env.step`, so an episode
is not cut off at Meta-World's 500-step horizon; success is still Meta-World's own `evaluate_state`. The
built-in corner cameras render upside down, so two of them are re-aimed: `front` faces the robot across the
table, `side` looks from its right.

Needs `pip install 'openrocode[metaworld]'` (Meta-World pins MuJoCo below 3.4).
"""

from __future__ import annotations

import numpy as np

from . import EnvTask

REQUIRES = ("metaworld", "gymnasium")

TASKS: dict[str, EnvTask] = {
    "reach-v3": EnvTask("reach-v3", "Move the gripper so that its fingertips touch the small red ball."),
    "push-v3": EnvTask("push-v3", "Push the dark red puck across the table to the small green ball."),
    "pick-place-v3": EnvTask("pick-place-v3", "Pick up the dark red puck and hold it at the small blue ball."),
    "button-press-topdown-v3": EnvTask("button-press-topdown-v3", "Press the red button on top of the yellow box "
                                                                  "all the way down."),
    "drawer-open-v3": EnvTask("drawer-open-v3", "Pull the green drawer open by its handle."),
    "drawer-close-v3": EnvTask("drawer-close-v3", "Push the open green drawer closed."),
    "door-open-v3": EnvTask("door-open-v3", "Open the door of the black safe by pulling its handle."),
    "window-open-v3": EnvTask("window-open-v3", "Slide the window open by pushing its handle sideways."),
    "faucet-open-v3": EnvTask("faucet-open-v3", "Turn the faucet on: swing its red handle round to the small green "
                                                "ball."),
    "coffee-button-v3": EnvTask("coffee-button-v3", "Press the button on the front of the coffee machine."),
}

CAMERAS = {   # shown name -> (model camera to re-aim, position, look-at point, fovy)
    "front": ("corner", (0.0, 1.35, 0.55), (0.0, 0.65, 0.05), 50),
    "side": ("corner2", (0.80, 0.55, 0.45), (0.0, 0.68, 0.05), 50),
}


def make_body(task: EnvTask, seed: int | None = None, width: int = 640, height: int = 480):
    return _body_class()(task.name, seed, width, height)


def _body_class():
    """Defined lazily so that listing tasks does not import MuJoCo or Meta-World."""
    import mujoco

    from ..body.base import ArmInfo, BodyInfo, Workspace
    from ..body.mj import ArmDriver, MujocoBody, geoms_of_bodies, place_camera

    class SawyerHand(ArmDriver):
        """Meta-World's hand: the mocap target follows commands; fingers are two actuators driven ±1."""

        def __init__(self, info, env, sides, hand_geoms):
            u = env.unwrapped
            super().__init__(info, u.model, u.data, "endEffector", sides, hand_geoms)
            self.u = u
            self.offset = u.data.mocap_pos[0] - u.tcp_center   # mocap target minus fingertip centre
            self.target = None

        def tcp(self):
            return np.asarray(self.u.tcp_center, dtype=float).copy()

        def commanded_tcp(self):
            return self.tcp() if self.target is None else self.target.copy()

        def set_target(self, tcp, yaw=None):
            self.target = np.asarray(tcp, dtype=float).copy()
            mocap = np.clip(self.target + self.offset, self.u.mocap_low, self.u.mocap_high)
            self.data.mocap_pos[0] = mocap

        def set_opening(self, opening_m):
            self.opening_cmd = float(np.clip(opening_m, 0.0, self.info.max_opening_m))
            g = 1.0 - 2.0 * self.opening_cmd / self.info.max_opening_m       # +1 closes
            self.data.ctrl[:2] = [g, -g]

        def opening(self):
            return max(0.0, min(float(mujoco.mj_geomDistance(self.model, self.data, a, b, 0.2, None))
                                for a in self.sides[0] for b in self.sides[1]))

    class MetaWorldBody(MujocoBody):
        """Implements `rocode.body.base.Body` on a Meta-World task."""

        def __init__(self, env_name: str, seed: int | None, width: int, height: int):
            import gymnasium as gym
            import metaworld  # noqa: F401  (registers the Meta-World environments)

            self.env_name, self._size = env_name, (width, height)
            seed = 0 if seed is None else seed
            # Meta-World only repeats a layout for the first reset of an env made with that seed
            self.env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
            u = self.env.unwrapped
            u.max_path_length = 10**9
            self.env.reset(seed=seed)
            self.model, self.data = u.model, u.data
            m = self.model
            for internal, pos, target, fovy in CAMERAS.values():
                place_camera(m, internal, pos, target, fovy)
            self.camera_alias = {shown: spec[0] for shown, spec in CAMERAS.items()}
            mujoco.mj_forward(m, self.data)
            sides = (geoms_of_bodies(m, ["leftpad", "leftclaw"]), geoms_of_bodies(m, ["rightpad", "rightclaw"]))
            hand = geoms_of_bodies(m, ["hand", "leftpad", "leftclaw", "rightpad", "rightclaw"])
            arm = ArmInfo("arm", "gripper", 0.09, turns=False)
            self.arms = {"arm": SawyerHand(arm, self.env, sides, hand)}
            self._static_geoms = {g for g in range(m.ngeom) if m.body_weldid[m.geom_bodyid[g]] == 0}
            lo, hi = u.mocap_low - self.arms["arm"].offset, u.mocap_high - self.arms["arm"].offset
            self._info = BodyInfo(
                name=f"metaworld-{env_name}",
                cameras=tuple(CAMERAS),
                has_depth=True,
                workspace=Workspace((lo[0], lo[1], 0.0), (hi[0], hi[1], hi[2]), 0.0),
                max_opening_m=arm.max_opening_m,
                arms=(arm,),
                robot=f"a Sawyer arm whose gripper always points straight down and cannot turn (Meta-World "
                      f"{env_name})",
            )
            self._init_render(width, height)
            self._settle_open()

        def _settle_open(self):
            a = self.arms["arm"]
            a.set_opening(a.info.max_opening_m)
            a.set_target(a.tcp())
            self._settle(100)

        def reset(self, seed: int | None = None) -> None:
            """A fresh environment made with this seed: Meta-World's later resets do not repeat layouts."""
            self.close()
            self.__init__(self.env_name, seed, *self._size)

        def ground_truth(self) -> dict:
            u = self.env.unwrapped
            objects, tilt = {}, {}
            for j in range(self.model.njnt):
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    b = self.model.jnt_bodyid[j]
                    name = self.model.body(b).name
                    objects[name] = self.data.xpos[b].copy()
                    tilt[name] = round(float(np.degrees(np.arccos(np.clip(self.data.xmat[b][8], -1, 1)))), 2)
            _, info = u.evaluate_state(u._get_obs(), np.zeros(4))
            return {"objects": objects, "tilt_deg": tilt, "held": self._held_body(self.arms["arm"]),
                    "tcp": self.arms["arm"].tcp(), "success": bool(info.get("success", 0) > 0),
                    "goal": np.asarray(u._target_pos, dtype=float).copy()}

        def close(self) -> None:
            super().close()
            env, self.env = getattr(self, "env", None), None
            if env is not None:
                env.close()

    return MetaWorldBody
