"""A scripted episode that exercises the whole harness without a model.

The script looks up object positions in ground truth and projects them into the latest pictures,
which a real model must instead do by looking. It exists to check an installation and for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .body.mujoco_tabletop import MujocoTabletop
from .core.events import Event
from .providers.base import Request
from .providers.scripted import ScriptedProvider
from .session import SessionConfig, SessionReport, run_session
from .tasks import COASTER_XY


def scripted_demo(out_dir: Path | None, subscribers: list[Callable[[Event], None]] = (),
                  seed: int | None = None) -> SessionReport:
    """Mug to coaster: one direct pointing step, then a program for the rest, then done."""
    body = MujocoTabletop(seed=seed)
    cfg = SessionConfig(task="mug_to_coaster", provider="scripted", policy="both", seed=seed, out_dir=out_dir)
    try:
        return run_session(cfg, provider=scripted_mug_to_coaster(body), subscribers=list(subscribers), body=body)
    finally:
        body.close()


def scripted_mug_to_coaster(body: MujocoTabletop) -> ScriptedProvider:
    """A provider that solves mug_to_coaster on this body by reading ground truth instead of looking."""

    def pixel(request: Request, point, camera: str = "front") -> tuple[int, int, str]:
        stored = request.images.latest()
        x, y = stored.observation.cameras[camera].project(np.asarray(point, dtype=float))
        return round(x), round(y), stored.id

    def hover_over_mug(request: Request):
        x, y, oid = pixel(request, body.ground_truth()["objects"]["mug"] + [0, 0, 0.045])
        return [("move_to", {"camera": "front", "x": x, "y": y, "observation": oid})]

    def pick_and_place_program(request: Request):
        cx, cy, oid = pixel(request, [*COASTER_XY, 0.006])
        code = f"""
robot.turn(90)                  # close across the mug's body, not its handle
r = robot.lower()
print("touched:", r.touched, "at", r.height_cm, "cm")
r = robot.grip("close")
if not r.holding:
    raise SystemExit("missed the mug")
robot.nudge("front", up_cm=12)
robot.move_to("front", {cx}, {cy}, hover_cm=10, observation="{oid}")
robot.lower()
robot.grip("open")
r = robot.nudge("side", up_cm=10)
print("finished at", r.height_cm, "cm, pictures", r.observation)
"""
        return [("run_code", {"code": code})]

    return ScriptedProvider([
        hover_over_mug,
        pick_and_place_program,
        [("done", {"success": True, "summary": "The mug is on the coaster."})],
    ])


def scripted_pick_and_place(body, moves: list[tuple[str, str]]) -> ScriptedProvider:
    """A provider that moves each object onto another, in order, by reading ground truth instead of looking.

    It shows that a task file's goal is reachable with this gripper; a model has to do the same by looking.
    """
    scene = body.scene
    # several objects going onto one target are set down side by side, not on top of each other
    slots: dict[tuple[str, str], float] = {}
    for target in {b for _, b in moves}:
        objs = [a for a, b in moves if b == target and scene[target].fixed]
        width = max((scene[a].reach_radius_cm * 2 + 1 for a in objs), default=0) / 100
        for i, a in enumerate(objs):
            slots[(a, target)] = (i - (len(objs) - 1) / 2) * width

    def step(obj: str, onto: str):
        def program(request: Request):
            gt = body.ground_truth()
            top = lambda name: gt["objects"][name] + [0, 0, scene[name].height_cm / 200]  # noqa: E731
            stored = request.images.latest()
            cams = stored.observation.cameras
            other = next(c for c in cams if c != "front")

            def click(point):
                """Pixel arguments for a point: one click with depth, the same spot in two cameras without."""
                x, y = (round(v) for v in cams["front"].project(point))
                if stored.observation.depth is not None:
                    return f'"front", {x}, {y}'
                x2, y2 = (round(v) for v in cams[other].project(point))
                return f'"front", {x}, {y}, camera2="{other}", x2={x2}, y2={y2}'

            pick = click(top(obj))
            place = click(top(onto) + [slots.get((obj, onto), 0.0), 0, 0])
            arm, away = "", ""
            if len(body.arms) > 1:   # an arm that reaches both ends of the move, the nearer one if both do
                ends = [top(obj) + [0, 0, 0.06], top(onto) + [0, 0, scene[obj].height_cm / 100 + 0.03]]
                able = [n for n, a in body.arms.items() if all(a.reachable(p) for p in ends)] or list(body.arms)
                name = min(able, key=lambda n: np.linalg.norm(body.arms[n].tcp()[:2] - gt["objects"][obj][:2]))
                arm = f", arm={name!r}"
                # then step back over to this arm's own end, out of the other arm's way
                # (5 cm past home: ALOHA wrist-camera mounts reach about 10 cm toward the other arm)
                home_x = body.robot.home_tcp[name][0]
                back = (home_x + np.sign(home_x) * 0.05 - top(onto)[0] - slots.get((obj, onto), 0.0)) * 100
                steps = [float(np.clip(back, -20, 20)), float(back - np.clip(back, -20, 20))]
                away = "".join(f"robot.nudge(\"front\", right_cm={d:.1f}, up_cm={5 if i == 0 else 0}{arm})\n"
                               for i, d in enumerate(steps) if abs(d) > 0.5 or i == 0)
            turn = f"robot.turn(90{arm})\n" if scene[obj].shape == "mug" else ""
            code = f"""
{turn}robot.move_to({pick}, hover_cm=6, observation="{stored.id}"{arm})
r = robot.lower({arm.lstrip(', ')})
robot.grip("close"{arm})
r = robot.nudge("front", up_cm=12{arm})
if not r.holding:
    raise SystemExit("missed {obj}")
robot.move_to({place}, hover_cm={scene[obj].height_cm + 3:g}, observation="{stored.id}"{arm})
robot.lower({arm.lstrip(', ')})
robot.grip("open"{arm})
robot.nudge("front", up_cm=10{arm})
{away}"""
            return [("run_code", {"code": code})]
        return program

    return ScriptedProvider([*(step(a, b) for a, b in moves),
                             [("done", {"success": True, "summary": "Moved everything."})]])
