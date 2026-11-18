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

    provider = ScriptedProvider([
        hover_over_mug,
        pick_and_place_program,
        [("done", {"success": True, "summary": "The mug is on the coaster."})],
    ])
    cfg = SessionConfig(task="mug_to_coaster", provider="scripted", policy="both", out_dir=out_dir)
    return run_session(cfg, provider=provider, subscribers=list(subscribers), body=body)
