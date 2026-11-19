"""Benchmark environments as openRocode bodies: their scenes, physics and success checks; our tools and traces.

A task id with a colon names an environment's task, e.g. `robosuite:lift` or `metaworld:pick-place-v3`. The
environment builds the scene and decides success (`env_success()` in the success expression); the model
drives it through the same pointing and code tools as any other body, and every episode is traced the same way.

Each environment is an optional install:

    pip install 'openrocode[robosuite]'    # robosuite 1.5: Lift, Stack, PickPlace, NutAssembly, Door, ...
    pip install 'openrocode[metaworld]'    # Meta-World v3: 50 tabletop tasks with a Sawyer gripper
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

ENVIRONMENTS = {
    "robosuite": "rocode.envs.robosuite_env",
    "metaworld": "rocode.envs.metaworld_env",
}


@dataclass(frozen=True)
class EnvTask:
    name: str            # the environment's own name for it, e.g. "Lift"
    instruction: str
    success: str = "env_success()"


def module(env: str):
    """The adapter module, after checking that the simulator it wraps is installed."""
    if env not in ENVIRONMENTS:
        raise ValueError(f"unknown environment {env!r}; known: {', '.join(ENVIRONMENTS)}")
    mod = importlib.import_module(ENVIRONMENTS[env])
    missing = [name for name in mod.REQUIRES if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(f"{env} needs {', '.join(missing)}, which is not installed: pip install 'openrocode[{env}]'")
    return mod


def tasks(env: str) -> dict[str, EnvTask]:
    """The environment's tasks by short id (the part after the colon). Does not import the simulator."""
    return importlib.import_module(ENVIRONMENTS[env]).TASKS if env in ENVIRONMENTS else {}


def make_body(task_id: str, seed: int | None, width: int, height: int, depth: bool = True):
    env, short = task_id.split(":", 1)
    return module(env).make_body(tasks(env)[short], seed=seed, width=width, height=height, depth=depth)
