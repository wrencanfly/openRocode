"""Tasks: what is on the table, an instruction for the model, and a success check the model never sees.

A task is defined the same way whether it ships with openRocode or comes from a file:

    Task("mug_to_coaster", "Put the red mug on the yellow coaster, standing upright, and let go of it.",
         success="on(mug, coaster) and upright(mug)")

`success` is an expression over the final state (see rocode/checks.py), a Python function that receives
`body.ground_truth()` and returns a bool, or None for an ungraded task: the episode is recorded and can be
watched, but nothing is scored. `scene` lists the objects (see rocode/scene.py); the default is the built-in
tabletop with a mug, a block and a coaster.

Task files are TOML (one scene, any number of tasks) or Python (a module-level TASKS list):

    [[object]]
    name = "tray"
    shape = "box"
    size_cm = [16, 10, 0.6]
    color = "white"
    at_cm = [-15, -5]
    fixed = true

    [[task]]
    id = "block_to_tray"
    instruction = "Put the blue block on the white tray."
    success = "on(block, tray)"

See docs/tasks.md for the whole format.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checks import Check, CheckError, Clause
from .scene import DEFAULT_SCENE, Scene, make_object

TASK_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
INLINE_ID = "custom"
COASTER_XY = np.array(DEFAULT_SCENE["coaster"].at_cm) / 100   # metres; used by the scripted demo


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    success: str | Callable[[dict], bool] | None = None
    scene: Scene = DEFAULT_SCENE
    source: str = "builtin"                     # builtin, a file path, or "inline" for a typed instruction
    _check: Check | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not TASK_ID.match(self.id):
            raise ValueError(f"task id {self.id!r}: use lowercase letters, digits and _, starting with a letter")
        if not str(self.instruction).strip():
            raise ValueError(f"task {self.id}: the instruction is empty")
        if isinstance(self.success, str):
            try:
                object.__setattr__(self, "_check", Check.parse(self.success, self.scene))
            except CheckError as err:
                raise ValueError(f"task {self.id}: {err}") from None
        elif self.success is not None and not callable(self.success):
            raise ValueError(f"task {self.id}: success must be an expression, a function or None")

    @property
    def graded(self) -> bool:
        return self.success is not None

    def evaluate(self, gt: dict) -> tuple[bool | None, list[Clause]]:
        """The verdict (None if ungraded) and, for an expression, how each part of it came out."""
        if self.success is None:
            return None, []
        if self._check is not None:
            return self._check.evaluate(gt)
        return bool(self.success(gt)), []

    def check(self, gt: dict) -> bool:
        return bool(self.evaluate(gt)[0])

    def spec(self) -> dict:
        """What a trace records, enough to run the task again."""
        success = (self._check.source if self._check is not None else
                   None if self.success is None else f"python:{getattr(self.success, '__qualname__', 'check')}")
        return {"id": self.id, "instruction": self.instruction, "success": success, "source": self.source,
                "scene": self.scene.to_dict()}


BUILTIN: tuple[Task, ...] = (
    Task("lift_block", "Pick up the blue block and hold it at least 10 cm above the table.",
         "held(block) and height_cm(block) > 10"),
    Task("mug_to_coaster", "Put the red mug on the yellow coaster, standing upright, and let go of it.",
         "on(mug, coaster) and upright(mug)"),
    Task("block_to_coaster", "Put the blue block on the yellow coaster and let go of it.", "on(block, coaster)"),
    Task("stack_block_on_mug", "Place the blue block on top of the red mug and let go of it.", "on(block, mug)"),
)
TASKS: dict[str, Task] = {t.id: t for t in BUILTIN}


# ---- task files -------------------------------------------------------------------------------------

OBJECT_KEYS = {"name", "shape", "size_cm", "color", "at_cm", "yaw_deg", "fixed", "mass_g", "jitter_cm"}
TASK_KEYS = {"id", "instruction", "success"}


def load_file(path: str | Path) -> tuple[Scene, list[Task]]:
    """The scene and tasks in a .toml or .py task file. Raises ValueError naming the file and the problem."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{p}: no such task file")
    try:
        if p.suffix == ".toml":
            return _load_toml(p)
        if p.suffix == ".py":
            return _load_python(p)
    except ValueError as err:
        raise ValueError(f"{p}: {err}") from None
    raise ValueError(f"{p}: task files are .toml or .py")


def _load_toml(p: Path) -> tuple[Scene, list[Task]]:
    try:
        data = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"not valid TOML ({err})") from None
    unknown = set(data) - {"object", "task"}
    if unknown:
        raise ValueError(f"unknown section(s) {', '.join(sorted(unknown))}; expected [[object]] and [[task]]")
    objects = []
    for i, o in enumerate(data.get("object", []), start=1):
        bad = set(o) - OBJECT_KEYS
        if bad:
            raise ValueError(f"object {i}: unknown key(s) {', '.join(sorted(bad))}")
        missing = {"name", "shape", "size_cm"} - set(o)
        if missing:
            raise ValueError(f"object {i}: missing {', '.join(sorted(missing))}")
        objects.append(make_object(**o))
    scene = Scene(tuple(objects)) if objects else DEFAULT_SCENE
    tasks = []
    for i, t in enumerate(data.get("task", []), start=1):
        bad = set(t) - TASK_KEYS
        if bad:
            raise ValueError(f"task {i}: unknown key(s) {', '.join(sorted(bad))}")
        if "id" not in t or "instruction" not in t:
            raise ValueError(f"task {i}: needs an id and an instruction")
        tasks.append(Task(t["id"], t["instruction"], t.get("success"), scene, str(p)))
    return scene, tasks


def _load_python(p: Path) -> tuple[Scene, list[Task]]:
    spec = importlib.util.spec_from_file_location(f"rocode_tasks_{p.stem}", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # the user's own file, run like any script they point us at
    tasks = list(getattr(module, "TASKS", []))
    if not all(isinstance(t, Task) for t in tasks):
        raise ValueError("TASKS must be a list of rocode.tasks.Task")
    tasks = [t if t.source != "builtin" else Task(t.id, t.instruction, t.success, t.scene, str(p)) for t in tasks]
    scene = getattr(module, "SCENE", None) or (tasks[0].scene if tasks else DEFAULT_SCENE)
    return scene, tasks


@dataclass
class Catalog:
    """Built-in tasks plus those from task files, looked up by id."""

    tasks: dict[str, Task]
    scenes: list[Scene]

    @classmethod
    def load(cls, files: Iterable[str | Path] = ()) -> Catalog:
        tasks, scenes = dict(TASKS), []
        for f in files:
            scene, found = load_file(f)
            scenes.append(scene)
            for t in found:
                if t.id in tasks:
                    raise ValueError(f"{f}: task id {t.id!r} is already defined ({tasks[t.id].source})")
                tasks[t.id] = t
        return cls(tasks, scenes)

    def resolve(self, name: str) -> Task:
        """A task by id, or, for text with spaces in it, an ungraded task with that instruction. The typed
        instruction runs in the scene of the one task file given, or in the default scene."""
        if name in self.tasks:
            return self.tasks[name]
        if " " in name.strip():
            if len(self.scenes) > 1:
                raise ValueError("a typed instruction needs one scene: pass at most one --task-file")
            return Task(INLINE_ID, name.strip(), None, self.scenes[0] if self.scenes else DEFAULT_SCENE, "inline")
        raise ValueError(f"unknown task {name!r}. Known: {', '.join(sorted(self.tasks))}. "
                         "To give an instruction instead, put it in quotes.")


def get(name: str | Task) -> Task:
    return name if isinstance(name, Task) else Catalog.load().resolve(name)
