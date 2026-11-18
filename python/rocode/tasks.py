"""Tasks: an instruction for the model and a ground-truth check the model never sees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

COASTER_XY = np.array([-0.14, -0.08])


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    check: Callable[[dict], bool]   # receives body.ground_truth()


def _on_coaster(gt: dict, name: str, tol: float = 0.035) -> bool:
    p = gt["objects"][name]
    return bool(np.linalg.norm(p[:2] - COASTER_XY) < tol and p[2] < 0.08 and gt["held"] != name)


def _block_on_mug(gt: dict) -> bool:
    mug, block = gt["objects"]["mug"], gt["objects"]["block"]
    return bool(np.linalg.norm(block[:2] - mug[:2]) < 0.03 and block[2] > mug[2] + 0.04 and gt["held"] != "block")


def _block_lifted(gt: dict) -> bool:
    return bool(gt["objects"]["block"][2] > 0.10 and gt["held"] == "block")


TASKS: dict[str, Task] = {t.id: t for t in [
    Task("lift_block", "Pick up the blue block and hold it at least 10 cm above the table.", _block_lifted),
    Task("mug_to_coaster", "Put the red mug on the yellow coaster, standing upright, and let go of it.",
         lambda gt: _on_coaster(gt, "mug")),
    Task("block_to_coaster", "Put the blue block on the yellow coaster and let go of it.",
         lambda gt: _on_coaster(gt, "block")),
    Task("stack_block_on_mug", "Place the blue block on top of the red mug and let go of it.", _block_on_mug),
]}
