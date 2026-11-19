"""The body contract: everything above this line is robot-agnostic.

A body owns sensing and motion for one robot (simulated or real). Coordinates here are world metres;
they never reach the model. Tools turn the model's pixels and view-relative directions into these calls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np


class CancelToken:
    """Set from any thread (e-stop, abort). Motion loops must check it every control period."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class CameraModel:
    """Pinhole model of one camera image. Pixels are (x right, y down)."""

    name: str
    width: int
    height: int
    focal_px: float
    position: np.ndarray  # camera centre, world
    rotation: np.ndarray  # world-from-camera; columns are camera x (right), y (up), z (backward)

    def deproject(self, x: float, y: float, depth_m: float) -> np.ndarray:
        """World point seen at pixel (x, y), `depth_m` along the optical axis."""
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        p_cam = np.array([(x - cx) * depth_m / self.focal_px, -(y - cy) * depth_m / self.focal_px, -depth_m])
        return self.position + self.rotation @ p_cam

    def project(self, point) -> tuple[float, float] | None:
        """Pixel (x, y) of a world point, or None if it is behind the camera."""
        p_cam = self.rotation.T @ (np.asarray(point, dtype=float) - self.position)
        depth = -p_cam[2]
        if depth <= 1e-6:
            return None
        cx, cy = (self.width - 1) / 2, (self.height - 1) / 2
        return (cx + self.focal_px * p_cam[0] / depth, cy - self.focal_px * p_cam[1] / depth)

    def horizontal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors on the table plane for 'right in this image' and 'away from this camera'."""
        right = self.rotation[:, 0].copy()
        away = -self.rotation[:, 2].copy()
        right[2] = away[2] = 0
        return right / np.linalg.norm(right), away / np.linalg.norm(away)


@dataclass(frozen=True)
class ArmState:
    tcp_xyz: np.ndarray        # fingertip centre, world metres
    yaw: float                 # hand rotation about vertical, radians
    opening_m: float           # measured gap between the finger pads (a hand: between thumb and fingers)
    opening_cmd_m: float       # commanded gap
    wrist_force_n: float       # contact force on the hand and anything it holds


@dataclass(frozen=True)
class Proprio:
    """The first arm's state as flat fields, and every arm's by name in `arms`."""

    tcp_xyz: np.ndarray
    yaw: float
    opening_m: float
    opening_cmd_m: float
    wrist_force_n: float
    arms: dict[str, ArmState] = field(default_factory=dict)
    base_xy_yaw: tuple[float, float, float] | None = None   # mobile bodies: where the base is, world frame

    @classmethod
    def of(cls, arms: dict[str, ArmState], base_xy_yaw=None) -> Proprio:
        first = next(iter(arms.values()))
        return cls(first.tcp_xyz, first.yaw, first.opening_m, first.opening_cmd_m, first.wrist_force_n, dict(arms),
                   base_xy_yaw)

    def arm(self, name: str | None = None) -> ArmState:
        if name is None or not self.arms:
            return ArmState(self.tcp_xyz, self.yaw, self.opening_m, self.opening_cmd_m, self.wrist_force_n)
        return self.arms[name]


@dataclass(frozen=True)
class Observation:
    """One synchronized snapshot. Immutable; tools refer to it by id after it is stored."""

    t: float
    rgb: dict[str, np.ndarray]                 # camera -> (H, W, 3) uint8
    depth: dict[str, np.ndarray] | None        # camera -> (H, W) metres along the optical axis
    cameras: dict[str, CameraModel]
    proprio: Proprio


@dataclass(frozen=True)
class Workspace:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    table_z: float = 0.0

    def contains(self, p) -> bool:
        return all(lo <= v <= hi for v, lo, hi in zip(p, self.lo, self.hi, strict=True))


@dataclass(frozen=True)
class ArmInfo:
    """One arm and what is on the end of it. Names are what the model calls them ("left", "right")."""

    name: str
    hand: Literal["gripper", "hand"] = "gripper"   # a parallel gripper, or a multi-fingered hand
    max_opening_m: float = 0.08


@dataclass(frozen=True)
class BaseInfo:
    """A body that can move itself around: wheels or legs."""

    kind: Literal["wheeled", "legged"]
    max_step_m: float = 1.0          # longest single drive command
    area: tuple[tuple[float, float], tuple[float, float]] = ((-3.0, -3.0), (3.0, 3.0))   # where it may go, x/y


@dataclass(frozen=True)
class BodyInfo:
    name: str
    cameras: tuple[str, ...]
    has_depth: bool
    workspace: Workspace
    max_opening_m: float
    arms: tuple[ArmInfo, ...] = ()           # empty: one arm called "hand" with max_opening_m
    base: BaseInfo | None = None
    robot: str = ""                           # what the model is told it controls, e.g. "a Franka Panda arm"
    surface: Literal["table", "floor"] = "table"

    def __post_init__(self) -> None:
        if not self.arms:
            object.__setattr__(self, "arms", (ArmInfo("hand", "gripper", self.max_opening_m),))

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms)

    def arm(self, name: str | None = None) -> ArmInfo:
        if name is None:
            return self.arms[0]
        for a in self.arms:
            if a.name == name:
                return a
        raise KeyError(f"no arm called {name!r}; this body has {', '.join(self.arm_names)}")


@dataclass(frozen=True)
class Guard:
    """Limits applied to one motion. Force is measured relative to the start of the motion."""

    max_force_n: float | None = 4.0
    speed_mps: float = 0.18


StopReason = Literal["reached", "contact", "blocked", "refused", "cancelled", "fault"]


@dataclass(frozen=True)
class Receipt:
    """What a motion actually did. `executed` is never guessed: unknown means unknown."""

    executed: Literal["none", "partial", "full", "unknown"]
    stop_reason: StopReason
    stop_confirmed: bool
    peak_force_n: float = 0.0
    detail: str = ""
    extra: dict = field(default_factory=dict)


class Body(Protocol):
    """Call every method except `stop` from the thread that created the body: simulators keep
    renderer state (an OpenGL context for MuJoCo) on that thread. `stop` may be called from any thread."""

    def info(self) -> BodyInfo: ...
    def observe(self) -> Observation: ...
    def move(self, goal_xyz, guard: Guard, cancel: CancelToken, arm: str | None = None) -> Receipt: ...
    def rotate(self, yaw: float, cancel: CancelToken, arm: str | None = None) -> Receipt: ...
    def grip(self, opening_m: float, cancel: CancelToken, arm: str | None = None) -> Receipt: ...
    def drive(self, forward_m: float, left_m: float, turn_rad: float, cancel: CancelToken) -> Receipt:
        """Mobile bodies only: move the base, in its own frame. Arms keep their pose relative to the base."""
        ...
    def stop(self) -> None: ...
    def reset(self, seed: int | None = None) -> None: ...
    def ground_truth(self) -> dict:
        """For scoring only. Nothing that talks to the model may call this."""
        ...
    def close(self) -> None:
        """Release renderers, connections and the like, from the creating thread. Safe to call twice.
        Whoever creates a body closes it."""
        ...
