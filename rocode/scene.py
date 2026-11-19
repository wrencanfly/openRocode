"""What is on the table: a list of objects, each with a shape, size, colour and place.

A scene is plain data, so a task file can describe one and a trace can record it. Lengths are centimetres in the
world frame: x to the right as seen from the front camera, y away from it, the origin at the table's centre.
The table top spans x -45..45 and y -35..35; the robot reaches x -40..40 and y -30..30.

Fixed objects (a coaster, a tray, a pad) are part of the table: they never move, and the gripper cannot pick
them up. Every other object is loose and, with a seed, starts a little off its place (`jitter_cm`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SHAPES = ("box", "cylinder", "sphere", "mug")
COLORS = {
    "red": (0.80, 0.15, 0.12), "blue": (0.15, 0.35, 0.80), "yellow": (0.93, 0.74, 0.20),
    "green": (0.20, 0.60, 0.28), "orange": (0.95, 0.50, 0.12), "purple": (0.50, 0.30, 0.70),
    "pink": (0.95, 0.55, 0.70), "brown": (0.50, 0.32, 0.18), "white": (0.92, 0.92, 0.90),
    "black": (0.10, 0.10, 0.11), "gray": (0.62, 0.63, 0.65),
}
RESERVED = {"floor", "table", "target", "gripper", "palm", "wrist", "tcp", "finger_left", "finger_right"}
NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
TABLE_HALF_CM = (45.0, 35.0)
DEFAULT_JITTER_CM = (4.0, 3.0)


@dataclass(frozen=True)
class SceneObject:
    name: str
    shape: str
    size_cm: tuple[float, float, float]       # full extents x, y, z; cylinders, mugs and spheres are round in x, y
    color: tuple[float, float, float]
    at_cm: tuple[float, float]
    yaw_deg: float = 0.0
    fixed: bool = False
    mass_g: float | None = None               # None: from the volume, at 0.6 g/cm³
    jitter_cm: tuple[float, float] = DEFAULT_JITTER_CM

    @property
    def height_cm(self) -> float:
        return self.size_cm[2]

    @property
    def footprint_radius_cm(self) -> float:
        """Half the narrowest horizontal extent: how far from the centre something on top still rests on it."""
        return min(self.size_cm[0], self.size_cm[1]) / 2

    @property
    def reach_radius_cm(self) -> float:
        """Radius of a circle that holds the whole object seen from above (a mug's handle included)."""
        x, y = self.size_cm[0] / 2, self.size_cm[1] / 2
        if self.shape == "box":
            return (x * x + y * y) ** 0.5
        return x + (1.4 if self.shape == "mug" else 0.0)

    @property
    def mass_kg(self) -> float:
        if self.mass_g is not None:
            return self.mass_g / 1000
        x, y, z = self.size_cm
        volume = {"box": x * y * z, "sphere": 3.14159 / 6 * x**3}.get(self.shape, 3.14159 / 4 * x * x * z)
        return round(volume * 0.6 / 1000, 4)

    def to_dict(self) -> dict:
        d = {"name": self.name, "shape": self.shape, "size_cm": list(self.size_cm), "color": list(self.color),
             "at_cm": list(self.at_cm), "yaw_deg": self.yaw_deg, "fixed": self.fixed, "mass_g": self.mass_g,
             "jitter_cm": list(self.jitter_cm)}
        return d


@dataclass(frozen=True)
class Scene:
    objects: tuple[SceneObject, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for o in self.objects:
            for n in (o.name, f"{o.name}_handle"):
                if n in seen or n in RESERVED:
                    raise ValueError(f"object name {o.name!r} is used twice or is reserved")
            seen |= {o.name, f"{o.name}_handle"}
            hx, hy = TABLE_HALF_CM
            r = o.reach_radius_cm
            if abs(o.at_cm[0]) + r > hx or abs(o.at_cm[1]) + r > hy:
                raise ValueError(f"{o.name} at {list(o.at_cm)} cm does not fit on the table "
                                 f"(x within ±{hx:g}, y within ±{hy:g}, edges included)")

    def __getitem__(self, name: str) -> SceneObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def names(self) -> list[str]:
        return [o.name for o in self.objects]

    @property
    def movable(self) -> tuple[SceneObject, ...]:
        return tuple(o for o in self.objects if not o.fixed)

    @property
    def fixed(self) -> tuple[SceneObject, ...]:
        return tuple(o for o in self.objects if o.fixed)

    def to_dict(self) -> dict:
        return {"objects": [o.to_dict() for o in self.objects]}

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        return cls(tuple(make_object(**o) for o in d.get("objects", [])))


def make_object(name: str, shape: str, size_cm, color="gray", at_cm=(0, 0), yaw_deg: float = 0.0,
                fixed: bool = False, mass_g: float | None = None, jitter_cm=None) -> SceneObject:
    """Build an object from loose input, as a task file writes it. Raises ValueError with a readable message.

    size_cm: box, a number (a cube) or [x, y, z]; cylinder and mug, [diameter, height]; sphere, a diameter.
    color: a name from COLORS or [r, g, b] in 0..1.
    """
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"object name {name!r}: use lowercase letters, digits and _, starting with a letter")
    if shape not in SHAPES:
        raise ValueError(f"{name}: shape {shape!r} is not one of {', '.join(SHAPES)}")
    size = _size(name, shape, size_cm)
    if isinstance(color, str):
        if color not in COLORS:
            raise ValueError(f"{name}: color {color!r} is not one of {', '.join(COLORS)} (or give [r, g, b])")
        rgb = COLORS[color]
    else:
        rgb = tuple(float(c) for c in color)
        if len(rgb) != 3 or not all(0 <= c <= 1 for c in rgb):
            raise ValueError(f"{name}: color must be a name or [r, g, b] with each in 0..1")
    at = tuple(float(v) for v in at_cm)
    if len(at) != 2:
        raise ValueError(f"{name}: at_cm must be [x, y]")
    jitter = (0.0, 0.0) if fixed and jitter_cm is None else tuple(float(v) for v in (jitter_cm or DEFAULT_JITTER_CM))
    if len(jitter) != 2 or min(jitter) < 0:
        raise ValueError(f"{name}: jitter_cm must be [x, y], not negative")
    if fixed and max(jitter) > 0:
        raise ValueError(f"{name}: a fixed object does not move, so it cannot have jitter")
    if fixed and shape == "mug":
        raise ValueError(f"{name}: a mug cannot be fixed")
    if not fixed and size[2] > 15:
        raise ValueError(f"{name}: loose objects can be at most 15 cm tall (the gripper starts 18 cm up)")
    if mass_g is not None and not 1 <= float(mass_g) <= 2000:
        raise ValueError(f"{name}: mass_g must be between 1 and 2000")
    return SceneObject(name, shape, size, rgb, at, float(yaw_deg), bool(fixed),
                       None if mass_g is None else float(mass_g), jitter)


def _size(name: str, shape: str, size_cm) -> tuple[float, float, float]:
    vals = [float(size_cm)] if isinstance(size_cm, (int, float)) else [float(v) for v in size_cm]
    if shape == "box" and len(vals) in (1, 3):
        size = (vals[0],) * 3 if len(vals) == 1 else tuple(vals)
    elif shape in ("cylinder", "mug") and len(vals) == 2:
        size = (vals[0], vals[0], vals[1])
    elif shape == "sphere" and len(vals) == 1:
        size = (vals[0],) * 3
    else:
        want = {"box": "a number or [x, y, z]", "cylinder": "[diameter, height]", "mug": "[diameter, height]",
                "sphere": "a diameter"}[shape]
        raise ValueError(f"{name}: size_cm for a {shape} is {want}")
    if not all(0.4 <= v <= 40 for v in size):
        raise ValueError(f"{name}: sizes must be between 0.4 and 40 cm")
    return size  # type: ignore[return-value]


# The tabletop the built-in tasks use. Changing it changes every built-in task and old seeds' layouts.
DEFAULT_SCENE = Scene((
    make_object("mug", "mug", [6, 9], "red", [10, 10], mass_g=150),
    make_object("block", "box", 4, "blue", [12, -12], yaw_deg=25, mass_g=50),
    make_object("coaster", "cylinder", [10, 0.6], "yellow", [-14, -8], fixed=True),
))
