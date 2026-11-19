"""Success conditions written as small expressions over the simulator's final state.

    on(mug, coaster) and upright(mug)
    height_cm(block) > 10 and held(block)
    on(red_block, tray) and on(blue_block, tray) and not on(green_block, tray)

Names are scene objects (or `table` for `on`). Lengths are centimetres, angles degrees. The expression is
parsed once, when the task is loaded, so a typo fails before any model is called. Evaluating it returns the
verdict and, for each part joined by `and`, whether it held and the values it read, so a failed episode
says which condition failed and by how much.

Predicates (all read the state at the end of the episode):

    on(a, b)             a rests on b: its centre is over b's footprint, above b's top and below b's top plus
                         a's own height, and the gripper is not holding it. b may be `table`.
    held(a)              the gripper is holding a.
    hand_empty()         the gripper is holding nothing.
    upright(a[, deg])    a's up axis leans at most deg from vertical (default 15).
    height_cm(a)         height of a's centre above the table top.
    distance_cm(a, b)    horizontal distance between the centres of a and b.
    x_cm(a), y_cm(a)     a's centre: x to the right as seen from the front camera, y away from it.
    tilt_deg(a)          how far a's up axis leans from vertical.

Combine them with and, or, not, comparisons (<, <=, >, >=) and numbers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np

from .scene import Scene

UPRIGHT_DEG = 15.0
TABLE = "table"

# name -> (object arguments, optional number arguments)
SIGNATURES = {
    "on": (2, 0), "held": (1, 0), "hand_empty": (0, 0), "upright": (1, 1), "height_cm": (1, 0),
    "distance_cm": (2, 0), "x_cm": (1, 0), "y_cm": (1, 0), "tilt_deg": (1, 0),
}
COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}


class CheckError(ValueError):
    pass


@dataclass
class Clause:
    expr: str
    ok: bool
    values: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"expr": self.expr, "ok": self.ok, "values": self.values}


@dataclass(frozen=True)
class Check:
    """A parsed success expression, bound to the scene whose objects it names."""

    source: str
    scene: Scene
    tree: ast.Expression

    @classmethod
    def parse(cls, source: str, scene: Scene) -> Check:
        text = " ".join(str(source).split())
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as err:
            raise CheckError(f"success {text!r}: not a valid expression ({err.msg})") from None
        _validate(tree.body, text, set(scene.names()))
        return cls(text, scene, tree)

    def clauses_of(self) -> list[ast.expr]:
        body = self.tree.body
        return list(body.values) if isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And) else [body]

    def evaluate(self, gt: dict) -> tuple[bool, list[Clause]]:
        """The verdict and one Clause per top-level `and` part."""
        clauses = []
        for node in self.clauses_of():
            values: dict = {}
            ok = bool(_eval(node, gt, self.scene, values, self.source))
            clauses.append(Clause(ast.get_source_segment(self.source, node) or ast.unparse(node), ok, values))
        return all(c.ok for c in clauses), clauses


NUMERIC = {"height_cm", "distance_cm", "x_cm", "y_cm", "tilt_deg"}


def _validate(node: ast.AST, src: str, names: set[str], in_compare: bool = False) -> None:
    def fail(msg: str):
        raise CheckError(f"success {src!r}: {msg}")

    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _validate(v, src, names)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _validate(node.operand, src, names)
    elif isinstance(node, ast.Compare):
        if not all(type(op) in COMPARE for op in node.ops):
            fail("only <, <=, > and >= compare numbers")
        for v in (node.left, *node.comparators):
            if isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool):
                continue
            if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                continue
            if isinstance(v, ast.Call) and _fname(v) in NUMERIC:
                _validate(v, src, names, in_compare=True)
                continue
            fail(f"{ast.unparse(v)!r} is not a number or a measurement such as height_cm(block)")
    elif isinstance(node, ast.Call):
        name = _fname(node)
        if name not in SIGNATURES:
            fail(f"unknown predicate {name or ast.unparse(node.func)!r}; use one of {', '.join(SIGNATURES)}")
        if name in NUMERIC and not in_compare:
            fail(f"{name} is a measurement; compare it with a number, e.g. {name}(...) > 10")
        n_obj, n_num = SIGNATURES[name]
        if node.keywords or not n_obj <= len(node.args) <= n_obj + n_num:
            fail(f"{name} takes {n_obj} object name(s)" + (f" and up to {n_num} number(s)" if n_num else ""))
        for i, a in enumerate(node.args):
            if i < n_obj:
                if not isinstance(a, ast.Name):
                    fail(f"{name}: {ast.unparse(a)!r} should be an object name")
                allowed = names | ({TABLE} if name == "on" and i == 1 else set())
                if a.id not in allowed:
                    fail(f"no object named {a.id!r} in the scene (objects: {', '.join(sorted(names))})")
            elif not (isinstance(a, ast.Constant) and isinstance(a.value, (int, float))):
                fail(f"{name}: {ast.unparse(a)!r} should be a number")
    else:
        fail(f"{ast.unparse(node)!r} is not a predicate; use and, or, not, comparisons and the predicates")


def _fname(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def _eval(node: ast.AST, gt: dict, scene: Scene, values: dict, src: str):
    if isinstance(node, ast.BoolOp):
        results = [_eval(v, gt, scene, values, src) for v in node.values]   # all of them, so values are complete
        return all(results) if isinstance(node.op, ast.And) else any(results)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, gt, scene, values, src)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, gt, scene, values, src)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        left = _eval(node.left, gt, scene, values, src)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval(right_node, gt, scene, values, src)
            if not COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    assert isinstance(node, ast.Call)
    name = _fname(node)
    args = [a.id if isinstance(a, ast.Name) else a.value for a in node.args]
    value = PREDICATES[name](gt, scene, *args)
    values[ast.get_source_segment(src, node) or ast.unparse(node)] = (
        round(float(value), 2) if isinstance(value, (float, np.floating)) else bool(value))
    return value


def _pos_cm(gt: dict, name: str) -> np.ndarray:
    return np.asarray(gt["objects"][name], dtype=float) * 100


def _on(gt: dict, scene: Scene, a: str, b: str) -> bool:
    pa = _pos_cm(gt, a)
    if gt.get("held") == a:
        return False
    if b == TABLE:
        return bool(pa[2] < scene[a].height_cm / 2 + 1.0)   # resting on the table top, give or take a centimetre
    pb = _pos_cm(gt, b)
    top = pb[2] + scene[b].height_cm / 2
    over = np.hypot(*(pa[:2] - pb[:2])) <= scene[b].footprint_radius_cm
    return bool(over and top < pa[2] < top + scene[a].height_cm)


PREDICATES = {
    "on": _on,
    "held": lambda gt, scene, a: gt.get("held") == a,
    "hand_empty": lambda gt, scene: gt.get("held") is None,
    "upright": lambda gt, scene, a, deg=UPRIGHT_DEG: (gt.get("tilt_deg") or {}).get(a, 0.0) <= deg,
    "height_cm": lambda gt, scene, a: float(_pos_cm(gt, a)[2]),
    "distance_cm": lambda gt, scene, a, b: float(np.hypot(*(_pos_cm(gt, a)[:2] - _pos_cm(gt, b)[:2]))),
    "x_cm": lambda gt, scene, a: float(_pos_cm(gt, a)[0]),
    "y_cm": lambda gt, scene, a: float(_pos_cm(gt, a)[1]),
    "tilt_deg": lambda gt, scene, a: float((gt.get("tilt_deg") or {}).get(a, 0.0)),
}
assert set(PREDICATES) == set(SIGNATURES)
