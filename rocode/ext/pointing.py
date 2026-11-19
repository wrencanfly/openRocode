"""Pointing tools: robot control as 3D computer use.

The model clicks pixels in a specific stored observation and nudges in directions it can see. All
geometry (depth, calibration, world frame) stays here. Results never contain world coordinates.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ..body.base import Guard, Receipt
from ..core.extension import Extension
from ..core.messages import ImageRef, Text
from ..core.observations import StoredObservation
from ..core.tooling import ToolContext, ToolResult

TRANSIT_CLEARANCE_M = 0.06
EMPTY_GRIP_M = 0.004

PROMPT = """\
Pointing tools. Every picture is labelled with its observation id (o1, o2, ...). Pixel ticks run along \
the edges every 50 px, labelled every 100 px; x grows to the right and y downward. The gripper's \
fingertip point is a cyan circle, with a dashed cyan line straight down to a cyan ellipse on the table \
directly below it: compare that ellipse with an object to judge whether the gripper is over it. A \
magenta cross marks the last point you clicked with move_to. Use more than one camera to judge \
alignment and height. Clicks refer to the pictures of one observation (the latest by default), so the \
gripper moving afterwards does not change what you clicked. A typical grasp: move_to the top of the \
object, check alignment in every view, turn so the fingers close across the object's narrow side, \
lower until contact, grip close, then nudge up before travelling."""


def _schema(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


@dataclass
class PointingState:
    cameras: tuple[str, ...]
    arms: tuple[str, ...] = ("hand",)
    last_target: np.ndarray | None = None
    yaws: dict = field(default_factory=dict)

    @property
    def multi(self) -> bool:
        return len(self.arms) > 1

    def with_arm(self, props: dict, required: tuple = ()) -> tuple[dict, tuple]:
        """Add the `arm` argument to a motion tool's schema when there is more than one arm."""
        if not self.multi:
            return props, required
        return {**props, "arm": {"type": "string", "enum": list(self.arms), "description": "Which arm moves."}}, \
            (*required, "arm")


@dataclass
class _Tool:
    state: PointingState
    name: str = ""
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    kind: str = "act"

    # -- shared helpers --------------------------------------------------------------------

    def report(self, ctx: ToolContext, message: str, receipt: Receipt | None = None,
               ok: bool = True, extra: dict | None = None, arm: str | None = None) -> ToolResult:
        stored = capture(ctx, self.state)
        proprio = stored.observation.proprio
        table_z = ctx.body.info().workspace.table_z
        names = self.state.arms if self.state.multi else (None,)
        lines, arms = [], {}
        for name in names:
            line, d = _arm_status(proprio.arm(name), table_z)
            lines.append(f"{name.capitalize()} arm: {line}" if name else line)
            arms[name] = d
        status = " ".join(lines) + f" Pictures: {stored.id}."
        content = (Text(f"{message}\n{status}"), *(ImageRef(stored.id, c) for c in self.state.cameras))
        acting = arms[arm if self.state.multi else None] if (arm or not self.state.multi) else {}
        data = {"observation": stored.id, **acting, "stop": receipt.stop_reason if receipt else None, **(extra or {})}
        if self.state.multi:
            data["arms"] = arms
        return ToolResult(content, is_error=not ok, receipt=receipt, data=data)

    def arm(self, args: dict) -> str | None:
        return args.get("arm") if self.state.multi else None


def _arm_status(p, table_z: float) -> tuple[str, dict]:
    closed = p.opening_cmd_m < 0.01
    holding = closed and p.opening_m > EMPTY_GRIP_M
    if holding:
        grip = f"closed on something {p.opening_m * 100:.1f} cm wide"
    elif closed:
        grip = "closed, empty"
    else:
        grip = f"open {p.opening_m * 100:.1f} cm"
    height = max(p.tcp_xyz[2] - table_z, 0) * 100
    line = f"Fingertips {height:.1f} cm above the table. Gripper {grip}. Contact force {p.wrist_force_n:.1f} N."
    return line, {"height_cm": round(height, 1), "opening_cm": round(p.opening_m * 100, 1), "holding": holding,
                  "contact_n": round(p.wrist_force_n, 1)}


def capture(ctx: ToolContext, state: PointingState) -> StoredObservation:
    obs = ctx.body.observe()
    oid = ctx.observations.next_id()
    views = {c: annotate(obs, c, state.last_target, oid, state.arms if state.multi else ()) for c in state.cameras}
    return ctx.observations.put(obs, views)


def _receipt_text(r: Receipt, success: str) -> str:
    return {
        "reached": success,
        "contact": f"Stopped early: touched something ({r.peak_force_n:.1f} N) and backed off slightly.",
        "blocked": "Stopped early: could not get all the way there.",
        "refused": f"Refused, nothing moved: {r.detail or 'not allowed'}.",
        "cancelled": f"Interrupted: {r.detail or 'cancelled'}.",
        "fault": f"The arm reported a fault: {r.detail}.",
    }[r.stop_reason]


class Look(_Tool):
    def __init__(self, state):
        super().__init__(state, "look", "Take fresh pictures from every camera without moving.", _schema({}), "observe")

    def execute(self, args, ctx):
        return self.report(ctx, "Fresh pictures.")


class MoveTo(_Tool):
    def __init__(self, state):
        super().__init__(state, "move_to", (
            "Move the gripper to hover above the point you click. Click the exact spot the fingertips should end "
            "up over, e.g. the centre of an object's top or a spot on the table. The arm rises, travels, and stops "
            "hover_cm above that surface point, so it does not hit what you clicked. It stops early on contact."),
            _schema(*state.with_arm({
                "camera": {"type": "string", "enum": list(state.cameras)},
                "x": {"type": "integer", "description": "Pixel column, 0 at the left edge."},
                "y": {"type": "integer", "description": "Pixel row, 0 at the top edge."},
                "hover_cm": {"type": "number", "minimum": 0, "maximum": 25, "default": 8},
                "observation": {"type": "string", "description": "Which pictures you clicked on, e.g. o4. "
                                                                 "Defaults to the latest."},
            }, ("camera", "x", "y"))))

    def execute(self, args, ctx):
        stored = ctx.observations.get(args["observation"]) if "observation" in args else ctx.observations.latest()
        if stored is None:
            return ToolResult.error(f"There are no pictures called {args.get('observation')}. Nothing moved.")
        surface = surface_point(stored, args["camera"], int(args["x"]), int(args["y"]),
                                ctx.body.info().workspace.table_z)
        arm = self.arm(args)
        if isinstance(surface, str):
            return self.report(ctx, surface + " Nothing moved.", Receipt("none", "refused", True), ok=False, arm=arm)
        hover = args.get("hover_cm", 8) / 100
        goal = surface + [0, 0, hover]
        if not ctx.body.info().workspace.contains(goal):
            return self.report(ctx, "That point is outside the area the arm can reach. Nothing moved.",
                               Receipt("none", "refused", True), ok=False, arm=arm)
        self.state.last_target = surface
        start = ctx.body.observe().proprio.arm(arm).tcp_xyz
        transit = min(max(start[2], goal[2]) + TRANSIT_CLEARANCE_M, ctx.body.info().workspace.hi[2])
        legs = [np.array([start[0], start[1], transit]), np.array([goal[0], goal[1], transit]), goal]
        receipt = None
        for i, leg in enumerate(legs):
            receipt = ctx.body.move(leg, Guard(), ctx.cancel, arm=arm)
            if receipt.stop_reason != "reached":
                if i and receipt.executed == "none":
                    receipt = Receipt("partial", receipt.stop_reason, receipt.stop_confirmed, receipt.peak_force_n,
                                      receipt.detail)
                break
        return self.report(ctx, _receipt_text(receipt, f"Hovering {hover * 100:g} cm above the point you clicked "
                                                       f"in {stored.id}."), receipt, arm=arm)


class Lower(_Tool):
    def __init__(self, state):
        super().__init__(state, "lower", (
            "Lower the gripper straight down until it touches something, or by cm if given. Use it to come down "
            "around an object before closing, or to set a held object down."),
            _schema(*state.with_arm({"cm": {"type": "number", "exclusiveMinimum": 0, "maximum": 30}})))

    def execute(self, args, ctx):
        arm = self.arm(args)
        table = ctx.body.info().workspace.table_z
        tcp = ctx.body.observe().proprio.arm(arm).tcp_xyz
        floor = table + 0.004
        goal_z = floor if "cm" not in args else max(tcp[2] - args["cm"] / 100, floor)
        receipt = ctx.body.move([tcp[0], tcp[1], goal_z], Guard(max_force_n=3.0, speed_mps=0.08), ctx.cancel, arm=arm)
        dropped = (tcp[2] - ctx.body.observe().proprio.arm(arm).tcp_xyz[2]) * 100
        if receipt.stop_reason == "contact":
            msg = f"Lowered {dropped:.1f} cm and stopped on contact."
        elif receipt.stop_reason == "reached" and "cm" not in args:
            msg = f"Lowered {dropped:.1f} cm to just above the table without touching anything."
        else:
            msg = _receipt_text(receipt, f"Lowered {dropped:.1f} cm.")
        # Reaching contact is what lower is for; report it as a completed motion.
        if receipt.stop_reason == "contact":
            receipt = Receipt(receipt.executed, "reached", receipt.stop_confirmed, receipt.peak_force_n, "contact")
        return self.report(ctx, msg, receipt, extra={"touched": receipt.detail == "contact"}, arm=arm)


class Nudge(_Tool):
    def __init__(self, state):
        super().__init__(state, "nudge", (
            "Move the gripper a short distance in directions as seen in one camera's pictures: right_cm toward the "
            "right of that picture, away_cm away from that camera along the table, up_cm straight up. Negative "
            "values go the other way. Stops early on contact."),
            _schema(*state.with_arm({
                "camera": {"type": "string", "enum": list(state.cameras)},
                "right_cm": {"type": "number", "minimum": -20, "maximum": 20, "default": 0},
                "away_cm": {"type": "number", "minimum": -20, "maximum": 20, "default": 0},
                "up_cm": {"type": "number", "minimum": -20, "maximum": 20, "default": 0},
            }, ("camera",))))

    def execute(self, args, ctx):
        arm = self.arm(args)
        obs = ctx.body.observe()
        right, away = obs.cameras[args["camera"]].horizontal_axes()
        delta = (right * args.get("right_cm", 0) + away * args.get("away_cm", 0)
                 + np.array([0, 0, 1.0]) * args.get("up_cm", 0)) / 100
        goal = obs.proprio.arm(arm).tcp_xyz + delta
        goal[2] = max(goal[2], ctx.body.info().workspace.table_z + 0.004)
        receipt = ctx.body.move(goal, Guard(speed_mps=0.10), ctx.cancel, arm=arm)
        return self.report(ctx, _receipt_text(receipt, "Nudged."), receipt, arm=arm)


class Turn(_Tool):
    def __init__(self, state):
        super().__init__(state, "turn", (
            "Rotate the gripper about the vertical axis. Positive is counterclockwise seen from above. The fingers "
            "close along the gripper's long axis, so turn until that axis crosses the object."),
            _schema(*state.with_arm({"degrees": {"type": "number", "minimum": -180, "maximum": 180}}, ("degrees",))))

    def execute(self, args, ctx):
        arm = self.arm(args)
        target = self.state.yaws.get(arm, 0.0) + np.deg2rad(args["degrees"])
        receipt = ctx.body.rotate(target, ctx.cancel, arm=arm)
        if receipt.stop_reason == "reached":
            self.state.yaws[arm] = target
        return self.report(ctx, _receipt_text(receipt, f"Turned {args['degrees']:g}°."), receipt, arm=arm)


class Grip(_Tool):
    def __init__(self, state):
        super().__init__(state, "grip", "Close or open the fingers. After closing, the result says whether "
                                        "something is between them.",
                         _schema(*state.with_arm({"action": {"type": "string", "enum": ["close", "open"]}},
                                                 ("action",))))

    def execute(self, args, ctx):
        arm = self.arm(args)
        info = ctx.body.info().arm(arm)
        target = 0.0 if args["action"] == "close" else info.max_opening_m
        receipt = ctx.body.grip(target, ctx.cancel, arm=arm)
        if receipt.stop_reason != "reached":
            return self.report(ctx, _receipt_text(receipt, ""), receipt, arm=arm)
        if args["action"] == "open":
            return self.report(ctx, "Opened.", receipt, arm=arm)
        width = ctx.body.observe().proprio.arm(arm).opening_m
        if width > EMPTY_GRIP_M:
            return self.report(ctx, f"Closed. Something {width * 100:.1f} cm wide is between the fingers.", receipt,
                               arm=arm)
        if info.hand == "hand":
            return self.report(ctx, "Closed the hand; it is not holding anything yet. A hand often finishes a grasp "
                                    "as it lifts: nudge up a little and check whether it is holding.", receipt, arm=arm)
        return self.report(ctx, "Closed on nothing: the fingers met.", receipt, arm=arm)


class Done(_Tool):
    def __init__(self, state):
        super().__init__(state, "done", "End the task. Call it once, when the pictures show the goal is met or "
                                        "you cannot make progress.",
                         _schema({"success": {"type": "boolean"}, "summary": {"type": "string", "maxLength": 2000}},
                                 ("success", "summary")), "control")

    def execute(self, args, ctx):
        return ToolResult((Text("Episode ended."),), data={"success": args["success"], "summary": args["summary"]},
                          ends_episode=True)


class Pointing(Extension):
    """`expose_motion=False` hides the motion tools from the model but keeps them callable from code."""

    name = "pointing"

    def __init__(self, cameras: tuple[str, ...], expose_motion: bool = True, arms: tuple[str, ...] = ("hand",),
                 robot: str = ""):
        self.state = PointingState(cameras, tuple(arms))
        motion = [MoveTo(self.state), Lower(self.state), Nudge(self.state), Turn(self.state), Grip(self.state)]
        self.tools = [Look(self.state), *(motion if expose_motion else []), Done(self.state)]
        self.internal_tools = [] if expose_motion else motion
        prompt = PROMPT if expose_motion else PROMPT.split(" A typical grasp")[0]
        if robot:
            prompt = f"You control {robot}.\n{prompt}"
        if self.state.multi:
            prompt += (f"\nThis robot has {len(arms)} arms: {', '.join(arms)}. Every motion tool takes `arm`. Each arm's "
                       "fingertip marker is labelled with its first letter. The arms can collide with each other; "
                       "move one out of the way before the other works near it.")
        self.prompt = prompt

    def first_look(self, ctx: ToolContext) -> ToolResult:
        return self.tools[0].execute({}, ctx)


# ---- geometry and pictures ---------------------------------------------------------------------


def surface_point(stored: StoredObservation, camera: str, x: int, y: int, table_z: float) -> np.ndarray | str:
    obs = stored.observation
    if camera not in obs.cameras:
        return f"{stored.id} has no camera called {camera}."
    cam = obs.cameras[camera]
    if not (0 <= x < cam.width and 0 <= y < cam.height):
        return f"({x}, {y}) is outside the {cam.width}×{cam.height} picture."
    if obs.depth is None or camera not in obs.depth:
        return f"The {camera} camera has no depth, so a single click cannot be placed in 3D."
    depth = obs.depth[camera]
    win = depth[max(0, y - 2): y + 3, max(0, x - 2): x + 3]   # an edge click should not land on the background
    point = cam.deproject(x, y, float(np.median(win)))
    if point[2] < table_z - 0.01:
        return f"({x}, {y}) in {stored.id} {camera} is off the table."
    return point


def annotate(obs, camera: str, target, oid: str, arms: tuple[str, ...] = ()) -> bytes:
    img = Image.fromarray(obs.rgb[camera])
    draw = ImageDraw.Draw(img)
    cam = obs.cameras[camera]
    w, h = img.size
    white = (255, 255, 255)
    for x in range(0, w, 50):
        long = x % 100 == 0
        draw.line([(x, 0), (x, 14 if long else 8)], fill=white)
        draw.line([(x, h - 1), (x, h - (14 if long else 8))], fill=white)
        if long and x:
            draw.text((x + 2, 14), str(x), fill=white)
    for y in range(0, h, 50):
        long = y % 100 == 0
        draw.line([(0, y), (14 if long else 8, y)], fill=white)
        draw.line([(w - 1, y), (w - (14 if long else 8), y)], fill=white)
        if long and y:
            draw.text((16, y - 6), str(y), fill=white)
    draw.rectangle([w - 92, h - 22, w - 4, h - 4], fill=(0, 0, 0))
    draw.text((w - 88, h - 19), f"{oid} {camera}", fill=white)

    for name in arms or (None,):
        tcp = obs.proprio.arm(name).tcp_xyz
        tip, floor = cam.project(tcp), cam.project([tcp[0], tcp[1], 0.0])
        if tip and floor:
            a, b = np.array(tip), np.array(floor)
            n = max(1, int(np.linalg.norm(b - a) // 6))
            for i in range(0, n, 2):
                draw.line([tuple(a + (b - a) * i / n), tuple(a + (b - a) * min(i + 1, n) / n)], fill=(0, 230, 255),
                          width=2)
            draw.ellipse([b[0] - 6, b[1] - 3, b[0] + 6, b[1] + 3], outline=(0, 230, 255), width=2)
            draw.ellipse([a[0] - 5, a[1] - 5, a[0] + 5, a[1] + 5], outline=(0, 230, 255), width=2)
            if name:
                draw.rectangle([a[0] + 7, a[1] - 7, a[0] + 17, a[1] + 6], fill=(0, 0, 0))
                draw.text((a[0] + 9, a[1] - 6), name[0].upper(), fill=(0, 230, 255))
    if target is not None and (pt := cam.project(target)):
        x, y = pt
        draw.line([(x - 6, y - 6), (x + 6, y + 6)], fill=(255, 0, 200), width=2)
        draw.line([(x - 6, y + 6), (x + 6, y - 6)], fill=(255, 0, 200), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
