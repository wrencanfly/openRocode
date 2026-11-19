"""Code as policy: the model writes a Python program that drives the robot.

The program runs in a separate process. Every robot call it makes travels back over a socket and runs
through the agent's normal tool path (validation, hooks, events, e-stop), so a program can do nothing
a direct tool call could not. Programs cannot see pictures: they act on pixels the model chose from
pictures it has already looked at, and end when the model needs to look again.

Isolation is a separate interpreter in isolated mode with a wall-clock and call budget. It is not a
security sandbox: the program runs with the user's permissions.
"""

from __future__ import annotations

import json
import os
import selectors
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..body.base import Receipt
from ..core.extension import Extension
from ..core.messages import ImageRef, Text, ToolCall
from ..core.tooling import ToolContext, ToolResult

ROBOT_TOOLS = ("look", "move_to", "lower", "nudge", "turn", "grip")

PRELUDE = r'''
import json as _json, os as _os, socket as _socket, sys as _sys

class Result(dict):
    """A robot call's outcome. Attribute access works: r.ok, r.text, r.holding, r.height_cm, r.stop ..."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

class _Robot:
    def __init__(self):
        sock = _socket.socket(fileno=int(_os.environ["ROCODE_RPC_FD"]))
        self._f = sock.makefile("rwb")

    def _call(self, tool, **args):
        args = {k: v for k, v in args.items() if v is not None}
        self._f.write(_json.dumps({"tool": tool, "args": args}).encode() + b"\n")
        self._f.flush()
        line = self._f.readline()
        if not line:
            raise SystemExit("the harness closed the robot connection")
        reply = _json.loads(line)
        if reply.get("abort"):
            print("[robot] " + reply["text"], file=_sys.stderr)
            raise SystemExit(3)
        return Result(ok=reply["ok"], text=reply["text"], **reply.get("data", {}))

    def look(self):
        return self._call("look")

    def move_to(self, camera, x, y, hover_cm=8, observation=None, arm=None):
        return self._call("move_to", camera=camera, x=int(x), y=int(y), hover_cm=hover_cm, observation=observation,
                          arm=arm)

    def lower(self, cm=None, arm=None):
        return self._call("lower", cm=cm, arm=arm)

    def nudge(self, camera, right_cm=0, away_cm=0, up_cm=0, arm=None):
        return self._call("nudge", camera=camera, right_cm=right_cm, away_cm=away_cm, up_cm=up_cm, arm=arm)

    def turn(self, degrees, arm=None):
        return self._call("turn", degrees=degrees, arm=arm)

    def grip(self, action, arm=None):
        return self._call("grip", action=action, arm=arm)

robot = _Robot()
'''

PROMPT = """\
Code tool. run_code runs a Python program that controls the robot through a ready-made `robot` object:
  robot.look()
  robot.move_to(camera, x, y, hover_cm=8, observation=None)
  robot.lower(cm=None)            # until contact when cm is omitted
  robot.nudge(camera, right_cm=0, away_cm=0, up_cm=0)
  robot.turn(degrees)
  robot.grip("close" | "open")
Each call blocks until the motion ends and returns a Result with: ok, text, observation (picture id), \
height_cm, opening_cm, holding, contact_n, stop ("reached", "contact", "blocked", "refused", "cancelled") \
and, for lower, touched. Programs cannot see pictures: choose pixels from pictures you have already \
looked at, then let the program handle the parts that need loops, retries or checks of the numbers \
above. print() what you want to read afterwards. When the next decision needs your eyes, end the \
program and look at the pictures returned with its result. The `robot` object is the only way to \
reach the robot; a program never sees world coordinates."""


class RunCode:
    name = "run_code"
    kind = "act"
    description = ("Run a Python program that controls the robot through the `robot` object described in the "
                   "system prompt. Returns the program's output, the robot calls it made, and the latest pictures.")
    input_schema = {"type": "object", "properties": {"code": {"type": "string", "maxLength": 20000}},
                    "required": ["code"], "additionalProperties": False}

    def __init__(self, cameras: tuple[str, ...], max_calls: int = 80, max_seconds: float = 180.0,
                 output_limit: int = 6000):
        self.cameras = cameras
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.output_limit = output_limit

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        parent, child = socket.socketpair()
        workdir = Path(tempfile.mkdtemp(prefix="rocode_prog_"))
        (workdir / "program.py").write_text(PRELUDE + "\n# ---- program ----\n" + args["code"])
        env = {"ROCODE_RPC_FD": str(child.fileno()), "PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen([sys.executable, "-I", "program.py"], cwd=workdir, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(child.fileno(),))
        child.close()
        out, err = [], []
        readers = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
                   threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
        for t in readers:
            t.start()

        calls: list[str] = []
        last_obs: str | None = None
        stop_note = ""
        deadline = time.time() + self.max_seconds
        rfile = parent.makefile("rb")
        sel = selectors.DefaultSelector()
        sel.register(parent, selectors.EVENT_READ)
        try:
            while True:
                if ctx.cancel.cancelled:
                    stop_note = f"Stopped by {ctx.cancel.reason}."
                    break
                if time.time() > deadline:
                    stop_note = f"Stopped: the program ran longer than {self.max_seconds:g} s."
                    break
                if not sel.select(timeout=0.1):
                    if proc.poll() is not None:
                        break
                    continue
                line = rfile.readline()
                if not line:
                    break
                request = json.loads(line)
                name, call_args = request.get("tool"), request.get("args", {})
                if len(calls) >= self.max_calls:
                    _send(parent, {"abort": True, "text": f"call budget of {self.max_calls} robot calls used up"})
                    stop_note = f"Stopped: the program made more than {self.max_calls} robot calls."
                    break
                if name not in ROBOT_TOOLS:
                    _send(parent, {"ok": False, "text": f"robot has no method {name}", "data": {}})
                    continue
                result = ctx.run_tool(ToolCall(f"{ctx.call_id}.{len(calls) + 1}", name, call_args))
                text = "\n".join(c.text for c in result.content if isinstance(c, Text))
                calls.append(f"{name}({_fmt(call_args)}) -> {text.splitlines()[0] if text else ''}")
                last_obs = result.data.get("observation", last_obs)
                if ctx.cancel.cancelled:
                    _send(parent, {"abort": True, "text": ctx.cancel.reason})
                    stop_note = f"Stopped by {ctx.cancel.reason}."
                    break
                _send(parent, {"ok": not result.is_error, "text": text, "data": result.data})
        finally:
            sel.close()
            if not stop_note:
                try:  # the socket closes a moment before the interpreter finishes exiting
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_note = "Stopped: the program stopped talking to the robot but did not exit."
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for t in readers:
                t.join(timeout=2)
            parent.close()

        code = proc.returncode
        exited = "finished" if code == 0 and not stop_note else f"ended with exit code {code}"
        lines = [stop_note or f"Program {exited} after {len(calls)} robot call(s)."]
        if calls:
            lines.append("Robot calls:\n" + "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(calls)))
        stdout, stderr = _tail("".join(out), self.output_limit), _tail("".join(err), self.output_limit)
        if stdout:
            lines.append("Output:\n" + stdout)
        if stderr:
            lines.append("Errors:\n" + stderr)
        content: list = [Text("\n".join(lines))]
        if last_obs:
            content += [ImageRef(last_obs, c) for c in self.cameras]
        failed = code != 0 or bool(stop_note)
        reason = "cancelled" if ctx.cancel.cancelled else ("fault" if failed else "reached")
        receipt = Receipt("none" if not calls else ("full" if not failed else "partial"), reason, True,
                          detail=stop_note or f"exit {code}")
        return ToolResult(tuple(content), is_error=failed, receipt=receipt,
                          data={"observation": last_obs, "calls": len(calls), "exit_code": code},
                          details={"code": args["code"], "stdout": "".join(out), "stderr": "".join(err)})


class CodeAsPolicy(Extension):
    name = "code"

    def __init__(self, cameras: tuple[str, ...], arms: tuple[str, ...] = ("hand",), **limits):
        self.tools = [RunCode(cameras, **limits)]
        self.prompt = PROMPT
        if len(arms) > 1:
            self.prompt += (f"\nEvery motion call takes arm= one of {', '.join(repr(a) for a in arms)}, e.g. "
                            f"robot.grip(\"close\", arm={arms[0]!r}). Results describe the arm that moved and, under "
                            "`arms`, every arm.")


def _send(sock: socket.socket, payload: dict) -> None:
    sock.sendall(json.dumps(payload).encode() + b"\n")


def _drain(stream, sink: list[str]) -> None:
    for chunk in iter(lambda: stream.readline(), b""):
        sink.append(chunk.decode(errors="replace"))


def _tail(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else f"[... {len(text) - limit} characters cut ...]\n" + text[-limit:]


def _fmt(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())
