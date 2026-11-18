"""A provider that plays a fixed plan. For tests and demos without an API key."""

from __future__ import annotations

from collections.abc import Callable

from ..core.messages import ToolCall
from .base import Reply, Request, Usage

# A step is a list of (tool_name, args) pairs, or a function of the request that returns one
# (so a script can react to what it has seen, e.g. compute a pixel from the latest observation).
Step = list[tuple[str, dict]] | Callable[[Request], list[tuple[str, dict]]]


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.requests: list[Request] = []
        self._n = 0

    def complete(self, request: Request) -> Reply:
        self.requests.append(request)
        if not self.steps:
            return Reply("Script finished.", (), "end_turn", None, Usage())
        step = self.steps.pop(0)
        calls = step(request) if callable(step) else step
        out = []
        for name, args in calls:
            self._n += 1
            out.append(ToolCall(f"call_{self._n}", name, dict(args)))
        return Reply("", tuple(out), "tool_use", None, Usage(input_tokens=1, output_tokens=1))
