"""Harness-level safety. The body enforces workspace and force limits on every motion; this gate adds
episode-level limits that need the agent's view of things."""

from __future__ import annotations

from ..core.events import Event, ToolEnd
from ..core.extension import Deny, Extension, Terminate
from ..core.messages import ToolCall
from ..core.tooling import Tool


class SafetyGate(Extension):
    name = "safety"

    def __init__(self, max_motions: int = 300, max_consecutive_faults: int = 3):
        self.max_motions = max_motions
        self.max_faults = max_consecutive_faults
        self.motions = 0
        self.faults = 0

    def before_tool(self, call: ToolCall, tool: Tool):
        if tool.kind != "act":
            return None
        if self.faults >= self.max_faults:
            return Terminate(f"the arm reported {self.faults} faults in a row; an operator should check it")
        if self.motions >= self.max_motions:
            return Deny(f"the motion budget for this episode ({self.max_motions}) is used up.")
        self.motions += 1
        return None

    def on_event(self, event: Event) -> None:
        # A program that raised is not an arm fault; only count what the body itself reported.
        if isinstance(event, ToolEnd) and event.receipt is not None and event.name != "run_code":
            self.faults = self.faults + 1 if event.receipt.get("stop_reason") == "fault" else 0
