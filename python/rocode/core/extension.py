"""Extensions bundle tools, prompt text and hooks. The core knows nothing else about them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .events import Event
from .messages import ToolCall
from .tooling import Tool, ToolResult

if TYPE_CHECKING:
    from ..providers.base import Request


@dataclass(frozen=True)
class Deny:
    """Refuse a call. The model is told `reason`; nothing is executed."""

    reason: str


@dataclass(frozen=True)
class Rewrite:
    args: dict


@dataclass(frozen=True)
class Terminate:
    """End the episode instead of running the call."""

    reason: str


Decision = Deny | Rewrite | Terminate


class Extension:
    """Override what you need. Hooks run in registration order."""

    name = "extension"
    tools: list[Tool] = []
    # Fixed for the whole episode: the system prompt and tool list never change mid-conversation.
    prompt: str = ""

    def before_request(self, request: "Request") -> "Request":
        return request

    def before_tool(self, call: ToolCall, tool: Tool) -> Decision | None:
        return None

    def after_tool(self, call: ToolCall, tool: Tool, result: ToolResult) -> ToolResult:
        return result

    def on_event(self, event: Event) -> None:
        pass
