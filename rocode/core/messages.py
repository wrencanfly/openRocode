"""Canonical conversation records. Providers translate these to and from each vendor's wire format.

History is append-only: nothing in this module mutates a message after it is appended, and
providers must render the same message to the same bytes on every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ImageRef:
    """An image held in the ObservationStore; bytes are looked up at request time."""

    observation_id: str
    camera: str


Content = Text | ImageRef


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class UserMessage:
    content: tuple[Content, ...]


@dataclass(frozen=True)
class OperatorMessage:
    """Something the human operator said mid-episode."""

    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str
    tool_calls: tuple[ToolCall, ...]
    # The provider's own blocks, replayed verbatim on later requests (thinking blocks included).
    raw: Any = field(default=None, compare=False, repr=False)
    provider: str = ""


@dataclass(frozen=True)
class ToolResultBlock:
    call_id: str
    content: tuple[Content, ...]
    is_error: bool = False


@dataclass(frozen=True)
class ToolResultsMessage:
    """All results for one assistant turn, in the order the calls were made."""

    results: tuple[ToolResultBlock, ...]


Message = UserMessage | OperatorMessage | AssistantMessage | ToolResultsMessage


def text_of(content: tuple[Content, ...]) -> str:
    return "\n".join(c.text for c in content if isinstance(c, Text))
