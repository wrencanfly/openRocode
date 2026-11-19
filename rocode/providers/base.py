"""The provider contract: one request in, one reply out. Everything vendor-specific stays behind it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from ..core.messages import Message, ToolCall
from ..core.observations import ObservationStore


@dataclass(frozen=True)
class ContextPolicy:
    """Keep images from only the most recent `keep_image_turns` tool results; older ones are dropped.

    Providers realize this in whatever way keeps their history valid (see each adapter).
    """

    keep_image_turns: int | None = None


@dataclass(frozen=True)
class Request:
    system: str
    tools: list[dict]
    messages: list[Message]
    images: ObservationStore
    effort: str = "medium"
    max_tokens: int = 16000
    context: ContextPolicy = ContextPolicy()

    def with_(self, **changes) -> "Request":
        return replace(self, **changes)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass(frozen=True)
class Reply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: str            # "tool_use" | "end_turn" | "max_tokens" | "refusal" | provider-specific
    raw: object = field(default=None, repr=False)   # replayed verbatim by the same provider
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    reasoning: str = ""         # the model's thinking or its summary, when the provider returns it; display only


class ProviderError(RuntimeError):
    """A request failed after the SDK's own retries. The episode ends; nothing physical is replayed."""


class Provider(Protocol):
    name: str
    model: str

    def complete(self, request: Request) -> Reply: ...
