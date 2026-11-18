"""Everything the core does is published as an event. Recorders, UIs and evaluators only subscribe.

Subscribers observe; they cannot change control flow (that is what hooks are for). A subscriber that
raises is reported once and then ignored, so a broken UI can never stop a robot mid-motion.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    t: float = field(default_factory=time.time, kw_only=True)

    @property
    def type(self) -> str:
        return type(self).__name__

    def to_dict(self) -> dict:
        return {"type": self.type, **_plain(asdict(self))}


@dataclass(frozen=True)
class EpisodeStart(Event):
    task: str
    config: dict


@dataclass(frozen=True)
class EpisodeEnd(Event):
    reason: str
    stats: dict


@dataclass(frozen=True)
class TurnStart(Event):
    turn: int


@dataclass(frozen=True)
class AssistantReply(Event):
    turn: int
    text: str
    tool_calls: list[dict]
    usage: dict
    stop_reason: str


@dataclass(frozen=True)
class ToolStart(Event):
    call_id: str
    name: str
    args: dict
    parent_id: str | None = None


@dataclass(frozen=True)
class ToolEnd(Event):
    call_id: str
    name: str
    is_error: bool
    text: str
    observation_ids: list[str]
    receipt: dict | None
    duration_s: float
    parent_id: str | None = None
    data: dict = field(default_factory=dict)   # the tool's structured, model-safe readings


@dataclass(frozen=True)
class ToolDenied(Event):
    call_id: str
    name: str
    reason: str


@dataclass(frozen=True)
class ObservationTaken(Event):
    observation_id: str
    cameras: list[str]


@dataclass(frozen=True)
class OperatorInput(Event):
    text: str


@dataclass(frozen=True)
class Paused(Event):
    pass


@dataclass(frozen=True)
class Resumed(Event):
    pass


@dataclass(frozen=True)
class EStop(Event):
    source: str


@dataclass(frozen=True)
class Notice(Event):
    """Anything worth a line in the log that is not one of the above (retries, budget warnings)."""

    text: str


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Callable[[Event], None]] = []
        self._broken: set[int] = set()
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subs.append(fn)

    def emit(self, event: Event) -> None:
        with self._lock:
            for i, fn in enumerate(self._subs):
                if i in self._broken:
                    continue
                try:
                    fn(event)
                except Exception as err:  # noqa: BLE001 - observers must never break the loop
                    self._broken.add(i)
                    print(f"[rocode] event subscriber {fn!r} failed and was disabled: {err}", file=sys.stderr)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
