"""Read episode traces. The format is documented in docs/trace-format.md.

A trace is a directory written by the Recorder extension:

    episode.jsonl       one event per line, flushed as it happens (readable while the episode runs)
    observations/       <observation id>_<camera>.png, every picture exactly as the model saw it
    result.json         how the episode ended: reason, counts, tokens, what the model claimed
    evaluation.json     the ground-truth verdict, written after the episode (never shown to the model)

    >>> from rocode.trace import load
    >>> trace = load("runs/mug_to_coaster-20261117-235411")
    >>> trace.config["model"], trace.success, len(trace.events)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .core.events import TRACE_VERSION

__all__ = ["TRACE_VERSION", "Trace", "load", "read_events", "read_json"]


def read_events(directory: str | Path) -> list[dict]:
    """Every complete event in episode.jsonl. Stops at a line a running episode is still writing."""
    out = []
    for line in (Path(directory) / "episode.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def read_json(path: str | Path) -> dict:
    """A JSON file's contents, or {} if it is missing or unreadable (e.g. an episode still running)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass
class Trace:
    dir: Path
    events: list[dict]
    result: dict
    evaluation: dict

    @property
    def start(self) -> dict | None:
        return next((e for e in self.events if e["type"] == "EpisodeStart"), None)

    @property
    def end(self) -> dict | None:
        return next((e for e in reversed(self.events) if e["type"] == "EpisodeEnd"), None)

    @property
    def version(self) -> int:
        # Traces recorded before the format was versioned are identical to version 1.
        return (self.start or {}).get("version", 1)

    @property
    def config(self) -> dict:
        return (self.start or {}).get("config", {})

    @property
    def running(self) -> bool:
        return self.end is None

    @property
    def success(self) -> bool | None:
        """The ground-truth verdict, or None while running or if the episode was not scored."""
        return self.evaluation.get("success")

    def image(self, observation_id: str, camera: str) -> Path:
        return self.dir / "observations" / f"{observation_id}_{camera}.png"


def load(directory: str | Path) -> Trace:
    d = Path(directory)
    trace = Trace(d, read_events(d), read_json(d / "result.json"), read_json(d / "evaluation.json"))
    if trace.version > TRACE_VERSION:
        raise ValueError(f"{d} uses trace format {trace.version}; this openRocode reads up to {TRACE_VERSION}. "
                         "Upgrade openRocode to read it.")
    return trace
