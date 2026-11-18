"""Writes an episode to disk: events, every picture the model saw, and the final result.

    <dir>/episode.jsonl        one event per line, no image bytes
    <dir>/observations/        <id>_<camera>.png exactly as shown to the model
    <dir>/result.json          end reason, counts, tokens, what the model claimed
    <dir>/evaluation.json      written by the session from ground truth (never shown to the model)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.events import EpisodeEnd, Event, ObservationTaken
from ..core.extension import Extension
from ..core.observations import ObservationStore


class Recorder(Extension):
    name = "recorder"

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        (self.dir / "observations").mkdir(parents=True, exist_ok=True)
        self._f = (self.dir / "episode.jsonl").open("w")
        self.store: ObservationStore | None = None

    def bind(self, store: ObservationStore) -> None:
        self.store = store

    def on_event(self, event: Event) -> None:
        self._f.write(json.dumps(event.to_dict(), default=str) + "\n")
        self._f.flush()
        if isinstance(event, ObservationTaken) and self.store is not None:
            item = self.store.get(event.observation_id)
            for camera, png in item.views.items():
                (self.dir / "observations" / f"{item.id}_{camera}.png").write_bytes(png)
        if isinstance(event, EpisodeEnd):
            (self.dir / "result.json").write_text(json.dumps({"reason": event.reason, **event.stats}, indent=2,
                                                             default=str))
            self._f.close()
