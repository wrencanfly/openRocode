"""Immutable store of observations and the exact image bytes the model was shown."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..body.base import Observation


@dataclass(frozen=True)
class StoredObservation:
    id: str
    observation: Observation
    views: dict[str, bytes]   # camera -> PNG bytes as shown to the model (annotations included)


class ObservationStore:
    """Ids are short and sequential (o1, o2, ...) so a model can cite them."""

    def __init__(self, on_put=None) -> None:
        self._items: dict[str, StoredObservation] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.on_put = on_put

    def next_id(self) -> str:
        """The id the next `put` will assign (so it can be drawn onto the pictures)."""
        return f"o{len(self._order) + 1}"

    def put(self, observation: Observation, views: dict[str, bytes]) -> StoredObservation:
        with self._lock:
            oid = self.next_id()
            item = StoredObservation(oid, observation, dict(views))
            self._items[oid] = item
            self._order.append(oid)
        if self.on_put:
            self.on_put(item)
        return item

    def get(self, oid: str) -> StoredObservation | None:
        return self._items.get(oid)

    def latest(self) -> StoredObservation | None:
        return self._items[self._order[-1]] if self._order else None

    def png(self, oid: str, camera: str) -> bytes:
        return self._items[oid].views[camera]

    def __iter__(self):
        return (self._items[i] for i in list(self._order))
