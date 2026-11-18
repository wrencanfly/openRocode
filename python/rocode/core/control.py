"""Operator controls, callable from any thread while an episode runs."""

from __future__ import annotations

import queue
import threading

from ..body.base import Body, CancelToken


class ControlChannel:
    """`say` and `pause` take effect at the next safe point; `estop` takes effect immediately."""

    def __init__(self) -> None:
        self._said: queue.Queue[str] = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self.cancel = CancelToken()
        self._body: Body | None = None
        self.estop_source: str | None = None

    def attach(self, body: Body) -> None:
        self._body = body

    def say(self, text: str) -> None:
        """Pass a message to the model before its next decision ("not that one, the left one")."""
        self._said.put(text)

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def estop(self, source: str = "operator") -> None:
        self.estop_source = source
        self.cancel.cancel(f"emergency stop ({source})")
        if self._body is not None:
            self._body.stop()
        self._running.set()  # wake a paused loop so it can end

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def wait_while_paused(self) -> None:
        self._running.wait()

    def drain_said(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._said.get_nowait())
            except queue.Empty:
                return out
