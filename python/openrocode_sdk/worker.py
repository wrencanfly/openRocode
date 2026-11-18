"""Unix Domain Socket NDJSON PolicyWorker base class for openRocode L3 plugins."""

from __future__ import annotations

import argparse
import json
import socket
from typing import Any


class PolicyWorker:
    """Subclass `PolicyWorker` to expose an RL / VLA / WAM policy to openRocode over UDS."""

    def on_init(self, skill_name: str, arguments: dict[str, Any]) -> None:
        pass

    def on_step(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def serve(self, socket_path: str) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(socket_path)
            reader = sock.makefile("r", encoding="utf-8")
            writer = sock.makefile("w", encoding="utf-8")
            for raw_line in reader:
                line = raw_line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                msg_type = msg.get("type")
                if msg_type == "init":
                    self.on_init(msg["skill_name"], msg.get("arguments", {}))
                    resp = {"status": "ready"}
                elif msg_type == "step":
                    resp = self.on_step(msg["state"])
                elif msg_type == "shutdown":
                    break
                else:
                    resp = {"status": "failed", "reason": f"unknown request type: {msg_type}"}
                writer.write(json.dumps(resp) + "\n")
                writer.flush()

    @classmethod
    def run_cli(cls) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--socket", required=True, help="Path to openRocode UDS socket")
        args, extra = parser.parse_known_args()
        worker = cls(*extra) if extra else cls()
        worker.serve(args.socket)
