#!/usr/bin/env python3
"""Example isolated L3 grasp policy worker over Unix Domain Socket.

Supports `--stall-at-step N` to simulate a hung CUDA / VLA inference step so Rust's
80ms S0/S1 Watchdog interception into `SafeHold` can be verified end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openrocode_sdk.worker import PolicyWorker  # noqa: E402


class MockGraspPolicyWorker(PolicyWorker):
    def __init__(self, stall_at_step: int | None = None) -> None:
        self.stall_at_step = stall_at_step
        self.step_idx = 0
        self.target_id = "red_mug"

    def on_init(self, skill_name: str, arguments: dict[str, Any]) -> None:
        self.target_id = arguments.get("target_id", "red_mug")
        self.step_idx = 0

    def on_step(self, state: dict[str, Any]) -> dict[str, Any]:
        self.step_idx += 1
        if self.stall_at_step is not None and self.step_idx >= self.stall_at_step:
            time.sleep(0.5)  # Exceeds 80ms watchdog budget -> triggers Rust SafeHold!

        ee = state["ee_xyz_m"]
        held = state.get("held_object_id")

        # Phase 2: Once grasped, lift clear of tabletop to z = 0.20m
        if held == self.target_id:
            if ee[2] >= 0.18:
                return {
                    "status": "completed",
                    "summary": f"L3 policy grasped and lifted '{self.target_id}' to z={ee[2]:.2f}m",
                }
            next_xyz = [ee[0], ee[1], min(ee[2] + 0.015, 0.20)]
            return {
                "status": "command",
                "command": {
                    "timestamp_ns": state["timestamp_ns"],
                    "target_positions": state["joint_positions"],
                    "target_ee_xyz_m": next_xyz,
                    "gripper_open_ratio": 0.0,
                },
            }

        # Phase 1: Converge onto target object and close gripper
        obj_xyz = None
        for obj in state.get("scene_objects", []):
            if obj["id"] == self.target_id:
                obj_xyz = obj["xyz_m"]
                break
        if obj_xyz is None:
            return {"status": "failed", "reason": f"object '{self.target_id}' not in scene"}

        dx = obj_xyz[0] - ee[0]
        dy = obj_xyz[1] - ee[1]
        dz = obj_xyz[2] - ee[2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist > 0.025:
            step = min(0.018, dist)
            next_xyz = [ee[0] + dx / dist * step, ee[1] + dy / dist * step, ee[2] + dz / dist * step]
            gripper = 1.0
        else:
            next_xyz = obj_xyz
            gripper = 0.0

        return {
            "status": "command",
            "command": {
                "timestamp_ns": state["timestamp_ns"],
                "target_positions": state["joint_positions"],
                "target_ee_xyz_m": next_xyz,
                "gripper_open_ratio": gripper,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--stall-at-step", type=int, default=None)
    args = parser.parse_args()
    worker = MockGraspPolicyWorker(stall_at_step=args.stall_at_step)
    worker.serve(args.socket)


if __name__ == "__main__":
    main()
