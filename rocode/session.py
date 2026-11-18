"""Assemble one episode from a config: body, extensions, provider, recorder; run it; score it."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .body.base import Body
from .body.mujoco_tabletop import MujocoTabletop
from .core.control import ControlChannel
from .core.events import Event, EventBus
from .core.extension import Extension
from .core.loop import Agent, Budget, EpisodeOutcome, LoopConfig
from .core.messages import ToolCall
from .ext.code import CodeAsPolicy
from .ext.pointing import Pointing
from .ext.recorder import Recorder
from .ext.safety import SafetyGate
from .providers.base import ContextPolicy, Provider
from .tasks import TASKS

Policy = Literal["tools", "code", "both"]


@dataclass
class SessionConfig:
    task: str
    provider: str = "anthropic"            # anthropic | openai
    model: str = "claude-opus-5-5"
    base_url: str | None = None            # for OpenAI-compatible servers
    policy: Policy = "both"                # which ways the model may act: direct tool calls, programs, or both
    effort: str = "medium"
    max_steps: int = 60
    max_seconds: float = 1800.0
    keep_image_turns: int | None = None    # trim images older than this many tool results
    seed: int | None = None
    out_dir: Path | None = None


@dataclass
class SessionReport:
    task: str
    success: bool
    outcome: EpisodeOutcome
    out_dir: Path | None
    extra: dict = field(default_factory=dict)


def make_provider(cfg: SessionConfig) -> Provider:
    if cfg.provider == "anthropic":
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(cfg.model)
    if cfg.provider == "openai":
        from .providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg.model, base_url=cfg.base_url)
    raise ValueError(f"unknown provider {cfg.provider!r} (expected anthropic or openai)")


def run_session(cfg: SessionConfig, provider: Provider | None = None, control: ControlChannel | None = None,
                subscribers: list[Callable[[Event], None]] = (), extra_extensions: list[Extension] = (),
                body: Body | None = None) -> SessionReport:
    task = TASKS[cfg.task]
    body = body or MujocoTabletop(seed=cfg.seed)
    cameras = body.info().cameras
    pointing = Pointing(cameras, expose_motion=cfg.policy != "code")
    extensions: list[Extension] = [SafetyGate(), pointing]
    if cfg.policy in ("code", "both"):
        extensions.append(CodeAsPolicy(cameras))
    extensions += list(extra_extensions)
    recorder = None
    if cfg.out_dir is not None:
        recorder = Recorder(cfg.out_dir)
        extensions.append(recorder)

    bus = EventBus()
    for fn in subscribers:
        bus.subscribe(fn)
    loop_cfg = LoopConfig(effort=cfg.effort, budget=Budget(max_steps=cfg.max_steps, max_turns=cfg.max_steps,
                                                           max_seconds=cfg.max_seconds),
                          context=ContextPolicy(cfg.keep_image_turns))
    agent = Agent(provider or make_provider(cfg), body, extensions, loop_cfg, control, bus)
    if recorder:
        recorder.bind(agent.observations)

    # The first look goes through the normal tool path so recordings and viewers see it like any other step.
    first_look = lambda: agent.run_tool(ToolCall("initial_look", "look", {})).content  # noqa: E731
    meta = {"task_id": task.id, "body": body.info().name, "policy": cfg.policy}
    outcome = agent.run(task.instruction, first_look, meta)
    truth = body.ground_truth()
    success = task.check(truth)
    if cfg.out_dir is not None:
        evaluation = {"task": task.id, "success": success, "claimed": (outcome.finish or {}).get("success"),
                      "ground_truth": {k: _plain(v) for k, v in truth.items()},
                      "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}}
        (Path(cfg.out_dir) / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    return SessionReport(task.id, success, outcome, cfg.out_dir)


def default_out_dir(root: Path, task: str) -> Path:
    return Path(root) / f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return [round(float(x), 4) for x in v]
    return v
