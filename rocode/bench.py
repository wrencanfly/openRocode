"""Run tasks x models x seeds and record every episode, so the dashboard can compare them.

Each episode is an ordinary recorded trace in the runs directory, tagged with the batch id. The plan and
progress live in <runs>/batches/<batch id>.json, so the dashboard can show what is still to come.

Episodes run one after another on the main thread: MuJoCo renders only on the thread that created it, and
parallel episodes would compete for the same API rate limits.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .body.base import Body
from .core.control import ControlChannel
from .providers.base import Provider
from .session import SessionConfig, make_provider, run_session
from .tasks import Task, get as get_task
from .view.compare import BATCH_ID, load_manifests  # noqa: F401 - re-exported for callers of the bench API
from .view.episodes import summarize

PROVIDERS = ("anthropic", "openai")


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        """"openai:gpt-6.1-sol", "anthropic:claude-opus-5-5", or a bare Claude model name."""
        provider, sep, model = text.partition(":")
        if not sep:
            provider, model = "anthropic", text
        if provider not in PROVIDERS or not model.strip():
            raise ValueError(f"{text!r}: expected PROVIDER:MODEL with PROVIDER one of {', '.join(PROVIDERS)}")
        return cls(provider, model.strip())

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", self.model).strip("-") or "model"


@dataclass
class BenchPlan:
    tasks: list[str | Task]               # ids of built-in tasks, or Tasks (from task files)
    models: list[ModelSpec]
    seeds: list[int | None] = field(default_factory=lambda: [None])
    policy: str = "both"
    effort: str = "medium"
    max_steps: int = 60
    max_seconds: float = 1800.0
    keep_image_turns: int | None = None
    base_url: str | None = None           # for every openai: model
    reasoning: bool = False               # openai: models: send reasoning_effort, record reasoning summaries
    depth: bool = True                    # False: RGB-only cameras for every episode

    def __post_init__(self) -> None:
        self.tasks = [get_task(t) for t in self.tasks]
        ungraded = [t.id for t in self.tasks if not t.graded]
        if ungraded:
            raise ValueError(f"a bench compares scores, but {', '.join(ungraded)} has no success check")
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("each task may appear once in a bench")
        if not self.tasks or not self.models or not self.seeds:
            raise ValueError("a bench needs at least one task, one model and one seed")

    def cells(self) -> list[tuple[int | None, Task, ModelSpec]]:
        """Seed-major order, so a batch stopped early still covers every model about equally."""
        return [(seed, task, m) for seed in self.seeds for task in self.tasks for m in self.models]

    def to_dict(self) -> dict:
        return {"tasks": [t.id for t in self.tasks], "task_sources": {t.id: t.source for t in self.tasks},
                "models": [str(m) for m in self.models], "seeds": self.seeds,
                "policy": self.policy, "effort": self.effort, "max_steps": self.max_steps,
                "max_seconds": self.max_seconds, "keep_image_turns": self.keep_image_turns,
                "reasoning": self.reasoning, "depth": self.depth}


def new_batch_id() -> str:
    return f"bench-{time.strftime('%Y%m%d-%H%M%S')}"


def episode_id(batch: str, task: str, model: ModelSpec, seed: int | None) -> str:
    return f"{task}-{batch}-{model.slug}-s{'default' if seed is None else seed}"


MakeBody = Callable[[SessionConfig], Body]
MakeProvider = Callable[[SessionConfig, Body], Provider]


class Bench:
    """One batch. Call run(); call stop() from a signal handler to end it after an emergency stop."""

    def __init__(self, plan: BenchPlan, runs_root: Path, batch_id: str | None = None,
                 make_body: MakeBody | None = None, make_provider_fn: MakeProvider | None = None,
                 on_start: Callable[[int, int, dict], None] | None = None,
                 on_done: Callable[[int, int, dict], None] | None = None):
        self.plan = plan
        self.root = Path(runs_root)
        self.id = batch_id or new_batch_id()
        if not BATCH_ID.match(self.id):
            raise ValueError(f"invalid batch id {self.id!r}")
        self.make_body = make_body or _default_body
        self.make_provider = make_provider_fn or (lambda cfg, body: make_provider(cfg))
        self.on_start, self.on_done = on_start, on_done
        self.control: ControlChannel | None = None
        self.stopping = False
        self.manifest = {
            "version": 1, "id": self.id, "created": time.time(), "status": "running", "plan": plan.to_dict(),
            "episodes": [{"id": episode_id(self.id, task.id, m, seed), "task": task.id, "provider": m.provider,
                          "model": m.model, "seed": seed, "status": "pending"} for seed, task, m in plan.cells()],
        }

    @property
    def manifest_path(self) -> Path:
        return self.root / "batches" / f"{self.id}.json"

    def stop(self) -> None:
        """Stop the running episode with an emergency stop and skip the rest."""
        self.stopping = True
        if self.control is not None:
            self.control.estop("bench stopped")

    def run(self) -> dict:
        self._save()
        cells = self.plan.cells()
        for i, ((seed, task, model), entry) in enumerate(zip(cells, self.manifest["episodes"]), start=1):
            if self.stopping:
                break
            cfg = SessionConfig(task=task, provider=model.provider, model=model.model,
                                base_url=self.plan.base_url if model.provider == "openai" else None,
                                policy=self.plan.policy, effort=self.plan.effort, max_steps=self.plan.max_steps,
                                max_seconds=self.plan.max_seconds, keep_image_turns=self.plan.keep_image_turns,
                                seed=seed, out_dir=self.root / entry["id"], batch=self.id,
                                reasoning=self.plan.reasoning and model.provider == "openai", depth=self.plan.depth)
            entry["status"] = "running"
            self._save()
            if self.on_start:
                self.on_start(i, len(cells), entry)
            self._run_one(cfg, entry)
            self._save()
            if self.on_done:
                self.on_done(i, len(cells), entry)
        self.manifest["status"] = "stopped" if self.stopping else "finished"
        self._save()
        return self.manifest

    def _run_one(self, cfg: SessionConfig, entry: dict) -> None:
        started = time.time()
        body = None
        try:
            body = self.make_body(cfg)
            provider = self.make_provider(cfg, body)   # before anything is recorded: config errors leave no trace
            self.control = ControlChannel()
            run_session(cfg, provider=provider, control=self.control, body=body)
        except Exception as err:  # noqa: BLE001 - one broken episode must not end the batch
            entry.update(status="error", error=f"{type(err).__name__}: {err}")
        else:
            summary = summarize(cfg.out_dir) or {}
            entry.update({k: summary.get(k) for k in ("status", "reason", "turns", "seconds", "usage")})
            if summary.get("error"):
                entry["error"] = summary["error"]
        finally:
            self.control = None
            if body is not None and hasattr(body, "close"):
                body.close()   # here, on the creating thread (see MujocoTabletop.close)
        entry["wall_seconds"] = round(time.time() - started, 1)

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, default=str))
        os.replace(tmp, self.manifest_path)


def _default_body(cfg: SessionConfig) -> Body:
    return get_task(cfg.task).make_body(seed=cfg.seed, depth=cfg.depth)
