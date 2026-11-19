"""Compare recorded episodes: success rate and cost per task and model, from traces and batch manifests.

Pure reading, no simulator imports, so the dashboard runs anywhere the runs directory is readable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median

BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
SCORED = {"succeeded", "failed", "ended"}   # the model ran until done or a budget, then ground truth judged

WHY = {
    "done": "Called done, but the ground-truth check failed",
    "budget_steps": "Ran out of steps",
    "budget_turns": "Ran out of turns",
    "budget_time": "Ran out of time",
    "refusal": "The model refused",
    "max_tokens": "Hit the output token limit",
    "estop": "Emergency stop",
}


def load_manifests(runs_root: Path) -> list[dict]:
    d = Path(runs_root) / "batches"
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("*.json"):
        try:
            m = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(m, dict) and BATCH_ID.match(str(m.get("id", ""))) and isinstance(m.get("episodes"), list):
            out.append(m)
    out.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return out


def why(e: dict) -> str:
    status = e.get("status")
    if status == "succeeded":
        return ""
    if status == "error":
        return e.get("error") or "The episode ended with an error"
    if status in ("pending", "running"):
        return ""
    return WHY.get(e.get("reason") or "", e.get("reason") or "")


def _cell(episodes: list[dict]) -> dict:
    # The rate counts only episodes the model played to a verdict. Harness or API errors and emergency
    # stops say nothing about the model, so they are reported beside it, not counted against it.
    done = [e for e in episodes if e["status"] in SCORED]
    ok = [e for e in done if e["status"] == "succeeded"]
    usage = [e.get("usage") or {} for e in done]
    return {
        "n": len(episodes),
        "scored": len(done),
        "succeeded": len(ok),
        "errors": sum(e["status"] == "error" for e in episodes),
        "stopped": sum(e["status"] == "stopped" for e in episodes),
        "running": sum(e["status"] == "running" for e in episodes),
        "pending": sum(e["status"] == "pending" for e in episodes),
        "rate": round(len(ok) / len(done), 3) if done else None,
        "turns_median": median([e["turns"] for e in done if e.get("turns") is not None] or [0]) if done else None,
        "seconds_median": median([e["seconds"] for e in done if e.get("seconds") is not None] or [0]) if done else None,
        "tokens_in_mean": round(mean([u.get("input_tokens") or 0 for u in usage])) if done else None,
        "tokens_out_mean": round(mean([u.get("output_tokens") or 0 for u in usage])) if done else None,
        "episodes": [{"id": e["id"], "status": e["status"], "seed": e.get("seed"), "why": why(e),
                      "turns": e.get("turns"), "seconds": e.get("seconds"), "has_trace": e.get("has_trace", True)}
                     for e in sorted(episodes, key=lambda e: (e.get("seed") is None, e.get("seed") or 0, e["id"]))],
    }


def _column_key(e: dict, by_policy: bool) -> str:
    return f"{e.get('model') or '?'}|{e.get('policy') or ''}" if by_policy else (e.get("model") or "?")


def compare(summaries: list[dict], manifests: list[dict], batch: str | None = None) -> dict:
    """Group episodes into a task x model matrix. With a batch id, only that batch, including pending runs."""
    manifest = next((m for m in manifests if m.get("id") == batch), None) if batch else None
    eps = [dict(s) for s in summaries if not batch or s.get("batch") == batch]
    if manifest is not None:
        seen = {e["id"]: e for e in eps}
        policy = (manifest.get("plan") or {}).get("policy")
        for m in manifest["episodes"]:
            e = seen.get(m.get("id"))
            if e is None:
                if m.get("status") in ("pending", "running", "error"):
                    eps.append({"id": m["id"], "task_id": m.get("task"), "model": m.get("model"), "policy": policy,
                                "seed": m.get("seed"), "status": "pending" if m.get("status") == "running"
                                and manifest.get("status") != "running" else m.get("status"),
                                "error": m.get("error", ""), "has_trace": False})
            elif e["status"] == "running" and manifest.get("status") != "running":
                e["status"], e["error"] = "error", m.get("error") or "The batch ended before this episode did"

    by_policy = len({e.get("policy") for e in eps}) > 1
    plan = (manifest or {}).get("plan") or {}
    task_order = list(plan.get("tasks") or [])
    col_order = [s.partition(":")[2] for s in plan.get("models") or []]
    for e in sorted(eps, key=lambda e: e.get("started") or 0):
        if e.get("task_id") not in task_order:
            task_order.append(e.get("task_id"))
    columns: dict[str, dict] = {}
    for model in col_order:
        if not by_policy:
            columns[model] = {"key": model, "model": model, "policy": None}
    for e in sorted(eps, key=lambda e: (e.get("model") or "", e.get("policy") or "")):
        key = _column_key(e, by_policy)
        columns.setdefault(key, {"key": key, "model": e.get("model"), "policy": e.get("policy") if by_policy else None})

    cells: dict[str, dict[str, dict]] = {}
    for task in task_order:
        row = {}
        for key in columns:
            group = [e for e in eps if e.get("task_id") == task and _column_key(e, by_policy) == key]
            if group:
                row[key] = _cell(group)
        cells[task] = row
    totals = {key: _cell([e for e in eps if _column_key(e, by_policy) == key]) for key in columns}

    batches = [{"id": m["id"], "created": m.get("created"), "status": m.get("status"),
                "planned": len(m["episodes"]),
                "done": sum(x.get("status") not in ("pending", "running") for x in m["episodes"]),
                "tasks": (m.get("plan") or {}).get("tasks", []), "models": (m.get("plan") or {}).get("models", []),
                "seeds": (m.get("plan") or {}).get("seeds", [])} for m in manifests]
    return {"batch": batch, "batches": batches, "tasks": task_order, "columns": list(columns.values()),
            "cells": cells, "totals": totals, "episodes": len(eps)}
