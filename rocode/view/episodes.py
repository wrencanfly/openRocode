"""Read recorded episodes (see ext/recorder.py) into the shapes the dashboard draws.

Works on finished and still-running episodes alike: the recorder flushes every event as it happens,
so a viewer in another process sees a live run by re-reading its directory.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..trace import read_events as _events
from ..trace import read_json as _json

EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,200}$")
IMAGE_NAME = re.compile(r"^o\d+_[A-Za-z0-9_-]+\.png$")


def episode_dir(root: Path, episode_id: str) -> Path | None:
    """The directory for an id, or None if the id is not a plain child of root holding an episode."""
    if not EPISODE_ID.match(episode_id or ""):
        return None
    d = root / episode_id
    return d if (d / "episode.jsonl").is_file() else None


def _status(start: dict, end: dict | None, evaluation: dict) -> str:
    if end is None:
        return "running"
    if end.get("reason") == "error":
        return "error"
    if end.get("reason") == "estop":
        return "stopped"
    if evaluation.get("success") is True:
        return "succeeded"
    if evaluation.get("success") is False:
        return "failed"
    return "ended"


def summarize(d: Path) -> dict | None:
    events = _events(d)
    start = next((e for e in events if e["type"] == "EpisodeStart"), None)
    if start is None:
        return None
    end = next((e for e in reversed(events) if e["type"] == "EpisodeEnd"), None)
    evaluation = _json(d / "evaluation.json")
    stats = (end or {}).get("stats", {})
    cfg = start.get("config", {})
    turns = sum(1 for e in events if e["type"] == "TurnStart")
    usage = stats.get("usage")
    if usage is None:  # still running: add up what each reply has used so far
        usage = {}
        for e in events:
            if e["type"] == "AssistantReply":
                for k, v in (e.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + (v or 0)
    return {
        "id": d.name,
        "task_id": cfg.get("task_id") or d.name.rsplit("-", 2)[0],
        "task": start.get("task", ""),
        "model": cfg.get("model", ""),
        "provider": cfg.get("provider", ""),
        "policy": cfg.get("policy"),
        "seed": cfg.get("seed"),
        "batch": cfg.get("batch"),
        "started": start.get("t"),
        "status": _status(start, end, evaluation),
        "reason": (end or {}).get("reason"),
        "claimed": evaluation.get("claimed"),
        "live": end is None,
        "turns": stats.get("turns", turns),
        "steps": stats.get("steps"),
        "seconds": stats.get("seconds") if end else round((events[-1]["t"] - start["t"]), 1),
        "usage": usage,
        "error": stats.get("error", ""),
    }


def list_episodes(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    out = [s for d in root.iterdir() if (d / "episode.jsonl").is_file() for s in [summarize(d)] if s]
    out.sort(key=lambda s: s["started"] or 0, reverse=True)
    return out


def load_episode(root: Path, episode_id: str) -> dict | None:
    d = episode_dir(root, episode_id)
    if d is None:
        return None
    events = _events(d)
    summary = summarize(d)
    if summary is None:
        return None
    t0 = summary["started"]
    observations: list[dict] = []
    readings: dict[str, dict] = {}
    turns: list[dict] = []
    before_first_turn: list[dict] = []
    calls: dict[str, dict] = {}
    notes: list[dict] = []
    end = None

    def rel(t):
        return round(t - t0, 2)

    for e in events:
        kind = e["type"]
        if kind == "ObservationTaken":
            observations.append({"id": e["observation_id"], "t": rel(e["t"]),
                                 "images": {c: f"files/{d.name}/observations/{e['observation_id']}_{c}.png"
                                            for c in e["cameras"]}})
        elif kind == "TurnStart":
            turns.append({"turn": e["turn"], "t": rel(e["t"]), "text": "", "calls": [], "notes": []})
        elif kind == "AssistantReply" and turns:
            turns[-1]["text"] = e.get("text", "")
            turns[-1]["reasoning"] = e.get("reasoning", "")
            turns[-1]["usage"] = e.get("usage", {})
        elif kind == "ToolStart":
            call = {"id": e["call_id"], "name": e["name"], "args": e.get("args", {}), "t": rel(e["t"]), "children": []}
            calls[e["call_id"]] = call
            parent = calls.get(e.get("parent_id") or "")
            if parent is not None:
                parent["children"].append(call)
            elif turns:
                turns[-1]["calls"].append(call)
            else:
                before_first_turn.append(call)
        elif kind == "ToolEnd" and e["call_id"] in calls:
            data = e.get("data") or {}
            calls[e["call_id"]]["result"] = {
                "text": e.get("text", ""), "error": e.get("is_error", False), "obs": e.get("observation_ids", []),
                "duration": e.get("duration_s"), "stop": (e.get("receipt") or {}).get("stop_reason"), "data": data}
            oid = data.get("observation")
            if oid and e["name"] != "run_code" and "height_cm" in data:
                readings[oid] = {k: data.get(k) for k in ("height_cm", "opening_cm", "holding", "contact_n")}
        elif kind == "ToolDenied" and e["call_id"] in calls:
            calls[e["call_id"]]["denied"] = e.get("reason", "")
        elif kind in ("OperatorInput", "Paused", "Resumed", "EStop", "Notice"):
            note = {"kind": kind, "t": rel(e["t"]), "text": e.get("text") or e.get("source") or ""}
            (turns[-1]["notes"] if turns else notes).append(note)
        elif kind == "EpisodeEnd":
            end = e

    for o in observations:
        o["readings"] = readings.get(o["id"])
    evaluation = _json(d / "evaluation.json")
    return {
        **summary,
        "finish": ((end or {}).get("stats") or {}).get("finish"),
        "reason": (end or {}).get("reason"),
        "success": evaluation.get("success"),
        "claimed": evaluation.get("claimed"),
        "observations": observations,
        "setup": before_first_turn,
        "turns": turns,
        "notes": notes,
    }
