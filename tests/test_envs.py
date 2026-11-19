"""Benchmark environments (robosuite, Meta-World) as bodies: their scenes and success checks, our tools."""

import importlib.util
import json

import numpy as np
import pytest

from rocode import envs
from rocode.body.base import CancelToken, Guard
from rocode.providers.base import Request
from rocode.providers.scripted import ScriptedProvider
from rocode.session import SessionConfig, run_session
from rocode.tasks import Catalog, env_task
from rocode.view.episodes import load_episode

HAVE = {name: importlib.util.find_spec(name) is not None for name in ("robosuite", "metaworld")}
needs = {name: pytest.mark.skipif(not ok, reason=f"{name} is not installed") for name, ok in HAVE.items()}


def test_env_tasks_resolve_without_importing_the_simulator():
    task = Catalog.load().resolve("robosuite:lift")
    assert task.env == "robosuite:lift" and task.success == "env_success()"
    assert task.spec()["env"] == "robosuite:lift" and "scene" not in task.spec()
    assert len(envs.tasks("metaworld")) >= 10
    with pytest.raises(ValueError, match="robosuite has no task 'lfit'"):
        env_task("robosuite:lfit")
    with pytest.raises(ValueError, match="unknown environment 'isaac'"):
        Catalog.load().resolve("isaac:lift")


@pytest.mark.skipif(HAVE["robosuite"], reason="robosuite is installed")
def test_a_missing_simulator_says_how_to_install_it():
    with pytest.raises(ValueError, match=r"pip install 'openrocode\[robosuite\]'"):
        Catalog.load().resolve("robosuite:lift").make_body()


def pick_and_place(body, pick, place=None, lift_cm=15):
    """Straight-line pick (and place) through the Body, as the pointing tools would move it."""
    c, arm = CancelToken(), body.arms["arm"]
    body.move(pick + [0, 0, 0.08], Guard(), c)
    body.move(pick + [0, 0, -0.005], Guard(max_force_n=3, speed_mps=0.08), c)
    body.grip(0.0, c)
    body.move(arm.tcp() + [0, 0, lift_cm / 100], Guard(), c)
    if place is not None:
        body.move(place, Guard(), c)


@needs["robosuite"]
@pytest.mark.parametrize("task_id", ["robosuite:lift", "robosuite:stack"])
def test_robosuite_tasks_are_solved_through_the_body(task_id):
    task = Catalog.load().resolve(task_id)
    body = task.make_body(seed=3)
    try:
        objs = body.ground_truth()["objects"]
        assert task.evaluate(body.ground_truth())[0] is False
        if task_id.endswith("lift"):
            pick_and_place(body, objs["cube"])
        else:
            pick_and_place(body, objs["cubeA"], objs["cubeB"] + [0, 0, 0.10])
            body.move(objs["cubeB"] + [0, 0, 0.0], Guard(max_force_n=3, speed_mps=0.08), CancelToken())
            body.grip(0.08, CancelToken())
            body.move(body.arms["arm"].tcp() + [0, 0, 0.08], Guard(), CancelToken())
        ok, clauses = task.evaluate(body.ground_truth())
        assert ok and clauses[0].expr == "env_success()"
    finally:
        body.close()


@needs["metaworld"]
def test_metaworld_pick_place_is_solved_and_the_wrist_does_not_turn():
    task = Catalog.load().resolve("metaworld:pick-place-v3")
    body = task.make_body(seed=0)
    try:
        assert body.info().cameras == ("front", "side") and not body.info().arms[0].turns
        assert body.rotate(1.0, CancelToken()).stop_reason == "refused"
        gt = body.ground_truth()
        pick_and_place(body, gt["objects"]["obj"], gt["goal"])
        assert task.evaluate(body.ground_truth())[0] is True
    finally:
        body.close()


@needs["robosuite"]
def test_an_env_episode_is_traced_like_any_other(tmp_path):
    """The model's side: pointing tools on rendered pictures, then the trace and the verdict."""
    task = Catalog.load().resolve("robosuite:lift")
    body = task.make_body(seed=1)

    def program(request: Request):
        stored = request.images.latest()
        cube = body.ground_truth()["objects"]["cube"]
        x, y = (round(v) for v in stored.observation.cameras["agentview"].project(cube + [0, 0, 0.02]))
        code = f"""
robot.move_to("agentview", {x}, {y}, hover_cm=6, observation="{stored.id}")
robot.lower()
robot.grip("close")
r = robot.nudge("agentview", up_cm=15)
print("holding:", r.holding)
"""
        return [("run_code", {"code": code})]

    provider = ScriptedProvider([program, [("done", {"success": True, "summary": "Lifted."})]])
    try:
        report = run_session(SessionConfig(task=task, provider="scripted", policy="code", seed=1,
                                           out_dir=tmp_path / "e"), provider=provider, body=body)
    finally:
        body.close()
    assert report.success
    start = json.loads((tmp_path / "e" / "episode.jsonl").read_text().splitlines()[0])
    assert start["config"]["task_spec"]["env"] == "robosuite:lift" and start["config"]["body"] == "robosuite-Lift"
    ep = load_episode(tmp_path, "e")
    assert ep["status"] == "succeeded" and ep["clauses"][0]["expr"] == "env_success()"
    assert np.all([o["readings"] for o in ep["observations"]])


@pytest.mark.parametrize("task_id", [pytest.param("robosuite:lift", marks=needs["robosuite"]),
                                     pytest.param("metaworld:push-v3", marks=needs["metaworld"])])
def test_a_seed_always_gives_the_same_layout(task_id):
    """Every model in a bench must see the same tables, so environments are rebuilt from the seed."""
    def layout(body):
        return {k: np.round(v, 5).tolist() for k, v in body.ground_truth()["objects"].items()}

    a, b = (Catalog.load().resolve(task_id).make_body(seed=7) for _ in range(2))
    try:
        first = layout(a)
        assert layout(b) == first
        a.reset(8)
        assert layout(a) != first
        a.reset(7)
        assert layout(a) == first
    finally:
        a.close()
        b.close()
