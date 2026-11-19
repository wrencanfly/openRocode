"""RGB-only cameras: one click lands on the table, two clicks of the same spot place it anywhere."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from rocode.body.mujoco_tabletop import MujocoTabletop
from rocode.core.observations import ObservationStore
from rocode.demo import scripted_pick_and_place
from rocode.ext.pointing import Pointing, locate
from rocode.session import SessionConfig, run_session
from rocode.tasks import Catalog

EXAMPLES = Path(__file__).parent.parent / "examples" / "tasks"


@pytest.fixture
def rgb_body():
    body = MujocoTabletop(depth=False)
    yield body
    body.close()


def stored(body):
    store = ObservationStore()
    return store.put(body.observe(), {})


def pixel(obs, camera, point):
    return tuple(round(v) for v in obs.cameras[camera].project(point))


def test_rgb_only_cameras_carry_no_depth(rgb_body):
    assert rgb_body.info().has_depth is False and rgb_body.observe().depth is None


def test_one_click_lands_on_the_table_and_two_clicks_place_a_point_in_the_air(rgb_body):
    s = stored(rgb_body)
    obs = s.observation
    coaster = rgb_body.ground_truth()["objects"]["coaster"]
    on_table = coaster * [1, 1, 0]
    point, note = locate(s, ("front", *pixel(obs, "front", on_table)), None, 0.0)
    assert np.linalg.norm(point - on_table) < 0.01 and "No depth" in note

    mug_top = rgb_body.ground_truth()["objects"]["mug"] + [0, 0, 0.045]
    point, note = locate(s, ("front", *pixel(obs, "front", mug_top)), ("side", *pixel(obs, "side", mug_top)), 0.0)
    assert np.linalg.norm(point - mug_top) < 0.01 and "two views" in note


def test_two_clicks_on_different_spots_are_refused_with_where_to_look(rgb_body):
    s = stored(rgb_body)
    obs = s.observation
    gt = rgb_body.ground_truth()["objects"]
    mug_top, block_top = gt["mug"] + [0, 0, 0.045], gt["block"] + [0, 0, 0.02]
    point, note = locate(s, ("front", *pixel(obs, "front", mug_top)), ("side", *pixel(obs, "side", block_top)), 0.0)
    assert point is None and "do not show the same spot" in note
    u, v = (int(n) for n in re.search(r"is \((\d+), (\d+)\)", note).groups())
    # the hint is a pixel on the front click's line of sight, so clicking it there lines the two up
    point, note = locate(s, ("front", *pixel(obs, "front", mug_top)), ("side", u, v), 0.0)
    assert point is not None and "two views" in note


def test_only_rgb_only_tools_ask_for_a_second_click():
    with_depth = {t.name: t.input_schema for t in Pointing(("front", "side")).tools}
    without = {t.name: t.input_schema for t in Pointing(("front", "side"), depth=False).tools}
    assert "camera2" not in with_depth["move_to"]["properties"]
    assert {"camera2", "x2", "y2"} <= set(without["move_to"]["properties"])
    assert "no depth" in Pointing(("front", "side"), depth=False).prompt


@pytest.mark.parametrize("task_id, moves", [
    ("stack_block_on_mug", [("block", "mug")]),
    ("stack_three", [("blue_block", "green_block"), ("red_block", "blue_block")]),
])
def test_tasks_are_solvable_without_depth(task_id, moves, tmp_path):
    task = Catalog.load(sorted(EXAMPLES.glob("*.toml"))).resolve(task_id)
    body = task.make_body(seed=1, depth=False)
    try:
        report = run_session(SessionConfig(task=task, provider="scripted", policy="code", seed=1, depth=False,
                                           out_dir=tmp_path), provider=scripted_pick_and_place(body, moves), body=body)
    finally:
        body.close()
    assert report.success, [c.to_dict() for c in report.extra["clauses"]]
    start = json.loads((tmp_path / "episode.jsonl").read_text().splitlines()[0])
    assert start["config"]["depth"] is False
    assert json.loads((tmp_path / "evaluation.json").read_text())["config"]["depth"] is False
