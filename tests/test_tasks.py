"""Task definitions: scenes, success expressions, task files, and that the shipped examples are solvable."""

import json
from pathlib import Path

import numpy as np
import pytest

from rocode.body.mujoco_tabletop import MujocoTabletop
from rocode.checks import Check, CheckError
from rocode.cli import main
from rocode.demo import scripted_pick_and_place
from rocode.providers.scripted import ScriptedProvider
from rocode.scene import DEFAULT_SCENE, Scene, make_object
from rocode.session import SessionConfig, run_session
from rocode.tasks import TASKS, Catalog, load_file
from rocode.view.episodes import load_episode, summarize

EXAMPLES = Path(__file__).parent.parent / "examples" / "tasks"
SCENE = Scene((make_object("cube", "box", 4, "red", [0, 0]), make_object("pad", "box", [10, 10, 1], "green", [20, 0],
                                                                         fixed=True)))


def gt(cube=(0, 0, 2), pad=(20, 0, 0.5), held=None, tilt=0.0):
    return {"objects": {"cube": np.array(cube) / 100, "pad": np.array(pad) / 100}, "held": held,
            "tilt_deg": {"cube": tilt}}


# ---- scenes -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"shape": "cone"}, "not one of"),
    ({"at_cm": [44, 0]}, "does not fit on the table"),
    ({"color": "teal"}, "color 'teal'"),
    ({"name": "gripper"}, "reserved"),
    ({"shape": "mug", "size_cm": [6, 9], "fixed": True}, "cannot be fixed"),
    ({"size_cm": [4, 4]}, "size_cm for a box"),
])
def test_scene_objects_are_validated(kwargs, message):
    args = {"name": "thing", "shape": "box", "size_cm": 4, "color": "red", "at_cm": [0, 0], **kwargs}
    with pytest.raises(ValueError, match=message):
        Scene((make_object(**args),))


def test_custom_scene_builds_and_reports_every_object():
    body = MujocoTabletop(scene=SCENE, seed=3)
    try:
        truth = body.ground_truth()
        assert set(truth["objects"]) == {"cube", "pad"} and set(truth["tilt_deg"]) == {"cube"}
        assert abs(truth["objects"]["cube"][2] - 0.02) < 0.002          # resting on the table
        assert np.allclose(truth["objects"]["pad"][:2], [0.20, 0.0])    # fixed objects do not jitter
    finally:
        body.close()


def test_default_scene_keeps_the_original_layout():
    assert DEFAULT_SCENE.names() == ["mug", "block", "coaster"]
    body = MujocoTabletop()
    try:
        truth = body.ground_truth()
        assert np.allclose(truth["objects"]["mug"][:2], [0.10, 0.10], atol=1e-4)
        assert np.allclose(truth["objects"]["block"][:2], [0.12, -0.12], atol=1e-4)
    finally:
        body.close()


# ---- success expressions ----------------------------------------------------------------------------

@pytest.mark.parametrize("expr, message", [
    ("on(cube, nowhere)", "no object named 'nowhere'"),
    ("inside(cube, pad)", "unknown predicate 'inside'"),
    ("height_cm(cube)", "is a measurement"),
    ("height_cm(cube) == 3", "only <, <=, > and >="),
    ("on(cube)", "takes 2 object name"),
    ("__import__('os')", "unknown predicate"),
    ("on(cube, pad) and", "not a valid expression"),
])
def test_bad_expressions_fail_when_loaded(expr, message):
    with pytest.raises(CheckError, match=message):
        Check.parse(expr, SCENE)


def test_on_means_resting_on_top_and_released():
    check = Check.parse("on(cube, pad)", SCENE)
    assert check.evaluate(gt(cube=(21, 1, 3)))[0]                     # on top, near the centre
    assert not check.evaluate(gt(cube=(21, 1, 3), held="cube"))[0]    # still in the hand
    assert not check.evaluate(gt(cube=(27, 0, 3)))[0]                 # centre past the pad's edge
    assert not check.evaluate(gt(cube=(20, 0, 12)))[0]                # hovering above it
    assert Check.parse("on(cube, table)", SCENE).evaluate(gt())[0]


def test_each_clause_reports_what_it_read():
    ok, clauses = Check.parse("on(cube, pad) and height_cm(cube) > 10 and upright(cube)", SCENE).evaluate(
        gt(cube=(21, 0, 3), tilt=40))
    assert not ok
    assert [(c.expr, c.ok) for c in clauses] == [("on(cube, pad)", True), ("height_cm(cube) > 10", False),
                                                 ("upright(cube)", False)]
    assert clauses[1].values == {"height_cm(cube)": 3.0}


# ---- task files and the catalog ---------------------------------------------------------------------

def test_toml_task_file(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text('[[object]]\nname = "cube"\nshape = "box"\nsize_cm = 4\ncolor = "red"\nat_cm = [0, 0]\n\n'
                 '[[task]]\nid = "lift_cube"\ninstruction = "Lift the cube."\nsuccess = "height_cm(cube) > 8"\n')
    scene, tasks = load_file(f)
    assert scene.names() == ["cube"] and tasks[0].scene is scene and tasks[0].source == str(f)
    assert tasks[0].spec()["success"] == "height_cm(cube) > 8"


@pytest.mark.parametrize("body, message", [
    ('[[task]]\nid = "t"\ninstruction = "x"\nsucess = "held(block)"\n', "unknown key"),
    ('[[task]]\nid = "t"\ninstruction = "x"\nsuccess = "held(cube)"\n', "no object named 'cube'"),
    ('[[task]]\nid = "Bad Id"\ninstruction = "x"\n', "task id"),
    ('[scene]\n', "unknown section"),
])
def test_task_file_errors_name_the_problem(tmp_path, body, message):
    f = tmp_path / "t.toml"
    f.write_text(body)
    with pytest.raises(ValueError, match=message) as err:
        load_file(f)
    assert str(f) in str(err.value)


def test_python_task_file_with_a_function_check(tmp_path):
    f = tmp_path / "mine.py"
    f.write_text("from rocode.tasks import Task\n"
                 "def mug_far_right(gt):\n    return gt['objects']['mug'][0] > 0.2\n"
                 "TASKS = [Task('push_mug_right', 'Push the mug to the right edge.', mug_far_right)]\n")
    task = Catalog.load([f]).resolve("push_mug_right")
    assert task.source == str(f) and task.spec()["success"] == "python:mug_far_right"


def test_catalog_resolves_ids_and_typed_instructions(tmp_path):
    catalog = Catalog.load([EXAMPLES / "sorting.toml"])
    assert catalog.resolve("mug_to_coaster") is TASKS["mug_to_coaster"]
    typed = catalog.resolve("Line the blocks up along the front edge")
    assert typed.id == "custom" and not typed.graded and typed.scene.names()[0] == "tray"
    with pytest.raises(ValueError, match="unknown task 'mug_to_tray'"):
        catalog.resolve("mug_to_tray")
    with pytest.raises(ValueError, match="already defined"):
        Catalog.load([EXAMPLES / "sorting.toml", EXAMPLES / "sorting.toml"])


# ---- every shipped task is unsolved at the start and solvable --------------------------------------

def all_tasks():
    catalog = Catalog.load(sorted(EXAMPLES.glob("*.toml")))
    tasks = list(catalog.tasks.values())
    try:
        import robot_descriptions  # noqa: F401
    except ImportError:   # arm and hand models come from MuJoCo Menagerie
        tasks = [t for t in tasks if t.scene.robot == "gripper"]
    return tasks


@pytest.mark.parametrize("task", all_tasks(), ids=lambda t: t.id)
def test_no_task_starts_solved(task):
    for seed in (None, 0, 1):
        body = MujocoTabletop(scene=task.scene, seed=seed)
        try:
            assert task.evaluate(body.ground_truth())[0] is False
        finally:
            body.close()


SOLUTIONS = {   # tasks for other robots are solved in test_robots.py
    "sort_blocks": [("red_block", "tray"), ("blue_block", "tray"), ("green_block", "pad")],
    "stack_three": [("blue_block", "green_block"), ("red_block", "blue_block")],
    "red_block_to_coaster": [("red_block", "coaster")],
    "block_on_can": [("blue_block", "can")],
}


@pytest.mark.parametrize("task_id", SOLUTIONS)
def test_example_tasks_are_solvable(task_id, tmp_path):
    task = Catalog.load(sorted(EXAMPLES.glob("*.toml"))).resolve(task_id)
    body = MujocoTabletop(scene=task.scene, seed=1)
    try:
        report = run_session(SessionConfig(task=task, provider="scripted", policy="code", seed=1, out_dir=tmp_path),
                             provider=scripted_pick_and_place(body, SOLUTIONS[task_id]), body=body)
    finally:
        body.close()
    assert report.success, [c.to_dict() for c in report.extra["clauses"]]
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    assert evaluation["check"] == task.success and all(c["ok"] for c in evaluation["clauses"])


# ---- traces ------------------------------------------------------------------------------------------

def test_typed_instruction_is_recorded_but_not_graded(tmp_path, body):
    task = Catalog.load().resolve("Wave at the camera")
    provider = ScriptedProvider([[("done", {"success": True, "summary": "Waved."})]])
    report = run_session(SessionConfig(task=task, provider="scripted", out_dir=tmp_path / "custom-1"),
                         provider=provider, body=body)
    assert report.extra["graded"] is False
    s = summarize(tmp_path / "custom-1")
    assert s["status"] == "ungraded" and s["task_id"] == "custom"
    start = json.loads((tmp_path / "custom-1" / "episode.jsonl").read_text().splitlines()[0])
    spec = start["config"]["task_spec"]
    assert spec["instruction"] == "Wave at the camera" and spec["success"] is None and spec["source"] == "inline"
    assert [o["name"] for o in spec["scene"]["objects"]] == ["mug", "block", "coaster"]


def test_failed_episode_says_which_condition_failed(tmp_path, body):
    provider = ScriptedProvider([[("done", {"success": True, "summary": "Done, I think."})]])
    run_session(SessionConfig(task="stack_block_on_mug", provider="scripted", out_dir=tmp_path / "e"),
                provider=provider, body=body)
    ep = load_episode(tmp_path, "e")
    assert ep["status"] == "failed" and ep["check"] == "on(block, mug)"
    assert ep["failed_checks"] == ["on(block, mug)"] and ep["clauses"][0]["ok"] is False


def test_cli_lists_and_previews_task_files(tmp_path, capsys):
    assert main(["tasks", "--task-file", str(EXAMPLES / "sorting.toml")]) == 0
    out = capsys.readouterr().out
    assert "sort_blocks" in out and "on(red_block, tray)" in out
    assert main(["preview", str(EXAMPLES / "distractors.toml"), "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["distractors-front.png", "distractors-side.png"]
    with pytest.raises(SystemExit):
        main(["tasks", "--task-file", str(tmp_path / "missing.toml")])
