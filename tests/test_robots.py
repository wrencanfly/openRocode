"""Robots other than the floating gripper: a Panda arm, two ALOHA arms and a LEAP hand, all through one Body."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("robot_descriptions")

from rocode.body.base import CancelToken, Guard
from rocode.body.mujoco_tabletop import MujocoTabletop
from rocode.demo import scripted_pick_and_place
from rocode.ext.pointing import Pointing
from rocode.scene import DEFAULT_SCENE, Scene, make_object
from rocode.session import SessionConfig, run_session
from rocode.tasks import Catalog, Task

EXAMPLES = Path(__file__).parent.parent / "examples" / "tasks"


@pytest.fixture
def make():
    bodies = []

    def build(robot, objects=DEFAULT_SCENE.objects, seed=None):
        bodies.append(MujocoTabletop(scene=Scene(tuple(objects), robot), seed=seed))
        return bodies[-1]
    yield build
    for b in bodies:
        b.close()


@pytest.mark.parametrize("robot, arms, cameras", [("panda", ("arm",), ("front", "side")),
                                                  ("aloha", ("left", "right"), ("front", "top")),
                                                  ("leap", ("hand",), ("front", "side"))])
def test_each_robot_builds_with_its_arms_and_cameras(make, robot, arms, cameras):
    body = make(robot)
    info = body.info()
    assert info.arm_names == arms and info.cameras == cameras and info.robot
    obs = body.observe()
    assert set(obs.proprio.arms) == set(arms) and all(img.mean() > 5 for img in obs.rgb.values())


def test_a_jointed_arm_reaches_points_and_refuses_what_it_cannot_reach(make):
    body, c = make("aloha", ()), CancelToken()
    r = body.move([-0.12, -0.10, 0.08], Guard(), c, arm="left")
    assert r.stop_reason == "reached" and np.linalg.norm(body.arms["left"].tcp() - [-0.12, -0.10, 0.08]) < 0.01
    far = body.move([0.30, 0.0, 0.05], Guard(), c, arm="left")      # the far end of the table: the right arm's side
    assert far.stop_reason == "refused" and "left arm cannot reach" in far.detail


def test_tools_take_an_arm_only_when_there_is_more_than_one():
    single = {t.name: t.input_schema for t in Pointing(("front",)).tools}
    double = {t.name: t.input_schema for t in Pointing(("front",), arms=("left", "right")).tools}
    assert "arm" not in single["move_to"]["properties"]
    assert double["move_to"]["properties"]["arm"]["enum"] == ["left", "right"]
    assert "arm" in double["grip"]["required"] and "arm" not in double["look"]["properties"]


def test_the_leap_hand_lifts_objects_of_different_shapes(make):
    c = CancelToken()
    for shape, size in (("box", 4), ("cylinder", [5, 8]), ("sphere", 5)):
        body = make("leap", (make_object("thing", shape, size, "red", [5, -5], jitter_cm=[0, 0]),))
        start = body.data.body("thing").xpos.copy()
        body.move([start[0], start[1], 0.15], Guard(), c)
        body.move([start[0], start[1], 0.004], Guard(max_force_n=3, speed_mps=0.08), c)
        body.grip(0.0, c)
        body.move(body.arms["hand"].tcp() + [0, 0, 0.12], Guard(max_force_n=None), c)   # the grasp closes as it lifts
        assert body.data.body("thing").xpos[2] - start[2] > 0.08, shape
        assert body.ground_truth()["held"] == "thing" and body.observe().proprio.opening_m > 0.004


SOLUTIONS = {
    ("panda", "mug_to_coaster"): [("mug", "coaster")],
    ("panda", "stack_three"): [("blue_block", "green_block"), ("red_block", "blue_block")],
    ("aloha", "both_blocks_to_tray"): [("red_block", "tray"), ("blue_block", "tray")],
    ("aloha", "yellow_to_pad"): [("yellow_block", "mat"), ("yellow_block", "pad")],   # a handover in the middle
    ("leap", "block_to_pad"): [("block", "pad")],
    ("leap", "can_to_tray"): [("can", "tray")],
}


@pytest.mark.parametrize("robot, task_id", SOLUTIONS)
def test_tasks_are_solvable_with_each_robot(robot, task_id, tmp_path):
    t = Catalog.load(sorted(EXAMPLES.glob("*.toml"))).resolve(task_id)
    task = Task(t.id, t.instruction, t.success, t.scene.with_robot(robot), t.source)
    body = task.make_body(seed=1)
    try:
        report = run_session(SessionConfig(task=task, provider="scripted", policy="code", seed=1, out_dir=tmp_path),
                             provider=scripted_pick_and_place(body, SOLUTIONS[(robot, task_id)]), body=body)
    finally:
        body.close()
    assert report.success, [c.to_dict() for c in report.extra["clauses"]]
    spec = json.loads((tmp_path / "episode.jsonl").read_text().splitlines()[0])["config"]["task_spec"]
    assert spec["scene"]["robot"] == robot


def test_robot_scenes_keep_object_angles_and_arm_joint_ranges(make):
    """MuJoCo 3.3.0 converts an attached robot's joint ranges with the parent's angle unit; robot scenes are
    therefore written in radians, and objects must keep their yaw."""
    gripper, panda = make("gripper"), make("panda")
    assert np.allclose(gripper.data.body("block").xquat, panda.data.body("block").xquat, atol=1e-6)
    ranges = panda.model.jnt_range[panda.arms["arm"].joints]
    assert ranges[0, 1] > 2.8   # joint 1 of a Panda turns ±2.9 rad
