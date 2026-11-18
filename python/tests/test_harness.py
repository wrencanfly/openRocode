"""Loop, pointing tools, code-as-policy, control channel. No API key needed: models are scripted."""

import threading
import time

import numpy as np

from conftest import make_agent, pixel, start
from rocode.core.events import ToolEnd
from rocode.core.extension import Deny, Extension
from rocode.core.loop import Budget, LoopConfig
from rocode.core.messages import OperatorMessage, Text, ToolResultsMessage
from rocode.core.tooling import validate
from rocode.demo import scripted_demo
from rocode.ext.pointing import surface_point
from rocode.tasks import COASTER_XY, TASKS


def mug_top(body):
    return body.ground_truth()["objects"]["mug"] + [0, 0, 0.045]


def results_of(agent):
    return [r for m in agent.messages if isinstance(m, ToolResultsMessage) for r in m.results]


def text(block):
    return "\n".join(c.text for c in block.content if isinstance(c, Text))


# ---- geometry ----------------------------------------------------------------------------------

def test_clicking_a_pixel_recovers_the_surface_point(body):
    obs = body.observe()
    for name, cam in obs.cameras.items():
        x, y = (round(v) for v in cam.project(mug_top(body)))
        assert np.linalg.norm(cam.deproject(x, y, obs.depth[name][y, x]) - mug_top(body)) < 0.004


def test_clicks_use_the_pictures_they_were_made_on(body):
    """After the gripper moves over the mug, a click on the old pictures still means the mug, not the gripper."""
    def step1(req):  # closed fingers hovering just above the mug hide its top from the front camera
        x, y = pixel(req, mug_top(body))
        return [("grip", {"action": "close"}), ("move_to", {"camera": "front", "x": x, "y": y, "hover_cm": 1})]

    clicked = {}

    def step2(req):
        old = req.images.get("o1")
        x, y = (round(v) for v in old.observation.cameras["front"].project(mug_top(body)))
        clicked["latest"] = surface_point(req.images.latest(), "front", x, y, 0.0)
        clicked["xy"] = (x, y)
        return [("move_to", {"camera": "front", "x": x, "y": y, "observation": "o1"})]

    agent, provider, pointing = make_agent(body, [step1, step2, [("done", {"success": True, "summary": ""})]])
    start(agent, pointing)
    assert np.linalg.norm(pointing.state.last_target - mug_top(body)) < 0.005
    # The same pixel in the newest pictures lands on the fingers, above the mug.
    assert clicked["latest"][2] > mug_top(body)[2] + 0.005


# ---- pointing tools through the loop -----------------------------------------------------------

def test_direct_tool_calls_complete_a_pick_and_place(body):
    def hover_mug(req):
        x, y = pixel(req, mug_top(body))
        return [("move_to", {"camera": "front", "x": x, "y": y})]

    def go_to_coaster(req):
        x, y = pixel(req, [*COASTER_XY, 0.006])
        return [("move_to", {"camera": "front", "x": x, "y": y, "hover_cm": 10})]

    steps = [hover_mug, [("turn", {"degrees": 90})], [("lower", {}), ("grip", {"action": "close"})],
             [("nudge", {"camera": "front", "up_cm": 12})], go_to_coaster,
             [("lower", {}), ("grip", {"action": "open"}), ("nudge", {"camera": "side", "up_cm": 8})],
             [("done", {"success": True, "summary": "placed"})]]
    agent, _, pointing = make_agent(body, steps, policy="tools")
    outcome = start(agent, pointing)
    assert outcome.reason == "done" and outcome.finish["success"]
    assert TASKS["mug_to_coaster"].check(body.ground_truth())
    assert "run_code" not in agent.tools


def test_results_never_reveal_world_coordinates(body):
    def hover(req):
        x, y = pixel(req, mug_top(body))
        return [("move_to", {"camera": "front", "x": x, "y": y}), ("lower", {})]

    agent, _, pointing = make_agent(body, [hover, [("done", {"success": False, "summary": ""})]])
    start(agent, pointing)
    gt = body.ground_truth()
    values = [*gt["tcp"], *gt["objects"]["mug"], *gt["objects"]["block"]]
    for block in results_of(agent):
        for v in values:
            assert f"{v:.3f}" not in text(block) and f"{v:.2f}" not in text(block).replace("cm", "")


def test_a_refused_action_skips_the_rest_of_the_turn(body):
    steps = [[("move_to", {"camera": "front", "x": 3, "y": 3}), ("lower", {}), ("look", {})],
             [("done", {"success": False, "summary": ""})]]
    agent, _, pointing = make_agent(body, steps)
    before = body.ground_truth()["tcp"]
    start(agent, pointing)
    first, second, third = results_of(agent)[:3]
    assert first.is_error and "off the table" in text(first)
    assert second.is_error and text(second).startswith("Skipped")
    assert not third.is_error          # observing is still allowed
    assert np.allclose(body.ground_truth()["tcp"], before, atol=1e-3)


def test_bad_arguments_are_rejected_without_executing(body):
    steps = [[("lower", {"cm": -3}), ("grip", {"action": "squeeze"}), ("turn", {"degrees": 10, "speed": 2})],
             [("done", {"success": False, "summary": ""})]]
    agent, _, pointing = make_agent(body, steps)
    start(agent, pointing)
    msgs = [text(b) for b in results_of(agent)[:3]]
    assert "greater than 0" in msgs[0] and "one of" in msgs[1] and "not a known argument" in msgs[2]


def test_closing_on_air_reports_empty(body):
    steps = [[("grip", {"action": "close"})], [("done", {"success": False, "summary": ""})]]
    agent, _, pointing = make_agent(body, steps)
    start(agent, pointing)
    assert "Closed on nothing" in text(results_of(agent)[0])


def test_hooks_can_deny_a_call(body):
    class NoTurning(Extension):
        name = "no_turning"

        def before_tool(self, call, tool):
            return Deny("turning is disabled here.") if call.name == "turn" else None

    agent, _, pointing = make_agent(body, [[("turn", {"degrees": 30})], [("done", {"success": False, "summary": ""})]],
                                    extra=[NoTurning()])
    start(agent, pointing)
    assert "Refused: turning is disabled" in text(results_of(agent)[0])


def test_budgets_end_the_episode(body):
    agent, _, pointing = make_agent(body, [[("look", {})]] * 10, config=LoopConfig(budget=Budget(max_steps=3)))
    assert start(agent, pointing).reason == "budget_steps"


# ---- code as policy ------------------------------------------------------------------------------

def test_code_mode_hides_motion_tools_from_the_model_but_not_from_programs(body):
    program = "r = robot.nudge('front', up_cm=3)\nprint('ok', r.ok, r.stop)"
    steps = [[("move_to", {"camera": "front", "x": 300, "y": 300})], [("run_code", {"code": program})],
             [("done", {"success": False, "summary": ""})]]
    agent, provider, pointing = make_agent(body, steps, policy="code")
    start(agent, pointing)
    exposed = {t["name"] for t in provider.requests[0].tools}
    assert exposed == {"look", "run_code", "done"}
    direct, prog = results_of(agent)[:2]
    assert "no tool called move_to" in text(direct)
    assert "ok True reached" in text(prog) and not prog.is_error


def test_program_errors_and_budgets_come_back_to_the_model(body):
    crash = "robot.look()\nraise ValueError('boom')"
    spin = "while True:\n    robot.look()"
    steps = [[("run_code", {"code": crash})], [("run_code", {"code": spin})],
             [("done", {"success": False, "summary": ""})]]
    agent, _, pointing = make_agent(body, steps)
    for ext in agent.extensions:
        if ext.name == "code":
            ext.tools[0].max_calls = 5
    start(agent, pointing)
    crashed, spun = results_of(agent)[:2]
    assert crashed.is_error and "ValueError: boom" in text(crashed)
    assert spun.is_error and "more than 5 robot calls" in text(spun)


def test_emergency_stop_interrupts_a_running_program(body):
    program = "for i in range(20):\n    robot.nudge('front', up_cm=1)\nprint('finished')"
    agent, _, pointing = make_agent(body, [[("run_code", {"code": program})],
                                           [("done", {"success": True, "summary": ""})]])
    seen = []

    def stop_after_two(event):
        if isinstance(event, ToolEnd) and event.parent_id:
            seen.append(event)
            if len(seen) == 2:
                agent.control.estop("test")

    agent.bus.subscribe(stop_after_two)
    outcome = start(agent, pointing)
    assert outcome.reason == "estop"
    assert len(seen) == 2
    assert "finished" not in text(results_of(agent)[0])


def test_scripted_demo_mixes_pointing_and_code_and_records_everything(tmp_path):
    report = scripted_demo(tmp_path / "ep")
    assert report.success and report.outcome.reason == "done"
    files = {p.name for p in (tmp_path / "ep").iterdir()}
    assert {"episode.jsonl", "result.json", "evaluation.json", "observations"} <= files
    assert len(list((tmp_path / "ep" / "observations").glob("*.png"))) >= 10


# ---- operator controls ---------------------------------------------------------------------------

def test_operator_messages_reach_the_model_before_its_next_decision(body):
    agent, provider, pointing = make_agent(body, [[("look", {})], [("done", {"success": False, "summary": ""})]])
    agent.control.say("Not the mug, the block.")
    start(agent, pointing)
    assert any(isinstance(m, OperatorMessage) for m in provider.requests[0].messages)


def test_pause_holds_the_loop_until_resume(body):
    """The loop stays on the thread that owns the renderer; the operator acts from another thread."""
    agent, provider, pointing = make_agent(body, [[("look", {})], [("done", {"success": False, "summary": ""})]])
    first_request = []
    agent.bus.subscribe(lambda e: first_request.append(time.time()) if e.type == "TurnStart" else None)
    agent.control.pause()
    resumed_at = time.time() + 0.5
    threading.Timer(0.5, agent.control.resume).start()
    assert start(agent, pointing).reason == "done"
    assert first_request[0] >= resumed_at - 0.05


def test_a_broken_subscriber_cannot_stop_the_robot(body):
    agent, _, pointing = make_agent(body, [[("look", {})], [("done", {"success": True, "summary": ""})]])
    agent.bus.subscribe(lambda e: 1 / 0)
    assert start(agent, pointing).reason == "done"


# ---- schema ------------------------------------------------------------------------------------

def test_schema_validation():
    s = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}, "s": {"type": "string"}},
         "required": ["n"], "additionalProperties": False}
    assert validate(s, {"n": 1}) is None
    assert "required" in validate(s, {})
    assert "integer" in validate(s, {"n": 1.5})
    assert "integer" in validate(s, {"n": True})
    assert "at least" in validate(s, {"n": -1})
    assert "not a known" in validate(s, {"n": 1, "x": 0})
