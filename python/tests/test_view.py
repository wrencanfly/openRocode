"""Dashboard data and server. Uses a scripted episode; no API key or browser needed."""

import json
import urllib.error
import urllib.request

import pytest

from rocode.core.control import ControlChannel
from rocode.demo import scripted_demo
from rocode.view.episodes import list_episodes, load_episode
from rocode.view.server import ViewerServer


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    report = scripted_demo(root / "mug_to_coaster-demo")
    assert report.success
    return root


def test_episode_list_and_detail(runs):
    [summary] = list_episodes(runs)
    assert summary["task_id"] == "mug_to_coaster" and summary["status"] == "succeeded" and not summary["live"]

    ep = load_episode(runs, summary["id"])
    assert ep["setup"][0]["name"] == "look"                   # the first look is a recorded step
    assert all(o["readings"] for o in ep["observations"])     # every picture has its readings
    program = [c for t in ep["turns"] for c in t["calls"] if c["name"] == "run_code"][0]
    assert len(program["children"]) == 8 and all("result" in c for c in program["children"])
    assert ep["observations"][-1]["readings"]["holding"] is False
    assert ep["finish"]["success"] is True and ep["success"] is True


def test_unknown_or_unsafe_ids_are_rejected(runs):
    for bad in ["../etc", "..", "", "a/b", "nope"]:
        assert load_episode(runs, bad) is None


@pytest.fixture
def server(runs):
    control = ControlChannel()
    s = ViewerServer(runs, port=0, control=control, live_id="mug_to_coaster-demo").start()
    yield s, control
    s.shutdown()


def get(s, path):
    with urllib.request.urlopen(s.url + path.lstrip("/")) as r:
        return r.status, r.read(), r.headers.get("Content-Type")


def post(s, body, token=None):
    headers = {"Content-Type": "application/json", **({"X-Rocode-Token": token} if token else {})}
    req = urllib.request.Request(s.url + "api/control", data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_server_serves_page_data_and_images(server):
    s, _ = server
    status, page, ctype = get(s, "/")
    assert status == 200 and s.token.encode() in page and b"mug_to_coaster-demo" in page and "html" in ctype
    _, body, _ = get(s, "/api/episodes")
    assert json.loads(body)["episodes"][0]["id"] == "mug_to_coaster-demo"
    _, body, _ = get(s, "/api/episodes/mug_to_coaster-demo")
    ep = json.loads(body)
    assert ep["controllable"] is True
    status, png, ctype = get(s, "/" + ep["observations"][0]["images"]["front"])
    assert status == 200 and ctype == "image/png" and png[:4] == b"\x89PNG"
    for bad in ["/files/../episode.jsonl", "/files/mug_to_coaster-demo/episode.jsonl",
                "/files/mug_to_coaster-demo/observations/../result.json", "/static/../server.py"]:
        with pytest.raises(urllib.error.HTTPError) as err:
            get(s, bad)
        assert err.value.code == 404


def test_controls_need_the_token_and_the_live_episode(server):
    s, control = server
    assert post(s, {"episode": "mug_to_coaster-demo", "action": "say", "text": "hi"})[0] == 403
    assert post(s, {"episode": "other", "action": "pause"}, s.token)[0] == 409
    assert post(s, {"episode": "mug_to_coaster-demo", "action": "fly"}, s.token)[0] == 400

    assert post(s, {"episode": "mug_to_coaster-demo", "action": "say", "text": "Left one, please."}, s.token)[0] == 200
    assert control.drain_said() == ["Left one, please."]
    status, body = post(s, {"episode": "mug_to_coaster-demo", "action": "pause"}, s.token)
    assert status == 200 and body["paused"] and control.paused
    post(s, {"episode": "mug_to_coaster-demo", "action": "resume"}, s.token)
    assert not control.paused
    post(s, {"episode": "mug_to_coaster-demo", "action": "stop"}, s.token)
    assert control.cancel.cancelled and control.estop_source == "dashboard"


def test_without_a_live_run_nothing_is_controllable(runs):
    s = ViewerServer(runs, port=0).start()
    try:
        _, body, _ = get(s, "/api/episodes/mug_to_coaster-demo")
        assert json.loads(body)["controllable"] is False
        assert post(s, {"episode": "mug_to_coaster-demo", "action": "stop"}, s.token)[0] == 409
    finally:
        s.shutdown()
