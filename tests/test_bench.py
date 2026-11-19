"""Batch runs and the comparison built from them. Scripted models stand in for real ones."""

import json
import urllib.error
import urllib.request

import pytest

from rocode.bench import Bench, BenchPlan, ModelSpec
from rocode.demo import scripted_mug_to_coaster
from rocode.providers.scripted import ScriptedProvider
from rocode.view.compare import compare, load_manifests
from rocode.view.episodes import list_episodes
from rocode.view.server import ViewerServer

SOLVER, QUITTER, BROKEN = ModelSpec("anthropic", "solver"), ModelSpec("anthropic", "quitter"), ModelSpec("openai", "broken")


def scripted_models(cfg, body):
    if cfg.model == "broken":
        raise RuntimeError("no API key for this provider")
    if cfg.model == "solver":
        provider = scripted_mug_to_coaster(body)
    else:  # claims success without doing anything
        provider = ScriptedProvider([[("done", {"success": True, "summary": "Done, I think."})]])
    provider.model = cfg.model
    return provider


def test_model_specs():
    assert ModelSpec.parse("openai:gpt-6.1-sol") == ModelSpec("openai", "gpt-6.1-sol")
    assert ModelSpec.parse("claude-opus-5-5") == ModelSpec("anthropic", "claude-opus-5-5")
    assert ModelSpec("openai", "org/model v2").slug == "org-model-v2"
    for bad in ["gemini:pro", "openai:", ""]:
        with pytest.raises(ValueError):
            ModelSpec.parse(bad)
    with pytest.raises(ValueError):
        BenchPlan(tasks=["fly_to_moon"], models=[SOLVER])


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    plan = BenchPlan(tasks=["mug_to_coaster"], models=[SOLVER, QUITTER, BROKEN], seeds=[0, 1])
    seen = []
    bench = Bench(plan, root, batch_id="bench-test", make_provider_fn=scripted_models,
                  on_done=lambda i, n, e: seen.append((i, n, e["status"])))
    manifest = bench.run()
    return root, manifest, seen


def test_batch_runs_every_cell_and_survives_broken_models(batch):
    root, manifest, seen = batch
    assert manifest["status"] == "finished" and [s[:2] for s in seen] == [(i, 6) for i in range(1, 7)]
    status = {(e["model"], e["seed"]): e["status"] for e in manifest["episodes"]}
    assert status == {("solver", 0): "succeeded", ("quitter", 0): "failed", ("broken", 0): "error",
                      ("solver", 1): "succeeded", ("quitter", 1): "failed", ("broken", 1): "error"}
    assert json.loads((root / "batches" / "bench-test.json").read_text()) == manifest
    recorded = list_episodes(root)
    assert len(recorded) == 4 and {e["batch"] for e in recorded} == {"bench-test"}
    assert {e["seed"] for e in recorded} == {0, 1}
    assert not any("broken" in p.name for p in root.iterdir())   # config errors leave no trace


def test_comparison_matrix(batch):
    root, _, _ = batch
    d = compare(list_episodes(root), load_manifests(root), "bench-test")
    assert d["tasks"] == ["mug_to_coaster"] and [c["key"] for c in d["columns"]] == ["solver", "quitter", "broken"]
    row = d["cells"]["mug_to_coaster"]
    assert (row["solver"]["succeeded"], row["solver"]["scored"], row["solver"]["rate"]) == (2, 2, 1.0)
    assert row["quitter"]["rate"] == 0.0
    # Episodes that never started are errors beside the rate, not failures of the model.
    assert (row["broken"]["scored"], row["broken"]["errors"], row["broken"]["rate"]) == (0, 2, None)
    assert all(e["why"] == "Called done, but on(mug, coaster) was false" for e in row["quitter"]["episodes"])
    assert [e["has_trace"] for e in row["broken"]["episodes"]] == [False, False]
    assert all("no API key" in e["why"] for e in row["broken"]["episodes"])
    assert d["batches"][0]["planned"] == 6 and d["batches"][0]["done"] == 6
    everything = compare(list_episodes(root), load_manifests(root))
    assert everything["episodes"] == 4    # without a batch: recorded episodes only


def test_stopping_a_batch_leaves_the_rest_pending(tmp_path):
    plan = BenchPlan(tasks=["mug_to_coaster"], models=[QUITTER], seeds=[0, 1, 2])
    bench = Bench(plan, tmp_path, batch_id="bench-stop", make_provider_fn=scripted_models,
                  on_done=lambda i, n, e: bench.stop())
    manifest = bench.run()
    assert manifest["status"] == "stopped"
    assert [e["status"] for e in manifest["episodes"]] == ["failed", "pending", "pending"]
    cell = compare(list_episodes(tmp_path), load_manifests(tmp_path), "bench-stop")["cells"]["mug_to_coaster"]["quitter"]
    assert (cell["scored"], cell["pending"]) == (1, 2)


def test_bench_closes_each_body_on_its_own_thread(tmp_path):
    # A renderer left to the garbage collector can be freed on a dashboard thread, which deadlocks on macOS.
    from rocode.body.mujoco_tabletop import MujocoTabletop

    bodies = []

    def make_body(cfg):
        bodies.append(MujocoTabletop(seed=cfg.seed))
        return bodies[-1]

    plan = BenchPlan(tasks=["mug_to_coaster"], models=[QUITTER, BROKEN], seeds=[0])
    Bench(plan, tmp_path, batch_id="bench-close", make_body=make_body, make_provider_fn=scripted_models).run()
    assert len(bodies) == 2 and all(b._rgb is None and b._depth is None for b in bodies)


def test_compare_api(batch):
    root, _, _ = batch
    s = ViewerServer(root, port=0).start()
    try:
        with urllib.request.urlopen(s.url + "api/compare?batch=bench-test") as r:
            assert json.loads(r.read())["cells"]["mug_to_coaster"]["solver"]["rate"] == 1.0
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(s.url + "api/compare?batch=../etc")
        assert err.value.code == 400
    finally:
        s.shutdown()
