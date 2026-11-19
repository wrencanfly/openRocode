"""The trace format: versioned, readable with rocode.trace, and fully documented."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from rocode.core import events as ev
from rocode.demo import scripted_demo
from rocode.trace import TRACE_VERSION, load

DOC = Path(__file__).resolve().parents[1] / "docs" / "trace-format.md"


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs") / "mug_to_coaster-demo"
    assert scripted_demo(d, seed=1).success
    return d


def test_trace_reads_back(trace_dir):
    trace = load(trace_dir)
    assert trace.version == TRACE_VERSION and not trace.running and trace.success is True
    assert trace.config["seed"] == 1 and trace.config["batch"] is None and trace.config["task_id"] == "mug_to_coaster"
    first = next(e for e in trace.events if e["type"] == "ObservationTaken")
    assert trace.image(first["observation_id"], first["cameras"][0]).is_file()
    nested = [e for e in trace.events if e["type"] == "ToolEnd" and e["parent_id"]]
    assert nested and all(e["receipt"]["executed"] in ("none", "partial", "full", "unknown")
                          for e in nested if e["receipt"])


def test_newer_traces_are_refused(trace_dir, tmp_path):
    lines = (trace_dir / "episode.jsonl").read_text().splitlines()
    start = json.loads(lines[0])
    start["version"] = TRACE_VERSION + 1
    (tmp_path / "episode.jsonl").write_text("\n".join([json.dumps(start), *lines[1:]]))
    with pytest.raises(ValueError, match="Upgrade"):
        load(tmp_path)


def test_unversioned_traces_read_as_version_1(trace_dir, tmp_path):
    lines = (trace_dir / "episode.jsonl").read_text().splitlines()
    start = json.loads(lines[0])
    del start["version"]
    (tmp_path / "episode.jsonl").write_text("\n".join([json.dumps(start), *lines[1:]]))
    assert load(tmp_path).version == 1


def test_every_event_and_field_is_documented():
    doc = DOC.read_text()
    sections = dict(re.findall(r"^### (\w+)\n(.*?)(?=^##)", doc + "\n##", flags=re.S | re.M))
    kinds = [c for c in vars(ev).values()
             if isinstance(c, type) and issubclass(c, ev.Event) and c is not ev.Event and dataclasses.is_dataclass(c)]
    assert kinds
    for kind in kinds:
        assert kind.__name__ in sections, f"docs/trace-format.md has no section for {kind.__name__}"
        for f in dataclasses.fields(kind):
            if f.name != "t":
                assert f"`{f.name}`" in sections[kind.__name__], f"{kind.__name__}.{f.name} is not documented"
