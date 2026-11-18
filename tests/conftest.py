import numpy as np
import pytest

from rocode.body.mujoco_tabletop import MujocoTabletop
from rocode.core.loop import Agent, LoopConfig
from rocode.ext.code import CodeAsPolicy
from rocode.ext.pointing import Pointing
from rocode.ext.safety import SafetyGate
from rocode.providers.scripted import ScriptedProvider


@pytest.fixture
def body():
    return MujocoTabletop()


def make_agent(body, steps, policy="both", extra=(), config=None, control=None):
    cams = body.info().cameras
    pointing = Pointing(cams, expose_motion=policy != "code")
    exts = [SafetyGate(), pointing, *([CodeAsPolicy(cams, max_seconds=30)] if policy != "tools" else []), *extra]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, body, exts, config or LoopConfig(), control)
    return agent, provider, pointing


def start(agent, pointing, task="test task"):
    return agent.run(task, lambda: pointing.first_look(agent.tool_context()).content)


def pixel(request_or_store, body_or_point, point=None, camera="front"):
    """Project a ground-truth point into the latest pictures (stands in for the model looking)."""
    store = getattr(request_or_store, "images", request_or_store)
    stored = store.latest()
    x, y = stored.observation.cameras[camera].project(np.asarray(point if point is not None else body_or_point,
                                                                 dtype=float))
    return round(x), round(y)
