# Contributing to openRocode

Thanks for helping. openRocode is young, so a focused pull request can change a lot. This guide covers setup, the rules the harness depends on, and the three most common kinds of contribution.

## Set up

```bash
git clone https://github.com/wrencanfly/openRocode.git
cd openRocode
python -m venv .venv && source .venv/bin/activate
pip install -e '.[sim,agent,openai,dev]'
pytest            # 79 tests, about 12 seconds, no API key
rocode demo       # a scripted episode end to end
```

Python 3.11 or newer. MuJoCo renders off-screen on macOS and Linux. Run the agent loop on the main thread: MuJoCo's renderer belongs to the thread that created it (`rocode/body/base.py` explains this).

## Rules the harness depends on

Please keep these true in every change. Reviews check them first.

1. **The model never sees or writes world coordinates.** Tool results describe pictures, centimetres and directions in a camera view. `tests/test_harness.py` checks results for coordinates.
2. **History is append-only.** Never edit or delete a message after it is sent. Claude's thinking blocks and prompt caches depend on it. Shorten context through the provider (see `rocode/providers/`), not by rewriting messages.
3. **Physical actions are never retried automatically.** Report what actually happened (`Receipt.executed`, `stop_reason`) and let the model decide.
4. **Safety does not depend on the model.** Every motion goes through a `Guard` and the body's workspace check. Programs from `run_code` go through the same hooks as direct calls.
5. **The core stays small.** `rocode/core/` knows no robots and no vendors. New capabilities belong in an extension (`rocode/ext/`), a provider, or a body.

Changes to events are changes to the [trace format](docs/trace-format.md), which other tools read. Adding an event or a field is fine: document it there (a test checks). Renaming, removing or changing the meaning of one needs `TRACE_VERSION` bumped in `rocode/core/events.py`.

## Common contributions

### Add a robot or simulator (a body)

Implement the `Body` protocol in `rocode/body/base.py`: `info`, `observe`, `move`, `rotate`, `grip`, `stop`, `reset`, `ground_truth` and `close`. Use `rocode/body/mujoco_tabletop.py` as the reference.

- `observe` returns RGB images, depth when you have it, and a `CameraModel` per camera. The pointing tools need depth and calibration to turn a click into a 3D target.
- `move` must stop on contact above `guard.max_force_n` (measured relative to the start of the motion), check `cancel` every control period, and return an honest `Receipt`.
- `ground_truth` is for scoring only. Nothing the model sees may call it.
- `close` frees renderers and connections. Whoever creates a body closes it, on the creating thread: a MuJoCo renderer left to the garbage collector can be freed on a dashboard thread, which deadlocks on macOS.

Add a test that runs the scripted demo, or an equivalent script, against your body.

### Add a task

Most tasks need no code: write a TOML task file with the objects, the instruction and a success expression, as described in [docs/tasks.md](docs/tasks.md). To add one to `examples/tasks/`, also add its solution to `SOLUTIONS` in `tests/test_tasks.py`, so the tests show it is reachable with this gripper; the tests already check that it does not start solved. Keep the instruction in plain language, and state the success condition precisely in the check, not in the instruction.

A new predicate goes in `rocode/checks.py`, with its signature in `SIGNATURES`, a line in the module docstring and in docs/tasks.md, and a test. Built-in tasks live in `BUILTIN` in `rocode/tasks.py`; changing `DEFAULT_SCENE` changes them all, and every old seed's layout.

### Add a model provider

Implement `Provider` from `rocode/providers/base.py`: one `complete(request) -> Reply` call. Render canonical messages to the vendor's format byte-identically on every request, replay assistant turns verbatim, and apply `ContextPolicy` in whatever way keeps the vendor's history valid. `tests/test_providers.py` shows the checks to copy. Register it in `make_provider` in `rocode/session.py`.

## Good places to start

Issues labelled [`good first issue`](https://github.com/wrencanfly/openRocode/labels/good%20first%20issue) and [`help wanted`](https://github.com/wrencanfly/openRocode/labels/help%20wanted) are ready to pick up. Bigger items on the roadmap:

- a LIBERO body, the second body behind the same contract
- an OpenTelemetry exporter for traces, so LLM observability tools (Langfuse, Phoenix) can show them
- an MCP server, so Claude Code, pi and other agents can drive a robot
- a simulated arm with real kinematics
- more tasks, and a Gemini provider

If you plan something large, open an issue first so we can agree on the shape before you write it.

## Pull requests

- One change per pull request. Explain the problem, the new behavior, and how you checked it.
- Run `pytest` and make sure it passes. Add tests for new behavior; none of them may need an API key or a network connection.
- If the change affects what the model sees (tool descriptions, results, prompts), say so in the pull request, and include a real run if you can: `rocode run <task> --view` records one under `runs/`, and a screenshot or a short excerpt of `rocode replay` is enough.
- Keep the dashboard dependency-free: plain HTML, CSS and JavaScript, following [DESIGN.md](DESIGN.md) in light and dark.
- Never commit API keys, `.env` files, or recordings under `runs/`.

## Reporting bugs

Open an issue with what you ran, what you expected, and what happened. Attach the episode's `episode.jsonl` and `result.json` if a run went wrong. They contain no API keys, but check them before sharing.

## License

By contributing you agree that your contributions are licensed under the [Apache-2.0 License](LICENSE).
