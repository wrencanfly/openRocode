# Trace format

Every episode openRocode runs is recorded as a **trace**: what the model was asked, every picture it saw, every tool call it made (including calls made from inside its programs), what the robot actually did, what a person said or pressed, and whether the task really succeeded. The dashboard, `rocode replay` and `rocode bench` all read this format, and so can your own tools.

This page describes **trace format version 1**. Read traces with `rocode.trace`:

```python
from rocode.trace import load

trace = load("runs/mug_to_coaster-20261117-235411")
trace.version            # 1
trace.config["model"]    # "gpt-6.1-sol"
trace.success            # True: the simulator's verdict, not the model's
for event in trace.events:
    if event["type"] == "ToolEnd" and event["receipt"]:
        print(event["name"], event["receipt"]["executed"], event["data"].get("height_cm"))
```

## Files

A trace is one directory, named after the episode (`<task>-<date>-<time>`, or `<task>-<batch>-<model>-s<seed>` when `rocode bench` ran it).

| File | Written | Contents |
|---|---|---|
| `episode.jsonl` | as events happen, one JSON object per line, flushed | the event stream below; never contains image bytes |
| `observations/<id>_<camera>.png` | when each picture is taken | every picture exactly as the model saw it, annotations included |
| `result.json` | when the episode ends | the `EpisodeEnd` reason and stats |
| `evaluation.json` | after the episode ends | the ground-truth verdict; never shown to the model |

A running episode has `episode.jsonl` and pictures but no `result.json` yet. Readers should ignore a last line that is not complete JSON: it is still being written.

## Versioning

`EpisodeStart.version` holds the format version. Traces recorded before it existed have no `version` field and are identical to version 1.

- **Additive changes keep the version**: a new event type, a new field, a new key in `config`, `data` or `stats`. Readers must ignore what they do not know.
- **Breaking changes bump it**: renaming or removing an event or field, or changing what a field means or its unit.
- `rocode.trace.load` refuses traces newer than the version it understands.

## Conventions

- `t` on every event is a Unix timestamp in seconds (float).
- Lengths the model sees are in **centimetres**, forces in **newtons**, angles in **degrees**. World coordinates appear only in `evaluation.json` (metres), which the model never sees.
- `call_id` identifies a tool call. Calls made from inside a `run_code` program have `parent_id` set to the program's `call_id`, so a trace nests the way the work did: turn, then program, then robot calls.
- The first picture is taken by a call with `call_id` `"initial_look"` before the first turn.

## Events

Every line of `episode.jsonl` is an object with a `type` naming one of the events below, a `t`, and that event's fields.

### EpisodeStart

The first event.

- `task`: the instruction given to the model, in plain language.
- `config`: how the episode was set up. Always `provider`, `model`, `effort`, `tools` (names the model may call directly) and `budget` (`max_steps`, `max_turns`, `max_seconds`). Sessions also add `task_id`, `body`, `policy` (`tools`, `code` or `both`), `seed` (null for the default layout) and `batch` (the `rocode bench` batch id, or null).
- `version`: the trace format version.

### TurnStart

The harness is about to ask the model for its next move.

- `turn`: 1-based turn number.

### AssistantReply

The model answered.

- `turn`: the turn this reply belongs to.
- `text`: what the model wrote besides tool calls (may be empty).
- `tool_calls`: list of `{id, name, args}`, in the order the model asked for them.
- `usage`: tokens for this reply: `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`.
- `stop_reason`: the provider's stop reason, e.g. `tool_use`, `end_turn`, `max_tokens`, `refusal`.
- `reasoning`: the model's thinking, or the provider's summary of it, when the provider returns one; empty otherwise. Recorded for people to read. Providers replay the original thinking blocks themselves; this text is never sent back to the model.

### ToolStart

A tool call begins, from the model or from a program.

- `call_id`, `name`, `args`: the call. `args` is exactly what was validated and run.
- `parent_id`: the `run_code` call this came from, or null for a call the model made directly.

### ToolEnd

A tool call finished. Every `ToolStart` has exactly one `ToolEnd`, even when the call was refused or failed.

- `call_id`, `name`, `parent_id`: as in `ToolStart`.
- `is_error`: true if the call failed or was refused. A physical action that stopped on contact is not an error; read `receipt`.
- `text`: the text result the model read.
- `observation_ids`: pictures attached to the result.
- `receipt`: for physical actions and programs, what actually happened, else null:
  - `executed`: `none`, `partial`, `full` or `unknown`. Never guessed.
  - `stop_reason`: `reached`, `contact`, `blocked`, `refused`, `cancelled` or `fault`.
  - `stop_confirmed`: whether the robot confirmed it stopped.
  - `peak_force_n`, `detail`, `extra`.
- `duration_s`: wall-clock seconds.
- `data`: structured readings the model also saw, flat and JSON-typed. Pointing tools report `observation`, `height_cm` (fingertips above the table), `opening_cm`, `holding`, `contact_n` and `stop`; `run_code` reports `observation`, `calls` and `exit_code`; `done` reports `success` and `summary`.

### ToolDenied

A hook refused a call before it ran. A `ToolEnd` with `is_error` true follows.

- `call_id`, `name`: the call.
- `reason`: why, as given by the hook.

### ObservationTaken

A picture set was taken and saved under `observations/`.

- `observation_id`: `o1`, `o2`, …, in order.
- `cameras`: camera names; the files are `<observation_id>_<camera>.png`.

### OperatorInput

A person sent the model a message (dashboard or terminal). The model reads it before its next turn.

- `text`: the message.

### Paused

A person paused the episode. The loop waits at a safe point; nothing moves.

### Resumed

A person resumed a paused episode.

### EStop

An emergency stop ended the episode.

- `source`: who stopped it, e.g. `dashboard`, `operator console`, `bench stopped`, or a hook's name and reason.

### Notice

Anything else worth recording, such as an action skipped because an earlier one in the same turn stopped early.

- `text`: the notice.

### EpisodeEnd

The last event. The same content is written to `result.json`.

- `reason`: `done`, `refusal`, `budget_steps`, `budget_turns`, `budget_time`, `estop`, `max_tokens` or `error`.
- `stats`: `steps` (top-level tool calls), `turns`, `seconds`, `usage` (summed over replies), `finish` (`{success, summary}` from the model's `done` call, or null) and `error` (text, or empty).

## evaluation.json

Written by the session after the episode, from the simulator's ground truth:

- `task`: the task id.
- `success`: whether the task's check passed. This is the verdict the dashboard and `rocode bench` report.
- `claimed`: what the model said in `done` (`true`, `false`, or null if it never called `done`).
- `ground_truth`: the simulator's final state, in metres in the world frame (for MuJoCo: `objects`, `held`, `tcp`).
- `config`: the full session config.

## Batches

`rocode bench` also writes `runs/batches/<batch id>.json`: the plan (`tasks`, `models`, `seeds`, `policy`, …) and one entry per planned episode with its `id`, `task`, `provider`, `model`, `seed` and `status` (`pending`, `running`, `succeeded`, `failed`, `error`, `stopped` or `ended`). An episode that could not start (for example, a missing API key) has `status` `error`, an `error` message, and no trace directory.
