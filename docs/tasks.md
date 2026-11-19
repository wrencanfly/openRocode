# Tasks

A task is three things: what is on the table, an instruction for the model, and a success check the model never sees. Built-in tasks and your own are defined the same way, run the same way, and are recorded the same way, so anything that works for `mug_to_coaster` works for yours: the dashboard, `rocode bench`, the Compare page and the trace format.

```bash
rocode tasks --task-file my_tasks.toml                 # check the file and list its tasks
rocode preview my_tasks.toml                           # pictures of the scene from both cameras, no model call
rocode run sort_blocks --task-file my_tasks.toml --view
rocode bench sort_blocks stack_three --task-file my_tasks.toml --repeat 5 \
  --model anthropic:claude-opus-5-5 --model openai:gpt-6.1-sol
```

To try something without writing a file, type the instruction in quotes. It runs in the default scene, or in the scene of the one `--task-file` you pass, and is recorded but not graded:

```bash
rocode run "Push the blue block to the left edge of the table" --view
```

## A task file

Task files are TOML. One file holds one scene and any number of tasks that use it. [`examples/tasks/`](../examples/tasks/) has two complete files.

```toml
[[object]]
name = "tray"
shape = "box"
size_cm = [16, 12, 0.6]
color = "white"
at_cm = [-18, -8]
fixed = true

[[object]]
name = "red_block"
shape = "box"
size_cm = 4
color = "red"
at_cm = [-8, 12]

[[task]]
id = "red_to_tray"
instruction = "Put the red block on the white tray."
success = "on(red_block, tray)"
```

A file with no `[[object]]` uses the default scene: a red `mug`, a blue `block` and a yellow `coaster`.

Every key is checked when the file loads. A misspelt key, an object that does not fit on the table or a success check that names a missing object stops the run before any model is called, with the file and the problem in the message.

## Objects

| Key | Meaning |
|---|---|
| `name` | How the success check refers to it. Lowercase letters, digits and `_`. The model never sees names; it sees pictures. |
| `shape` | `box`, `cylinder`, `sphere` or `mug` (a cylinder with a handle on its +x side). |
| `size_cm` | A box: a number (a cube) or `[x, y, z]`. A cylinder or mug: `[diameter, height]`. A sphere: its diameter. |
| `color` | `red`, `blue`, `yellow`, `green`, `orange`, `purple`, `pink`, `brown`, `white`, `black`, `gray`, or `[r, g, b]` from 0 to 1. |
| `at_cm` | `[x, y]` of its centre. x points right as seen from the front camera, y away from it, and the origin is the table's centre. |
| `yaw_deg` | Rotation about the vertical. Default 0. |
| `fixed` | `true` for things that are part of the table: coasters, trays, pads, walls. The gripper cannot move them. Default `false`. |
| `mass_g` | Default: from the volume, at 0.6 g/cm³. |
| `jitter_cm` | `[x, y]`: how far a seed may move it from `at_cm`. Default `[4, 3]` for loose objects, none for fixed ones. |

The table top spans x from -45 to 45 cm and y from -35 to 35 cm; the gripper reaches x from -40 to 40 and y from -30 to 30. The fingers open to 9 cm, so anything wider than that cannot be gripped across. Loose objects can be at most 15 cm tall.

`rocode run --seed N` and `rocode bench --seeds` place each loose object within its jitter, redrawing until none overlap. The same seed always gives the same layout, so every model in a bench sees the same tables.

## Success checks

`success` is an expression over the state at the end of the episode. It reads the simulator, never the model's claims.

| Predicate | True when |
|---|---|
| `on(a, b)` | `a` rests on `b`: its centre is over `b`'s footprint, above `b`'s top and lower than `b`'s top plus `a`'s height, and the gripper is not holding it. `b` may be `table`. |
| `held(a)` | The gripper is holding `a`. |
| `hand_empty()` | The gripper is holding nothing. |
| `upright(a)`, `upright(a, deg)` | `a`'s up axis leans at most `deg` degrees (default 15) from vertical. |

| Measurement | Value |
|---|---|
| `height_cm(a)` | Height of `a`'s centre above the table top. |
| `distance_cm(a, b)` | Horizontal distance between the centres of `a` and `b`. |
| `x_cm(a)`, `y_cm(a)` | `a`'s centre, in the same axes as `at_cm`. |
| `tilt_deg(a)` | How far `a`'s up axis leans from vertical. |

Combine them with `and`, `or`, `not`, parentheses, numbers and `<`, `<=`, `>`, `>=`:

```toml
success = "on(mug, coaster) and upright(mug)"
success = "held(block) and height_cm(block) > 10"
success = "on(red_block, tray) and not on(green_block, tray)"
success = "x_cm(block) < -30 and on(block, table)"
success = "distance_cm(red_block, blue_block) < 6 or on(red_block, blue_block)"
```

Each part joined by a top-level `and` is judged and recorded on its own, with the values it read. A failed episode therefore says which condition failed and by how much, in `evaluation.json`, in the dashboard and on the Compare page:

```
✗ on(red_block, blue_block)
✓ on(green_block, table)
✗ height_cm(block) > 10   (height_cm(block) = 6.4)
```

Leave `success` out to make an ungraded task: it runs and is recorded, but has no verdict and cannot be benchmarked.

## Writing a good task

- Say the goal in the instruction, the way you would to a person, and state the exact condition in `success`. Instructions should not mention centimetre tolerances.
- Give look-alike objects names that tell them apart in the check (`red_block`, `red_mug`), but describe them in the instruction as a person would see them.
- Check the start: `rocode preview` shows the table, and a task whose check already passes at the start is a bug. `tests/test_tasks.py` checks this for every file in `examples/tasks/`.
- Check that it can be done. `rocode.demo.scripted_pick_and_place` solves pick-and-place goals by reading ground truth; `tests/test_tasks.py` uses it to show that every example task is reachable with this gripper.

## Python task files

For a check the expressions cannot state, write a `.py` file with a module-level `TASKS` list. `success` may be a function that receives `body.ground_truth()` (`objects`: centres in metres, `tilt_deg`, `held`, `tcp`) and returns a bool:

```python
from rocode.scene import Scene, make_object
from rocode.tasks import Task

SCENE = Scene((make_object("ball", "sphere", 4.5, "orange", [0, 10]),))

def ball_near_front_edge(gt):
    return gt["objects"]["ball"][1] < -0.28

TASKS = [Task("ball_to_front", "Roll the orange ball to the front edge of the table.", ball_near_front_edge, SCENE)]
```

The file runs as ordinary Python, so only load files you trust. A function check is recorded as `python:<name>` and gives a verdict but no per-part breakdown.

## In the trace

Every episode records its whole task in `EpisodeStart.config.task_spec`: the instruction, the success expression, where it came from and the full scene. A trace is enough to see exactly what was asked and how it was judged, even after the task file has changed. See [trace-format.md](trace-format.md).
