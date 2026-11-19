# Robots and environments

openRocode drives every robot through one body contract (`rocode/body/base.py`), so the pointing tools, `run_code`, the safety gate, traces, `rocode bench` and the dashboard work the same whichever robot or simulator is underneath. There are two ways to get a different robot:

- **A robot at the openRocode tabletop.** Set `robot` in a task file. The scene, the objects and the success expression stay yours.
- **A benchmark environment.** Name one of its tasks, such as `robosuite:lift` or `metaworld:pick-place-v3`. The environment builds its own scene and decides success; openRocode supplies the tools and the record.

```bash
pip install -e '.[sim,robots]'                  # Panda, ALOHA and LEAP models (MuJoCo Menagerie)
pip install -e '.[sim,robosuite,metaworld]'     # benchmark environments (MuJoCo 3.2 or 3.3)

rocode preview examples/tasks/aloha_bimanual.toml
rocode run both_blocks_to_tray --task-file examples/tasks/aloha_bimanual.toml --view
rocode tasks --env robosuite --env metaworld
rocode run robosuite:stack --view
rocode bench metaworld:pick-place-v3 metaworld:drawer-open-v3 --repeat 5 --model ... --model ...
```

## Robots at the tabletop

| `robot` | What it is | Arms | Hand | Cameras |
|---|---|---|---|---|
| `gripper` (default) | A parallel gripper that floats above the table. No arm, so no reach or joint limits. | `hand` | gripper, 9 cm | front, side |
| `panda` | A Franka Emika Panda arm behind the table, facing the front camera. | `arm` | gripper, 8 cm | front, side |
| `aloha` | Two ViperX 300 arms (the ALOHA setup), one at each end of the table. Each reaches its own half and about 10 cm past the middle. | `left`, `right` | gripper, 5.5 cm | front, top |
| `leap` | A four-fingered LEAP hand, palm down, floating above the table. | `hand` | 16-joint hand | front, side |

```toml
robot = "aloha"

[[object]]
name = "red_block"
...
```

The arm and hand models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) through the `robot_descriptions` package, which downloads them on first use.

**Jointed arms** (Panda, ALOHA) follow the commanded fingertip point with damped least-squares inverse kinematics, the hand pointing straight down and turned by the commanded yaw. A point the arm cannot reach is refused before anything moves ("the left arm cannot reach that point"), the same way a point outside the workspace is. Arms are gravity-compensated and run with MuJoCo's implicit integrator.

**More than one arm.** Every motion tool and every `robot.*` call takes `arm`, the tool results describe each arm, and each arm's fingertip marker carries its initial. Arms can collide with each other; the model has to move one out of the way. `examples/tasks/aloha_bimanual.toml` includes a handover: a block that only the left arm can reach has to end up where only the right arm can reach.

**The LEAP hand** closes into one power grasp: `grip close` curls the fingers into a fist around the point under its marker. On a table the fingers finish closing as the hand lifts, so the hand reports holding something from touch on both sides of the grasp, and its grip result says to lift and check. The open hand is about 20 cm across, so tasks for it spread objects out (`examples/tasks/leap_hand.toml`).

A robot changes what is reachable, not what is asked: any task file can be tried with any robot by changing `robot`. `tests/test_robots.py` solves tasks with each robot using a scripted policy that reads ground truth, to show they are reachable.

## Benchmark environments

| Environment | Tasks | Robot | Needs |
|---|---|---|---|
| `robosuite` | `lift`, `stack`, `pick_place_can`, `pick_place_milk`, `nut_assembly_square` | Franka Panda | `robosuite>=1.5`, MuJoCo below 3.4 |
| `metaworld` | `reach-v3`, `push-v3`, `pick-place-v3`, `button-press-topdown-v3`, `drawer-open-v3`, `drawer-close-v3`, `door-open-v3`, `window-open-v3`, `faucet-open-v3`, `coffee-button-v3` | Sawyer (fixed wrist) | `metaworld>=3.0`, MuJoCo below 3.4 |

An environment task's success expression is `env_success()`: the environment's own check (robosuite's `_check_success()`, Meta-World's `evaluate_state`). It can be combined with other predicates in a task of your own, e.g. `env_success() and hand_empty()`.

How each one is driven:

- **robosuite.** The environment keeps its operational-space controller. Each control step (20 Hz) the commanded fingertip point and yaw become a delta action, so robosuite's physics and success check run as they would for any policy. Pictures come from `agentview` and a second camera of the environment, rendered with depth for the pointing tools.
- **Meta-World.** The Sawyer hand is a mocap-driven gripper that always points down and cannot turn, so the `turn` tool is not offered. openRocode moves the mocap target and the fingers directly instead of calling `env.step`, so an episode is not cut off at Meta-World's 500-step horizon. Meta-World's corner cameras render upside down; two of them are re-aimed as `front` (facing the robot across the table) and `side`.

**Seeds.** A seed always gives the same layout, so every model in a bench sees the same tables. Both simulators only repeat a layout when the environment is made with the seed, so `reset(seed)` rebuilds the environment (under a second).

**Not yet included.** robosuite's Door, Wipe and ToolHang need the wrist to pitch, and the pointing tools keep the hand pointing down. Two-arm robosuite tasks, LIBERO, RoboCasa, ManiSkill and the Isaac-based suites are not wrapped yet.

## Writing a body

A body implements `info`, `observe`, `move`, `rotate`, `grip`, `drive`, `stop`, `reset`, `ground_truth` and `close` (see `rocode/body/base.py`). For anything in MuJoCo, subclass `MujocoBody` (`rocode/body/mj.py`) and give it one `ArmDriver` per arm; the motion loops, force guard, grasp sensing, cameras and receipts are shared:

| You have | Driver |
|---|---|
| A hand welded to a mocap body | `MocapArm` |
| Position-controlled arm joints | `JointArm` (IK toward the fingertip pose) |
| A benchmark's action vector | a small driver that holds the target, as in `rocode/envs/robosuite_env.py` |

If the simulator advances several physics steps per action, set `substeps`; the loops then count in actions. `rocode/envs/` shows two complete wrappers.
