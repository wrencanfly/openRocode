# Skill: `trace_path` (L2 On-the-fly Synthesized Trajectory)

## Overview
Allows the LLM to synthesize continuous spatial trajectories directly without a learned VLA—ideal for wiping spills, stirring, pouring arcs, or obstacle-clearing paths.

## Arguments
- `shape` (`"circle" | "waypoints"`): Trajectory parametrization.
- For `"circle"`:
  - `center_xyz_m` (`[float, float, float]`): Circle center in workspace meters.
  - `radius_m` (`float`): Radius in meters (e.g. `0.05`).
  - `cycles` (`int`, default `2`): Number of revolutions.
  - `duration_sec` (`float`, default `2.0`): Total execution duration.
- For `"waypoints"`:
  - `points_xyz_m` (`[[float, float, float], ...]`): Ordered 3D waypoints.
  - `duration_sec` (`float`, default `2.0`): Total execution duration.
