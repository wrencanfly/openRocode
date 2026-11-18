# Skill: `move_ee` (L1 Direct Spatial Primitive)

## Overview
Built-in zero-VLA spatial primitive allowing the LLM to directly command end-effector motion in task space (`torso` or `world` frame) via minimum-jerk spline interpolation and S0 contact-force guarding.

## Arguments
- `frame` (`"torso" | "world"`): Reference coordinate frame.
- `target_xyz_m` (`[float, float, float]`): Target `[x, y, z]` in meters (or relative delta when `relative = true`).
- `target_rpy_rad` (`[float, float, float]`, optional): Target orientation `[roll, pitch, yaw]` in radians.
- `stop_on_contact_n` (`float`, optional): Stop early and report contact pose if sensed force exceeds threshold (useful for guarded probing / surface touch).
