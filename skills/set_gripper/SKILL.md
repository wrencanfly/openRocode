# Skill: `set_gripper` (L1 Direct Primitive)

## Overview
Directly control gripper aperture (`0.0` closed to `1.0` open) with effort-bounded closure detection so the LLM knows whether an object was actually grasped or the fingers closed on empty air.

## Arguments
- `open_ratio` (`float`): Target aperture in `[0.0, 1.0]`.
- `max_effort_fraction` (`float`, optional): Effort limit in `[0.1, 1.0]` for delicate objects.
