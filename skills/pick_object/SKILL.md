# Skill: `pick_object`

## When to Invoke
Call `pick_object` when the target rigid or deformable object is visible in the wrist or head RGB camera and the robot base is already parked within `0.85 m` of the object.

## Arguments
- `target_label` (`string`): Semantic description of the object to grasp.
- `image_point_1000` (`[int, int]`): Normalized `[x, y]` click in `[0, 1000]` on the head RGB frame.

## Failure Codes & Recovery Guidance
- `precondition_refused`: Gripper is already holding an object or base is out of range. Place held object or navigate closer first.
- `grasp_slip`: Contact lost during lift after bounded local retries; inspect new object pose in failure trace before retrying.
- `safety_envelope_violation` / `heartbeat_timeout`: S0 engaged `SafeHold`; verify obstacle clearance before re-issuing.
