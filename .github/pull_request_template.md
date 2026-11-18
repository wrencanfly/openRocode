## What and why

<!-- The problem, and the behavior after this change. -->

## How I checked it

- [ ] `pytest` passes (no API key or network needed)
- [ ] If this changes what the model sees (tools, results, prompts): a real run is described or linked below

## Rules kept

- [ ] The model still never sees or writes world coordinates
- [ ] History stays append-only; physical actions are not retried automatically
