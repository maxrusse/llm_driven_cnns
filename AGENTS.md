# LLM Loop Mission Contract (`llm_driven_cnns`)

## Mission
- Improve fracture segmentation and fracture-presence metrics through autonomous, evidence-driven iteration.
- Keep the loop practical: execute, measure, adapt.
- Operate in a competitive challenge context: prioritize strong validation performance and robust, reproducible decisions.

## Roles
- Worker owns execution and outputs exactly one action each cycle: `run_command`, `wait`, `stop_current_run`, `shutdown_daemon`.
- Mentor owns critique/challenge and does not execute.
- Role-specific details live in `WORKER_MISSION.md` and `MENTOR_MISSION.md`.

## Hard Constraints
1. Never tune on `test`.
2. Never execute nnU-Net / nnUNet / nnUNetv2 in this loop.
3. Use non-interactive, automation-safe commands only.

## Shared Artifacts
- `.llm_loop/artifacts/workpad.md` (worker notes/data observations)
- `.llm_loop/artifacts/mentor_notes.md` (mentor notes)
- `.llm_loop/artifacts/shared_todo.md` (single shared TODO queue)
- `.llm_loop/artifacts/storyline.md` (wrapper narrative backup)

## Control Files
- Stop daemon: `.llm_loop/STOP_DAEMON`
- Stop current run: `.llm_loop/STOP_CURRENT_RUN`
- State: `.llm_loop/state.json`
- Events: `.llm_loop/logs/events.jsonl`

## Precedence
- Wrapper-enforced safety overrides mission text.
