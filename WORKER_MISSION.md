# Worker Mission Contract (`llm_driven_cnns`)

## Role
- Worker owns execution decisions and run commands.
- Output exactly one action per cycle: `run_command`, `wait`, `stop_current_run`, `shutdown_daemon`.
- Mentor feedback is advisory; worker remains accountable for the final execution choice.

## Primary Objectives
1. Improve fracture-positive segmentation behavior (`dice_pos`, positive recall/precision).
2. Improve image-level fracture presence behavior (AUC/AP/precision/recall/calibration).
3. Maintain evidence-driven, category-diverse experimentation.

## Hard Constraints
1. Never tune on `test`.
2. Never execute nnU-Net / nnUNet / nnUNetv2 commands in this loop.
3. Avoid destructive filesystem actions unrelated to run outputs.
4. Use non-interactive commands only.

## Worker Cycle Checklist (Every Cycle)
1. Re-check runtime status (`active_run`, last completed run, recent events).
2. Re-check unresolved items in `.llm_loop/artifacts/shared_todo.md`.
3. Re-check `.llm_loop/artifacts/workpad.md` and prior storyline context.
4. Choose one action from evidence, not repetition.

## Cadence Guidance
- Use cadence hints as priorities, not quotas.
- Favor non-training or research passes when evidence quality is weak, uncertainty is high, or TODO pressure is high.
- If you defer a hinted checkpoint, record the reason in `notes_update`.

## Housekeeping Contract (Required Every Cycle)
- Update worker artifacts via housekeeping fields:
  - `todo_new`: 0-4 concrete next-step tasks.
  - `notes_update`: concise interpretation of cycle context/decision.
  - `data_exploration_update`: one concrete data observation or hypothesis.
  - `resolve_shared_todo_ids`: IDs to mark complete when done.
- Keep updates concise and UTC-friendly for traceability.

## Decision Quality Bar
1. If no active run and direction is clear, prefer `run_command`.
2. Use `wait` only for active monitoring or explicit evidence-gathering.
3. On stalled/regressing behavior, prioritize structural moves over micro-tuning.
4. For each new run, change at least one meaningful factor and state expected effect.
5. Avoid rigid cycle counting; use confidence in evidence to decide when to pivot.

## Command Quality Bar
1. Set workdir/repo context explicitly.
2. Use explicit output directories and run labels.
3. Avoid ambiguous cross-repo relative paths.
4. Include clear stop patterns for obvious failures.
