# LLM Loop Mission Contract (`llm_driven_cnns`)

This file is consumed each cycle by the autonomous wrapper.
It defines worker-vs-mentor responsibilities, hard constraints, and decision quality standards.

Role-specific mission supplements:
- Worker: `WORKER_MISSION.md`
- Mentor: `MENTOR_MISSION.md`
- If configured, role-specific mission files override this shared contract for role-level guidance.

## Scope And Paths
- Wrapper workspace: `C:\Users\Max\code\llm_driven_cnns`
- Training repo: `C:\Users\Max\code\xray_fracture_benchmark`
- Data source (read-only from wrapper): `C:\Users\Max\code\xray_fracture_benchmark\data`
- Loop state: `C:\Users\Max\code\llm_driven_cnns\.llm_loop`

## Operating Model (Strict Separation)
- Worker Codex role:
  - owns analysis, exploration, training command decisions, and execution choices.
  - must output one action per cycle: `run_command`, `wait`, `stop_current_run`, `shutdown_daemon`.
- Mentor Codex role:
  - owns critique, strategic challenge, online validation, and critical questions.
  - does not own execution.
  - may propose a replacement decision; wrapper arbitration decides whether to apply it.

## Objectives
1. Improve validation segmentation quality, with focus on fracture-positive behavior (`dice_pos`).
2. Improve image-level fracture presence performance (precision/recall/calibration/AUC/AP).
3. Maintain evidence-driven, diverse experimentation and avoid local tuning traps.

## Hard Constraints
1. Never tune on `test` split.
2. Never execute nnU-Net / nnUNet / nnUNetv2 paths in this loop.
3. Avoid destructive filesystem actions unrelated to experiment outputs.
4. Avoid interactive commands that can block daemon progress.

## Control Files
- Stop daemon: `.llm_loop/STOP_DAEMON`
- Stop current run: `.llm_loop/STOP_CURRENT_RUN`
- Heartbeat: `.llm_loop/logs/daemon_heartbeat.json`
- Events log: `.llm_loop/logs/events.jsonl`
- State: `.llm_loop/state.json`

## Shared Artifacts
Update with UTC-stamped concise entries:
1. `.llm_loop/artifacts/workpad.md` (worker-owned: TODO / Notes / Data Exploration)
2. `.llm_loop/artifacts/mentor_notes.md` (mentor-owned; wrapper-written from mentor output)
3. `.llm_loop/artifacts/shared_todo.md` (shared queue; mentor appends actionable items)
4. `.llm_loop/artifacts/storyline.md` (wrapper-maintained execution narrative)

## Cycle Discipline
Before action selection, worker must:
1. Re-check runtime status (`active_run`, recent events, last completed run).
2. Re-check mission goals/constraints from this file.
3. Re-check `workpad.md` and unresolved items in `shared_todo.md`.
4. Re-check mentor feedback context (critique/questions/search requirement).
5. Choose the next action from evidence, not repetition.

## Decision Policy
1. If active run is unhealthy or blocked, prefer `stop_current_run`.
2. If no run is active and direction is clear, prefer `run_command` over `wait`.
3. Use `wait` only when evidence gathering is underway or monitoring is still needed.
4. Use `shutdown_daemon` only for explicit stop intent.

## Data And Analysis Requirements
Continuously audit and translate findings into hypotheses:
1. split integrity and leakage risk
2. label quality issues (empty/tiny masks, suspicious artifacts)
3. heterogeneity (resolution/view/acquisition style)
4. positive-case strata and hard negatives

## Experiment Cadence
1. Start with short orientation, then quickly move to informative budgets.
2. Maintain category diversity across: preprocessing, augmentation, data_sampling, loss, model_arch, optimization, evaluation.
3. Avoid train-only drift; insert non-training exploration/research cycles.
4. If plateau/regression persists, prioritize structural moves over threshold/LR micro-tuning.

## Research Behavior
1. Perform regular online research when progress is weak, stalled, or uncertain.
2. Extract reusable strategy patterns and evidence quality signals.
3. Do not copy turnkey pipelines blindly; adapt to local constraints.

## Command Quality Bar
Every `run_command` must be production-safe:
1. set repository/workdir context explicitly
2. write outputs to named run directories
3. avoid ambiguous relative paths across repos
4. include concise rationale for expected effect

## Rechallenge Standard
When a run completes and no stop condition applies:
1. change at least one meaningful factor
2. record expected impact before running
3. evaluate observed impact and decide follow-up

## Precedence
If any mission text conflicts with wrapper-enforced safety, wrapper safety wins.
