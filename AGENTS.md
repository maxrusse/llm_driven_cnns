# LLM Driven CNNs Contract

## Purpose
This repository is a clean control plane for LLM-driven CNN experimentation.
The LLM is always the decision-maker for run/start/stop actions.
Primary target is the X-ray fracture challenge training loop.

## Scope
- Workspace root: `C:\Users\Max\code\llm_driven_cnns`
- Data source (read-only): `C:\Users\Max\code\xray_fracture_benchmark\data`
- Runtime state: `C:\Users\Max\code\llm_driven_cnns\.llm_loop`

## Driver Seat Rules
1. No automatic config auto-selection.
2. No hidden leaderboard logic.
3. LLM decides whether to run, wait, stop current run, or stop daemon.
4. The wrapper only enforces minimal safety:
   - stop flags
   - process timeout windows
   - basic liveness heartbeat
5. Never tune on test split.

## Control Files
- Stop daemon: `.llm_loop/STOP_DAEMON`
- Stop current run immediately: `.llm_loop/STOP_CURRENT_RUN`
- Heartbeat: `.llm_loop/logs/daemon_heartbeat.json`
- Cycle events: `.llm_loop/logs/events.jsonl`
- Runtime state: `.llm_loop/state.json`

## Expected Workflow
1. `scripts/install_tools.ps1`
2. `scripts/startup.ps1`
3. Monitor with `scripts/status.ps1`
4. Stop with `scripts/stop_llm_daemon.ps1`
5. Reset to clean state with `scripts/clean_fresh.ps1`

## Rechallenge Protocol
1. When a run finishes, do not idle by default.
2. Rechallenge the prior run with one deliberate change (for example lr schedule, augmentation, loss weighting, sampling, or architecture block).
3. Record what changed and why in loop events/todo.
4. Keep improving until explicit stop criteria are met.

## Scientific Working Style (Soft Guidance)
1. Use hypothesis-driven iteration:
   - State expected direction before each run (what should improve and why).
   - After each run, compare expected vs observed behavior.
2. Always record uncertainty:
   - Note confidence level and what evidence is still missing.
   - Flag when a decision is based on noisy or limited validation signal.
3. Run periodic explore vs exploit self-checks:
   - If recent cycles are too similar, deliberately explore a different idea class (preprocessing, augmentation, sampling, architecture, or loss design).
   - If a direction is consistently improving, exploit with controlled follow-ups.
4. After regressions, write a short failure-analysis note:
   - Most likely cause.
   - What was ruled out.
   - Next best corrective experiment.

## Initial Execution Agenda
1. Start with a fast baseline run on the challenge stack to validate pipeline health.
2. Perform an explicit model-selection bracket before deep tuning:
   - Compare at least 3 architecture/backbone candidates under equal fast budget.
   - Select winner based on validation evidence.
   - Record winner in `.llm_loop/artifacts/MODEL_SELECTION_DONE.md`.
3. Then iterate with targeted rechallenges focused on metric improvement.
4. Keep each rechallenge change deliberate and traceable.

## Subagent Note
The planner prompt explicitly allows Codex to delegate to subagents when that improves execution quality.
