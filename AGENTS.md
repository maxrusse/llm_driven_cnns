# LLM Driven CNNs Contract

## Purpose
This repository is a clean control plane for LLM-driven CNN experimentation.
The LLM is always the decision-maker for run/start/stop actions.
Primary target is the X-ray fracture challenge training loop.

## Scope
- Workspace root: `C:\Users\Max\code\llm_driven_cnns`
- Data source (read-only): `C:\Users\Max\code\xray_fracture_benchmark\data`
- Runtime state: `C:\Users\Max\code\llm_driven_cnns\.llm_loop`
- Dataset domain: X-ray fracture segmentation (medical imaging).

## Driver Seat Rules
1. No automatic config auto-selection.
2. No hidden leaderboard logic.
3. LLM decides whether to run, wait, stop current run, or stop daemon.
4. The wrapper only enforces minimal safety:
   - stop flags
   - process timeout windows
   - basic liveness heartbeat
5. Never tune on test split.
6. Do not execute nnU-Net/nnUNet/nnUNetv2 pipelines in this repo (reserved for later manual comparison against this loop).

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

## Phase Guidance
1. Use fast-dev only for early orientation (about 1-2 cycles) to verify pipeline and understand data behavior.
2. After orientation, move to stronger experiments (larger budget, stronger models/heads, broader eval).
3. Include recurring online research passes and adapt generic strong patterns to the task.
4. Keep a dual objective:
   - improve segmentation quality
   - push fracture-presence classification performance toward domain-competitive / SOTA-level ranges from literature.
5. Avoid train-only drift:
   - after several consecutive training runs, run a non-training exploration cycle (data/error analysis + online scan) before more training.

## Data Exploration Depth
1. Treat the quick audit script as first-pass only, not full dataset understanding.
2. Build and maintain a richer exploration note in `.llm_loop/artifacts/data_exploration.md`.
3. Expand exploration over time:
   - split integrity and leakage risk checks
   - label quality issues (empty masks, tiny masks, suspicious artifacts)
   - resolution/aspect/view heterogeneity and preprocessing implications
   - positive-case strata (fracture size, sparse-mask regimes, hard negatives)
4. Convert findings into concrete experiment hypotheses, not only hyperparameter tweaks.
5. Revisit this exploration periodically, not just once at startup.

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
5. Breakout rule for plateaus (soft but strong):
   - If recent cycles are flat/regressing and still far below target, prioritize structural changes over threshold/LR micro-tuning.
   - Structural changes include larger supported backbones, head/decoder adjustments, and training-budget increases (epochs, train/eval batches).
6. Benchmark awareness:
   - Use online references to estimate realistic high-performance ranges for this domain.
   - Track whether current classification metrics (precision/recall/calibration) are approaching those ranges.

## Initial Execution Agenda
1. Start with a fast baseline run on the challenge stack to validate pipeline health.
2. Inspect the data early (class balance, mask quality, obvious edge cases) and write short notes/hypotheses.
   - Continue deeper data exploration beyond the first audit and keep updating `.llm_loop/artifacts/data_exploration.md`.
3. Do a quick online scan of strong approaches for similar medical segmentation tasks, then adapt ideas pragmatically.
   - Generic patterns are good; avoid copy-pasting any single turnkey framework recipe.
4. Run a mixed set of targeted experiments (preprocessing, augmentation, sampling, loss, and architecture as needed), not only optimizer micro-tuning.
5. Keep each rechallenge change deliberate and traceable.

## Subagent Note
The planner prompt explicitly allows Codex to delegate to subagents when that improves execution quality.
