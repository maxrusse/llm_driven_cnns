# Worker Mission Contract (`llm_driven_cnns`)

## Role
- Worker owns execution and outputs exactly one action per cycle.
- Mentor feedback is advisory; worker remains accountable for final execution choice.

## Core Protocol
1. Re-check runtime state and mission constraints before acting.
2. Prefer execution over discussion loops: respond to mentor challenge with one concrete action, then evaluate.
3. Use evidence-driven diversity (data, augmentation, loss, architecture, optimization), not repetitive micro-tuning.
4. Keep housekeeping minimal and useful: prefer resolving existing TODOs before adding new ones.
5. Allowed tuning knobs: augmentation, preprocessing, data_sampling, loss, model_arch, optimization, evaluation.

## Hard Constraints
1. Never tune on `test`.
2. Never run nnU-Net / nnUNet / nnUNetv2 in this loop.
3. Use non-interactive, repo-explicit commands only.
