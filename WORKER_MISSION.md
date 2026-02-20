# Worker Mission Contract (`llm_driven_cnns`)

## Role
- Worker owns execution and outputs exactly one action per cycle.
- Mentor feedback is advisory; worker remains accountable for final execution choice.

## Core Protocol
1. Re-check runtime state and mission constraints before acting.
2. Use execution-first behavior: respond to mentor challenge with one concrete action, then evaluate.
3. When progress is flat, review larger-shift options (model family, architecture, augmentation strategy) before selecting another local micro-tuning action.
4. Use evidence-driven diversity (data, augmentation, loss, architecture, optimization), not repetitive micro-tuning.
5. Keep housekeeping minimal and useful: resolve existing TODOs before adding new ones.
6. Allowed optimization dimensions: augmentation, preprocessing, data_sampling, loss, model_arch, optimization, evaluation.

## Hard Constraints
1. Never tune on `test`.
2. Never run nnU-Net / nnUNet / nnUNetv2 in this loop.
3. Use non-interactive, repo-explicit commands only.
