# Reinforcement Learning Updates (October 2025)

This note summarises the reinforcement-learning changes that landed with the latest EasyDeL refresh. For a permanent reference, see `RL_ALGORITHMS.md`.

## Highlights
- `GRPOTrainer` now subsumes PPO-style variants from ProRL, DAPO, and VinePPO via configuration knobs (asymmetric clipping, selectable advantage estimators, length shaping, KL/entropy guards).
- Preference-based trainers (`DPOTrainer`, `ORPOTrainer`) remain unchanged except for richer metrics (KL, entropy, logit margin in DPO).
- New utility helpers (`compute_group_advantages`, `compute_length_reward`, `update_ema`) power these features and are unit-tested in `tests/trainers/grpo_utils_test.py`.
- Documentation (`docs/trainers/grpo.md`) outlines the new config switches; `RL_ALGORITHMS.md` documents the overall RL strategy.

## File Touch Points
- `easydel/trainers/group_relative_policy_optimization/` — core trainer, config, and math updates.
- `easydel/trainers/direct_preference_optimization_trainer/_fn.py` — metric additions.
- `easydel/trainers/training_utils.py` — shared helpers.
- `tests/trainers/grpo_utils_test.py` — unit coverage.
- `docs/trainers/grpo.md`, `RL_ALGORITHMS.md` — user-facing guidance.

No standalone PPO/GSPO/VinePPO trainers shipped in this release; functionality lives behind the expanded GRPO configuration.
