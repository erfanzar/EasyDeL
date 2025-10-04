# EasyDeL Reinforcement Learning Algorithms

This document captures the current reinforcement-learning and post-training support in EasyDeL, focusing on how recent research features (ProRL, DAPO, VinePPO) were integrated into the existing GRPO trainer. It also summarises other post-training trainers shipped with the framework (SFT, ORPO, distillation, reward modelling).

## 1. Unified GRPO Stack

All policy-gradient–style reinforcement learning now funnels through `GRPOTrainer` (`easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`). Rather than maintaining separate PPO, DAPO, or VinePPO trainers, we expanded GRPO so each functionality is toggled through configuration.

### Key Capabilities
-
- **Group Advantage Normalisation (`adv_estimator="group"`)** — legacy GRPO behaviour with z-score normalisation across sampled completions.
- **Monte-Carlo/GAE Advantages (`adv_estimator="gae"`)** — activates VinePPO-style returns; rewards are passed through optional discounted estimators controlled by `gae_gamma` and `gae_lambda`.
- **Truncated Returns (`adv_estimator="truncated"`)** — short-horizon credit assignment as described in VinePPO; length governed by `truncated_return_k`.
- **Asymmetric Clipping (`epsilon_low`, `epsilon_high`)** — implements DAPO’s decoupled clip window so beneficial updates have higher headroom than detrimental ones.
- **Per-token Weighting (`per_token_weighting=True`)** — token-level loss balance identical to DAPO’s GSPO-token formulation.
- **Length Shaping (`length_shaping`)** — linear or punitive schedules that penalise overlong completions, matching DAPO’s overlength reward strategy.
- **KL/Entropy Management (`kl_target`, `reference_reset_style`, `entropy_floor`)** — ProRL-inspired moving-average tracking; once KL grows too high or entropy collapses, the reference policy is reset (either hard copy or mixup via `ref_model_mixup_alpha`).
- **Sampling Diagnostics (`enforce_mixed_sampling`, `positive_reward_threshold`)** — ensures diverse completions per prompt and jitters rewards if every rollout matches.

All features are optional; omitting them reverts to the original GRPO training loop, guaranteeing backwards compatibility.

### Relevant Files
-
- `easydel/trainers/group_relative_policy_optimization/grpo_config.py`
- `easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
- `easydel/trainers/group_relative_policy_optimization/_fn.py`
- `easydel/trainers/training_utils.py`

## 2. Other Post-Training Trainers in EasyDeL

While GRPO handles multi-sample RLHF updates, EasyDeL also bundles other post-training strategies that often precede or complement RL:

| Trainer | Path | Purpose |
| ------- | ---- | ------- |
| `SFTTrainer` | `easydel/trainers/supervised_fine_tuning_trainer/` | Supervised fine-tuning (instruction-tuning) with optional packing, LoRA, etc. |
| `DistillationTrainer` | `easydel/trainers/distillation_trainer/` | Teacher–student distillation, including pooling heads and mixture tokens. |
| `RewardTrainer` | `easydel/trainers/reward_trainer/` | Trains scalar reward models consumed by RLHF loops. |
| `DPOTrainer` | `easydel/trainers/direct_preference_optimization_trainer/` | Direct Preference Optimization; now reports KL, entropy, logit margins per batch. |
| `ORPOTrainer` | `easydel/trainers/odds_ratio_preference_optimization_trainer/` | Odds-Ratio Preference Optimization (classification-style preference learning). |

Together these cover the common post-training pipeline stages: supervised warm-up (SFT/distillation), reward modelling, preference optimisation (DPO/ORPO), and finally RL (GRPO with ProRL/DAPO/VinePPO options).

## 3. Reasons for a Unified Trainer

Maintaining a single RL trainer keeps the codebase DRY:
- **Shared infrastructure** — generation, reward gathering, sharding, checkpointing, and logging already exist inside `GRPOTrainer`. Extending it avoids duplicating that machinery across four near-identical trainers.
- **Composable features** — users can mix and match (e.g., DAPO-style clipping with VinePPO advantages) by toggling config values instead of swapping trainer classes.
- **Backwards compatibility** — existing configs continue to work because each new argument defaults to the legacy behaviour.
- **Lower maintenance cost** — bug fixes, optimisation, and instrumentation live in one place.

We can still introduce a dedicated PPO or VinePPO trainer later if requirements diverge significantly; for now, the single configurable pipeline is easier to test and reason about.

## 4. Quick Reference: New GRPOConfig Fields

A non-exhaustive list of new options (see source for defaults and docstrings):

```python
GRPOConfig(
    epsilon_low=0.2,
    epsilon_high=0.2,
    adv_estimator="group",   # or "gae", "truncated"
    gae_gamma=0.99,
    gae_lambda=0.95,
    truncated_return_k=1,
    per_token_weighting=True,
    length_shaping="none",   # "linear" or "punitive"
    length_reward_scale=1.0,
    kl_target=None,
    kl_horizon=100,
    reference_reset_style="hard",  # "mix" or "none"
    entropy_floor=None,
    enforce_mixed_sampling=True,
    dynamic_sampling_jitter=1e-3,
)
```

Refer to `docs/trainers/grpo.md` for usage examples.

## 5. Testing

The new helper functions are covered by `tests/trainers/grpo_utils_test.py`; this verifies group advantage calculations and DAPO-style length shaping. Full trainer tests still require JAX/TPU setup and are expected to run in the project’s CI environment.

---

_Last updated: 2025-10-03_
