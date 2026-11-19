# RL Safety: Off-Policy Contract, Unscorable Rewards, Async Admission

Configuration knobs that keep online RL updates honest when rollouts come from eSurge, rewards can fail, or rollouts
are generated asynchronously. Field names below are `GRPOConfig` fields (inherited by the GRPO family) unless stated
otherwise.

## Off-policy contract (`easydel/trainers/_shared/off_policy.py`)

Per action token, with learner and behaviour (eSurge sampling) log-probs:

```text
lr_t = clip(logp_learner_t - logp_behavior_t, -20, 20)    # non-finite -> 0, counted
k3_t = expm1(lr_t) - lr_t                                 # >= 0
w_t  = exp(lr_t)
K3_b = mean over the sequence's action tokens of k3_t
```

| field                                 | default | effect                                                                                               |
|---------------------------------------|---------|------------------------------------------------------------------------------------------------------|
| `off_policy_k3_threshold`             | `None`  | sequences with `K3_b` above it (or a non-finite log ratio, even when `None`) get zero loss; normalisers keep their token counts |
| `off_policy_min_acceptance`           | `None`  | if `accepted_rows / eligible_rows` is below it, parameters and optimizer state are left unchanged (`off_policy/step_skipped = 1`); a batch without behaviour log-probs is skipped. Must be in `(0, 1]` |
| `startup_behavior_k3_max`             | `None`  | first-step parity check on token-mean K3; a violation skips the first update and raises `RuntimeError` |
| `startup_behavior_tail_fraction_max`  | `0.01`  | first-step bound on the fraction of tokens with `abs(lr_t) > 0.25`                                      |

The first three must be positive when set and require `esurge_importance_sampling_correction=True` (otherwise
`__post_init__` raises).

Logged metrics (`N` = action tokens):

| metric                           | formula                                              |
|----------------------------------|------------------------------------------------------|
| `off_policy/behavior_k3`         | `sum k3_t / N`                                       |
| `off_policy/ess`                 | `(sum w_t)^2 / (N * sum w_t^2)` (`1` = on-policy)    |
| `off_policy/chi2`                | `sum (w_t - 1)^2 / N`                                |
| `off_policy/tail_fraction`       | `count(abs(lr_t) > 0.25) / N`                               |
| `off_policy/accepted_fraction`   | `accepted_rows / eligible_rows` (`1` if none eligible) |
| `off_policy/nonfinite_tokens`    | count of non-finite log ratios                       |
| `off_policy/step_skipped`        | `1` when the update gate closed                      |

Diagnostics are accumulated as additive sums, so gradient accumulation preserves every ratio.

### Scheduled (MPMD) update gate

Under a pipeline schedule (`mpmd_scheduler`), GRPO's scheduled loss adapter reduces the same sums across ranks on the
host and sets the `update_gate` aux key: `accepted_fraction >= off_policy_min_acceptance` and, on the flagged first
step, the parity bounds. Without behaviour log-probs the acceptance gate fails closed. The trainer applies stage-local
gradients only when the gate is open; a missing or non-finite gate value closes it. Every rank computes the gate from
the same host-replicated values.

## Unscorable rewards

A reward function returning `None` marks a sample unscorable (judge or sandbox failure).
`discard_unscorable_groups`:

| value            | behaviour                                                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `None` (default) | per-function opt-in: a `None` from a function whose `discard_unscorable` attribute is `True` drops the whole prompt group (zero advantages; its tokens and sequences leave every loss normaliser). A `None` from any other function becomes NaN and is skipped by the nan-aware reward statistics |
| `True`           | a `None` from any function drops the group                                                                                       |
| `False`          | never drop; every `None` uses the nan-aware aggregation                                                                          |

`make_environment_reward_fn` sets `discard_unscorable = True`; see [RL environments](environments.md).

With `gradient_accumulation_steps > 1`, a batch carrying `difficulty_weights` (dataset weights or discarded groups)
normalises every microbatch by the full-batch weighted count, so a fully or mostly discarded microbatch does not
dilute the gradient. Under `environment_factory`, a `reset` or `step` whose `info` sets `env_error` or `discard`
reports a NaN environment reward (never the placeholder `0.0`), so the opted-in environment column drops the group.
Dynamic sampling gives resampled groups weight `1`, never the zero weight of the discarded group they replace.

## eSurge importance sampling

`esurge_importance_sampling_correction` (default `True`) corrects the learner/rollout mismatch.
`esurge_importance_sampling_mode` (default `"sequence_mask"`), with `cap` = `esurge_importance_sampling_cap`
(default `3.0`) and `cap_low` = `esurge_importance_sampling_cap_low` (default `None` -> `1 / cap`):

| mode                                     | effect                                                                                     |
|------------------------------------------|--------------------------------------------------------------------------------------------|
| `token_truncate`, `sequence_truncate`    | truncate the importance ratio at `cap`                                                     |
| `token_mask`, `sequence_mask`            | mask tokens / sequences whose ratio exceeds `cap`                                          |
| `token_icepop`, `sequence_icepop`        | double-sided band mask outside `[cap_low, cap]` (sequence variant extends it per sequence) |
| `token_log_mask`, `sequence_log_mask`    | drop when `abs(log ratio) > log(cap)`, keep the rest with weight `1`; the sequence variant uses the summed log ratio; requires `cap > 1` |

IcePop modes with `cap_low=None` require `cap >= 1`; an explicit `cap_low` must not exceed `cap`.

## Loss and advantage variants

- **GMPO** (`GMPOConfig`, `loss_type="gmpo"`, `epsilon = epsilon_high = 0.4`): the sequence ratio is the geometric
  mean `exp(mean_t log clip(r_t))` of advantage-sign-aware clipped token ratios, computed in log space. Token-level IS
  weights `w_t` fold into the mean: tokens with `w_t = 0` leave it, others add `stop_gradient(log w_t)`. A row
  whose tokens all have `w_t = 0` gets sequence ratio `0` (zero loss).
- **`scale_rewards`**: `"group"` (default) `A / (std_group(r) + eps)`, `"batch"` `A / (std_batch(r) + eps)`,
  `"batch_advantage"` `(A - mean_batch(A)) / (std_batch(A) + eps)` over group-centred `A = r - baseline`, `"none"`.
  `True` / `False` map to `"group"` / `"none"`.
- **REINFORCE++** (`ReinforcePlusPlusConfig`) defaults to `scale_rewards="batch_advantage"` and `kl_estimator="k2"`.

## Prefix-shared forward (`GRPOConfig.prefix_shared_forward`, default `False`)

With the flag on, the policy loss forward runs each prompt once per group instead of once per completion. A capture
pass records every layer's key/value states for the prompts. A fork pass then scores the `G` completions against the
gathered prefix, with visibility from segment ids and causality from absolute positions.

- **Supported:** dense full-attention models with `attn_mechanism` in `PREFIX_SHARED_ATTENTION_MECHANISMS`
  (`easydel/caching/shared_prefix`): `vanilla`, `sdpa`, `flash_attn2` and `blocksparse`. All four match the
  repeated-prompt forward on TPU for log-probs and every parameter update. `flash_attn2` was checked with fp32 matmul
  precision pinned (XLA kernel at `HIGHEST`, or the Pallas kernel). At the default precision the XLA flash kernel uses
  bf16 passes on TPU, about 6e-3 off for output and 9e-3 for gradients, with masking unchanged. `blocksparse` was
  checked at 512-token prompts and completions, where the TPU kernel itself runs; shorter inputs use its fallback.
- **Rejected:** `completion_chunk_size`, `top_entropy_quantile < 1`, `mpmd_scheduler`, sliding-window or
  non-full-attention layers, quantized KV, attention dropout, and other attention operations.
- **When it pays off:** the saving tracks `(G-1)·P / (G·(P+C))` for prompt length `P` and completion length `C`.
  Measured on 4 TPU v5p chips, 697M-parameter dense model, `vanilla` attention, bf16, 8 prompts per step:

| P | C | G | step time off → on | activation temps off → on |
|---|---|---|---|---|
| 1024 | 256 | 8 | 0.316 s → 0.142 s | 31.1 → 14.2 GiB |
| 2048 | 512 | 8 | 0.805 s → 0.301 s | 90.9 → 41.8 GiB |
| 256 | 1024 | 4 | 0.173 s → 0.170 s | 17.0 → 19.7 GiB |
| 512 | 2048 | 8 (4 prompts) | 0.378 s → 0.382 s | 51.3 → 59.2 GiB |

  Turn it on when prompts are at least as long as completions. When completions are much longer than prompts
  (`C ≥ 4P`) it gave no speedup, roughly doubled compile time and, without rematerialization, used more memory (at
  P=512, C=2048, G=8 with 8 prompts it ran out of HBM where the default path fits).

## Async rollouts (`TrainingArguments`)

| field                 | default | meaning                                                         |
|-----------------------|---------|-----------------------------------------------------------------|
| `async_mode`          | `False` | background rollout generation against a params-only snapshot; also enables eSurge async scheduling and overlap |
| `max_inflight_tasks`  | `32`    | queued/in-flight rollout tasks                                  |
| `max_staleness`       | `1`     | maximum lag, in rollout units (policy versions)                 |
| `queue_maxsize`       | `0`     | `0` uses `max_inflight_tasks`                                    |
| `weight_sync_steps`   | `1`     | rollouts between snapshot refreshes                             |
| `request_timeout`, `first_rollout_timeout` | `120.0`, `3600.0` | seconds                                 |

All bookkeeping runs in **rollout units**: one consumed rollout advances `state.step` by
`_optimizer_steps_per_rollout()` (default `1`), and a step lag converts to a version lag by floor division.
**Deadline admission**: a rollout anchored at version `v` must be consumed by `v + max_staleness`; a lookahead
rollout is started only when every queued deadline stays feasible (`admission_is_feasible`), otherwise it is deferred.
An over-stale rollout found at take time is regenerated. Counters: `async_rollout/admission_deferred`,
`async_rollout/stale_dropped`.

## PPO

- **Async epochs**: `PPOTrainer._optimizer_steps_per_rollout()` returns `num_ppo_epochs`, so each rollout counts as
  one version regardless of how many epochs replay it.
- **Value pretraining** (`value_pretrain_steps`, default `0`; needs `vf_coef > 0`): while
  `state.step < value_pretrain_steps` the loss is `vf_coef * vf_loss` on detached hidden states, and
  `freeze_policy_during_value_pretrain` restores every non-value-head parameter and its optimizer moments to the
  pre-step value (so weight decay and momentum do not move the policy). Counted in optimizer steps, so each PPO epoch
  counts.
