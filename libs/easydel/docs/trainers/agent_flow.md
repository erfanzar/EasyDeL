# AgentFlow Trainer and Agentic Rollout Plumbing

`AgentFlowTrainer` (registry name `agent_flow`, config `AgentFlowConfig`) wraps an existing EasyDeL trainer and runs
multi-turn environment episodes for it. Environments come from [RL environments](environments.md).

## How it works

In `context_mode="append"` (default) each episode is one token sequence: every turn generates from the running token
list, appends the generated action ids (`action_mask=1`) and the observation ids (`action_mask=0`), and the inner
trainer trains on exactly that sequence, so old log-probs stay aligned. `"sliding_window"` (with
`sliding_window_turns`) and `"reset_with_summary"` truncate the generation context but still emit the full trajectory.

Episode reward: `reward_mode` = `"sum"` (default), `"terminal"`, or `"discounted_sum"` with `gamma`.

**Template-delta observations.** Unless `skip_apply_chat_template=True`, an observation is rendered as
`render(history + [obs as user], add_generation_prompt=True)` minus `render(history)` (string-prefix difference), so
preambles (default system block, BOS) are not re-emitted mid-trajectory. The assistant turn's closing markup is
prepended (minus EOS when the action already ended with it). If the prefix check fails, a standalone rendering is
used. Prompts already ending with the template's generation prompt are not templated twice.

## Configuration

| `AgentFlowConfig` field          | default    | notes                                                                       |
|----------------------------------|------------|-----------------------------------------------------------------------------|
| `base_trainer_type`              | `"grpo"`   | registry name of the inner trainer                                          |
| `base_config`                    | `None`     | inner config instance or dict; when `None`, built from the shared fields    |
| `environment`                    | `None`     | name, mapping spec, or factory                                              |
| `environment_backend`            | `"local"`  | `"local"`, `"kiri"`, `"docker"`, `"auto"`                                   |
| `environment_endpoints`          | `None`     | daemon endpoints, or `"shm:N"` / `"pod-shm:N"`; builds a `SandboxPool`     |
| `max_turns`, `max_workers`       | `8`, `8`   |                                                                             |
| `group_size`                     | `None`     | defaults to the inner `num_generations` / `num_return_sequences`, else `1` |
| `per_turn_max_tokens`, `max_obs_tokens_per_turn` | `None` | per-turn generation and observation token caps                     |

Supported inner trainers include `grpo`, `gmpo`, `gdpo`, `reinforce_plus_plus`, `ppo`, `vapo`, `gkd`,
`on_policy_distillation` and `online_dpo`. When no `reward_funcs` are given, GRPO-family and OnlineDPO inner trainers
receive a placeholder reward (`agent_flow_supplied_reward`) that raises if ever called: the wrapper supplies episode
rewards itself.

### OnlineDPO inner trainer

- Group size is at least `2` (from `group_size`, else the inner `generation_num_return_sequences`, else `2`).
- `select_group_preference_pairs`: per group, among valid episodes (not `env_error`, finite reward) the first episode
  with the maximum reward is `chosen` and the first with the minimum is `rejected`. Groups with fewer than two
  candidates, all-equal rewards, or different prompts yield no pair.
- Completions carry the whole trajectory under the attention mask; `chosen_loss_mask` / `rejected_loss_mask` equal the
  action masks, so only generated tokens enter the DPO log-prob sums.
- The batch keeps one row per group for a static shape; a group without a pair gets all-zero loss masks. The DPO loss
  marks pairs with at least one scored token on both sides as `pair_valid` and averages only over those.
- Metrics: `agent_flow/dpo_pairs`, `dpo_groups`, `dpo_skipped_ties`, `dpo_skipped_insufficient_valid`,
  `dpo_skipped_prompt_mismatch`, `dpo_reward_margin`.

### eLarge

```yaml
trainer:
  trainer_type: agent_flow
  base_trainer_type: grpo
  environment: code_judge          # name or mapping
  environment_backend: local       # agent_flow default; other trainers default to kiri
  environment_endpoints: null      # or ["shm:4"] / explicit endpoints
  max_turns: 8
```

`environment`, `environment_backend` and `environment_endpoints` are eLarge-only keys; they are forwarded only to
argument classes that declare them (AgentFlow, AsyncGRPO). For GRPO-family `trainer_type`s, a `trainer.environment`
spec without a Python factory builds `AutoEnvironment.as_factory(spec, runtime=...)` (a `SandboxPool` when endpoints
are set, else `KiriRuntime(backend=...)`) and adds a zero-weight placeholder reward so the environment reward is
counted once.

## On-policy distillation

`OnPolicyDistillationConfig`:

- **Streamed divergences**: with `logits_chunk_size > 0`, plain forward KL uses the chunked KL kernel; the other
  objectives (`beta` JSD, `cakld_gamma`, `loss_top_k`, `loss_add_tail`) and trust-region mode stream the vocabulary
  (online logsumexp, running top-k) from hidden states without materialising `[B, T, V]` logits
  (`_streamed_divergences.py`, `VocabChunkedProjection` / `FullVocabProjection`).
- **`distillation_mode="trust_region"`**: speculative-decoding-style acceptance of sampled tokens, with a teacher
  top-k forward KL on rejected tokens (`trust_region_top_k`, default `64`; `trust_region_seed`). All log-probs use the
  **sampling** temperature (`trust_region_sampling_temperature`: `generation_temperature` if set, else
  `temperature_sampling`), not the distillation `temperature`, and **no `T**2` factor** is applied. A non-positive
  sampling temperature raises. The acceptance ratio is exact only for an untruncated, unpenalised student proposal,
  so in this mode generation top-k/top-p default to off (`0` / `1.0`), and the config refuses
  `generate_with_teacher=True`, any top-k/top-p/min-p truncation, and presence/frequency/repetition penalties.

## eSurge weight-version provenance

- `CompletionOutput.weight_version_segments`: `[(start_token_index, version), ...]`; token `i` was sampled with the
  version of the last segment starting at or before `i`. `oldest_weight_version` / `newest_weight_version` summarise
  it; `None` when not recorded. Trainers expand it per token (`expand_weight_version_segments`). The segments record
  the weights each token was **sampled** with only; the version of the KV a token attends to is not recorded.
- `eSurge.update_model_weights(..., version=None, keep_requests=False)` installs a strictly increasing version
  (`None` = current + 1); concurrent calls are serialised, so the version never decreases and is never reused. By
  default the engine must be idle (it raises instead of draining). With `keep_requests=True` the swap happens between
  two model steps and in-flight requests continue (tokens after the swap carry the new version; earlier KV entries are
  kept and **not tracked**, so a prompt prefilled, fully or partly, under the old weights and sampled under the new
  ones reports only the new version). Limits: requires `enable_prefix_caching=False`, single-host engines, and a graph
  whose compiled programs the runner already holds.
- `eSurge.abort_requests_before_version(min_version)` aborts unfinished requests that emitted any token with a version
  below `min_version`; requests without tokens yet are kept (even if their prompt KV was computed under older weights).
  Returns the aborted parent request ids. It is a sampling-version fence, **not** a bound on KV staleness: after a
  `keep_requests=True` swap, kept requests may still attend to KV computed under older weights.
