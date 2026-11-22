# Single-Rollout Asynchronous Optimization (SAO) Trainer

Single-Rollout Asynchronous Optimization (SAO) is an online actor-critic
algorithm for long agentic trajectories. It consumes one rollout per prompt,
allows rollouts to arrive in completion order, and uses direct double-sided
importance sampling (DIS) to reject policy tokens that are too stale.

## Overview

`SAOTrainer` has three distinct model states:

- the trainable actor;
- a separately optimized critic with its own optimizer, step counter, and
  nested checkpoint.

An optional frozen reference policy is available when `kl_coef` is non-zero,
but the public SAO objective does not require it and the SAO default is zero.

For every generated action token, the trainer computes

```text
r_t = exp(log pi_current(a_t | s_t) - log pi_rollout(a_t | s_t))
```

and retains the token only when

```text
1 - epsilon_low < r_t < 1 + epsilon_high
```

The comparisons are strict. Rejected tokens have zero policy gradient, while
the ratio for accepted tokens is detached before applying the score-function
loss. Observations remain in the attention context but never enter the actor
or critic loss masks.

The default values reproduce the hyperparameters disclosed for the GLM-5.2
reasoning recipe: one rollout, global batch size `128`, actor learning rate
`1e-6`, critic learning rate `5e-6`, 10 critic learning-rate warmup steps, two
critic updates per actor update, `epsilon_low=0.3`, `epsilon_high=5.0`, and
length-adaptive GAE with `alpha=1.5`. The coding recipe can be selected with
`epsilon_low=0.8, epsilon_high=3.0`.

## Configuration

```python
import easydel as ed

config = ed.SAOConfig(
    model_name="sao-agent",
    save_directory="sao-checkpoints",
    max_prompt_length=4096,
    max_completion_length=4096,
    total_batch_size=128,
    learning_rate=1e-6,
    critic_learning_rate=5e-6,
    critic_warmup_steps=10,
    critic_pretrain_steps=0,
    critic_updates_per_actor_update=2,
    dis_epsilon_low=0.3,
    dis_epsilon_high=5.0,
    policy_gae_mode="length_adaptive",
    length_adaptive_alpha=1.5,
    rollout_logprob_source="engine",
)
```

Important settings:

- `critic_trainable_mode="full_except_attention"` freezes critic attention
  parameters while training its other parameters and value head.
- `rollout_logprob_source="engine"` requires the rollout provider to preserve
  target-aligned behavior log probabilities, as required by SAO. The
  `"recompute"` mode is an explicit synchronous/debug fallback when an
  immutable rollout snapshot is available.
- `max_inflight_rollouts`, `rollout_queue_size`, `weight_sync_steps`, and
  `max_policy_staleness` configure an asynchronous rollout provider.
- `global_token_normalization=True` uses one action-token denominator across
  the complete learner batch, including under gradient accumulation.

## Basic Usage

The prompt-only path uses EasyDeL's normal generation and reward plumbing. It
is useful for validating the trainer without a multi-turn environment.

```python
import easydel as ed
from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    model_id,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
)
dataset = load_dataset("openai/gsm8k", "main", split="train")


def exact_answer_reward(completions, answer, **_):
    return [
        float(reference.strip() in str(completion))
        for completion, reference in zip(completions, answer, strict=True)
    ]


trainer = ed.SAOTrainer(
    arguments=ed.SAOConfig(
        model_name="sao-gsm8k",
        max_prompt_length=1024,
        max_completion_length=1024,
        total_batch_size=8,
    ),
    model=model,
    reward_funcs=exact_answer_reward,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## eLarge Configuration

The unified eLarge runner recognizes `sao`,
`single_rollout_asynchronous_optimization`, and
`single-rollout-asynchronous-optimization`.

```yaml
config:
  model:
    name_or_path: Qwen/Qwen3-0.6B
  reward_model:
    name_or_path: OpenAssistant/reward-model-deberta-v3-large-v2
    task: sequence-classification
    extra_kwargs:
      num_labels: 1
  mixture:
    informs:
      - type: hf
        data_files: openai/gsm8k
        split: train
  trainer:
    trainer_type: sao
    max_prompt_length: 1024
    max_completion_length: 1024
    total_batch_size: 8
    dis_epsilon_low: 0.3
    dis_epsilon_high: 5.0
actions:
  - train
```

Run it with:

```bash
python -m easydel.scripts.elarge --config sao.yaml
```

## Dataset Format

The built-in rollout path expects prompt rows. Conversational prompts may be
represented as chat messages; plain text is accepted when
`skip_apply_chat_template=True`.

```json
{
  "prompt": [{"role": "user", "content": "Solve 27 * 14."}],
  "answer": "378"
}
```

Reward-side metadata such as `answer` is preserved and routed only when the
reward callable requests it by name.

## Asynchronous and Agentic Rollouts

Pass `rollout_provider` for multi-turn environments or true completion-order
collection. The provider returns full trajectory rows. All arrays except
`input_ids` and `attention_mask` are aligned with `input_ids[:, 1:]`.

```python
def rollout_provider(*, trainer, state, batch, is_train):
    del trainer, state, batch, is_train
    return {
        "input_ids": input_ids,              # [batch, sequence]
        "attention_mask": attention_mask,    # [batch, sequence]
        "action_mask": action_mask,          # [batch, sequence - 1]
        "rewards": rewards,                  # [batch, sequence - 1]
        "done_mask": done_mask,              # [batch, sequence - 1]
        "behavior_logps": behavior_logps,    # required in engine mode
        "bootstrap_value": bootstrap_value,  # optional [batch]
    }
```

`easydel.trainers._online_rl.BoundedAsyncRolloutCoordinator` provides bounded
worker concurrency, queue backpressure, policy-version metadata, completion
order, cancellation, timeout, and staleness accounting. The coordinator does
not copy a sharded model automatically; its `PolicySnapshot.state` must be an
inference-safe snapshot owned by the caller.

Take that snapshot with
`easydel.trainers._shared.OwnedPolicySnapshot.from_training_state`, which is the
same mechanism AsyncGRPO uses. It copies parameters and buffers only (the
optimizer transform and slots are dropped) and exposes an explicit `release()`
so the outgoing copy can be freed *before* the next one is allocated -- keeping
exactly one extra copy of the policy resident. The copy is mandatory: the
compiled train step is compiled with `donate_argnums=(0,)`, so generating from
the live state races with buffer donation.

```python
from easydel.trainers._shared import OwnedPolicySnapshot

snapshot = OwnedPolicySnapshot.from_training_state(
    state,
    policy_step=int(jax.device_get(state.step)),
    cache_scope_key=f"{state.esurge_cache_scope_key}-rollout",
)
coordinator.submit(payload, snapshot=PolicySnapshot(version=snapshot.policy_step, state=snapshot.state))
...
snapshot.release()  # only once no worker is reading it
```

Pass a `cache_scope_key` distinct from the training state's so the snapshot's
eSurge engine is cached separately, and release only between rollouts: the
engine keeps its own reference to the arrays it last generated from, and picks
up the replacement's weights through the normal engine weight refresh.

## Checkpoints

The actor uses the standard EasyDeL checkpoint layout. The critic is stored in
the checkpoint's `critic/` subdirectory, with
`online_rl_checkpoint.json` recording actor and critic steps. Resume restores
both states before optional checkpoint cleanup. Saving first marks the
manifest incomplete; it becomes resumable only after both halves finish, and
restore validates the trainer algorithm plus actor and critic steps.

## Practical Notes

1. A prompt-only run is synchronous and recomputes its rollout statistics;
   asynchronous behavior with exact rollout logs begins when an asynchronous
   `rollout_provider` is supplied.
2. Engine log probabilities must correspond to the exact sampled tokens. Use
   recomputation only with a preserved rollout-policy snapshot.
3. CPU tests validate masking, loss math, state plumbing, and shapes. They do
   not validate TPU eSurge overlap, Mosaic lowering, or throughput.
4. The public paper does not disclose the complete production rollout system,
   anti-hacking prompts, or every GLM-5.2 training detail. EasyDeL exposes the
   disclosed algorithm and explicit extension points instead of claiming
   bit-identical reproduction.

## References

- [Single-Rollout Asynchronous Optimization](https://arxiv.org/abs/2607.07508)
- [GLM-5.2: Advancing Open Foundation Models](https://z.ai/blog/glm-5.2)
