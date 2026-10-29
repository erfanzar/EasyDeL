# Compaction Reinforcement Learning (CompactionRL) Trainer

Compaction Reinforcement Learning trains an agent to summarize its own working
context and continue a long task from that summary. `CompactionRLTrainer`
implements the critic-based, group-size-one objective described for GLM-5.2
and exposes reusable context-compaction and tool-guard controls.

## Overview

An agentic trajectory consists of atomic assistant-action and environment-
observation pairs. Compaction is checked only after a complete pair has been
recorded. The trigger is strict:

```text
context_budget - current_tokens < compaction_threshold
```

At a compaction boundary, the same policy generates a summary. The controller
reconstructs the context from:

1. the original system messages;
2. a resume message containing the policy summary;
3. the most recent complete action/observation pairs that fit.

The number of retained pairs starts at `recent_steps` and decreases until the
new context fits. Copied context and observations are attended but not
optimized. Newly sampled execution and summary tokens are optimized.

Every execution and summary segment receives the same final task reward. The
trainer computes local, skip-observation GAE and applies the published
cross-segment correction

```text
(gamma * lambda) ** N_after
```

where `N_after` counts optimized tokens in later segments of the same
trajectory. The actor uses one clipped-PPO loss normalized over all generated
action and summary tokens. A separate critic is updated before actor
advantages are refreshed.

## Configuration

```python
import easydel as ed

config = ed.CompactionRLConfig(
    model_name="compaction-agent",
    save_directory="compaction-rl-checkpoints",
    context_budget=65_536,
    compaction_threshold=10_240,
    max_compactions=3,
    recent_steps=2,
    max_assistant_tokens=10_240,
    max_summary_tokens=2_048,
    max_turns=250,
    total_batch_size=128,
    learning_rate=2e-6,
    weight_decay=0.0,
    critic_learning_rate=3e-6,
    critic_pretrain_steps=50,
    critic_updates_per_actor_update=2,
    critic_trainable_mode="all",
    critic_gae_mode="match_policy",
    train_summary_tokens=True,
    cross_trajectory_gae=True,
    global_token_normalization=True,
)
```

The defaults match the disclosed 64K recipe where the paper provides a value.
EasyDeL exposes AdamW rather than a separate Adam enum; with the default
`weight_decay=0.0`, its update is equivalent to Adam.
The exact production summary instruction and resume prompt are not public.
`summary_instruction` and `resume_template` therefore use explicit EasyDeL
defaults and should be treated as tunable policy prompts.

## Basic Usage

Without a rollout provider, CompactionRL uses the inherited one-segment
prompt/completion path. That validates the actor, critic, reward, and PPO loss,
but it does not compact context.

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


def task_reward(completions, answer, **_):
    return [
        float(reference.strip() in str(completion))
        for completion, reference in zip(completions, answer, strict=True)
    ]


trainer = ed.CompactionRLTrainer(
    arguments=ed.CompactionRLConfig(
        model_name="compaction-gsm8k",
        max_prompt_length=4096,
        max_completion_length=2048,
        context_budget=8192,
        compaction_threshold=2048,
        critic_pretrain_steps=0,
        total_batch_size=4,
    ),
    model=model,
    reward_funcs=task_reward,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Full Agentic Rollout Contract

True CompactionRL requires a `rollout_provider` that runs the environment,
invokes the policy summarizer through `CompactionController`, and returns
one full-context row per execution or summary segment. This separation is
load-bearing: a post-compaction segment was sampled under a reconstructed
context and must not causally attend to the discarded pre-compaction history.
The canonical path is to build one immutable `OnlineRLTrajectory` per segment
and batch those rows with `OnlineRLTrajectoryCollator`:

```python
import numpy as np

from easydel.trainers._online_rl import (
    OnlineRLTrajectory,
    OnlineRLTrajectoryCollator,
    TrajectorySegment,
)

collator = OnlineRLTrajectoryCollator(
    max_length=65_536,
    pad_token_id=tokenizer.pad_token_id,
)


def rollout_provider(*, trainer, state, batch, is_train):
    segment_rows, final_scores = run_agentic_environment(
        trainer=trainer,
        state=state,
        prompts=batch,
        is_train=is_train,
    )
    learner_batch = collator(segment_rows)
    # Repeat each trajectory's final task score for all of its segment rows.
    learner_batch["scores"] = np.asarray(final_scores, dtype=np.float32)
    return learner_batch
```

Each row stores its actual full causal context, token-aligned `action_mask`,
exact behavior log probabilities, and exactly one `TrajectorySegment`.
Rows from the same task share `trajectory_id`; their increasing `segment_id`
values define chronological order. The collator shifts masks and log
probabilities onto `input_ids[:, 1:]`. `NaN` is only a missing/non-action
log-probability sentinel; the learner validates action positions and sanitizes
every non-action position before JAX tracing.

When `scores` is supplied, the learner places each row's score at that
segment's final action token. Providers may instead materialize the repeated
task reward directly in each row's token-level `rewards`. A row containing
more than one optimized segment is rejected because ordinary 2-D causal
attention cannot represent a compaction reset.

The shared `easydel.trainers.agentic_rollout` package provides:

- `AtomicInteractionStep`, which keeps an assistant action and observation
  indivisible;
- `CompactionController`, which implements the strict trigger and fit loop;
- `TwoStageToolGuard`, which applies a deterministic rule filter followed by
  an optional batched intent judge;
- `ToolDispatcher`, which prevents blocked calls from executing and inserts a
  stable dummy observation so the rollout continues;
- redacted guard events containing hashes and bounded reason codes rather than
  raw tool arguments or observations.

The same policy must be used for execution and summary sampling. A
`PolicySummary` includes both text and generated token IDs so summary ownership
can be represented in the actor mask. Call `CompactionController.maybe_compact`
with the immutable `original_system_messages` and the exact
`history_prefix_messages` currently visible before the atomic steps. The first
prefix includes the original user instruction; after a compaction it includes
the prior resume-summary message instead. Summary generation sees that full
current history, while reconstruction retains only the original system prompt,
the new resume-summary message, and the fitting complete-step tail.

## eLarge Configuration

eLarge recognizes `compaction_rl`, `compaction-rl`, and
`compaction_reinforcement_learning`.

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
    trainer_type: compaction_rl
    context_budget: 65536
    compaction_threshold: 10240
    max_compactions: 3
    recent_steps: 2
    max_prompt_length: 55296
    max_completion_length: 10240
    total_batch_size: 4
actions:
  - train
```

Run the one-segment configuration with:

```bash
python -m easydel.scripts.elarge --config compaction_rl.yaml
```

For real environment interaction, construct `eLargeModel` programmatically and
pass `rollout_provider`, `environment_factory`, `tools`, `tool_guard`, and
optionally `compaction_controller` to `train()`. Callables are runtime objects
and are intentionally not serialized into YAML.

## Dataset Format

The learner starts from prompt rows. Environment observations, tool results,
summary segments, action masks, and behavior log probabilities are produced by
the rollout provider rather than stored in the source dataset.

```json
{
  "prompt": [
    {"role": "system", "content": "Use the available tools and verify the final result."},
    {"role": "user", "content": "Complete this long-running coding task."}
  ],
  "task_id": "example-001"
}
```

## Tool-Call Guarding

The optional two-stage guard follows this flow:

```text
normalized tool call
  -> deterministic rule result
  -> optional intent judge for reviewed calls
  -> execute allowed call OR insert dummy observation
  -> continue the same rollout
```

A blocked call is never executed. The dummy observation has
`terminated=False` and `truncated=False`, so it does not turn suspicious
behavior into a shortcut termination reward.

The exact GLM-5.2 anti-hacking rules, judge model, and prompts are not public.
EasyDeL supplies the disclosed control flow and injectable rule/judge
contracts, not undisclosed production policy.

## Checkpoints and Validation

Actor checkpoints use the standard EasyDeL layout; the independent critic is
saved under `critic/`. A paired manifest stays incomplete until both states
finish saving and validates the trainer algorithm plus both steps on resume.
CPU tests cover the trigger boundary, atomic retention,
summary token ownership, cross-segment factors, global token normalization,
guard non-execution, redacted telemetry, and configuration/registry surfaces.

TPU eSurge overlap, long-context memory use, environment throughput, and
end-to-end reward quality require a hardware run and are not established by
the CPU suite.

## References

- [Compaction Reinforcement Learning](https://arxiv.org/abs/2607.05378)
- [GLM-5.2: Advancing Open Foundation Models](https://z.ai/blog/glm-5.2)
