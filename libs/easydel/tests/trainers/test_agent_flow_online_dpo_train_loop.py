# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end AgentFlow + OnlineDPO through the real ``trainer.train()`` loop (accelerator only).

A tiny Llama is trained for two steps on a scripted two-turn environment with deterministic rewards:

* ``pair-*`` prompts: the two episodes of each group act ``"aa"`` (reward 4) and ``"bb"`` (reward 0), so every
  group yields a valid chosen/rejected pair and the step must move the parameters;
* ``tie-*`` prompts: both episodes act ``"aa"``, every group is a tie, all pair rows are placeholders
  (``pair_valid == 0``), so the loss and gradient are exactly zero and the parameters must not move.

RMSProp without momentum and ``weight_decay=0`` make a zero gradient a zero update regardless of step order
(Adam would carry momentum from an earlier pair step; weight decay would shrink parameters on its own).
"""

from __future__ import annotations

import re

import easydel as ed
import jax
import numpy as np
import spectrax as spx
from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from easydel.environments._base import AgenticEnvironment, ResetResult, StepResult
from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from jax import numpy as jnp

VOCAB = 128
_PROMPT_RE = re.compile(r"^(tie|pair)-\d+")


class _CharTokenizer:
    """ASCII tokenizer: one token per character (ids < 128), 0 is padding."""

    pad_token = "\x00"
    eos_token = "\x01"
    pad_token_id = 0
    eos_token_id = 1
    bos_token = None
    chat_template = None
    padding_side = "left"

    def __len__(self):
        return VOCAB

    def __call__(self, text, add_special_tokens: bool = False, **kwargs):
        ids = [ord(c) % VOCAB for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens: bool = True, **kwargs) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) > 1)

    def batch_decode(self, rows, skip_special_tokens: bool = True, **kwargs):
        return [self.decode(r, skip_special_tokens=skip_special_tokens) for r in rows]


class _ScriptedEnv(AgenticEnvironment):
    """Two turns; each step's reward is the number of ``a`` characters in the action."""

    def __init__(self, prompt: str = "", **kwargs):
        self._prompt = prompt
        self._turn = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        self._turn = 0
        return ResetResult(observation=self._prompt, info={})

    def step(self, action: str) -> StepResult:
        self._turn += 1
        done = self._turn >= 2
        return StepResult(
            observation="" if done else "o1",
            reward=float(action.count("a")),
            terminated=done,
            truncated=False,
            info={},
        )


def _make_generate_turn():
    """Deterministic policy stub: per prompt, episodes alternate "aa"/"bb" on pair prompts, always "aa" on ties."""
    episode_counter: dict[str, int] = {}
    current_parity: dict[str, int] = {}

    def generate_turn(state, prompt_ids, prompt_text, turn_idx):
        match = _PROMPT_RE.match(prompt_text)
        assert match is not None, prompt_text
        key = match.group(0)
        if turn_idx == 0:
            current_parity[key] = episode_counter.get(key, 0) % 2
            episode_counter[key] = episode_counter.get(key, 0) + 1
        text = "aa" if key.startswith("tie") or current_parity[key] == 0 else "bb"
        return [ord(c) for c in text], text, None

    return generate_turn


def _tiny_llama(tokenizer):
    config = ed.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    )
    config.pad_token_id = tokenizer.pad_token_id
    with config.mesh:
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return model.shard_model()


def test_agent_flow_online_dpo_trains_on_pairs_and_ignores_all_tie_steps(tmp_path):
    n = jax.device_count()  # one prompt group per device keeps the pair batch evenly shardable
    tokenizer = _CharTokenizer()
    prompts = [f"tie-{i}" for i in range(n)] + [f"pair-{i}" for i in range(n)]
    arguments = OnlineDPOConfig(
        model_name="agent-flow-online-dpo-e2e",
        save_directory=str(tmp_path),
        max_training_steps=2,
        num_train_epochs=1,
        total_batch_size=n,
        learning_rate=1e-2,
        optimizer="rmsprop",
        weight_decay=0.0,
        beta=0.1,
        max_prompt_length=16,
        max_new_tokens=48,
        max_length=64,
        do_eval=False,
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        log_steps=1_000_000,
        report_steps=1_000_000,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    inner = OnlineDPOTrainer(
        arguments=arguments,
        model=_tiny_llama(tokenizer),
        reward_funcs=lambda **kw: [0.0] * len(kw["completions"]),
        train_dataset=Dataset.from_dict({"prompt": prompts}),
        processing_class=tokenizer,
    )
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            base_trainer_type="online_dpo",
            max_turns=2,
            group_size=2,
            max_workers=1,
            reward_mode="sum",
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _ScriptedEnv(**kw),
        generate_turn_fn=_make_generate_turn(),
    )

    params_before = [np.asarray(x) for x in jax.device_get(jax.tree_util.tree_leaves(inner.model_state.graphstate))]
    records: list[dict] = []
    original_on_step_end = inner.on_step_end

    def _record_step_end(state, metrics, step):
        other = dict(metrics.other_metrics or {})
        records.append(
            {
                "step": int(step),
                "loss": float(jax.device_get(metrics.loss)),
                "max_grad_norm": float(jax.device_get(metrics.max_grad_norm)),
                "dpo_pairs": float(other["agent_flow/dpo_pairs"]),
                "dpo_groups": float(other["agent_flow/dpo_groups"]),
                "params": [np.asarray(x) for x in jax.device_get(jax.tree_util.tree_leaves(state.graphstate))],
            }
        )
        return original_on_step_end(state=state, metrics=metrics, step=step)

    inner.on_step_end = _record_step_end

    output = flow.train()

    assert int(jax.device_get(output.state.step)) == 2
    assert len(records) == 2, [r["step"] for r in records]
    # Each step consumed one batch of a single kind: one with every group tied, one with every group paired.
    assert sorted(r["dpo_pairs"] for r in records) == [0.0, float(n)]
    assert all(r["dpo_groups"] == float(n) for r in records)

    prev = params_before
    for record in records:
        assert np.isfinite(record["loss"])
        moved = any(not np.array_equal(a, b) for a, b in zip(prev, record["params"], strict=True))
        if record["dpo_pairs"] == 0.0:
            # All pairs invalid: loss and gradient are exactly zero, so the parameters are untouched.
            assert record["loss"] == 0.0
            assert record["max_grad_norm"] == 0.0
            assert not moved
        else:
            assert record["loss"] > 0.0
            assert record["max_grad_norm"] > 0.0
            assert moved
        prev = record["params"]
