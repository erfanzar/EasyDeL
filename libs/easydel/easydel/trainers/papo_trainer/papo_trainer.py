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
"""PAPO objective shaping for GRPO rollouts."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from ..metrics import _host_mean_float, _host_scalar_float
from .papo_config import PAPOConfig


@Registry.register("trainer", "papo")
class PAPOTrainer(GRPOTrainer):
    """PAPO trainer using GRPO rollouts plus PAPO objective shaping.

    The trainer keeps GRPO generation and reward scoring intact, then modifies
    the model batch by masking completion-token objectives and optionally adding
    perception/DER reward columns into the computed advantages.
    """

    arguments: PAPOConfig
    _papo_reward_keys = (
        ("papo_perception_reward", "perception_reward", "perception_rewards"),
        ("papo_der_reward1", "der_reward1", "der_rewards1"),
        ("papo_der_reward2", "der_reward2", "der_rewards2"),
    )

    def __init__(
        self,
        arguments: PAPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: object | list[object],
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | list[object] | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
        tools: list[dict | str | tp.Callable[..., object]] | None = None,
        environment_factory: tp.Callable[[], object] | None = None,
    ) -> None:
        """Initialize PAPO on top of the standard GRPO rollout trainer.

        PAPO does not replace GRPO generation or reward scoring. It validates
        its config, delegates rollout setup to ``GRPOTrainer``, and later
        applies PAPO-specific completion masks and optional perception/DER
        reward columns during batch preprocessing.
        """
        if not isinstance(arguments, PAPOConfig):
            raise TypeError(f"arguments must be PAPOConfig, got {type(arguments)}")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
            tools=tools,
            environment_factory=environment_factory,
        )

    def _extract_papo_reward_columns(self, batch: tp.Mapping[str, object]) -> dict[str, object]:
        """Collect PAPO reward side columns under canonical metric names.

        The input dataset may use several alias names for perception and DER
        reward signals. This method returns only the first matched column for
        each canonical PAPO signal.
        """
        columns: dict[str, object] = {}
        for canonical_name, aliases in zip(
            ("perception", "der1", "der2"),
            self._papo_reward_keys,
            strict=True,
        ):
            for key in aliases:
                if key in batch:
                    columns[canonical_name] = batch[key]
                    break
        return columns

    def _papo_objective_mask(self, completion_mask: jax.Array, *, step: int | None = None) -> jax.Array:
        """Build the token-level objective mask requested by PAPO config.

        Prefix, suffix, alternate (honoring ``mask_ratio`` via evenly spaced
        Bresenham selection), and step-seeded pseudo-random masking all
        preserve existing padding masks without signed ``int32`` overflow.
        The output is multiplied into the GRPO completion mask before loss
        computation.
        """
        mask_ratio = float(self.arguments.mask_ratio)
        if mask_ratio <= 0.0:
            return completion_mask
        if mask_ratio >= 1.0:
            return jnp.zeros_like(completion_mask)
        positions = jnp.arange(completion_mask.shape[1], dtype=jnp.int32)[None, :]
        active_lengths = jnp.maximum(jnp.sum(completion_mask, axis=1, keepdims=True), 1)
        if self.arguments.mask_type == "prefix":
            keep_mask = positions >= jnp.floor(active_lengths * mask_ratio).astype(jnp.int32)
        elif self.arguments.mask_type == "suffix":
            keep_mask = positions < jnp.ceil(active_lengths * (1.0 - mask_ratio)).astype(jnp.int32)
        elif self.arguments.mask_type == "alternate":
            # Evenly space masked positions at density ``mask_ratio``; matches
            # ``(positions % 2) == 0`` exactly when ``mask_ratio == 0.5``.
            pos_f = positions.astype(jnp.float32)
            is_masked = jnp.floor((pos_f + 1.0) * mask_ratio) > jnp.floor(pos_f * mask_ratio)
            keep_mask = ~is_masked
        else:
            if step is None:
                step = int(getattr(self, "_papo_mask_step", 0))
                self._papo_mask_step = step + 1
            base_seed = int(getattr(self.arguments, "seed", 0) or 0)
            key = jax.random.fold_in(jax.random.PRNGKey(base_seed), int(step))
            samples = jax.random.uniform(key, shape=completion_mask.shape, dtype=jnp.float32)
            keep_mask = samples >= mask_ratio
        return completion_mask * keep_mask.astype(completion_mask.dtype)

    def _apply_papo_reward_columns(
        self,
        model_batch: dict[str, jax.Array],
        reward_columns: dict[str, object],
        local_prompt_count: int,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int]]:
        """Add weighted PAPO reward columns into completion advantages.

        Reward columns can be prompt-shaped or completion-shaped. Prompt-shaped
        rewards are repeated over the generation factor; mismatched shapes are
        skipped and reported in metrics instead of being broadcast implicitly.
        """
        if not reward_columns:
            return model_batch, {"papo/reward_columns": 0}
        num_prompts = int(model_batch["prompt_ids"].shape[0])
        num_completions = int(model_batch["completion_ids"].shape[0])
        generation_factor = max(num_completions // max(num_prompts, 1), 1)
        reward_delta = jnp.zeros((num_completions,), dtype=jnp.float32)
        metrics: dict[str, float | int] = {"papo/reward_columns": 0}
        for name, weight in (
            ("perception", self.arguments.perception_loss_weight),
            ("der1", self.arguments.der_loss_weight1),
            ("der2", self.arguments.der_loss_weight2),
        ):
            value = reward_columns.get(name)
            if value is None or weight == 0.0:
                continue
            rewards = jnp.asarray(value, dtype=jnp.float32).reshape(-1)
            if local_prompt_count != num_prompts and rewards.shape[0] in (
                local_prompt_count,
                local_prompt_count * generation_factor,
            ):
                rewards = self._all_gather(rewards)
            if rewards.shape[0] == num_prompts:
                rewards = rewards.repeat(generation_factor, axis=0)
            if rewards.shape[0] != num_completions:
                metrics[f"papo/{name}_reward_skipped_shape_mismatch"] = 1
                continue
            reward_delta = reward_delta + float(weight) * rewards
            metrics[f"papo/{name}_reward_mean"] = _host_mean_float(rewards)
            metrics["papo/reward_columns"] += 1
        if metrics["papo/reward_columns"] == 0:
            return model_batch, metrics
        advantages = model_batch["advantages"]
        return {**model_batch, "advantages": advantages + reward_delta.reshape(advantages.shape)}, metrics

    def _apply_papo_objective_shaping(
        self,
        model_batch: dict[str, jax.Array],
        reward_columns: dict[str, object],
        local_prompt_count: int,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int]]:
        """Apply PAPO completion masking and reward-column shaping.

        The method updates ``completion_mask`` and ``num_items_in_batch`` before
        adding optional reward deltas. It also reports the effective kept-token
        fraction so masking behavior is visible in logs.
        """
        original_mask = model_batch["completion_mask"]
        shaped_mask = self._papo_objective_mask(original_mask)
        model_batch = {
            **model_batch,
            "completion_mask": shaped_mask,
            "completion_attention_mask": model_batch.get("completion_attention_mask", original_mask),
            "num_items_in_batch": jnp.sum(shaped_mask),
        }
        model_batch, reward_metrics = self._apply_papo_reward_columns(model_batch, reward_columns, local_prompt_count)
        original_tokens = jnp.maximum(jnp.sum(original_mask), 1)
        kept_tokens = jnp.sum(shaped_mask)
        return model_batch, {
            "papo/mask_ratio": float(self.arguments.mask_ratio),
            "papo/kept_token_fraction": _host_scalar_float(kept_tokens / original_tokens),
            **reward_metrics,
        }

    def _store_buffered_grpo_batch(
        self,
        model_batch: dict[str, jax.Array],
        metrics: dict[str, float | int | str],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Apply PAPO shaping to the full un-sliced batch BEFORE GRPO slices it into distinct chunks.

        When ``steps_per_generation > 1``, ``GRPOTrainer._store_buffered_grpo_batch``
        slices ``model_batch`` into ``steps_per_generation`` distinct minibatches
        inside ``_buffered_grpo_queue``. Shaping here ensures PAPO reward columns
        match the full ``S * B`` prompt count and every queued minibatch retains
        its own distinct shaped prompts, masks, and advantages.
        """
        pending = getattr(self, "_papo_pending_shaping", None)
        if pending is not None:
            reward_columns, local_prompt_count = pending
            self._papo_pending_shaping = None
            self._papo_shaped_in_store = True
            model_batch, papo_metrics = self._apply_papo_objective_shaping(
                model_batch,
                reward_columns,
                local_prompt_count,
            )
            metrics = {**metrics, **papo_metrics}
        return super()._store_buffered_grpo_batch(model_batch, metrics)

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Run GRPO preprocessing, then apply PAPO shaping to the model batch.

        Reward side columns are captured from the raw batch before GRPO purifies
        or transforms it. On training steps, shaping runs inside
        :meth:`_store_buffered_grpo_batch` on the full un-sliced batch prior to
        ``steps_per_generation`` chunk slicing; on eval (or monkeypatched parent
        calls that bypass ``_store_buffered_grpo_batch``), shaping falls back here.
        """
        if is_train and getattr(self, "_buffered_grpo_remaining", 0) > 0:
            # GRPO's cache holds the already-shaped PAPO batch (including the
            # original prompt's reward columns), not the new dataloader row.
            return super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)

        batch = self._apply_user_data_collator(batch)
        reward_columns = self._extract_papo_reward_columns(batch)
        local_prompt_count = int(batch["input_ids"].shape[0])
        self._papo_pending_shaping = (reward_columns, local_prompt_count)
        self._papo_shaped_in_store = False
        try:
            model_batch, metrics = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
        finally:
            self._papo_pending_shaping = None

        if not self._papo_shaped_in_store:
            model_batch, papo_metrics = self._apply_papo_objective_shaping(
                model_batch,
                reward_columns,
                local_prompt_count,
            )
            metrics = {**metrics, **papo_metrics}
            if is_train and getattr(self, "_buffered_grpo_remaining", 0) > 0:
                self._buffered_grpo_batch = (dict(model_batch), dict(metrics))
                queued = getattr(self, "_buffered_grpo_queue", None)
                if isinstance(queued, list) and queued:
                    self._buffered_grpo_queue = [
                        (dict(model_batch), dict(metrics)) for _ in range(int(self._buffered_grpo_remaining))
                    ]
        return model_batch, metrics
