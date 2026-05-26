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
"""GRPO replay-buffer trainer extension."""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from .replay_buffer import _ReplayBuffer


@Registry.register("trainer", ["grpo_with_replay_buffer", "grpo_replay_buffer"])
class GRPOWithReplayBufferTrainer(GRPOTrainer):
    """GRPO trainer variant that replaces flat rollout groups from replay.

    Groups with non-zero advantage variance are stored in a host-side
    score-prioritized buffer. Later groups with flat advantages can be replaced
    with sampled buffered groups of compatible tensor shape.
    """

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Initialize GRPO and attach a bounded host replay buffer.

        The constructor accepts the same positional/keyword contract as
        ``GRPOTrainer``. ``arguments`` is inspected before delegation only to
        recover ``replay_buffer_size`` and ``seed``; model setup, reward setup,
        tokenizer handling, and eSurge generation remain owned by the base GRPO
        trainer. After base initialization, ``self.replay_buffer`` holds
        score-prioritized rollout groups and is inert when capacity is zero.
        """
        arguments = kwargs.get("arguments")
        if arguments is None and args:
            arguments = args[0]
        super().__init__(*args, **kwargs)
        replay_buffer_size = int(getattr(arguments, "replay_buffer_size", 0) or 0)
        self.replay_buffer = _ReplayBuffer(replay_buffer_size, seed=getattr(arguments, "seed", None))

    @staticmethod
    def _pad_sequence_axis(array: jax.Array, target_length: int, *, value: float | int, left: bool) -> jax.Array:
        """Pad or truncate axis 1 to ``target_length`` for replayed tensors.

        Prompt tensors are left-padded to preserve suffix-aligned prompts, while
        completion tensors are right-padded to preserve generated token order.
        When a replayed tensor is longer than the active batch shape, the same
        side convention is used for truncation.
        """
        current_length = int(array.shape[1])
        if current_length == target_length:
            return array
        if current_length > target_length:
            return array[:, current_length - target_length :] if left else array[:, :target_length]
        delta = target_length - current_length
        pad_width = [(0, 0), (delta, 0) if left else (0, delta), *[(0, 0)] * (array.ndim - 2)]
        return jnp.pad(array, tuple(pad_width), constant_values=value)

    def _replay_known_batch_keys(self) -> set[str]:
        """Return model-batch keys known to be safe for replay substitution.

        Replay replacement only runs when every key in the current model batch
        is listed here. That prevents silently dropping or incorrectly slicing
        future side tensors introduced by other GRPO extensions.
        """
        return {
            "advantages",
            "completion_ids",
            "completion_mask",
            "difficulty_weights",
            "importance_sampling_ratio",
            "num_items_in_batch",
            "old_per_token_logps",
            "prompt_ids",
            "prompt_mask",
            "ref_per_token_logps",
            "sampling_per_token_logps",
        }

    def _apply_replay_buffer(
        self,
        model_batch: dict[str, jax.Array],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int]]:
        """Store useful rollout groups and replace flat groups from replay.

        The method keeps replay strictly local to known GRPO tensors. If a batch
        carries additional keys, it skips replacement rather than guessing how
        those side tensors should be sliced or padded.
        """
        replay_buffer = getattr(self, "replay_buffer", None)
        if replay_buffer is None or replay_buffer.max_size <= 0:
            return model_batch, {"replay_buffer_size": 0, "replay_groups_added": 0, "replay_groups_replaced": 0}

        unsupported_keys = set(model_batch) - self._replay_known_batch_keys()
        if unsupported_keys:
            return model_batch, {
                "replay_buffer_size": len(replay_buffer),
                "replay_groups_added": 0,
                "replay_groups_replaced": 0,
                "replay_skipped_unsupported_keys": 1,
            }

        prompt_ids = model_batch["prompt_ids"]
        completion_ids = model_batch["completion_ids"]
        num_groups = int(prompt_ids.shape[0])
        num_generations = int(getattr(self.arguments, "num_generations", 0) or getattr(self, "num_generations", 0) or 0)
        if num_groups <= 0 or num_generations <= 0 or int(completion_ids.shape[0]) != num_groups * num_generations:
            return model_batch, {
                "replay_buffer_size": len(replay_buffer),
                "replay_groups_added": 0,
                "replay_groups_replaced": 0,
                "replay_skipped_shape_mismatch": 1,
            }

        advantages = model_batch["advantages"]
        advantage_groups = advantages.reshape(num_groups, num_generations, *advantages.shape[1:])
        advantage_magnitude = jnp.max(jnp.abs(advantage_groups).reshape(num_groups, -1), axis=1)
        has_variance = np.asarray(jax.device_get(advantage_magnitude > 1e-8), dtype=bool)
        added = 0
        for group_index, keep_group in enumerate(has_variance):
            if not keep_group:
                continue
            start = group_index * num_generations
            end = start + num_generations
            group_advantages = advantage_groups[group_index]
            score = float(
                jax.device_get(
                    jnp.sum(jnp.abs(group_advantages)) * (jnp.std(group_advantages.astype(jnp.float32)) + 1e-8)
                )
            )
            replay_group = {
                "advantages": group_advantages,
                "completion_ids": model_batch["completion_ids"][start:end],
                "completion_mask": model_batch["completion_mask"][start:end],
                "prompt_ids": model_batch["prompt_ids"][group_index : group_index + 1],
                "prompt_mask": model_batch["prompt_mask"][group_index : group_index + 1],
            }
            for key in (
                "difficulty_weights",
                "importance_sampling_ratio",
                "old_per_token_logps",
                "ref_per_token_logps",
                "sampling_per_token_logps",
            ):
                if key in model_batch:
                    replay_group[key] = model_batch[key][start:end]
            replay_buffer.add(score, replay_group)
            added += 1

        replace_group_indices = [int(index) for index, keep_group in enumerate(has_variance) if not keep_group]
        sampled_groups = replay_buffer.sample(len(replace_group_indices))
        if not sampled_groups:
            return model_batch, {
                "replay_buffer_size": len(replay_buffer),
                "replay_groups_added": added,
                "replay_groups_replaced": 0,
            }

        updated_batch = dict(model_batch)
        pad_token_id = int(getattr(self, "_pad_token_id", 0) or 0)
        target_prompt_len = max(
            int(updated_batch["prompt_ids"].shape[1]),
            *(int(group["prompt_ids"].shape[1]) for group in sampled_groups),
        )
        target_completion_len = max(
            int(updated_batch["completion_ids"].shape[1]),
            *(int(group["completion_ids"].shape[1]) for group in sampled_groups),
        )
        updated_batch["prompt_ids"] = self._pad_sequence_axis(
            updated_batch["prompt_ids"], target_prompt_len, value=pad_token_id, left=True
        )
        updated_batch["prompt_mask"] = self._pad_sequence_axis(
            updated_batch["prompt_mask"], target_prompt_len, value=0, left=True
        )
        for key, value in list(updated_batch.items()):
            if key in {
                "completion_ids",
                "completion_mask",
                "importance_sampling_ratio",
                "old_per_token_logps",
                "ref_per_token_logps",
                "sampling_per_token_logps",
            }:
                pad_value = pad_token_id if key == "completion_ids" else 0
                updated_batch[key] = self._pad_sequence_axis(value, target_completion_len, value=pad_value, left=False)

        replaced = 0
        for group_index, sampled_group in zip(replace_group_indices, sampled_groups, strict=False):
            start = group_index * num_generations
            end = start + num_generations
            sampled_prompt_ids = self._pad_sequence_axis(
                sampled_group["prompt_ids"], target_prompt_len, value=pad_token_id, left=True
            )
            sampled_prompt_mask = self._pad_sequence_axis(
                sampled_group["prompt_mask"], target_prompt_len, value=0, left=True
            )
            sampled_completion_ids = self._pad_sequence_axis(
                sampled_group["completion_ids"], target_completion_len, value=pad_token_id, left=False
            )
            sampled_completion_mask = self._pad_sequence_axis(
                sampled_group["completion_mask"], target_completion_len, value=0, left=False
            )
            updated_batch["prompt_ids"] = (
                updated_batch["prompt_ids"].at[group_index : group_index + 1].set(sampled_prompt_ids)
            )
            updated_batch["prompt_mask"] = (
                updated_batch["prompt_mask"].at[group_index : group_index + 1].set(sampled_prompt_mask)
            )
            updated_batch["completion_ids"] = updated_batch["completion_ids"].at[start:end].set(sampled_completion_ids)
            updated_batch["completion_mask"] = (
                updated_batch["completion_mask"].at[start:end].set(sampled_completion_mask)
            )
            updated_batch["advantages"] = (
                updated_batch["advantages"]
                .at[start:end]
                .set(sampled_group["advantages"].reshape(updated_batch["advantages"][start:end].shape))
            )
            for key in (
                "difficulty_weights",
                "importance_sampling_ratio",
                "old_per_token_logps",
                "ref_per_token_logps",
                "sampling_per_token_logps",
            ):
                if key not in updated_batch or key not in sampled_group:
                    continue
                sampled_value = sampled_group[key]
                if key in {
                    "importance_sampling_ratio",
                    "old_per_token_logps",
                    "ref_per_token_logps",
                    "sampling_per_token_logps",
                }:
                    sampled_value = self._pad_sequence_axis(sampled_value, target_completion_len, value=0.0, left=False)
                updated_batch[key] = (
                    updated_batch[key].at[start:end].set(sampled_value.reshape(updated_batch[key][start:end].shape))
                )
            replaced += 1

        updated_batch["num_items_in_batch"] = jnp.sum(updated_batch["completion_mask"])
        return updated_batch, {
            "replay_buffer_size": len(replay_buffer),
            "replay_groups_added": added,
            "replay_groups_replaced": replaced,
        }

    def _store_buffered_grpo_batch(
        self,
        model_batch: dict[str, jax.Array],
        metrics: dict[str, float | int | str],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Apply replay-buffer substitution before normal GRPO batch buffering.

        The replay pass may add metrics describing inserted and replaced groups.
        Those metrics are merged into the original rollout metrics and then
        passed to the parent GRPO reuse cache unchanged.
        """
        model_batch, replay_metrics = self._apply_replay_buffer(model_batch)
        merged_metrics = {**metrics, **replay_metrics}
        return super()._store_buffered_grpo_batch(model_batch, merged_metrics)
