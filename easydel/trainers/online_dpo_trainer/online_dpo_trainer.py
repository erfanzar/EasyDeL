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
"""Online DPO built from eSurge completions."""

from __future__ import annotations

import typing as tp
from itertools import chain

import jax
import numpy as np
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..direct_preference_optimization_trainer import DPOTrainer
from ..trainer import Trainer
from .online_dpo_config import OnlineDPOConfig


@Registry.register("trainer", "online_dpo")
class OnlineDPOTrainer(DPOTrainer):
    """Online DPO trainer using eSurge generation and EasyDeL's JAX DPO loss.

    The trainer accepts raw prompt batches, generates two completions per prompt
    with local EasyDeL/eSurge inference, scores them with reward functions, and
    converts the winner/loser pair into a normal DPO batch for the parent loss.
    """

    arguments: OnlineDPOConfig

    def __init__(
        self,
        arguments: OnlineDPOConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reward_funcs: object | list[object],
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | list[object] | None = None,
        data_collator: tp.Callable[..., object] | None = None,
    ) -> None:
        """Initialize online DPO reward routing and the inherited DPO trainer.

        Args:
            arguments: Online DPO config with generation parameters and reward
                weights.
            model: Initialized EasyDeL policy module or state. String model ids
                are intentionally not loaded here.
            reward_funcs: One or more initialized reward states or Python reward
                callables used to score the two generated completions.
            reference_model: Optional initialized reference model/state for the
                parent DPO machinery.
            train_dataset: Raw prompt dataset consumed before on-policy
                generation.
            eval_dataset: Optional evaluation dataset or named evaluation
                dataset mapping.
            processing_class: Tokenizer or processor used for prompt encoding
                and completion decoding.
            reward_processing_classes: Optional processors paired one-to-one
                with ``reward_funcs``.
            data_collator: Optional collator override for raw prompt batches.

        Raises:
            TypeError: If ``arguments`` is not ``OnlineDPOConfig``.
            ValueError: If no reward function is provided, a reward model id is
                passed instead of an initialized object, or reward processors
                and weights do not match the reward function count.
        """
        if not isinstance(arguments, OnlineDPOConfig):
            raise TypeError(f"arguments must be OnlineDPOConfig, got {type(arguments)}")
        reward_func_list = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        if not reward_func_list:
            raise ValueError("`reward_funcs` must contain at least one callable or EasyDeL reward state.")
        for reward_func in reward_func_list:
            if isinstance(reward_func, str):
                raise ValueError(
                    "EasyDeL OnlineDPO does not accept reward model ids. "
                    "Pass an initialized EasyDeL reward state or a Python reward function."
                )
        if reward_processing_classes is None:
            reward_processing_class_list = [None] * len(reward_func_list)
        elif isinstance(reward_processing_classes, list):
            reward_processing_class_list = reward_processing_classes
        else:
            reward_processing_class_list = [reward_processing_classes]
        if len(reward_processing_class_list) != len(reward_func_list):
            raise ValueError("The number of reward processing classes must match the number of reward functions.")
        if arguments.reward_weights is not None and len(arguments.reward_weights) != len(reward_func_list):
            raise ValueError("`reward_weights` must match the number of reward functions.")
        self.reward_funcs = reward_func_list
        self.reward_processing_classes = reward_processing_class_list
        self.reward_func_names = [getattr(func, "__name__", None) or type(func).__name__ for func in reward_func_list]
        self.reward_weights = jnp.asarray(
            arguments.reward_weights if arguments.reward_weights is not None else [1.0] * len(reward_func_list),
            dtype=jnp.float32,
        )
        super().__init__(
            arguments=arguments,
            model=model,
            reference_model=reference_model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    @staticmethod
    def _online_prompt_collator(examples: list[dict[str, object]]) -> dict[str, object]:
        """Collate raw prompt rows without tokenizing before online generation.

        Online DPO must keep prompt text and side-channel fields intact until
        generation. The collator therefore returns lists per key instead of
        applying tokenizer padding or truncation.
        """
        if not examples:
            return {}
        keys = set().union(*(example.keys() for example in examples))
        return {key: [example.get(key) for example in examples] for key in keys}

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the raw prompt collator used by Grain dataloaders.

        ``max_sequence_length`` and ``truncation_mode`` are accepted for trainer
        interface compatibility, but online DPO generation handles sequence
        length later through eSurge generation settings.
        """
        del max_sequence_length, truncation_mode
        return self._online_prompt_collator

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the raw prompt collator used by TFDS-style dataloaders.

        The TFDS path uses the same raw prompt representation as Grain so
        generated completions can be produced from identical prompt payloads.
        """
        del max_sequence_length, truncation_mode
        return self._online_prompt_collator

    def _build_source_from_dataset(self, dataset: object | None):
        """Build a sharded source for raw online prompts without preprocessing.

        DPO tokenization is intentionally delayed until after online generation
        creates chosen/rejected completions. The returned source is therefore a
        direct sharded view of the user dataset.
        """
        return self._to_sharded_source(dataset)

    def _get_preprocess_transform(self) -> None:
        """Disable dataset preprocessing because prompts are generated online.

        Parent DPO preprocessing expects chosen/rejected rows, which raw Online
        DPO datasets do not have yet. This method makes that delayed conversion
        explicit.
        """
        return None

    def configure_dataloaders(self):
        """Use the base trainer dataloader path with online prompt collation.

        Calling :class:`Trainer` directly avoids DPO's eager preference-pair
        preprocessing while preserving EasyDeL's normal dataloader setup.
        """
        return Trainer.configure_dataloaders(self)

    def _online_generation_overrides(self) -> dict[str, object]:
        """Return eSurge generation overrides for two online DPO completions.

        The override dictionary is filtered to omit ``None`` values and always
        requests two return sequences because the trainer needs a pair to choose
        a preferred and rejected completion.
        """
        overrides: dict[str, object] = {
            "num_return_sequences": 2,
            "max_new_tokens": self.arguments.generation_max_new_tokens,
            "temperature": self.arguments.generation_temperature,
            "top_p": self.arguments.generation_top_p,
            "top_k": self.arguments.generation_top_k,
            "repetition_penalty": self.arguments.generation_repetition_penalty,
        }
        overrides["num_return_sequences"] = 2
        return {key: value for key, value in overrides.items() if value is not None}

    @staticmethod
    def _make_online_dpo_labels(
        prompt_mask: jax.Array,
        completion_ids: jax.Array,
        completion_mask: jax.Array,
        *,
        label_pad_token_id: int,
    ) -> jax.Array:
        """Build DPO labels with prompt positions masked out.

        Prompt tokens are filled with ``label_pad_token_id`` so the supervised
        loss only scores generated completion tokens. Padding positions in the
        completion are masked with the same label pad id.
        """
        prompt_labels = jnp.full(prompt_mask.shape, label_pad_token_id, dtype=completion_ids.dtype)
        completion_labels = jnp.where(completion_mask.astype(bool), completion_ids, label_pad_token_id)
        return jnp.concatenate([prompt_labels, completion_labels], axis=-1)

    def _score_online_dpo_rewards(
        self,
        *,
        prompts: list[object],
        completions: list[object],
        completion_ids: jax.Array,
        batch: dict[str, object],
    ) -> jax.Array:
        """Score generated completions with reward models/functions.

        EasyDeL reward states are called with tokenized prompt+completion text;
        Python reward callables receive prompts, completions, completion ids, and
        any original batch side channels accepted by their signature.
        """
        sidechannels = {
            key: value
            for key, value in batch.items()
            if key not in {"prompt", "input_ids", "attention_mask", "prompt_input_ids", "prompt_attention_mask"}
        }
        reward_rows = []
        for reward_func, reward_processing_class in zip(self.reward_funcs, self.reward_processing_classes, strict=False):
            if isinstance(reward_func, EasyDeLState):
                if reward_processing_class is None:
                    raise ValueError("EasyDeLState reward functions require a matching reward processing class.")
                texts = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions, strict=False)]
                reward_inputs = dict(
                    reward_processing_class(
                        texts,
                        return_tensors="np",
                        padding="max_length",
                        padding_side="right",
                        add_special_tokens=False,
                        truncation=True,
                        return_attention_mask=True,
                        max_length=self.arguments.max_length,
                    )
                )
                reward_output = reward_func.apply_fn(
                    reward_func.graphdef,
                    reward_func.graphstate,
                    reward_func.graphother,
                    reward_inputs,
                )
                reward_values = reward_output.logits[:, 0]
            else:
                reward_call_kwargs = self._build_reward_call_kwargs(
                    reward_func,
                    prompts=prompts,
                    completions=completions,
                    completion_ids=np.asarray(jax.device_get(completion_ids)).tolist(),
                    max_length=self.arguments.max_length,
                    batch=batch,
                    **sidechannels,
                )
                reward_values = reward_func(**reward_call_kwargs)
            reward_rows.append(
                jnp.asarray([jnp.nan if value is None else value for value in reward_values], dtype=jnp.float32)
            )
        rewards_per_func = jnp.stack(reward_rows, axis=1)
        return jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)

    def _build_online_dpo_batch(
        self,
        *,
        prompt_ids: jax.Array,
        prompt_mask: jax.Array,
        completion_ids: jax.Array,
        completion_mask: jax.Array,
        rewards: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Convert paired online completions into a chosen/rejected DPO batch.

        For each prompt, the first two generated completions are compared by
        reward. The higher-scoring completion becomes ``chosen`` and the other
        becomes ``rejected``; optional EOS penalties are applied before ranking.
        """
        prompt_count = int(prompt_ids.shape[0])
        generation_factor = int(completion_ids.shape[0]) // max(prompt_count, 1)
        if generation_factor < 2:
            raise RuntimeError("OnlineDPO requires at least two generated completions per prompt.")
        adjusted_rewards = rewards
        if self.arguments.missing_eos_penalty is not None:
            eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
            has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=-1)
            adjusted_rewards = rewards - float(self.arguments.missing_eos_penalty) * (~has_eos).astype(jnp.float32)
        first_indices = jnp.arange(prompt_count) * generation_factor
        second_indices = first_indices + 1
        first_rewards = adjusted_rewards[first_indices]
        second_rewards = adjusted_rewards[second_indices]
        choose_first = first_rewards >= second_rewards
        chosen_indices = jnp.where(choose_first, first_indices, second_indices)
        rejected_indices = jnp.where(choose_first, second_indices, first_indices)

        chosen_completion_ids = completion_ids[chosen_indices]
        rejected_completion_ids = completion_ids[rejected_indices]
        chosen_completion_mask = completion_mask[chosen_indices]
        rejected_completion_mask = completion_mask[rejected_indices]
        chosen_input_ids = jnp.concatenate([prompt_ids, chosen_completion_ids], axis=-1)
        rejected_input_ids = jnp.concatenate([prompt_ids, rejected_completion_ids], axis=-1)
        chosen_attention_mask = jnp.concatenate([prompt_mask, chosen_completion_mask], axis=-1)
        rejected_attention_mask = jnp.concatenate([prompt_mask, rejected_completion_mask], axis=-1)

        eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
        chosen_has_eos = jnp.any(jnp.isin(chosen_completion_ids, eos_tokens), axis=-1)
        rejected_has_eos = jnp.any(jnp.isin(rejected_completion_ids, eos_tokens), axis=-1)
        score_margin = adjusted_rewards[chosen_indices] - adjusted_rewards[rejected_indices]

        batch = {
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": self._make_online_dpo_labels(
                prompt_mask,
                chosen_completion_ids,
                chosen_completion_mask,
                label_pad_token_id=self.arguments.label_pad_token_id,
            ),
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": self._make_online_dpo_labels(
                prompt_mask,
                rejected_completion_ids,
                rejected_completion_mask,
                label_pad_token_id=self.arguments.label_pad_token_id,
            ),
        }
        metrics = {
            "online_dpo/reward_margin": float(jnp.nanmean(score_margin)),
            "online_dpo/chosen_reward": float(jnp.nanmean(adjusted_rewards[chosen_indices])),
            "online_dpo/rejected_reward": float(jnp.nanmean(adjusted_rewards[rejected_indices])),
            "online_dpo/chosen_contains_eos": float(jnp.mean(chosen_has_eos.astype(jnp.float32))),
            "online_dpo/rejected_contains_eos": float(jnp.mean(rejected_has_eos.astype(jnp.float32))),
        }
        return batch, metrics

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, object],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Generate, score, and collate online DPO pairs for one raw prompt batch.

        Already-tokenized DPO preference batches are delegated to the parent
        implementation. Raw prompt batches are generated through eSurge, scored,
        converted into chosen/rejected tensors, and then passed through the DPO
        shared step path.
        """
        if "chosen_input_ids" in batch and "rejected_input_ids" in batch:
            return super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)

        prompts = batch.get("prompt")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if prompts is None and input_ids is None:
            raise ValueError("OnlineDPO batches must contain `prompt` text or pretokenized `input_ids`.")

        with capture_time() as generation_time_fn:
            results = self.generate_unified(
                input_ids=None if input_ids is None else jnp.asarray(input_ids),
                attention_mask=None if attention_mask is None else jnp.asarray(attention_mask),
                prompts=prompts,
                state=state,
                apply_chat_template=False,
                shard_inputs=False,
                all_gather=False,
                config_overrides=self._online_generation_overrides(),
            )
        completion_ids = results.completion_ids
        completion_mask = results.completion_mask
        prompt_ids = results.prompt_ids
        prompt_mask = results.prompt_mask

        completion_texts = self._coerce_generation_texts(results.text, fallback=results.raw_text)
        if not completion_texts:
            completion_texts = self._decode_prompt_batch(
                self.processing_class,
                np.asarray(jax.device_get(completion_ids), dtype=np.int64),
                True,
                self._pad_token_id,
                True,
                np.asarray(jax.device_get(completion_mask), dtype=np.int32),
            )
        prompt_records = list(results.completion_prompts or [])
        if len(prompt_records) != int(completion_ids.shape[0]):
            base_prompts = prompts
            if base_prompts is None:
                base_prompts = self._decode_prompt_batch(
                    self.processing_class,
                    np.asarray(jax.device_get(prompt_ids), dtype=np.int64),
                    True,
                    self._pad_token_id,
                    True,
                    np.asarray(jax.device_get(prompt_mask), dtype=np.int32),
                )
            prompt_list = list(base_prompts) if isinstance(base_prompts, list) else [base_prompts]
            generation_factor = int(completion_ids.shape[0]) // max(len(prompt_list), 1)
            prompt_records = list(chain.from_iterable([prompt] * generation_factor for prompt in prompt_list))

        with capture_time() as rewarding_time_fn:
            rewards = self._score_online_dpo_rewards(
                prompts=prompt_records,
                completions=completion_texts,
                completion_ids=completion_ids,
                batch=batch,
            )
        dpo_batch, metrics = self._build_online_dpo_batch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            rewards=rewards,
        )
        metrics["online_dpo/generation_time"] = generation_time_fn()
        metrics["online_dpo/rewarding_time"] = rewarding_time_fn()
        return dpo_batch, metrics
