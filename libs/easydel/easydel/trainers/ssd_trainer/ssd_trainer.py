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
"""SSD eSurge self-distillation trainer."""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from jax import numpy as jnp

from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..base_trainer import TrainerConfigureFunctionOutput
from ..group_relative_policy_optimization import GRPOTrainer
from ..sdft_trainer import _zero_reward_func
from ..training_utils import (
    compile_trainer_step,
)
from ._fn import ssd_step
from .ssd_config import SSDConfig


@Registry.register("trainer", "ssd")
class SSDTrainer(GRPOTrainer):
    """Simple self-distillation through eSurge generation and JAX CE loss.

    SSD uses GRPO's generation stack to create one completion per prompt, then
    trains the policy with completion-token cross entropy on those generated
    tokens. It does not use reward advantages for the optimization step.
    """

    arguments: SSDConfig

    def __init__(
        self,
        arguments: SSDConfig,
        model: object,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
    ) -> None:
        """Initialize SSD through the GRPO generation infrastructure.

        SSD reuses GRPO's eSurge rollout, tokenizer, dataset, and dataloader
        setup, but installs a zero reward function and later replaces the GRPO
        optimization step with the SSD cross-entropy step. The constructor only
        wires those inherited pieces; it does not load models from string ids.
        """
        if not isinstance(arguments, SSDConfig):
            raise TypeError(f"arguments must be SSDConfig, got {type(arguments)}")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=_zero_reward_func,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=None,
            data_tokenize_fn=data_tokenize_fn,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile SSD train/eval steps and return trainer runtime functions.

        The compiled functions share the same SSD step with different
        ``is_training`` static flags. The method also installs the static args
        used by the base trainer execution loop and prepares the streaming
        checkpoint manager.
        """
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)
        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,
            self.arguments.logprob_vocab_chunk_size,
            None,
        )
        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            self.arguments.logprob_vocab_chunk_size,
            None,
        )
        static_argnums = tuple(range(2, 9))
        sharded_training_step_function = compile_trainer_step(
            ssd_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        sharded_evaluation_step_function = compile_trainer_step(
            ssd_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        sharded_training_step_function.static_argnums_ = static_argnums
        sharded_evaluation_step_function.static_argnums_ = static_argnums
        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    @staticmethod
    def _ssd_keep_completion_text(text: str) -> bool:
        """Return whether generated SSD text is non-empty enough for training.

        The heuristic filters whitespace-only and extremely short single-line
        completions, which are usually generation failures rather than useful
        self-distillation targets.
        """
        stripped = text.strip()
        return bool(stripped) and (stripped.count("\n") > 0 or len(stripped) >= 10)

    def _filter_ssd_completion_mask(
        self, completion_ids: jax.Array, completion_mask: jax.Array, texts: list[str]
    ) -> jax.Array:
        """Zero completion masks for generated SSD rows filtered as empty.

        Filtering is controlled by ``SSDConfig.filter_empty``. Shape mismatches
        between decoded texts and completion rows are treated as a no-op to
        avoid masking valid tokens based on incomplete decoding metadata.
        """
        if not getattr(self.arguments, "filter_empty", True):
            return completion_mask
        keep = np.asarray([self._ssd_keep_completion_text(text) for text in texts], dtype=np.int32)
        if keep.shape[0] != int(completion_mask.shape[0]):
            return completion_mask
        return completion_mask * jnp.asarray(keep, dtype=completion_mask.dtype)[:, None]

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Generate SSD completions and convert them into a CE model batch.

        Training can reuse a buffered generated batch when GRPO reuse is active.
        Otherwise the method runs local generation, builds completion masks,
        optionally filters empty decoded completions, and returns prompt /
        completion tensors plus timing metrics.
        """
        if is_train:
            cached = self._take_buffered_grpo_batch()
            if cached is not None:
                return cached
        batch = self._apply_user_data_collator(batch)
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
            generation_time = generation_time_fn()
            prompt_ids = results.prompt_ids
            prompt_mask = results.prompt_mask
            completion_ids = results.completion_ids
            completion_mask = self._make_attn_mask(completion_ids)
            texts = self._coerce_generation_texts(results.text, fallback=results.raw_text)
            if not texts:
                completion_ids_array = np.asarray(jax.device_get(completion_ids), dtype=np.int64)
                completion_mask_array = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
                texts = self._decode_prompt_batch(
                    self.processing_class,
                    completion_ids_array,
                    skip_special_tokens=True,
                    pad_token_id=self._pad_token_id,
                    pop_pad_tokens=True,
                    attention_mask=completion_mask_array,
                )
            completion_mask = self._filter_ssd_completion_mask(completion_ids, completion_mask, texts)

        preprocessing_time = preprocessing_time_fn()
        completion_lengths = jnp.sum(completion_mask, axis=-1)
        metrics_dict: dict[str, float | int | str] = {
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "completions/mean_length": float(jnp.mean(completion_lengths)),
            "completions/min_length": float(jnp.min(completion_lengths)),
            "completions/max_length": float(jnp.max(completion_lengths)),
            "ssd/active_sample_ratio": float(jnp.mean((completion_lengths > 0).astype(jnp.float32))),
        }
        model_batch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": jnp.sum(completion_mask),
        }
        if is_train:
            return self._store_buffered_grpo_batch(model_batch, metrics_dict)
        return model_batch, metrics_dict
