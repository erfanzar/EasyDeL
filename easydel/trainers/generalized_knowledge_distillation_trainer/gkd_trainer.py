# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Generalized Knowledge Distillation trainer."""

from __future__ import annotations

import random
import typing as tp

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from transformers import GenerationConfig

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import deepcopy_model

from ..supervised_fine_tuning_trainer import SFTTrainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import gkd_step
from .gkd_config import GKDConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger(__name__)


@Registry.register("trainer", "gkd")
class GKDTrainer(SFTTrainer):
    """Generalized Knowledge Distillation trainer with optional on-policy sampling.

    Implements GKD training which distills knowledge from a teacher model to a student
    model using a generalized Jensen-Shannon divergence objective. Supports on-policy
    sampling where student-generated completions are used for training.

    Args:
        arguments: GKD-specific training configuration.
        processing_class: Tokenizer or processor for text encoding.
        model: Student model to train (EasyDeLBaseModule or EasyDeLState).
        teacher_model: Teacher model for distillation.
        train_dataset: Training dataset.
        eval_dataset: Optional evaluation dataset.
        formatting_func: Optional function to format dataset examples.
        data_collator: Optional custom data collator.
    """

    arguments: GKDConfig

    def __init__(
        self,
        arguments: GKDConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        formatting_func: tp.Callable | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        if not isinstance(arguments, GKDConfig):
            raise TypeError(f"`arguments` must be a `GKDConfig`, received {type(arguments)}.")

        if isinstance(model, EasyDeLState):
            student_state = model
        else:
            student_state = model.to_state()  # type: ignore[union-attr]

        self.lmbda = float(arguments.lmbda)
        self.seq_kd = bool(arguments.seq_kd)
        self._on_policy_rng = random.Random(getattr(arguments, "seed", None))
        self._warned_missing_prompt = False
        self.gkd_generate_function = None

        if teacher_model is None:
            teacher_state = deepcopy_model(student_state)
        elif isinstance(teacher_model, EasyDeLState):
            teacher_state = teacher_model
        else:
            teacher_state = teacher_model.to_state()

        self.teacher_state = teacher_state

        super().__init__(
            arguments=arguments,
            processing_class=processing_class,
            model=student_state,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            data_collator=data_collator,
        )

        if teacher_model is None:
            self.teacher_state = deepcopy_model(self.model_state)

        if arguments.disable_dropout:
            self.model_state.model.eval()
            self.teacher_state.model.eval()

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("GKDTrainer requires a tokenizer with a defined `pad_token_id`.")

        self.generation_config = GenerationConfig(
            max_new_tokens=self.arguments.max_new_tokens,
            temperature=self.arguments.temperature,
            do_sample=True,
            top_k=0,
            pad_token_id=self.pad_token_id,
        )
        shard_inputs = getattr(self.arguments, "generation_shard_inputs", True)
        try:
            self.gkd_generate_function = self.create_generate_function(
                generation_config=self.generation_config,
                shard_inputs=shard_inputs,
            )
        except Exception as exc:  # pragma: no cover - generation is optional
            self.gkd_generate_function = None
            logger.warning("Failed to initialize GKD generation function, on-policy sampling disabled: %s", exc)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
        """
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,
            float(self.arguments.beta),
            float(self.arguments.temperature),
        )

        static_argnums = (3, 4, 5, 6, 7, 8, 9)
        sharded_training_step_function = ejit(
            gkd_step,
            in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
        )
        sharded_training_step_function.static_argnums_ = static_argnums

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            float(self.arguments.beta),
            float(self.arguments.temperature),
        )

        sharded_evaluation_step_function = ejit(
            gkd_step,
            in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
        )
        sharded_evaluation_step_function.static_argnums_ = static_argnums

        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function
        self._train_shared_fn_extra_args = (self.teacher_state,)
        self._eval_shared_fn_extra_args = (self.teacher_state,)

        teacher_flops = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=False)
        self._extra_forward_flops_per_token = teacher_flops

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Optionally generate on-policy samples before training.

        Args:
            state: Current model state.
            batch: Input batch.
            is_train: Whether this is a training step.

        Returns:
            Processed batch and metrics dictionary.
        """
        # Purify batch first to handle list of dicts (uncollated batch)
        batch = self._purify_batch(batch)
        if not is_train or self.gkd_generate_function is None:
            return batch, {}

        batch_copy = dict(batch)
        info: dict[str, float | int | str] = {}

        if self.seq_kd:
            generated, meta = self._apply_generation(self.teacher_state, batch_copy, source="teacher")
            if generated is not None:
                batch_copy = generated
                info.update(meta)

        if self._should_use_student_sampling():
            generated, meta = self._apply_generation(state, batch_copy, source="student")
            if generated is not None:
                batch_copy = generated
                info.update(meta)

        return batch_copy, info

    def _should_use_student_sampling(self) -> bool:
        """Determine whether to use student on-policy sampling for this batch.

        Returns:
            True if student sampling should be performed.
        """
        if self.lmbda <= 0:
            return False
        return self._on_policy_rng.random() <= self.lmbda

    def _apply_generation(
        self,
        generator_state: EasyDeLState,
        batch: dict[str, jax.Array],
        *,
        source: str,
    ) -> tuple[dict[str, jax.Array] | None, dict[str, float]]:
        """Generate completions using the specified model.

        Args:
            generator_state: Model state to use for generation.
            batch: Input batch containing prompts.
            source: Source identifier for logging ('teacher' or 'student').

        Returns:
            Generated batch and generation metrics.
        """
        prompts = self._extract_prompts(batch)
        if prompts is None:
            if not self._warned_missing_prompt:
                logger.warning(
                    "Unable to derive prompts from batch; on-policy sampling is disabled until prompts are available."
                )
                self._warned_missing_prompt = True
            return None, {}
        prompt_ids, prompt_mask, prompt_lengths = prompts
        prompt_seq_len = int(prompt_ids.shape[1])

        try:
            sequences, _, _ = jax.block_until_ready(
                self.gkd_generate_function(generator_state, prompt_ids, prompt_mask)  # type: ignore[arg-type]
            )
        except Exception as exc:  # pragma: no cover - generation failures are rare
            logger.warning("Failed to generate %s continuations: %s", source, exc)
            return None, {}

        new_batch = self._build_batch_from_sequences(batch, sequences, prompt_seq_len=prompt_seq_len)
        attention_mask = np.asarray(new_batch["attention_mask"])
        completion_lengths = attention_mask.sum(axis=-1) - np.clip(prompt_lengths, 0, attention_mask.shape[1])
        info = {
            f"gkd_completion_length_{source}": float(np.mean(completion_lengths)),
            f"gkd_prompt_length_{source}": float(np.mean(prompt_lengths)),
        }
        return new_batch, info

    def _extract_prompts(self, batch: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array, np.ndarray] | None:
        """Extract prompt token IDs and masks from batch.

        Args:
            batch: Input batch.

        Returns:
            Tuple of (prompt_ids, prompt_mask, prompt_lengths) or None if prompts cannot be extracted.
        """
        if "prompt_input_ids" in batch and "prompt_attention_mask" in batch:
            prompt_ids = jnp.asarray(batch["prompt_input_ids"])
            prompt_mask = jnp.asarray(batch["prompt_attention_mask"]).astype(jnp.int32)
            prompt_lengths = np.asarray(np.sum(np.asarray(prompt_mask), axis=-1), dtype=np.int32)
            return prompt_ids, prompt_mask, prompt_lengths

        input_ids = np.asarray(batch["input_ids"])
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask_np = (input_ids != self.pad_token_id).astype(np.int32)
        else:
            attention_mask_np = np.asarray(attention_mask)
        valid_tokens = attention_mask_np.astype(bool)

        completion_mask = batch.get("completion_mask")
        labels = batch.get("labels")
        if completion_mask is not None:
            completion_mask_np = np.asarray(completion_mask)
            prompt_token_mask = valid_tokens & (~completion_mask_np.astype(bool))
        elif labels is not None:
            labels_np = np.asarray(labels)
            prompt_token_mask = valid_tokens & (labels_np == -100)
        else:
            return None

        prompt_lengths = np.asarray(prompt_token_mask.sum(axis=-1), dtype=np.int32)
        max_prompt = int(prompt_lengths.max()) if prompt_lengths.size else 0
        if max_prompt <= 0:
            return None

        batch_size = input_ids.shape[0]
        prompt_ids_np = np.full((batch_size, max_prompt), self.pad_token_id, dtype=input_ids.dtype)
        prompt_mask_np = np.zeros((batch_size, max_prompt), dtype=np.int32)
        for idx, length in enumerate(prompt_lengths.tolist()):
            if length <= 0:
                continue
            tokens = input_ids[idx][prompt_token_mask[idx]]
            if tokens.size == 0:
                continue
            length = int(min(tokens.size, max_prompt))
            prompt_ids_np[idx, -length:] = tokens[-length:]
            prompt_mask_np[idx, -length:] = 1

        return jnp.asarray(prompt_ids_np), jnp.asarray(prompt_mask_np), prompt_lengths

    def _build_batch_from_sequences(
        self,
        original_batch: dict[str, jax.Array],
        sequences: jax.Array,
        *,
        prompt_seq_len: int,
    ) -> dict[str, jax.Array]:
        """Construct training batch from generated sequences.

        Args:
            original_batch: Original input batch.
            sequences: Generated token sequences.
            prompt_seq_len: Prompt token array length used during generation (includes padding).

        Returns:
            New batch with generated completions and appropriate masks/labels.
        """
        seq_np = np.asarray(sequences)
        seq_len = seq_np.shape[1]
        max_len = self.arguments.max_length
        prompt_cutoff = int(max(prompt_seq_len, 0))
        if max_len is not None and seq_len > max_len:
            start = seq_len - max_len
            seq_np = seq_np[:, start:]
            prompt_cutoff = max(prompt_cutoff - start, 0)
            seq_len = max_len

        attention_mask = (seq_np != self.pad_token_id).astype(np.int32)
        completion_mask = np.zeros_like(attention_mask)
        if prompt_cutoff < seq_len:
            completion_mask[:, prompt_cutoff:] = 1
        completion_mask *= attention_mask

        labels_dtype = np.asarray(original_batch.get("labels", seq_np)).dtype
        labels = seq_np.astype(labels_dtype, copy=True)
        labels[attention_mask == 0] = -100
        if prompt_cutoff > 0:
            labels[:, :prompt_cutoff] = -100

        new_batch = dict(original_batch)
        new_batch["input_ids"] = jnp.asarray(seq_np.astype(np.asarray(original_batch["input_ids"]).dtype, copy=False))
        attn_dtype = np.asarray(original_batch.get("attention_mask", attention_mask)).dtype
        new_batch["attention_mask"] = jnp.asarray(attention_mask.astype(attn_dtype, copy=False))
        new_batch["completion_mask"] = jnp.asarray(completion_mask.astype(attn_dtype, copy=False))
        new_batch["labels"] = jnp.asarray(labels)
        if "position_ids" in original_batch:
            position_ids = np.broadcast_to(np.arange(seq_len, dtype=np.int32), seq_np.shape)
            new_batch["position_ids"] = jnp.asarray(position_ids)
        return new_batch
