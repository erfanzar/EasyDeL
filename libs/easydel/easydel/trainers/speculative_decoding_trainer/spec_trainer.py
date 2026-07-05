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

"""Speculative-decoding trainer for target-matched draft models.

The trainer owns two EasyDeL states: a trainable drafter and a frozen target.
Each step runs the target under ``stop_gradient``, runs the drafter, and
optimizes the drafter logits against the target distribution.  The same code
path handles normal causal-LM drafters, multi-token draft heads, and drafters
that need target logits or hidden states as extra inputs.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import spectrax as spx
from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import SFTPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import speculative_decoding_step
from .spec_config import SpeculativeDecodingConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

logger = get_logger(__name__)


@Registry.register("trainer", ["speculative_decoding", "spec_decoding", "spec"])
class SpeculativeDecodingTrainer(Trainer):
    """Train a speculative-decoding drafter against a frozen verifier model.

    ``SpeculativeDecodingTrainer`` fits a smaller or cheaper drafter so its
    proposed token distribution matches the target model used during
    speculative decoding.  The target state is never updated; it is only used
    to produce teacher logits, optional hidden states, and acceptance-style
    metrics for the drafter.

    The training objective is the weighted mixture configured by
    :class:`SpeculativeDecodingConfig`: target-distribution KL plus optional
    supervised CE on labels.  A plain causal LM can be used directly through
    its ``logits`` output.  Draft-specific modules can instead expose
    ``draft_logits`` / ``mtp_logits`` or a method such as
    ``compute_mtp_chain`` that returns multi-step draft logits.

    The compiled step accepts both drafter and target states, so normal SPMD
    training and SpectraX scheduled MPMD both keep the target forward inside
    the same train/eval step instead of precomputing a separate artifact.
    """

    target_state: EasyDeLState
    arguments: SpeculativeDecodingConfig

    def __init__(
        self,
        arguments: SpeculativeDecodingConfig,
        processing_class: ProcessingClassType,
        drafter_model: EasyDeLBaseModule | EasyDeLState | None = None,
        target_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
    ):
        """Create the trainer and materialize drafter/target states.

        The drafter is converted with ``trainable_selector`` so only the
        intended parameters receive optimizer updates.  The target is converted
        to an ``EasyDeLState`` in eval mode and stored separately as
        ``target_state``; gradients never flow through that state.  Passing
        ``model`` is supported as a launcher-compatible alias for
        ``drafter_model``.

        Args:
            arguments (SpeculativeDecodingConfig): Speculative-decoding trainer
                configuration.
            processing_class (ProcessingClassType): Tokenizer or processor used
                by the SFT-style preprocessing fallback.
            drafter_model (EasyDeLBaseModule | EasyDeLState | None): Trainable
                drafter module or state. ``model`` is accepted as an alias for
                launcher compatibility. Defaults to ``None``.
            target_model (EasyDeLBaseModule | EasyDeLState | None): Frozen
                target module or state. Defaults to ``None``.
            train_dataset (Dataset | None): Training dataset. Defaults to
                ``None``.
            eval_dataset (Dataset | dict[str, Dataset] | None): Optional
                evaluation dataset or a dictionary of per-split datasets.
                Defaults to ``None``.
            data_collator (DataCollatorForCompletionOnlyLM | None): Optional
                completion-only collator. Defaults to ``None``.
            model (EasyDeLBaseModule | EasyDeLState | None): Alias for
                ``drafter_model``. Defaults to ``None``.

        Raises:
            TypeError: If ``arguments`` is not a
                :class:`SpeculativeDecodingConfig`.
            ValueError: If both ``drafter_model`` and ``model`` are provided
                and they are not the same object.
        """
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        if not isinstance(arguments, SpeculativeDecodingConfig):
            raise TypeError("passed argument must be a `SpeculativeDecodingConfig`.")
        if drafter_model is not None and model is not None and drafter_model is not model:
            raise ValueError("Pass only one of `drafter_model` or `model`.")
        if drafter_model is None:
            drafter_model = model

        self.arguments = arguments
        drafter_model = self._resolve_model(drafter_model, role="drafter model")
        target_model = self._resolve_target_model(
            target_model=target_model,
            target_model_revision=arguments.target_model_revision,
        )

        # When the caller does not pass an explicit trainable_selector, defer to
        # the drafter model's own property so that e.g. DSpark's
        # freeze_embeddings_and_lm_head is honoured automatically. Reassign the
        # local `arguments` (not self.arguments) so the SAME instance reaches
        # super().__init__ — otherwise the base trainer re-binds the original
        # and every arguments-derived re-split disagrees with the drafter state.
        if arguments.trainable_selector is None and hasattr(drafter_model, "trainable_selector"):
            arguments = dataclasses.replace(arguments, trainable_selector=drafter_model.trainable_selector)

        if not isinstance(drafter_model, EasyDeLState):
            drafter_model = drafter_model.to_state(trainable_selector=arguments.trainable_selector)
        if not isinstance(target_model, EasyDeLState):
            target_model.eval()
            target_model = target_model.to_state()
        if arguments.disable_dropout:
            drafter_model = disable_state_dropout(drafter_model)
            target_model = disable_state_dropout(target_model)

        self.target_state = target_model

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=drafter_model,
            data_collator=data_collator,
            processing_class=processing_class,
        )

    @staticmethod
    def _resolve_model(
        model: EasyDeLBaseModule | EasyDeLState | None,
        *,
        role: str,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Return an already initialized module/state for a required role.

        This trainer deliberately does not turn string model IDs into loaded
        modules.  Launchers must construct the drafter and target explicitly so
        the caller controls checkpoint conversion, mesh placement, and
        sharding before training starts.

        Args:
            model (EasyDeLBaseModule | EasyDeLState | None): An initialized
                EasyDeL module or state, or ``None``.
            role (str): Human-readable name for the role (e.g.
                ``"drafter model"`` or ``"target model"``), used in error
                messages.

        Returns:
            EasyDeLBaseModule | EasyDeLState: The same module or state object
            that was passed in.

        Raises:
            ValueError: If ``model`` is ``None`` or a string model identifier.
        """
        if model is None:
            raise ValueError(f"`{role}` must be provided for speculative-decoding training.")
        if isinstance(model, str):
            reject_string_model_id(model, role=role)
        return model

    @staticmethod
    def _resolve_target_model(
        *,
        target_model: EasyDeLBaseModule | EasyDeLState | None,
        target_model_revision: str | None,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Validate and return the frozen verifier model/state.

        ``target_model_revision`` is metadata for external loaders, not an
        in-trainer download hook.  By this point the target must already be an
        initialized EasyDeL module or state.

        Args:
            target_model (EasyDeLBaseModule | EasyDeLState | None): Frozen
                target module or state.
            target_model_revision (str | None): Optional revision string for
                external loaders. Must be ``None`` inside the trainer.

        Returns:
            EasyDeLBaseModule | EasyDeLState: The validated target module or
            state.

        Raises:
            ValueError: If ``target_model`` is ``None``.
            ValueError: If ``target_model_revision`` is not ``None``.
        """
        if target_model_revision is not None:
            raise ValueError("`target_model_revision` is only metadata for externally loaded targets.")
        return SpeculativeDecodingTrainer._resolve_model(target_model, role="target model")

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile train and eval steps that carry both model states.

        The returned functions take the drafter state, batch, and frozen target
        state.  Without ``mpmd_scheduler`` they are regular ``spx.jit``
        functions.  With a SpectraX MPMD schedule they use
        ``compile_trainer_step`` so the target forward, drafter forward, loss,
        and drafter backward are all represented inside the scheduled step.

        Returns:
            TrainerConfigureFunctionOutput: The compiled train/eval step
            functions, mesh, and checkpoint manager.
        """
        self._runtime_trace(
            "configure_functions.speculative_decoding",
            mpmd_scheduler=self.arguments.mpmd_scheduler,
            gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
        )
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,
            self.arguments.temperature,
            self.arguments.alpha,
            self.arguments.target_logits_attr,
            self.arguments.draft_logits_attr,
            self.arguments.draft_chain_method,
            self.arguments.drafter_forward_method,
            self.arguments.target_forward_method,
            self.arguments.num_draft_tokens,
            self.arguments.draft_target_offset,
            bool(self.arguments.target_output_hidden_states),
            bool(self.arguments.pass_target_outputs),
            straight_through_emulator,
        )
        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            self.arguments.temperature,
            self.arguments.alpha,
            self.arguments.target_logits_attr,
            self.arguments.draft_logits_attr,
            self.arguments.draft_chain_method,
            self.arguments.drafter_forward_method,
            self.arguments.target_forward_method,
            self.arguments.num_draft_tokens,
            self.arguments.draft_target_offset,
            bool(self.arguments.target_output_hidden_states),
            bool(self.arguments.pass_target_outputs),
            None,
        )

        static_argnums = tuple(range(3, 20))
        in_shardings = (self.state_shardings, empty_sharding, self.target_state.shardings)

        self._runtime_trace("train.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_training_step_function = spx.jit(
                speculative_decoding_step,
                in_shardings=in_shardings,
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
            )
        else:
            sharded_training_step_function = compile_trainer_step(
                speculative_decoding_step,
                in_shardings=in_shardings,
                out_shardings=(self.state_shardings, empty_sharding),
                donate_argnums=(0,),
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
                arguments=self.arguments,
            )
        self._runtime_trace("train.compile_wrapper.end")

        self._runtime_trace("eval.compile_wrapper.begin")
        if self.arguments.mpmd_scheduler is None:
            sharded_evaluation_step_function = spx.jit(
                speculative_decoding_step,
                in_shardings=in_shardings,
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
            )
        else:
            sharded_evaluation_step_function = compile_trainer_step(
                speculative_decoding_step,
                in_shardings=in_shardings,
                out_shardings=empty_sharding,
                static_argnums=static_argnums,
                mesh=self.model.mesh,
                schedule=self.arguments.mpmd_scheduler,
                arguments=self.arguments,
            )
        self._runtime_trace("eval.compile_wrapper.end")

        sharded_training_step_function.static_argnums_ = static_argnums
        sharded_evaluation_step_function.static_argnums_ = static_argnums

        try:
            self._extra_forward_flops_per_token = self.target_state.model.flops_per_token(
                include_loss=False,
                include_backward=False,
            )
        except Exception:
            self._extra_forward_flops_per_token = 0.0
        self._extra_backward_flops_per_token = 0.0

        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def _get_preprocess_transform(self) -> SFTPreprocessTransform | None:
        """Build the raw-text tokenizer transform when the dataset needs one.

        Pretokenized sources flow through unchanged.  Raw text sources reuse
        the SFT transform so assistant/completion masking, packing, and
        truncation behave the same way as standard language-model training.

        Returns:
            SFTPreprocessTransform | None: An SFT preprocessing transform for
            raw text datasets, or ``None`` when the source is already
            tokenized.
        """
        if self._is_pretokenized():
            return None
        text_field = getattr(self.arguments, "dataset_text_field", None) or "text"
        mask_prompt = bool(getattr(self.arguments, "assistant_only_loss", False))
        completion_only_loss = getattr(self.arguments, "completion_only_loss", None)
        if completion_only_loss is not None:
            mask_prompt = bool(completion_only_loss)
        return SFTPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
            text_field=text_field,
            mask_prompt=mask_prompt,
            padding=False
            if getattr(self.arguments, "sequence_packing", False) or self._user_data_collator
            else "max_length",
            truncation_mode=self.arguments.truncation_mode,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the first train shard already yields ``input_ids``.

        Returns:
            bool: ``True`` if the training source exists, does not require
            row-level preprocessing, and the first shard sample contains an
            ``input_ids`` key. ``False`` otherwise.
        """
        if self._train_source is None:
            return False
        if self._source_requires_row_preprocessing(self._train_source):
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    @staticmethod
    def _source_requires_row_preprocessing(source: tp.Any) -> bool:
        """Return whether a wrapped source may mix raw and tokenized rows.

        Mixed and shuffled sources need row-level preprocessing because a
        single shard sample is not enough to prove the whole source schema.

        Args:
            source (tp.Any): A dataset source object, potentially wrapped by
                mixers or shufflers.

        Returns:
            bool: ``True`` if the source (or any nested wrapper) is a
            ``MixedShardedSource`` or ``ShuffledShardedSource``, indicating
            row-level preprocessing is required.
        """
        current = source
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if type(current).__name__ in {"MixedShardedSource", "ShuffledShardedSource"}:
                return True
            current = getattr(current, "_source", None)
        return False

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Normalize masks so KL and CE are applied to the same positions.

        The base trainer handles tokenization, packing, and collator output.
        This hook folds ``assistant_masks`` into ``completion_mask`` and, when
        labels are missing, derives ``-100`` labels from the completion and
        attention masks.  That keeps the speculative KL denominator aligned
        with the optional supervised CE denominator.

        Args:
            state (EasyDeLState): Current drafter state (unused directly but
                required by the base trainer signature).
            batch (dict[str, tp.Any]): Dictionary of tensors or arrays produced
                by the data loader or collator. May contain
                ``assistant_masks``, ``attention_mask``, ``completion_mask``,
                ``input_ids``, and ``labels``.
            is_train (bool): Whether this is a training batch (unused directly
                but forwarded to the base class).

        Returns:
            tuple[dict[str, tp.Any], dict[str, float | int | str]]: A tuple
            ``(batch, infos)`` where ``batch`` is the updated dictionary with
            normalized ``completion_mask`` and ``labels``, and ``infos`` is a
            metadata dictionary from the base class.
        """
        batch, infos = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)

        if "assistant_masks" in batch:
            if "completion_mask" not in batch:
                batch["completion_mask"] = batch["assistant_masks"]
            batch.pop("assistant_masks", None)

        attention_mask = batch.get("attention_mask")
        completion_mask = batch.get("completion_mask")

        if completion_mask is not None:
            completion_mask_np = np.asarray(completion_mask)
            if attention_mask is not None:
                completion_mask_np = completion_mask_np * np.asarray(attention_mask)
            completion_dtype = (
                np.asarray(attention_mask).dtype if attention_mask is not None else completion_mask_np.dtype
            )
            batch["completion_mask"] = completion_mask_np.astype(completion_dtype, copy=False)

            if "labels" not in batch and "input_ids" in batch:
                labels = np.asarray(batch["input_ids"]).astype(np.int32, copy=True)
                labels[completion_mask_np == 0] = -100
                if attention_mask is not None:
                    labels[np.asarray(attention_mask) == 0] = -100
                batch["labels"] = labels

        if "labels" in batch and "completion_mask" not in batch:
            labels_np = np.asarray(batch["labels"])
            if (labels_np == -100).any():
                completion_mask_np = (labels_np != -100).astype(np.int32)
                if attention_mask is not None:
                    completion_mask_np = completion_mask_np * np.asarray(attention_mask)
                batch["completion_mask"] = completion_mask_np

        return batch, infos

    @property
    def _train_shared_fn_extra_args(self) -> tuple[EasyDeLState]:
        """Return the frozen target state passed beside every train batch.

        Returns:
            tuple[EasyDeLState]: A one-element tuple containing
            :attr:`target_state`.
        """
        return (self.target_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[EasyDeLState]:
        """Return the frozen target state passed beside every eval batch.

        Returns:
            tuple[EasyDeLState]: A one-element tuple containing
            :attr:`target_state`.
        """
        return (self.target_state,)
