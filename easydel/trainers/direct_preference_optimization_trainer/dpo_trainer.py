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
"""Direct Preference Optimization (DPO) trainer.

DPO -- introduced by Rafailov et al. (2023) -- aligns a language model
to human preferences without a reward model.  Given pairs
``(prompt, chosen, rejected)`` the trainer maximises a simple
log-likelihood-ratio margin between the policy and a frozen reference,
optionally with several loss variants (sigmoid, IPO, hinge, KTO-style,
SimPO, etc.).
"""

from __future__ import annotations

import typing as tp
from collections import defaultdict
from functools import partial

import jax
import numpy as np
from eformer.loggings import get_logger
from tqdm.autonotebook import tqdm

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.traversals import deepcopy_model

from ..base_trainer import TrainerConfigureFunctionOutput  # pyright: ignore[reportPrivateLocalImportUsage]
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import DPOPreprocessTransform
from ..trainer.trainer import Trainer
from ..training_configurations import MetricsType
from ..training_utils import compile_trainer_auxiliary, compile_trainer_step, resolve_straight_through_emulator
from ..utils import DataCollatorForPreferenceGrain, DataCollatorForPreferenceTFDS
from ._fn import concatenated_forward, evaluation_step, training_step
from .dpo_config import DPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "dpo")
class DPOTrainer(Trainer):
    """Trainer for Direct Preference Optimization (DPO).

    This trainer implements the Direct Preference Optimization algorithm for training
    language models from human preferences without requiring a separate reward model.
    DPO directly optimizes the policy to match human preferences by maximizing the
    likelihood of preferred completions relative to rejected ones.

    The trainer uses lazy preprocessing transforms that are applied during iteration,
    providing better performance than eager HF .map() calls.

    Attributes:
        arguments (DPOConfig): Configuration object containing all training parameters.
        processing_class: Tokenizer or processor for data preprocessing.
        reference_state (EasyDeLState): Reference model state for KL divergence computation.
        padding_value (int): Token ID used for padding sequences.

    Example:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     loss_type="sigmoid",
        ...     max_length=512,
        ...     learning_rate=5e-6
        ... )
        >>> trainer = DPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reference_model=reference_model,
        ...     processing_class=tokenizer,
        ...     train_dataset=preference_dataset
        ... )
        >>> trainer.train()

    Note:
        The trainer expects datasets with 'prompt', 'chosen', and 'rejected' columns.
        These will be automatically tokenized via lazy transforms during iteration.
    """

    arguments: DPOConfig

    def __init__(
        self,
        arguments: DPOConfig | None,
        model: EasyDeLBaseModule | EasyDeLState,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        data_collator: tp.Callable | None = None,
    ):
        """Initialize the DPO trainer.

        Resolves the policy and reference states (deep-copying the
        policy when no reference is provided), configures the padding
        value, builds the default preference collators, and forwards
        construction to :class:`Trainer`.

        Args:
            arguments: DPO-specific training configuration.  Required.
            model: Policy module or state.
            reference_model: Optional reference module/state; deep-
                copied from ``model`` when omitted.
            processing_class: Tokenizer/processor used to encode
                preference triples.
            train_dataset: Training dataset of preference triples.
            eval_dataset: Optional evaluation dataset.
            data_collator: Optional custom collator; otherwise the
                default :class:`DataCollatorForPreferenceTFDS`/
                :class:`DataCollatorForPreferenceGrain` is built.

        Raises:
            ValueError: If ``arguments`` or ``processing_class`` is
                missing, or no padding token can be determined.
            TypeError: If ``arguments`` is not a :class:`DPOConfig`.
        """
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if not isinstance(arguments, DPOConfig):
            raise TypeError(f"arguments must be DPOConfig, got {type(arguments)}")
        if processing_class is None:
            raise ValueError("processing_class must be specified to tokenize a DPO dataset.")

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class
        self.is_encoder_decoder = arguments.is_encoder_decoder
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._apply_pad_token_override(processing_class, arguments.pad_token)
        self.padding_free = self._resolve_padding_free(arguments)

        # Determine padding value
        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )
        arguments.padding_value = self.padding_value

        # Setup data collators
        self.input_data_collator_tfds = (
            DataCollatorForPreferenceTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                pad_to_multiple_of=arguments.pad_to_multiple_of,
            )
            if data_collator is None
            else data_collator
        )
        self.input_data_collator_grain = (
            DataCollatorForPreferenceGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                pad_to_multiple_of=arguments.pad_to_multiple_of,
            )
            if data_collator is None
            else data_collator
        )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        reference_model = self._resolve_reference_model(reference_model)
        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)
        if reference_model is None:
            reference_model = deepcopy_model(model)
        if not isinstance(reference_model, EasyDeLState):
            reference_model = reference_model.to_state(trainable_selector=arguments.trainable_selector)

        self.reference_state: EasyDeLState | None = reference_model
        if arguments.disable_dropout:
            model, reference_model = self._disable_state_dropout(model, reference_model)
            self.reference_state = reference_model

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    @staticmethod
    def _resolve_reference_model(
        reference_model: EasyDeLBaseModule | EasyDeLState | None,
    ) -> EasyDeLBaseModule | EasyDeLState | None:
        """Reject accidental string references."""
        if isinstance(reference_model, str):
            reject_string_model_id(reference_model, role="reference model")
        return reference_model

    def _effective_reference_free(self) -> bool:
        """Apply `force_use_ref_model` to the reference-free loss switch."""
        return bool(self.arguments.reference_free and not self.arguments.force_use_ref_model)

    @staticmethod
    def _apply_pad_token_override(processing_class: object, pad_token: str | None) -> None:
        """Apply a configured pad token to a tokenizer or processor wrapper."""
        if pad_token is None:
            return
        tokenizer = getattr(processing_class, "tokenizer", processing_class)
        tokenizer.pad_token = pad_token

    @staticmethod
    def _resolve_padding_free(arguments: DPOConfig) -> bool:
        """Resolve DPO padding-free mode, matching TRL's temporary fallback."""
        if not arguments.padding_free:
            return False
        logger.warning(
            "`padding_free=True` is temporarily unavailable for DPO and is being disabled. "
            "Falling back to standard padded preference batches."
        )
        arguments.padding_free = False
        return False

    @staticmethod
    def _disable_state_dropout(*states: EasyDeLState | None) -> tuple[EasyDeLState | None, ...]:
        """Put EasyDeL state modules into eval mode when requested."""
        return tuple(disable_state_dropout(state) for state in states)

    def _get_preprocess_transform(self) -> DPOPreprocessTransform | None:
        """Get DPO preprocessing transform for ShardedDataSource.

        Returns a transform that handles:
        - Prompt extraction from chosen/rejected
        - Chat template application
        - Triple tokenization (prompt, chosen, rejected)

        Returns:
            DPOPreprocessTransform or None if data is already tokenized.
        """
        if self._is_pretokenized():
            return None

        return self._build_preprocess_transform()

    def _build_preprocess_transform(self) -> DPOPreprocessTransform:
        """Construct a :class:`DPOPreprocessTransform` from the trainer's config.

        Returns:
            A transform that tokenizes ``(prompt, chosen, rejected)``
            triples using the trainer's processing class and configured
            length caps.
        """
        return DPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            tools=getattr(self.arguments, "tools", None),
            label_pad_token_id=self.arguments.label_pad_token_id,
        )

    @staticmethod
    def _source_is_pretokenized(source: "ShardedDataSource | None") -> bool:
        """Detect whether ``source`` already contains DPO-tokenized fields.

        Args:
            source: Optional :class:`ShardedDataSource`.

        Returns:
            ``True`` when the first sample exposes ``prompt_input_ids``;
            ``False`` for empty / missing sources.
        """
        if source is None:
            return False
        try:
            sample = next(iter(source.open_shard(source.shard_names[0])))
            return "prompt_input_ids" in sample
        except (StopIteration, IndexError):
            return False

    @staticmethod
    def _source_has_reference_logps(source: "ShardedDataSource | None") -> bool:
        """Detect whether ``source`` already contains precomputed reference logps.

        Args:
            source: Optional :class:`ShardedDataSource`.

        Returns:
            ``True`` when the first sample exposes either of the
            recognised reference-logp column-pair conventions
            (``ref_chosen_logps``/``ref_rejected_logps`` or the legacy
            ``reference_*_log_probs`` aliases).
        """
        if source is None:
            return False
        try:
            sample = next(iter(source.open_shard(source.shard_names[0])))
        except (StopIteration, IndexError):
            return False
        return ("ref_chosen_logps" in sample and "ref_rejected_logps" in sample) or (
            "reference_chosen_log_probs" in sample and "reference_rejected_log_probs" in sample
        )

    def _build_source_from_dataset(
        self,
        dataset: "Dataset | IterableDataset | ShardedDataSource | None",
    ) -> "ShardedDataSource | None":
        """Wrap a dataset in a sharded source, applying DPO tokenization if needed.

        Args:
            dataset: Raw dataset (may be ``None``).

        Returns:
            A :class:`ShardedDataSource` carrying either the
            already-tokenized rows or a freshly attached DPO transform.
        """
        source = self._to_sharded_source(dataset)
        if source is None or self._source_is_pretokenized(source):
            return source
        return source.transform(self._build_preprocess_transform())

    def _is_pretokenized(self) -> bool:
        """Check whether the training source already carries DPO tokenized fields.

        Returns:
            ``True`` if the training :class:`ShardedDataSource` exposes
            ``prompt_input_ids`` (and is therefore pre-tokenized);
            ``False`` if the dataset is raw text and needs a
            :class:`DPOPreprocessTransform`.
        """
        return self._source_is_pretokenized(self._train_source)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled DPO training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, partials
        the chosen/rejected concatenated forward with all
        tokenisation knobs from the config, compiles it through
        :func:`compile_trainer_auxiliary` (so it can be reused as a
        stand-alone reference scoring path), and finally compiles
        :func:`training_step` and :func:`evaluation_step` with the
        right input/output shardings, donated argnums, and the active
        MPMD pipeline schedule.

        Returns:
            ``TrainerConfigureFunctionOutput`` with the sharded
            training / evaluation step callables, the reference mesh,
            and the streaming checkpoint manager.
        """
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        partial_concatenated_forward = partial(
            concatenated_forward,
            is_encoder_decoder=self.arguments.is_encoder_decoder,
            label_pad_token_id=self.arguments.label_pad_token_id,
            padding_value=self.padding_value,
            max_length=self.arguments.max_length,
            truncation_mode=self.arguments.truncation_mode,
            aux_loss_enabled=self.arguments.aux_loss_enabled,
            loss_type=self.arguments.loss_type,
            logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            use_weighting=self.arguments.use_weighting,
            ld_alpha=self.arguments.ld_alpha,
        )

        jited_concatenated_forward = compile_trainer_auxiliary(
            partial_concatenated_forward,
            mesh=mesh,
            out_shardings=(empty_sharding,),
            static_argnames=(
                "is_encoder_decoder",
                "label_pad_token_id",
                "padding_value",
                "max_length",
                "truncation_mode",
                "aux_loss_enabled",
                "loss_type",
                "use_weighting",
            ),
        )
        effective_reference_free = self._effective_reference_free()

        self._train_shared_fn_static_args = (
            self.scheduler,
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.loss_weights,
            self.arguments.f_divergence_type,
            self.arguments.f_alpha_divergence_coef,
            self.arguments.use_weighting,
            self.arguments.discopop_tau,
            self.arguments.ld_alpha,
            self.arguments.rpo_alpha,
            effective_reference_free,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        sharded_training_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            training_step,
            in_shardings=(
                self.state_shardings,
                empty_sharding,
                self.reference_state.shardings,
            ),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=sharded_training_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.loss_weights,
            self.arguments.f_divergence_type,
            self.arguments.f_alpha_divergence_coef,
            self.arguments.use_weighting,
            self.arguments.discopop_tau,
            self.arguments.ld_alpha,
            self.arguments.rpo_alpha,
            effective_reference_free,
            self.arguments.step_partition_spec,
        )

        sharded_evaluation_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            evaluation_step,
            in_shardings=(
                self.state_shardings,
                empty_sharding,
                self.reference_state.shardings,
            ),
            out_shardings=empty_sharding,
            static_argnums=sharded_evaluation_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("eval.compile_wrapper.end")

        sharded_training_step_function.static_argnums_ = sharded_training_static_argnums
        sharded_evaluation_step_function.static_argnums_ = sharded_evaluation_static_argnums

        self.arguments.ensure_checkpoint_path()
        self.concatenated_forward = jited_concatenated_forward
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        flops_per_tkn = self.reference_state.model.flops_per_token(include_loss=True, include_backward=True)
        self._extra_forward_flops_per_token = flops_per_tkn
        self._extra_backward_flops_per_token = flops_per_tkn

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the pre-built Grain preference collator.

        Args:
            max_sequence_length: Maximum sequence length (accepted for
                interface parity with the base trainer; the collator is
                already configured at construction time).
            truncation_mode: Truncation strategy (``"keep_end"`` /
                ``"keep_start"``); also unused here because the collator
                applies the trainer's configured mode.

        Returns:
            The cached :class:`DataCollatorForPreferenceGrain` instance
            wired up in :meth:`__init__`.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the pre-built TFDS preference collator.

        Args:
            max_sequence_length: Maximum sequence length (unused;
                see :meth:`create_grain_collect_function`).
            truncation_mode: Truncation strategy (unused for the same
                reason).

        Returns:
            The cached :class:`DataCollatorForPreferenceTFDS` instance
            wired up in :meth:`__init__`.
        """
        return self.input_data_collator_tfds

    def configure_dataloaders(self):
        """Build train/eval dataloaders, optionally caching reference logps.

        When ``arguments.precompute_ref_log_probs`` is set (and the
        dataset does not already carry ``ref_chosen_logps`` /
        ``ref_rejected_logps``), iterates the dataset once through the
        reference model and stores the per-example reference logps as
        new dataset columns. This eliminates the per-step reference
        forward during training at the cost of a larger materialised
        dataset.

        Returns:
            The base trainer's dataloader objects after any reference
            precomputation step has run.
        """
        if self.dataset_train is not None:
            if self._source_has_reference_logps(self._train_source):
                self._precomputed_train_ref_log_probs = True
            if self.arguments.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
                self._precomputed_train_ref_log_probs = self._precompute_reference_log_probs_for_split(
                    dataset_attr="dataset_train",
                    source_attr="_train_source",
                    batch_size=self.arguments.precompute_ref_batch_size or self.training_batch_size,
                    is_train=True,
                    desc="Train dataset reference log probs",
                )

        if self.dataset_eval is not None:
            if self._source_has_reference_logps(self._eval_source):
                self._precomputed_eval_ref_log_probs = True
            if self.arguments.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
                self._precomputed_eval_ref_log_probs = self._precompute_reference_log_probs_for_split(
                    dataset_attr="dataset_eval",
                    source_attr="_eval_source",
                    batch_size=self.arguments.precompute_ref_batch_size or self.evaluation_batch_size,
                    is_train=False,
                    desc="Eval dataset reference log probs",
                )

        return super().configure_dataloaders()

    def _precompute_reference_log_probs_for_split(
        self,
        *,
        dataset_attr: str,
        source_attr: str,
        batch_size: int,
        is_train: bool,
        desc: str,
    ) -> bool:
        """Precompute and attach reference log-probs for one dataset split.

        Iterates the dataloader once (in eval mode), runs the reference
        model's concatenated forward to obtain
        ``(ref_chosen_logp, ref_rejected_logp)`` per example, and adds
        the resulting columns back onto the underlying HF dataset.  The
        sharded source is rebuilt so subsequent iterations consume the
        cached values.

        Args:
            dataset_attr: Trainer attribute holding the HF dataset to
                augment (e.g. ``"dataset_train"``).
            source_attr: Trainer attribute holding the corresponding
                :class:`ShardedDataSource` (e.g. ``"_train_source"``).
            batch_size: Batch size to use during precomputation.
            is_train: Whether the split is the training split.
            desc: Description shown by the tqdm progress bar.

        Returns:
            ``True`` when reference logps were successfully computed and
            attached, ``False`` when the split is missing or its
            backing dataset does not support ``add_column``.
        """
        dataset = getattr(self, dataset_attr)
        source = getattr(self, source_attr)

        if dataset is None or source is None:
            return False
        if not hasattr(dataset, "add_column"):
            logger.warning(
                "`precompute_ref_log_probs=True` requires a dataset that supports `add_column`; "
                f"falling back to on-the-fly reference scoring for `{dataset_attr}`."
            )
            return False

        ref_chosen_logps: list[np.ndarray] = []
        ref_rejected_logps: list[np.ndarray] = []
        data_collator = self.data_collator or (lambda batch: batch)
        batch_iterator = self._create_dataloader_from_source(
            source=source,
            batch_size=batch_size,
            is_train=is_train,
            shuffle=False,
            num_epochs=1,
            drop_remainder=False,
        )

        for raw_batch in tqdm(iterable=batch_iterator, desc=desc):
            padded_batch = self._purify_batch(data_collator(raw_batch))
            ref_chosen_logp, ref_rejected_logp = self.compute_reference_log_probs(padded_batch)
            ref_chosen_logps.append(np.asarray(ref_chosen_logp))
            ref_rejected_logps.append(np.asarray(ref_rejected_logp))

        if not ref_chosen_logps:
            return False

        updated_dataset = dataset.add_column(
            name="ref_chosen_logps",
            column=np.concatenate(ref_chosen_logps, axis=0),
        )
        updated_dataset = updated_dataset.add_column(
            name="ref_rejected_logps",
            column=np.concatenate(ref_rejected_logps, axis=0),
        )
        setattr(self, dataset_attr, updated_dataset)
        setattr(self, source_attr, self._build_source_from_dataset(updated_dataset))
        return True

    def compute_reference_log_probs(
        self,
        padded_batch: dict,
    ) -> tuple[tp.Any, tp.Any]:
        """Score a preference batch under the frozen reference model.

        Used by :meth:`_precompute_reference_log_probs_for_split` to
        cache reference logps and by ad-hoc evaluation. Falls back to
        the policy model when no separate reference state is wired up
        (typical for ``reference_free`` runs that still want
        diagnostics).

        Args:
            padded_batch: A DPO batch already padded by the data
                collator.

        Returns:
            ``(ref_chosen_logps, ref_rejected_logps)``: per-example
            summed log-probs of the chosen and rejected completions
            under the reference.
        """
        reference_model = self.model_state.model if self.reference_state is None else self.reference_state.model
        reference_model.eval()
        forward_fn = getattr(self, "concatenated_forward", None)
        if forward_fn is None:
            outs = concatenated_forward(
                reference_model,
                batch=padded_batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=self.arguments.aux_loss_enabled,
                loss_type=self.arguments.loss_type,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                use_weighting=self.arguments.use_weighting,
            )
        else:
            outs = forward_fn(reference_model, batch=padded_batch)
        return outs["chosen_logps"], outs["rejected_logps"]

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        """Forward the reference state alongside the shared training step."""
        return (self.reference_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        """Forward the reference state alongside the shared evaluation step."""
        return (self.reference_state,)

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Sync the reference model from the policy at the configured cadence.

        When ``arguments.sync_ref_model`` is enabled and ``step`` is a
        multiple of ``arguments.ref_model_sync_steps``, the reference
        state's ``graphstate`` is mixed toward the current policy using
        ``arguments.ref_model_mixup_alpha``. This implements the
        TR-DPO-style reference moving average used by TRL.

        Args:
            state: Current policy state after the step's optimizer update.
            metrics: Training metrics collected this step.
            step: Global training step index.

        Returns:
            ``(state, metrics)`` -- both forwarded unchanged; the
            reference state is mutated in place on ``self``.
        """
        if (
            self.arguments.sync_ref_model
            and self.reference_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            alpha = self.arguments.ref_model_mixup_alpha
            new_graphstate = jax.tree_util.tree_map(
                lambda new, old: alpha * new + (1 - alpha) * old,
                deepcopy_model(state.graphstate),
                self.reference_state.graphstate,
            )
            self.reference_state = self.reference_state.replace(graphstate=new_graphstate)
        return state, metrics
