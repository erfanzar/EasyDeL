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
from __future__ import annotations

import typing as tp
from collections import defaultdict
from functools import partial

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..base_trainer import TrainerConfigureFunctionOutput  # pyright: ignore[reportPrivateLocalImportUsage]
from ..prompt_transforms import ORPOPreprocessTransform
from ..trainer.trainer import Trainer
from ..training_utils import compile_trainer_auxiliary, compile_trainer_step, resolve_straight_through_emulator
from ..utils import DPODataCollatorWithPaddingGrain, DPODataCollatorWithPaddingTFDS
from ._fn import concatenated_forward, orpo_step, orpo_training_step
from .orpo_config import ORPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "orpo")
class ORPOTrainer(Trainer):
    """Odds Ratio Preference Optimization trainer.

    ORPO is a reference-free preference optimization method that directly
    optimizes the odds ratio between preferred and rejected responses.
    Unlike DPO, ORPO doesn't require a reference model, making it more
    memory-efficient while maintaining competitive performance.

    The trainer uses lazy preprocessing transforms that are applied during
    iteration, providing better performance than eager HF .map() calls.

    Attributes:
        arguments: ORPOConfig with training hyperparameters
        processing_class: Tokenizer or processor for text encoding
        padding_value: Token ID used for padding

    Example:
        >>> config = ORPOConfig(
        ...     per_device_train_batch_size=4,
        ...     orpo_beta=0.1,
        ...     learning_rate=5e-6,
        ...     max_prompt_length=512,
        ...     max_completion_length=512
        ... )
        >>> trainer = ORPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=preference_dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: ORPOConfig

    def __init__(
        self,
        arguments: ORPOConfig | None,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        data_collator: DPODataCollatorWithPaddingTFDS | DPODataCollatorWithPaddingGrain | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
    ):
        """Initialize an ORPO trainer.

        Args:
            arguments (ORPOConfig | None): Training configuration. Required.
            model (EasyDeLBaseModule | EasyDeLState | None): Base module or
                pre-built state to train. Plain modules are converted to a
                state via ``model.to_state(...)``.
            data_collator (DPODataCollatorWithPaddingTFDS |
                DPODataCollatorWithPaddingGrain | None): Optional explicit
                collator. When ``None`` default DPO-style padding collators
                are constructed for both Grain and TFDS pipelines.
            train_dataset (Dataset | IterableDataset | ShardedDataSource | None):
                Training dataset; preference pairs with ``chosen``/``rejected``
                fields (or pre-tokenized columns).
            eval_dataset (Dataset | IterableDataset | ShardedDataSource |
                dict[str, Dataset] | None): Optional evaluation dataset(s).
            processing_class (ProcessingClassType): Tokenizer or processor
                used to tokenize prompts/responses. Required.

        Raises:
            ValueError: If ``arguments`` is None, if ``processing_class`` is
                None, or if no ``padding_value`` can be resolved.
            TypeError: If ``arguments`` is not an :class:`ORPOConfig`.
        """
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if not isinstance(arguments, ORPOConfig):
            raise TypeError(f"arguments must be ORPOConfig, got {type(arguments)}")
        if processing_class is None:
            raise ValueError("processing_class must be specified to tokenize an ORPO dataset.")

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class
        self.is_encoder_decoder = arguments.is_encoder_decoder

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
                    "`padding_value` is not specified in `ORPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please set `tokenizer.pad_token` before instantiating the trainer."
                )
        arguments.padding_value = self.padding_value

        # Setup data collators
        self.input_data_collator_tfds = (
            DPODataCollatorWithPaddingTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                prepadded=True,
            )
            if data_collator is None
            else data_collator
        )
        self.input_data_collator_grain = (
            DPODataCollatorWithPaddingGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                prepadded=True,
            )
            if data_collator is None
            else data_collator
        )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> ORPOPreprocessTransform | None:
        """Build the lazy ORPO preprocessing transform attached to a :class:`ShardedDataSource`.

        The transform tokenises a row that contains raw ``prompt`` /
        ``chosen`` / ``rejected`` fields, applies the configured chat
        template (and any tool schema in ``arguments.tools``), trims to
        ``max_prompt_length`` / ``max_completion_length``, and replaces
        prompt-side label positions with ``label_pad_token_id`` so that
        the cross-entropy term in the ORPO NLL only scores completion
        tokens. The transform is intentionally lazy: it runs at iteration
        time inside the data loader rather than eagerly via
        ``Dataset.map``.

        Returns:
            An :class:`ORPOPreprocessTransform` configured against the
            current tokenizer/processor, or ``None`` when the training
            source already exposes pre-tokenised fields (detected by
            :meth:`_is_pretokenized`) and no further work is required.
        """

        if self._is_pretokenized():
            return None

        return ORPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            tools=getattr(self.arguments, "tools", None),
            label_pad_token_id=self.arguments.label_pad_token_id,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the bound training source already carries tokenised ORPO columns.

        Opens the first shard of ``self._train_source``, peeks at one
        row, and reports whether it exposes the tokeniser-output column
        ``"prompt_input_ids"``. When present, the trainer skips the
        :class:`ORPOPreprocessTransform` and feeds the source rows
        directly to the collator. When the source is missing or empty
        the method returns ``False`` so that the preprocessing transform
        is still attached as a defensive default.

        Returns:
            ``True`` when the first sample of the first shard contains a
            ``"prompt_input_ids"`` field; ``False`` if the source is
            ``None``, the shard list is empty, the shard yields no
            rows, or the field is absent.
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "prompt_input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled ORPO training/eval step functions and supporting auxiliaries.

        The method wires three compiled artefacts that the base
        :class:`Trainer` loop later calls:

        * **``concatenated_forward``** -- a stateless utility that
          concatenates the chosen and rejected completions, runs a
          single forward pass, and returns per-sequence log-probabilities
          (and, when in chunked mode, vocabulary-tiled logsumexp). It is
          captured as ``self.concatenated_forward`` for use by metric
          probes outside the JITted step.
        * **Sharded training step** -- :func:`orpo_training_step`
          partial-applied with ``concatenated_forward``, the ORPO
          ``beta``, the scheduler, ``loss_config``, partition spec,
          gradient-accumulation count, and the resolved straight-through
          quantisation emulator. Static argnums freeze those Python-side
          values into the compiled cache so the JAX cache hits across
          steps.
        * **Sharded evaluation step** -- :func:`orpo_step` compiled with
          the same statics plus the literal mode tag ``"eval"`` so
          dropout/labels/logging are routed through the eval branch.

        Sharding is built from the trainer's mesh: state is donated and
        round-trips at ``self.state_shardings`` while the batch and
        scalar metrics ride on a fully replicated sharding. Compilation
        honours ``arguments.mpmd_scheduler`` so MPMD-pipelined runs
        forward to the scheduled-loss adapter automatically.

        Side effects:
            Caches ``self._train_shared_fn_static_args``,
            ``self._eval_shared_fn_static_args``,
            ``self.concatenated_forward``,
            ``self._extra_forward_flops_per_token``, and
            ``self._extra_backward_flops_per_token`` for downstream FLOP
            accounting; ensures the checkpoint directory exists.

        Returns:
            A :class:`TrainerConfigureFunctionOutput` carrying the two
            compiled step functions, the active mesh, and the streaming
            checkpoint manager.
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
            padding_value=self.arguments.padding_value,
            max_length=self.arguments.max_length,
            logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
        )
        jited_concatenated_forward = compile_trainer_auxiliary(
            partial_concatenated_forward,
            mesh=mesh,
            static_argnames=("is_encoder_decoder", "label_pad_token_id", "padding_value", "max_length"),
        )

        self._train_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.scheduler,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        train_static_argnums = (2, 3, 4, 5, 6, 7, 8)

        sharded_training_step_function = compile_trainer_step(
            orpo_training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=train_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.scheduler,
            "eval",
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            None,
        )

        eval_static_argnums = (2, 3, 4, 5, 6, 7, 8, 9)
        sharded_evaluation_step_function = compile_trainer_step(
            orpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=eval_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        self._extra_forward_flops_per_token = 0
        self._extra_backward_flops_per_token = 0

        self.arguments.ensure_checkpoint_path()
        self.concatenated_forward = jited_concatenated_forward
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

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
        """Return the Grain collator that batches ORPO preference pairs.

        The base :class:`Trainer` calls this hook when assembling a
        Grain-driven loader. ORPO ignores ``max_sequence_length`` and
        ``truncation_mode`` because the per-pair lengths
        (``max_prompt_length`` / ``max_completion_length``) and
        truncation policy were already resolved when
        :class:`DPODataCollatorWithPaddingGrain` was constructed in
        ``__init__``. The returned callable left-pads the chosen and
        rejected branches symmetrically (``prepadded=True``) so that
        downstream concatenation in :func:`concatenated_forward` lines
        up token positions across the two branches.

        Args:
            max_sequence_length: Accepted for interface compatibility
                with :class:`Trainer`; unused.
            truncation_mode: Accepted for interface compatibility with
                :class:`Trainer`; unused.

        Returns:
            The pre-instantiated Grain DPO/ORPO padding collator.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the TFDS collator that batches ORPO preference pairs.

        Mirrors :meth:`create_grain_collect_function` for the TFDS
        loader path. The collator was constructed in ``__init__`` with
        the ORPO-specific prompt/completion budgets and the resolved
        ``padding_value`` / ``label_pad_token_id``; the arguments to
        this method are kept only to satisfy the
        :class:`Trainer`-imposed signature.

        Args:
            max_sequence_length: Accepted for interface compatibility
                with :class:`Trainer`; unused.
            truncation_mode: Accepted for interface compatibility with
                :class:`Trainer`; unused.

        Returns:
            The pre-instantiated TFDS DPO/ORPO padding collator.
        """
        return self.input_data_collator_tfds
