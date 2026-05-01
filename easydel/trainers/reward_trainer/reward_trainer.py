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

import jax
from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput
from easydel.utils import Registry

from ..prompt_transforms import RewardPreprocessTransform
from ..trainer import Trainer
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ..utils import RewardDataCollatorWithPaddingGrain, RewardDataCollatorWithPaddingTFDS
from ._fn import evaluation_step, training_step
from .reward_config import RewardConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "reward")
class RewardTrainer(Trainer):
    """Reward model trainer for RLHF pipelines.

    Trains reward models that learn to score responses based on human preferences.
    The reward model is typically used in the RLHF pipeline to provide feedback
    signals for policy optimization methods like PPO or DPO.

    The trainer uses lazy preprocessing transforms that are applied during
    iteration, providing better performance than eager HF .map() calls.

    Attributes:
        arguments: RewardConfig with training hyperparameters
        processing_class: Tokenizer or processor for text encoding

    Example:
        >>> config = RewardConfig(
        ...     per_device_train_batch_size=8,
        ...     learning_rate=2e-5,
        ...     max_length=512
        ... )
        >>> trainer = RewardTrainer(
        ...     arguments=config,
        ...     model=reward_model,
        ...     train_dataset=preference_dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()

    Note:
        The dataset should contain 'chosen' and 'rejected' columns with
        text examples representing preferred and non-preferred responses.
    """

    def __init__(
        self,
        arguments: RewardConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        data_collator: RewardDataCollatorWithPaddingTFDS | RewardDataCollatorWithPaddingGrain | None = None,
    ):
        """Initialize the reward-model trainer.

        Args:
            arguments (RewardConfig): Training configuration; must be a
                :class:`RewardConfig`.
            processing_class (ProcessingClassType): Tokenizer / processor.
                Required.
            model (EasyDeLBaseModule | EasyDeLState | None): The reward
                model (typically with a scalar regression head). Plain
                modules are converted to a state via ``model.to_state(...)``.
            train_dataset (Dataset | IterableDataset | ShardedDataSource | None):
                Preference training dataset.
            eval_dataset (Dataset | IterableDataset | ShardedDataSource |
                dict[str, Dataset] | None): Optional evaluation dataset(s).
            data_collator (RewardDataCollatorWithPaddingTFDS |
                RewardDataCollatorWithPaddingGrain | None): Optional explicit
                data collator. When ``None``, default reward-model padding
                collators are constructed for both Grain and TFDS pipelines.

        Raises:
            TypeError: If ``arguments`` is not a :class:`RewardConfig`.
            ValueError: If ``processing_class`` is None.
        """
        if not isinstance(arguments, RewardConfig):
            raise TypeError("passed argument must be a `RewardConfig`.")
        if processing_class is None:
            raise ValueError("processing_class must be specified.")

        if getattr(processing_class, "pad_token", None) is None:
            processing_class.pad_token = processing_class.eos_token

        self.arguments = arguments

        # Setup data collators
        if data_collator is None:
            self.input_data_collator_tfds = RewardDataCollatorWithPaddingTFDS(
                processing_class,
                max_length=arguments.max_length,
                truncation_mode=arguments.truncation_mode,
            )
            self.input_data_collator_grain = RewardDataCollatorWithPaddingGrain(
                processing_class,
                max_length=arguments.max_length,
                truncation_mode=arguments.truncation_mode,
            )
        else:
            self.input_data_collator_tfds = data_collator
            self.input_data_collator_grain = data_collator

        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> RewardPreprocessTransform | None:
        """Build the lazy preference-pair preprocessing transform.

        Reward training expects each row to expose a ``chosen`` and a
        ``rejected`` text. The transform tokenises both branches
        independently, truncates to ``arguments.max_length``, and emits
        the matched columns ``input_ids_chosen`` /
        ``attention_mask_chosen`` / ``input_ids_rejected`` /
        ``attention_mask_rejected`` that the reward collator then pads to
        a common length. The transform runs lazily inside the data
        loader.

        Returns:
            A :class:`RewardPreprocessTransform`, or ``None`` if
            :meth:`_is_pretokenized` indicates the source already
            provides ``input_ids_chosen`` and the transform would be a
            no-op.
        """

        if self._is_pretokenized():
            return None

        return RewardPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the bound source already exposes tokenised reward columns.

        Peeks at the first row of the first shard and reports whether
        the column ``"input_ids_chosen"`` is present. The presence of
        that field is the signal the trainer uses to skip
        :class:`RewardPreprocessTransform` and route the source rows
        directly to the reward collator. Defensive against unset
        sources, empty shard lists and shards that yield no rows.

        Returns:
            ``True`` when the first sample of the first shard contains
            ``"input_ids_chosen"``; ``False`` otherwise.
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids_chosen" in sample
        except (StopIteration, IndexError):
            return False

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile the reward-model training and evaluation steps.

        The reward objective is the Bradley-Terry pairwise loss

        ``L = -log sigmoid(r_chosen - r_rejected)
              + center_coef * (r_chosen + r_rejected)^2``

        where the optional centring term anchors the reward to mean zero
        when ``arguments.center_rewards_coefficient`` is non-zero. The
        method:

        * Resolves the optional straight-through emulator from the
          configured quantisation knobs so the reward backbone can be
          traced under simulated quantisation while keeping the
          gradient path differentiable.
        * Caches the static-argument tuples for both phases under
          ``self._train_shared_fn_static_args`` and
          ``self._eval_shared_fn_static_args``; eval drops the
          scheduler / accumulation / STE entries because eval is a
          single-pass forward.
        * JITs :func:`training_step` with state donation and a
          ``(state, metrics)`` output and :func:`evaluation_step` with a
          metrics-only output. Both share the state's resident
          shardings; the batch and metrics ride on a fully replicated
          sharding.

        Side effects:
            Stores the compiled functions' ``static_argnums_`` for
            downstream introspection and ensures the checkpoint
            directory exists.

        Returns:
            A :class:`TrainerConfigureFunctionOutput` carrying the two
            compiled step functions, the active mesh, and the streaming
            checkpoint manager.
        """
        empty_sharding = jax.sharding.NamedSharding(
            spec=PartitionSpec(),
            mesh=self.model.mesh,
        )
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
            self.arguments.center_rewards_coefficient,
            straight_through_emulator,
        )

        sharded_training_static_argnums = (2, 3, 4, 5, 6, 7)
        sharded_training_step_function = compile_trainer_step(
            training_step,
            static_argnums=sharded_training_static_argnums,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.center_rewards_coefficient,
        )

        sharded_evaluation_static_argnums = (2, 3, 4)
        sharded_evaluation_step_function = compile_trainer_step(
            evaluation_step,
            static_argnums=sharded_evaluation_static_argnums,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        sharded_training_step_function.static_argnums_ = sharded_training_static_argnums
        sharded_evaluation_step_function.static_argnums_ = sharded_evaluation_static_argnums

        mesh = self.model.mesh
        self.arguments.ensure_checkpoint_path()
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
        """Return the cached Grain collator that pads chosen/rejected pairs.

        The pre-instantiated collator already knows the configured
        ``max_length`` and ``truncation_mode``; this method simply
        returns it so the base :class:`Trainer` can hand it to the
        loader. Both branches of each preference pair are padded to the
        same length so the reward backbone can run them under a single
        compiled forward.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; unused.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; unused.

        Returns:
            The :class:`RewardDataCollatorWithPaddingGrain` instance
            built in ``__init__``.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the cached TFDS collator that pads chosen/rejected pairs.

        TFDS analogue of :meth:`create_grain_collect_function`. The
        collator was pre-built in ``__init__`` with the reward training
        ``max_length`` and ``truncation_mode``; this method simply
        returns it.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; unused.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; unused.

        Returns:
            The :class:`RewardDataCollatorWithPaddingTFDS` instance
            built in ``__init__``.
        """
        return self.input_data_collator_tfds
