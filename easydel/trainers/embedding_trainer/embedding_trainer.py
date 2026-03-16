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

"""EmbeddingTrainer for contrastive embedding model training.

This module implements the EmbeddingTrainer, which trains dense text
embedding models using contrastive learning objectives. It supports:

- **InfoNCE**: In-batch negatives with temperature-scaled cross-entropy.
- **MNRL**: Multiple Negatives Ranking Loss (sentence-transformers standard).
- **Triplet**: Margin-based triplet loss with explicit negatives.
- **Matryoshka**: Multi-dimensional loss for variable-size embeddings.

The trainer expects datasets with ``query`` and ``positive`` columns (and
optionally ``negative``). It tokenizes both sides, forward-passes them
through the embedding model separately, and computes contrastive loss
over the similarity matrix.
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import get_logger

from ..prompt_transforms import EmbeddingPreprocessTransform
from ..trainer import Trainer
from ..trainer._fn import evaluation_step
from ..trainer.trainer import TrainerConfigureFunctionOutput
from ..utils import EmbeddingDataCollatorGrain, EmbeddingDataCollatorTFDS
from ._fn import embedding_training_step
from .embedding_config import EmbeddingConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "embedding")
class EmbeddingTrainer(Trainer):
    """Contrastive embedding trainer for dense text representations.

    Trains embedding models (e.g. ``Qwen2ForEmbedding``) using contrastive
    objectives where the model learns to produce similar embeddings for
    semantically related text pairs and dissimilar embeddings for unrelated pairs.

    The trainer handles:
    - Tokenization of query/positive/negative text columns
    - Batched forward passes through the embedding model
    - Contrastive loss computation (InfoNCE, MNRL, or triplet)
    - Optional Matryoshka multi-dim loss
    - Standard training loop with gradient accumulation, checkpointing, etc.

    Args:
        arguments: EmbeddingConfig with training hyperparameters.
        model: Embedding model (e.g. ``Qwen2ForEmbedding``) or EasyDeLState.
        train_dataset: Dataset with ``query`` and ``positive`` columns.
        eval_dataset: Optional evaluation dataset.
        processing_class: Tokenizer for text tokenization.
        data_collator: Optional custom data collator.

    Example:
        >>> config = EmbeddingConfig(
        ...     loss_type="infonce",
        ...     temperature=0.05,
        ...     total_batch_size=128,
        ...     learning_rate=2e-5,
        ...     max_length=512,
        ... )
        >>> trainer = EmbeddingTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    arguments: EmbeddingConfig

    def __init__(
        self,
        arguments: EmbeddingConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict | None = None,
        processing_class: ProcessingClassType | None = None,
        data_collator: tp.Callable | None = None,
    ):
        if not isinstance(arguments, EmbeddingConfig):
            raise TypeError(f"arguments must be EmbeddingConfig, got {type(arguments)}")

        self._embedding_config = arguments

        pad_token_id = getattr(processing_class, "pad_token_id", None) if processing_class else None
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)

        if isinstance(model, EasyDeLState):
            super().__init__(
                model_state=model,
                arguments=arguments,
                dataset_train=train_dataset,
                dataset_eval=eval_dataset,
                processing_class=processing_class,
                data_collator=data_collator,
            )
        else:
            super().__init__(
                model=model,
                arguments=arguments,
                dataset_train=train_dataset,
                dataset_eval=eval_dataset,
                processing_class=processing_class,
                data_collator=data_collator,
            )

    def _get_preprocess_transform(self) -> EmbeddingPreprocessTransform | None:
        """Tokenize query/positive/negative text columns.

        Returns a transform that converts raw text columns into tokenized
        input_ids and attention_mask tensors with the ``query_``/``positive_``/
        ``negative_`` prefixes.
        """
        if self._is_pretokenized():
            return None

        return EmbeddingPreprocessTransform(
            tokenizer=self.processing_class,
            max_length=self.arguments.max_length or 512,
            query_field=self.arguments.query_field,
            positive_field=self.arguments.positive_field,
            negative_field=self.arguments.negative_field,
        )

    def _is_pretokenized(self) -> bool:
        """Check if the dataset is already tokenized."""
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "query_input_ids" in sample
        except Exception:
            return False

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training/eval step functions.

        Overrides the default ``training_step`` with
        ``embedding_training_step`` which handles the contrastive loss.
        """
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=self.model.mesh)

        _step_fn = partial(
            embedding_training_step,
            loss_type=self._embedding_config.loss_type,
            temperature=self._embedding_config.temperature,
            margin=self._embedding_config.margin,
            normalize=self._embedding_config.normalize_embeddings,
            matryoshka_dims=self._embedding_config.matryoshka_dims,
            learning_rate_fn=self.scheduler,
            partition_spec=self.arguments.step_partition_spec,
            gradient_accumulation_steps=self.arguments.gradient_accumulation_steps,
        )

        self._train_shared_fn_extra_args = ()
        self._train_shared_fn_static_args = ()

        sharded_training_step_function = ejit(
            _step_fn,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
        )
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            static_argnums=(2, 3),
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
        )

        mesh = self.model.mesh
        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a data collator for embedding training."""
        return EmbeddingDataCollatorTFDS(
            pad_token_id=self.padding_value,
            max_length=self.arguments.max_length or 512,
            has_negatives=self.arguments.negative_field is not None,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create a Grain data collator for embedding training."""
        return EmbeddingDataCollatorGrain(
            pad_token_id=self.padding_value,
            max_length=self.arguments.max_length or 512,
            has_negatives=self.arguments.negative_field is not None,
        )
