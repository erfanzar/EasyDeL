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

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import ejit
from easydel.utils.registery import Registry
from easydel.utils.traversals import deepcopy_model

from ..prompt_transforms import BCOPreprocessTransform
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import resolve_straight_through_emulator
from ..utils import BCODataCollatorGrain, BCODataCollatorTFDS
from ._fn import RunningMoments, concatenated_forward, evaluation_step, training_step
from .bco_config import BCOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "bco")
class BCOTrainer(Trainer):
    """Binary Classifier Optimization (BCO) trainer.

    Implements BCO training which aligns language models using binary feedback.
    Supports Unbiased Data Marginalization (UDM) for handling distribution mismatch
    between desirable and undesirable examples.

    Args:
        arguments: BCO-specific training configuration.
        model: Policy model to train.
        reference_model: Reference model for computing log probability ratios.
        processing_class: Tokenizer or processor.
        train_dataset: Training dataset with prompt, completion, and label.
        eval_dataset: Optional evaluation dataset.
        data_collator: Optional custom data collator.
        embedding_func: Optional embedding function for UDM.
        embedding_tokenizer: Optional tokenizer for UDM embeddings.
    """

    arguments: BCOConfig

    def __init__(
        self,
        arguments: BCOConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        data_collator: BCODataCollatorTFDS | BCODataCollatorGrain | None = None,
        embedding_func: tp.Callable | None = None,
        embedding_tokenizer: ProcessingClassType | None = None,
    ):
        if not isinstance(arguments, BCOConfig):
            raise TypeError(f"`arguments` must be a `BCOConfig`, received {type(arguments)}")
        if processing_class is None:
            raise ValueError("`processing_class` must be provided to tokenise a BCO dataset.")
        if train_dataset is None:
            raise ValueError("`train_dataset` must be provided for BCOTrainer.")

        self.arguments = arguments
        self.processing_class = processing_class
        self.embedding_func = embedding_func
        self.embedding_tokenizer = embedding_tokenizer
        self.beta = arguments.beta
        self.running = RunningMoments()
        seed = getattr(arguments, "seed", None)
        self._rng = np.random.default_rng(seed)

        if isinstance(model, EasyDeLState):
            model_state = model
        else:
            model_state = model.to_state()

        if reference_model is None:
            reference_state = deepcopy_model(model_state)
        elif isinstance(reference_model, EasyDeLState):
            reference_state = reference_model
        else:
            reference_state = reference_model.to_state()

        if arguments.is_encoder_decoder is not None:
            self.is_encoder_decoder = arguments.is_encoder_decoder
        else:
            self.is_encoder_decoder = getattr(model_state.model.config, "is_encoder_decoder", False)
            self.arguments.is_encoder_decoder = self.is_encoder_decoder

        if getattr(processing_class, "pad_token_id", None) is None and hasattr(processing_class, "eos_token"):
            processing_class.pad_token = processing_class.eos_token

        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            pad_token_id = getattr(processing_class, "pad_token_id", None)
            if pad_token_id is None and hasattr(processing_class, "tokenizer"):
                pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "`padding_value` is not specified and tokenizer has no pad token. "
                    "Please set `processing_class.pad_token` before instantiating the trainer."
                )
            self.padding_value = pad_token_id
            self.arguments.padding_value = pad_token_id

        if data_collator is None:
            self.input_data_collator_tfds = BCODataCollatorTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            self.input_data_collator_grain = BCODataCollatorGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        else:
            self.input_data_collator_tfds = data_collator
            self.input_data_collator_grain = data_collator

        self.max_length = arguments.max_length
        self.max_prompt_length = arguments.max_prompt_length
        self.max_completion_length = arguments.max_completion_length
        self.truncation_mode = arguments.truncation_mode
        self.label_pad_token_id = arguments.label_pad_token_id

        self.reference_state = reference_state
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self.clf_weights: tuple[np.ndarray, float] | None = None

        # BCOPreprocessTransform is an ExpandTransform that handles:
        # - Extract prompts from chosen/rejected
        # - Unpair preference data (1 pair → 2 examples)
        # - Apply chat template
        # - Tokenize
        # All preprocessing is done lazily via the transform during iteration.

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model_state,
            data_collator=self.input_data_collator_tfds,
            processing_class=processing_class,
        )

        # Train UDM classifier after BaseTrainer sets up _train_source
        if self.embedding_func is not None and self.embedding_tokenizer is not None:
            self._train_density_ratio_classifier()

    def _get_preprocess_transform(self) -> BCOPreprocessTransform | None:
        """Get BCO preprocessing transform for ShardedDataSource.

        BCOPreprocessTransform is an ExpandTransform that handles the full
        preprocessing pipeline: extract prompts, unpair preference data,
        apply chat template, and tokenize.
        """
        if self._is_pretokenized():
            return None
        return BCOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            label_pad_token_id=self.arguments.label_pad_token_id,
            embedding_tokenizer=self.embedding_tokenizer,
            tools=getattr(self.arguments, "tools", None),
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has tokenized fields."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "prompt_input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def _vectorize_prompt(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Convert prompt tokens to embeddings using the embedding function.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.

        Returns:
            Prompt embeddings.
        """
        if self.embedding_func is None or self.embedding_tokenizer is None:
            return np.array([])
        input_ids = np.where(
            input_ids == self.processing_class.pad_token_id,
            getattr(self.embedding_tokenizer, "pad_token_id", self.processing_class.pad_token_id),
            input_ids,
        )
        embeddings = self.embedding_func(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        return np.asarray(embeddings)

    def _train_density_ratio_classifier(self):
        """Train UDM density ratio classifier by iterating over transformed source.

        Iterates over _train_source to collect embeddings for desirable and
        undesirable examples, then fits a logistic regression classifier.
        """
        if self._train_source is None:
            logger.warning("Cannot train UDM classifier: _train_source is not available.")
            return

        desirable_embeddings: list[np.ndarray] = []
        undesirable_embeddings: list[np.ndarray] = []
        sample_size = self.arguments.prompt_sample_size

        # Iterate through the transformed source to collect embeddings
        for shard_name in self._train_source.shard_names:
            for example in self._train_source.open_shard(shard_name):
                if "embedding_input_ids" not in example:
                    continue

                emb = self._vectorize_prompt(
                    np.asarray(example["embedding_input_ids"], dtype=np.int32),
                    np.asarray(example["embedding_attention_mask"], dtype=np.int32),
                )
                if emb.size == 0:
                    continue

                pooled = emb.mean(axis=0)
                if example.get("label", True):
                    desirable_embeddings.append(pooled)
                else:
                    undesirable_embeddings.append(pooled)

                # Early stop if we have enough samples
                if len(desirable_embeddings) >= sample_size and len(undesirable_embeddings) >= sample_size:
                    break
            else:
                continue
            break

        if not desirable_embeddings or not undesirable_embeddings:
            logger.warning("UDM was requested but dataset does not include both desirable and undesirable samples.")
            return

        # Random sample down to prompt_sample_size
        n_d = min(len(desirable_embeddings), sample_size)
        n_u = min(len(undesirable_embeddings), sample_size)

        if n_d < len(desirable_embeddings):
            indices = self._rng.choice(len(desirable_embeddings), size=n_d, replace=False)
            desirable_embeddings = [desirable_embeddings[i] for i in indices]
        if n_u < len(undesirable_embeddings):
            indices = self._rng.choice(len(undesirable_embeddings), size=n_u, replace=False)
            undesirable_embeddings = [undesirable_embeddings[i] for i in indices]

        chosen_embeddings = np.stack(desirable_embeddings, axis=0)
        rejected_embeddings = np.stack(undesirable_embeddings, axis=0)

        embeddings = np.concatenate([chosen_embeddings, rejected_embeddings], axis=0)
        labels = np.concatenate(
            [np.ones(len(chosen_embeddings), dtype=np.float32), np.zeros(len(rejected_embeddings), dtype=np.float32)],
            axis=0,
        )

        weights, bias = self._fit_logistic_regression(embeddings, labels)
        self.clf_weights = (weights, bias)
        logger.info("Trained UDM classifier for BCO density ratio estimation.")

    def _fit_logistic_regression(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-5,
    ) -> tuple[np.ndarray, float]:
        """Fit logistic regression classifier using gradient descent.

        Args:
            embeddings: Feature embeddings.
            labels: Binary labels.
            lr: Learning rate.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple of (weights, bias).
        """
        weights = np.zeros(embeddings.shape[1], dtype=np.float32)
        bias = 0.0
        n_pos = np.count_nonzero(labels)
        n_neg = labels.size - n_pos
        pos_weight = 0.5 if n_pos == 0 else 0.5 * labels.size / (2 * n_pos)
        neg_weight = 0.5 if n_neg == 0 else 0.5 * labels.size / (2 * n_neg)

        for _ in range(max_iter):
            logits = embeddings @ weights + bias
            preds = 1.0 / (1.0 + np.exp(-logits))
            weights_vec = np.where(labels == 1, pos_weight, neg_weight)
            error = (preds - labels) * weights_vec
            grad_w = embeddings.T @ error / embeddings.shape[0]
            grad_b = error.mean()
            weights_prev = weights.copy()
            bias_prev = bias
            weights -= lr * grad_w
            bias -= lr * grad_b
            if np.linalg.norm(weights - weights_prev) < tol and abs(bias - bias_prev) < tol:
                break
        return weights.astype(np.float32), float(bias)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
        """
        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_block=self.arguments.quantization_block,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        def forward_fn(model, batch):
            return concatenated_forward(
                model,
                batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=getattr(self.model_state.model, "output_router_logits", False),
            )

        self.concatenated_forward = ejit(forward_fn, static_argnames=())

        self._train_shared_fn_static_args = (
            self.scheduler,
            forward_fn,
            self.arguments.beta,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        ref_sharding = self.reference_state.shardings if self.reference_state is not None else empty_sharding

        train_static_argnums = (3, 4, 5, 6, 7, 8, 9)
        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=train_static_argnums,
        )

        self._eval_shared_fn_static_args = (forward_fn, self.arguments.beta)
        eval_static_argnums = (3, 4)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=empty_sharding,
            static_argnums=eval_static_argnums,
        )

        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function
        self._train_shared_fn_extra_args = (self.reference_state,)
        self._eval_shared_fn_extra_args = (self.reference_state,)

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
        """Create data collator for Grain data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            Grain-compatible data collator.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for TFDS data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            TFDS-compatible data collator.
        """
        return self.input_data_collator_tfds

    def compute_reference_log_probs(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Compute reference model log probabilities for a batch.

        Args:
            batch: Input batch.

        Returns:
            Completion log probabilities from reference model.
        """
        if self.reference_state is None:
            reference_model = self.model_state.model
        else:
            reference_model = self.reference_state.model
        outputs = self.concatenated_forward(reference_model, batch)
        return outputs["completion_logps"]

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, np.ndarray],
        is_train: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, tp.Any]]:
        """Preprocess batch by adding running moments and UDM weights.

        Args:
            state: Current model state.
            batch: Input batch.
            is_train: Whether this is a training step.

        Returns:
            Processed batch with additional fields and metrics dictionary.
        """
        # Purify batch first to handle list of dicts (uncollated batch)
        batch = self._purify_batch(batch)
        batch["running_mean"] = jnp.asarray(self.running.mean, dtype=jnp.float32)
        informations: dict[str, tp.Any] = {}

        if self.clf_weights is not None and "embedding_input_ids" in batch:
            weights, bias = self.clf_weights
            embeddings = self._vectorize_prompt(
                np.asarray(batch["embedding_input_ids"]),
                np.asarray(batch["embedding_attention_mask"]),
            )
            if embeddings.size > 0:
                if embeddings.ndim == 3:
                    pooled = embeddings.mean(axis=1)
                else:
                    pooled = embeddings
                logits = pooled @ weights + bias
                prob = 1.0 / (1.0 + np.exp(-logits))
                ratio = prob / (1.0 - prob + 1e-8)
                ratio = np.clip(ratio, self.arguments.min_density_ratio, self.arguments.max_density_ratio)
                weights_array = np.where(np.asarray(batch["label"], dtype=bool), 1.0, ratio)
                batch["udm_weights"] = jnp.asarray(weights_array, dtype=jnp.float32)
                informations["udm_ratio_mean"] = float(ratio.mean())

        return batch, informations
