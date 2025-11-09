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
from datasets import Dataset, IterableDataset
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import ejit
from easydel.utils.registery import Registry
from easydel.utils.traversals import deepcopy_model

from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt, maybe_unpair_preference_dataset
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..utils import BCODataCollatorGrain, BCODataCollatorTFDS
from ._fn import RunningMoments, concatenated_forward, evaluation_step, training_step
from .bco_config import BCOConfig

logger = get_logger(__name__)


def _tokenize(
    batch: dict[str, list[tp.Any]],
    tokenizer,
    embedding_tokenizer=None,
) -> dict[str, list[tp.Any]]:
    """Tokenize prompts and completions for BCO training.

    Args:
        batch: Dictionary containing 'prompt' and 'completion' lists.
        tokenizer: Primary tokenizer for model inputs.
        embedding_tokenizer: Optional tokenizer for UDM embeddings.

    Returns:
        Dictionary with tokenized prompt, answer, and optional embedding fields.
    """
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]

    prompt_and_completion = [
        prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True)
    ]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    answer_input_ids = [f[len(p) :] for f, p in zip(full_input_ids, prompt_input_ids, strict=True)]
    answer_attention_mask = [f[len(p) :] for f, p in zip(full_attention_mask, prompt_attention_mask, strict=True)]

    full_input_ids = [np.asarray(f) for f in full_input_ids]

    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]
    for idx, (prompt_ids, full_ids, start_idx) in enumerate(
        zip(prompt_input_ids, full_input_ids, response_token_ids_start_idx, strict=True)
    ):
        if not np.array_equal(prompt_ids, full_ids[:start_idx]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r] for f, r in zip(full_input_ids, response_token_ids_start_idx, strict=True)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]
    answer_input_ids = [f[r:] for f, r in zip(full_input_ids, response_token_ids_start_idx, strict=True)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx, strict=True)]

    output = dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
    )

    if embedding_tokenizer is not None:
        embedding_tokenized = embedding_tokenizer(batch["prompt"], truncation=True, add_special_tokens=False)
        output.update(
            {
                "embedding_input_ids": embedding_tokenized["input_ids"],
                "embedding_attention_mask": embedding_tokenized["attention_mask"],
            }
        )

    return output


def _process_tokens(
    example: dict[str, tp.Any],
    *,
    prefix: str,
    tokenizer,
    is_encoder_decoder: bool,
    max_length: int,
    truncation_mode: tp.Literal["keep_end", "keep_start"],
    label_pad_token_id: int,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    model: tp.Any = None,
) -> dict[str, tp.Any]:
    """Process and truncate tokenized sequences for BCO training.

    Args:
        example: Single example with prompt, completion, and label.
        prefix: Prefix for output keys.
        tokenizer: Tokenizer for special tokens.
        is_encoder_decoder: Whether model uses encoder-decoder architecture.
        max_length: Maximum total sequence length.
        truncation_mode: How to truncate sequences.
        label_pad_token_id: Token ID for padding labels.
        max_prompt_length: Maximum prompt length.
        max_completion_length: Maximum completion length.
        model: Optional model for encoder-decoder input preparation.

    Returns:
        Dictionary with processed sequences and labels.
    """
    prompt = example["prompt"]
    completion = example["completion"]
    batch = {f"{prefix}label": example["label"]}

    if not is_encoder_decoder:
        prompt_tokens = {
            "prompt_input_ids": example["prompt_input_ids"],
            "prompt_attention_mask": example["prompt_attention_mask"],
            "answer_input_ids": example["answer_input_ids"],
            "answer_attention_mask": example["answer_attention_mask"],
        }

        full_len = len(prompt_tokens["prompt_input_ids"]) + len(prompt_tokens["answer_input_ids"])

        available_length = max_length
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id

        if bos_token_id is not None and (
            len(prompt_tokens["prompt_input_ids"]) == 0 or prompt_tokens["prompt_input_ids"][0] != bos_token_id
        ):
            available_length -= 1
        if eos_token_id is not None and (
            len(prompt_tokens["answer_input_ids"]) == 0 or prompt_tokens["answer_input_ids"][-1] != eos_token_id
        ):
            available_length -= 1

        if full_len > available_length:
            for key in ("prompt_input_ids", "prompt_attention_mask"):
                if truncation_mode == "keep_start":
                    prompt_tokens[key] = prompt_tokens[key][:max_prompt_length]
                elif truncation_mode == "keep_end":
                    prompt_tokens[key] = prompt_tokens[key][-max_prompt_length:]
                else:
                    raise ValueError(f"Unsupported truncation mode: {truncation_mode}")

        full_len = len(prompt_tokens["prompt_input_ids"]) + len(prompt_tokens["answer_input_ids"])
        if full_len > available_length:
            for key in ("answer_input_ids", "answer_attention_mask"):
                prompt_tokens[key] = prompt_tokens[key][: max(available_length - max_prompt_length, 0)]

        completion_ids = prompt_tokens["prompt_input_ids"] + prompt_tokens["answer_input_ids"]
        completion_attention = prompt_tokens["prompt_attention_mask"] + prompt_tokens["answer_attention_mask"]

        if bos_token_id is not None:
            if len(prompt_tokens["prompt_input_ids"]) == 0 or prompt_tokens["prompt_input_ids"][0] != bos_token_id:
                completion_ids = [bos_token_id, *completion_ids]
                completion_attention = [1, *completion_attention]
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]

        if eos_token_id is not None:
            if len(prompt_tokens["answer_input_ids"]) == 0 or prompt_tokens["answer_input_ids"][-1] != eos_token_id:
                completion_ids = [*completion_ids, eos_token_id]
                completion_attention = [*completion_attention, 1]

        completion_labels = completion_ids[:]
        completion_labels[: len(prompt_tokens["prompt_input_ids"])] = [label_pad_token_id] * len(
            prompt_tokens["prompt_input_ids"]
        )

        batch[f"{prefix}prompt_input_ids"] = prompt_tokens["prompt_input_ids"]
        batch[f"{prefix}prompt_attention_mask"] = prompt_tokens["prompt_attention_mask"]
        batch[f"{prefix}completion_input_ids"] = completion_ids
        batch[f"{prefix}completion_attention_mask"] = completion_attention
        batch[f"{prefix}completion_labels"] = completion_labels

    else:
        completion_tokens = tokenizer(
            completion,
            truncation=True,
            max_length=max_completion_length,
            add_special_tokens=True,
        )
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=True,
        )

        batch[f"{prefix}prompt_input_ids"] = prompt_tokens["input_ids"]
        batch[f"{prefix}prompt_attention_mask"] = prompt_tokens["attention_mask"]
        batch[f"{prefix}completion_labels"] = completion_tokens["input_ids"]
        batch[f"{prefix}completion_attention_mask"] = completion_tokens["attention_mask"]

        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                labels=np.asarray(batch[f"{prefix}completion_labels"])
            )
            batch[f"{prefix}completion_decoder_input_ids"] = np.asarray(decoder_input_ids)

    return batch


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
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
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

        train_dataset = self._prepare_dataset(train_dataset, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {key: self._prepare_dataset(ds, key) for key, ds in eval_dataset.items()}
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, "eval")

        self.reference_state = reference_state

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        self.clf_weights: tuple[np.ndarray, float] | None = None
        if self.embedding_func is not None and self.embedding_tokenizer is not None:
            self._train_density_ratio_classifier(train_dataset)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model_state,
            data_collator=None,
        )

    def _prepare_dataset(self, dataset: Dataset | IterableDataset, name: str):
        """Prepare dataset by extracting prompts, applying templates, and tokenizing.

        Args:
            dataset: Raw dataset to process.
            name: Dataset name for logging.

        Returns:
            Processed dataset with tokenized fields.
        """
        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = self.arguments.dataset_num_proc

        dataset = dataset.map(maybe_extract_prompt, desc=f"Extracting prompt for {name}", **map_kwargs)
        dataset = maybe_unpair_preference_dataset(dataset, self.arguments.dataset_num_proc, desc=f"Unpairing {name}")
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": self.processing_class},
            desc=f"Applying chat template to {name}",
            **map_kwargs,
        )

        dataset = dataset.map(
            _tokenize,
            batched=True,
            fn_kwargs={"tokenizer": self.processing_class, "embedding_tokenizer": self.embedding_tokenizer},
            desc=f"Tokenising {name}",
            **map_kwargs,
        )

        process_kwargs = {
            "prefix": "",
            "tokenizer": self.processing_class,
            "is_encoder_decoder": self.is_encoder_decoder,
            "max_length": self.max_length,
            "truncation_mode": self.truncation_mode,
            "label_pad_token_id": self.label_pad_token_id,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
        }

        dataset = dataset.map(
            _process_tokens,
            fn_kwargs=process_kwargs,
            desc=f"Processing {name}",
            **map_kwargs,
        )
        return dataset

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

    def _get_prompt_embeddings(self, dataset: Dataset, sample_size: int) -> np.ndarray:
        """Extract and pool prompt embeddings from dataset samples.

        Args:
            dataset: Dataset to sample from.
            sample_size: Number of samples to use.

        Returns:
            Stacked prompt embeddings.
        """
        n_samples = min(len(dataset), sample_size)
        indices = self._rng.choice(len(dataset), size=n_samples, replace=False)
        sampled = dataset.select(indices)
        embeddings = []
        for record in sampled:
            emb = self._vectorize_prompt(
                np.asarray(record["embedding_input_ids"], dtype=np.int32),
                np.asarray(record["embedding_attention_mask"], dtype=np.int32),
            )
            embeddings.append(emb.mean(axis=0))
        return np.stack(embeddings, axis=0)

    def _train_density_ratio_classifier(self, train_dataset: Dataset):
        """Train UDM density ratio classifier using desirable/undesirable prompt embeddings.

        Args:
            train_dataset: Training dataset containing labeled examples.
        """
        desirable = train_dataset.filter(lambda x: x["label"], num_proc=self.arguments.dataset_num_proc)
        undesirable = train_dataset.filter(lambda x: not x["label"], num_proc=self.arguments.dataset_num_proc)

        if len(desirable) == 0 or len(undesirable) == 0:
            logger.warning("UDM was requested but dataset does not include both desirable and undesirable samples.")
            return

        chosen_embeddings = self._get_prompt_embeddings(desirable, self.arguments.prompt_sample_size)
        rejected_embeddings = self._get_prompt_embeddings(undesirable, self.arguments.prompt_sample_size)

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
        )

        ref_sharding = self.reference_state.shardings if self.reference_state is not None else empty_sharding

        train_static_argnums = (3, 4, 5, 6, 7, 8)
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
