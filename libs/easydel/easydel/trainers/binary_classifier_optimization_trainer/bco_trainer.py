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
"""Binary Classifier Optimization (BCO) trainer.

BCO -- introduced as an extension of KTO -- aligns language models from
unpaired desirable / undesirable completions by minimising a logistic
loss against a reference policy.  It optionally uses User-Driven Modeling
(UDM) embeddings to estimate density ratios and reweight examples,
helping when the desirable and undesirable distributions are severely
imbalanced.
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils.registery import Registry
from easydel.utils.traversals import deepcopy_model

from ..model_loading import disable_state_dropout, reject_string_model_id
from ..prompt_transforms import BCOPreprocessTransform
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_auxiliary, compile_trainer_step, resolve_straight_through_emulator
from ..utils import BCODataCollatorGrain, BCODataCollatorTFDS
from ._fn import RunningMoments, concatenated_forward, evaluation_step, training_step
from .bco_config import BCOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "bco")
class BCOTrainer(Trainer):
    """Trainer for Binary Classifier Optimization (BCO).

    Implements the unpaired BCO objective from Jung et al. 2024 (a
    cousin of KTO): each row carries a single binary ``label``
    (desirable vs. undesirable) and the loss treats the implicit
    reward ``r = beta * (logp - logp_ref)`` as the score of an
    implicit binary classifier whose decision boundary is the running
    in-batch reward mean ``delta``. The trainer maintains ``delta``
    across steps via :class:`RunningMoments` and (optionally) trains a
    UDM (Unbiased Data Marginalization) density-ratio classifier on
    prompt embeddings to correct for distribution mismatch between
    the desirable and undesirable streams.

    See :func:`compute_bco_loss` for the loss form and
    :func:`training_step` for the per-step pipeline.

    Attributes:
        arguments: :class:`BCOConfig` controlling losses, lengths, UDM
            knobs, and the inherited ``TrainingArguments`` surface.
        processing_class: Tokenizer/processor used for prompt/completion
            encoding.
        beta: Cached copy of ``arguments.beta`` used by the loss
            closure.
        running: Running mean / variance tracker for the BCO
            ``delta`` threshold.
        embedding_func: Optional embedding callable used by UDM to
            estimate density ratios on prompts.
        embedding_tokenizer: Optional separate tokenizer for the UDM
            embedding feed.
        reference_state: Frozen reference :class:`EasyDeLState`; falls
            back to a deep copy of the policy when none is provided.
    """

    supports_sequence_packing: tp.ClassVar[bool] = False  # RL/online or paired-preference: warn-and-ignore packing

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
        """Initialize the BCO trainer.

        Wires up the policy and reference states, copies the policy to a
        frozen reference when one is not provided, configures padding /
        encoder-decoder handling, and seeds the running-moments helper
        used for density-ratio estimation.

        Args:
            arguments: BCO-specific training configuration.
            model: Policy model module or state.
            reference_model: Optional reference model; defaults to a
                deep copy of ``model`` when omitted.
            processing_class: Tokenizer/processor used for prompt and
                completion encoding.
            train_dataset: Training dataset with ``prompt``,
                ``completion`` and ``label`` fields.
            eval_dataset: Optional evaluation dataset.
            data_collator: Optional custom collator; otherwise the
                default :class:`BCODataCollatorTFDS` /
                :class:`BCODataCollatorGrain` is used.
            embedding_func: Optional callable that maps prompts to
                fixed-length embeddings used by the UDM density
                estimator.
            embedding_tokenizer: Optional separate tokenizer for the
                UDM embedding feed.

        Raises:
            TypeError: If ``arguments`` is not a :class:`BCOConfig`.
            ValueError: If ``processing_class`` or ``train_dataset`` is
                missing.
        """
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

        model = self._resolve_policy_model(model)
        reference_model = self._resolve_reference_model(reference_model)

        if isinstance(model, EasyDeLState):
            model_state = model
        else:
            model_state = model.to_state(trainable_selector=arguments.trainable_selector)

        if reference_model is None:
            reference_state = deepcopy_model(model_state)
        elif isinstance(reference_model, EasyDeLState):
            reference_state = reference_model
        else:
            reference_state = reference_model.to_state(trainable_selector=arguments.trainable_selector)

        if arguments.disable_dropout:
            model_state, reference_state = self._disable_state_dropout(model_state, reference_state)

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

    @staticmethod
    def _resolve_policy_model(
        model: EasyDeLBaseModule | EasyDeLState,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Reject accidental string policy identifiers."""
        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        return model

    @staticmethod
    def _resolve_reference_model(
        reference_model: EasyDeLBaseModule | EasyDeLState | None,
    ) -> EasyDeLBaseModule | EasyDeLState | None:
        """Reject accidental string references."""
        if isinstance(reference_model, str):
            reject_string_model_id(reference_model, role="reference model")
        return reference_model

    @staticmethod
    def _disable_state_dropout(*states: EasyDeLState | None) -> tuple[EasyDeLState | None, ...]:
        """Put BCO policy/reference state modules into eval mode when requested."""
        return tuple(disable_state_dropout(state) for state in states)

    def _get_preprocess_transform(self) -> BCOPreprocessTransform | None:
        """Build the lazy BCO preprocessing transform for ``ShardedDataSource``.

        :class:`BCOPreprocessTransform` is an expand-style transform
        that runs the full preprocessing pipeline (extract prompts from
        chosen/rejected pairs, unpair preference data into desirable /
        undesirable rows, apply the chat template, and tokenise) during
        iteration rather than ahead of time.

        Returns:
            The configured :class:`BCOPreprocessTransform`, or ``None``
            if the dataset is already tokenised (carries
            ``prompt_input_ids``).
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
        """Return ``True`` when the train dataset already carries token ids.

        Peeks at the first sample of the first shard and checks for the
        presence of ``prompt_input_ids``. Used to short-circuit the
        preprocessing transform when callers feed in an already-
        tokenised dataset.

        Returns:
            ``True`` if the dataset is pre-tokenised, ``False`` if not
            (including when ``_train_source`` is missing or the first
            shard is empty).
        """
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "prompt_input_ids" in sample
        except (StopIteration, IndexError):
            return False

    @staticmethod
    def _source_has_reference_logps(source: "ShardedDataSource | None") -> bool:
        """Return whether a BCO source already carries reference logps."""
        if source is None:
            return False
        try:
            sample = next(iter(source.open_shard(source.shard_names[0])))
        except (StopIteration, IndexError):
            return False
        return "reference_logps" in sample

    def _build_source_from_dataset(
        self,
        dataset: "Dataset | IterableDataset | ShardedDataSource | None",
    ) -> "ShardedDataSource | None":
        """Wrap a dataset in a sharded source, applying BCO tokenization if needed."""
        source = self._to_sharded_source(dataset)
        if source is None or self._source_has_reference_logps(source) or self._source_is_pretokenized(source):
            return source
        return source.transform(self._get_preprocess_transform())

    def _vectorize_prompt(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Embed prompt tokens via the UDM embedding function.

        Rewrites the policy's ``pad_token_id`` to the embedding
        tokenizer's pad id (so cross-tokenizer setups still produce
        well-formed inputs), invokes ``self.embedding_func``, and
        unwraps tuple outputs by taking the first element (matching the
        HF convention for ``(last_hidden_state, pooler_output, ...)``).

        Args:
            input_ids: ``[batch, seq]`` token id array under the policy
                tokenizer.
            attention_mask: Matching ``[batch, seq]`` attention mask.

        Returns:
            ``np.ndarray`` of prompt embeddings. Returns an empty array
            when ``embedding_func`` or ``embedding_tokenizer`` is
            unset.
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
        """Fit the UDM density-ratio classifier on prompt embeddings.

        Streams up to ``prompt_sample_size`` desirable and ``prompt_sample_size``
        undesirable embeddings from the transformed train source,
        pools them along the token axis, and fits a class-balanced
        logistic regression via :meth:`_fit_logistic_regression`. The
        resulting ``(weights, bias)`` pair is cached on
        ``self.clf_weights`` and consumed by
        :meth:`_preprocess_batch_input` to compute per-example UDM
        density-ratio weights at train time.

        A warning is logged (and the method becomes a no-op) when:

        - ``_train_source`` is not available,
        - the source yields no ``embedding_input_ids``,
        - or one of the two label streams is empty.
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
        """Fit a class-balanced logistic regression by full-batch gradient descent.

        Runs at most ``max_iter`` gradient steps on the binary
        cross-entropy with per-class reweighting so the desirable /
        undesirable streams contribute equally regardless of frequency.
        Stops early when both the weight delta L2 and the bias delta
        fall below ``tol``.

        Args:
            embeddings: ``[n_samples, embedding_dim]`` feature matrix.
            labels: ``[n_samples]`` binary labels (1 for desirable, 0
                for undesirable).
            lr: Gradient descent learning rate.
            max_iter: Maximum number of full-batch GD iterations.
            tol: Convergence tolerance on the parameter delta.

        Returns:
            Tuple ``(weights, bias)`` with the fitted classifier
            parameters (``weights`` is float32, ``bias`` is a Python
            float).
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
        """Build the JIT-compiled BCO training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, builds the
        BCO concatenated forward closure (with encoder-decoder /
        truncation / vocab-chunk-size knobs baked in) and compiles it
        via :func:`compile_trainer_auxiliary` for stand-alone reference
        scoring. Then compiles :func:`training_step` and
        :func:`evaluation_step` with the right input/output shardings,
        donate args (training only), and the active MPMD pipeline
        schedule.

        Returns:
            :class:`TrainerConfigureFunctionOutput` carrying the sharded
            training and evaluation step callables, the model mesh, and
            the streaming checkpoint manager.
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

        def forward_fn(model, batch):
            """Compute the BCO concatenated forward pass for ``batch``.

            Args:
                model: The current policy or reference model module.
                batch: The collated BCO batch.

            Returns:
                A 4-tuple ``(chosen_logps, rejected_logps, chosen_logits,
                rejected_logits)`` plus optional aux-loss components.
            """
            return concatenated_forward(
                model,
                batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=getattr(self.model_state.model, "output_router_logits", False),
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            )

        self.concatenated_forward = compile_trainer_auxiliary(forward_fn, mesh=mesh, static_argnames=())

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
        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=train_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (forward_fn, self.arguments.beta)
        eval_static_argnums = (3, 4)
        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=empty_sharding,
            static_argnums=eval_static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("eval.compile_wrapper.end")

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
        """Return the BCO Grain-compatible data collator.

        Both ``max_sequence_length`` and ``truncation_mode`` are
        accepted for API symmetry with the base trainer; the cached
        :class:`BCODataCollatorGrain` already encapsulates BCO-specific
        padding and label masking rules and is returned unchanged.

        Args:
            max_sequence_length: Maximum sequence length (unused; kept
                for symmetry).
            truncation_mode: Truncation mode (unused; kept for
                symmetry).

        Returns:
            The :class:`BCODataCollatorGrain` instance configured in
            ``__init__``.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Return the BCO TFDS-compatible data collator.

        Both arguments are accepted for API symmetry with the base
        trainer; the cached :class:`BCODataCollatorTFDS` already
        encapsulates BCO-specific padding and label masking rules.

        Args:
            max_sequence_length: Maximum sequence length (unused).
            truncation_mode: Truncation mode (unused).

        Returns:
            The :class:`BCODataCollatorTFDS` instance configured in
            ``__init__``.
        """
        return self.input_data_collator_tfds

    def compute_reference_log_probs(self, batch: dict[str, np.ndarray]) -> jax.Array:
        """Score the reference model on ``batch`` and return completion logps.

        Uses ``self.reference_state.model`` when available, falling back
        to the policy itself when no reference state is configured.
        Runs through the compiled :attr:`concatenated_forward` so the
        same tokenization and chunking knobs apply.

        Args:
            batch: BCO batch with the ``prompt_*``/``completion_*``
                fields expected by :func:`concatenated_forward`.

        Returns:
            ``[batch]`` array of summed completion log probabilities
            under the reference model.
        """
        if self.reference_state is None:
            reference_model = self.model_state.model
        else:
            reference_model = self.reference_state.model
        forward_fn = getattr(self, "concatenated_forward", None)
        if forward_fn is None:
            outputs = concatenated_forward(
                reference_model,
                batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=getattr(self.model_state.model, "output_router_logits", False),
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            )
        else:
            outputs = forward_fn(reference_model, batch)
        return outputs["completion_logps"]

    def configure_dataloaders(self):
        """Build dataloaders, optionally materializing BCO reference logps."""
        if self.dataset_train is not None:
            if self._source_has_reference_logps(self._train_source):
                self._precomputed_train_ref_log_probs = True
            if self.arguments.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
                self._precomputed_train_ref_log_probs = self._precompute_reference_log_probs_for_split(
                    dataset_attr="dataset_train",
                    source_attr="_train_source",
                    batch_size=self.training_batch_size,
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
                    batch_size=self.evaluation_batch_size,
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
        """Precompute BCO reference logps for materializable one-row-per-example splits."""
        dataset = getattr(self, dataset_attr)
        source = getattr(self, source_attr)
        if dataset is None or source is None:
            return False
        if not hasattr(dataset, "add_column"):
            logger.warning(
                "`precompute_ref_log_probs=True` requires a dataset that supports `add_column`; "
                "falling back to on-the-fly BCO reference scoring for `%s`.",
                dataset_attr,
            )
            return False

        reference_logps: list[np.ndarray] = []
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
            reference_logps.append(np.asarray(self.compute_reference_log_probs(padded_batch)))

        if not reference_logps:
            return False

        reference_values = np.concatenate(reference_logps, axis=0)
        dataset_len = len(dataset) if hasattr(dataset, "__len__") else None
        if dataset_len is not None and int(dataset_len) != int(reference_values.shape[0]):
            logger.warning(
                "`precompute_ref_log_probs=True` cannot materialize BCO reference logps for `%s` because "
                "the preprocessing transform changes the row count (%s dataset rows -> %s scored rows). "
                "Falling back to on-the-fly reference scoring.",
                dataset_attr,
                dataset_len,
                reference_values.shape[0],
            )
            return False

        updated_dataset = dataset.add_column(
            name="reference_logps",
            column=reference_values,
        )
        setattr(self, dataset_attr, updated_dataset)
        setattr(self, source_attr, self._build_source_from_dataset(updated_dataset))
        return True

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, np.ndarray],
        is_train: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, tp.Any]]:
        """Inject the running BCO ``delta`` and optional UDM weights into the batch.

        Pipeline:

        1. Purify the (possibly list-of-dict) batch into a flat
           ``dict[str, array]``.
        2. Stash ``self.running.mean`` as ``running_mean`` (the BCO
           ``delta`` threshold) so the JIT-compiled step can consume it
           as a scalar tensor.
        3. When the UDM classifier has been fitted, embed each prompt,
           compute the logistic density ratio
           ``p / (1 - p) clipped to [min_density_ratio,
           max_density_ratio]``, and broadcast it into ``udm_weights``
           for the undesirable rows (desirable rows are forced to
           weight 1.0).

        Args:
            state: Current model state (passed through; not mutated
                here).
            batch: Input batch produced by the collator.
            is_train: Whether this is a training step. Accepted for
                interface compatibility; BCO behaves the same way for
                training and evaluation.

        Returns:
            Tuple ``(processed_batch, informations)`` where
            ``processed_batch`` carries the extra ``running_mean`` and
            optional ``udm_weights`` fields, and ``informations`` is a
            dict of scalar diagnostics (currently the mean UDM ratio
            when UDM is active).
        """
        batch = self._apply_user_data_collator(batch)
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
