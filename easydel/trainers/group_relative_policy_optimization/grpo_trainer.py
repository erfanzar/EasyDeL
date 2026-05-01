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
"""Group Relative Policy Optimization (GRPO) trainer.

GRPO -- DeepSeek (2024) -- replaces the PPO critic with a group-
relative advantage: rewards are mean-centred (and optionally
standardised) within each prompt group of online rollouts.  This
trainer drives the online generation pipeline, calls the registered
reward functions / models, computes reference log-probabilities for
the KL penalty, and dispatches to the JIT-compiled GRPO step.
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import numpy as np
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import NamedSharding
from spectrax import with_sharding_constraint
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import (  # pyright: ignore[reportPrivateLocalImportUsage]
    capture_time,
    get_logger,
)
from easydel.utils.traversals import deepcopy_model

from ..prompt_transforms import GRPOPreprocessTransform
from ..prompt_utils import apply_chat_template
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ..training_utils import (
    compile_trainer_step,
    extract_generation_model_kwargs,
    filter_kwargs_for_callable,
    normalize_generation_model_kwargs,
    repeat_prompt_aligned_model_kwargs,
    resolve_straight_through_emulator,
    sanitize_model_call_kwargs,
    slice_prompt_aligned_model_kwargs,
    strip_prompt_only_scoring_model_kwargs,
    validate_prompt_aligned_generation_model_kwargs,
)
from ._fn import get_per_token_logps, grpo_step
from .grpo_config import GRPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)
RewardFunc = EasyDeLBaseModule | EasyDeLState | tp.Callable[[list, list], list[float]]


def _fileaf(x):
    """``is_leaf`` predicate that stops at JAX arrays.

    Args:
        x: Pytree leaf candidate.

    Returns:
        ``True`` when ``x`` is a JAX array.
    """
    return isinstance(x, jax.Array)


def delete_tree(pytree):
    """Eagerly free every JAX-array leaf inside ``pytree``.

    Used after a generation pass to release on-device buffers before
    the next compiled step allocates new ones.

    Args:
        pytree: A pytree mixing JAX arrays with Python objects.

    Returns:
        A pytree of ``None`` leaves, returned for caller symmetry; the
        side effect is the in-place buffer deletion.
    """
    return jax.tree_util.tree_map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None,
        pytree,
        is_leaf=_fileaf,
    )


@Registry.register("trainer", "grpo")
class GRPOTrainer(Trainer):
    """Group Relative Policy Optimization trainer for RLHF.

    GRPO is a reinforcement learning method that optimizes policies by comparing
    responses within groups, providing more stable training than standard PPO.
    It uses relative scoring within batches to reduce variance and improve
    convergence in preference-based learning tasks.

    Key features:
    - Group-based advantage normalization
    - Stable policy updates with KL regularization
    - Support for multiple reward models
    - Efficient generation and scoring pipeline

    Attributes:
        arguments: GRPOConfig instance with training hyperparameters
        ref_state: Reference model state for KL divergence computation
        processing_class: Tokenizer or processor for text encoding
        reward_processing_classes: Optional separate processors for reward models
        generation_config: Configuration for response generation
        data_tokenize_fn: Function to tokenize dataset samples

    Example:
        >>> config = GRPOConfig(
        ...     per_device_train_batch_size=4,
        ...     grpo_n_samples=4,
        ...     grpo_beta=0.1,
        ...     learning_rate=1e-6
        ... )
        >>> trainer = GRPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: GRPOConfig  # type hinting
    reward_processing_classes: list | None

    def __init__(
        self,
        arguments: GRPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        """Initialize the GRPO trainer.

        Resolves the policy state, deep-copies it into a frozen
        reference state, sets up reward modules / callables (lifting
        :class:`EasyDeLBaseModule` rewards into compiled
        :class:`EasyDeLState` apply functions when needed), and forwards
        construction to :class:`Trainer`.

        Args:
            arguments: GRPO-specific training configuration.
            model: Policy module or state.
            reward_funcs: Single reward callable / module / state, or a
                list of them.  Reward modules are converted to
                :class:`EasyDeLState` automatically.
            train_dataset: Optional dataset of prompts.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer/processor; defaults to
                ``AutoTokenizer.from_pretrained(model.config._name_or_path)``.
            reward_processing_classes: Per-reward tokenizer overrides;
                falls back to ``AutoTokenizer`` based on each reward
                model's ``config._name_or_path``.
            data_tokenize_fn: Optional custom tokenization callable.

        Raises:
            ValueError: If ``arguments`` is ``None`` or the reward-
                weight count does not match the number of rewards.
            TypeError: If ``arguments`` is not a :class:`GRPOConfig`.
        """
        if arguments is None:
            raise ValueError(
                "You Have to pass `arguments` that will be used for training, but you have passed `arguments=None`"
            )
        if not isinstance(arguments, GRPOConfig):
            raise TypeError(f"arguments type must be `GRPOConfig` but got {type(arguments)}")
        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.loss_type = arguments.loss_type.lower() if isinstance(arguments.loss_type, str) else arguments.loss_type
        self.epsilon = arguments.epsilon
        self.epsilon_high = arguments.epsilon_high
        self.delta = arguments.delta
        self.importance_sampling_level = arguments.importance_sampling_level
        if isinstance(self.importance_sampling_level, str):
            self.importance_sampling_level = self.importance_sampling_level.lower()
        self.scale_rewards = arguments.scale_rewards
        if isinstance(self.scale_rewards, str):
            self.scale_rewards = self.scale_rewards.lower()
        self.top_entropy_quantile = arguments.top_entropy_quantile
        self.ref_logps_chunk_size = arguments.ref_logps_chunk_size

        if not isinstance(model, EasyDeLState):
            model = model.to_state(trainable_selector=arguments.trainable_selector)

        self.ref_state = deepcopy_model(model=model)

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.model.config._name_or_path,
                padding_side="left",
            )
        self.processing_class = processing_class
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.processing_class, "tokenizer"):
            pad_token_id = getattr(self.processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        empty_sharding = replicated_named_sharding(model.model.mesh)
        if not isinstance(reward_processing_classes, list):
            raise TypeError(f"reward_processing_classes must be a list, got {type(reward_processing_classes)}")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=False)
        ):
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state(trainable_selector=arguments.trainable_selector)
                    sharding = reward_func.shardings

                    def apply_fn(gd, gs, gt, batch):
                        """Sharded reward-model forward used as the state's ``apply_fn``.

                        Args:
                            gd: Reward-module graphdef.
                            gs: Trainable graphstate.
                            gt: Frozen graphother (stop-gradient applied).
                            batch: Tokenized input batch.

                        Returns:
                            The reward module's output (typically a
                            ``logits`` field with the per-example score).
                        """
                        gt = jax.tree_util.tree_map(
                            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                            gt,
                        )
                        module = spx.bind(gd, gs.merge(gt, copy=False))
                        batch = with_sharding_constraint(
                            arr=batch,
                            sharding=self.arguments.step_partition_spec,
                            mesh=module.mesh,
                            ignore_mpmd=True,
                        )
                        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), batch)
                        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
                        return module(**call_kwargs)

                    apply_fn = compile_trainer_step(
                        apply_fn,
                        mesh=model.model.mesh,
                        static_argnums=(0,),
                        in_shardings=(sharding.graphstate, sharding.graphother, empty_sharding),
                        out_shardings=empty_sharding,
                    )
                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_model_name = reward_func.model.config._name_or_path
                    try:
                        reward_processing_class = AutoTokenizer.from_pretrained(reward_model_name)
                    except ValueError as exc:
                        if "tiktoken" in str(exc).lower():
                            reward_processing_class = AutoTokenizer.from_pretrained(reward_model_name, use_fast=False)
                        else:
                            raise
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
                reward_funcs[i] = reward_func

        if arguments.reward_weights is not None and len(arguments.reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Expected {len(reward_funcs)} reward weights, but got {len(arguments.reward_weights)} instead."
            )

        self.reward_weights = jnp.asarray(
            arguments.reward_weights if arguments.reward_weights is not None else [1.0] * len(reward_funcs),
            dtype="f4",
        )
        self.reward_func_names = [getattr(func, "__name__", None) or func.__class__.__name__ for func in reward_funcs]

        self.num_generations = arguments.num_generations
        self.reward_processing_classes = reward_processing_classes
        self.reward_funcs = reward_funcs
        self.arguments = arguments
        self._initialize_conversational_flags(train_dataset, eval_dataset)

        self.data_tokenize_fn = data_tokenize_fn

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Get GRPO preprocessing transform for ShardedDataSource."""

        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            tools=getattr(self.arguments, "tools", None),
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has tokenized fields."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for Grain data loading."""
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for TFDS data loading."""
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    @property
    def step_sharding(self):
        """Return the :class:`NamedSharding` used for per-step batch tensors."""
        return NamedSharding(
            mesh=self.model.mesh,
            spec=self.arguments.step_partition_spec,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled GRPO training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, wires up
        the captured static-arg tuples for both training and
        evaluation passes (``num_generations``, ``beta``, loss type,
        clip bounds, importance-sampling level, chunk sizes, ...),
        compiles :func:`grpo_step` once for each mode under the
        active MPMD pipeline schedule, and registers the sharded
        reference-model forward callable used inside
        :meth:`_preprocess_batch_input`.

        Returns:
            ``TrainerConfigureFunctionOutput`` with the sharded
            training / evaluation step callables, the model mesh, and
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

        self._train_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
            self.loss_type,
            self.epsilon,
            self.epsilon_high,
            self.delta,
            self.importance_sampling_level,
            self.top_entropy_quantile,
            self.arguments.completion_chunk_size,
            self.arguments.max_loss_completion_tokens,
            self.arguments.logprob_vocab_chunk_size,
            straight_through_emulator,
        )

        static_argnames = tuple(range(2, 19))

        sharded_training_step_function = compile_trainer_step(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        self._eval_shared_fn_static_args = (
            self.num_generations,
            self.arguments.beta,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
            self.loss_type,
            self.epsilon,
            self.epsilon_high,
            self.delta,
            self.importance_sampling_level,
            self.top_entropy_quantile,
            self.arguments.completion_chunk_size,
            self.arguments.max_loss_completion_tokens,
            self.arguments.logprob_vocab_chunk_size,
            straight_through_emulator,
        )

        sharded_evaluation_step_function = compile_trainer_step(
            grpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, model_kwargs=None, graphdef=None):
            """Sharded reference-model per-token log-prob forward.

            Stops gradients through the reference parameters, applies
            the trainer's step partition spec, and dispatches to
            :func:`get_per_token_logps` with the configured vocab chunk
            size.

            Args:
                graphtree: Reference trainable graphstate.
                graphother: Reference frozen graphother.
                ids: Token id array ``[batch, seq_len]``.
                mask: Attention mask ``[batch, seq_len]``.
                model_kwargs: Optional dict of additional model kwargs
                    (forwarded after normalization).
                graphdef: Reference graphdef captured via partial.

            Returns:
                ``[batch, seq_len]`` reference log-probabilities.
            """
            graphother = jax.tree_util.tree_map(
                lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                graphother,
            )
            apply = spx.bind(graphdef, graphtree.merge(graphother, copy=False))
            with apply.mesh:
                ids = with_sharding_constraint(
                    ids,
                    self.arguments.step_partition_spec,
                    mesh=apply.mesh,
                    ignore_mpmd=True,
                )
                mask = with_sharding_constraint(
                    mask,
                    self.arguments.step_partition_spec,
                    mesh=apply.mesh,
                    ignore_mpmd=True,
                )
                model_kwargs = normalize_generation_model_kwargs(
                    model_kwargs,
                    model_callable=getattr(apply, "forward", apply),
                )
                return get_per_token_logps(
                    apply,
                    ids,
                    mask,
                    self.arguments.max_prompt_length,
                    model_kwargs=model_kwargs,
                    logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                )

        self.compute_refmodel_logps = compile_trainer_step(
            partial(_compute_refmodel_logps, graphdef=self.ref_state.graphdef),
            mesh=mesh,
            static_argnames=("graphdef"),
            in_shardings=(
                self.ref_state.shardings.graphstate,
                self.ref_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
                {key: None for key in normalize_generation_model_kwargs(None).keys()},
            ),
            out_shardings=empty_sharding,
        )

        sharded_training_step_function.static_argnums_ = static_argnames
        sharded_evaluation_step_function.static_argnums_ = static_argnames

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
        """Run online generation, score completions, and assemble the GRPO batch.

        For every prompt this hook:

        1. Calls :meth:`generate_unified` to draw ``num_generations``
           completions per prompt.
        2. Computes the reference-model per-token log-probabilities
           via the compiled :meth:`compute_refmodel_logps`.
        3. Calls each registered reward function and combines results
           with the configured weights.
        4. Packs prompt/completion ids, masks, advantages, and reward
           breakdown into a JAX-friendly batch.

        Args:
            state: Current policy state.
            batch: Raw batch from the dataloader (containing prompt
                tokens and any per-prompt side-channel metadata).
            is_train: Whether the call is during training or eval.

        Returns:
            A ``(batch, info)`` tuple where ``batch`` is the JAX-pure
            dict consumed by ``grpo_step`` and ``info`` is a dict of
            timing / reward metrics.
        """
        reward_batch = self._extract_reward_batch_sidechannels(batch)
        batch = self._purify_batch(batch)
        if reward_batch:
            reward_batch = {**batch, **reward_batch}
        else:
            reward_batch = batch
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]
            prompt_model_kwargs = extract_generation_model_kwargs(
                batch,
                model_callable=getattr(state.model, "forward", state.model),
            )
            scoring_prompt_model_kwargs = strip_prompt_only_scoring_model_kwargs(prompt_model_kwargs)
            validate_prompt_aligned_generation_model_kwargs(
                scoring_prompt_model_kwargs,
                prompt_batch_size=prompt_ids.shape[0],
            )

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    model_kwargs=prompt_model_kwargs,
                    state=state,
                    apply_chat_template=False,  # GRPO doesn't apply chat template to prompts
                    shard_inputs=False,  # Already sharded
                    all_gather=False,  # We'll handle gathering ourselves
                )
                sequences = results.sequences
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_prompts = results.completion_prompts

            generation_time = generation_time_fn()
            prompt_completion_ids = sequences

            completion_mask = self._make_attn_mask(completion_ids)
            if self.arguments.mask_truncated_completions:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                completion_mask = completion_mask * has_eos[:, None].astype(completion_mask.dtype)
            # Derive how many completions we have per prompt instead of trusting config-only value.
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            ridmask = prompt_mask.repeat(generation_factor, 0)
            repeated_prompt_model_kwargs = repeat_prompt_aligned_model_kwargs(
                scoring_prompt_model_kwargs,
                generation_factor,
                prompt_batch_size=prompt_mask.shape[0],
            )
            normalized_repeated_model_kwargs = normalize_generation_model_kwargs(
                repeated_prompt_model_kwargs,
                model_callable=getattr(self.ref_state.model, "forward", self.ref_state.model),
            )
            prompt_completion_mask = jnp.concatenate([ridmask, completion_mask], -1)

            with capture_time() as token_logps_time_fn:
                if self.ref_logps_chunk_size is not None and prompt_completion_ids.shape[0] > self.ref_logps_chunk_size:
                    ref_chunks: list[jax.Array] = []
                    full_batch_size = int(prompt_completion_ids.shape[0])
                    for start in range(0, full_batch_size, self.ref_logps_chunk_size):
                        end = min(start + self.ref_logps_chunk_size, full_batch_size)
                        ref_chunks.append(
                            self.compute_refmodel_logps(
                                self.ref_state.graphstate,
                                self.ref_state.graphother,
                                prompt_completion_ids[start:end],
                                prompt_completion_mask[start:end],
                                slice_prompt_aligned_model_kwargs(
                                    normalized_repeated_model_kwargs,
                                    start,
                                    end,
                                    prompt_batch_size=full_batch_size,
                                ),
                            )
                        )
                    ref_per_token_logps = jnp.concatenate(ref_chunks, axis=0)
                else:
                    ref_per_token_logps = self.compute_refmodel_logps(
                        self.ref_state.graphstate,
                        self.ref_state.graphother,
                        prompt_completion_ids,
                        prompt_completion_mask,
                        normalized_repeated_model_kwargs,
                    )
            token_logps_time = token_logps_time_fn()

            raw_completions_text = self._coerce_generation_texts(
                results.raw_text,
                fallback=results.text,
            )
            clean_completions_text = self._coerce_generation_texts(
                results.text,
                fallback=raw_completions_text,
            )
            if not raw_completions_text or not clean_completions_text:
                host_completion_ids = np.asarray(jax.device_get(completion_ids), dtype=np.int64)
                host_completion_mask = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
                if not raw_completions_text:
                    raw_completions_text = self._decode_prompt_batch(
                        self.processing_class,
                        host_completion_ids,
                        skip_special_tokens=False,
                        pad_token_id=self._pad_token_id,
                        pop_pad_tokens=True,
                        attention_mask=host_completion_mask,
                    )
                if not clean_completions_text:
                    clean_completions_text = self._decode_prompt_batch(
                        self.processing_class,
                        host_completion_ids,
                        skip_special_tokens=True,
                        pad_token_id=self._pad_token_id,
                        pop_pad_tokens=True,
                        attention_mask=host_completion_mask,
                    )

            is_conversational = self.train_is_conversational if is_train else self.eval_is_conversational

            if is_conversational:
                raw_completions = [[{"role": "assistant", "content": completion}] for completion in raw_completions_text]
                clean_completions = [
                    [{"role": "assistant", "content": completion}] for completion in clean_completions_text
                ]
            else:
                raw_completions = raw_completions_text
                clean_completions = clean_completions_text
            target_len = len(clean_completions_text) or len(raw_completions_text) or int(completion_ids.shape[0])
            reasoning_records = self._coerce_optional_generation_texts(
                results.reasoning,
                target_len=target_len,
            )
            tool_call_records = self._coerce_generation_metadata_list(
                results.tool_calls,
                target_len=target_len,
            )
            structured_clean_completions = (
                self._build_structured_assistant_messages(
                    clean_completions_text,
                    tool_calls=tool_call_records,
                )
                if is_conversational
                else clean_completions
            )

            rewards_per_func = jnp.full(
                (prompt_ids.shape[0] * generation_factor, len(self.reward_funcs)),
                jnp.nan,
                dtype="f4",
            )
            with capture_time() as rewarding_time_fn:
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes, strict=False)
                ):
                    if isinstance(reward_func, EasyDeLState):
                        if is_conversational:
                            messages = [
                                {"messages": p + c}
                                for p, c in zip(completion_prompts, structured_clean_completions, strict=False)
                            ]
                            texts = [
                                apply_chat_template(
                                    x,
                                    reward_processing_class,
                                    tools=self._reward_chat_template_tools(),
                                )["text"]
                                for x in messages
                            ]
                        else:
                            texts = [p + c for p, c in zip(completion_prompts, clean_completions, strict=False)]

                        rew = reward_func.apply_fn(
                            reward_func.graphdef,
                            reward_func.graphstate,
                            reward_func.graphother,
                            dict(
                                reward_processing_class(
                                    texts,
                                    return_tensors="np",
                                    padding="max_length",
                                    padding_side="right",
                                    add_special_tokens=False,
                                    truncation=True,
                                    return_attention_mask=True,
                                    max_length=self.arguments.max_length,
                                )
                            ),
                        ).logits[:, 0]
                    else:
                        in_prompts = completion_prompts
                        reward_call_kwargs = self._build_reward_call_kwargs(
                            reward_func,
                            prompts=in_prompts,
                            completions=clean_completions,
                            raw_completions=raw_completions,
                            completion_texts=clean_completions_text,
                            raw_text=raw_completions_text,
                            reasoning=reasoning_records,
                            tool_calls=tool_call_records,
                            max_length=self.arguments.max_length,
                            batch=reward_batch,
                        )
                        output_reward_func = reward_func(**reward_call_kwargs)
                        rew = jnp.array(
                            [val if val is not None else jnp.nan for val in output_reward_func],
                            dtype="f4",
                        )
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()
            log_completion_length = jnp.sum(completion_mask, -1)

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            rewards_per_func = self._all_gather(rewards_per_func)
            scoring_prompt_model_kwargs = jax.tree_util.tree_map(
                lambda x: self._all_gather(x) if isinstance(x, jax.Array) else x,
                scoring_prompt_model_kwargs,
            )

            with capture_time() as grouped_comp_time_fn:
                generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
                generation_factor = max(generation_factor, 1)
                rewards = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)
                mean_grouped_rewards = jnp.nanmean(rewards.reshape(-1, generation_factor), axis=-1)
                advantages = rewards - mean_grouped_rewards.repeat(generation_factor, axis=0)

                if self.scale_rewards in ("group", "none"):
                    std_rewards = jnp.nanstd(rewards.reshape(-1, generation_factor), axis=-1)
                    std_rewards = std_rewards.repeat(generation_factor, axis=0)
                elif self.scale_rewards == "batch":
                    std_rewards = jnp.nanstd(rewards)
                    std_rewards = jnp.broadcast_to(std_rewards, advantages.shape)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be 'batch', 'group', or 'none'."
                    )
                is_std_zero = jnp.isclose(std_rewards, 0.0)
                if self.scale_rewards != "none":
                    advantages = advantages / (std_rewards + 1e-4)
                advantages = jnp.nan_to_num(advantages)
            grouped_comp_time = grouped_comp_time_fn()
        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.sum(completion_mask, -1)
        metrics_dict: dict[str, float | int | str] = {
            "reward_mean": float(jnp.nanmean(rewards, -1)),
            "reward_std": float(jnp.nanmean(std_rewards)),
            "completion_length": float(jnp.mean(completion_length)),
            "grouped_comp_time": grouped_comp_time,
            "rewarding_time": rewarding_time,
            "token_logps_time": token_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
            "frac_reward_zero_std": float(jnp.mean(is_std_zero.astype(jnp.float32))),
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = float(jnp.nanmean(rewards_per_func[:, i]))
        self._log_training_generations_to_wandb(
            state=state,
            prompts=completion_prompts,
            completions=clean_completions_text,
            completion_lengths=log_completion_length,
            generation_time=generation_time,
            reasoning=reasoning_records,
            tool_calls=tool_call_records,
            source="policy",
        )

        # i don't care who you are and what you do.
        # ill find you and ill gather u...
        return (
            {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
                "num_items_in_batch": jnp.sum(completion_mask),
                **scoring_prompt_model_kwargs,
            },
            metrics_dict,
        )

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Post-step hook that optionally synchronizes the reference model.

        When ``sync_ref_model`` is enabled in the training arguments, this
        method performs an exponential moving average (EMA) update of the
        reference model parameters toward the current policy parameters
        every ``ref_model_sync_steps`` steps.

        Args:
            state: The current model state after the training step.
            metrics: Metrics collected during the training step.
            step: The current global training step number.

        Returns:
            tuple[EasyDeLState, MetricsType]: The (possibly unchanged) state
                and metrics, passed through for further processing.
        """

        if (
            self.arguments.sync_ref_model
            and self.ref_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            alpha = self.arguments.ref_model_mixup_alpha
            new_graphstate = jax.tree_util.tree_map(
                lambda new, old: alpha * new + (1 - alpha) * old,
                deepcopy_model(state.graphstate),
                self.ref_state.graphstate,
            )
            self.ref_state = self.ref_state.replace(graphstate=new_graphstate)
        return state, metrics
