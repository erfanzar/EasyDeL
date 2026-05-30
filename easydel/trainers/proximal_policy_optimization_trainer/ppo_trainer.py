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

"""Proximal Policy Optimization (PPO) trainer for RLHF.

This module implements the PPOTrainer class for training language models with
Proximal Policy Optimization. PPO is a policy gradient method that uses clipped
surrogate objectives to ensure stable policy updates during reinforcement learning
from human feedback (RLHF).

The trainer supports:
- Online generation of completions
- Multiple reward functions with configurable weights
- Value head for advantage estimation (GAE)
- KL penalty against a frozen reference model
- Clipped policy and value function objectives
"""

from __future__ import annotations

import typing as tp
from functools import partial

import jax
import numpy as np
import spectrax as spx
from jax import numpy as jnp
from spectrax import with_sharding_constraint
from transformers import AutoTokenizer

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLTimerError
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time, get_logger  # pyright: ignore[reportPrivateLocalImportUsage]
from easydel.utils.traversals import deepcopy_model

from .._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from ..model_loading import reject_string_model_id
from ..prompt_transforms import GRPOPreprocessTransform
from ..prompt_utils import apply_chat_template
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import (
    compile_trainer_step,
    filter_kwargs_for_callable,
    resolve_straight_through_emulator,
    sanitize_model_call_kwargs,
)
from ._fn import ppo_step
from .modeling_value_head import CausalLMWithValueHead
from .ppo_config import PPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)
RewardFunc = EasyDeLBaseModule | EasyDeLState | tp.Callable[[list, list], list[float]]


@Registry.register("trainer", "ppo")
class PPOTrainer(Trainer):
    """Proximal Policy Optimization trainer for RLHF.

    PPO is a policy gradient method that uses clipped surrogate objectives to
    ensure stable policy updates. This trainer implements online PPO where
    completions are generated on-the-fly and scored by reward functions.

    Key features:
    - Online generation with configurable sampling parameters
    - Value head for advantage estimation via GAE
    - KL penalty against a frozen reference model
    - Support for multiple weighted reward functions
    - Clipped policy and value function objectives

    The training loop:
    1. Sample prompts from the dataset
    2. Generate completions using the current policy
    3. Score completions with reward function(s)
    4. Compute advantages using GAE with the value head
    5. Update policy with clipped PPO objective

    Attributes:
        arguments: PPOConfig with training hyperparameters.
        ref_state: Frozen reference model for KL computation.
        reward_funcs: List of reward functions/models.
        reward_weights: Weights for combining multiple rewards.
        processing_class: Tokenizer for text encoding.
        num_generations: Number of completions per prompt.

    Example:
        >>> config = PPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_return_sequences=4,
        ...     kl_coef=0.05,
        ...     cliprange=0.2,
        ...     learning_rate=1e-6
        ... )
        >>> trainer = PPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    supports_sequence_packing: tp.ClassVar[bool] = False  # RL/online or paired-preference: warn-and-ignore packing

    arguments: PPOConfig

    def __init__(
        self,
        arguments: PPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc] | None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType | None = None,
        reward_processing_classes: ProcessingClassType | None = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        """Initialize a PPO trainer with value-head and reference-model wiring.

        Wraps the policy in :class:`CausalLMWithValueHead` when no value
        head is attached, deep-copies the wrapped state to form the
        frozen reference policy ``self.ref_state``, normalises the
        reward-function list (including JITting reward-model forwards),
        and finally chains into the base :class:`Trainer`.

        Args:
            arguments (PPOConfig): Training hyperparameters.
            model (EasyDeLBaseModule | EasyDeLState | None): Policy model
                to train. Wrapped with a value head when needed and
                converted to an :class:`EasyDeLState`.
            reward_funcs (RewardFunc | list[RewardFunc]): Reward
                source(s) for scoring completions. Each entry is either
                a callable accepting ``(prompts, completions, ...)`` and
                returning a list of floats, or an
                :class:`EasyDeLBaseModule` / :class:`EasyDeLState`
                reward model whose last logit acts as the scalar score.
            train_dataset (Dataset | IterableDataset | ShardedDataSource | None):
                Prompt-only training dataset.
            eval_dataset (Dataset | IterableDataset | ShardedDataSource |
                dict[str, Dataset] | None): Optional evaluation
                dataset(s).
            processing_class (ProcessingClassType | None): Tokenizer
                used both for prompt encoding and rollout decoding.
                Defaults to an :class:`AutoTokenizer` derived from
                ``model.config._name_or_path`` when ``None``.
            reward_processing_classes (ProcessingClassType | None):
                Optional per-reward-function tokenizers. When ``None``,
                each reward model defaults to its own
                :class:`AutoTokenizer`.
            data_tokenize_fn (tp.Callable | None): Optional custom
                tokenisation function used by reward-side data
                pipelines.

        Raises:
            ValueError: If ``arguments`` is ``None``, ``model`` is
                ``None``, or :attr:`PPOConfig.reward_weights` length
                does not match the number of reward functions.
            TypeError: If ``arguments`` is not a :class:`PPOConfig`, or
                if ``reward_processing_classes`` cannot be normalised to
                a list.
        """
        if arguments is None:
            raise ValueError("PPOTrainer requires `arguments`.")
        if not isinstance(arguments, PPOConfig):
            raise TypeError(f"arguments type must be `PPOConfig` but got {type(arguments)}")
        self.arguments = arguments

        model = self._resolve_policy_model(model, arguments)

        # Ensure we have a value head attached.
        if isinstance(model, EasyDeLState):
            module = model.model
            if not hasattr(module, "value_head"):
                model = CausalLMWithValueHead(module, rngs=spx.Rngs(0)).to_state(
                    trainable_selector=arguments.trainable_selector
                )
        else:
            if not hasattr(model, "value_head"):
                model = CausalLMWithValueHead(model, rngs=spx.Rngs(0))
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

        if reward_funcs is None:
            raise ValueError("`reward_funcs` must be provided for PPO training.")
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
            reward_func = self._resolve_reward_func_model(reward_func)
            if isinstance(reward_func, EasyDeLBaseModule | EasyDeLState):
                if isinstance(reward_func, EasyDeLBaseModule):
                    reward_func = reward_func.to_state(trainable_selector=arguments.trainable_selector)
                    sharding = reward_func.shardings

                    def apply_fn(gs, gt, batch, graphdef):
                        """Sharded reward-model forward used to score completions.

                        Args:
                            gs: Reward model graph state (parameters).
                            gt: Reward model auxiliary state.
                            batch: Tokenized batch passed to the reward model.
                            graphdef: Reward model graph definition.

                        Returns:
                            The reward model's full output (typically with a
                            ``logits`` field used as the scalar score).
                        """
                        gt = jax.tree_util.tree_map(
                            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                            gt,
                        )
                        module = spx.bind(graphdef, gs.merge(gt, copy=False))
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
                        partial(apply_fn, graphdef=reward_func.graphdef),
                        mesh=model.model.mesh,
                        static_argnames=("graphdef",),
                        in_shardings=(sharding.graphstate, sharding.graphother, empty_sharding),
                        out_shardings=empty_sharding,
                    )
                    reward_func = reward_func.replace(apply_fn=apply_fn)

                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.model.config._name_or_path)
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

    @staticmethod
    def _resolve_policy_model(
        model: EasyDeLBaseModule | EasyDeLState | None,
        arguments: PPOConfig,
    ) -> EasyDeLBaseModule | EasyDeLState:
        """Resolve PPO policy models from initialized EasyDeL objects."""
        if model is None:
            raise ValueError("`model` must be provided for PPO training.")
        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        return model

    @staticmethod
    def _resolve_reward_func_model(reward_func: RewardFunc) -> RewardFunc:
        """Reject string reward identifiers and prepare initialized reward modules."""
        if isinstance(reward_func, str):
            reject_string_model_id(reward_func, role="reward model")
        return reward_func

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Build the lazy prompt-only preprocessing transform for the rollout source.

        PPO drives rollouts from prompt-only data: the dataset is
        expected to provide a chat-templatable conversation (or a raw
        prompt string when ``skip_apply_chat_template`` is true), and
        completions are sampled inside :meth:`_preprocess_batch_input`.
        This hook returns the GRPO-style transform that applies the
        chat template (with any tool schemas in ``arguments.tools``)
        and tokenises the prompt to ``arguments.max_prompt_length``.
        The transform is lazy so it runs inside the data loader rather
        than via an eager ``Dataset.map``.

        Returns:
            A :class:`GRPOPreprocessTransform` configured against the
            current tokenizer, prompt budget, tool schema, and chat-
            template flag, or ``None`` when :meth:`_is_pretokenized`
            indicates the source is already tokenised.
        """
        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            tools=getattr(self.arguments, "tools", None),
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
            chat_template_kwargs=self.arguments.chat_template_kwargs,
        )

    def _is_pretokenized(self) -> bool:
        """Detect whether the bound training source already exposes tokenised prompts.

        Peeks at the first row of the first shard of
        ``self._train_source`` and reports whether it carries an
        ``"input_ids"`` field. When present the trainer skips the
        prompt preprocessing transform and feeds rows directly to the
        prompt collator. Defensive against missing sources, empty shard
        lists and shards that yield no rows.

        Returns:
            ``True`` when the first sample of the first shard contains
            ``"input_ids"``; ``False`` if the source is unset, empty,
            or the field is absent.
        """
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
        """Construct the Grain collator that left-pads prompt-only PPO batches.

        PPO operates on prompt-only batches; rollouts (and therefore
        completions) are produced inside :meth:`_preprocess_batch_input`.
        The collator pads tokenised prompts to
        ``arguments.max_prompt_length`` using ``self.padding_value`` so
        generation always sees a contiguous, right-aligned prefix.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; ignored -- the prompt budget is taken
                from :class:`PPOConfig`.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; the GRPO collator left-pads instead of
                truncating.

        Returns:
            A freshly built :class:`GRPODataCollatorGrain`.
        """
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
        """Construct the TFDS collator that left-pads prompt-only PPO batches.

        TFDS analogue of :meth:`create_grain_collect_function`. Returns
        the :class:`GRPODataCollatorTFDS` configured with PPO's prompt
        budget and pad token; completions are produced on-the-fly inside
        :meth:`_preprocess_batch_input`.

        Args:
            max_sequence_length: Accepted for compatibility with
                :class:`Trainer`; ignored.
            truncation_mode: Accepted for compatibility with
                :class:`Trainer`; ignored.

        Returns:
            A freshly built :class:`GRPODataCollatorTFDS`.
        """
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build and JIT-compile the PPO step plus rollout-time helpers.

        Wires four compiled artefacts that the training loop consumes:

        * **Sharded training step** -- :func:`ppo_step` partial-applied
          with ``prompt_length``, the clipped-loss coefficients
          (``cliprange``, ``vf_coef``, ``cliprange_value``,
          ``entropy_coef``), the optional ``logprob_vocab_chunk_size``,
          ``loss_config``, scheduler, partition spec, gradient
          accumulation count, the ``is_train=True`` flag, and the
          resolved straight-through emulator. The state is donated.
        * **Sharded eval step** -- :func:`ppo_step` compiled with the
          same statics but ``is_train=False`` and no state donation.
        * **``compute_refmodel_logps``** -- compiled helper used by
          :meth:`_preprocess_batch_input` to score the rollout under the
          frozen reference policy and return per-token completion log
          probabilities. The reference graphdef is curried in as a
          static argument so the same compile cache is reused.
        * **``compute_rollout_logps_values``** -- compiled helper that
          scores a rollout under the *current* policy and returns both
          per-token log probabilities and the value-head predictions
          used as ``V_old`` for GAE.

        Both helpers detect the headless-LM-head path via
        :func:`resolve_lmhead_chunksize` and project through the LM head
        in token/vocab chunks when ``logprob_vocab_chunk_size`` is set,
        avoiding materialising ``[batch, seq, vocab]`` logit tensors on
        very large vocabularies.

        Returns:
            TrainerConfigureFunctionOutput: Container with
            ``sharded_training_step_function``,
            ``sharded_evaluation_step_function``, the active ``mesh``,
            and the streaming ``checkpoint_manager``.
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

        prompt_length = int(self.arguments.max_prompt_length)

        self._train_shared_fn_static_args = (
            prompt_length,
            float(self.arguments.cliprange),
            float(self.arguments.vf_coef),
            float(self.arguments.cliprange_value),
            0.0 if self.arguments.entropy_coef is None else float(self.arguments.entropy_coef),
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
            straight_through_emulator,
        )
        static_argnums = tuple(range(2, 14))
        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            ppo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("train.compile_wrapper.end")

        self._eval_shared_fn_static_args = (
            prompt_length,
            float(self.arguments.cliprange),
            float(self.arguments.vf_coef),
            float(self.arguments.cliprange_value),
            0.0 if self.arguments.entropy_coef is None else float(self.arguments.entropy_coef),
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_train
            straight_through_emulator,
        )
        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            ppo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        self._runtime_trace("eval.compile_wrapper.end")

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            """Compute frozen reference-model per-token log probabilities.

            Args:
                graphtree: Reference model graph state (parameters).
                graphother: Reference model auxiliary state (non-parameters).
                ids (jax.Array): Full ``[batch, seq]`` token ids
                    (prompt + completion).
                mask (jax.Array): Attention mask of shape ``[batch, seq]``.
                graphdef: Reference model graph definition for binding.

            Returns:
                jax.Array: Per-token log probabilities for the completion
                portion only, shape ``[batch, completion_len]``.

            Raises:
                ValueError: If headless mode is active but the reference
                    model does not return ``last_hidden_state``.
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
                target_ids = ids[:, prompt_length:]
                call_kwargs = {"input_ids": ids, "attention_mask": mask}
                lmhead_chunksize = resolve_lmhead_chunksize(apply)
                if lmhead_chunksize is not None:
                    call_kwargs["apply_lm_head"] = False
                outputs = apply(**call_kwargs)
                if outputs.logits is None and lmhead_chunksize is not None:
                    hidden_states = outputs.last_hidden_state
                    if hidden_states is None:
                        raise ValueError("Reference model outputs do not provide last_hidden_state for PPO scoring.")
                    score_hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
                    token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
                        apply,
                        score_hidden_states,
                        target_ids,
                        token_chunk_size=lmhead_chunksize,
                        vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                        return_entropy=False,
                    )
                    return token_log_probs
                logits = outputs.logits
                if logits is None:
                    raise ValueError("Reference model outputs do not provide logits for PPO scoring.")
                logits = logits[:, prompt_length - 1 :]
                logits = logits[:, :-1, :]
                token_log_probs, _ = compute_token_logps_and_entropies_chunked(
                    logits,
                    target_ids,
                    return_entropy=False,
                    chunk_size=self.arguments.logprob_vocab_chunk_size,
                )
                return token_log_probs

        self.compute_refmodel_logps = compile_trainer_step(
            partial(_compute_refmodel_logps, graphdef=self.ref_state.graphdef),
            mesh=mesh,
            static_argnames=("graphdef",),
            in_shardings=(
                self.ref_state.shardings.graphstate,
                self.ref_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )

        def _compute_rollout_logps_values(graphtree, graphother, ids, mask, graphdef):
            """Compute policy log-probabilities and value-head predictions for a rollout.

            Args:
                graphtree: Policy graph state (parameters).
                graphother: Policy auxiliary state (non-parameters).
                ids (jax.Array): Full ``[batch, seq]`` token ids.
                mask (jax.Array): Attention mask of shape ``[batch, seq]``.
                graphdef: Policy graph definition for binding.

            Returns:
                tuple[jax.Array, jax.Array]: ``(token_log_probs, values)`` for
                the completion portion, both shaped ``[batch, completion_len]``.

            Raises:
                ValueError: If hidden states are not available from the
                    forward pass.
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
                target_ids = ids[:, prompt_length:]
                call_kwargs = {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "output_hidden_states": True,
                }
                lmhead_chunksize = resolve_lmhead_chunksize(apply)
                if lmhead_chunksize is not None:
                    call_kwargs["apply_lm_head"] = False
                outputs = apply(**call_kwargs)

                hidden_states = getattr(outputs, "last_hidden_state", None)
                if hidden_states is None:
                    hidden_states = getattr(outputs, "hidden_states", None)
                    if hidden_states is None:
                        raise ValueError("Model outputs do not provide hidden states; cannot compute value outputs.")
                    hidden_states = hidden_states[-1]

                if outputs.logits is None and lmhead_chunksize is not None:
                    score_hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
                    token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
                        apply,
                        score_hidden_states,
                        target_ids,
                        token_chunk_size=lmhead_chunksize,
                        vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                        return_entropy=False,
                    )
                else:
                    logits = outputs.logits
                    if logits is None:
                        raise ValueError("Model outputs do not provide logits for PPO scoring.")
                    logits = logits[:, prompt_length - 1 :]
                    logits = logits[:, :-1, :]
                    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
                        logits,
                        target_ids,
                        return_entropy=False,
                        chunk_size=self.arguments.logprob_vocab_chunk_size,
                    )

                values_full = apply.value_head(hidden_states).squeeze(-1)
                values = values_full[:, prompt_length - 1 : -1]
                return token_log_probs, values

        self.compute_rollout_logps_values = compile_trainer_step(
            partial(_compute_rollout_logps_values, graphdef=self.model_state.graphdef),
            mesh=mesh,
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=(empty_sharding, empty_sharding),
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _execute_train_step(
        self,
        state: EasyDeLState,
        batch,
    ) -> tuple[EasyDeLState, LossMetrics, BaseException | None]:
        """Execute one PPO update, including multiple PPO epochs when configured."""
        if self.arguments.num_ppo_epochs == 1:
            return super()._execute_train_step(state=state, batch=batch)

        if self.pruning_module is not None:
            state = state.replace(
                graphstate=self.pruning_module.pre_forward_update(
                    state.graphstate,
                    state.opt_state,
                )
            )
        metrics = LossMetrics()
        try:
            self._runtime_trace("execute_train_step.preprocess.begin", batch=self._runtime_batch_summary(batch))
            batch, informations = self._preprocess_batch_input(
                state=state,
                batch=batch,
                is_train=True,
            )
            self._runtime_trace(
                "execute_train_step.preprocess.end",
                batch=self._runtime_batch_summary(batch),
                information_keys=tuple(informations.keys()) if isinstance(informations, dict) else None,
            )

            for ppo_epoch in range(self.arguments.num_ppo_epochs):
                self._runtime_trace("execute_train_step.compiled_call.begin", ppo_epoch=ppo_epoch)
                state, metrics = jax.block_until_ready(
                    self.sharded_training_step_function(
                        state,
                        batch,
                        *self._train_shared_fn_extra_args,
                        *self._train_shared_fn_static_args,
                    )
                )
                self._runtime_trace(
                    "execute_train_step.compiled_call.end",
                    step=int(jax.device_get(state.step)),
                    metrics_type=type(metrics).__name__,
                    ppo_epoch=ppo_epoch,
                )

            if len(informations) != 0:
                if metrics.other_metrics is not None:
                    informations.update(metrics.other_metrics)
                metrics = metrics.replace(other_metrics=informations)

            if self.pruning_module is not None:
                state = state.replace(
                    graphstate=self.pruning_module.post_gradient_update(
                        state.graphstate,
                        state.opt_state,
                    )
                )
            return state, metrics, None
        except (
            KeyboardInterrupt,
            EasyDeLTimerError,
            EasyDeLBreakRequest,
            TypeError,
        ) as run_exception:
            self._runtime_trace(
                "execute_train_step.control_exception",
                exc_type=type(run_exception).__name__,
                exc=str(run_exception),
            )
            return state, metrics, run_exception
        except Exception as run_exception:
            self._runtime_trace(
                "execute_train_step.exception",
                exc_type=type(run_exception).__name__,
                exc=str(run_exception),
            )
            if self._is_memory_oom_exception(run_exception):
                annotated_exception = self._augment_memory_oom_exception(run_exception)
                logger.error(str(annotated_exception))
                return state, metrics, annotated_exception
            raise

    def _masked_whiten(self, x: jax.Array, mask: jax.Array, *, shift_mean: bool) -> jax.Array:
        """Whiten ``x`` over the masked elements (variance scaling, optional centering).

        Computes the masked first and second moments under ``mask``,
        then either centres-and-scales (``shift_mean=True``, used for
        advantages where a zero-mean target is desired) or scales only
        (``shift_mean=False``, used for rewards when
        :class:`PPOConfig.whiten_rewards` is enabled and absolute reward
        magnitudes need to be preserved). A small constant ``1e-8`` is
        added before the square root to keep gradients well-defined when
        the masked variance collapses to zero.

        Args:
            x: Tensor of shape ``(batch, completion_len)`` to whiten.
            mask: Same-shape binary mask selecting completion positions
                that participate in the moment estimation.
            shift_mean: If ``True`` subtract the masked mean before
                scaling; if ``False`` only scale by the standard
                deviation.

        Returns:
            ``x`` whitened to unit variance over the masked positions
            (and zero mean when ``shift_mean`` is set). The masking is
            *not* re-applied to the output -- callers multiply by
            ``mask`` again to zero out padded positions.
        """
        mask = mask.astype(x.dtype)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        mean = jnp.sum(x * mask) / denom
        var = jnp.sum(jnp.square(x - mean) * mask) / denom
        std = jnp.sqrt(var + 1e-8)
        if shift_mean:
            x = x - mean
        return x / std

    def _compute_gae(
        self,
        rewards: jax.Array,
        values: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Estimate per-token advantages and returns via Generalised Advantage Estimation.

        GAE estimates the advantage at each completion position from a
        backward recursion on the temporal-difference residuals

        ``delta_t = r_t + gamma * V_{t+1} * mask_{t+1} - V_t``

        ``A_t = delta_t + gamma * lambda * A_{t+1} * mask_{t+1}``

        where ``gamma`` is :class:`PPOConfig.gamma` and ``lambda`` is
        :class:`PPOConfig.lam`. The recursion is implemented with
        :func:`jax.lax.scan` over the time axis reversed in place; the
        ``mask`` and shifted ``mask_next`` zero out contributions past
        the end of each completion. Returns are reconstructed as
        ``R_t = A_t + V_t`` so the value-function loss in
        :func:`ppo_step` regresses against a consistent target.

        Args:
            rewards: ``(batch, completion_len)`` per-token rewards
                already containing the score-on-final-token plus the
                token-wise KL penalty.
            values: ``(batch, completion_len)`` rollout-time value
                predictions ``V_t`` aligned with ``rewards``.
            mask: Same-shape binary mask selecting valid completion
                positions; positions past EOS are zero.

        Returns:
            ``(advantages, returns)`` -- both ``(batch, completion_len)``.
            ``advantages`` carries ``A_t`` (zeroed outside the mask) and
            ``returns`` carries ``A_t + V_t``.
        """
        rewards = rewards.astype(jnp.float32)
        values = values.astype(jnp.float32)
        mask = mask.astype(jnp.float32)

        batch_size, _gen_len = rewards.shape
        values_next = jnp.concatenate([values[:, 1:], jnp.zeros((batch_size, 1), dtype=values.dtype)], axis=1)
        mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros((batch_size, 1), dtype=mask.dtype)], axis=1)

        gamma = float(self.arguments.gamma)
        lam = float(self.arguments.lam)

        def scan_fn(adv_next, inputs):
            """Single backward GAE recursion step.

            Args:
                adv_next: Advantage at the next timestep ``A_{t+1}``.
                inputs: Tuple ``(r_t, v_t, v_{t+1}, m_t, m_{t+1})`` for the
                    current step.

            Returns:
                tuple: ``(A_t, A_t)`` -- carry and stacked output for ``scan``.
            """
            r_t, v_t, v_next_t, m_t, m_next_t = inputs
            delta = r_t + gamma * v_next_t * m_next_t - v_t
            adv_t = delta + gamma * lam * adv_next * m_next_t
            adv_t = adv_t * m_t
            return adv_t, adv_t

        inputs = (
            rewards[:, ::-1].T,
            values[:, ::-1].T,
            values_next[:, ::-1].T,
            mask[:, ::-1].T,
            mask_next[:, ::-1].T,
        )
        _, adv_rev = jax.lax.scan(scan_fn, jnp.zeros((batch_size,), dtype=values.dtype), inputs)
        advantages = adv_rev.T[:, ::-1]
        returns = advantages + values
        return advantages, returns

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Roll out a PPO batch (generate, score, build advantages) outside the gradient.

        Executes the full PPO rollout pipeline that the JITted
        :func:`ppo_step` consumes:

        1. Generate completions from the prompt batch via
           :meth:`generate_unified` using the *current* policy.
        2. Build the completion mask (optionally zeroing out completions
           that hit ``max_completion_length`` without an EOS when
           ``mask_truncated_completions`` is enabled).
        3. Score every position under both the frozen reference
           (:attr:`compute_refmodel_logps`) and the rollout policy
           (:attr:`compute_rollout_logps_values`) to capture
           ``ref_per_token_logps``, ``old_logps`` and ``old_values``.
        4. Run each registered reward function (callable or
           :class:`EasyDeLState`) and combine the results with
           :attr:`reward_weights` to a scalar ``score`` per completion,
           subtracting :attr:`PPOConfig.missing_eos_penalty` when no EOS
           is emitted.
        5. Build per-token rewards as
           ``r_t = -kl_coef * KL_t`` (with the chosen ``kl_estimator``)
           and add the scalar score at the last completion position.
           Optionally whiten the rewards.
        6. Compute GAE advantages and returns with :meth:`_compute_gae`
           and optionally whiten the advantages.
        7. All-gather every output tensor so the downstream JIT receives
           a replicated, well-formed batch.

        Args:
            state (EasyDeLState): Current PPO state used both as the
                rollout policy and as the source of value-head
                predictions.
            batch (dict[str, jax.Array]): Tokenised prompt-only batch
                with ``input_ids`` / ``attention_mask`` and (when
                present) reward-function side-channels.
            is_train (bool): When ``True`` use the train-side
                conversational-mode detection, otherwise the eval-side
                flag.

        Returns:
            tuple[dict[str, jax.Array], dict[str, float | int | str]]:
            Pair of ``(prepared_batch, metrics_dict)`` where
            ``prepared_batch`` holds ``input_ids``, ``attention_mask``,
            ``completion_mask``, ``old_logps``, ``old_values``,
            ``advantages`` and ``returns``; ``metrics_dict`` carries
            wall-clock stats and aggregate score / KL diagnostics, plus
            one entry per reward function.
        """
        reward_batch = self._extract_reward_batch_sidechannels(batch)
        batch = self._purify_batch(batch)
        if reward_batch:
            reward_batch = {**batch, **reward_batch}
        else:
            reward_batch = batch
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
                sequences = results.sequences
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_prompts = results.completion_prompts

            generation_time = generation_time_fn()

            completion_mask = self._make_attn_mask(completion_ids)
            if self.arguments.mask_truncated_completions:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                completion_mask = completion_mask * has_eos[:, None].astype(completion_mask.dtype)

            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            prompt_mask_rep = prompt_mask.repeat(generation_factor, 0)
            attention_mask = jnp.concatenate([prompt_mask_rep, completion_mask], axis=1)
            input_ids = sequences

            with capture_time() as ref_logps_time_fn:
                ref_per_token_logps = self.compute_refmodel_logps(
                    self.ref_state.graphstate,
                    self.ref_state.graphother,
                    input_ids,
                    attention_mask,
                )
            ref_logps_time = ref_logps_time_fn()

            with capture_time() as rollout_stats_time_fn:
                old_logps, old_values = self.compute_rollout_logps_values(
                    state.graphstate,
                    state.graphother,
                    input_ids,
                    attention_mask,
                )
            rollout_stats_time = rollout_stats_time_fn()

            raw_completions_text = self._coerce_generation_texts(
                results.raw_text,
                fallback=results.text,
            )
            completions_text = self._coerce_generation_texts(
                results.text,
                fallback=raw_completions_text,
            )
            if not raw_completions_text or not completions_text:
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
                if not completions_text:
                    completions_text = self._decode_prompt_batch(
                        self.processing_class,
                        host_completion_ids,
                        skip_special_tokens=True,
                        pad_token_id=self._pad_token_id,
                        pop_pad_tokens=True,
                        attention_mask=host_completion_mask,
                    )

            is_conv = self.train_is_conversational if is_train else self.eval_is_conversational
            if completion_prompts:
                first_prompt = completion_prompts[0]
                if not isinstance(first_prompt, list):
                    is_conv = False
            else:
                is_conv = False
            if is_conv:
                raw_completions = [[{"role": "assistant", "content": completion}] for completion in raw_completions_text]
                completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
            else:
                raw_completions = raw_completions_text
                completions = completions_text
            target_len = len(completions_text) or len(raw_completions_text) or int(completion_ids.shape[0])
            reasoning_records = self._coerce_optional_generation_texts(
                results.reasoning,
                target_len=target_len,
            )
            tool_call_records = self._coerce_generation_metadata_list(
                results.tool_calls,
                target_len=target_len,
            )
            structured_completions = (
                self._build_structured_assistant_messages(
                    completions_text,
                    tool_calls=tool_call_records,
                )
                if is_conv
                else completions
            )

            rewards_per_func = jnp.full((completion_ids.shape[0], len(self.reward_funcs)), jnp.nan, dtype="f4")
            with capture_time() as rewarding_time_fn:
                for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes, strict=False)
                ):
                    if isinstance(reward_func, EasyDeLState):
                        if is_conv:
                            messages = [
                                {"messages": p + c}
                                for p, c in zip(completion_prompts, structured_completions, strict=False)
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
                            texts = [p + c for p, c in zip(completion_prompts, completions, strict=False)]

                        rew = reward_func.apply_fn(
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
                        reward_call_kwargs = self._build_reward_call_kwargs(
                            reward_func,
                            prompts=completion_prompts,
                            completions=completions,
                            raw_completions=raw_completions,
                            completion_texts=completions_text,
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
            log_completion_length = jnp.sum(completion_mask, axis=1)

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            input_ids = self._all_gather(input_ids)
            attention_mask = self._all_gather(attention_mask)
            old_logps = self._all_gather(old_logps)
            old_values = self._all_gather(old_values)
            ref_per_token_logps = self._all_gather(ref_per_token_logps)
            rewards_per_func = self._all_gather(rewards_per_func)

            scores = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)
            if self.arguments.missing_eos_penalty is not None:
                eos_tokens = jnp.asarray(self._eos_token_id).reshape(-1)
                has_eos = jnp.any(jnp.isin(completion_ids, eos_tokens), axis=1)
                scores = scores - (~has_eos).astype(scores.dtype) * float(self.arguments.missing_eos_penalty)

            logr = ref_per_token_logps - old_logps
            if self.arguments.kl_estimator == "k1":
                kl = -logr
            else:
                kl = jnp.exp(logr) - 1.0 - logr
            non_score_reward = -float(self.arguments.kl_coef) * kl
            rewards = non_score_reward

            lengths = jnp.sum(completion_mask, axis=1).astype(jnp.int32)
            last_idx = jnp.maximum(lengths - 1, 0)
            batch_idx = jnp.arange(rewards.shape[0])
            rewards = rewards.at[batch_idx, last_idx].add(scores.astype(rewards.dtype))
            rewards = rewards * completion_mask

            if self.arguments.whiten_rewards:
                rewards = self._masked_whiten(rewards, completion_mask, shift_mean=False)
                rewards = rewards * completion_mask

            advantages, returns = self._compute_gae(rewards, old_values, completion_mask)
            if self.arguments.whiten_advantages:
                advantages = self._masked_whiten(advantages, completion_mask, shift_mean=True)
                advantages = advantages * completion_mask

        preprocessing_time = preprocessing_time_fn()

        token_count = jnp.maximum(jnp.sum(completion_mask), 1.0)
        metrics_dict: dict[str, float | int | str] = {
            "score_mean": float(jnp.nanmean(scores)),
            "reward_mean": float(jnp.sum(rewards) / token_count),
            "mean_kl": float(jnp.sum(kl * completion_mask) / token_count),
            "completion_length": float(jnp.mean(jnp.sum(completion_mask, axis=1))),
            "rewarding_time": rewarding_time,
            "rollout_stats_time": rollout_stats_time,
            "ref_logps_time": ref_logps_time,
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }
        for i, reward_func_name in enumerate(self.reward_func_names):
            metrics_dict[reward_func_name] = float(jnp.nanmean(rewards_per_func[:, i]))
        self._log_training_generations_to_wandb(
            state=state,
            prompts=completion_prompts,
            completions=completions_text,
            completion_lengths=log_completion_length,
            generation_time=generation_time,
            reasoning=reasoning_records,
            tool_calls=tool_call_records,
            source="policy",
        )

        return (
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "old_logps": old_logps,
                "old_values": old_values,
                "advantages": advantages,
                "returns": returns,
            },
            metrics_dict,
        )
