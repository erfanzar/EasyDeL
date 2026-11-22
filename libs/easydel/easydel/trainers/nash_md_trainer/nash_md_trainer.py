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
"""Nash Mirror Descent (Nash-MD) trainer.

Nash-MD -- Munos et al. (2023) -- searches for the Nash equilibrium
of a self-play game by descending the regularised expected reward
against a mixture between the current policy and a frozen reference,
using a preference oracle to score head-to-head completions.  The
trainer subclasses :class:`GRPOTrainer` for the online generation /
reward pipeline and overrides only the loss/step.
"""

from __future__ import annotations

import collections
import collections.abc
import copy
import typing as tp

import jax
import jax.nn
import numpy as np
import spectrax as spx
from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]
from eformer.loggings import get_logger
from jax import numpy as jnp
from spectrax import with_sharding_constraint
from transformers import GenerationConfig

from easydel.inference.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from .._shared import copy_frozen_policy, drop_optimizer_state, esurge_generation
from ..base_trainer import GenerationResults
from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..metrics import _host_mean_float
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import compile_trainer_step, resolve_straight_through_emulator
from ._fn import nash_md_step
from .nash_md_config import NashMDConfig

logger = get_logger(__name__)

if tp.TYPE_CHECKING:
    pass


def _ensure_state(
    model: EasyDeLBaseModule | EasyDeLState | None,
    *,
    trainable_selector: tp.Any = "parameters",
) -> EasyDeLState | None:
    """Convert a module to an ``EasyDeLState`` if needed.

    Args:
        model: An :class:`EasyDeLBaseModule`, an :class:`EasyDeLState`,
            or ``None``. Already-promoted states are returned as-is.
        trainable_selector: Selector forwarded to
            :meth:`EasyDeLBaseModule.to_state` when promoting a raw
            module into a state.

    Returns:
        The corresponding :class:`EasyDeLState`, or ``None`` when
        ``model`` is ``None``.
    """
    if model is None:
        return None
    if isinstance(model, EasyDeLState):
        return model
    return model.to_state(trainable_selector=trainable_selector)


class GeometricMixtureLogitsProcessor(LogitsProcessor):
    """Replace policy logits with the policy/reference geometric-mixture logits."""

    def __init__(
        self,
        ref_model: EasyDeLBaseModule,
        mixture_coef: float | jax.Array,
        pad_token_id: int | None,
        prompt_mask: jnp.ndarray | None = None,
    ):
        self.ref_model = ref_model
        self.mixture_coef = mixture_coef if isinstance(mixture_coef, jax.Array) else float(mixture_coef)
        self.pad_token_id = pad_token_id
        self.prompt_mask = prompt_mask

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        seq_len = input_ids.shape[1]
        seq_positions = jnp.arange(seq_len)[None, :]
        attention_mask = seq_positions < cur_len
        if self.prompt_mask is not None:
            prompt_len = self.prompt_mask.shape[1]
            if prompt_len < seq_len:
                full_valid_mask = jnp.concatenate(
                    [
                        self.prompt_mask.astype(jnp.bool_),
                        jnp.ones((input_ids.shape[0], seq_len - prompt_len), dtype=jnp.bool_),
                    ],
                    axis=1,
                )
            else:
                full_valid_mask = self.prompt_mask[:, :seq_len].astype(jnp.bool_)
            attention_mask = attention_mask & full_valid_mask
        elif self.pad_token_id is not None:
            attention_mask = attention_mask & (input_ids != jnp.asarray(self.pad_token_id, dtype=input_ids.dtype))

        ref_outputs = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask.astype(jnp.int32),
        )
        ref_logits = jax.lax.dynamic_slice_in_dim(ref_outputs.logits, cur_len - 1, 1, axis=1)[:, 0, :]
        mix = jnp.asarray(self.mixture_coef, dtype=jnp.float32)
        mixed_logits = mix * ref_logits.astype(jnp.float32) + (1.0 - mix) * scores.astype(jnp.float32)
        return jax.nn.log_softmax(mixed_logits, axis=-1)


@Registry.register("trainer", ["nash-md", "nash_md"])
class NashMDTrainer(GRPOTrainer):
    """Trainer for Nash Mirror Descent (Nash-MD) preference optimization.

    Implements the Nash-MD algorithm (Munos et al. 2024): each step
    samples auxiliary completions from the *geometric mixture* of the
    policy and reference logits. Following TRL, ``mixture_coef`` is the
    reference-logit coefficient:
    ``logits_mix = mixture_coef * logits_ref + (1 - mixture_coef) * logits_policy``,
    treats those mixture samples as the opposing player, and updates
    the policy via a mirror-descent step regularised by the
    token-level KL against the reference. Both ``mixture_coef`` and
    ``beta`` may be provided as epoch-wise schedules in the config.

    Inherits the GRPO data plumbing (rollout generator, reward
    routing, advantage construction) from :class:`GRPOTrainer`; only
    the loss step (:func:`nash_md_step`) and the rollout post-
    processing (mixture sampling + oracle preference scoring) differ.

    See :func:`nash_md_step` for the loss form.

    Attributes:
        arguments: :class:`NashMDConfig` controlling the mirror-descent
            schedules and the inherited ``GRPOConfig`` surface.
        reference_state: Frozen reference :class:`EasyDeLState` used
            to evaluate the mixture and the KL-regularisation term.
        reward_funcs: Single oracle reward callable / module that
            scores head-to-head ``(policy_completion,
            mixture_completion)`` pairs.
    """

    arguments: NashMDConfig
    _MIXTURE_GENERATE_CACHE_MAXSIZE: int = 4

    def __init__(
        self,
        arguments: NashMDConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reward_funcs: collections.abc.Sequence[tp.Callable] | tp.Callable,
        *,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
        processing_class: ProcessingClassType,
        reward_processing_classes: ProcessingClassType | list[ProcessingClassType] | None = None,
    ):
        """Initialize the Nash-MD trainer.

        Delegates the heavy lifting (state setup, reward routing, online
        generation pipeline) to :class:`GRPOTrainer`, then attaches the
        Nash-MD specific schedules (``beta``, ``mixture_coef``,
        ``missing_eos_penalty``).

        Args:
            arguments: Nash-MD-specific training configuration.
            model: Policy module or state.
            reward_funcs: Single reward callable / module / state, or a
                list with exactly one element (Nash-MD scores
                head-to-head completions, so multi-reward isn't
                supported).
            reference_model: Optional reference module/state; when
                omitted the GRPO trainer's deep-copy default is used.
            train_dataset: Optional training dataset of prompts.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer/processor.
            reward_processing_classes: Optional per-reward tokenizer
                overrides.

        Raises:
            ValueError: If ``reward_funcs`` is missing or contains more
                than one entry.
        """
        if reward_funcs is None:
            raise ValueError("NashMDTrainer requires at least one reward function.")
        reward_funcs_list = (
            list(reward_funcs)
            if isinstance(reward_funcs, collections.abc.Sequence) and not isinstance(reward_funcs, (str, bytes))
            else [reward_funcs]
        )
        if len(reward_funcs_list) != 1:
            raise ValueError("NashMDTrainer only supports a single reward function/model.")
        reward_funcs_list = [
            drop_optimizer_state(rf) if isinstance(rf, EasyDeLState) else rf
            for rf in reward_funcs_list
        ]
        self._reference_model = reference_model
        self._current_step = 0
        self._eval_ref_graphdef = None
        self._mixture_generate_cache: collections.OrderedDict[
            tuple[tp.Any, ...], tuple[tp.Any, tp.Any, tp.Callable[..., tp.Any]]
        ] = collections.OrderedDict()
        self._mixture_prng_key: jax.Array | None = None
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs_list,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
        )
        self._beta_schedule = arguments.beta
        self._mixture_schedule = arguments.mixture_coef
        self.missing_eos_penalty = arguments.missing_eos_penalty
        self.num_generations = 1
        self._sync_current_step(getattr(self, "model_state", None))

    def _create_reference_state(self, model: EasyDeLState) -> EasyDeLState:
        """Keep a mixture-generation reference even when KL regularization is zero."""
        ref_state = _ensure_state(self._reference_model, trainable_selector=self.arguments.trainable_selector)
        # Do not retain a caller's optimizer-bearing state after freezing it.
        self._reference_model = None
        if ref_state is None:
            return copy_frozen_policy(model)
        ref_state = drop_optimizer_state(ref_state)
        return self._share_esurge_cache_scope_if_compatible(ref_state, primary_state=model)

    def _sync_current_step(self, state: EasyDeLState | None) -> None:
        """Record the host-side step counter from a live (non-donated) state."""
        step = getattr(state, "step", None)
        if step is None:
            return
        try:
            self._current_step = int(jax.device_get(step))
        except Exception:
            pass

    def _schedule_value(self, schedule: tp.Any, default: float, *, step: int | None = None) -> float:
        """Resolve the current value of an epoch-wise schedule.

        Scalars are returned as ``float(schedule)``. Sequences are
        indexed by the current epoch (estimated from
        ``_current_step // steps_per_epoch``) and clamped to the last
        entry once training overruns the schedule. Empty sequences
        and ``None`` fall back to ``default``.

        Args:
            schedule: Either a scalar value, a non-empty sequence of
                per-epoch values, or ``None``.
            default: Value returned when ``schedule`` is ``None`` or
                an empty sequence.
            step: Optional explicit step override; defaults to the host-side
                ``_current_step`` counter (avoiding reads on donated
                ``self.model_state.step`` buffers).

        Returns:
            The float-valued current schedule entry.
        """
        if isinstance(schedule, collections.abc.Sequence):
            if not schedule:
                return default
            if self.max_training_steps is None or self.arguments.num_train_epochs <= 0:
                return float(schedule[0])
            steps_per_epoch = max(self.max_training_steps // self.arguments.num_train_epochs, 1)
            current_step = int(getattr(self, "_current_step", 0)) if step is None else int(step)
            idx = min(max(current_step, 0) // steps_per_epoch, len(schedule) - 1)
            return float(schedule[idx])
        if schedule is None:
            return default
        return float(schedule)

    def _current_beta_value(self) -> float:
        """Get current beta value from schedule.

        Returns:
            Current beta value.
        """
        return self._schedule_value(self._beta_schedule, 0.1)

    def _current_mixture_coef(self) -> float:
        """Get current mixture coefficient from schedule.

        Returns:
            Current mixture coefficient.
        """
        return self._schedule_value(self._mixture_schedule, 0.5)

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Return the current ``beta`` schedule value as a hashable Python float."""
        return (float(self._current_beta_value()),)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any, ...]:
        """Return the current ``beta`` schedule value as a hashable Python float."""
        return (float(self._current_beta_value()),)

    def _configure_eval_config_override(self):
        """Apply evaluation config overrides to the reference forward as well."""
        super()._configure_eval_config_override()
        self._eval_ref_graphdef = None
        overrides = getattr(self.arguments, "eval_config_overrides", None)
        if self._eval_graphdef is not None and overrides and self.ref_state is not None:
            ref_graphstate = self.ref_state.graphstate
            if hasattr(ref_graphstate, "items"):
                selected_keys = {(collection, path) for collection, path, _ in ref_graphstate.items()}

                def ref_selector(v, path):
                    return (v.kind, path) in selected_keys
            else:
                ref_selector = None
            self._eval_ref_graphdef = self._build_config_override_graphdef(
                model=self.ref_state.model,
                overrides=overrides,
                reference_graphstate=ref_graphstate,
                trainable_selector=ref_selector,
                label="eval_config_overrides (reference)",
                eval_mode=True,
            )

    def on_step_end(self, state: EasyDeLState, metrics, step: int):
        """Track host-side step progress and delegate reference syncing to GRPO."""
        self._current_step = int(step)
        return super().on_step_end(state=state, metrics=metrics, step=step)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Build the JIT-compiled Nash-MD training/evaluation step functions.

        Resolves the optional QAT straight-through emulator, captures
        the static argument tuple (``beta`` schedule, vocab chunk
        size, partition spec, ...), and compiles
        :func:`nash_md_step` once for training (with state donation)
        and once for evaluation. Also installs the sharded
        per-token-logp callable used by the rollout pipeline to score
        the mixture and reference completions.

        Returns:
            ``TrainerConfigureFunctionOutput`` with the sharded
            training / evaluation step callables, the model mesh,
            and the streaming checkpoint manager.
        """
        self._clear_mixture_generate_cache()
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
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,
            straight_through_emulator,
        )
        self._eval_shared_fn_static_args = (
            self.arguments.logprob_vocab_chunk_size,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,
            straight_through_emulator,
        )

        static_argnums = (3, 4, 5, 6, 7, 8, 9)
        self._runtime_trace("train.compile_wrapper.begin")
        sharded_training_step_function = compile_trainer_step(
            nash_md_step,
            in_shardings=(self.state_shardings, empty_sharding, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("train.compile_wrapper.end")
        self._runtime_trace("eval.compile_wrapper.begin")
        sharded_evaluation_step_function = compile_trainer_step(
            nash_md_step,
            in_shardings=(self.eval_state_shardings, empty_sharding, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
            mesh=self.model.mesh,
            schedule=self.arguments.mpmd_scheduler,
            arguments=self.arguments,
        )
        self._runtime_trace("eval.compile_wrapper.end")

        def _build_ref_logps_fn(graphdef: spx.GraphDef):
            def _compute_model_logps(
                graphtree: spx.State,
                graphother: spx.State,
                ids: jax.Array,
                mask: jax.Array,
                prompt_length: int,
            ):
                """Sharded model per-token log-prob forward used for reference scoring.

                Stops gradients through the frozen graphother, applies the
                trainer's step partition spec, and dispatches to
                :func:`get_per_token_logps` with the configured vocab chunk
                size and dynamic prompt width.
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
                    return get_per_token_logps(
                        apply,
                        ids,
                        mask,
                        prompt_length,
                        logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                    )

            compiled = compile_trainer_step(
                _compute_model_logps,
                mesh=mesh,
                static_argnums=(4,),
                in_shardings=(
                    self.ref_state.shardings.graphstate,
                    self.ref_state.shardings.graphother,
                    empty_sharding,
                    empty_sharding,
                ),
                out_shardings=empty_sharding,
            )

            def _call(
                graphtree: spx.State,
                graphother: spx.State,
                ids: jax.Array,
                mask: jax.Array,
                prompt_length: int | None = None,
            ):
                resolved_prompt_length = (
                    self.arguments.max_prompt_length if prompt_length is None else int(prompt_length)
                )
                return compiled(graphtree, graphother, ids, mask, resolved_prompt_length)

            return _call

        self.compute_refmodel_logps = _build_ref_logps_fn(self.ref_state.graphdef)
        eval_ref_graphdef = getattr(self, "_eval_ref_graphdef", None)
        self.eval_compute_refmodel_logps = (
            _build_ref_logps_fn(eval_ref_graphdef)
            if eval_ref_graphdef is not None
            else self.compute_refmodel_logps
        )

        self.arguments.ensure_checkpoint_path()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def _score_rewards(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        raw_text: list[str] | None = None,
        reasoning: list[str | None] | None = None,
        tool_calls: list[tp.Any | None] | None = None,
        batch: dict[str, tp.Any] | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute reward scores for prompt-completion pairs.

        Iterates over each registered reward function (or
        :class:`EasyDeLState` reward model), routes the prompt /
        completion strings through the configured tokenizer when
        scoring with a model, and accumulates a per-function score plus
        a name-keyed breakdown for logging.

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings.
            raw_text: Optional raw (pre-decoded) completion strings to
                forward to reward functions that consume them
                directly. Defaults to ``None``.
            reasoning: Optional per-completion reasoning traces (one
                per completion, ``None`` when unavailable).
            tool_calls: Optional per-completion tool-call metadata.
            batch: Optional original batch dict (forwarded to reward
                functions that accept it).

        Returns:
            Tuple ``(total_rewards, reward_breakdown)`` where
            ``total_rewards`` is a ``[batch]`` float32 array summed
            across reward functions and ``reward_breakdown`` maps each
            reward function's name to its ``[batch]`` score array.

        Raises:
            ValueError: If an :class:`EasyDeLState` reward model is
                supplied without a matching
                ``reward_processing_class``.
        """
        rewards = []
        breakdown: dict[str, jnp.ndarray] = {}
        for reward_func, reward_processing_class in zip(
            self.reward_funcs,
            self.reward_processing_classes,
            strict=True,
        ):
            name = getattr(reward_func, "__name__", None) or reward_func.__class__.__name__
            if isinstance(reward_func, EasyDeLState):
                if reward_processing_class is None:
                    raise ValueError("Reward processing class must be provided when using EasyDeL reward models.")
                texts = [p + c for p, c in zip(prompts, completions, strict=False)]
                tokenized = reward_processing_class(
                    texts,
                    return_tensors="np",
                    padding="max_length",
                    padding_side="right",
                    add_special_tokens=False,
                    truncation=True,
                    return_attention_mask=True,
                    max_length=self.arguments.max_length,
                )
                logits = reward_func.apply_fn(
                    reward_func.graphdef,
                    reward_func.graphstate,
                    reward_func.graphother,
                    dict(tokenized),
                ).logits[:, 0]
                values = jnp.asarray(logits, dtype=jnp.float32)
            else:
                reward_call_kwargs = self._build_reward_call_kwargs(
                    reward_func,
                    prompts=prompts,
                    completions=completions,
                    raw_completions=raw_text,
                    prompt_texts=prompts,
                    completion_texts=completions,
                    raw_text=raw_text,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    max_length=self.arguments.max_length,
                    batch=batch,
                )
                outputs = reward_func(**reward_call_kwargs)
                values = jnp.asarray(np.asarray(outputs, dtype=np.float32))
            rewards.append(values)
            breakdown[name] = values

        if not rewards:
            zeros = jnp.zeros((len(prompts),), dtype=jnp.float32)
            return zeros, {}

        stacked = jnp.stack(rewards, axis=1)
        totals = stacked.sum(axis=1)
        return totals, breakdown

    def _clear_mixture_generate_cache(self) -> None:
        """Clear cached compiled mixture-generation executables."""
        cache = getattr(self, "_mixture_generate_cache", None)
        if cache is not None:
            cache.clear()

    def _get_cached_mixture_fn(
        self,
        key: tuple[tp.Any, ...],
        *,
        policy_graphdef: tp.Any,
        ref_graphdef: tp.Any,
    ) -> tp.Callable[..., tp.Any] | None:
        """Return a cached mixture executable when both ``GraphDef`` identities match."""
        cache = getattr(self, "_mixture_generate_cache", None)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        cached_policy_gd, cached_ref_gd, compiled = entry
        if cached_policy_gd is policy_graphdef and cached_ref_gd is ref_graphdef:
            if isinstance(cache, collections.OrderedDict):
                cache.move_to_end(key)
            else:
                cache.pop(key, None)
                cache[key] = entry
            return compiled
        cache.pop(key, None)
        return None

    def _put_cached_mixture_fn(
        self,
        key: tuple[tp.Any, ...],
        *,
        policy_graphdef: tp.Any,
        ref_graphdef: tp.Any,
        compiled: tp.Callable[..., tp.Any],
    ) -> None:
        """Insert a compiled mixture executable into the bounded LRU cache."""
        cache = getattr(self, "_mixture_generate_cache", None)
        if cache is None:
            cache = collections.OrderedDict()
            self._mixture_generate_cache = cache
        entry = (policy_graphdef, ref_graphdef, compiled)
        if isinstance(cache, collections.OrderedDict):
            cache[key] = entry
            cache.move_to_end(key)
        else:
            cache.pop(key, None)
            cache[key] = entry
        max_size = max(1, int(getattr(self, "_MIXTURE_GENERATE_CACHE_MAXSIZE", 4)))
        while len(cache) > max_size:
            if isinstance(cache, collections.OrderedDict):
                cache.popitem(last=False)
            else:
                cache.pop(next(iter(cache)))

    def release_generation_runtime(
        self,
        *,
        state: EasyDeLState | None = None,
        extra_states: collections.abc.Iterable[EasyDeLState | None] = (),
        clear_esurge_compiled_cache: bool = False,
    ) -> None:
        """Release eSurge runtimes and drop cached mixture executables on full/cache teardown."""
        if state is None or clear_esurge_compiled_cache:
            self._clear_mixture_generate_cache()
        super().release_generation_runtime(
            state=state,
            extra_states=extra_states,
            clear_esurge_compiled_cache=clear_esurge_compiled_cache,
        )

    def _can_use_cached_kv_mixture(
        self,
        policy_state: tp.Any,
        ref_state: tp.Any,
    ) -> bool:
        """Return whether both policy and reference support KV-cached JIT mixture decoding."""
        if "rollout" in self.__dict__:
            return False
        if not isinstance(policy_state, EasyDeLState) or not isinstance(ref_state, EasyDeLState):
            return False
        policy_model = getattr(policy_state, "model", None)
        ref_model = getattr(ref_state, "model", None)
        if policy_model is None or ref_model is None:
            return False
        if getattr(getattr(policy_model, "config", None), "is_encoder_decoder", False) or getattr(
            getattr(ref_model, "config", None), "is_encoder_decoder", False
        ):
            return False
        required = ("prepare_inputs_for_generation", "_call_generation_model_step", "update_inputs_for_generation")
        return all(hasattr(policy_model, attr) and hasattr(ref_model, attr) for attr in required)

    def _can_use_cached_fallback_mixture(
        self,
        policy_state: tp.Any,
        ref_state: tp.Any,
    ) -> bool:
        """Return whether policy and reference support compiled fallback mixture decoding."""
        if "rollout" in self.__dict__:
            return False
        if not isinstance(policy_state, EasyDeLState) or not isinstance(ref_state, EasyDeLState):
            return False
        policy_model = getattr(policy_state, "model", None)
        ref_model = getattr(ref_state, "model", None)
        if policy_model is None or ref_model is None:
            return False
        return hasattr(policy_model, "generate") and callable(ref_model)

    def _get_compiled_mixture_generate_fn(
        self,
        *,
        policy_state: EasyDeLState,
        ref_state: EasyDeLState,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_ids: tuple[int, ...],
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tp.Callable[..., tuple[jax.Array, jax.Array]]:
        """Return a cached, KV-cached JIT mixture decoder for ``(policy, ref)``."""
        key = (
            "kv_cached",
            id(policy_state.graphdef),
            id(ref_state.graphdef),
            int(max_new_tokens),
            int(pad_token_id),
            tuple(int(t) for t in eos_token_ids),
            bool(do_sample),
            float(temperature),
            int(top_k),
            float(top_p),
        )
        cached = self._get_cached_mixture_fn(
            key,
            policy_graphdef=policy_state.graphdef,
            ref_graphdef=ref_state.graphdef,
        )
        if cached is not None:
            return cached

        policy_graphdef = policy_state.graphdef
        ref_graphdef = ref_state.graphdef
        mesh = policy_state.model.mesh
        empty_sharding = replicated_named_sharding(mesh)

        warpers = LogitsProcessorList()
        if do_sample:
            if temperature > 0.0 and temperature != 1.0:
                warpers.append(TemperatureLogitsWarper(float(temperature)))
            if top_k > 0:
                warpers.append(TopKLogitsWarper(int(top_k)))
            if 0.0 < top_p < 1.0:
                warpers.append(TopPLogitsWarper(float(top_p)))

        def _strip_mismatched_masks(model_kwargs: dict[str, tp.Any], running_len: int) -> dict[str, tp.Any]:
            call_kwargs = dict(model_kwargs)
            for mask_key in ("attention_mask", "decoder_attention_mask"):
                mask = call_kwargs.get(mask_key, None)
                if mask is None:
                    continue
                if hasattr(mask, "shape") and len(mask.shape) > 0 and mask.shape[-1] != running_len:
                    call_kwargs.pop(mask_key, None)
            return call_kwargs

        def _mixture_generate(
            policy_gs: spx.State,
            policy_go: spx.State,
            ref_gs: spx.State,
            ref_go: spx.State,
            prompt_ids: jax.Array,
            prompt_mask: jax.Array,
            mixture_coef: jax.Array,
            prng_key: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            policy_module = spx.bind(policy_graphdef, policy_gs.merge(policy_go, copy=False))
            ref_module = spx.bind(ref_graphdef, ref_gs.merge(ref_go, copy=False))

            batch_size, prompt_len = prompt_ids.shape
            max_length = prompt_len + max_new_tokens
            pad_id = jnp.asarray(pad_token_id, dtype=jnp.int32)
            eos_arr = jnp.asarray(eos_token_ids, dtype=jnp.int32) if eos_token_ids else None
            mix = jnp.asarray(mixture_coef, dtype=jnp.float32)

            prompt_ids_i32 = prompt_ids.astype(jnp.int32)
            prompt_mask_i32 = prompt_mask.astype(jnp.int32)
            sequences = jnp.full((batch_size, max_length), pad_id, dtype=jnp.int32)
            sequences = jax.lax.dynamic_update_slice(sequences, prompt_ids_i32, (0, 0))
            completion_mask = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
            is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

            with policy_module.mesh:
                policy_kwargs = policy_module.prepare_inputs_for_generation(
                    prompt_ids_i32,
                    max_length=max_length,
                    pad_token_id=pad_id,
                    starts=None,
                    attention_mask=prompt_mask_i32,
                )
                ref_kwargs = ref_module.prepare_inputs_for_generation(
                    prompt_ids_i32,
                    max_length=max_length,
                    pad_token_id=pad_id,
                    starts=None,
                    attention_mask=prompt_mask_i32,
                )

                def cond_fn(loop_state):
                    cur_len, _, _, _, finished, _, _, _ = loop_state
                    return ~((cur_len >= max_length) | jnp.all(finished))

                def body_fn(loop_state):
                    cur_len, seqs, comp_mask, running_token, finished, key_state, p_kwargs, r_kwargs = loop_state
                    running_len = running_token.shape[1]
                    p_out = policy_module._call_generation_model_step(
                        policy_module,
                        running_token,
                        _strip_mismatched_masks(p_kwargs, running_len),
                    )
                    r_out = ref_module._call_generation_model_step(
                        ref_module,
                        running_token,
                        _strip_mismatched_masks(r_kwargs, running_len),
                    )
                    p_logits = p_out.logits[:, -1].astype(jnp.float32)
                    r_logits = r_out.logits[:, -1].astype(jnp.float32)
                    mixed_logits = jax.nn.log_softmax(mix * r_logits + (1.0 - mix) * p_logits, axis=-1)

                    if do_sample:
                        step_key, next_key = jax.random.split(key_state)
                        warped = warpers(seqs, mixed_logits, cur_len)
                        sampled = jax.random.categorical(step_key, warped, axis=-1).astype(jnp.int32)
                    else:
                        next_key = key_state
                        sampled = jnp.argmax(mixed_logits, axis=-1).astype(jnp.int32)

                    active = ~finished
                    next_token = jnp.where(active, sampled, pad_id)
                    if eos_arr is not None:
                        next_finished = finished | jnp.isin(next_token, eos_arr)
                    else:
                        next_finished = finished

                    step_idx = cur_len - prompt_len
                    next_comp_mask = jax.lax.dynamic_update_slice(
                        comp_mask,
                        active.astype(jnp.int32)[:, None],
                        (0, step_idx),
                    )
                    next_seqs = jax.lax.dynamic_update_slice(seqs, next_token[:, None], (0, cur_len))
                    next_p_kwargs = policy_module.update_inputs_for_generation(p_out, p_kwargs)
                    next_r_kwargs = ref_module.update_inputs_for_generation(r_out, r_kwargs)
                    return (
                        cur_len + 1,
                        next_seqs,
                        next_comp_mask,
                        next_token[:, None],
                        next_finished,
                        next_key,
                        next_p_kwargs,
                        next_r_kwargs,
                    )

                init_state = (
                    jnp.asarray(prompt_len, dtype=jnp.int32),
                    sequences,
                    completion_mask,
                    prompt_ids_i32,
                    is_sent_finished,
                    prng_key,
                    policy_kwargs,
                    ref_kwargs,
                )
                if prompt_len > 1:
                    init_state = body_fn(init_state)
                final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
                return final_state[1], final_state[2]

        compiled = compile_trainer_step(
            _mixture_generate,
            mesh=mesh,
            in_shardings=(
                policy_state.shardings.graphstate,
                policy_state.shardings.graphother,
                ref_state.shardings.graphstate,
                ref_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=(empty_sharding, empty_sharding),
        )
        self._put_cached_mixture_fn(
            key,
            policy_graphdef=policy_graphdef,
            ref_graphdef=ref_graphdef,
            compiled=compiled,
        )
        return compiled

    def _get_compiled_fallback_mixture_generate_fn(
        self,
        *,
        policy_state: EasyDeLState,
        ref_state: EasyDeLState,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_ids: tuple[int, ...],
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tp.Callable[..., jax.Array]:
        """Return a cached JIT fallback mixture decoder taking both states and masks as traced args."""
        key = (
            "fallback",
            id(policy_state.graphdef),
            id(ref_state.graphdef),
            int(max_new_tokens),
            int(pad_token_id),
            tuple(int(t) for t in eos_token_ids),
            bool(do_sample),
            float(temperature),
            int(top_k),
            float(top_p),
        )
        cached = self._get_cached_mixture_fn(
            key,
            policy_graphdef=policy_state.graphdef,
            ref_graphdef=ref_state.graphdef,
        )
        if cached is not None:
            return cached

        policy_graphdef = policy_state.graphdef
        ref_graphdef = ref_state.graphdef
        mesh = policy_state.model.mesh
        empty_sharding = replicated_named_sharding(mesh)

        base_gen_config = getattr(policy_state.model, "generation_config", None)
        if base_gen_config is not None:
            config_copy = copy.deepcopy(base_gen_config)
        else:
            model_cfg = getattr(policy_state.model, "config", None)
            try:
                config_copy = (
                    GenerationConfig.from_model_config(model_cfg)
                    if model_cfg is not None
                    else GenerationConfig()
                )
            except Exception:
                config_copy = GenerationConfig()
        extra_gen_kwargs: dict[str, tp.Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "num_return_sequences": 1,
            "pad_token_id": int(pad_token_id),
        }
        if eos_token_ids:
            extra_gen_kwargs["eos_token_id"] = eos_token_ids[0] if len(eos_token_ids) == 1 else list(eos_token_ids)
        if do_sample:
            if temperature > 0.0:
                extra_gen_kwargs["temperature"] = float(temperature)
            if top_k > 0:
                extra_gen_kwargs["top_k"] = int(top_k)
            if 0.0 < top_p < 1.0:
                extra_gen_kwargs["top_p"] = float(top_p)

        if config_copy is not None:
            for attr_key in list(extra_gen_kwargs.keys()):
                if hasattr(config_copy, attr_key):
                    setattr(config_copy, attr_key, extra_gen_kwargs.pop(attr_key))

        def _fallback_mixture_generate(
            policy_gs: spx.State,
            policy_go: spx.State,
            ref_gs: spx.State,
            ref_go: spx.State,
            prompt_ids: jax.Array,
            prompt_mask: jax.Array,
            mixture_coef: jax.Array,
            prng_key: jax.Array,
        ) -> jax.Array:
            policy_module = spx.bind(policy_graphdef, policy_gs.merge(policy_go, copy=False))
            ref_module = spx.bind(ref_graphdef, ref_gs.merge(ref_go, copy=False))
            mixture_logits_processor = LogitsProcessorList(
                [
                    GeometricMixtureLogitsProcessor(
                        ref_model=ref_module,
                        mixture_coef=mixture_coef,
                        pad_token_id=pad_token_id,
                        prompt_mask=prompt_mask,
                    )
                ]
            )
            with policy_module.mesh:
                call_kwargs: dict[str, tp.Any] = {
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "prng_key": prng_key,
                    "logits_processor": mixture_logits_processor,
                    **extra_gen_kwargs,
                }
                if config_copy is not None:
                    call_kwargs["generation_config"] = config_copy
                outputs = policy_module.generate(**call_kwargs)
                return outputs.sequences if hasattr(outputs, "sequences") else outputs

        compiled = compile_trainer_step(
            _fallback_mixture_generate,
            mesh=mesh,
            in_shardings=(
                policy_state.shardings.graphstate,
                policy_state.shardings.graphother,
                ref_state.shardings.graphstate,
                ref_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
                empty_sharding,
                empty_sharding,
            ),
            out_shardings=empty_sharding,
        )
        self._put_cached_mixture_fn(
            key,
            policy_graphdef=policy_graphdef,
            ref_graphdef=ref_graphdef,
            compiled=compiled,
        )
        return compiled

    def _generate_mixture_rollout(
        self,
        *,
        policy_state: EasyDeLState,
        ref_state: EasyDeLState,
        prompt_ids: jax.Array,
        prompt_mask: jax.Array,
        mixture_coef: float,
        use_kv_cache: bool = True,
    ) -> GenerationResults:
        """Generate geometric-mixture completions using cached dual-KV or fallback decoding."""
        sampling_params = esurge_generation.resolve_sampling(self.arguments, {"num_return_sequences": 1})
        max_new_tokens = int(
            getattr(sampling_params, "max_tokens", None) or getattr(self.arguments, "max_completion_length", 64) or 64
        )
        pad_token_id = 0 if getattr(self, "_pad_token_id", None) is None else int(self._pad_token_id)
        raw_eos = getattr(self, "_eos_token_id", None)
        if raw_eos is None:
            raw_eos = getattr(self.processing_class, "eos_token_id", None)
        if raw_eos is None:
            eos_token_ids: tuple[int, ...] = ()
        elif isinstance(raw_eos, (list, tuple)):
            eos_token_ids = tuple(int(tok) for tok in raw_eos if tok is not None)
        else:
            eos_token_ids = (int(raw_eos),)

        temperature = float(getattr(sampling_params, "temperature", 1.0) or 0.0)
        do_sample_override = getattr(self.arguments, "generation_do_sample", None)
        do_sample = (temperature > 0.0) if do_sample_override is None else bool(do_sample_override)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        raw_top_p = getattr(sampling_params, "top_p", 1.0)
        top_p = 1.0 if raw_top_p is None else float(raw_top_p)

        prompt_len = int(prompt_ids.shape[1])
        if max_new_tokens <= 0:
            empty_comp = jnp.zeros((prompt_ids.shape[0], 0), dtype=prompt_ids.dtype)
            sequences = prompt_ids
            completion_ids = empty_comp
            completion_mask = empty_comp
        else:
            prng_key = getattr(self, "_mixture_prng_key", None)
            if prng_key is None:
                prng_key = jax.random.PRNGKey(int(getattr(self.arguments, "seed", 0) or 0))
            self._mixture_prng_key, step_key = jax.random.split(prng_key)
            if use_kv_cache:
                compiled = self._get_compiled_mixture_generate_fn(
                    policy_state=policy_state,
                    ref_state=ref_state,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_ids=eos_token_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                sequences, completion_mask = compiled(
                    policy_state.graphstate,
                    policy_state.graphother,
                    ref_state.graphstate,
                    ref_state.graphother,
                    prompt_ids,
                    prompt_mask,
                    jnp.asarray(mixture_coef, dtype=jnp.float32),
                    step_key,
                )
                completion_ids = sequences[:, prompt_len : prompt_len + max_new_tokens]
            else:
                compiled_fallback = self._get_compiled_fallback_mixture_generate_fn(
                    policy_state=policy_state,
                    ref_state=ref_state,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_ids=eos_token_ids,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                sequences = compiled_fallback(
                    policy_state.graphstate,
                    policy_state.graphother,
                    ref_state.graphstate,
                    ref_state.graphother,
                    prompt_ids,
                    prompt_mask,
                    jnp.asarray(mixture_coef, dtype=jnp.float32),
                    step_key,
                )
                if sequences.shape[1] >= prompt_len:
                    completion_ids = sequences[:, prompt_len : prompt_len + max_new_tokens]
                else:
                    completion_ids = sequences[:, :max_new_tokens]
                if getattr(self, "_eos_token_id", None) is not None:
                    completion_mask = self._make_attn_mask(completion_ids)
                else:
                    completion_mask = jnp.ones_like(completion_ids, dtype=jnp.int32)

        processor = self._get_processing_class()
        host_prompt_ids = np.asarray(jax.device_get(prompt_ids), dtype=np.int64)
        host_prompt_mask = np.asarray(jax.device_get(prompt_mask), dtype=np.int32)
        host_comp_ids = np.asarray(jax.device_get(completion_ids), dtype=np.int64)
        host_comp_mask = np.asarray(jax.device_get(completion_mask), dtype=np.int32)
        decoded_prompts = self._decode_prompt_batch(
            processor, host_prompt_ids, False, pad_token_id, True, host_prompt_mask
        )
        completion_texts = self._decode_prompt_batch(
            processor, host_comp_ids, True, pad_token_id, True, host_comp_mask
        )
        raw_completion_texts = self._decode_prompt_batch(
            processor, host_comp_ids, False, pad_token_id, True, host_comp_mask
        )
        return GenerationResults(
            generation_results=completion_texts,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            sequences=sequences,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            decoded_prompts=decoded_prompts,
            completion_prompts=decoded_prompts,
            text=completion_texts,
            reasoning=[None] * len(completion_texts),
            tool_calls=[None] * len(completion_texts),
            raw_text=raw_completion_texts,
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Generate policy and mixture completions, score them, and assemble the loss batch.

        Runs three logical phases:

        1. Generate a policy completion for each prompt, then generate
           an opposing completion from the policy/reference geometric
           logit mixture.
        2. Score both the policy completion and the chosen mixture
           completion with the oracle reward function, apply the
           optional missing-EOS penalty, and convert the score
           difference into a Bradley-Terry preference probability via
           :func:`jax.nn.sigmoid`.
        3. Compute reference per-token logps over the
           prompt+completion sequence and package the all-gathered
           tensors that :func:`nash_md_step` consumes.

        Also logs per-source completions to W&B and records the
        current ``beta`` / ``mixture_coef`` schedule values.

        Args:
            state: Current policy ``EasyDeLState``.
            batch: Input batch containing ``input_ids`` and
                ``attention_mask`` for the prompts.
            is_train: Whether this is a training step (unused
                directly, kept for API parity with the GRPO base).

        Returns:
            Tuple ``(processed_batch, metrics_dict)`` where
            ``processed_batch`` is the dict fed to
            :func:`nash_md_step` (prompts, completions, masks,
            ``ref_token_logps``, ``probabilities``) and
            ``metrics_dict`` carries reward/length/timing diagnostics
            for the trainer's logger.
        """
        self._sync_current_step(state)
        batch = self._apply_user_data_collator(batch)
        batch = self._purify_batch(batch)
        with capture_time() as preprocessing_time_fn:
            prompt_ids = batch["input_ids"]
            prompt_mask = batch["attention_mask"]
            active_ref_state = self.ref_state
            eval_ref_graphdef = getattr(self, "_eval_ref_graphdef", None)
            if not is_train and eval_ref_graphdef is not None and active_ref_state is not None:
                active_ref_state = active_ref_state.replace(graphdef=eval_ref_graphdef)

            with capture_time() as generation_time_fn:
                results = self.rollout(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                    config_overrides={"num_return_sequences": 1},
                )
                jax.block_until_ready(results.sequences)
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
                completion_mask = results.completion_mask
            generation_time = generation_time_fn()

            # Sample mixture completions from the policy/reference geometric
            # logit mixture, matching TRL's Nash-MD rollout semantics.
            mixture_coef = self._current_mixture_coef()
            with capture_time() as _mixture_time_fn:
                if mixture_coef <= 0.0:
                    mixture_results = self.rollout(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        state=state,
                        use_esurge=True,
                        apply_chat_template=False,
                        shard_inputs=False,
                        all_gather=False,
                        config_overrides={"num_return_sequences": 1},
                    )
                elif mixture_coef >= 1.0:
                    mixture_results = self.rollout(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        state=active_ref_state,
                        use_esurge=True,
                        apply_chat_template=False,
                        shard_inputs=False,
                        all_gather=False,
                        config_overrides={"num_return_sequences": 1},
                    )
                elif self._can_use_cached_kv_mixture(state, active_ref_state):
                    mixture_results = self._generate_mixture_rollout(
                        policy_state=state,
                        ref_state=active_ref_state,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        mixture_coef=mixture_coef,
                        use_kv_cache=True,
                    )
                elif self._can_use_cached_fallback_mixture(state, active_ref_state):
                    mixture_results = self._generate_mixture_rollout(
                        policy_state=state,
                        ref_state=active_ref_state,
                        prompt_ids=prompt_ids,
                        prompt_mask=prompt_mask,
                        mixture_coef=mixture_coef,
                        use_kv_cache=False,
                    )
                else:
                    mixture_logits_processor = LogitsProcessorList(
                        [
                            GeometricMixtureLogitsProcessor(
                                ref_model=active_ref_state.model,
                                mixture_coef=mixture_coef,
                                pad_token_id=self._pad_token_id,
                                prompt_mask=prompt_mask,
                            )
                        ]
                    )
                    mixture_results = self.rollout(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        state=state,
                        use_esurge=False,
                        apply_chat_template=False,
                        shard_inputs=False,
                        all_gather=False,
                        config_overrides={"num_return_sequences": 1},
                        logits_processor=mixture_logits_processor,
                    )
                jax.block_until_ready(mixture_results.sequences)
                mixture_completion_ids = mixture_results.completion_ids
                mixture_completion_mask = mixture_results.completion_mask
            mixture_generation_time = _mixture_time_fn()

            prompt_completion_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            attention_mask = jnp.concatenate([prompt_mask, completion_mask], axis=1)
            prompt_length = int(prompt_ids.shape[-1])
            ref_logps_fn = (
                self.compute_refmodel_logps
                if is_train
                else getattr(self, "eval_compute_refmodel_logps", self.compute_refmodel_logps)
            )

            with capture_time() as ref_logps_time_fn:
                ref_token_logps = ref_logps_fn(
                    active_ref_state.graphstate,
                    active_ref_state.graphother,
                    prompt_completion_ids,
                    attention_mask,
                    prompt_length=prompt_length,
                )
            ref_logps_time = ref_logps_time_fn()

        preprocessing_time = preprocessing_time_fn()

        host_prompt_ids = np.asarray(jax.device_get(prompt_ids))
        prompts_text = list(self.processing_class.batch_decode(host_prompt_ids, skip_special_tokens=True))
        model_completions_text = self._coerce_generation_texts(results.text)
        ref_completions_text = self._coerce_generation_texts(mixture_results.text)
        if not model_completions_text:
            host_completion_ids = np.asarray(jax.device_get(completion_ids))
            model_completions_text = list(
                self.processing_class.batch_decode(host_completion_ids, skip_special_tokens=True)
            )
        if not ref_completions_text:
            host_ref_completion_ids = np.asarray(jax.device_get(mixture_results.completion_ids))
            ref_completions_text = list(
                self.processing_class.batch_decode(host_ref_completion_ids, skip_special_tokens=True)
            )
        model_raw_texts = self._coerce_generation_texts(
            results.raw_text,
            fallback=model_completions_text,
        )
        ref_raw_texts = self._coerce_generation_texts(
            mixture_results.raw_text,
            fallback=ref_completions_text,
        )
        model_reasoning = self._coerce_optional_generation_texts(
            results.reasoning,
            target_len=len(model_completions_text),
        )
        model_tool_calls = self._coerce_generation_metadata_list(
            results.tool_calls,
            target_len=len(model_completions_text),
        )
        ref_reasoning = self._coerce_optional_generation_texts(
            mixture_results.reasoning,
            target_len=len(ref_completions_text),
        )
        ref_tool_calls = self._coerce_generation_metadata_list(
            mixture_results.tool_calls,
            target_len=len(ref_completions_text),
        )

        mixture_completions_text = ref_completions_text
        mixture_raw_texts = ref_raw_texts
        mixture_reasoning = ref_reasoning
        mixture_tool_calls = ref_tool_calls

        model_scores, model_breakdown = self._score_rewards(
            prompts_text,
            model_completions_text,
            raw_text=model_raw_texts,
            reasoning=model_reasoning,
            tool_calls=model_tool_calls,
            batch=batch,
        )
        mixture_scores, _ = self._score_rewards(
            prompts_text,
            mixture_completions_text,
            raw_text=mixture_raw_texts,
            reasoning=mixture_reasoning,
            tool_calls=mixture_tool_calls,
            batch=batch,
        )

        penalty = self.missing_eos_penalty
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if penalty is not None and eos_token_id is not None:
            eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            eos_tensor = jnp.asarray(eos_ids)
            policy_contains_eos = (jnp.isin(completion_ids, eos_tensor) & completion_mask.astype(bool)).any(axis=1)
            mixture_contains_eos = (
                jnp.isin(mixture_completion_ids, eos_tensor) & mixture_completion_mask.astype(bool)
            ).any(axis=1)
            model_scores = model_scores - jnp.where(policy_contains_eos, 0.0, penalty)
            mixture_scores = mixture_scores - jnp.where(mixture_contains_eos, 0.0, penalty)

        probabilities = jax.nn.sigmoid(model_scores - mixture_scores)
        completion_lengths = completion_mask.sum(axis=1)
        beta_value = float(self._current_beta_value())

        metrics_dict: dict[str, float | int | str] = {
            "rewards/chosen": _host_mean_float(model_scores),
            "rewards/rejected": _host_mean_float(mixture_scores),
            "rewards/margins": _host_mean_float(model_scores - mixture_scores),
            "rewards/probabilities": _host_mean_float(probabilities),
            "completion_length": _host_mean_float(completion_lengths),
            "generation_time": generation_time,
            "mixture_generation_time": mixture_generation_time,
            "reference_logps_time": ref_logps_time,
            "preprocessing_time": preprocessing_time,
            "mixture_coef": float(mixture_coef),
            "beta": beta_value,
        }
        for name, values in model_breakdown.items():
            metrics_dict[f"reward/{name}"] = _host_mean_float(values)
        if is_train:
            self._log_training_generations_to_wandb(
                state=state,
                prompts=prompts_text,
                completions=model_completions_text,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                generation_time=generation_time,
                reasoning=model_reasoning,
                tool_calls=model_tool_calls,
                source="policy",
            )
            self._log_training_generations_to_wandb(
                state=state,
                prompts=prompts_text,
                completions=mixture_completions_text,
                completion_ids=mixture_completion_ids,
                completion_mask=mixture_completion_mask,
                generation_time=mixture_generation_time,
                reasoning=mixture_reasoning,
                tool_calls=mixture_tool_calls,
                source="mixture",
            )

        processed_batch = {
            "prompt_ids": self._all_gather(prompt_ids),
            "prompt_mask": self._all_gather(prompt_mask),
            "completion_ids": self._all_gather(completion_ids),
            "completion_mask": self._all_gather(completion_mask),
            "ref_token_logps": self._all_gather(ref_token_logps),
            "probabilities": self._all_gather(probabilities),
            "beta": self._all_gather(jnp.full((prompt_ids.shape[0],), beta_value, dtype=jnp.float32)),
        }
        return processed_batch, metrics_dict
