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

"""Self-Distillation Policy Optimization (SDPO) trainer for EasyDeL.

SDPO (arXiv:2601.20802) replaces GRPO's scalar-reward advantages with
dense token-level advantages derived from a *self-teacher*: the current
policy re-prompted with ``(question, feedback, original_response)``.

The trainer supports two feedback modes:

1. **Rich feedback** - a ``feedback_func`` callable returns a string for
   every completion describing what went wrong (runtime errors, judge
   evaluations, etc.).  This is the primary SDPO setting (Sections 4-5 of
   the paper).

2. **Self-feedback** (default when ``feedback_func=None``) - successful
   rollouts from the same group are used as implicit feedback for failed
   ones (Section 3 of the paper).  No environment feedback is needed; only
   a binary reward signal to distinguish pass from fail.

Algorithmically, SDPO is identical to GRPO except that:
- Per-token advantages come from ``log π_θ(y_t|x,f,y<t) / π_θ(y_t|x,y<t)``
  instead of normalised scalar rewards.
- The loss is a distillation loss (KL or JSD) rather than a clipped PG loss.
- No reward normalisation or clipping is needed.
"""

from __future__ import annotations

import typing as tp
from functools import partial

import flax
import jax
import numpy as np
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time, get_logger
from easydel.utils.traversals import deepcopy_model

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from ..prompt_utils import apply_chat_template
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_configurations import MetricsType
from ..training_utils import resolve_straight_through_emulator
from ._fn import sdpo_step
from .sdpo_config import SDPOConfig

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)

FeedbackFunc = tp.Callable[
    [list[str], list[str], list[float]],
    list[str],
]
RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa

_FEEDBACK_CORRECT = "Your previous attempt was correct.\n"
_FEEDBACK_TEMPLATE_SOLUTION = "Correct solution:\n{solution}\n\n"
_FEEDBACK_TEMPLATE_ENV = "The following is feedback from your unsuccessful earlier attempt:\n{feedback}\n\n"
_FEEDBACK_TEMPLATE_SOLVE = "Correctly solve the original question.\n"


def _build_feedback_separator(
    *,
    is_successful: bool,
    env_feedback: str,
    correct_solution: str | None,
) -> str:
    """Return the text block inserted between the prompt and the completion.

    Args:
        is_successful: Whether the original attempt received a positive reward.
        env_feedback: Textual environment feedback (error message, judge note…).
            May be empty when only a correct_solution is available.
        correct_solution: Best completion from the same rollout group.
            ``None`` if the group has no successful attempt yet.

    Returns:
        The separator string to be tokenised and inserted in the teacher input.
    """
    if is_successful:
        return _FEEDBACK_CORRECT

    parts: list[str] = []
    if correct_solution is not None:
        parts.append(_FEEDBACK_TEMPLATE_SOLUTION.format(solution=correct_solution))
    if env_feedback:
        parts.append(_FEEDBACK_TEMPLATE_ENV.format(feedback=env_feedback))
    parts.append(_FEEDBACK_TEMPLATE_SOLVE)
    return "".join(parts)


@Registry.register("trainer", "sdpo")
class SDPOTrainer(GRPOTrainer):
    """Self-Distillation Policy Optimization trainer.

    Extends :class:`GRPOTrainer` by replacing scalar-reward advantages with
    dense per-token advantages produced by a self-teacher — the same model
    conditioned on the original prompt, environment feedback, and the
    student's own attempt.

    The only required change compared to a GRPO setup is:

    1. Replace ``GRPOConfig`` with ``SDPOConfig``.
    2. Optionally pass a ``feedback_func`` that converts ``(prompts,
       completions, rewards)`` into per-completion feedback strings.
       If omitted the trainer uses successful rollouts as implicit feedback.

    Attributes:
        arguments: :class:`SDPOConfig` instance.
        feedback_func: Optional callable returning per-completion feedback.
        teacher_prompt_length: Static length of the teacher prefix
            (``max_prompt_length + effective_feedback_length``).

    Example:
        >>> def my_feedback(prompts, completions, rewards):
        ...     return [f"Error: {c}" if r == 0 else "" for c, r in zip(completions, rewards)]
        ...
        >>> config = SDPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_generations=4,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     max_feedback_length=128,
        ...     distillation_type="jsd",
        ...     learning_rate=1e-6,
        ... )
        >>> trainer = SDPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=my_reward_fn,
        ...     feedback_func=my_feedback,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    arguments: SDPOConfig

    def __init__(
        self,
        arguments: SDPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
        feedback_func: FeedbackFunc | None = None,
    ):
        assert isinstance(arguments, SDPOConfig), f"arguments must be SDPOConfig, got {type(arguments)}"

        self.feedback_func = feedback_func
        self._effective_feedback_length = arguments.max_feedback_length
        self.teacher_prompt_length = arguments.max_prompt_length + self._effective_feedback_length

        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )
        self._configure_teacher_context()

    @staticmethod
    def _resolve_model_context_window(config: tp.Any) -> int | None:
        """Best-effort extraction of model context window from config."""
        candidates: list[int] = []
        for attr in (
            "granted_mask_max_position_embedding",
            "granted_freq_max_position_embedding",
            "mask_max_position_embeddings",
            "freq_max_position_embeddings",
            "max_position_embeddings",
            "max_context_length",
            "n_positions",
        ):
            value = getattr(config, attr, None)
            if isinstance(value, (int, np.integer)) and int(value) > 0:
                candidates.append(int(value))
        return min(candidates) if candidates else None

    def _configure_teacher_context(self) -> None:
        """Cap feedback tokens so prompt+feedback+completion fits model context."""
        base_len = int(self.arguments.max_prompt_length + self.arguments.max_completion_length)
        requested_feedback_len = int(self.arguments.max_feedback_length)
        requested_teacher_len = base_len + requested_feedback_len

        model_config = getattr(self.model_state.model, "config", None)
        model_context_window = (
            self._resolve_model_context_window(model_config) if model_config is not None else None
        )
        if model_context_window is None and self.arguments.max_length is not None:
            model_context_window = int(self.arguments.max_length)

        if model_context_window is None:
            teacher_total_len = requested_teacher_len
        else:
            if model_context_window < base_len:
                raise ValueError(
                    "SDPO requires context window >= prompt+completion "
                    f"({base_len}), but model supports {model_context_window}."
                )
            teacher_total_len = min(requested_teacher_len, model_context_window)

        self._effective_feedback_length = max(teacher_total_len - base_len, 0)
        self.teacher_prompt_length = self.arguments.max_prompt_length + self._effective_feedback_length

        if self._effective_feedback_length < requested_feedback_len:
            logger.warning(
                "Truncating SDPO feedback tokens from %d to %d to fit context window "
                "(prompt+completion=%d, model_context=%d).",
                requested_feedback_len,
                self._effective_feedback_length,
                base_len,
                model_context_window,
            )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure and JIT-compile the SDPO training / evaluation steps."""
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        shared_static = (
            self.num_generations,
            self.teacher_prompt_length,
            self.arguments.beta,
            self.arguments.distillation_type,
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )
        static_argnames = tuple(range(2, 12))

        self._train_shared_fn_static_args = (*shared_static, True, straight_through_emulator)

        sharded_training_step_function = ejit(
            sdpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (*shared_static, False, straight_through_emulator)

        sharded_evaluation_step_function = ejit(
            sdpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        def _compute_refmodel_logps(graphtree, graphother, ids, mask, graphdef):
            apply = flax.nnx.merge(graphdef, graphtree, graphother)
            with apply.mesh:
                ids = with_sharding_constraint(ids, self.arguments.step_partition_spec)
                mask = with_sharding_constraint(mask, self.arguments.step_partition_spec)
                return get_per_token_logps(apply, ids, mask, self.arguments.max_prompt_length)

        self.compute_refmodel_logps = ejit(
            partial(_compute_refmodel_logps, graphdef=self.model_state.graphdef),
            static_argnames=("graphdef",),
            in_shardings=(
                self.model_state.shardings.graphstate,
                self.model_state.shardings.graphother,
                empty_sharding,
                empty_sharding,
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

    def _get_self_feedback(
        self,
        completions: list[str],
        rewards: jax.Array,
        generation_factor: int,
    ) -> tuple[list[str], list[str | None]]:
        """Derive feedback without a rich-feedback function.

        For each rollout group the best-rewarded completion is used as the
        "correct solution" for the other attempts (Section 3 of the paper).

        Args:
            completions: Decoded completion strings (length ``B*G``).
            rewards: Per-completion scalar rewards, shape ``[B*G]``.
            generation_factor: Number of completions per prompt (``G``).

        Returns:
            ``(feedback_texts, env_feedbacks)`` - both lists of length ``B*G``.
            ``env_feedbacks`` is unused here (all empty strings) but kept for
            interface consistency with the rich-feedback path.
        """
        rewards_np = np.asarray(jax.device_get(rewards))
        n_prompts = len(completions) // generation_factor
        feedback_texts: list[str] = []
        env_feedbacks: list[str] = [""] * len(completions)

        for i in range(n_prompts):
            sl = slice(i * generation_factor, (i + 1) * generation_factor)
            group_completions = completions[sl]
            group_rewards = rewards_np[sl]

            best_idx = int(np.argmax(group_rewards))
            best_reward = group_rewards[best_idx]
            best_completion = group_completions[best_idx]
            correct_solution = best_completion if best_reward > 0 else None

            for j in range(generation_factor):
                is_successful = bool(group_rewards[j] > 0)
                sep = _build_feedback_separator(
                    is_successful=is_successful,
                    env_feedback="",
                    correct_solution=correct_solution if not is_successful else None,
                )
                feedback_texts.append(sep)

        return feedback_texts, env_feedbacks

    def _get_rich_feedback(
        self,
        completion_prompts: list,
        completions: list,
        rewards: jax.Array,
    ) -> list[str]:
        """Get feedback strings from the user-supplied ``feedback_func``.

        Args:
            completion_prompts: Prompt texts (or conversation dicts) repeated
                ``G`` times (one per completion).
            completions: Decoded completion strings.
            rewards: Per-completion scalar rewards.

        Returns:
            List of feedback separator strings, one per completion.
        """
        rewards_list = [float(r) for r in jax.device_get(rewards)]
        raw_feedbacks = self.feedback_func(
            prompts=(
                completion_prompts if isinstance(completion_prompts[0], str) else [str(p) for p in completion_prompts]
            ),
            completions=completions if isinstance(completions[0], str) else [str(c) for c in completions],
            rewards=rewards_list,
        )
        feedback_texts = []
        for raw_fb, reward in zip(raw_feedbacks, rewards_list, strict=False):
            is_successful = reward > 0
            sep = _build_feedback_separator(
                is_successful=is_successful,
                env_feedback=raw_fb or "",
                correct_solution=None,
            )
            feedback_texts.append(sep)
        return feedback_texts

    def _tokenize_feedback_separators(
        self,
        feedback_texts: list[str],
    ) -> tuple[jax.Array, jax.Array]:
        """Tokenise feedback separator strings and pad to effective feedback length.

        Args:
            feedback_texts: List of ``B*G`` feedback separator strings.

        Returns:
            ``(feedback_ids, feedback_mask)`` both ``[B*G, effective_feedback_length]``.
        """
        if self._effective_feedback_length <= 0:
            batch_size = len(feedback_texts)
            empty = jnp.zeros((batch_size, 0), dtype=jnp.int32)
            return empty, empty

        enc = self.processing_class(
            feedback_texts,
            add_special_tokens=False,
            max_length=self._effective_feedback_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return jnp.asarray(enc["input_ids"]), jnp.asarray(enc["attention_mask"])

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Build the SDPO batch from raw prompts.

        Steps:
        1. Generate ``G`` completions per prompt (identical to GRPO).
        2. Evaluate rewards from ``reward_funcs`` (for logging + feedback).
        3. Build feedback separator texts (rich or self-feedback).
        4. Tokenise feedback separators → ``feedback_ids`` / ``feedback_mask``.
        5. Construct teacher context:
           ``teacher_ids = [prompt_ids | feedback_ids | completion_ids]``
        6. Optionally compute ref-model log-probs when ``beta > 0``.
        7. Return the batch dict consumed by :func:`sdpo_step`.
        """
        batch = self._purify_batch(batch)

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
            ridmask = prompt_mask.repeat(generation_factor, 0)

            # HF tokenizers expect Python / NumPy token ids; JAX arrays can fail in Rust decode bindings.
            completion_ids_for_decode = np.asarray(jax.device_get(completion_ids)).tolist()
            completions_text = self.processing_class.batch_decode(completion_ids_for_decode, skip_special_tokens=True)
            is_conv = self.train_is_conversational if is_train else self.eval_is_conversational
            if is_conv:
                completions = [[{"role": "assistant", "content": c}] for c in completions_text]
            else:
                completions = completions_text

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
                        if is_conv:
                            messages = [
                                {"messages": p + c} for p, c in zip(completion_prompts, completions, strict=False)
                            ]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(completion_prompts, completions, strict=False)]
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
                        output_reward_func = reward_func(
                            prompts=completion_prompts,
                            completions=completions,
                            max_length=self.arguments.max_length,
                            batch=batch,
                        )
                        rew = jnp.array(
                            [v if v is not None else jnp.nan for v in output_reward_func],
                            dtype="f4",
                        )
                    rewards_per_func = rewards_per_func.at[:, i].set(rew.reshape(-1))
            rewarding_time = rewarding_time_fn()

            rewards = jnp.nansum(rewards_per_func * self.reward_weights[None, :], axis=1)

            if self.feedback_func is not None:
                feedback_texts = self._get_rich_feedback(
                    completion_prompts=completion_prompts,
                    completions=completions_text,
                    rewards=rewards,
                )
            else:
                feedback_texts, _ = self._get_self_feedback(
                    completions=completions_text,
                    rewards=rewards,
                    generation_factor=generation_factor,
                )

            with capture_time() as teacher_build_time_fn:
                feedback_ids, feedback_mask = self._tokenize_feedback_separators(feedback_texts)

                rids = prompt_ids.repeat(generation_factor, 0)

                teacher_ids = jnp.concatenate([rids, feedback_ids, completion_ids], axis=1)
                teacher_mask = jnp.concatenate([ridmask, feedback_mask, completion_mask], axis=1)
            teacher_build_time = teacher_build_time_fn()

            prompt_completion_ids = sequences
            with capture_time() as token_logps_time_fn:
                if self.arguments.beta != 0.0:
                    ref_per_token_logps = self.compute_refmodel_logps(
                        self.ref_state.graphstate,
                        self.ref_state.graphother,
                        prompt_completion_ids,
                        jnp.concatenate([ridmask, completion_mask], -1),
                    )
                else:
                    ref_per_token_logps = None
            token_logps_time = token_logps_time_fn()

            prompt_ids = self._all_gather(prompt_ids)
            prompt_mask = self._all_gather(prompt_mask)
            completion_ids = self._all_gather(completion_ids)
            completion_mask = self._all_gather(completion_mask)
            teacher_ids = self._all_gather(teacher_ids)
            teacher_mask = self._all_gather(teacher_mask)
            rewards_per_func = self._all_gather(rewards_per_func)
            if ref_per_token_logps is not None:
                ref_per_token_logps = self._all_gather(ref_per_token_logps)

        preprocessing_time = preprocessing_time_fn()
        completion_length = jnp.sum(completion_mask, -1)

        metrics_dict: dict[str, float | int | str] = {
            "reward_mean": float(jnp.nanmean(rewards)),
            "reward_std": float(jnp.nanstd(rewards)),
            "completion_length": float(jnp.mean(completion_length)),
            "generation_time": generation_time,
            "rewarding_time": rewarding_time,
            "teacher_build_time": teacher_build_time,
            "token_logps_time": token_logps_time,
            "preprocessing_time": preprocessing_time,
        }
        for i, name in enumerate(self.reward_func_names):
            metrics_dict[name] = float(jnp.nanmean(rewards_per_func[:, i]))

        out_batch: dict[str, jax.Array] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_ids": teacher_ids,
            "teacher_mask": teacher_mask,
            "num_items_in_batch": jnp.sum(completion_mask),
        }
        if ref_per_token_logps is not None:
            out_batch["ref_per_token_logps"] = ref_per_token_logps

        return out_batch, metrics_dict

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Post-step hook - syncs reference model when requested."""
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
