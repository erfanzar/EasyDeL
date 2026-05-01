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
"""Configuration dataclass for the GRPO trainer.

Group Relative Policy Optimization (DeepSeek, 2024) replaces the
critic of PPO with group-relative advantage normalization: rewards
inside each prompt group are mean-centred and (optionally) standardised
to provide a low-variance learning signal.  :class:`GRPOConfig`
holds the temperature, KL penalty, group / generation sizes, the chunk
sizes for memory efficiency, and the reference-model handling knobs.
"""

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "grpo")
@dataclass
class GRPOConfig(TrainingArguments):
    """Configuration class for Group Relative Policy Optimization (GRPO) training.

    GRPO (DeepSeekMath, Shao et al. 2024) replaces PPO's value-network
    baseline with a *group-relative* advantage: for each prompt the
    policy samples ``num_generations`` completions, scores them with
    one or more reward functions, and standardises the rewards within
    the group (mean / std). The standardised rewards become the
    advantages that drive a clipped-PPO-style policy gradient,
    optionally regularised by a KL penalty against a reference model.

    This trainer supports several GRPO loss variants exposed via
    ``loss_type``: the canonical ``"grpo"``, the
    batch-normalised-policy-gradient ``"bnpo"``, the unbiased
    ``"dr_grpo"``, the dynamic-clipping ``"dapo"`` (default in EasyDeL),
    and the constant-importance-sampling ``"cispo"``. Importance
    sampling can be applied at the token level (``"token"``) or
    aggregated per sequence (``"sequence"``).

    Construct with dict-literal kwargs, e.g.:

    >>> cfg = GRPOConfig(num_generations=8, beta=0.04, loss_type="dapo",
    ...                  max_prompt_length=512, max_completion_length=256)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"GRPO"``).
        remove_unused_columns: When ``False``, dataset columns are kept
            so reward functions can read auxiliary fields.
        max_prompt_length: Maximum prompt-only token budget. Default
            ``512``.
        max_completion_length: Maximum completion token budget. Default
            ``256``. Must satisfy
            ``max_prompt_length + max_completion_length <= max_length``.
        dataset_num_proc: Worker count for ``Dataset.map`` calls.
        learning_rate: Optimizer learning rate. Default ``1e-6``.
        beta: KL-regularisation coefficient against the reference
            model. ``0.0`` disables the KL term entirely.
        epsilon: Lower (and default upper) clipping bound for
            importance-sampling weights, mirroring PPO.
        epsilon_high: Optional asymmetric upper clip; falls back to
            ``epsilon`` when ``None``.
        delta: Optional two-sided dynamic clipping bound (DAPO).
            ``None`` disables dynamic clipping.
        sync_ref_model: When ``True``, the reference model is
            periodically refreshed from a moving average of the
            policy.
        ref_model_mixup_alpha: Polyak mixing coefficient used when
            syncing the reference (``new_ref = alpha * ref + (1 - alpha) * policy``).
        ref_model_sync_steps: Optimizer-step interval between
            reference syncs.
        num_iterations: Number of optimizer updates per generated
            batch (PPO-style multi-epoch updates).
        loss_type: One of ``"grpo"``, ``"bnpo"``, ``"dr_grpo"``,
            ``"dapo"``, ``"cispo"``. Default ``"dapo"``.
        importance_sampling_level: ``"token"`` or ``"sequence"``.
        reward_weights: Optional weights for combining multiple reward
            functions. Length must match the reward-function list.
        scale_rewards: Reward scaling strategy: ``"group"`` (default),
            ``"batch"``, ``"none"``. ``True``/``False`` are accepted
            and mapped to ``"group"``/``"none"``.
        tools: Optional tool registry forwarded to the reward
            functions.
        skip_apply_chat_template: When ``True``, the prompt is taken
            verbatim from the dataset (no chat-template application).
        num_return_sequences: Number of completions to sample per
            prompt. Mirrored on ``num_generations`` for TRL parity.
        num_generations: Alias of ``num_return_sequences`` (kept for
            TRL compatibility); both fields are kept in sync after
            ``__post_init__``.
        temperature: Sampling temperature.
        top_p, top_k, presence_penalty, frequency_penalty, min_p,
            repetition_penalty: Standard generation knobs.
        generation_kwargs: Extra kwargs forwarded to the generation
            engine.
        chat_template_kwargs: Extra kwargs for chat-template
            application during generation.
        mask_truncated_completions: When ``True``, completions that
            did not terminate with EOS are dropped from the loss to
            avoid biasing the gradient toward truncated trajectories.
        top_entropy_quantile: Keeps only the top fraction (by token
            entropy) of completion tokens in the loss. ``1.0``
            disables filtering.
        ref_logps_chunk_size: Sequence-axis chunk size for the
            reference-model log-prob forward. ``None`` disables
            chunking.
        completion_chunk_size: Sequence-axis chunk size for the
            policy completion-loss computation. ``None`` disables
            chunking.
        max_loss_completion_tokens: Optional cap on the number of
            completion tokens contributing to the loss.
        logprob_vocab_chunk_size: Vocab-axis chunk size for
            :func:`compute_token_logps_and_entropies_chunked` when
            scoring completions. ``None`` disables chunking.
    """

    trainer_prefix: str | None = field(
        default="GRPO",
        metadata={"help": "default prefix name for trainer."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "The maximum length of the completion."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for dataset processing."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The learning rate."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "The beta parameter for GRPO."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower clipping bound for importance sampling weights."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={"help": "Upper clipping bound for importance sampling weights. If None, defaults to `epsilon`."},
    )
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Optional two-sided clipping bound. If set, importance weights are additionally clipped to `delta`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync the reference model with the policy model."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "The alpha parameter for mixing the reference model with the policy model."},
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={"help": "The number of steps between syncing the reference model."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "How many optimizer updates to perform per generated batch."},
    )
    loss_type: str = field(
        default="dapo",
        metadata={"help": "Loss variant to use. One of ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo']."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={"help": "Importance sampling applied per 'token' or aggregated per 'sequence'."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Optional weights for each reward function. Must match the number of reward functions if set."
        },
    )
    scale_rewards: str | bool = field(
        default="group",
        metadata={
            "help": "Reward scaling strategy: 'group', 'batch', 'none', or the booleans True/False for group/none."
        },
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for training."},
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "whenever to skip extracting prompt from dataset."},
    )
    num_return_sequences: int = field(
        default=4,
        metadata={
            "help": (
                "The number of sequences to return for each input prompt. Used during sampling to "
                "generate multiple completions per prompt."
            )
        },
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Alias for num_return_sequences to keep parity with TRL's interface."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature used during generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p nucleus sampling parameter."},
    )
    top_k: int | None = field(
        default=None,
        metadata={"help": "Top-k sampling parameter. None disables top-k."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty applied during generation."},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty applied during generation."},
    )
    min_p: float | None = field(
        default=None,
        metadata={"help": "Minimum token probability threshold (see HF top-p-min sampling)."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty applied during generation."},
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Additional generation kwargs forwarded to the generation config."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application during generation."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "If True, drop completions that do not terminate with EOS from the loss calculation."},
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Keep only the top quantile of tokens by entropy in the loss (1.0 disables filtering)."},
    )
    ref_logps_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Chunk size for reference-model log-prob computation. "
                "Set to `None` to disable chunking; `0` is accepted for backward compatibility."
            )
        },
    )
    completion_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Chunk size for completion-loss computation. "
                "Set to `None` to disable chunked completion loss; `0` is accepted for backward compatibility."
            )
        },
    )
    max_loss_completion_tokens: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional cap on completion tokens used by the GRPO loss. "
                "Set to `None` to disable truncation; `0` is accepted for backward compatibility."
            )
        },
    )
    logprob_vocab_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Vocabulary chunk size used when computing per-token log probabilities and entropies. "
                "Set to `None` to disable chunking."
            )
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize GRPO-specific config invariants.

        Resolves the legacy ``max_sequence_length`` alias, derives
        ``max_completion_length`` from ``max_length`` /
        ``max_prompt_length`` when left at the class default, ensures
        ``max_length == max_prompt_length + max_completion_length``,
        keeps ``num_generations`` and ``num_return_sequences`` in
        sync, copies ``temperature`` into ``generation_temperature``
        when not set, defaults ``epsilon_high`` to ``epsilon``,
        normalises the various chunk-size aliases (``0`` -> ``None``)
        and converts ``scale_rewards`` boolean shorthands to their
        canonical string values. Finally defers to the base
        :class:`TrainingArguments.__post_init__`.

        Args:
            max_sequence_length: Legacy alias for ``max_length``.
            quantization_block: Legacy alias for the quantization group
                size; forwarded to the base class.

        Raises:
            ValueError: If ``max_length`` is smaller than
                ``max_prompt_length`` or
                ``max_prompt_length + max_completion_length``.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)

        default_completion = type(self).__dataclass_fields__["max_completion_length"].default
        if self.max_length is not None:
            if self.max_length < self.max_prompt_length:
                raise ValueError(
                    f"`max_length` ({self.max_length}) must be >= `max_prompt_length` ({self.max_prompt_length})."
                )
            max_allowed_completion = self.max_length - self.max_prompt_length

            # Keep legacy behavior when completion length is left at class default:
            # infer completion from max_length and max_prompt_length.
            if self.max_completion_length == default_completion:
                self.max_completion_length = max_allowed_completion
            elif self.max_completion_length > max_allowed_completion:
                raise ValueError(
                    "`max_prompt_length + max_completion_length` "
                    f"({self.max_prompt_length} + {self.max_completion_length}) must be <= `max_length` "
                    f"({self.max_length})."
                )

        self.max_length = self.max_prompt_length + self.max_completion_length

        if self.num_generations is None:
            self.num_generations = self.num_return_sequences
        else:
            self.num_return_sequences = self.num_generations
        if self.generation_temperature is None:
            self.generation_temperature = self.temperature

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        if self.scale_rewards is True:
            self.scale_rewards = "group"
        elif self.scale_rewards is False:
            self.scale_rewards = "none"

        if self.ref_logps_chunk_size is not None:
            normalized_ref_chunk_size = int(self.ref_logps_chunk_size)
            self.ref_logps_chunk_size = normalized_ref_chunk_size if normalized_ref_chunk_size > 0 else None
        if self.completion_chunk_size is not None:
            normalized_completion_chunk_size = int(self.completion_chunk_size)
            self.completion_chunk_size = (
                normalized_completion_chunk_size if normalized_completion_chunk_size > 0 else None
            )
        if self.max_loss_completion_tokens is not None:
            normalized_max_loss_completion_tokens = int(self.max_loss_completion_tokens)
            self.max_loss_completion_tokens = (
                normalized_max_loss_completion_tokens if normalized_max_loss_completion_tokens > 0 else None
            )
        if self.logprob_vocab_chunk_size is not None:
            normalized_chunk_size = int(self.logprob_vocab_chunk_size)
            self.logprob_vocab_chunk_size = normalized_chunk_size if normalized_chunk_size > 0 else None

        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
