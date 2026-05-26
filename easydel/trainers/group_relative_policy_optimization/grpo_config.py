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
    the constant-importance-sampling ``"cispo"``, the
    soft-advantage-policy-optimization ``"sapo"``, the
    length-unbiased sequence ``"luspo"``, and the variational sequence
    soft-policy ``"vespo"`` variants. Importance
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
        shuffle_dataset: TRL-compatible alias for
            ``shuffle_train_dataset``. ``None`` leaves the base setting
            unchanged.
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
        generation_batch_size: Optional effective generation batch
            size. When set, it is converted into
            ``steps_per_generation`` using ``total_batch_size``.
        steps_per_generation: Number of optimizer steps that reuse a
            generated/scored GRPO batch before sampling again.
        loss_type: One of ``"grpo"``, ``"bnpo"``, ``"dr_grpo"``,
            ``"dapo"``, ``"cispo"``, ``"sapo"``, ``"luspo"``,
            ``"vespo"``. Default ``"dapo"``.
        disable_dropout: Whether to put policy, reference, and reward
            model modules into eval mode during trainer construction.
        sapo_temperature_pos: Positive-advantage soft-clipping
            temperature used by ``loss_type="sapo"``.
        sapo_temperature_neg: Negative-advantage soft-clipping
            temperature used by ``loss_type="sapo"``.
        vespo_k_pos: Gamma-weight exponent for positive VESPO advantages.
        vespo_lambda_pos: Gamma-weight decay for positive VESPO advantages.
        vespo_k_neg: Gamma-weight exponent for negative VESPO advantages.
        vespo_lambda_neg: Gamma-weight decay for negative VESPO advantages.
        importance_sampling_level: ``"token"`` or ``"sequence"``.
        reward_weights: Optional weights for combining multiple reward
            functions. Length must match the reward-function list.
        multi_objective_aggregation: Reward aggregation strategy when
            multiple reward functions are configured.
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
        num_generations_eval: Number of completions to sample per
            prompt during evaluation. ``None`` reuses
            ``num_generations``.
        temperature: Sampling temperature.
        top_p, top_k, presence_penalty, frequency_penalty,
            repetition_penalty: Standard eSurge generation knobs.
        chat_template_kwargs: Extra kwargs for chat-template
            application during generation.
        pad_to_multiple_of: If set, prompt batches are padded to a
            multiple of this value while still truncating prompt content
            to ``max_prompt_length``.
        mask_truncated_completions: When ``True``, completions that
            did not terminate with EOS are dropped from the loss to
            avoid biasing the gradient toward truncated trajectories.
        top_entropy_quantile: Keeps only the top fraction (by token
            entropy) of completion tokens in the loss. ``1.0``
            disables filtering.
        off_policy_mask_threshold: Optional sequence-level forward-KL
            threshold for dropping negative-advantage off-policy
            samples from the policy objective.
        use_bias_correction_kl: When ``True``, multiply the KL penalty
            by the same importance-sampling coefficient used by the
            policy objective.
        log_completions: When ``True``, log a small sample of generated
            prompt/completion pairs through the trainer logger.
        num_completions_to_print: Optional cap on the number of
            completion rows emitted by ``log_completions``.
        log_unique_prompts: When logging completions, keep only the
            first completion for each distinct prompt.
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
    shuffle_dataset: bool | None = field(
        default=None,
        metadata={"help": "TRL-compatible alias for `shuffle_train_dataset`."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The learning rate."},
    )
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": (
                "TRL compatibility field. EasyDeL keeps model parameter dtypes under the model/sharding config, "
                "so this flag does not require a trainer-side cast path."
            )
        },
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
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Effective batch size used for generation. Mutually exclusive with `steps_per_generation`; "
                "when set, EasyDeL derives `steps_per_generation = generation_batch_size // total_batch_size`."
            )
        },
    )
    steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Number of optimizer steps to reuse a generated GRPO batch before sampling again."},
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Loss variant to use. One of ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo', 'luspo', 'vespo']."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the policy, reference, and reward models."},
    )
    sapo_temperature_pos: float = field(
        default=1.0,
        metadata={"help": "Positive-advantage soft-clipping temperature for loss_type='sapo'."},
    )
    sapo_temperature_neg: float = field(
        default=1.05,
        metadata={"help": "Negative-advantage soft-clipping temperature for loss_type='sapo'."},
    )
    vespo_k_pos: float = field(
        default=2.0,
        metadata={"help": "VESPO gamma-weight exponent for positive advantages."},
    )
    vespo_lambda_pos: float = field(
        default=3.0,
        metadata={"help": "VESPO gamma-weight decay for positive advantages."},
    )
    vespo_k_neg: float = field(
        default=3.0,
        metadata={"help": "VESPO gamma-weight exponent for negative advantages."},
    )
    vespo_lambda_neg: float = field(
        default=2.0,
        metadata={"help": "VESPO gamma-weight decay for negative advantages."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": (
                "Importance sampling applied per 'token', aggregated per 'sequence', or GSPO-token 'sequence_token'."
            )
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Optional weights for each reward function. Must match the number of reward functions if set."
        },
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={
            "help": (
                "Multi-reward aggregation strategy. 'sum_then_normalize' first sums weighted reward functions and "
                "then normalizes. 'normalize_then_sum' normalizes each reward function within prompt groups before "
                "weighted summation and batch-level advantage normalization."
            )
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
    environment_factory: tp.Callable[[], object] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional factory that creates an agentic environment for each generated completion. "
                "When set, GRPO steps the environment with the generated completion and exposes the "
                "environment feedback to reward functions."
            )
        },
    )
    tool_caller: str | None = field(
        default=None,
        metadata={
            "help": (
                "Tool call parser used with environment_factory/tools. Either a registered inference parser name "
                "or a regex pattern prefixed with 'regex:'."
            )
        },
    )
    max_tool_calls_per_step: int = field(
        default=5,
        metadata={"help": "Maximum number of tool calls executed during one environment step."},
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of GRPO tool-call regenerate iterations. None means keep iterating until no tool "
                "calls remain or the completion token budget is exhausted; 0 disables the loop."
            )
        },
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
    num_generations_eval: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of generations per prompt during evaluation. None uses the training `num_generations` value."
            )
        },
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
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty applied during generation."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application during generation."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, GRPO prompt batches are padded to a multiple of this value."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "If True, drop completions that do not terminate with EOS from the loss calculation."},
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Keep only the top quantile of tokens by entropy in the loss (1.0 disables filtering)."},
    )
    off_policy_mask_threshold: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional off-policy sequence mask threshold. Negative-advantage samples with mean forward KL above "
                "this value are removed from the policy objective."
            )
        },
    )
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={"help": "Whether to importance-weight the KL penalty for bias correction."},
    )
    esurge_importance_sampling_correction: bool = field(
        default=True,
        metadata={"help": "eSurge importance-sampling correction toggle."},
    )
    esurge_importance_sampling_mode: str = field(
        default="sequence_mask",
        metadata={"help": "eSurge importance-sampling mode."},
    )
    esurge_importance_sampling_cap: float = field(
        default=3.0,
        metadata={"help": "eSurge importance-sampling cap."},
    )
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log sampled prompt/completion pairs during GRPO preprocessing."},
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Maximum number of completion rows to print when log_completions=True."},
    )
    log_unique_prompts: bool = field(
        default=False,
        metadata={"help": "When logging completions, keep only the first completion for each unique prompt."},
    )
    log_completions_hub_repo: str | None = field(
        default=None,
        metadata={"help": "Hugging Face Hub repository for completion logs. Upload is not supported yet."},
    )
    use_transformers_paged: bool = field(
        default=False,
        metadata={"help": "Deprecated TRL paged-generation flag. Not supported by EasyDeL GRPO."},
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
        if self.log_completions_hub_repo is not None:
            if not self.log_completions:
                raise ValueError("`log_completions_hub_repo` requires `log_completions=True`.")
        if self.shuffle_dataset is not None:
            self.shuffle_train_dataset = bool(self.shuffle_dataset)

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
        if self.num_generations_eval is not None and self.num_generations_eval <= 0:
            raise ValueError("`num_generations_eval` must be a positive integer when set.")
        if self.num_iterations <= 0:
            raise ValueError("`num_iterations` must be a positive integer.")
        if not (0.0 <= self.ref_model_mixup_alpha <= 1.0):
            raise ValueError(f"`ref_model_mixup_alpha` must be in [0.0, 1.0]; got {self.ref_model_mixup_alpha}.")
        if self.ref_model_sync_steps <= 0:
            raise ValueError(f"`ref_model_sync_steps` must be positive; got {self.ref_model_sync_steps}.")
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            raise ValueError("`generation_batch_size` and `steps_per_generation` are mutually exclusive.")
        if self.generation_batch_size is not None:
            if self.generation_batch_size <= 0:
                raise ValueError("`generation_batch_size` must be a positive integer when set.")
            if self.generation_batch_size % self.total_batch_size != 0:
                raise ValueError("`generation_batch_size` must be divisible by `total_batch_size` in EasyDeL.")
            self.steps_per_generation = max(1, self.generation_batch_size // self.total_batch_size)
        elif self.steps_per_generation is None:
            self.steps_per_generation = 1
        elif self.steps_per_generation <= 0:
            raise ValueError("`steps_per_generation` must be a positive integer when set.")
        self.generation_batch_size = self.total_batch_size * self.steps_per_generation
        if self.generation_temperature is None:
            self.generation_temperature = self.temperature

        self.loss_type = self.loss_type.lower() if isinstance(self.loss_type, str) else self.loss_type
        if self.loss_type not in {"grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo", "dppo"}:
            raise ValueError(
                "`loss_type` must be one of 'grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo', "
                "'luspo', 'vespo', or 'dppo'. "
                f"Got {self.loss_type!r}."
            )
        if self.sapo_temperature_pos <= 0.0:
            raise ValueError(f"`sapo_temperature_pos` must be positive; got {self.sapo_temperature_pos}.")
        if self.sapo_temperature_neg <= 0.0:
            raise ValueError(f"`sapo_temperature_neg` must be positive; got {self.sapo_temperature_neg}.")
        if self.vespo_k_pos <= 0.0:
            raise ValueError(f"`vespo_k_pos` must be positive; got {self.vespo_k_pos}.")
        if self.vespo_lambda_pos <= 0.0:
            raise ValueError(f"`vespo_lambda_pos` must be positive; got {self.vespo_lambda_pos}.")
        if self.vespo_k_neg <= 0.0:
            raise ValueError(f"`vespo_k_neg` must be positive; got {self.vespo_k_neg}.")
        if self.vespo_lambda_neg <= 0.0:
            raise ValueError(f"`vespo_lambda_neg` must be positive; got {self.vespo_lambda_neg}.")
        if self.max_tool_calls_per_step <= 0:
            raise ValueError("`max_tool_calls_per_step` must be positive.")
        if self.max_tool_calling_iterations is not None and self.max_tool_calling_iterations < 0:
            raise ValueError("`max_tool_calling_iterations` must be non-negative or None.")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("`pad_to_multiple_of` must be a positive integer when set.")
        if not (0.0 <= self.top_entropy_quantile <= 1.0):
            raise ValueError("`top_entropy_quantile` must be in [0.0, 1.0].")
        if self.off_policy_mask_threshold is not None and self.off_policy_mask_threshold <= 0.0:
            raise ValueError("`off_policy_mask_threshold` must be positive when set.")
        if self.esurge_importance_sampling_mode not in {
            "token_truncate",
            "token_mask",
            "sequence_truncate",
            "sequence_mask",
        }:
            raise ValueError(
                "`esurge_importance_sampling_mode` must be one of 'token_truncate', 'token_mask', "
                "'sequence_truncate', or 'sequence_mask'."
            )
        if self.esurge_importance_sampling_cap <= 0.0:
            raise ValueError("`esurge_importance_sampling_cap` must be positive.")
        if self.num_completions_to_print is not None and self.num_completions_to_print <= 0:
            raise ValueError("`num_completions_to_print` must be a positive integer when set.")

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        if self.scale_rewards is True:
            self.scale_rewards = "group"
        elif self.scale_rewards is False:
            self.scale_rewards = "none"
        if isinstance(self.multi_objective_aggregation, str):
            self.multi_objective_aggregation = self.multi_objective_aggregation.lower()
        if self.multi_objective_aggregation not in {"sum_then_normalize", "normalize_then_sum"}:
            raise ValueError(
                "`multi_objective_aggregation` must be 'sum_then_normalize' or 'normalize_then_sum'. "
                f"Got {self.multi_objective_aggregation!r}."
            )

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
