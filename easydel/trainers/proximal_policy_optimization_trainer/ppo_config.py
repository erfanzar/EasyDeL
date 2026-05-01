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

"""Configuration classes for PPO training.

This module defines PPOConfig, a dataclass that holds all hyperparameters
for Proximal Policy Optimization training including:
- Prompt/completion length limits
- PPO-specific parameters (clip range, KL coefficient, GAE lambda)
- Value function training parameters
- Generation sampling parameters
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "ppo")
@dataclass
class PPOConfig(TrainingArguments):
    """Hyperparameters for RLHF-style Proximal Policy Optimization.

    PPO optimises a stochastic policy ``pi_theta`` against advantages
    estimated from a learned value function ``V_phi`` while constraining
    each update to a *proximal* region of the rollout policy
    ``pi_old``. The clipped surrogate objective minimised here is

    ``L_pi = E_t[ min( r_t * A_t, clip(r_t, 1 - cliprange, 1 + cliprange) * A_t ) ]``

    where ``r_t = exp(logp_theta(a_t|s_t) - logp_old(a_t|s_t))`` is the
    importance ratio. The total loss adds a clipped value-function
    regression ``vf_coef * L_v`` (see ``cliprange_value``) and an
    optional entropy bonus controlled by ``entropy_coef``. Per-token
    rewards combine an external reward (typically a reward model
    scoring the *whole* completion) with a token-wise KL penalty
    ``-kl_coef * KL(pi_theta || pi_ref)`` against a frozen reference
    policy; advantages are computed via Generalised Advantage Estimation
    with discount ``gamma`` and trace-decay ``lam``. Rollouts are
    generated outside the gradient computation with the sampling
    parameters below and replayed for ``num_ppo_epochs`` minibatch
    epochs.

    Attributes:
        trainer_prefix: Prefix used when naming logs / checkpoints / W&B
            runs. Default: ``"PPO"``.
        remove_unused_columns: Whether the base trainer should drop
            columns from the dataset that are not consumed by PPO.
            Default ``False`` to preserve any side-channel metadata.
        max_prompt_length: Maximum prompt length in tokens; prompts are
            left-padded so generation always sees a right-aligned prefix.
        max_completion_length: Maximum number of new tokens generated
            per rollout sample; right-padded.
        dataset_num_proc: Number of worker processes used by the
            preprocessing transform. ``None`` runs sequentially.
        learning_rate: Optimiser step size for the joint
            policy + value-head update.
        num_ppo_epochs: Minibatch epochs reused per rollout batch.
            Currently best-effort -- keep at ``1`` for parity with the
            v1 reference implementation.
        kl_coef: Coefficient on the per-token KL penalty added to the
            score-only reward, i.e. ``r_kl_t = -kl_coef * KL_t``.
        kl_estimator: Choice of KL estimator: ``"k1"`` uses the
            unbiased single-sample estimator
            ``logp - logp_ref``; ``"k3"`` uses the variance-reduced
            estimator ``exp(logp_ref - logp) - 1 - (logp_ref - logp)``.
        cliprange: PPO policy-ratio clip parameter ``epsilon`` used in
            the clipped surrogate.
        vf_coef: Scalar weight on the value-function loss in the joint
            objective.
        cliprange_value: Symmetric clip range for the value function
            update, applied as
            ``V_clipped = V_old + clip(V_phi - V_old, -range, +range)``.
        gamma: Reward discount factor used by GAE.
        lam: GAE trace-decay parameter.
        whiten_rewards: If ``True``, normalise per-token rewards to
            zero-mean / unit-variance before running GAE.
        whiten_advantages: If ``True``, normalise the GAE advantages
            after computation. Strongly recommended.
        entropy_coef: Optional entropy-regularisation weight; ``None`` or
            non-positive disables the bonus, otherwise the loss adds
            ``-entropy_coef * H[pi_theta]``.
        missing_eos_penalty: Optional penalty subtracted from the score
            of completions that fail to emit an EOS token before the
            length cap.
        tools: Tool/function-calling schemas forwarded to the chat
            template during prompt assembly.
        reward_weights: Per-reward-function weights used when several
            reward callables are combined into a scalar score; length
            must match the number of reward functions.
        skip_apply_chat_template: When ``True`` the dataset is treated
            as already chat-templated and only tokenisation is applied.
        num_return_sequences: Number of completions sampled per prompt
            during rollout.
        num_generations: Alias for ``num_return_sequences`` kept for
            cross-trainer parity; the two are reconciled in
            ``__post_init__``.
        temperature: Rollout sampling temperature.
        top_p: Nucleus-sampling cumulative-probability cutoff.
        top_k: Optional top-k sampling cutoff (``None`` disables top-k).
        presence_penalty: Per-token presence penalty applied during
            sampling.
        frequency_penalty: Per-token frequency penalty applied during
            sampling.
        min_p: Optional minimum-probability filter (HF "top-p-min").
        repetition_penalty: Multiplicative repetition penalty applied
            to already-generated tokens.
        generation_kwargs: Extra keyword arguments forwarded verbatim to
            the underlying generation config.
        chat_template_kwargs: Extra keyword arguments forwarded to chat
            template rendering during prompt assembly.
        mask_truncated_completions: When ``True``, completions that hit
            ``max_completion_length`` without emitting EOS are masked
            out of the loss.
        logprob_vocab_chunk_size: Optional vocabulary chunking for the
            log-probability / entropy reductions used by the loss; set
            to ``None`` (or non-positive) to disable chunking.
    """

    trainer_prefix: str | None = field(
        default="PPO",
        metadata={"help": "Default prefix name for trainer outputs/checkpoints."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt (left-padded)."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "The maximum length of the completion (right-padded)."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for PPO fine-tuning."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of PPO epochs per rollout batch. (Currently best-effort; keep 1 for parity.)"},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient for the non-score reward: r_kl = -kl_coef * KL(pi||ref)."},
    )
    kl_estimator: tp.Literal["k1", "k3"] = field(
        default="k1",
        metadata={"help": "KL estimator to use ('k1' or 'k3')."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clip range for the policy objective."},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value-function loss coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Value-function clip range."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor for GAE/returns."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda for GAE."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten (scale) token rewards before GAE."},
    )
    whiten_advantages: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages to zero mean / unit variance."},
    )
    entropy_coef: float | None = field(
        default=None,
        metadata={
            "help": "Optional entropy bonus coefficient. Set to `None` to disable; `0` is accepted for backward compatibility."
        },
    )
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={"help": "Optional penalty subtracted from score when no EOS is generated."},
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for chat-template function calling."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Optional weights for each reward function. Must match the number of reward functions if set."
        },
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "If True, skip extracting prompt from dataset via chat template."},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of completions per prompt during generation."},
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Alias for num_return_sequences to keep parity with other trainers."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for rollout generation."},
    )
    top_p: float = field(
        default=0.95,
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
        metadata={"help": "Additional generation kwargs forwarded to generation config."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application during generation."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "If True, drop completions that do not terminate with EOS from loss calculation."},
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
        """Validate and reconcile PPO-specific length and sampling parameters.

        - Forwards a deprecated ``max_sequence_length`` to the base class.
        - Reconciles ``max_length`` with ``max_prompt_length`` /
          ``max_completion_length``.
        - Mirrors ``num_return_sequences`` and ``num_generations``.
        - Synchronizes ``generation_temperature`` with the rollout
          ``temperature`` when not explicitly set.
        - Clamps ``entropy_coef`` to ``None`` when non-positive.
        - Normalizes ``logprob_vocab_chunk_size`` to ``None`` when not
          positive.

        Args:
            max_sequence_length (int | None): Deprecated alias for
                ``max_length``; forwarded to the base class.
            quantization_block (int | None): Optional quantization block
                size forwarded to the base class.

        Raises:
            ValueError: If ``max_length`` is smaller than
                ``max_prompt_length``.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)

        if self.max_length is not None:
            if self.max_length < self.max_prompt_length:
                raise ValueError(
                    f"`max_length` ({self.max_length}) must be >= `max_prompt_length` ({self.max_prompt_length})."
                )
            self.max_completion_length = self.max_length - self.max_prompt_length

        self.max_length = self.max_prompt_length + self.max_completion_length

        if self.num_generations is None:
            self.num_generations = self.num_return_sequences
        else:
            self.num_return_sequences = self.num_generations
        if self.generation_temperature is None:
            self.generation_temperature = self.temperature
        if self.entropy_coef is not None:
            normalized_entropy_coef = float(self.entropy_coef)
            self.entropy_coef = normalized_entropy_coef if normalized_entropy_coef > 0.0 else None
        if self.logprob_vocab_chunk_size is not None:
            normalized_chunk_size = int(self.logprob_vocab_chunk_size)
            self.logprob_vocab_chunk_size = normalized_chunk_size if normalized_chunk_size > 0 else None

        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
