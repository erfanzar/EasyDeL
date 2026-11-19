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
"""Configuration dataclass for the on-policy distillation trainer.

Defines :class:`OnPolicyDistillationConfig`, which extends
:class:`DistillationConfig` with the online-generation knobs
(prompt/completion lengths, sampling temperatures, top-k/top-p,
penalties) used to roll out student (or teacher) sequences during
training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..distillation_trainer.distillation_config import DistillationConfig


@Registry.register("trainer-arguments", "on_policy_distillation")
@dataclass
class OnPolicyDistillationConfig(DistillationConfig):
    """Configuration for on-policy knowledge distillation training.

    Unlike standard (offline) distillation which operates on a fixed dataset
    with pre-computed teacher outputs, on-policy distillation generates
    completions from prompts during training and distills teacher knowledge
    on the student's own generated sequences.

    This configuration extends DistillationConfig with parameters for
    controlling the online generation process during training.

    Attributes:
        trainer_prefix: Prefix used for trainer logs, checkpoints, and W&B
            runs. Defaults to ``"OnPolicyDistillation"``.
        remove_unused_columns: Whether the base trainer should drop columns
            not consumed by the trainer. Defaults to ``False`` to keep raw
            metadata available to reward / generation hooks.
        max_prompt_length: Maximum number of tokens for prompts.
        max_completion_length: Maximum number of new tokens to generate per
            prompt.
        num_generations_per_prompt: How many completions to sample per
            prompt.
        temperature_sampling: Sampling temperature for generation (separate
            from the distillation temperature used to soften logits for the
            KL loss).
        top_k: Top-k sampling parameter for generation; ``None`` disables
            top-k.
        top_p: Top-p (nucleus) sampling parameter for generation.
        presence_penalty: Presence penalty applied during generation.
        frequency_penalty: Frequency penalty applied during generation.
        repetition_penalty: Repetition penalty applied during generation.
        generate_with_teacher: If ``True``, the teacher model generates
            completions instead of the student. Useful for comparison or
            curriculum-style distillation.
        skip_apply_chat_template: Whether to skip chat template application
            on prompts.
        distillation_mode: ``"divergence"`` or ``"trust_region"``
            (speculative-decoding-style acceptance of sampled tokens).
        trust_region_top_k: Teacher top-k support of the rejected-token
            forward KL in trust-region mode. Defaults to ``64``.
        trust_region_seed: Base seed of the per-step acceptance draw.

    In trust-region mode every log-prob (acceptance ratio, policy-gradient term and rejected-token top-k KL)
    is computed at the *sampling* temperature (:attr:`trust_region_sampling_temperature`), not the
    distillation ``temperature``, and no ``T**2`` factor is applied to either branch.
    """

    trainer_prefix: str | None = field(
        default="OnPolicyDistillation",
        metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens for prompts."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate per prompt."},
    )
    num_generations_per_prompt: int = field(
        default=1,
        metadata={"help": "How many completions to sample per prompt."},
    )
    temperature_sampling: float = field(
        default=0.7,
        metadata={
            "help": "Sampling temperature for generation. Separate from the distillation "
            "temperature which controls logit softening for KL loss."
        },
    )
    top_k: int | None = field(
        default=50,
        metadata={"help": "Top-k sampling parameter for generation. None disables top-k."},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p (nucleus) sampling parameter for generation."},
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
    generate_with_teacher: bool = field(
        default=False,
        metadata={
            "help": "If True, the teacher model generates completions instead of the student. "
            "Useful for comparison or curriculum-style distillation."
        },
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to skip chat template application on prompts."},
    )
    distillation_mode: str = field(
        default="divergence",
        metadata={
            "help": (
                "'divergence' (default) minimizes the selected divergence (forward KL, `beta` JSD/reverse KL, "
                "`cakld_gamma`, `loss_top_k`). 'trust_region' applies speculative-decoding-style acceptance to "
                "each sampled token: with P = min(exp(t_lp - s_lp), 1) and M ~ Bernoulli(P), accepted tokens get "
                "a reverse-KL policy gradient sg(s_lp - t_lp) * s_lp and rejected tokens the teacher-top-k forward "
                "KL (`trust_region_top_k`)."
            )
        },
    )
    trust_region_top_k: int = field(
        default=64,
        metadata={"help": "Teacher top-k support of the rejected-token forward KL in trust-region mode."},
    )
    trust_region_seed: int = field(
        default=0,
        metadata={"help": "Base seed of the per-step trust-region acceptance draw (folded with the step)."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Validate and reconcile prompt/completion/total length budgets.

        If both ``max_length`` and ``max_completion_length`` are configured,
        ensures their sum does not exceed ``max_length``. If only
        ``max_length`` is provided, ``max_completion_length`` is derived as
        ``max_length - max_prompt_length``. ``max_length`` is then
        renormalized as the sum.

        Args:
            max_sequence_length (int | None): Deprecated alias forwarded to
                the base class for backward compatibility.
            quantization_block (int | None): Optional quantization block
                size forwarded to the base class.

        Raises:
            ValueError: If ``max_length`` is smaller than ``max_prompt_length``
                or if ``max_prompt_length + max_completion_length`` exceeds
                ``max_length``.
        """
        if self.distillation_mode not in ("divergence", "trust_region"):
            raise ValueError(
                f"`distillation_mode` must be 'divergence' or 'trust_region'; got {self.distillation_mode!r}."
            )
        self.trust_region_top_k = int(self.trust_region_top_k)
        if self.trust_region_top_k < 1:
            raise ValueError("`trust_region_top_k` must be >= 1.")
        self.trust_region_seed = int(self.trust_region_seed)
        default_prompt = type(self).__dataclass_fields__["max_prompt_length"].default
        default_completion = type(self).__dataclass_fields__["max_completion_length"].default
        if self.max_prompt_length is None:
            self.max_prompt_length = default_prompt
        if self.max_completion_length is None:
            self.max_completion_length = default_completion
        if self.max_length is not None:
            if self.max_length < self.max_prompt_length:
                raise ValueError(
                    f"`max_length` ({self.max_length}) must be >= `max_prompt_length` ({self.max_prompt_length})."
                )
            max_allowed_completion = self.max_length - self.max_prompt_length
            if self.max_completion_length == default_completion:
                self.max_completion_length = max_allowed_completion
            elif self.max_completion_length > max_allowed_completion:
                raise ValueError(
                    "`max_prompt_length + max_completion_length` "
                    f"({self.max_prompt_length} + {self.max_completion_length}) must be <= `max_length` "
                    f"({self.max_length})."
                )

        self.max_length = self.max_prompt_length + self.max_completion_length

        on_policy_fields = {
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        # Inherit sampling defaults before offline validation neutralizes them.
        if self.generation_top_p is None:
            self.generation_top_p = self.top_p
        if self.generation_top_k is None:
            self.generation_top_k = self.top_k
        self.max_prompt_length = None  # type: ignore[assignment]
        self.max_completion_length = None  # type: ignore[assignment]
        self.top_p = 1.0
        self.top_k = 0
        try:
            if hasattr(super(), "__post_init__"):
                super().__post_init__(
                    max_sequence_length=None,
                    quantization_block=quantization_block,
                )
        finally:
            self.max_prompt_length = on_policy_fields["max_prompt_length"]
            self.max_completion_length = on_policy_fields["max_completion_length"]
            self.top_p = on_policy_fields["top_p"]
            self.top_k = on_policy_fields["top_k"]

        if self.distillation_mode == "trust_region":
            if self.generation_do_sample is False:
                raise ValueError(
                    "`distillation_mode='trust_region'` scores sampled tokens; it cannot be used with "
                    "`generation_do_sample=False`."
                )
            sampling_temperature = self.trust_region_sampling_temperature
            if sampling_temperature is None or not float(sampling_temperature) > 0.0:
                raise ValueError(
                    "`distillation_mode='trust_region'` needs a positive sampling temperature "
                    f"(`temperature_sampling` / `generation_temperature`); got {sampling_temperature!r}."
                )

    @property
    def trust_region_sampling_temperature(self) -> float | None:
        """Temperature the completions are actually sampled at (``generation_temperature`` wins when set)."""
        value = getattr(self, "generation_temperature", None)
        if value is None:
            value = self.temperature_sampling
        return None if value is None else float(value)

    __hash__ = hash_fn
