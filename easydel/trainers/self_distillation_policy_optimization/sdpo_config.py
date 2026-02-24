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

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", "sdpo")
@dataclass
class SDPOConfig(GRPOConfig):
    """Configuration for Self-Distillation Policy Optimization (SDPO).

    SDPO converts tokenized environment feedback into a dense learning signal
    without any external teacher or reward model by using the current policy
    as a self-teacher. The model is prompted with its original attempt plus
    feedback, and the resulting next-token distribution is distilled back into
    the student policy.

    This approach removes the information bottleneck of scalar outcome rewards
    (as in GRPO) by leveraging rich textual feedback — runtime errors, failing
    tests, or LLM judge evaluations — to provide token-level credit assignment.

    When no rich feedback function is supplied the trainer falls back to the
    paper's "learning without rich feedback" mode (Section 3): successful
    rollouts from the same group are used as implicit feedback for failed ones.

    Reference:
        "Reinforcement Learning via Self-Distillation" (arXiv:2601.20802)

    Key hyperparameters vs GRPO:
    - No reward-derived advantages: the self-teacher provides per-token
      advantages = log(teacher_logp / student_logp).
    - ``distillation_type``: loss variant used to match student to teacher.
    - ``max_feedback_length``: maximum tokens reserved for the feedback
      separator that is inserted between the prompt and the original
      completion in the teacher context.
    - ``beta``: optional KL penalty toward a frozen reference model (same as
      GRPO; default 0 because the self-teacher already acts as a regulariser).
    """

    trainer_prefix: str | None = field(
        default="sdpotrainer",
        metadata={"help": "Default prefix name for the trainer checkpoint directory."},
    )

    max_feedback_length: int = field(
        default=256,
        metadata={
            "help": (
                "Maximum number of tokens reserved for the feedback separator "
                "that is inserted between the original prompt and the original "
                "completion in the teacher context. Feedback strings that are "
                "longer will be truncated; shorter ones are right-padded with "
                "the tokenizer's pad token."
            )
        },
    )

    distillation_type: str = field(
        default="jsd",
        metadata={
            "help": (
                "Loss used to match the student to the self-teacher. "
                "'kl'  - forward KL(student || teacher) approximated at the "
                "         sampled token: loss = student_logp - teacher_logp. "
                "'jsd' - (recommended, Section 2.3) symmetric JSD approximated "
                "         at the sampled token using the log-sum-exp mixture: "
                "         loss = student_logp - log((student_prob + teacher_prob)/2). "
                "         Bounded in [-log 2, 0] per token; more stable than KL."
            )
        },
    )

    beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL-divergence penalty toward the frozen reference model "
                "(inherited from GRPOConfig). Defaults to 0 for SDPO because "
                "the self-distillation objective already regularises the policy. "
                "Set >0 to additionally penalise deviation from the initial model."
            )
        },
    )

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None):
        if self.distillation_type not in ("kl", "jsd"):
            raise ValueError(f"`distillation_type` must be 'kl' or 'jsd', got '{self.distillation_type}'.")
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )

    __hash__ = hash_fn
