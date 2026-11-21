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
"""Configuration for :class:`AgentFlowTrainer` multi-turn agentic RL."""

from __future__ import annotations

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils import Registry

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "agent_flow")
@auto_pytree
class AgentFlowConfig(TrainingArguments):
    """Configuration for multi-turn :class:`AgentFlowTrainer` wrapping any RL/distillation trainer.

    Args:
        base_trainer_type: Registry name of the underlying trainer algorithm
            (e.g. ``"grpo"``, ``"gmpo"``, ``"gdpo"``, ``"reinforce_plus_plus"``,
            ``"ppo"``, ``"vapo"``, ``"gkd"``, ``"on_policy_distillation"``).
        base_config: Optional pre-built ``TrainingArguments`` or dict of overrides
            used when constructing the underlying trainer.
        environment: Environment name, mapping specification, or ``None``.
        environment_backend: Sandbox backend (``"kiri"``, ``"docker"``, ``"auto"``,
            or ``"local"``).
        environment_endpoints: Optional explicit list of Kiri/Docker endpoints.
        max_turns: Maximum interaction turns per multi-turn episode.
        context_mode: Multi-turn context strategy: ``"append"`` (default),
            ``"sliding_window"``, or ``"reset_with_summary"``.
        sliding_window_turns: Number of recent turns retained when
            ``context_mode == "sliding_window"``.
        gamma: Discount factor applied when ``reward_mode == "discounted_sum"``
            and for per-turn return-to-go attribution.
        reward_mode: How multi-turn step rewards aggregate into episode reward:
            ``"sum"``, ``"terminal"``, or ``"discounted_sum"``.
        max_workers: Maximum concurrent environment rollout workers.
        group_size: Optional rollouts per prompt group (defaults to the inner
            trainer's ``num_generations`` / ``num_return_sequences`` or ``1``).
        max_prompt_length: Maximum prompt token length.
        max_completion_length: Maximum completion/trajectory token length.
        num_generations: Optional group size forwarded to the base config.
        skip_apply_chat_template: Skip chat template formatting on prompts.
    """

    trainer_prefix: str | None = field(
        default="AgentFlow",
        metadata={"help": "Default prefix for AgentFlow checkpoints and logs."},
    )
    base_trainer_type: str = field(
        default="grpo",
        metadata={"help": "Registry name of the underlying RL/distillation trainer."},
    )
    base_config: tp.Any = field(
        default=None,
        metadata={"help": "Optional pre-built base trainer config or dict of overrides."},
    )
    environment: tp.Any = field(
        default=None,
        metadata={"help": "Environment name, mapping specification, or factory."},
    )
    environment_backend: str = field(
        default="local",
        metadata={"help": "Sandbox backend for environment execution ('local', 'kiri', 'docker', 'auto')."},
    )
    environment_endpoints: list[str] | str | None = field(
        default=None,
        metadata={
            "help": "Optional explicit sandbox daemon endpoints, or 'shm:N' / 'pod-shm:N' to auto-boot in /dev/shm."
        },
    )
    max_turns: int = field(
        default=8,
        metadata={"help": "Maximum interaction turns per multi-turn episode."},
    )
    context_mode: str = field(
        default="append",
        metadata={"help": "Multi-turn context strategy: 'append', 'sliding_window', or 'reset_with_summary'."},
    )
    sliding_window_turns: int | None = field(
        default=None,
        metadata={"help": "Number of recent turns kept when context_mode='sliding_window'."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor for multi-turn return-to-go and discounted_sum rewards."},
    )
    reward_mode: str = field(
        default="sum",
        metadata={"help": "Episode reward aggregation: 'sum', 'terminal', or 'discounted_sum'."},
    )
    max_workers: int = field(
        default=8,
        metadata={"help": "Maximum concurrent environment rollout workers."},
    )
    group_size: int | None = field(
        default=None,
        metadata={"help": "Optional explicit number of episodes per prompt group."},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length in tokens."},
    )
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "Maximum completion/trajectory length in tokens."},
    )
    per_turn_max_tokens: int | None = field(
        default=None,
        metadata={"help": "Optional per-turn generation token cap passed to rollout()."},
    )
    max_obs_tokens_per_turn: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional per-turn observation token cap. When None and per_turn_max_tokens is set, "
                "auto-caps observations so max_turns * per_turn_max_tokens + observations fit inside "
                "max_completion_length."
            )
        },
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Optional number of generations per prompt forwarded to base config."},
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Skip chat template formatting on prompts."},
    )

    def __post_init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__(*args, **kwargs)
        if self.max_turns <= 0:
            raise ValueError(f"`max_turns` must be positive; got {self.max_turns}.")
        if self.context_mode not in {"append", "sliding_window", "reset_with_summary"}:
            raise ValueError(
                f"`context_mode` must be 'append', 'sliding_window', or 'reset_with_summary'; got {self.context_mode!r}."
            )
        if (
            self.context_mode == "sliding_window"
            and self.sliding_window_turns is not None
            and self.sliding_window_turns <= 0
        ):
            raise ValueError(f"`sliding_window_turns` must be positive; got {self.sliding_window_turns}.")
        if self.reward_mode not in {"sum", "terminal", "discounted_sum"}:
            raise ValueError(f"`reward_mode` must be 'sum', 'terminal', or 'discounted_sum'; got {self.reward_mode!r}.")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError(f"`gamma` must be in [0, 1]; got {self.gamma}.")
        if self.max_workers <= 0:
            raise ValueError(f"`max_workers` must be positive; got {self.max_workers}.")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError(f"`group_size` must be positive; got {self.group_size}.")
        self.environment_backend = str(self.environment_backend).strip().lower()
        if self.environment_backend not in {"local", "kiri", "docker", "auto"}:
            raise ValueError(
                f"`environment_backend` must be 'local', 'kiri', 'docker', or 'auto'; got {self.environment_backend!r}."
            )
        self.base_trainer_type = str(self.base_trainer_type).strip().lower()
        if self.base_trainer_type in {"agent_flow", "agent-flow"}:
            raise ValueError("`base_trainer_type` cannot be 'agent_flow' (recursive wrapping is disallowed).")
        if not _is_supported_base_trainer_type(self.base_trainer_type):
            raise ValueError(
                f"`base_trainer_type` {self.base_trainer_type!r} is not supported by AgentFlowTrainer. "
                "AgentFlow adapts episodes for the GRPO family, the PPO family, online DPO, and the on-policy "
                f"distillation family; supported values: {sorted(AGENT_FLOW_BASE_TRAINER_TYPES)} "
                "(or any registered subclass of those trainers)."
            )


# Registered trainer names whose classes ``AgentFlowTrainer._adapt_episodes_to_trainer`` knows how to feed:
# OnlineDPOTrainer (preference pairs), PPOTrainer subclasses (returns/advantages/values), the on-policy
# distillation family (GKD / OnPolicyDistillation / SDFT / SDPO / SeqKD / SSD), and GRPOTrainer subclasses
# (the default advantage layout). Offline trainers (sft, dpo, kto, ...) would receive a batch they cannot use.
AGENT_FLOW_BASE_TRAINER_TYPES: frozenset[str] = frozenset(
    {
        # GRPO family.
        "agentic-moshpit",
        "async_grpo",
        "dppo",
        "gdpo",
        "gfpo",
        "gmpo",
        "grpo",
        "grpo_replay_buffer",
        "grpo_with_replay_buffer",
        "gspo",
        "gspo_token",
        "minillm",
        "nash-md",
        "nash_md",
        "nemo_gym",
        "papo",
        "reinforce_plus_plus",
        "rloo",
        "rlvr",
        "xpo",
        # PPO family.
        "compaction-rl",
        "compaction_reinforcement_learning",
        "compaction_rl",
        "ppo",
        "sao",
        "single-rollout-asynchronous-optimization",
        "single_rollout_asynchronous_optimization",
        "vapo",
        # Preference pairs.
        "online_dpo",
        # On-policy distillation family.
        "gkd",
        "on_policy_distillation",
        "sdft",
        "sdpo",
        "self_distillation",
        "seq_kd",
        "ssd",
    }
)


def _is_supported_base_trainer_type(name: str) -> bool:
    """Return whether AgentFlow can adapt episodes for the trainer registered as ``name``.

    Built-in names are checked against :data:`AGENT_FLOW_BASE_TRAINER_TYPES`. Any other name is accepted
    only when it is registered as a trainer whose class subclasses one of the supported families, so
    user-registered GRPO/PPO/OnlineDPO/distillation variants keep working.
    """
    if name in AGENT_FLOW_BASE_TRAINER_TYPES:
        return True
    trainer_cls = Registry.get("trainer", name)
    if not isinstance(trainer_cls, type):
        return False
    from ..generalized_knowledge_distillation_trainer import GKDTrainer
    from ..group_relative_policy_optimization import GRPOTrainer
    from ..on_policy_distillation_trainer import OnPolicyDistillationTrainer
    from ..online_dpo_trainer import OnlineDPOTrainer
    from ..proximal_policy_optimization_trainer import PPOTrainer
    from ..seq_kd_trainer import SeqKDTrainer

    return issubclass(
        trainer_cls,
        (GRPOTrainer, PPOTrainer, OnlineDPOTrainer, GKDTrainer, OnPolicyDistillationTrainer, SeqKDTrainer),
    )


__all__ = ["AGENT_FLOW_BASE_TRAINER_TYPES", "AgentFlowConfig"]
