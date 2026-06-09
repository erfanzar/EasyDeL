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

"""Agentic MoshPit Self-Play Training with EasyDeL eLarge.

Trains a model through self-play: the model plays three roles simultaneously
using different system prompts:

1. **Questioner** — Generates challenging questions on a given topic.
   Uses ``enable_thinking=False`` to produce clean question text without
   internal reasoning tokens.
2. **Solver** — Reasons through the problem over multiple turns, optionally
   using tools (Python code execution). Thinking is preserved in the
   training signal so the model learns the full reasoning chain.
3. **Verifier** — Grades the solver's answer on a 0-10 scale returned
   inside ``<reward>N</reward>`` tags. Uses ``enable_thinking=False``
   to produce a clean structured score.

No dataset is required — questions are generated on-the-fly from the
topic string. This creates an infinite curriculum that naturally adapts
as the model improves.

How it works:
    1. ``eLargeModel`` loads and shards the model. eSurge is compiled for
       fast batched generation.
    2. Each training step:
       a. The questioner generates ``group_size × num_env_groups`` problems
          (batched, one eSurge call).
       b. The solver works through each problem over ``max_steps`` turns
          (batched per step, with tool-call interception).
       c. The verifier scores all answers (batched, one eSurge call).
       d. GRPO advantages are computed across the group and the policy
          is updated with DAPO loss.
    3. All three roles use the trainer's own model via ``LocalQuestionGenerator``,
       which injects the trainer's ``generate_fn`` at rollout time.

Usage:
    python examples/post_training/agentic_moshpit_self_play.py
    python examples/post_training/agentic_moshpit_self_play.py --topic "calculus"
    python examples/post_training/agentic_moshpit_self_play.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel
from easydel.trainers.agentic_moshpit import (
    LocalQuestionGenerator,
    PythonCodeTool,
    SelfPlayEnvironment,
)


@dataclass
class SelfPlayArgs:
    """Command-line arguments for self-play training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    topic: str = field(
        default="arithmetic and algebra word problems involving money, distances, or time",
        metadata={"help": "Topic for the questioner to generate problems about."},
    )
    output_dir: str = field(
        default="outputs/agentic-moshpit-self-play",
        metadata={"help": "Directory to save checkpoints."},
    )
    max_steps: int = field(
        default=500,
        metadata={"help": "Maximum number of training steps."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Total batch size across all devices."},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of gradient accumulation steps."},
    )
    group_size: int = field(
        default=2,
        metadata={"help": "Number of rollouts per environment seed."},
    )
    num_env_groups: int = field(
        default=2,
        metadata={"help": "Number of distinct environment seeds per batch."},
    )
    max_steps_per_episode: int = field(
        default=3,
        metadata={"help": "Maximum turns per episode."},
    )
    max_length: int = field(
        default=8192,
        metadata={"help": "Maximum total sequence length."},
    )
    max_prompt_length: int = field(
        default=4096,
        metadata={"help": "Maximum prompt length in tokens."},
    )
    wandb: bool = field(
        default=True,
        metadata={"help": "Enable Weights & Biases logging."},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "W&B entity (team or username)."},
    )


def main():
    parser = DataClassArgumentParser(
        SelfPlayArgs,
        description="Agentic MoshPit Self-Play Training with EasyDeL",
    )
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length
    half_steps = args.max_steps // 2

    generator = LocalQuestionGenerator(
        questioner_system_prompt=(
            "You are a creative teacher who designs challenging problems. "
            "Given a topic, create a single original problem that requires "
            "multi-step reasoning and has a clear numeric answer. "
            "Output ONLY the problem statement, nothing else."
        ),
        verifier_system_prompt=(
            "You are a fair grader. Given a question and a student's answer, "
            "determine if the reasoning is sound and the final result is correct. "
            "Output your score as an integer from 0 to 10 inside <reward></reward> tags."
        ),
    )

    topic = args.topic

    def env_factory() -> SelfPlayEnvironment:
        return SelfPlayEnvironment(
            topic=topic,
            generator=generator,
            verify=True,
            answer_pattern=r"\\boxed\{.+\}",
            max_steps_override=args.max_steps_per_episode,
        )

    python_tool = PythonCodeTool(timeout=5.0, max_output_length=2048)

    elm = eLargeModel(
        {
            "model": {
                "name_or_path": args.model,
                "tokenizer": args.model,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
            },
            "sharding": {
                "axis_dims": (1, -1, 1, 2, 1),
                "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            "base_config": {
                "values": {
                    "freq_max_position_embeddings": args.max_length,
                    "mask_max_position_embeddings": args.max_length,
                    "attn_mechanism": ed.AttentionMechanisms.AUTO,
                    "attn_dtype": "bf16",
                    "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
                }
            },
            "esurge": {
                "max_model_len": args.max_length,
                "max_num_seqs": 16,
                "page_size": 64,
                "hbm_utilization": 0.9,
                "enable_prefix_caching": True,
                "verbose": True,
                "max_num_batched_tokens": 2048,
                "use_aot_forward": True,
                "tool_parser": "hermes",
                "reasoning_parser": "qwen3_reasoning",
                "data_parallelism_axis": "fsdp",
            },
            "trainer": {
                "trainer_type": "agentic-moshpit",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 8e-6,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": max_completion_length,
                "max_steps": args.max_steps_per_episode,
                "group_size": args.group_size,
                "num_env_groups": args.num_env_groups,
                "reward_mode": "episode",
                "advantage_estimator": "grpo",
                "beta": 0.04,
                "loss_type": "dapo",
                "scale_rewards": "group",
                "system_prompt": (
                    "You are a problem solver. Think step by step. "
                    "Use the python_code tool if you need to compute something. "
                    "When you have the final answer, write it as \\boxed{<answer>}."
                ),
                "tool_caller": "hermes",
                "esurge_page_size": 64,
                "esurge_hbm_utilization": 0.6,
                "esurge_max_num_seqs": 16,
                "esurge_max_num_batched_tokens": 2048,
                "esurge_enable_prefix_caching": True,
                "esurge_data_parallelism_axis": "fsdp",
                "generation_temperature": 0.6,
                "generation_top_p": 0.95,
                "generation_top_k": 64,
                "generation_do_sample": True,
                "log_steps": 1,
                "save_steps": 100,
                "save_total_limit": 2,
                "generation_interval": 50,
                "generation_prompts": [
                    "A water tank is being filled by two pipes and drained by a third. Pipe A fills at 5 gallons/min, Pipe B at 3 gallons/min, and Pipe C drains at 2 gallons/min. If the tank holds 360 gallons, how long until it's full?",
                    "You have 12 coins, one of which is counterfeit and weighs differently. Using a balance scale at most 3 times, how do you find the counterfeit coin and determine if it's heavier or lighter?",
                    "A train 150 meters long passes a bridge 850 meters long in 40 seconds. What is the speed of the train in km/h?",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
                "benchmark_interval": half_steps,
                "benchmarks": [
                    {
                        "name": "gsm8k",
                        "tasks": ["gsm8k"],
                        "num_fewshot": 0,
                        "max_new_tokens": args.max_length,
                        "confirm_run_unsafe_code": True,
                    },
                ],
            },
        }
    )

    elm.train(
        env_factory=env_factory,
        tools=[python_tool],
    )


if __name__ == "__main__":
    main()
