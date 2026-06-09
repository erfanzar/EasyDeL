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

"""Agentic MoshPit Math Training with EasyDeL eLarge.

Trains a multi-turn math agent using the Agentic MoshPit trainer. Instead
of single-turn generation + reward scoring (like GRPO), the model interacts
with an environment over multiple turns:

    reset() → model generates → env.step() → model generates → ... → done

The agent can use tools (e.g. Python code execution) to solve problems.
Rewards come from the environment at episode termination — typically by
checking whether the agent's final answer matches the ground truth.

How it works:
    1. ``eLargeModel`` loads and shards the model. eSurge is compiled for
       fast batched generation during the multi-turn rollout phase.
    2. Each training step runs ``group_size × num_env_groups`` parallel
       episodes through the environment. Episodes with the same seed share
       the same problem but get independent rollouts for group-relative
       advantage estimation.
    3. The ``GSM8KCodingEnv`` environment:
       - On ``reset()``: samples a GSM8K problem as the initial observation.
       - On ``step(action)``: checks for a ``\\boxed{answer}`` in the action
         and returns reward 1.0 if correct, 0.0 if wrong or not found.
    4. The ``PythonCodeTool`` lets the agent write and execute Python code
       during reasoning, with results fed back as observations.
    5. Trajectories are collated into prompt/response masks and trained
       with the same GRPO/DAPO loss used in standard GRPO training.

Usage:
    python examples/post_training/agentic_moshpit_math.py
    python examples/post_training/agentic_moshpit_math.py --model Qwen/Qwen3-8B
    python examples/post_training/agentic_moshpit_math.py --max_steps_per_episode 5
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel
from easydel.trainers.agentic_moshpit import (
    AgenticEnvironment,
    PythonCodeTool,
    ResetResult,
    StepResult,
)


@dataclass
class AgenticMoshPitMathArgs:
    """Command-line arguments for Agentic MoshPit math training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="openai/gsm8k",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/agentic-moshpit-math",
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


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower().strip()


class GSM8KCodingEnv(AgenticEnvironment):
    """Multi-turn math environment backed by GSM8K problems.

    On ``reset(seed)``, selects a problem from the dataset using the seed
    as an index. On ``step(action)``, checks whether the agent produced a
    ``\\boxed{answer}`` matching the ground truth. If no boxed answer is
    found, the environment continues (the agent gets another turn).
    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._question: str = ""
        self._gold: str = ""

    def reset(self, seed: int | None = None) -> ResetResult:
        idx = (seed or 0) % len(self._dataset)
        example = self._dataset[idx]
        self._question = example["question"]
        answer_text = example["answer"]
        self._gold = _normalize_number(answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text)
        return ResetResult(observation=self._question)

    def step(self, action: str) -> StepResult:
        match = _BOXED_RE.search(action)
        if match:
            pred = _normalize_number(match.group(1))
            correct = pred == self._gold
            return StepResult(
                observation="Correct!" if correct else f"Wrong. Expected {self._gold}.",
                reward=1.0 if correct else 0.0,
                terminated=True,
            )
        return StepResult(
            observation="Please provide your final answer as \\boxed{answer}.",
            reward=0.0,
            terminated=False,
        )

    def close(self) -> None:
        pass


def main():
    parser = DataClassArgumentParser(
        AgenticMoshPitMathArgs,
        description="Agentic MoshPit Math Training with EasyDeL",
    )
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length
    half_steps = args.max_steps // 2

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    def env_factory() -> GSM8KCodingEnv:
        return GSM8KCodingEnv(dataset=gsm8k)

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
                    "You are a math problem solver. Think step by step. "
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
                    "A rectangular garden is 3 meters longer than it is wide. If a walkway 1.5 meters wide surrounds it and the total area including the walkway is 200 square meters, find the garden dimensions.",
                    "Two cyclists start from the same point. One goes north at 15 km/h, the other goes east at 20 km/h. After 3 hours, how far apart are they?",
                    "A factory produces widgets. On Monday it produces 100. Each subsequent day it produces 12%% more than the previous day. How many total widgets are produced Monday through Friday?",
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
