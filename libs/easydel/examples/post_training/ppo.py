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

"""PPO (Proximal Policy Optimization) with EasyDeL eLarge.

Trains a language model using PPO for RLHF. This is the classic approach
with a policy model, reference model, and reward function.

How it works:
    1. ``eLargeModel`` loads and shards the policy model. A frozen
       reference copy is automatically created.
    2. Each training step:
       a. The policy generates completions for prompts from the dataset.
       b. A reward function scores each completion.
       c. Advantages are computed using GAE (Generalized Advantage
          Estimation) or REINFORCE-style returns.
       d. The policy is updated with the PPO clipped surrogate loss,
          keeping it close to the reference via a KL penalty.
    3. Unlike GRPO which uses group-relative baselines, PPO can
       optionally use a value head (critic) for variance reduction.

PPO is well-suited for training with learned reward models and
provides fine-grained control over the RL optimization process.

Usage:
    python examples/post_training/ppo.py
    python examples/post_training/ppo.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class PPOArgs:
    """Command-line arguments for PPO training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "HuggingFace dataset name."},
    )
    dataset_split: str = field(
        default="train[:10%]",
        metadata={"help": "Dataset split to use for training."},
    )
    output_dir: str = field(
        default="outputs/ppo",
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
        default=4,
        metadata={"help": "Number of gradient accumulation steps."},
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of completions per prompt."},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum total sequence length."},
    )
    max_prompt_length: int = field(
        default=512,
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


def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Simple length-based reward for demonstration.

    In practice, replace this with a learned reward model or
    a task-specific verifiable reward function.
    """
    return [min(len(c.split()) / 100.0, 1.0) for c in completions]


def main():
    parser = DataClassArgumentParser(PPOArgs, description="PPO Training with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length

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
                "trainer_type": "ppo",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 5e-7,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": max_completion_length,
                "beta": 0.04,
                "num_return_sequences": args.num_generations,
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
                    "Explain how gradient descent works and why the learning rate matters.",
                    "Write a Python implementation of binary search that handles edge cases.",
                    "What would happen to Earth's climate if the Moon suddenly disappeared?",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
            },
        }
    )

    train_dataset = load_dataset(args.dataset, split=args.dataset_split)

    elm.train(
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
    )


if __name__ == "__main__":
    main()
