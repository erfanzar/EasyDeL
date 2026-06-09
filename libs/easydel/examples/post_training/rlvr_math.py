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

"""RLVR Math Training with EasyDeL eLarge.

Trains a language model on GSM8K math problems using Reinforcement Learning
with Verifiable Rewards (RLVR). Unlike standard GRPO where you supply custom
reward functions, RLVR automatically builds a reward pipeline from verifiable
specifications in the config:

- ``MathVerifier``: Extracts ``\\boxed{...}`` from completions and compares
  against the gold answer in the dataset's ``answer_key`` column.
- ``FormatVerifier``: Optional regex-based check that completions follow
  a required reasoning structure (e.g. ``<think>...</think>\\boxed{...}``).
- ``LengthPenaltyVerifier``: Penalizes completions that exceed a target
  length to encourage concise reasoning.

The entire pipeline — model loading, sharding, eSurge inference, RLVR
training loop, and lm-eval benchmarking — is driven through a single
``eLargeModel`` configuration dict.

RLVR is ideal when ground-truth answers are available and you want
zero-setup RL training without writing custom reward functions.

Usage:
    python examples/post_training/rlvr_math.py
    python examples/post_training/rlvr_math.py --model Qwen/Qwen3-8B
    python examples/post_training/rlvr_math.py --num_generations 16 --max_length 8192
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class RLVRMathArgs:
    """Command-line arguments for RLVR math training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="openai/gsm8k",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/rlvr-math",
        metadata={"help": "Directory to save checkpoints."},
    )
    max_steps: int = field(
        default=500,
        metadata={"help": "Maximum number of training steps."},
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Total batch size across all devices."},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of gradient accumulation steps."},
    )
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of completions generated per prompt."},
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum total sequence length (prompt + completion)."},
    )
    max_prompt_length: int = field(
        default=1024,
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


def prepare_dataset(dataset_name: str = "openai/gsm8k", split: str = "train"):
    """Load and format GSM8K for RLVR training.

    The RLVR trainer's built-in MathVerifier reads the ``answer`` column
    to score completions, so the dataset must provide gold answers.
    """
    ds = load_dataset(dataset_name, "main", split=split)

    def _format(example):
        answer_text = example["answer"]
        final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text
        return {
            "prompt": [{"role": "user", "content": example["question"]}],
            "answer": final_answer,
        }

    return ds.map(_format, remove_columns=ds.column_names)


def main():
    parser = DataClassArgumentParser(RLVRMathArgs, description="RLVR Math Training with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length
    half_steps = args.max_steps // 2

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
                "trainer_type": "rlvr",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 5e-6,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": max_completion_length,
                "beta": 0.04,
                "loss_type": "dapo",
                "num_return_sequences": args.num_generations,
                "scale_rewards": "group",
                "answer_key": "answer",
                "format_pattern": r"<think>.*?</think>.*\\boxed\{.+\}",
                "format_reward_weight": 0.1,
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
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
                "generation_interval": 50,
                "generation_prompts": [
                    "A store offers a 20%% discount followed by an additional 15%% discount. Is this the same as a single 35%% discount? Prove your answer.",
                    "How many ways can you make change for $1 using quarters, dimes, nickels, and pennies?",
                    "If the sum of three consecutive even numbers is 78, what are the numbers?",
                ],
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

    train_dataset = prepare_dataset(args.dataset)

    elm.train(train_dataset=train_dataset)


if __name__ == "__main__":
    main()
