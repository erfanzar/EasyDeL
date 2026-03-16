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

"""Supervised Fine-Tuning (SFT) with EasyDeL eLarge.

Trains a language model on instruction-following data using standard
cross-entropy loss on the assistant tokens. The chat template is
automatically applied by the trainer so the dataset only needs to
provide ``messages`` in OpenAI format.

How it works:
    1. ``eLargeModel`` loads and shards the model across devices.
    2. The SFT trainer tokenizes each conversation using the model's
       chat template, masks out non-assistant tokens, and trains with
       standard next-token prediction loss.
    3. No reward model, reference model, or generation is needed —
       this is pure supervised learning on demonstrations.

This is typically the first step in an RLHF pipeline: SFT → DPO/GRPO.

Usage:
    python examples/post_training/sft.py
    python examples/post_training/sft.py --model Qwen/Qwen3-8B --max_length 4096
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class SFTArgs:
    """Command-line arguments for SFT training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "HuggingFace instruction dataset name."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use."},
    )
    output_dir: str = field(
        default="outputs/sft",
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
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length."},
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
    parser = DataClassArgumentParser(SFTArgs, description="SFT Training with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

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
            "trainer": {
                "trainer_type": "sft",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 2e-5,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "max_length": args.max_length,
                "log_steps": 1,
                "save_steps": 100,
                "save_total_limit": 2,
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
                "generation_interval": 50,
                "generation_prompts": [
                    "Explain the difference between a compiler and an interpreter to a 10-year-old.",
                    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes.",
                    "What are the trade-offs between microservices and monolithic architectures?",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
            },
        }
    )

    train_dataset = load_dataset(args.dataset, split=args.dataset_split)

    elm.train(train_dataset=train_dataset)


if __name__ == "__main__":
    main()
