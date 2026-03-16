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

"""GKD (Generalized Knowledge Distillation) with EasyDeL eLarge.

Distills a larger teacher model into a smaller student using generalized
knowledge distillation. GKD combines standard KD (matching teacher output
distributions) with on-policy distillation (student generates, teacher
scores) for more effective knowledge transfer.

How it works:
    1. ``eLargeModel`` loads both the student model (being trained) and
       a frozen teacher model (specified via the ``teacher`` config key).
    2. The student generates completions on-policy, then the teacher's
       output distribution on those completions is used as a soft target.
    3. The GKD loss combines forward KL divergence (matching teacher
       distribution) with a supervised SFT component on the training data.
    4. This on-policy approach avoids the train-test distribution mismatch
       that plagues standard KD, where the student only sees teacher-
       generated text during training.

Usage:
    python examples/post_training/gkd.py
    python examples/post_training/gkd.py --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-1.7B
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class GKDArgs:
    """Command-line arguments for GKD training."""

    student: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "Student model name or HuggingFace path."},
    )
    teacher: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Teacher model name or HuggingFace path."},
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
        default="outputs/gkd",
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
    parser = DataClassArgumentParser(GKDArgs, description="GKD Training with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    elm = eLargeModel(
        {
            "model": {
                "name_or_path": args.student,
                "tokenizer": args.student,
                "task": "auto-bind",
            },
            "teacher": {
                "name_or_path": args.teacher,
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
                "trainer_type": "gkd",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 1e-5,
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
                    "Explain how transformers work in neural networks, focusing on the attention mechanism.",
                    "Write a Python generator that yields the Fibonacci sequence indefinitely.",
                    "What is the CAP theorem and why does it matter for distributed systems?",
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
