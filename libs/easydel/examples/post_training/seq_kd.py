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

"""Sequence-level Knowledge Distillation (SeqKD) with EasyDeL eLarge.

Performs sequence-level knowledge distillation where the teacher model first
generates complete sequences, and the student is then trained via standard
supervised fine-tuning on those teacher-generated sequences. Unlike token-level
KD (which matches per-token distributions), SeqKD transfers knowledge at the
sequence level by treating the teacher's outputs as gold-standard training data.

How it works:
    1. ``eLargeModel`` loads both the student (trainable) and teacher (frozen)
       models, sharding them across devices via the ``sharding`` config.
    2. For each batch of prompts, the teacher generates full completions using
       eSurge (EasyDeL's paged-attention inference engine).
    3. The student is trained with a standard cross-entropy loss on the
       teacher-generated sequences, effectively performing SFT on synthetic
       data produced by the teacher.
    4. This approach is simpler than token-level KD (no temperature tuning or
       KL computation needed) and naturally transfers the teacher's generation
       style, reasoning patterns, and output formatting. However, it discards
       the full distributional information available from the teacher.

Usage:
    python examples/post_training/seq_kd.py
    python examples/post_training/seq_kd.py --model Qwen/Qwen3-4B --teacher_model Qwen/Qwen3-8B
    python examples/post_training/seq_kd.py --batch_size 8 --max_length 4096
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class SeqKDArgs:
    """Command-line arguments for sequence-level knowledge distillation."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Student model name or HuggingFace path."},
    )
    teacher_model: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Teacher model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/seq-kd",
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
        metadata={"help": "Maximum total sequence length (prompt + completion)."},
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


def format_prompts(example):
    """Extract the user prompt from ultrafeedback_binarized chosen messages."""
    messages = example["chosen"]
    prompt = [msg for msg in messages if msg["role"] == "user"]
    return {"prompt": prompt}


def main():
    parser = DataClassArgumentParser(SeqKDArgs, description="Sequence-level Knowledge Distillation with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length

    elm = eLargeModel(
        {
            "model": {
                "name_or_path": args.model,
                "tokenizer": args.model,
                "task": "auto-bind",
            },
            "teacher": {
                "name_or_path": args.teacher_model,
                "tokenizer": args.teacher_model,
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
                "trainer_type": "seq_kd",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 1e-5,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": max_completion_length,
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
                    "Compare and contrast REST and GraphQL APIs, including when each is most appropriate.",
                    "Write a Python context manager that temporarily changes the working directory and restores it on exit.",
                    "Explain how consistent hashing works and why it is useful for distributed caching systems.",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
            },
        }
    )

    train_dataset = load_dataset(args.dataset, split="train_prefs").map(
        format_prompts, remove_columns=["chosen", "rejected"]
    )

    elm.train(train_dataset=train_dataset)


if __name__ == "__main__":
    main()
