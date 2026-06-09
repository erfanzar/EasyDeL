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

"""CPO (Contrastive Preference Optimization) with EasyDeL eLarge.

Trains a language model on paired preference data without needing a
frozen reference model copy. CPO uses a contrastive loss that directly
optimizes the policy to prefer chosen over rejected completions,
making it more memory-efficient than DPO.

How it works:
    1. ``eLargeModel`` loads and shards a single model — no reference
       model is needed, saving ~50%% GPU memory compared to DPO.
    2. The dataset contains ``(prompt, chosen, rejected)`` triples.
    3. CPO computes a contrastive loss that maximizes the likelihood
       gap between chosen and rejected completions, controlled by
       ``beta``. An optional SFT component (``loss_type="simpo"``)
       adds a supervised term on the chosen completions.

CPO is ideal when GPU memory is limited and you want DPO-quality
alignment without the reference model overhead.

Usage:
    python examples/post_training/cpo.py
    python examples/post_training/cpo.py --model Qwen/Qwen3-8B --beta 0.1
    python examples/post_training/cpo.py --loss_type simpo
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class CPOArgs:
    """Command-line arguments for CPO training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "HuggingFace preference dataset name."},
    )
    dataset_split: str = field(
        default="train[:10%]",
        metadata={"help": "Dataset split to use for training."},
    )
    output_dir: str = field(
        default="outputs/cpo",
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
        metadata={"help": "Maximum total sequence length."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum prompt length in tokens."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "CPO temperature for contrastive loss."},
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "Loss variant: sigmoid (standard CPO) or simpo (with SFT term)."},
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
    parser = DataClassArgumentParser(CPOArgs, description="CPO Training with EasyDeL")
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
            "trainer": {
                "trainer_type": "cpo",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": 8e-6,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": max_completion_length,
                "beta": args.beta,
                "loss_type": args.loss_type,
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
                    "Derive the quadratic formula from first principles and explain each step.",
                    "Write a Rust function that implements a thread-safe LRU cache.",
                    "Explain why P vs NP is considered the most important open problem in computer science.",
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
