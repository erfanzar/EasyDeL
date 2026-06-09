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

"""BCO (Binary Classifier Optimization) with EasyDeL eLarge.

Trains a language model using Binary Classifier Optimization (BCO) on a
preference dataset. Unlike DPO which requires paired (chosen, rejected)
comparisons, BCO treats preference alignment as a binary classification
problem -- each response is independently classified as desirable or
undesirable using a learned density ratio estimator.

How it works:
    1. ``eLargeModel`` loads and shards the model across devices using the
       ``loader`` and ``sharding`` sections of the config. A frozen
       reference copy is automatically created for computing reference
       log-probabilities.
    2. BCO samples a pool of ``prompt_sample_size`` prompts to fit a running
       estimate of the implicit reward distribution via density ratios.
    3. For each training example, the trainer computes the log-probability
       ratio between the policy and the frozen reference model, then
       classifies it as chosen or rejected using a binary cross-entropy
       objective scaled by ``beta``.
    4. The density ratio is clamped between ``min_density_ratio`` and
       ``max_density_ratio`` to prevent training instabilities from extreme
       likelihood ratios.
    5. eSurge (EasyDeL's paged-attention inference engine) handles generation
       previews at ``generation_interval`` steps so you can monitor output
       quality during training.

Usage:
    python examples/post_training/bco.py
    python examples/post_training/bco.py --model Qwen/Qwen3-8B --batch_size 8
    python examples/post_training/bco.py --max_length 4096 --wandb
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class BCOArgs:
    """Command-line arguments for BCO training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "HuggingFace preference dataset name."},
    )
    output_dir: str = field(
        default="outputs/bco",
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


def main():
    parser = DataClassArgumentParser(BCOArgs, description="BCO Training with EasyDeL")
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
                "trainer_type": "bco",
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
                "beta": 0.1,
                "prompt_sample_size": 1024,
                "min_density_ratio": 0.5,
                "max_density_ratio": 10.0,
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
                    "A patient presents with chest pain, shortness of breath, and elevated troponin levels. Walk through the differential diagnosis and explain your reasoning for each possibility.",
                    "Classify the following argument as valid or invalid and explain why: 'All mammals are warm-blooded. All whales are mammals. Therefore, all warm-blooded animals are whales.'",
                    "Given a dataset with 95%% class imbalance, compare and contrast at least three strategies for building a reliable binary classifier.",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
            },
        }
    )

    train_dataset = load_dataset(args.dataset, split="train[:10%]")

    elm.train(train_dataset=train_dataset)


if __name__ == "__main__":
    main()
