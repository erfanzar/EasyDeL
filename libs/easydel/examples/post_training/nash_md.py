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

"""Nash-MD (Nash Mixture-of-Decoders) with EasyDeL eLarge.

Trains a language model by finding the Nash equilibrium policy through
iterative mirror descent with a mixture-of-decoders formulation. The
current policy generates completions that are scored by a reward function,
and the policy is updated using mirror descent toward the Nash
equilibrium -- the point where no further improvement is possible against
the mixture of the current policy and the reference model.

How it works:
    1. ``eLargeModel`` loads and shards the policy model across devices
       using the ``loader`` and ``sharding`` sections of the config. A
       frozen reference copy is automatically created for mirror descent.
    2. Each training step uses eSurge (EasyDeL's paged-attention inference
       engine configured in the ``esurge`` section) to generate completions,
       which are scored by the provided ``reward_funcs``.
    3. The ``mixture_coef`` parameter controls the interpolation between
       the current policy and the reference model when constructing the
       mixture-of-decoders target distribution.
    4. Nash-MD applies a mirror descent update rule with KL regularization
       (scaled by ``beta``) that provably converges to the Nash equilibrium
       of the preference game, providing stronger theoretical guarantees
       than standard DPO or PPO.
    5. Generation previews at ``generation_interval`` steps let you monitor
       output quality throughout training.

Usage:
    python examples/post_training/nash_md.py
    python examples/post_training/nash_md.py --model Qwen/Qwen3-8B --wandb
    python examples/post_training/nash_md.py --batch_size 8 --max_length 4096
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class NashMDArgs:
    """Command-line arguments for Nash-MD training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/nash-md",
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


def length_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Length-based reward function for Nash-MD.

    Scores completions based on response length, rewarding detailed
    answers up to a soft ceiling of 200 words. Replace with a learned
    reward model or task-specific verifier for production use.
    """
    return [min(len(c.split()) / 200.0, 1.0) for c in completions]


def main():
    parser = DataClassArgumentParser(NashMDArgs, description="Nash-MD Training with EasyDeL")
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
                "trainer_type": "nash-md",
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
                "mixture_coef": 0.5,
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
                    "Explain the halting problem and why it fundamentally limits what automated software verification tools can guarantee.",
                    "Design a fair algorithm to divide a heterogeneous set of goods among N agents with different valuations, and prove it satisfies envy-freeness.",
                    "What are the philosophical implications of Godel's incompleteness theorems for the foundations of mathematics and artificial intelligence?",
                ],
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
            },
        }
    )

    train_dataset = load_dataset(args.dataset, split="train[:10%]")

    elm.train(
        train_dataset=train_dataset,
        reward_funcs=[length_reward],
    )


if __name__ == "__main__":
    main()
