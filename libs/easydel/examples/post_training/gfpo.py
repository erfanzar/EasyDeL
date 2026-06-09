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

"""GFPO Math Training with EasyDeL eLarge.

Trains a language model on GSM8K math word problems using Generalized Flow
Policy Optimization (GFPO). GFPO extends standard policy optimization by
treating the generation process as a flow-matching problem, optimizing the
policy to match a target distribution defined by reward-weighted trajectories.

How it works:
    1. ``eLargeModel`` loads and shards the model across devices using the
       ``loader`` and ``sharding`` sections of the config.
    2. During training, the GFPO trainer generates ``num_return_sequences``
       completions per prompt using eSurge (EasyDeL's paged-attention
       inference engine configured in the ``esurge`` section).
    3. An ``accuracy_reward`` function scores each completion by checking
       whether ``\\boxed{answer}`` matches the GSM8K ground truth.
    4. GFPO uses the reward signals to compute flow-based policy gradients,
       aligning the model's generation distribution with the reward-optimal
       trajectory distribution.
    5. lm-eval GSM8K benchmarks run automatically at 50 %% and 100 %% of
       training (controlled by ``benchmark_interval``).

Usage:
    python examples/post_training/gfpo.py
    python examples/post_training/gfpo.py --model Qwen/Qwen3-8B --wandb
    python examples/post_training/gfpo.py --batch_size 16 --num_generations 16
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class GFPOMathArgs:
    """Command-line arguments for GFPO math training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="openai/gsm8k",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/gfpo-math",
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
        metadata={"help": "Number of completions generated per prompt for GFPO."},
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


def accuracy_reward(
    prompts: list[str],
    completions: list[str],
    *,
    batch: dict | None = None,
    **kwargs,
) -> list[float]:
    """Score completions by checking if the final boxed answer matches ground truth.

    Extracts ``\\boxed{...}`` from the completion and compares against the
    ground-truth answer from the dataset's ``answer`` column. Returns 1.0
    for correct, 0.0 for incorrect or missing answers.
    """
    if batch is None or "answer" not in batch:
        return [0.0] * len(completions)

    answers = batch["answer"]
    rewards = []
    for completion, gold in zip(completions, answers, strict=False):
        pred = _extract_boxed(completion)
        gold_clean = _normalize_number(str(gold))
        pred_clean = _normalize_number(pred) if pred else ""
        rewards.append(1.0 if pred_clean == gold_clean else 0.0)
    return rewards


def _extract_boxed(text: str) -> str | None:
    """Extract the content inside ``\\boxed{...}`` from model output."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def _normalize_number(s: str) -> str:
    """Normalize a numeric string for comparison.

    Handles commas, dollar signs, trailing zeros, and other formatting
    differences between predicted and ground-truth answers.
    """
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".")
    try:
        return str(float(s))
    except ValueError:
        return s.lower().strip()


def prepare_dataset(dataset_name: str = "openai/gsm8k", split: str = "train"):
    """Load and format GSM8K for GFPO training.

    Converts each example into a prompt with the question and extracts
    the numeric answer from GSM8K's ``answer`` field (which contains
    step-by-step reasoning followed by ``#### <number>``).
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
    parser = DataClassArgumentParser(GFPOMathArgs, description="GFPO Math Training with EasyDeL")
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
                "trainer_type": "gfpo",
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
                "use_wandb": args.wandb,
                "wandb_entity": args.wandb_entity,
                "generation_interval": 50,
                "generation_prompts": [
                    "A store offers a 20%% discount on all items, then an additional 15%% off the discounted price for members. If a jacket originally costs $180, how much does a member pay?",
                    "In how many ways can you distribute 12 identical balls into 4 distinct boxes such that each box has at least 2 balls?",
                    "A car travels the first half of a journey at 40 km/h and the second half at 60 km/h. If the total distance is 240 km, what is the average speed for the entire journey?",
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

    elm.train(
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward],
    )


if __name__ == "__main__":
    main()
