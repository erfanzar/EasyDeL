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

"""GSPO (Group Sequence Policy Optimization) with EasyDeL eLarge.

Trains a language model using GSPO, which extends GRPO by operating at
the sequence level rather than the token level for policy gradient
computation. Like GRPO, it generates multiple completions per prompt and
uses group-relative advantages, but computes the importance ratio and
loss at the sequence level for more stable optimization.

How it works:
    1. ``eLargeModel`` loads and shards the model. eSurge is compiled
       for fast batched generation.
    2. For each prompt, ``num_generations`` completions are sampled.
       Reward functions score each completion.
    3. Group-relative advantages are computed within each prompt group.
    4. The GSPO loss uses sequence-level log-probability ratios
       (sum of per-token log-probs) rather than per-token ratios,
       which provides lower variance gradients at the cost of some
       granularity.

GSPO is well-suited for math reasoning and code generation tasks
where the correctness of the entire sequence matters more than
individual token decisions.

Usage:
    python examples/post_training/gspo.py
    python examples/post_training/gspo.py --model Qwen/Qwen3-8B
    python examples/post_training/gspo.py --num_generations 16 --batch_size 16
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class GSPOArgs:
    """Command-line arguments for GSPO training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="openai/gsm8k",
        metadata={"help": "HuggingFace dataset name."},
    )
    output_dir: str = field(
        default="outputs/gspo-math",
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
        metadata={"help": "Maximum total sequence length."},
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
    """Binary reward: 1.0 if boxed answer matches ground truth, else 0.0."""
    if batch is None or "answer" not in batch:
        return [0.0] * len(completions)

    answers = batch["answer"]
    rewards = []
    for completion, gold in zip(completions, answers, strict=False):
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        pred = match.group(1).strip() if match else ""
        gold_clean = str(gold).strip().replace(",", "")
        pred_clean = pred.replace(",", "")
        try:
            rewards.append(1.0 if float(pred_clean) == float(gold_clean) else 0.0)
        except ValueError:
            rewards.append(1.0 if pred_clean.lower() == gold_clean.lower() else 0.0)
    return rewards


def prepare_dataset(dataset_name: str = "openai/gsm8k", split: str = "train"):
    """Load and format GSM8K for GSPO training."""
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
    parser = DataClassArgumentParser(GSPOArgs, description="GSPO Training with EasyDeL")
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
                "trainer_type": "gspo",
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
                "scale_rewards": "group",
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
                    "A farmer has 200 meters of fencing. What dimensions should a rectangular pen have to maximize the enclosed area?",
                    "Three pipes can fill a pool in 6, 8, and 12 hours respectively. If all three are opened, how long to fill the pool?",
                    "A ball is dropped from 100 meters. Each bounce reaches 60%% of the previous height. What is the total distance traveled?",
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

    train_dataset = prepare_dataset(args.dataset)

    elm.train(
        train_dataset=train_dataset,
        reward_funcs=[accuracy_reward],
    )


if __name__ == "__main__":
    main()
