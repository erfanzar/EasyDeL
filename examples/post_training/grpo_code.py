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

"""GRPO Code Training with EasyDeL eLarge.

Trains a coding model using GRPO on verifiable Python coding problems.
Each prompt asks the model to solve a coding problem; the generated code
is executed against test cases and scored automatically.

How it works:
    1. ``eLargeModel`` loads and shards the model, then compiles eSurge
       for fast batched generation during the GRPO rollout phase.
    2. For each prompt the model generates ``num_generations`` candidate
       solutions. Two reward functions score each completion:
       - ``code_reward``: Extracts the Python code block from the completion,
         executes it against the test cases in a sandboxed subprocess, and
         returns the fraction of tests passed (0.0–1.0).
       - ``format_reward``: Checks that the completion wraps code in a
         proper markdown ```python ... ``` block with reasoning tags.
    3. GRPO computes group-relative advantages and updates the policy with
       the DAPO loss variant.

Dataset format:
    The dataset must have a ``problem_statement`` column (the coding prompt)
    and a ``verification_info`` column containing ``{"test_cases": [...]}``
    with input/output pairs for execution-based verification.

Usage:
    python examples/post_training/grpo_code.py
    python examples/post_training/grpo_code.py --model Qwen/Qwen3-4B --max_steps 200
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel.infra.elarge import eLargeModel


@dataclass
class GRPOCodeArgs:
    """Command-line arguments for GRPO code training."""

    model: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Model name or HuggingFace path."},
    )
    dataset: str = field(
        default="open-r1/verifiable-coding-problems-python",
        metadata={"help": "HuggingFace dataset with coding problems."},
    )
    output_dir: str = field(
        default="outputs/grpo-code",
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
        metadata={"help": "Number of completions generated per prompt for GRPO."},
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


def _extract_python_code(text: str) -> str:
    """Extract Python code from a markdown code block."""
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else ""


def _run_code_against_tests(code: str, test_cases: list[dict]) -> float:
    """Execute code against test cases in a subprocess and return pass rate."""
    if not code or not test_cases:
        return 0.0

    passed = 0
    for case in test_cases:
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                input=case.get("input", ""),
                text=True,
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                continue
            expected = case.get("output", "").strip()
            actual = result.stdout.strip()
            if all(
                a.strip() == e.strip()
                for a, e in zip(actual.splitlines(), expected.splitlines(), strict=False)
            ):
                passed += 1
        except (subprocess.TimeoutExpired, OSError):
            continue
    return passed / len(test_cases)


def code_reward(
    prompts: list[str],
    completions: list[str],
    *,
    batch: dict | None = None,
    **kwargs,
) -> list[float]:
    """Score completions by extracting and executing Python code against test cases.

    Extracts ``python ... `` code blocks from completions, runs them in a
    sandboxed subprocess with each test case's input, and checks stdout
    against expected output. Returns the fraction of tests passed (0.0–1.0).
    """
    if batch is None or "verification_info" not in batch:
        return [0.0] * len(completions)

    verification_info = batch["verification_info"]
    rewards = []
    for completion, info in zip(completions, verification_info, strict=False):
        code = _extract_python_code(completion)
        if isinstance(info, str):
            info = json.loads(info)
        test_cases = info.get("test_cases", [])
        rewards.append(_run_code_against_tests(code, test_cases))
    return rewards


def format_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Score completions for proper code formatting.

    Awards partial credit for:
    - 0.25 for ``<think>`` reasoning block
    - 0.25 for ``</think>`` closing tag
    - 0.25 for a ```python code block
    - 0.25 for closing ``` after the code
    """
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion:
            score += 0.25
        if "</think>" in completion:
            score += 0.25
        if "```python" in completion:
            score += 0.25
        if completion.count("```") >= 2:
            score += 0.25
        rewards.append(score)
    return rewards


def prepare_dataset(dataset_name: str, prompt_column: str = "problem_statement"):
    """Load and format a coding problems dataset for GRPO training."""
    ds = load_dataset(dataset_name, split="train")

    def _format(example):
        return {
            "prompt": [{"role": "user", "content": example[prompt_column]}],
            "verification_info": example.get("verification_info", "{}"),
        }

    return ds.map(_format, remove_columns=[c for c in ds.column_names if c != "verification_info"])


def main():
    parser = DataClassArgumentParser(GRPOCodeArgs, description="GRPO Code Training with EasyDeL")
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
                "trainer_type": "grpo",
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
                    "Write a Python function that finds the longest palindromic substring in a given string.",
                    "Implement a function that merges k sorted linked lists into one sorted list.",
                    "Write a Python class that implements a min-heap with insert, extract_min, and peek operations.",
                ],
                "benchmark_interval": half_steps,
                "benchmarks": [
                    {
                        "name": "humaneval",
                        "tasks": ["humaneval"],
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
        reward_funcs=[code_reward, format_reward],
    )


if __name__ == "__main__":
    main()
