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

"""RLVR trainer smoke test with GSM8K math verification.

Uses the GSM8K dataset with a MathVerifier to train a model using
Reinforcement Learning with Verifiable Rewards. The verifier extracts
``\\boxed{}`` or ``####`` answers from completions and compares them
against gold answers for binary rewards.

This is the single-turn variant of agentic training: generate a
completion, verify correctness, compute group-relative advantages,
and update the policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from .._common import (
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    logger.info("Loading GSM8K dataset...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    logger.info("Loaded %d GSM8K problems.", len(gsm8k))

    trainer_args = make_config(
        ed.RLVRConfig,
        "rlvr-gsm8k",
        overrides={
            "max_prompt_length": 1024,
            "max_completion_length": 1024,
            "max_length": 2048,
            "num_return_sequences": 4,
            "answer_key": "answer",
            "format_pattern": r"\\boxed\{.+\}",
            "format_reward_weight": 0.1,
            "max_len_mask": True,
            "loss_type": "dapo",
            "beta": 0.04,
            "scale_rewards": "group",
        },
    )

    def format_gsm8k(example):
        return {
            "prompt": [{"role": "user", "content": example["question"]}],
            "answer": example["answer"],
        }

    train_dataset = gsm8k.map(format_gsm8k, remove_columns=gsm8k.column_names)

    logger.info("Launching RLVR trainer on GSM8K.")
    trainer = ed.RLVRTrainer(
        arguments=trainer_args,
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("RLVR GSM8K run finished.")


if __name__ == "__main__":
    main()
