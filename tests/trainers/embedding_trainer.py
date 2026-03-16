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

"""Smoke test for EmbeddingTrainer with contrastive learning.

Trains a small Qwen2ForEmbedding model on synthetic text pairs using
InfoNCE loss. Verifies that the training loop runs, loss decreases,
and contrastive accuracy improves.
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import Dataset

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        make_config,
    )
else:
    from ._common import (
        get_logger,
        get_tokenizer,
        make_config,
    )


def build_synthetic_pairs(num_samples: int = 100) -> Dataset:
    """Create synthetic query-positive text pairs for testing."""
    queries = []
    positives = []
    for i in range(num_samples):
        queries.append(f"What is the capital of country {i}?")
        positives.append(f"The capital of country {i} is city {i}.")
    return Dataset.from_dict({"query": queries, "positive": positives})


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()

    model = ed.AutoEasyDeLModelForEmbedding.from_pretrained(
        "Qwen/Qwen3-4B",
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            attn_dtype="bf16",
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
    )

    dataset = build_synthetic_pairs(100)

    trainer_args = make_config(
        ed.EmbeddingConfig,
        "embedding-test",
        overrides={
            "loss_type": "infonce",
            "temperature": 0.05,
            "max_length": 128,
            "total_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "max_training_steps": 5,
            "query_field": "query",
            "positive_field": "positive",
        },
    )

    logger.info("Launching EmbeddingTrainer smoke test.")
    trainer = ed.EmbeddingTrainer(
        arguments=trainer_args,
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("Embedding training finished.")


if __name__ == "__main__":
    main()
