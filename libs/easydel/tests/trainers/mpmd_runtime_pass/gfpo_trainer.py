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

"""Smoke test for GFPO (Group Filtered Policy Optimization) trainer.

GFPO generates more samples per prompt, then filters to keep the most
efficient ones based on length and reward-per-token metrics. This reduces
response length inflation while maintaining accuracy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        mpmd_generation_length_overrides,
    )
else:
    from ._common import (
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        mpmd_generation_length_overrides,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    policy_model = load_causal_lm_model()

    trainer_args = make_config(
        ed.GFPOConfig,
        "gfpo",
        overrides={
            **mpmd_generation_length_overrides(),
            "num_train_epochs": 1,
            "total_batch_size": 2,
            "num_generations": 8,  # Generate more samples
            "num_remains_in_group": 4,  # Keep top 4 after filtering
            "generation_num_return_sequences": 8,
            "filter_by_length": True,
            "filter_by_efficiency": True,
        },
    )

    lightweight = os.environ.get("EASYDEL_RUNTIME_LIGHTWEIGHT", "0").lower() in {"1", "true", "yes", "on"}
    expected_generations = 3 if lightweight else 8
    expected_remains = 2 if lightweight else 4
    assert trainer_args.num_generations == expected_generations, (
        f"Expected {expected_generations}, got {trainer_args.num_generations}"
    )
    assert trainer_args.num_remains_in_group == expected_remains, (
        f"Expected {expected_remains}, got {trainer_args.num_remains_in_group}"
    )
    trainer_args.generation_num_return_sequences = expected_generations
    trainer_args.num_return_sequences = expected_generations
    assert trainer_args.filter_by_length is True
    assert trainer_args.filter_by_efficiency is True

    dataset = load_preference_dataset()

    logger.info("Launching GFPO trainer smoke test.")
    logger.info(
        "GFPO config: num_generations=%s, num_remains_in_group=%s",
        trainer_args.num_generations,
        trainer_args.num_remains_in_group,
    )

    trainer = ed.GFPOTrainer(
        arguments=trainer_args,
        model=policy_model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    output = trainer.train()
    step = int(jax.device_get(output.state.step))
    assert step >= int(os.environ.get("EASYDEL_MPMD_MAX_TRAINING_STEPS", "1")), (
        f"Expected GFPO training to advance, got step={step}"
    )
    logger.info("GFPO run finished at step=%s.", step)


if __name__ == "__main__":
    main()
