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

import sys
from pathlib import Path

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
    )
else:
    from ._common import (  # type: ignore
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    policy_model = load_causal_lm_model()

    trainer_args = make_config(
        ed.GFPOConfig,
        "gfpo",
        overrides={
            "max_prompt_length": 512,
            "max_completion_length": 256,
            "max_sequence_length": 768,
            "num_train_epochs": 1,
            "total_batch_size": 2,
            "num_generations": 8,       # Generate more samples
            "num_remains_in_group": 4,  # Keep top 4 after filtering
            "filter_by_length": True,
            "filter_by_efficiency": True,
        },
    )

    # Verify GFPO-specific parameters
    assert trainer_args.num_generations == 8, f"Expected 8, got {trainer_args.num_generations}"
    assert trainer_args.num_remains_in_group == 4, f"Expected 4, got {trainer_args.num_remains_in_group}"
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
    trainer.train()
    logger.info("GFPO run finished.")


if __name__ == "__main__":
    main()
