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

"""Smoke test for GSPO (Group Sequence Policy Optimization) trainer.

GSPO is a variant of GRPO that uses sequence-level importance sampling
instead of token-level, with smaller clipping bounds (3e-4 to 4e-4) and
no KL regularization by default.
"""

from __future__ import annotations

# pyright: reportPrivateLocalImportUsage=false
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
    from ._common import (
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
        ed.GSPOConfig,
        "gspo",
        overrides={
            "max_prompt_length": 512,
            "max_completion_length": 256,
            "max_length": 768,
            "num_train_epochs": 1,
            "total_batch_size": 2,
        },
    )

    # Verify GSPO-specific defaults are set correctly
    assert trainer_args.importance_sampling_level == "sequence", (
        f"Expected 'sequence', got {trainer_args.importance_sampling_level}"
    )
    assert trainer_args.epsilon == 3e-4, f"Expected 3e-4, got {trainer_args.epsilon}"
    assert trainer_args.epsilon_high == 4e-4, f"Expected 4e-4, got {trainer_args.epsilon_high}"
    assert trainer_args.beta == 0.0, f"Expected 0.0, got {trainer_args.beta}"

    dataset = load_preference_dataset()

    logger.info("Launching GSPO trainer smoke test.")
    logger.info(
        "GSPO config: epsilon=%s, epsilon_high=%s, beta=%s, importance_sampling_level=%s",
        trainer_args.epsilon,
        trainer_args.epsilon_high,
        trainer_args.beta,
        trainer_args.importance_sampling_level,
    )

    trainer = ed.GSPOTrainer(
        arguments=trainer_args,
        model=policy_model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("GSPO run finished.")


if __name__ == "__main__":
    main()
