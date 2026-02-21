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

"""Smoke test for PPO trainer.

Runs a tiny PPO training loop (1 step) to exercise:
- rollout generation (optionally via eSurge)
- reward computation wiring
- PPO update (policy + value head) via the jitted step function
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

import easydel as ed

REWARD_MODEL_REPO = "Ray2333/Gemma-2B-rewardmodel-baseline"

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        load_sequence_classifier_model,
        make_config,
    )
else:
    from ._common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        load_sequence_classifier_model,
        make_config,
    )


def main() -> None:
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()
    reward_tokenizer = get_tokenizer(REWARD_MODEL_REPO)
    reward_tokenizer.padding_side = "right"
    reward_model = load_sequence_classifier_model(REWARD_MODEL_REPO)
    max_training_steps = 1024
    dataset = load_preference_dataset()
    args = make_config(
        ed.PPOConfig,
        "proximal-policy-optimization",
        overrides={
            "max_prompt_length": 128,
            # Keep total sequence lengths TPU-friendly (blocksparse kernels often
            # require multiples of 128).
            "max_completion_length": 128,
            "max_length": 256,
            "max_sequence_length": None,  # avoid deprecated `max_sequence_length` warnings
            "num_return_sequences": 2,
            "generation_num_return_sequences": 2,
            "use_esurge_generation": True,
            "generation_top_p": 0.9,
            "generation_temperature": 0.7,
            "use_wandb": False,
            "do_last_save": False,
            "save_optimizer_state": False,
            "max_training_steps": max_training_steps,
            "total_batch_size": 2,
            "gradient_accumulation_steps": 1,
        },
    )

    logger.info(f"Launching PPO trainer smoke test ({max_training_steps} step).")
    trainer = ed.PPOTrainer(
        arguments=args,
        model=model,
        reward_funcs=reward_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_processing_classes=reward_tokenizer,
    )
    output = trainer.train()
    step = int(jax.device_get(output.state.step))
    assert step >= max_training_steps, f"Expected PPO training to advance step >= {max_training_steps}, got {step}"
    logger.info("PPO run finished at step=%s.", step)


if __name__ == "__main__":
    main()
