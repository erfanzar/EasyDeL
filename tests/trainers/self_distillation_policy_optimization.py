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
    model = load_causal_lm_model()

    trainer_args = make_config(
        ed.SDPOConfig,
        "self-distillation-policy-optimization",
        overrides={
            "max_prompt_length": 1024,
            "max_completion_length": 1024,
            "max_length": 2048,
            "num_return_sequences": 4,
            "max_feedback_length": 256,
            "distillation_type": "jsd",
        },
    )

    dataset = load_preference_dataset()

    logger.info("Launching SDPO trainer smoke test (self-feedback mode).")
    trainer = ed.SDPOTrainer(
        arguments=trainer_args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("SDPO self-feedback run finished.")


def dummy_feedback_fn(*, prompts, completions, rewards, **_):
    return [f"Reward={r:.3f}" if r <= 0 else "" for r in rewards]


def main_rich_feedback():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()


    trainer_args = make_config(
        ed.SDPOConfig,
        "self-distillation-policy-optimization-rich",
        overrides={
            "max_prompt_length": 1024,
            "max_completion_length": 1024,
            "max_length": 2048,
            "num_return_sequences": 4,
            "max_feedback_length": 256,
            "distillation_type": "jsd",
        },
    )

    dataset = load_preference_dataset()

    logger.info("Launching SDPO trainer smoke test (rich feedback mode).")
    trainer = ed.SDPOTrainer(
        arguments=trainer_args,
        model=model,
        reward_funcs=dummy_reward_fn,
        feedback_func=dummy_feedback_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("SDPO rich-feedback run finished.")


if __name__ == "__main__":
    main()
