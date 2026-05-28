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

import os
import sys
from pathlib import Path

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from _common import (  # type: ignore
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )
else:
    from .._common import (
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

    overrides = {
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "max_length": 1024,
        "num_train_epochs": 1,
        "total_batch_size": 2,
    }
    if os.environ.get("EASYDEL_RUNTIME_LIGHTWEIGHT", "0").lower() in {"1", "true", "yes", "on"}:
        overrides.update({"alpha": 0.0, "beta": 0.0})

    trainer_args = make_config(
        ed.XPOConfig,
        "xpo",
        overrides=overrides,
    )

    dataset = load_preference_dataset()

    logger.info("Launching XPO trainer smoke test.")
    trainer = ed.XPOTrainer(
        arguments=trainer_args,
        model=policy_model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("XPO run finished.")


if __name__ == "__main__":
    main()
