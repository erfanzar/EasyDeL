# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        run_trainer,
    )
else:
    from ._common import (
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        run_trainer,
    )

REFERENCE_MODEL_REPO = "Qwen/Qwen3-4B"


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    policy_model = load_causal_lm_model()
    reference_model = load_causal_lm_model(REFERENCE_MODEL_REPO)

    trainer_args = make_config(
        ed.KTOConfig,
        "kto",
        overrides={"beta": 0.1, "loss_type": "kto"},
    )

    dataset = load_preference_dataset()

    run_trainer(
        ed.KTOTrainer,
        model=policy_model,
        arguments=trainer_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        logger=logger,
        extra_kwargs={"reference_model": reference_model},
    )


if __name__ == "__main__":
    main()
