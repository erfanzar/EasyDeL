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

# pyright: reportPrivateLocalImportUsage=false
import sys
from pathlib import Path

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        build_reward_dataset,
        get_logger,
        get_tokenizer,
        load_sequence_classifier_model,
        make_config,
    )
else:
    from ._common import (
        build_reward_dataset,
        get_logger,
        get_tokenizer,
        load_sequence_classifier_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_sequence_classifier_model()

    trainer_args = make_config(
        ed.RewardConfig,
        "reward-modeling",
        overrides={"total_batch_size": 1},
    )

    dataset = build_reward_dataset()

    logger.info("Initializing RewardTrainer.")
    trainer = ed.RewardTrainer(
        model=model,
        processing_class=tokenizer,
        arguments=trainer_args,
        train_dataset=dataset,
    )
    logger.info("Starting reward model training.")
    trainer.train()
    logger.info("Reward training complete.")


if __name__ == "__main__":
    main()
