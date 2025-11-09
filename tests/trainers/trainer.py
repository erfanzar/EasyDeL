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
        build_lm_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from ._common import (
        build_lm_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    trainer_args = make_config(
        ed.TrainingArguments,
        "base-trainer",
    )

    dataset = build_lm_dataset(tokenizer)

    logger.info("Running base Trainer smoke test.")
    trainer = ed.Trainer(
        arguments=trainer_args,
        model=model,
        dataset_train=dataset,
    )
    trainer.train()
    logger.info("Trainer smoke test completed.")


if __name__ == "__main__":
    main()
