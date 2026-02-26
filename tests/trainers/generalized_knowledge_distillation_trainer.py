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
    student_model = load_causal_lm_model()
    student_state = student_model.to_state()
    teacher_model = load_causal_lm_model("Qwen/Qwen3-4B")
    teacher_state = teacher_model.to_state()

    trainer_args = make_config(
        ed.GKDConfig,
        "gkd",
        overrides={
            "dataset_text_field": None,
            "lmbda": 0.0,  # disable on-policy sampling for the smoke test
            "max_new_tokens": 16,
        },
    )

    dataset = build_lm_dataset(tokenizer)

    logger.info("Instantiating GKDTrainer.")
    trainer = ed.GKDTrainer(
        arguments=trainer_args,
        processing_class=tokenizer,
        model=student_state,
        teacher_model=teacher_state,
        train_dataset=dataset,
    )
    logger.info("Launching GKD training.")
    trainer.train()
    logger.info("GKD training run finished.")


if __name__ == "__main__":
    main()
