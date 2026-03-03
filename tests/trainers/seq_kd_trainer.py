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
from easydel.utils.traversals import deepcopy_model

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )
else:
    from ._common import (
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()

    student_module = load_causal_lm_model()
    student_state = student_module.to_state()
    teacher_state = deepcopy_model(student_state)

    trainer_args = make_config(
        ed.SeqKDConfig,
        "seq-kd",
        overrides={
            "max_prompt_length": 256,
            "max_completion_length": 256,
            "num_generations_per_prompt": 1,
        },
    )

    dataset = load_preference_dataset()

    logger.info("Instantiating SeqKDTrainer.")
    trainer = ed.SeqKDTrainer(
        arguments=trainer_args,
        processing_class=tokenizer,
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=dataset,
    )
    logger.info("Starting SeqKD fine-tune.")
    trainer.train()
    logger.info("SeqKD run finished.")


if __name__ == "__main__":
    main()
