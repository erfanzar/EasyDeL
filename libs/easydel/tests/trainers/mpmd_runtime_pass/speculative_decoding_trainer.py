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

"""Runtime-pass script for the speculative-decoding trainer (MPMD).

Builds a causal-LM drafter and target model, constructs a speculative-decoding
training configuration, and launches a ``SpeculativeDecodingTrainer`` fine-tuning
run under the MPMD (multi-process multi-device) runtime pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed
from easydel.utils.traversals import deepcopy_model

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
    """Run a speculative-decoding trainer fine-tuning loop.

    Loads a tokenizer and causal-LM model, deep-copies the model state to
    obtain both drafter and target weights, builds a speculative-decoding
    training configuration and dataset, instantiates a
    ``SpeculativeDecodingTrainer``, and runs training.
    """
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()

    drafter_module = load_causal_lm_model()
    drafter_state = drafter_module.to_state()
    target_state = deepcopy_model(drafter_state)

    trainer_args = make_config(
        ed.SpeculativeDecodingConfig,
        "speculative_decoding",
        overrides={
            "alpha": 0.8,
            "temperature": 1.0,
            "num_draft_tokens": 1,
        },
    )

    dataset = build_lm_dataset(tokenizer)

    logger.info("Instantiating SpeculativeDecodingTrainer.")
    trainer = ed.SpeculativeDecodingTrainer(
        arguments=trainer_args,
        processing_class=tokenizer,
        drafter_model=drafter_state,
        target_model=target_state,
        train_dataset=dataset,
    )
    logger.info("Starting speculative-decoding drafter fine-tune.")
    trainer.train()
    logger.info("Speculative-decoding run finished.")


if __name__ == "__main__":
    main()
