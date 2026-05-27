# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

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
    model = load_causal_lm_model()
    args = make_config(
        ed.RLOOConfig,
        "rloo",
        overrides={
            "max_prompt_length": 64,
            "max_completion_length": 32,
            "max_length": 96,
            "num_return_sequences": 2,
            "num_generations": 2,
            "total_batch_size": 2,
        },
    )
    trainer = ed.RLOOTrainer(
        arguments=args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=load_preference_dataset(),
        processing_class=tokenizer,
    )
    logger.info("Launching RLOO trainer runtime pass.")
    trainer.train()
    logger.info("RLOO run finished.")


if __name__ == "__main__":
    main()
