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
        build_prompt_dataset,
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from .._common import (
        build_prompt_dataset,
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()
    args = make_config(
        ed.OnlineDPOConfig,
        "online-dpo",
        overrides={"max_prompt_length": 512, "max_completion_length": 512, "max_length": 1024, "max_new_tokens": 512},
    )
    trainer = ed.OnlineDPOTrainer(
        arguments=args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=build_prompt_dataset(),
        processing_class=tokenizer,
    )
    logger.info("Launching OnlineDPO trainer runtime pass.")
    trainer.train()
    logger.info("OnlineDPO run finished.")


if __name__ == "__main__":
    main()
