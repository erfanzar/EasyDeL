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
        ed.SDFTConfig,
        "sdft",
        overrides={
            "max_prompt_length": 64,
            "max_completion_length": 32,
            "max_length": 96,
            "max_feedback_length": 32,
            "num_generations": 1,
            "num_return_sequences": 1,
            "teacher_prompt_template": "{prompt}\n{privileged_context}",
            "generate_from_teacher": True,
        },
    )
    train_dataset = load_preference_dataset().map(
        lambda sample: {"context": "Use the high-quality chosen response as privileged guidance."},
    )
    trainer = ed.SDFTTrainer(
        arguments=args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    logger.info("Launching SDFT trainer runtime pass.")
    trainer.train()
    logger.info("SDFT run finished.")


if __name__ == "__main__":
    main()
