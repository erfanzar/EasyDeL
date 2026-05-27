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
        MAX_TOTAL_LENGTH,
        build_prm_dataset,
        get_logger,
        get_tokenizer,
        load_token_classifier_model,
        make_config,
    )
else:
    from .._common import (
        MAX_TOTAL_LENGTH,
        build_prm_dataset,
        get_logger,
        get_tokenizer,
        load_token_classifier_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_token_classifier_model(tokenizer)
    args = make_config(
        ed.PRMConfig,
        "prm",
        overrides={"total_batch_size": 1, "max_length": MAX_TOTAL_LENGTH, "max_completion_length": 256},
    )
    trainer = ed.PRMTrainer(
        arguments=args,
        model=model,
        processing_class=tokenizer,
        train_dataset=build_prm_dataset(),
    )
    logger.info("Launching PRM trainer runtime pass.")
    trainer.train()
    logger.info("PRM run finished.")


if __name__ == "__main__":
    main()
