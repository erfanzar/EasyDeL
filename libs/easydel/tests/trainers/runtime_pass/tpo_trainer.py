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
    from _common import build_tpo_dataset, get_logger, get_tokenizer, load_causal_lm_model, make_config  # type: ignore
else:
    from .._common import build_tpo_dataset, get_logger, get_tokenizer, load_causal_lm_model, make_config


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()
    args = make_config(ed.TPOConfig, "tpo")
    trainer = ed.TPOTrainer(
        arguments=args,
        model=model,
        train_dataset=build_tpo_dataset(),
        processing_class=tokenizer,
    )
    logger.info("Launching TPO trainer runtime pass.")
    trainer.train()
    logger.info("TPO run finished.")


if __name__ == "__main__":
    main()
