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
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )
else:
    from .._common import (
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
    trainer_args = make_config(ed.DPOConfig, "bema-dpo")
    callback = ed.BEMACallback(
        update_freq=1,
        update_after=0,
        update_ref_model=True,
        ref_model_update_freq=1,
        ref_model_update_after=0,
    )
    trainer = ed.BEMADPOTrainer(
        arguments=trainer_args,
        model=model,
        train_dataset=load_preference_dataset(),
        processing_class=tokenizer,
        bema_callback=callback,
    )
    logger.info("Launching BEMA DPO trainer runtime pass.")
    trainer.train()
    logger.info("BEMA DPO run finished.")


if __name__ == "__main__":
    main()
