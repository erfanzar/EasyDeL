# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed
from easydel.utils.traversals import deepcopy_model

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from _common import build_lm_dataset, get_logger, get_tokenizer, load_causal_lm_model, make_config  # type: ignore
else:
    from .._common import build_lm_dataset, get_logger, get_tokenizer, load_causal_lm_model, make_config


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    student_state = load_causal_lm_model().to_state()
    teacher_state = deepcopy_model(student_state)
    args = make_config(
        ed.GOLDConfig,
        "gold",
        overrides={"alpha": 0.5, "temperature": 2.0, "num_generations": 1, "num_return_sequences": 1},
    )
    trainer = ed.GOLDTrainer(
        arguments=args,
        processing_class=tokenizer,
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=build_lm_dataset(tokenizer),
    )
    logger.info("Launching GOLD trainer runtime pass.")
    trainer.train()
    logger.info("GOLD run finished.")


if __name__ == "__main__":
    main()
