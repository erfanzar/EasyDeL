# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import sys
from pathlib import Path

from datasets import Dataset

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from _common import get_logger, get_tokenizer, load_causal_lm_model, make_config  # type: ignore
else:
    from .._common import get_logger, get_tokenizer, load_causal_lm_model, make_config


class _Env:
    def __init__(self, metadata=None, agent_ref=None, request_timeout=None):
        self.metadata = metadata or {}
        self.agent_ref = agent_ref
        self.request_timeout = request_timeout

    def step(self, action):
        del action
        return {"observation": "ok", "reward": float(self.metadata.get("reward", 0.1)), "terminated": True}


def _environment_factory(metadata=None, agent_ref=None, request_timeout=None):
    return _Env(metadata=metadata, agent_ref=agent_ref, request_timeout=request_timeout)


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()
    args = make_config(
        ed.NeMoGymConfig,
        "nemo-gym",
        overrides={
            "max_prompt_length": 512,
            "max_completion_length": 512,
            "max_length": 1024,
            "num_generations_eval": 1,
        },
    )
    dataset = Dataset.from_list(
        [
            {"prompt": "Say ok.", "metadata": {"reward": 0.1}, "agent_ref": {"name": "smoke-a"}},
            {"prompt": "Say done.", "metadata": {"reward": 0.2}, "agent_ref": {"name": "smoke-b"}},
        ]
    )
    trainer = ed.NeMoGymTrainer(
        arguments=args,
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=_environment_factory,
    )
    logger.info("Launching NeMoGym trainer runtime pass.")
    trainer.train()
    logger.info("NeMoGym run finished.")


if __name__ == "__main__":
    main()
