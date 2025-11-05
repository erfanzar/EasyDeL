from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        dummy_reward_fn,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
    )
else:
    from ._common import (
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

    trainer_args = make_config(
        ed.GRPOConfig,
        "group-relative-policy-optimization",
        overrides={
            "max_prompt_length": 64,
            "max_completion_length": 64,
            "max_sequence_length": 128,
            "num_return_sequences": 2,
        },
    )

    dataset = load_preference_dataset()

    logger.info("Launching GRPO trainer smoke test.")
    trainer = ed.GRPOTrainer(
        arguments=trainer_args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("GRPO run finished.")


if __name__ == "__main__":
    main()
