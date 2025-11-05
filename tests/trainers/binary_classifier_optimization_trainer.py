from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        run_trainer,
    )
else:
    from ._common import (
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        load_preference_dataset,
        make_config,
        run_trainer,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    trainer_args = make_config(ed.BCOConfig, "binary-classifier-optimization", overrides={"beta": 0.1})

    dataset = load_preference_dataset()

    run_trainer(
        ed.BCOTrainer,
        model=model,
        arguments=trainer_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        logger=logger,
    )


if __name__ == "__main__":
    main()
