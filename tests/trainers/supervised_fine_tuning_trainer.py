from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        build_sft_text_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from ._common import (
        build_sft_text_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()

    trainer_args = make_config(
        ed.SFTConfig,
        "supervised-fine-tuning",
        overrides={"dataset_text_field": "text", "packing": False},
    )

    dataset = build_sft_text_dataset(tokenizer=tokenizer)

    logger.info("Starting SFT run.")
    trainer = ed.SFTTrainer(
        arguments=trainer_args,
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    logger.info("SFT run finished.")


if __name__ == "__main__":
    main()
