from __future__ import annotations

import sys
from pathlib import Path

import easydel as ed
from easydel.utils.traversals import deepcopy_model

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from _common import (  # type: ignore
        build_lm_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )
else:
    from ._common import (
        build_lm_dataset,
        get_logger,
        get_tokenizer,
        load_causal_lm_model,
        make_config,
    )


def main():
    logger = get_logger(__name__)
    tokenizer = get_tokenizer()

    student_module = load_causal_lm_model()
    student_state = student_module.to_state()
    teacher_state = deepcopy_model(student_state)

    trainer_args = make_config(
        ed.DistillationConfig,
        "distillation",
        overrides={
            "alpha": 0.5,
            "temperature": 2.0,
            "hidden_state_loss_weight": 0.05,
            "hidden_state_layers": (-1,),
        },
    )

    dataset = build_lm_dataset(tokenizer)

    logger.info("Instantiating DistillationTrainer.")
    trainer = ed.DistillationTrainer(
        arguments=trainer_args,
        processing_class=tokenizer,
        student_model=student_state,
        teacher_model=teacher_state,
        train_dataset=dataset,
    )
    logger.info("Starting knowledge distillation fine-tune.")
    trainer.train()
    logger.info("Distillation run finished.")


if __name__ == "__main__":
    main()
