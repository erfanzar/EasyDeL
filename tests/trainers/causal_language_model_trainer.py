import logging

import jax.numpy as jnp
from datasets import Dataset, IterableDataset

import easydel as ed
from easydel.infra.loss_utils import LossConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TOTAL_BATCH_SIZE = 5
UPPER = 2000
NUM_TRAIN_EPOCHS = 4
SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-4
WARMUP_STEPS = 5
SAVE_STEPS = 1000
DO_LAST_SAVE = False
# Derived Constants
NUM_TRAIN_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_EVAL_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
MAX_TRAINING_STEPS = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
MAX_EVALUATION_STEPS = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE


def create_model(sequence_length=SEQUENCE_LENGTH, dtype=jnp.float32):
    """Creates the model."""
    logging.info("Creating model...")
    config = ed.Xerxes2Config(
        vocab_size=32000,
        hidden_size=64,
        num_attention_heads=8,
        num_hidden_layers=4,
        intermediate_size=128,
        max_position_embeddings=sequence_length,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        num_experts=2,
        num_experts_per_tok=1,
    )

    model = ed.Xerxes2ForCausalLM(config=config, dtype=dtype, param_dtype=dtype, rngs=ed.Rngs(0))

    model = model.shard_model()
    logging.info("Model created.")
    return model


def create_dummy_dataset(
    num_rows: int,
    sequence_length: int = SEQUENCE_LENGTH,
    use_iterable_dataset: bool = False,
):
    """Creates a dummy dataset."""
    logging.info(f"Creating {'iterable' if use_iterable_dataset else 'regular'} dataset with {num_rows} rows.")

    def data_generator(num_rows: int):
        ones = jnp.ones((sequence_length,), dtype="i4")
        for _ in range(num_rows):
            yield {
                "attention_mask": ones,
                "input_ids": ones.at[-1].set(0),
                "labels": ones.at[-1].set(0),
            }

    if not use_iterable_dataset:
        dataset = Dataset.from_generator(data_generator, gen_kwargs={"num_rows": num_rows})
    else:
        dataset = IterableDataset.from_generator(
            data_generator,
            gen_kwargs={"num_rows": num_rows},
        )

    return dataset


def create_training_args(
    sequence_length=SEQUENCE_LENGTH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_training_steps=MAX_TRAINING_STEPS,
    max_evaluation_steps=MAX_EVALUATION_STEPS,
):
    """Creates training arguments."""
    logging.info("Creating training arguments...")
    training_args = ed.TrainingArguments(
        save_directory="tmp-files",
        loss_config=LossConfig(z_loss=0.0008),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        total_batch_size=TOTAL_BATCH_SIZE,
        gradient_accumulation_steps=2,
        max_training_steps=max_training_steps,
        max_evaluation_steps=max_evaluation_steps,
        do_train=True,
        do_eval=True,
        max_sequence_length=sequence_length,
        track_memory=True,
        use_wandb=False,
        weight_distribution_log_steps=5,
        learning_rate=learning_rate,
        do_last_save=DO_LAST_SAVE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        save_optimizer_state=True,
        per_epoch_training_steps=NUM_TRAIN_EXAMPLES,
        per_epoch_evaluation_steps=NUM_TRAIN_EXAMPLES,
        # training_time_limit="80Min",
        wandb_entity="erfanzar",
        model_name="CausalLanguageModelTrainerTest",
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        clip_grad=1.0,
        warmup_steps=warmup_steps,
        report_steps=10,
        log_steps=100,
        progress_bar_type="tqdm",
        use_grain=True,
        resume_if_possible=True,
    )
    logging.info("Training arguments created.")
    return training_args


def main(use_iterable_dataset: bool = False):
    model = create_model()
    train_dataset = create_dummy_dataset(NUM_TRAIN_EXAMPLES, use_iterable_dataset=use_iterable_dataset)
    eval_dataset = create_dummy_dataset(NUM_EVAL_EXAMPLES, use_iterable_dataset=use_iterable_dataset)
    training_args = create_training_args()

    trainer = ed.Trainer(
        arguments=training_args,
        model=model,
        dataset_train=train_dataset,
        dataset_eval=eval_dataset,
    )

    # logging.info("Compiling AOT...")
    # trainer.compile_aot()
    # logging.info("AOT COMP finished.")

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")


if __name__ == "__main__":
    main(use_iterable_dataset=False)
