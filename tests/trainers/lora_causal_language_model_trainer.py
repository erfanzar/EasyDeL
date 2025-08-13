import logging

import flax
import jax
import jax.numpy as jnp
from datasets import Dataset, IterableDataset

import easydel as ed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
TOTAL_BATCH_SIZE = 8
UPPER = 600
NUM_TRAIN_EPOCHS = 2
SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-4
WARMUP_STEPS = 5
SAVE_STEPS = 10

# Derived Constants
NUM_TRAIN_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_EVAL_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
MAX_TRAINING_STEPS = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
MAX_EVALUATION_STEPS = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE


def create_model(sequence_length=SEQUENCE_LENGTH, dtype=jnp.float32):
    """Creates the model."""
    logging.info("Creating model...")
    config = ed.LlamaConfig(
        head_dim=16,
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=128,
        max_position_embeddings=sequence_length,
        attn_dtype=dtype,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    )

    model = ed.LlamaForCausalLM(
        config=config,
        dtype=dtype,
        param_dtype=dtype,
        rngs=flax.nnx.Rngs(0),
    )

    model = model.shard_model()
    model = model.apply_lora_to_layers(32, ".*(q_proj|k_proj).*")
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
        model_name="TrainerTest",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        total_batch_size=TOTAL_BATCH_SIZE,
        gradient_accumulation_steps=2,
        max_training_steps=max_training_steps,
        max_evaluation_steps=max_evaluation_steps,
        do_train=True,
        do_eval=False,
        max_sequence_length=sequence_length,
        track_memory=True,
        use_wandb=False,
        learning_rate=learning_rate,
        do_last_save=True,
        # save_steps=SAVE_STEPS,
        # save_total_limit=5,
        # save_optimizer_state=True,
        training_time_limit="10sec",
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        clip_grad=1.0,
        warmup_steps=warmup_steps,
    )
    logging.info("Training arguments created.")
    return training_args


def main(use_iterable_dataset: bool = True):
    # Device selection (choose GPU if available, else CPU)
    devices = jax.devices("gpu")
    if not devices:
        logging.warning("No GPU found, using CPU.")
        devices = jax.devices("cpu")

    model = create_model()
    train_dataset = create_dummy_dataset(
        NUM_TRAIN_EXAMPLES,
        use_iterable_dataset=use_iterable_dataset,
    )
    eval_dataset = create_dummy_dataset(
        NUM_EVAL_EXAMPLES,
        use_iterable_dataset=use_iterable_dataset,
    )
    training_args = create_training_args()

    trainer = ed.Trainer(
        arguments=training_args,
        model=model,
        dataset_train=train_dataset,
        dataset_eval=eval_dataset,
    )
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")


if __name__ == "__main__":
    main(use_iterable_dataset=True)
