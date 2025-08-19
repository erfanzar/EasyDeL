import logging

import flax
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

import easydel as ed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants (can be moved to a config if needed)
SEQUENCE_LENGTH = 128
NUM_TRAIN_EPOCHS = 4
TOTAL_BATCH_SIZE = 8
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_NAME = "LDJnr/Pure-Dove"
TRAIN_SPLIT = 500
EVAL_SPLIT = 1000
LEARNING_RATE = 3e-4
WARMUP_STEPS = 5


def create_model_and_tokenizer(sequence_length=SEQUENCE_LENGTH, dtype=jnp.float32):
    """Creates model and tokenizer."""
    logging.info(f"Loading model: {MODEL_NAME}")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ed.LlamaForCausalLM(
        config=config,
        dtype=dtype,
        param_dtype=dtype,
        rngs=flax.nnx.Rngs(0),
    )

    model = model.shard_model()
    logging.info("Model and tokenizer created successfully.")

    return model, tokenizer


def create_datasets(dataset_name=DATASET_NAME, train_split=TRAIN_SPLIT, eval_split=EVAL_SPLIT):
    """Loads and splits the dataset."""
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    train_dataset = dataset.select(range(train_split))
    eval_dataset = dataset.select(range(train_split, eval_split))
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Evaluation dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def create_sft_config(
    sequence_length=SEQUENCE_LENGTH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
):
    """Creates SFT config."""
    logging.info("Creating SFT config")
    config = ed.SFTConfig(
        save_directory="tmp-files",
        model_name="SFT-TrainerTest",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        total_batch_size=TOTAL_BATCH_SIZE,
        gradient_accumulation_steps=2,
        do_train=True,
        do_eval=True,
        max_sequence_length=sequence_length,
        track_memory=True,
        use_wandb=False,
        learning_rate=learning_rate,
        do_last_save=True,
        save_steps=350,
        save_total_limit=5,
        save_optimizer_state=True,
        training_time_limit="80Min",
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        clip_grad=1.0,
        warmup_steps=warmup_steps,
        packing=True,
        num_of_sequences=sequence_length,
        evaluation_steps=200,
    )
    logging.info("SFT config created successfully.")
    return config


def main():
    # Device selection (choose GPU if available, else CPU)
    devices = jax.devices("gpu")
    if not devices:
        logging.warning("No GPU found, using CPU.")
        devices = jax.devices("cpu")

    model, tokenizer = create_model_and_tokenizer()
    train_dataset, eval_dataset = create_datasets()
    sft_config = create_sft_config()
    prompter = ed.trainers.create_prompt_creator(tokenizer)

    trainer = ed.SFTTrainer(
        arguments=sft_config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=prompter,
    )
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")


if __name__ == "__main__":
    main()
