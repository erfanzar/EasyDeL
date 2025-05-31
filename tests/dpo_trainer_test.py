import logging

import jax
from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    REPO = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info("Loading tokenizer from the repository.")

    tokenizer = AutoTokenizer.from_pretrained(REPO)

    logger.info("Loading model with specified configuration and precision settings.")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        REPO,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism="vanilla",
            attn_dtype=jnp.float32,
            attn_softmax_dtype=jnp.float32,
        ),
        precision=jax.lax.Precision.HIGHEST,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    # Define training arguments
    trainer_args = ed.DPOConfig(
        max_completion_length=256,
        max_prompt_length=256,
        max_length=512,
        num_train_epochs=1,
        total_batch_size=1,
        log_steps=5,
        do_last_save=True,
        use_wandb=False,
        shuffle_train_dataset=False,
        save_optimizer_state=False,
        progress_bar_type="json",
        save_steps=100,
        save_total_limit=1,
        learning_rate=1e-6,
    )

    logger.info("Loading a subset of the training dataset.")
    # Load a 10% subset of the training dataset
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:10%]")

    logger.info("Initializing the DPOTrainer.")
    # Initialize the trainer with the model, training arguments, dataset, and tokenizer
    trainer = ed.DPOTrainer(
        model=model,
        arguments=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training.")
    # Start the training process
    trainer.train()
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
