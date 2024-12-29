import os
import sys

import flax

# Local imports (assuming easydel is in parent directory or installed)
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, "..", ".."))

import easydel as ed  # this is first import

import logging
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# Constants
SEQUENCE_LENGTH = 128
NUM_TRAIN_EXAMPLES = 500
NUM_EVAL_EXAMPLES = 500
TOTAL_BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 5e-4
DATASET_NAME = "orpo-explorers/OpenHermesPreferences-10k"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def create_model_and_tokenizer(
	model_name_or_path=MODEL_NAME, sequence_length=SEQUENCE_LENGTH, dtype=jnp.float32
):
	"""Creates model and tokenizer."""
	logging.info(f"Loading model: {model_name_or_path}")
	config = ed.LlamaConfig(
		head_dim=16,
		hidden_size=64,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=1,
		intermediate_size=128,
		max_position_embeddings=sequence_length * 3,
		attn_dtype=dtype,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	)

	model = ed.LlamaForCausalLM(
		config=config,
		dtype=dtype,
		param_dtype=dtype,
		rngs=flax.nnx.Rngs(0),
	)
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	logging.info("Model and tokenizer loaded.")
	return model, tokenizer


def create_datasets(
	dataset_name=DATASET_NAME,
	train_split=NUM_TRAIN_EXAMPLES,
	eval_split=NUM_EVAL_EXAMPLES,
):
	"""Loads and splits the dataset."""
	logging.info(f"Loading dataset: {dataset_name}")
	dataset = load_dataset(dataset_name, split="train")
	train_dataset = dataset.select(range(0, train_split))
	eval_dataset = dataset.select(range(train_split, train_split + eval_split))
	logging.info(f"Train dataset size: {len(train_dataset)}")
	logging.info(f"Evaluation dataset size: {len(eval_dataset)}")
	return train_dataset, eval_dataset


def create_orpo_config(sequence_length=SEQUENCE_LENGTH, learning_rate=LEARNING_RATE):
	"""Creates ORPO config."""
	logging.info("Creating ORPO config")
	config = ed.ORPOConfig(
		model_name="odds_ratio_preference_optimization_trainer_test",
		save_directory="tmp-files",
		num_train_epochs=NUM_TRAIN_EPOCHS,
		total_batch_size=TOTAL_BATCH_SIZE,
		gradient_accumulation_steps=2,
		track_memory=False,
		use_wandb=False,
		learning_rate=learning_rate,
		do_last_save=True,
		max_prompt_length=sequence_length,
		max_length=sequence_length * 2,
		max_completion_length=sequence_length * 2,
	)
	logging.info("ORPO config created.")
	return config


def orpo_main():
	# Device selection (choose GPU if available, else CPU)
	devices = jax.devices("gpu")
	if not devices:
		logging.warning("No GPU found, using CPU.")
		devices = jax.devices("cpu")
	jax.default_device(devices[0])

	model, tokenizer = create_model_and_tokenizer()
	train_dataset, eval_dataset = create_datasets()
	orpo_config = create_orpo_config()
	if tokenizer.chat_template is None:
		tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
	trainer = ed.ORPOTrainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		tokenizer=tokenizer,
		dataset_num_proc=4,
		arguments=orpo_config,
	)

	logging.info("Starting training...")
	out = trainer.train()
	logging.info("Training finished.")
	print(out)


if __name__ == "__main__":
	orpo_main()
