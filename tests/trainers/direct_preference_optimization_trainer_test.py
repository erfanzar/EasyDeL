import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, "..", ".."))

import logging

import easydel as ed

import datasets
import flax
import jax.numpy as jnp
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_datasets(dataset_size=1000, train_split=500):
	"""Loads, preprocesses, and splits the dataset."""
	logging.info("Loading dataset...")
	dataset: datasets.Dataset = (
		datasets.concatenate_datasets(
			[
				datasets.load_dataset(
					"argilla/ultrafeedback-binarized-preferences", split="train"
				),
			]
		)
		.shuffle()
		.shuffle()
	).select(range(dataset_size))
	dataset = (
		dataset.rename_column("chosen_response", "chosen")
		.rename_column("rejected_response", "rejected")
		.rename_column("instruction", "prompt")
	)
	train_dataset = dataset.select(range(0, train_split))
	eval_dataset = dataset.select(range(train_split, dataset_size))
	logging.info(f"Train dataset size: {len(train_dataset)}")
	logging.info(f"Evaluation dataset size: {len(eval_dataset)}")
	return train_dataset, eval_dataset


def create_model_and_tokenizer(model_name_or_path):
	"""Creates model and tokenizer."""
	logging.info(f"Loading model and tokenizer: {model_name_or_path}")
	conf = ed.LlamaConfig(
		hidden_size=128,
		intermediate_size=256,
		num_hidden_layers=4,
		num_attention_heads=8,
		num_key_value_heads=4,
		max_position_embeddings=512,
		use_scan_mlp=False,
		attention_bias=False,
		attn_dtype=jnp.float32,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	)
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	model = ed.LlamaForCausalLM(
		config=conf,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=flax.nnx.Rngs(0),
	)
	ref_model = ed.LlamaForCausalLM(
		config=conf,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=flax.nnx.Rngs(0),
	)

	logging.info("Model and tokenizer created successfully.")
	return model, ref_model, tokenizer


def create_dpo_config(
	model_name="direct_preference_optimization_trainer",
	save_directory="tmp-files",
	total_batch_size=1,
	learning_rate=7e-5,
	max_length=512,
	max_completion_length=256,
	max_prompt_length=256,
):
	"""Create DPO configurations"""
	logging.info("Creating DPO config")
	config = ed.DPOConfig(
		num_train_epochs=4,
		model_name=model_name,
		save_directory=save_directory,
		loss_type="sigmoid",
		total_batch_size=total_batch_size,
		use_wandb=False,
		learning_rate=learning_rate,
		learning_rate_end=9e-6,
		warmup_steps=100,
		optimizer=ed.EasyDeLOptimizers.ADAMW,
		scheduler=ed.EasyDeLSchedulers.COSINE,
		weight_decay=0.02,
		max_sequence_length=max_length,
		gradient_accumulation_steps=1,
		step_start_point=0,
		do_last_save=False,
		training_time_limit=None,
		track_memory=True,
		max_length=max_length,
		max_completion_length=max_completion_length,
		max_prompt_length=max_prompt_length,
		beta=0.1,
		save_steps=50,
		# evaluation_steps=100,
	)
	logging.info("DPO config created successfully.")
	return config


def main():
	model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"
	train_dataset, eval_dataset = create_datasets()
	model, ref_model, tokenizer = create_model_and_tokenizer(model_name_or_path)
	arguments = create_dpo_config()

	dpo_trainer = ed.DPOTrainer(
		model=model,
		reference_model=ref_model,
		train_dataset=train_dataset,
		# eval_dataset=eval_dataset,
		processing_class=tokenizer,
		arguments=arguments,
	)

	dpo_trainer.train()


if __name__ == "__main__":
	main()
