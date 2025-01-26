import os
import sys

import jax

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import flax
import transformers
from jax import numpy as jnp

import easydel as ed

MODEL_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
MAX_LENGTH = 1024
SHARDING_AXIS_DIMS = (1, -1, 1, 1)

DTYPE = jnp.bfloat16
PARAM_DTYPE = jnp.bfloat16
EPOCHS = 3
BATCH_SIZE = 8 * jax.process_count()
GRAD_ACCUMULATION_STEPS = 1
WARPUP_STEPS = 100
LEARNING_RATE = 1e-5
LEARNING_RATE_END = 8e-6

if jax.process_index() == 0:
	print("DTYPE", DTYPE)
	print("PARAM_DTYPE", PARAM_DTYPE)
	print("EPOCHS", EPOCHS)
	print("BATCH_SIZE", BATCH_SIZE)
	print("GRAD_ACCUMULATION_STEPS", GRAD_ACCUMULATION_STEPS)
	print("WARPUP_STEPS", WARPUP_STEPS)
	print("LEARNING_RATE", LEARNING_RATE)
	print("LEARNING_RATE_END", LEARNING_RATE_END)

TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_REPO_ID)
TOKENIZER.padding_side = "left"
TOKENIZER.pad_token = TOKENIZER.eos_token

PA_AXIS = ed.PartitionAxis()


def create_dataset():
	import datasets

	dataset = datasets.load_dataset(
		"PowerInfer/QWQ-LONGCOT-500K",
		split="train[:5%]",
		streaming=False,
	)

	def to_ids(sample):
		ids = TOKENIZER.apply_chat_template(
			[
				{"role": "user", "content": sample["prompt"]},
				{"role": "assistant", "content": sample["response"]},
			],
			max_length=MAX_LENGTH,
			padding="max_length",
			return_tensors="jax",
			return_dict=True,
			truncation=True,
		)
		input_ids = ids["input_ids"]
		attention_mask = ids["attention_mask"]
		ids["input_ids"] = input_ids.squeeze(0)  # [-MAX_LENGTH:]
		ids["attention_mask"] = attention_mask.squeeze(0)  # [-MAX_LENGTH:]
		return ids

	return dataset.map(
		to_ids,
		# batched=False,
		remove_columns=["prompt", "response"],
		# num_proc=os.cpu_count(),
	)


if ed.__version__ == "0.1.0":
	TRAIN_ARGUMENTS = ed.TrainingArguments(
		model_name="Runtime-Tests",
		save_directory="/home/erfan/runner/model",
		num_train_epochs=EPOCHS,
		learning_rate=LEARNING_RATE,
		learning_rate_end=LEARNING_RATE_END,
		warmup_steps=WARPUP_STEPS,
		optimizer=ed.EasyDeLOptimizers.ADAMW,
		scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
		weight_decay=0.02,
		total_batch_size=BATCH_SIZE,
		max_sequence_length=MAX_LENGTH,
		gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
		do_last_save=False,
		save_steps=500,
		save_total_limit=1,
		progress_bar_type="json",
		max_training_steps=100_000,
	)

	MODEL = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		MODEL_REPO_ID,
		auto_shard_model=True,
		sharding_axis_dims=SHARDING_AXIS_DIMS,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=MAX_LENGTH,
			mask_max_position_embeddings=MAX_LENGTH,
			attn_dtype=DTYPE,
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		platform=ed.EasyDeLPlatforms.JAX,
		param_dtype=PARAM_DTYPE,
		dtype=DTYPE,
		partition_axis=PA_AXIS,
	)
	trainer = ed.Trainer(
		model=MODEL,
		arguments=TRAIN_ARGUMENTS,
		dataset_train=create_dataset(),
	)

	output = trainer.train()

	output.state.save_state("/home/erfan/model-ckpt")
elif ed.__version__ == "0.0.80":
	TRAIN_ARGUMENTS = ed.TrainingArguments(
		model_name="Runtime-Tests",
		save_dir="/home/erfan/runner/model",
		num_train_epochs=EPOCHS,
		learning_rate=LEARNING_RATE,
		learning_rate_end=LEARNING_RATE_END,
		warmup_steps=WARPUP_STEPS,
		optimizer=ed.EasyDeLOptimizers.ADAMW,
		scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
		weight_decay=0.02,
		total_batch_size=BATCH_SIZE,
		max_sequence_length=MAX_LENGTH,
		gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
		do_last_save=False,
		save_steps=500,
		save_total_limit=1,
	)
	MODEL, PARAMS = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		MODEL_REPO_ID,
		auto_shard_params=True,
		sharding_axis_dims=SHARDING_AXIS_DIMS,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=MAX_LENGTH,
			mask_max_position_embeddings=MAX_LENGTH,
			attn_dtype=DTYPE,
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		platform=ed.EasyDeLPlatforms.JAX,
		param_dtype=PARAM_DTYPE,
		dtype=DTYPE,
		partition_axis=PA_AXIS,
	)
	TRAIN_ARGUMENTS.gradient_checkpointing = (
		ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE
	)
	trainer = ed.CausalLanguageModelTrainer(
		model=MODEL,
		arguments=TRAIN_ARGUMENTS,
		dataset_train=create_dataset(),
	)

	output = trainer.train(model_parameters=flax.core.FrozenDict({"params": PARAMS}))

	output.state.save_state("/home/erfan/model-ckpt")
