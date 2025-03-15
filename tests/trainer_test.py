import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import transformers
from jax import numpy as jnp

import easydel as ed

repo_id = "meta-llama/Llama-3.2-1B-Instruct"
max_length = 2048


dtype = param_dtype = jnp.bfloat16
epoch = 3
batch_size = 24 * jax.process_count()
gradient_accum_step = 1
warmup_steps = 100
learning_rate = 1e-5
learning_rate_end = 8e-6

processor = transformers.AutoTokenizer.from_pretrained(repo_id)
processor.padding_side = "left"

if processor.pad_token is None:
	processor.pad_token = processor.eos_token


def create_dataset():
	import datasets

	dataset = datasets.load_dataset(
		"PowerInfer/QWQ-LONGCOT-500K",
		split="train",
		streaming=False,
	)

	def to_ids(sample):
		ids = processor.apply_chat_template(
			[
				{"role": "user", "content": sample["prompt"]},
				{"role": "assistant", "content": sample["response"]},
			],
			max_length=max_length,
			padding="max_length",
			return_tensors="jax",
			return_dict=True,
			truncation=True,
		)
		input_ids = ids["input_ids"]
		attention_mask = ids["attention_mask"]
		ids["input_ids"] = input_ids.squeeze(0)
		ids["attention_mask"] = attention_mask.squeeze(0)
		return ids

	return dataset.select(range(4000)).map(to_ids, remove_columns=["prompt", "response"])


arguments = ed.TrainingArguments(
	model_name="Runtime-Tests",
	num_train_epochs=epoch,
	learning_rate=learning_rate,
	learning_rate_end=learning_rate_end,
	warmup_steps=warmup_steps,
	optimizer=ed.EasyDeLOptimizers.ADAMW,
	scheduler=ed.EasyDeLSchedulers.COSINE,
	weight_decay=0.02,
	wandb_entity="erfanzar",
	total_batch_size=batch_size,
	max_sequence_length=max_length,
	gradient_accumulation_steps=gradient_accum_step,
	do_last_save=False,
	save_steps=500,
	save_total_limit=1,
	progress_bar_type="json",
	max_training_steps=100_000,
)

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
	repo_id,
	auto_shard_model=True,
	sharding_axis_dims=(1, -1, 1, 1),
	config_kwargs=ed.EasyDeLBaseConfigDict(
		freq_max_position_embeddings=max_length,
		mask_max_position_embeddings=max_length,
		gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
		kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		attn_dtype=param_dtype,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	),
	param_dtype=param_dtype,
	dtype=dtype,
	partition_axis=ed.PartitionAxis(),
)
trainer = ed.Trainer(
	model=model,
	arguments=arguments,
	dataset_train=create_dataset(),
)

output = trainer.train()

output.state.save_state("/home/erfan/model-ckpt")
