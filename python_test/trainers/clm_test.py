import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../../src",
	)
)
# import jax

from easydel import (
	AttentionMechanisms,
	CausalLanguageModelTrainer,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
	FlaxLlamaForCausalLM,
	LlamaConfig,
	TrainArguments,
)
import fjformer

# jax.config.update("jax_platform_name", "cpu")  # CPU Test !
import flax.core
from datasets import Dataset, IterableDataset
from jax import numpy as jnp
from jax import random

TOTAL_BATCH_SIZE = 8
UPPER = 200
NUM_TRAIN_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_EVAL_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_TRAIN_EPOCHS = 1
rng = fjformer.GenerateRNG()


def main(use_iterable_dataset: bool):
	sequence_length = 1024
	max_training_steps = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
	max_evaluation_steps = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE
	config = LlamaConfig(
		head_dim=128,
		hidden_size=512,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=4,
		intermediate_size=1024,
		max_position_embeddings=sequence_length,
		attn_dtype=jnp.float16,
		attn_mechanism=AttentionMechanisms.flash_attn2,
		block_k=64,
		block_q=128,
	)

	dtype = jnp.float16
	model = FlaxLlamaForCausalLM(
		config=config,
		_do_init=True,
		dtype=dtype,
		param_dtype=dtype,
	)
	params = model.shard_params(model.params)
	new_rng = rng.rng

	def data_generator(num_rows: int):
		for _ in range(num_rows):
			yield {
				"attention_mask": jnp.ones((sequence_length,), dtype="i4"),
				"input_ids": random.randint(new_rng, (sequence_length,), 0, 32000, dtype="i4"),
			}

	if not use_iterable_dataset:
		example_train_data = Dataset.from_generator(
			data_generator,
			gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES},
		)
		example_eval_data = Dataset.from_generator(
			data_generator,
			gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES},
		)
	else:
		example_train_data = IterableDataset.from_generator(
			data_generator,
			gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES},
		)
		example_eval_data = IterableDataset.from_generator(
			data_generator,
			gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES},
		)
	trainer = CausalLanguageModelTrainer(
		arguments=TrainArguments(
			model_name="CLM_TEST",
			num_train_epochs=NUM_TRAIN_EPOCHS,
			total_batch_size=TOTAL_BATCH_SIZE,
			gradient_accumulation_steps=2,
			max_training_steps=max_training_steps,
			max_evaluation_steps=max_evaluation_steps,
			model_class=type(model),
			do_train=True,
			do_eval=True,
			max_sequence_length=sequence_length,
			configs_to_initialize_model_class={
				"config": model.config,
				"input_shape": (1, 1),
				"dtype": dtype,
				"param_dtype": dtype,
			},
			dtype=dtype,
			param_dtype=dtype,
			track_memory=True,
			use_wandb=True,
			learning_rate=3e-4,
			label_smoothing_factor=0.1,
			train_on_inputs=True,
			do_last_save=True,
			training_time="80Min",
			optimizer=EasyDeLOptimizers.ADAMW,
			scheduler=EasyDeLSchedulers.COSINE,
			clip_grad=1.0,
			warmup_steps=5,
		),
		dataset_train=example_train_data,
		dataset_eval=example_eval_data,
	)

	output = trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))
	# trainer.save_pretrained(output.state, to_torch=True)
	exit(0)


if __name__ == "__main__":
	main(use_iterable_dataset=True)
