import os
import sys

import flax
import flax.nnx

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)
os.environ["EKERNEL_OPS"] = "false"
import fjformer
from datasets import Dataset, IterableDataset
from jax import numpy as jnp


from easydel import (
	AttentionMechanisms,
	Trainer,
	EasyDeLOptimizers,
	EasyDeLSchedulers,
	LlamaForCausalLM,
	LlamaConfig,
	TrainingArguments,
)

TOTAL_BATCH_SIZE = 1
UPPER = 5
NUM_TRAIN_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_EVAL_EXAMPLES = TOTAL_BATCH_SIZE * UPPER
NUM_TRAIN_EPOCHS = 2
rng = fjformer.GenerateRNG()


def main(use_iterable_dataset: bool):
	sequence_length = 128
	max_training_steps = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
	max_evaluation_steps = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE
	config = LlamaConfig(
		head_dim=16,
		hidden_size=64,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=1,
		intermediate_size=128,
		max_position_embeddings=sequence_length,
		attn_dtype=jnp.float32,
		attn_mechanism=AttentionMechanisms.VANILLA,
	)

	dtype = jnp.float32
	model = LlamaForCausalLM(
		config=config,
		dtype=dtype,
		param_dtype=dtype,
		rngs=flax.nnx.Rngs(0),
	)
	model = model.shard_model()

	def data_generator(num_rows: int):
		ones = jnp.ones((sequence_length,), dtype="i4")

		for _ in range(num_rows):
			yield {
				"attention_mask": ones,
				"input_ids": ones.at[-1].set(0),
				"labels": ones.at[-1].set(0),
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

	trainer = Trainer(
		arguments=TrainingArguments(
			model_name="CLM_TEST",
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
			learning_rate=3e-4,
			do_last_save=True,
			save_optimizer_state=True,
			training_time="80Min",
			optimizer=EasyDeLOptimizers.ADAMW,
			scheduler=EasyDeLSchedulers.COSINE,
			clip_grad=1.0,
			warmup_steps=5,
		),
		model=model,
		dataset_train=example_train_data,
		dataset_eval=example_eval_data,
	)
	trainer.train()
	# trainer.save_pretrained(output.state, to_torch=True)
	exit(0)


if __name__ == "__main__":
	main(use_iterable_dataset=True)
