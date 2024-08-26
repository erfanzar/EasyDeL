import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../../src",
	)
)
import jax

jax.config.update("jax_platform_name", "cpu")  # CPU Test !
from datasets import Dataset, IterableDataset
import flax.core
from easydel import (
	AttentionMechanisms,
	CausalLanguageModelTrainer,
	FlaxMistralForCausalLM,
	MistralConfig,
	TrainArguments,
	LoraRaptureConfig,
)

from jax import numpy as jnp, random

NUM_TRAIN_EXAMPLES = 100
NUM_EVAL_EXAMPLES = 12
TOTAL_BATCH_SIZE = 2
NUM_TRAIN_EPOCHS = 1


def main(use_iterable_dataset: bool):
	sequence_length = 512
	max_training_steps = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
	max_evaluation_steps = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE
	config = MistralConfig(
		hidden_size=128,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=4,
		intermediate_size=256,
		gradient_checkpointing="",
		max_position_embeddings=sequence_length,
		attn_dtype=jnp.float16,
		attn_mechanism=AttentionMechanisms.pallas_flash,
		block_k=128,
		block_q=128,
	)

	model = FlaxMistralForCausalLM(config=config, _do_init=True)
	params = model.params

	def data_generator(num_rows: int):
		for _ in range(num_rows):
			yield {
				"attention_mask": jnp.ones((sequence_length,), dtype="i4"),
				"input_ids": random.randint(
					random.PRNGKey(0), (sequence_length,), 0, 32000, dtype="i4"
				),
			}

	if not use_iterable_dataset:
		example_train_data = Dataset.from_generator(
			data_generator, gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES}
		)
		example_eval_data = Dataset.from_generator(
			data_generator, gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES}
		)
	else:
		example_train_data = IterableDataset.from_generator(
			data_generator, gen_kwargs={"num_rows": NUM_TRAIN_EXAMPLES}
		)
		example_eval_data = IterableDataset.from_generator(
			data_generator, gen_kwargs={"num_rows": NUM_EVAL_EXAMPLES}
		)
	dtype = jnp.float16
	trainer = CausalLanguageModelTrainer(
		arguments=TrainArguments(
			model_name="LORA_CLM_TEST",
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
			track_memory=False,
			use_wandb=False,
			learning_rate=5e-4,
			label_smoothing_factor=0.1,
			z_loss=0.0001,
			train_on_inputs=True,
			do_last_save=True,
			rapture_config=LoraRaptureConfig(
				parameters=flax.core.FrozenDict({"params": params}),
				lora_dim=64,
				fully_fine_tune_parameters=["o_proj"],
				lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj"],
				tune_vectors=True,
				verbose=False,
				dtype=jnp.float32,
			),
		),
		dataset_train=example_train_data,
		dataset_eval=example_eval_data,
	)

	output = trainer.train()
	trainer.save_pretrained(output.state, to_torch=True)


if __name__ == "__main__":
	main(use_iterable_dataset=True)
