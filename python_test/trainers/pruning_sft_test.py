import fjformer
import functools
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

import flax.core
from datasets import load_dataset
from easydel import FlaxMistralForCausalLM, MistralConfig, TrainArguments
from easydel.trainers import conversations_formatting_function
from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer
from jax import numpy as jnp
from transformers import AutoTokenizer


def main():
	sequence_length = 128
	config = MistralConfig(
		hidden_size=128,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=4,
		intermediate_size=256,
		gradient_checkpointing="",
		max_position_embeddings=sequence_length,
	)

	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

	def prompter(sample):
		return [
			conversations_formatting_function(tokenizer, messages_field="messages")(sample)
		]

	train_dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split="train_sft")

	model = FlaxMistralForCausalLM(config=config, _do_init=True)
	params = model.shard_params(model.params)

	dtype = jnp.float32
	trainer = SFTTrainer(
		arguments=TrainArguments(
			model_name="SFTTrainer_TEST",
			num_train_epochs=3,
			total_batch_size=2,
			gradient_accumulation_steps=2,
			use_wandb=False,
			model_class=type(model),
			do_train=True,
			do_eval=False,
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
			learning_rate=5e-4,
			label_smoothing_factor=0.1,
			z_loss=0.0001,
			train_on_inputs=True,
			save_steps=500,
			save_total_limit=1,
			do_last_save=False,
			pruning_module=fjformer.jaxpruner.MagnitudePruning(
				sparsity_distribution_fn=functools.partial(
					fjformer.jaxpruner.sparsity_distributions.uniform, sparsity=0.8
				),
				scheduler=fjformer.jaxpruner.sparsity_schedules.OneShotSchedule(0),
			),
		),
		train_dataset=train_dataset,
		eval_dataset=None,  # we don't have eval dataset rn :)
		tokenizer=tokenizer,
		dataset_text_field=None,
		formatting_func=prompter,
		packing=True,
		num_of_sequences=1024,
		chars_per_token=2.1,
		dataset_num_proc=32,
	)
	return trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
	res = main()
