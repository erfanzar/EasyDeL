import os
import sys

import flax.nnx

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)
# jax.config.update("jax_platform_name", "cpu")  # CPU Test !

import flax
import jax  # noqa
import datasets
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed  # noqa
from easydel import (
	AttentionMechanisms,
	DPOConfig,
	DPOTrainer,
	LlamaForCausalLM,
	LlamaConfig,
)


def main():
	train_dataset: datasets.Dataset = (
		datasets.concatenate_datasets(
			[
				datasets.load_dataset(
					"argilla/ultrafeedback-binarized-preferences", split="train"
				),
			]
		)
		.shuffle()
		.shuffle()
	).select(range(1000))
	train_dataset = train_dataset.rename_column("chosen_response", "chosen")
	train_dataset = train_dataset.rename_column("rejected_response", "rejected")
	train_dataset = train_dataset.rename_column("instruction", "prompt")

	max_length = 512
	max_completion_length = 256
	max_prompt_length = 256

	# assert len(jax.devices("cpu")) == 8, "XLA Device manipulation failed."
	with jax.default_device(jax.devices("gpu")[0]):
		model_name_or_path = "erfanzar/LLamaStory-70M"
		conf = LlamaConfig(
			hidden_size=128,
			intermediate_size=256,
			num_hidden_layers=4,
			num_attention_heads=8,
			num_key_value_heads=4,
			max_position_embeddings=512,
			use_scan_mlp=False,
			attention_bias=False,
			attn_dtype=jnp.float16,
			attn_mechanism=AttentionMechanisms.VANILLA,
		)
		arguments = DPOConfig(
			num_train_epochs=4,
			model_name="direct_preference_optimization_trainer",
			save_directory="tmp-files",
			loss_type="kto",
			total_batch_size=8,
			use_wandb=False,
			learning_rate=7e-5,
			learning_rate_end=9e-6,
			warmup_steps=100,
			optimizer=ed.EasyDeLOptimizers.ADAMW,
			scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
			weight_decay=0.02,
			max_sequence_length=max_length,
			gradient_accumulation_steps=1,
			step_start_point=0,
			do_last_save=False,
			training_time="7H",
			force_batch_and_gradient_accumulation_steps_calculation=False,
			track_memory=True,
			max_length=max_length,
			max_completion_length=max_completion_length,
			max_prompt_length=max_prompt_length,
			beta=0.1,
		)

		processing_class = AutoTokenizer.from_pretrained(model_name_or_path)

		if processing_class.pad_token is None:
			processing_class.pad_token = processing_class.eos_token

		if processing_class.pad_token_id is None:
			processing_class.pad_token_id = processing_class.eos_token_id

		model = LlamaForCausalLM(
			config=conf,
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			rngs=flax.nnx.Rngs(0),
		)
		ref_model = LlamaForCausalLM(
			config=conf,
			dtype=jnp.float32,
			param_dtype=jnp.float32,
			rngs=flax.nnx.Rngs(0),
		)

		dpo_trainer = DPOTrainer(
			model=model,
			ref_model=ref_model,
			train_dataset=train_dataset,
			eval_dataset=None,
			processing_class=processing_class,
			arguments=arguments,
		)

		dpo_trainer.train()


if __name__ == "__main__":
	main()
