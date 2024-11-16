import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)
os.environ["EKERNEL_OPS"] = "false"
# import jax

# jax.config.update("jax_platform_name", "cpu")  # CPU Test !

import flax.core
from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed
from easydel.trainers import create_prompt_creator


def main():
	sequence_length = 128
	config = ed.MistralConfig(
		hidden_size=128,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=4,
		intermediate_size=256,
		gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
		max_position_embeddings=sequence_length,
		attn_dtype=jnp.float16,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
		block_k=512,
		block_q=512,
		hardware_abstraction=False,
	)

	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

	train_dataset = load_dataset("LDJnr/Pure-Dove", split="train[:5%]")

	model = ed.FlaxMistralForCausalLM(config=config, _do_init=True)
	params = model.shard_params(model.params)

	prompter = create_prompt_creator(tokenizer)
	dtype = jnp.float32
	trainer = ed.SFTTrainer(
		arguments=ed.TrainingArguments(
			model_name="SFTTrainer_TEST",
			num_train_epochs=3,
			total_batch_size=2,
			optimizer=ed.EasyDeLOptimizers.ADAMW,
			scheduler=ed.EasyDeLSchedulers.COSINE,
			gradient_accumulation_steps=2,
			use_wandb=False,
			do_train=True,
			do_eval=False,
			max_sequence_length=sequence_length,
			dtype=dtype,
			param_dtype=dtype,
			track_memory=False,
			learning_rate=5e-4,
			label_smoothing_factor=0.1,
			z_loss=0.0001,
			train_on_inputs=True,
			# save_steps=500,
			save_total_limit=1,
			do_last_save=True,
		),
		model=model,
		train_dataset=train_dataset,
		tokenizer=tokenizer,
		formatting_func=prompter,
		packing=True,
		num_of_sequences=sequence_length,
		# chars_per_token=2.1,
		dataset_num_proc=32,
	)
	return trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
	res = main()
