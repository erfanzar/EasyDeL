import os
import sys

import flax

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)

from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed
from easydel.trainers import create_prompt_creator

NUM_TRAIN_EPOCHS = 4
TOTAL_BATCH_SIZE = 8


def main():
	sequence_length = 128
	config = ed.LlamaConfig(
		head_dim=16,
		hidden_size=64,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=1,
		intermediate_size=128,
		max_position_embeddings=sequence_length,
		attn_dtype=jnp.float32,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	)

	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

	train_dataset = load_dataset("LDJnr/Pure-Dove", split="train[:5%]")
	dtype = jnp.float32
	model = ed.LlamaForCausalLM(
		config=config,
		dtype=dtype,
		param_dtype=dtype,
		rngs=flax.nnx.Rngs(0),
	)
	model = model.shard_model()

	prompter = create_prompt_creator(tokenizer)
	trainer = ed.SFTTrainer(
		arguments=ed.TrainingArguments(
			save_directory="tmp-files",
			model_name="SFT-TrainerTest",
			num_train_epochs=NUM_TRAIN_EPOCHS,
			total_batch_size=TOTAL_BATCH_SIZE,
			gradient_accumulation_steps=2,
			do_train=True,
			do_eval=False,
			max_sequence_length=sequence_length,
			track_memory=True,
			use_wandb=False,
			learning_rate=3e-4,
			do_last_save=True,
			save_steps=10,
			save_total_limit=5,
			save_optimizer_state=True,
			training_time="80Min",
			optimizer=ed.EasyDeLOptimizers.ADAMW,
			scheduler=ed.EasyDeLSchedulers.COSINE,
			clip_grad=1.0,
			warmup_steps=5,
		),
		model=model,
		train_dataset=train_dataset,
		tokenizer=tokenizer,
		formatting_func=prompter,
		packing=True,
		num_of_sequences=sequence_length,
	)
	return trainer.train()


if __name__ == "__main__":
	res = main()
