import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)
# import jax

# jax.config.update("jax_platform_name", "cpu")  # CPU Test !

import easydel as ed
import flax
import jax
from datasets import load_dataset
from jax import numpy as jnp
from transformers import AutoTokenizer

SEQUENCE_LENGTH = 128
NUM_TRAIN_EXAMPLES = 50
NUM_EVAL_EXAMPLES = 12
TOTAL_BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 3
MAX_TRAINING_STEPS = NUM_TRAIN_EXAMPLES // TOTAL_BATCH_SIZE * NUM_TRAIN_EPOCHS
MAX_EVALUATION_STEPS = NUM_EVAL_EXAMPLES // TOTAL_BATCH_SIZE


def orpo_main():
	#####################
	# Model & Tokenizer #
	#####################
	with jax.default_device(jax.devices("gpu")[0]):
		model_name_or_path = "erfanzar/LLamaStory-70M"
		config = ed.LlamaConfig(
			head_dim=16,
			hidden_size=64,
			num_attention_heads=8,
			num_key_value_heads=4,
			num_hidden_layers=1,
			intermediate_size=128,
			max_position_embeddings=SEQUENCE_LENGTH * 3,
			attn_dtype=jnp.float32,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		)

		dtype = jnp.float32
		model = ed.LlamaForCausalLM(
			config=config,
			dtype=dtype,
			param_dtype=dtype,
			rngs=flax.nnx.Rngs(0),
		)
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	#################
	# Dataset       #
	#################
	train_dataset = load_dataset(
		"orpo-explorers/OpenHermesPreferences-10k", split="train[:3%]"
	)
	if tokenizer.chat_template is None:
		tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

	################
	# Training     #
	################
	trainer = ed.ORPOTrainer(
		model=model,
		train_dataset=train_dataset,
		tokenizer=tokenizer,
		dataset_num_proc=4,
		arguments=ed.ORPOConfig(
			model_name="odds_ratio_preference_optimization_trainer_test",
			save_directory="tmp-files",
			num_train_epochs=NUM_TRAIN_EPOCHS,
			total_batch_size=TOTAL_BATCH_SIZE,
			gradient_accumulation_steps=2,
			track_memory=False,
			use_wandb=False,
			learning_rate=5e-4,
			do_last_save=True,
			max_prompt_length=SEQUENCE_LENGTH,
			max_length=SEQUENCE_LENGTH * 2,
			max_completion_length=SEQUENCE_LENGTH * 2,
		),
	)

	out = trainer.train()
	print(out)


if __name__ == "__main__":
	orpo_main()
