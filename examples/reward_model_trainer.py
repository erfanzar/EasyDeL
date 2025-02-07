"""
Usage Example:

python reward_model_trainer.py \
    --repo_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --max_length 2048 \
    --num_train_epochs 1 \
    --total_batch_size 32 \
    --warmup_steps 50 \
    --optimizer ADAMW \
    --scheduler COSINE \
    --sharding_axis_dims "1,1,1,-1"
	
"""

import argparse
import easydel as ed
from datasets import load_dataset
from transformers import AutoTokenizer
from jax import numpy as jnp
import jax


def parse_args():
	parser = argparse.ArgumentParser(description="Train a reward model using EasyDeL")

	# Model configuration
	parser.add_argument(
		"--repo_id",
		type=str,
		default="meta-llama/Llama-3.1-8B-Instruct",
		help="Hugging Face repository ID for model and tokenizer",
	)
	parser.add_argument(
		"--max_length",
		type=int,
		default=2048,
		help="Maximum sequence length for model inputs",
	)
	parser.add_argument(
		"--sharding_axis_dims",
		type=str,
		default="1,1,1,-1",
		help="Sharding dimensions for model parallelism as comma-separated integers",
	)
	parser.add_argument(
		"--center_rewards_coefficient",
		type=float,
		default=0.1,
		help="Coefficient to incentivize the reward model to output mean-zero rewards.",
	)

	# Dataset configuration
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="trl-lib/ultrafeedback_binarized",
		help="Name of the dataset to load from Hugging Face Hub",
	)

	# Training configuration
	parser.add_argument(
		"--num_train_epochs", type=int, default=1, help="Number of training epochs"
	)
	parser.add_argument(
		"--total_batch_size",
		type=int,
		default=32,
		help="Total batch size across all devices",
	)
	parser.add_argument(
		"--warmup_steps",
		type=int,
		default=50,
		help="Number of warmup steps for learning rate scheduler",
	)

	# Optimization configuration
	parser.add_argument(
		"--optimizer",
		type=str,
		default="ADAMW",
		choices=[opt.name for opt in ed.EasyDeLOptimizers],
		help="Optimizer to use for training",
	)
	parser.add_argument(
		"--scheduler",
		type=str,
		default="COSINE",
		choices=[sched.name for sched in ed.EasyDeLSchedulers],
		help="Learning rate scheduler to use",
	)

	# Logging and saving
	parser.add_argument(
		"--log_steps", type=int, default=1, help="Number of steps between logging metrics"
	)
	parser.add_argument(
		"--save_steps",
		type=int,
		default=1000,
		help="Number of steps between checkpoint saves",
	)
	parser.add_argument(
		"--save_total_limit",
		type=int,
		default=1,
		help="Maximum number of checkpoints to keep",
	)
	parser.add_argument(
		"--progress_bar_type",
		type=str,
		default="json",
		choices=["tqdm", "json", "rich"],
		help="Type of progress bar to use",
	)

	# Boolean flags
	parser.add_argument(
		"--no_do_last_save",
		action="store_false",
		dest="do_last_save",
		help="Disable saving final checkpoint",
	)
	parser.add_argument(
		"--no_use_wandb",
		action="store_false",
		dest="use_wandb",
		help="Disable Weights & Biases logging",
	)
	parser.add_argument(
		"--save_optimizer_state",
		action="store_true",
		help="Save optimizer state with checkpoints",
	)
	parser.add_argument(
		"--no_process_zero_is_admin",
		action="store_false",
		dest="process_zero_is_admin",
		help="Disable process zero admin privileges",
	)

	return parser.parse_args()


def main():
	args = parse_args()

	# Convert string arguments to appropriate types
	sharding_axis_dims = tuple(map(int, args.sharding_axis_dims.split(",")))
	optimizer = getattr(ed.EasyDeLOptimizers, args.optimizer)
	scheduler = getattr(ed.EasyDeLSchedulers, args.scheduler)

	# Load dataset and tokenizer
	dataset = load_dataset(args.dataset_name)
	tokenizer = AutoTokenizer.from_pretrained(args.repo_id)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	# Initialize model
	model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
		args.repo_id,
		auto_shard_model=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=args.max_length,
			mask_max_position_embeddings=args.max_length,
			attn_dtype=jnp.bfloat16,
			attn_softmax_dtype=jnp.float32,
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		platform=ed.EasyDeLPlatforms.JAX,
		param_dtype=jnp.bfloat16,
		dtype=jnp.bfloat16,
		precision=jax.lax.Precision.HIGH,
		partition_axis=ed.PartitionAxis(),
	)
	model.config.pad_token_id = tokenizer.pad_token_id

	# Initialize trainer
	trainer = ed.RewardTrainer(
		model=model,
		processing_class=tokenizer,
		arguments=ed.RewardConfig(
			center_rewards_coefficient=args.center_rewards_coefficient,
			max_length=args.max_length,
			max_sequence_length=args.max_length,
			num_train_epochs=args.num_train_epochs,
			total_batch_size=args.total_batch_size,
			log_steps=args.log_steps,
			do_last_save=args.do_last_save,
			use_wandb=args.use_wandb,
			save_optimizer_state=args.save_optimizer_state,
			progress_bar_type=args.progress_bar_type,
			save_steps=args.save_steps,
			save_total_limit=args.save_total_limit,
			optimizer=optimizer,
			scheduler=scheduler,
			warmup_steps=args.warmup_steps,
			process_zero_is_admin=args.process_zero_is_admin,
		),
		train_dataset=dataset["train"],
		eval_dataset=None,
	)

	# Start training
	trainer.train()


if __name__ == "__main__":
	main()
