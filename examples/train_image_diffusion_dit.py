#!/usr/bin/env python3
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example training script for DiT (Diffusion Transformer) image diffusion model.

This script demonstrates how to train a DiT model on image datasets using
rectified flow and the EasyDeL framework.
"""

import jax
import jax.numpy as jnp
from datasets import load_dataset
from flax import nnx as nn

from easydel.modules.dit import DiTConfig, DiTForImageDiffusion
from easydel.trainers.image_diffusion_trainer import ImageDiffusionConfig, ImageDiffusionTrainer


def main():
	# Configuration
	# DiT model configuration
	model_config = DiTConfig(
		image_size=32,  # 32x32 images (CIFAR-10 size)
		patch_size=2,  # 2x2 patches
		in_channels=3,  # RGB images
		hidden_size=384,  # Smaller for faster training
		num_hidden_layers=12,
		num_attention_heads=6,
		num_classes=10,  # CIFAR-10 has 10 classes
		class_dropout_prob=0.1,  # For classifier-free guidance
		learn_sigma=False,  # Don't predict variance
		use_conditioning=True,  # Use class conditioning
	)

	# Training configuration
	training_config = ImageDiffusionConfig(
		# Model settings
		model_name="dit-cifar10",
		# Dataset settings
		dataset_name="cifar10",
		dataset_image_field="img",  # CIFAR-10 uses 'img'
		dataset_label_field="label",
		image_size=32,
		# Training settings
		num_train_epochs=100,
		learning_rate=1e-4,
		per_device_train_batch_size=128,
		per_device_eval_batch_size=128,
		gradient_accumulation_steps=1,
		# Diffusion settings
		num_train_timesteps=1000,
		prediction_type="velocity",  # Rectified flow uses velocity prediction
		class_conditional=True,
		class_dropout_prob=0.1,
		min_snr_gamma=5.0,  # Enable Min-SNR weighting
		# Optimization
		optimizer="adamw",
		weight_decay=0.01,
		warmup_steps=1000,
		# Logging and checkpointing
		logging_steps=100,
		save_steps=5000,
		eval_steps=1000,
		save_total_limit=3,
		# Paths
		output_dir="./outputs/dit-cifar10",
		# Mixed precision
		dtype=jnp.bfloat16,
		param_dtype=jnp.bfloat16,
		# Sharding
		sharding_array=(1, -1, 1, 1),  # Shard on sequence dimension
	)

	# Load dataset
	print("Loading dataset...")
	dataset = load_dataset("cifar10")

	def preprocess_images(examples):
		"""Preprocess images to be in [-1, 1] range."""
		images = jnp.array(examples[training_config.dataset_image_field])
		# Normalize to [-1, 1]
		images = (images.astype(jnp.float32) / 127.5) - 1.0
		return {
			"pixel_values": images,
			"labels": jnp.array(examples[training_config.dataset_label_field]),
		}

	train_dataset = dataset["train"].map(
		preprocess_images,
		batched=True,
		remove_columns=dataset["train"].column_names,
	)

	eval_dataset = dataset["test"].map(
		preprocess_images,
		batched=True,
		remove_columns=dataset["test"].column_names,
	)

	# Initialize model
	print("Initializing model...")
	rngs = nn.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))
	model = DiTForImageDiffusion(
		config=model_config,
		dtype=training_config.dtype,
		param_dtype=training_config.param_dtype,
		rngs=rngs,
	)

	print(f"Model initialized with {sum(x.size for x in jax.tree.leaves(model.parameters()))} parameters")

	# Initialize trainer
	print("Initializing trainer...")
	trainer = ImageDiffusionTrainer(
		arguments=training_config,
		model=model,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		seed=42,
		dtype=training_config.dtype,
	)

	# Train
	print("Starting training...")
	trainer.train()
	print("Training complete!")

	# Save final model
	print(f"Saving final model to {training_config.output_dir}/final_model")
	trainer.save_checkpoint(f"{training_config.output_dir}/final_model")


if __name__ == "__main__":
	main()
