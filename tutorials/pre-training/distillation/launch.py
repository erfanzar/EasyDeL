# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates large-scale model distillation using EasyDeL on TPUs.
It showcases several advanced features:
- Distilling knowledge from a large "teacher" model (Qwen3-14B) into a smaller,
  custom-defined "student" model.
- Training on a massive, web-scale dataset ('tiiuae/falcon-refinedweb') by
  streaming it directly from the source, avoiding the need for local downloads.
- Leveraging Ray for distributed execution across a TPU pod.
"""

import os
import pprint

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

# Initialize Ray for distributed computing. This must be done once per application.
ray.init()

# --- Configuration Constants ---
# The large, powerful model that will act as the "teacher".
TEACHER_MODEL_ID = "Qwen/Qwen3-14B"

# Your Weights & Biases entity (username or organization) for experiment logging.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# --- Environment and TPU Configuration ---
# These environment variables are passed to each Ray worker to ensure they have
# access to necessary tokens and use efficient shared memory for caching.
TPU_EXECUTION_ENV_VARS = {
    "EASYDEL_AUTO": "1",  # Enables EasyDeL's automatic sharding configuration.
    "HF_TOKEN": os.environ.get("HF_TOKEN_FOR_EASYDEL", ""),  # Hugging Face token.
    "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",  # RAM-disk for dataset cache.
    "HF_HOME": "/dev/shm/huggingface",  # RAM-disk for model cache.
    "HF_DATASETS_OFFLINE": "0",  # Allow online dataset access.
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY_FOR_EASYDEL", ""),  # W&B API key.
}

# Additional pip packages to install on each Ray worker environment.
TPU_PIP_PACKAGES = []

# Print the environment variables for verification.
pprint.pprint(TPU_EXECUTION_ENV_VARS)

# Defines the TPU environment for Ray, specifying the accelerator type and worker setup.
tpu_config = TpuAcceleratorConfig(
    "v4-64",  # Using a TPU v4 pod slice with 64 chips. Adjust to your hardware.
    execution_env={
        "env_vars": TPU_EXECUTION_ENV_VARS,
        "pip": TPU_PIP_PACKAGES,
    },
)


@execute(tpu_config)
@ray.remote
def main():
    """
    The main function for the distillation training process, executed as a
    remote task on the TPU cluster via Ray.
    """
    # Imports are inside the function to ensure they are available in the
    # separate Ray worker process.
    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    import easydel as ed

    # --- Basic Training Parameters ---
    max_length = 4096
    total_batch_size = 128

    # --- Tokenizer Setup ---
    # The tokenizer MUST be loaded from the teacher model to ensure that both the
    # student and teacher share the exact same vocabulary and token mappings.
    processor = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    processor.padding_side = "left"  # Crucial for causal language models.
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    # --- Streaming Dataset Setup ---
    # This section configures a streaming dataset, which is essential for training
    # on datasets too large to fit in memory or download completely.
    # 1. Define the dataset source information.
    informs = [
        ed.TextDatasetInform(content_field="content", path="tiiuae/falcon-refinedweb", split="train"),
        # ed.TextDatasetInform( # sample of reading from bucket.
        #     content_field="text",
        #     data_files="gs://your-bucket/raw/dclm/a3b142c/**",
        #     split="train",
        #     path=ed.DatasetType.JSON,
        # ),
        # ed.TextDatasetInform(
        #     content_field="content",
        #     data_files="gs://your-bucket/raw/starcoderdata-720c8c/9fc30b5/**/*.parquet",
        #     split="train",
        # ),
        # ed.TextDatasetInform(
        #     content_field="text",
        #     data_files="gs://your-bucket/raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927/**",
        #     split="train",
        #     path=ed.DatasetType.JSON,
        # ),
        # ed.TextDatasetInform(
        #     content_field="text",
        #     data_files="gs://your-bucket/raw/dolma/v1.7/*.json.gz",
        #     split="train",
        # ),
    ]
    # 2. Combine sources into a mixture (here, just one source).
    mixture = ed.DatasetMixture(batch_size=1, informs=informs)
    # 3. Create the live, iterable, streaming dataset.
    train_dataset = ed.DataManager.create_dataset_from_mixture(mixture)

    # --- Teacher Model Loading ---
    # Load the large teacher model and automatically shard it across all TPU devices.
    teacher_model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        auto_shard_model=True,
        # Shard across data-parallel and fully-sharded data-parallel dimensions.
        sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS,
        ),
    )

    # --- Student Model Definition ---
    # We define the student model's architecture from scratch using an EasyDeL config object.
    # This allows us to create a much smaller model than the teacher.
    student_model = ed.Qwen3ForCausalLM(
        config=ed.Qwen3Config(
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=4096 * 2,
            num_hidden_layers=16,  # Key change: Much smaller than the teacher model.
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            use_sliding_window=False,
            sliding_window=4096,
            sharding_axis_dims=(1, jax.process_count(), 1, -1, 1),
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS,
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
        ),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        rngs=ed.Rngs(0),
    ).shard_model()  # Shard the newly created student model across devices.

    # --- Distillation Configuration ---
    # The DistillationConfig holds all hyperparameters for the distillation process.
    arguments = ed.DistillationConfig(
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        use_wandb=True,
        wandb_entity=WANDB_ENTITY,
        do_last_save=True,
        max_sequence_length=max_length,
        # This is MANDATORY for streaming datasets. It tells the trainer how many
        # steps constitute one "epoch". Should be ~ (total_dataset_size // total_batch_size).
        per_epoch_training_steps=98_000_000,
        learning_rate=2e-4,
        learning_rate_end=7e-6,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.COSINE,
        # --- Key Distillation Hyperparameters ---
        # `temperature`: Softens the teacher's output probabilities, providing a richer
        # signal for the student. Higher values make the distribution "softer".
        temperature=2.0,
        # `alpha`: Balances the distillation loss (mimicking the teacher's soft labels)
        # and the standard SFT loss (learning the correct token). 0.9 means 90% of the
        # loss comes from distillation.
        alpha=0.9,
        # Saving to a GCS bucket is highly recommended for large-scale training.
        save_directory="gs://your-bucket/distillation",
        save_steps=1_000,
        save_total_limit=0,
        save_optimizer_state=False,
        clip_grad=1.0,
        report_steps=10,
        log_steps=5,
        progress_bar_type="json",
    )

    # --- Data Processing Functions ---
    def process_sample_data(sample: dict) -> dict[str, jax.Array]:
        """Tokenizes a single text sample from the streaming dataset."""
        out = processor(
            sample["content"],  # The `content_field` we defined in `TextDatasetInform`
            padding="max_length",
            max_length=max_length,
            return_tensors="jax",
            truncation=True,
        )
        # The original script flattens the output; we preserve this logic.
        out = {k: (v.reshape(-1) if hasattr(v, "shape") else v) for k, v in out.items()}
        return out

    def extract_column_names(dataset) -> list[str] | None:
        """A utility to get column names from the first sample of a dataset."""
        if hasattr(dataset, "column_names") and len(dataset.column_names) != 0:
            return dataset.column_names
        keys = None
        for _sample in dataset:
            keys = list(_sample.keys())  # Ensure it's a list
            break
        return keys

    # Use `.map()` to apply the tokenization function on-the-fly to the streaming dataset.
    # The `remove_columns` argument cleans up the original text column after processing.
    processed_dataset = train_dataset.map(
        process_sample_data,
        remove_columns=extract_column_names(train_dataset),
    )

    # --- Trainer Setup and Execution ---
    # The `DistillationTrainer` is a specialized EasyDeL trainer that orchestrates
    # the entire distillation process.
    trainer = ed.DistillationTrainer(
        arguments=arguments,
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataset=processed_dataset,  # The live, tokenized, streaming dataset.
        eval_dataset=None,  # No evaluation dataset is used in this example.
        processing_class=processor,
    )

    trainer.train()


if __name__ == "__main__":
    # This is the script's entry point. The `@execute` and `@ray.remote` decorators
    # on `main` ensure that the function is launched as a distributed task on
    # the configured TPU resources via Ray.
    out = main()
