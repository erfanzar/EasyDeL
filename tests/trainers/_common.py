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

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import numpy as np
from datasets import Dataset, IterableDataset, load_dataset
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed

if jax.default_backend() == "tpu":
    MODEL_REPO = "Qwen/Qwen3-4B"
else:
    MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

PREFERENCE_DATASET = "trl-lib/ultrafeedback_binarized"
PREFERENCE_SPLIT = "train[:50%]"
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
MAX_TOTAL_LENGTH = MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH
MAX_TRAINING_STEP = 512
SAVE_ROOT = Path("tmp-files") / "trainer-smoke-tests"


def _ensure_logging_configured() -> None:
    # Configure root logger once; subsequent calls are cheap no-ops.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def get_logger(name: str) -> logging.Logger:
    _ensure_logging_configured()
    return logging.getLogger(name)


def _prepare_save_dir(name: str) -> str:
    path = SAVE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@lru_cache(maxsize=4)
def get_tokenizer(model_repo: str = MODEL_REPO):
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model_kwargs() -> dict[str, Any]:
    return dict(
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=MAX_TOTAL_LENGTH,
            mask_max_position_embeddings=MAX_TOTAL_LENGTH,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            use_expert_tensor_mode=False,
            moe_force_xla_gmm=True,
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
    )


def load_causal_lm_model(model_repo: str | None = None) -> ed.AutoEasyDeLModelForCausalLM:
    repo = model_repo or MODEL_REPO
    tokenizer = get_tokenizer(repo)
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(repo, **_load_model_kwargs())
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_sequence_classifier_model() -> ed.AutoEasyDeLModelForSequenceClassification:
    tokenizer = get_tokenizer()
    model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
        MODEL_REPO, num_labels=1, **_load_model_kwargs()
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_preference_dataset(split: str = PREFERENCE_SPLIT):
    return load_dataset(PREFERENCE_DATASET, split=split)


def build_sft_text_dataset(
    split: str = PREFERENCE_SPLIT,
    template: Callable[[dict[str, Any]], str] | None = None,
    tokenizer=None,
) -> Dataset | IterableDataset:
    dataset = load_preference_dataset(split)
    if template is None:

        def template(sample):
            return sample["chosen"]

    out = dataset.map(lambda sample: {"messages": template(sample)}, remove_columns=dataset.column_names)
    return out


def build_lm_dataset(
    tokenizer,
    split: str = PREFERENCE_SPLIT,
    max_length: int = MAX_TOTAL_LENGTH,
) -> Dataset | IterableDataset:
    text_dataset = build_sft_text_dataset(split, tokenizer=tokenizer)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[np.ndarray]]:
        encoded = tokenizer.apply_chat_template(
            batch["messages"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_dict=True,
        )
        return encoded

    return text_dataset.map(tokenize, batched=True, remove_columns=["messages"])


def make_config(
    config_cls,
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> Any:
    base_args = {
        "save_directory": _prepare_save_dir(name),
        "num_train_epochs": 1,
        "total_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "log_steps": 1,
        "learning_rate": 8e-6,
        "max_training_steps": MAX_TRAINING_STEP,
        "max_sequence_length": MAX_TOTAL_LENGTH,
        "save_steps": 1_000,
        "save_total_limit": 1,
        "save_optimizer_state": False,
        "do_last_save": True,
        "use_wandb": True,
        "wandb_entity": "erfanzar",
        "shuffle_train_dataset": False,
        "progress_bar_type": "json",
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "max_length": MAX_TOTAL_LENGTH,
        "generation_top_p": 0.95,
        "generation_top_k": 64,
        "generation_temperature": 0.7,
        "generation_do_sample": True,
        "generation_num_return_sequences": 4,
        "generation_max_new_tokens": 2048,
        "generation_interval": 100,
        "generation_prompts": ["Here's Fibo in c++"],
        "generation_use_train_prompts": False,
        "generation_log_to_wandb": True,
    }

    if overrides:
        base_args.update(overrides)

    field_names = set(getattr(config_cls, "__dataclass_fields__", {}).keys())
    filtered = {key: value for key, value in base_args.items() if key in field_names}
    return config_cls(**filtered)


def run_trainer(
    trainer_factory: Callable[..., Any],
    *,
    model,
    arguments,
    tokenizer,
    train_dataset,
    logger: logging.Logger,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    extra_kwargs = extra_kwargs or {}
    logger.info("Instantiating %s", trainer_factory.__name__)
    trainer = trainer_factory(
        model=model,
        arguments=arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        **extra_kwargs,
    )
    logger.info("Launching training loop.")
    trainer.train()
    logger.info("Finished training run.")


def build_reward_dataset(split: str = PREFERENCE_SPLIT):
    dataset = load_preference_dataset(split)
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset split {split} missing columns: {missing}")
    return dataset


def dummy_reward_fn(*, prompts: Iterable[Any] | None = None, completions: Iterable[Any] | None = None, **_: Any):
    # print("Lengths", len(prompts), len(completions))
    # print("Prompt ===> ", prompts[0])
    # print("COMPL ===> ", completions[0])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[1])
    # print("COMPL ===> ", completions[1])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[2])
    # print("COMPL ===> ", completions[2])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[3])
    # print("COMPL ===> ", completions[3])
    # print("FIRST 4")
    # print("=" * 15)
    # print("Prompt ===> ", prompts[4])
    # print("COMPL ===> ", completions[4])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[5])
    # print("COMPL ===> ", completions[5])
    # print("THESE 2 should be match")
    # print("=" * 15)
    # print("Prompt ===> ", prompts[8])
    # print("COMPL ===> ", completions[8])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[9])
    # print("COMPL ===> ", completions[9])
    # print("=" * 15)
    # print("Prompt ===> ", prompts[10])
    # print("COMPL ===> ", completions[10])
    # print("THESE 3 should be match")
    # print("=" * 15)
    if completions is None:
        return [0.1]
    if isinstance(completions, list | tuple):
        length = len(completions)
    else:
        completions = list(completions)
        length = len(completions)
    return [0.1] * max(length, 1)
