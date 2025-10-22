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


"""Comprehensive trainer configuration support for ELM.

This module extends the ELM configuration system to support all trainer types
available in EasyDeL, including DPO, ORPO, GRPO, SFT, Reward, and Distillation trainers.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class LossConfig(TypedDict, total=False):
    ignore_index: NotRequired[int]
    label_smoothing: NotRequired[float]
    z_loss: NotRequired[float]
    loss_normalizing_factor: NotRequired[
        Literal[
            "NO_WEIGHT_NUM_REAL_TARGET_TOKENS",
            "NUM_REAL_TARGET_TOKENS",
            "NUM_TOTAL_TARGET_TOKENS",
            "AVERAGE_PER_SEQUENCE",
        ]
    ]
    num_labels: NotRequired[str | None]
    problem_type: NotRequired[str | None]
    divide_weight_sum: NotRequired[bool]
    shift_tokens: NotRequired[bool]
    break_on_nan: NotRequired[bool]
    reduction: NotRequired[Literal["none", "mean", "sum"] | None]
    num_classification_labels: NotRequired[int | None]
    classification_problem_type: NotRequired[
        Literal[
            "regression",
            "single_label_classification",
            "multi_label_classification",
        ]
        | None
    ]
    chunk_vocab_size: NotRequired[int | None]
    chunk_token_size: NotRequired[int | None]
    chunk_block_size: NotRequired[int | None]
    compute_dtype: NotRequired[Literal["fp32", "bf16"] | None]


class BaseTrainerCfg(TypedDict, total=False):
    """Base configuration shared by all trainers (TrainingArguments)."""

    trainer_type: NotRequired[Literal["sft", "base", "dpo", "grpo", "orpo", "reward", "distillation"]]
    learning_rate: NotRequired[float]
    learning_rate_end: NotRequired[float | None]
    num_train_epochs: NotRequired[int]
    max_training_steps: NotRequired[int | None]
    per_epoch_training_steps: NotRequired[int | None]
    per_epoch_evaluation_steps: NotRequired[int | None]
    total_batch_size: NotRequired[int]
    eval_batch_size: NotRequired[int | None]
    gradient_accumulation_steps: NotRequired[int]

    optimizer: NotRequired[str]
    scheduler: NotRequired[str]
    warmup_steps: NotRequired[int]
    weight_decay: NotRequired[float]
    clip_grad: NotRequired[float | None]
    extra_optimizer_kwargs: NotRequired[dict]
    custom_scheduler: NotRequired[Any]

    dataloader_num_workers: NotRequired[int | None]
    dataloader_pin_memory: NotRequired[bool | None]
    remove_unused_columns: NotRequired[bool]
    ids_to_pop_from_dataset: NotRequired[list[str] | None]
    shuffle_train_dataset: NotRequired[bool]
    shuffle_seed_train: NotRequired[int]
    use_data_collactor: NotRequired[bool]
    use_grain: NotRequired[bool]
    grain_shard_index: NotRequired[int | None]
    grain_shard_count: NotRequired[int | None]
    offload_dataset: NotRequired[bool]
    offload_device_type: NotRequired[str]
    offload_device_index: NotRequired[int]

    do_train: NotRequired[bool]
    do_eval: NotRequired[bool]
    do_last_save: NotRequired[bool]
    is_fine_tuning: NotRequired[bool]
    init_tx: NotRequired[bool]
    train_on_inputs: NotRequired[bool]
    aux_loss_enabled: NotRequired[bool]
    training_time_limit: NotRequired[str | None]
    step_start_point: NotRequired[int | None]
    resume_if_possible: NotRequired[bool]
    truncation_mode: NotRequired[Literal["keep_end", "keep_start"]]
    max_sequence_length: NotRequired[int | None]
    save_interval_minutes: NotRequired[float | None]
    save_steps: NotRequired[int | None]
    save_total_limit: NotRequired[int | None]
    save_directory: NotRequired[str]
    save_optimizer_state: NotRequired[bool]
    remove_ckpt_after_load: NotRequired[bool]

    evaluation_steps: NotRequired[int | None]
    max_evaluation_steps: NotRequired[int | None]

    log_steps: NotRequired[int]
    report_steps: NotRequired[int]
    log_all_workers: NotRequired[bool]
    log_grad_norms: NotRequired[bool]
    report_metrics: NotRequired[bool]
    metrics_to_show_in_rich_pbar: NotRequired[list[str] | None]
    progress_bar_type: NotRequired[Literal["tqdm", "rich", "json"]]
    weight_distribution_pattern: NotRequired[str]
    weight_distribution_log_steps: NotRequired[int]
    verbose: NotRequired[bool]
    process_zero_is_admin: NotRequired[bool]
    use_wandb: NotRequired[bool]
    wandb_entity: NotRequired[str | None]
    wandb_name: NotRequired[str | None]
    trainer_prefix: NotRequired[str | None]

    backend: NotRequired[str | None]
    auto_shard_states: NotRequired[bool]
    performance_mode: NotRequired[bool]
    track_memory: NotRequired[bool | float]
    low_mem_usage: NotRequired[bool]

    model_name: NotRequired[str | None]
    model_parameters: NotRequired[dict | None]
    frozen_parameters: NotRequired[str | None]

    loss_config: NotRequired[LossConfig | None]
    jax_distributed_config: NotRequired[dict | None]
    step_partition_spec: NotRequired[Any]
    state_apply_fn_kwarguments_to_model: NotRequired[dict | None]

    sparsify_module: NotRequired[bool]
    sparse_module_type: NotRequired[str]
    pruning_module: NotRequired[Any]

    tx_mu_dtype: NotRequired[Any]


class DPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Direct Preference Optimization trainer (DPOConfig)."""

    beta: NotRequired[float]
    label_smoothing: NotRequired[float]
    loss_type: NotRequired[
        Literal[
            "sigmoid",
            "hinge",
            "ipo",
            "exo_pair",
            "nca_pair",
            "robust",
            "bco_pair",
            "sppo_hard",
            "aot",
            "aot_pair",
            "apo_zero",
            "apo_down",
        ]
    ]
    use_weighting: NotRequired[bool]
    label_pad_token_id: NotRequired[int]
    padding_value: NotRequired[int | None]
    max_length: NotRequired[int | None]
    max_prompt_length: NotRequired[int | None]
    max_completion_length: NotRequired[int | None]
    is_encoder_decoder: NotRequired[bool | None]
    disable_dropout: NotRequired[bool]
    precompute_ref_log_probs: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    reference_free: NotRequired[bool]
    force_use_ref_model: NotRequired[bool]
    sync_ref_model: NotRequired[bool]
    ref_model_mixup_alpha: NotRequired[float]
    ref_model_sync_steps: NotRequired[int]
    rpo_alpha: NotRequired[float | None]
    tools: NotRequired[list[dict | Any] | None]


class ORPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Odds Ratio Preference Optimization trainer (ORPOConfig)."""

    beta: NotRequired[float]
    max_length: NotRequired[int | None]
    max_prompt_length: NotRequired[int | None]
    max_completion_length: NotRequired[int | None]
    disable_dropout: NotRequired[bool]
    label_pad_token_id: NotRequired[int]
    padding_value: NotRequired[int | None]
    generate_during_eval: NotRequired[bool]
    is_encoder_decoder: NotRequired[bool | None]
    dataset_num_proc: NotRequired[int | None]


class GRPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Group Relative Policy Optimization trainer (GRPOConfig)."""

    beta: NotRequired[float]
    max_prompt_length: NotRequired[int]
    max_completion_length: NotRequired[int]
    dataset_num_proc: NotRequired[int | None]
    sync_ref_model: NotRequired[bool]
    ref_model_mixup_alpha: NotRequired[float]
    ref_model_sync_steps: NotRequired[int]
    tools: NotRequired[list[dict | Any] | None]
    skip_apply_chat_template: NotRequired[bool]
    num_return_sequences: NotRequired[int]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    temperature: NotRequired[float]


class SFTTrainerCfg(BaseTrainerCfg):
    """Configuration for Supervised Fine-Tuning trainer (SFTConfig)."""

    dataset_text_field: NotRequired[str | None]
    add_special_tokens: NotRequired[bool]
    packing: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    dataset_batch_size: NotRequired[int]
    dataset_kwargs: NotRequired[dict[str, Any] | None]
    eval_packing: NotRequired[bool | None]
    num_of_sequences: NotRequired[int]


class RewardTrainerCfg(BaseTrainerCfg):
    """Configuration for Reward Model trainer (RewardConfig)."""

    max_sequence_length: NotRequired[int | None]
    disable_dropout: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    center_rewards_coefficient: NotRequired[float | None]


class DistillationTrainerCfg(BaseTrainerCfg):
    """Configuration for Knowledge Distillation trainer (DistillationConfig)."""

    temperature: NotRequired[float]
    alpha: NotRequired[float]


class TrainerConfig(
    ORPOTrainerCfg,
    GRPOTrainerCfg,
    SFTTrainerCfg,
    RewardTrainerCfg,
    DistillationTrainerCfg,
    BaseTrainerCfg,
    DPOTrainerCfg,
): ...


def normalize_trainer_config(config: dict[str, Any]) -> TrainerConfig:
    """Normalize and validate trainer configuration.

    This function takes raw trainer configuration and applies appropriate defaults
    based on the trainer type, ensuring all required fields are present.

    Args:
        config: Raw trainer configuration dictionary

    Returns:
        Normalized trainer configuration with proper type and defaults applied

    Example:
        >>> config = {"trainer_type": "dpo", "learning_rate": 2e-6}
        >>> normalized = normalize_trainer_config(config)
        >>> normalized["beta"]
        0.1
    """
    from copy import deepcopy

    config = deepcopy(config)
    trainer_type = config.get("trainer_type", "sft").lower()

    defaults = {
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "total_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "scheduler": "none",
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "remove_unused_columns": True,
        "shuffle_train_dataset": True,
        "shuffle_seed_train": 64871,
        "use_data_collactor": True,
        "use_grain": True,
        "offload_dataset": False,
        "offload_device_type": "cpu",
        "offload_device_index": 0,
        "do_train": True,
        "do_eval": False,
        "do_last_save": True,
        "is_fine_tuning": True,
        "init_tx": True,
        "train_on_inputs": True,
        "aux_loss_enabled": False,
        "resume_if_possible": True,
        "truncation_mode": "keep_end",
        "max_sequence_length": 4096,
        "save_directory": "EasyDeL-Checkpoints",
        "save_optimizer_state": True,
        "remove_ckpt_after_load": False,
        "log_steps": 10,
        "report_steps": 5,
        "log_all_workers": False,
        "log_grad_norms": True,
        "report_metrics": True,
        "progress_bar_type": "tqdm",
        "weight_distribution_pattern": r".*",
        "weight_distribution_log_steps": 50,
        "verbose": True,
        "process_zero_is_admin": True,
        "use_wandb": True,
        "auto_shard_states": True,
        "performance_mode": False,
        "track_memory": False,
        "low_mem_usage": True,
        "sparsify_module": False,
        "sparse_module_type": "bcoo",
    }

    if trainer_type == "dpo":
        defaults.update(
            {
                "trainer_prefix": "dpotrainer",
                "learning_rate": 1e-6,
                "beta": 0.1,
                "label_smoothing": 0.0,
                "loss_type": "sigmoid",
                "use_weighting": False,
                "label_pad_token_id": -100,
                "max_length": 512,
                "max_prompt_length": 256,
                "disable_dropout": True,
                "precompute_ref_log_probs": False,
                "reference_free": False,
                "force_use_ref_model": False,
                "sync_ref_model": False,
                "ref_model_mixup_alpha": 0.9,
                "ref_model_sync_steps": 64,
            }
        )
    elif trainer_type == "orpo":
        defaults.update(
            {
                "trainer_prefix": "orpotrainer",
                "learning_rate": 1e-6,
                "beta": 0.1,
                "max_length": 1024,
                "max_prompt_length": 512,
                "disable_dropout": True,
                "label_pad_token_id": -100,
                "generate_during_eval": False,
            }
        )
    elif trainer_type == "grpo":
        defaults.update(
            {
                "trainer_prefix": "grpotrainer",
                "learning_rate": 1e-6,
                "remove_unused_columns": False,
                "max_prompt_length": 512,
                "max_completion_length": 256,
                "beta": 0.04,
                "sync_ref_model": False,
                "ref_model_mixup_alpha": 0.9,
                "ref_model_sync_steps": 64,
                "skip_apply_chat_template": False,
                "num_return_sequences": 1,
                "top_p": 0.95,
                "top_k": 50,
                "temperature": 0.7,
            }
        )
    elif trainer_type == "sft":
        defaults.update(
            {
                "trainer_prefix": "sfttrainer",
                "learning_rate": 2e-5,
                "add_special_tokens": False,
                "packing": False,
                "dataset_batch_size": 1000,
                "num_of_sequences": 1024,
            }
        )
    elif trainer_type == "reward":
        defaults.update(
            {
                "trainer_prefix": "rewardtrainer",
                "max_sequence_length": 1024,
                "disable_dropout": True,
                "center_rewards_coefficient": 0.1,
                "remove_unused_columns": False,
            }
        )
    elif trainer_type == "distillation":
        defaults.update(
            {
                "trainer_prefix": "distillationtrainer",
                "temperature": 2.0,
                "alpha": 0.9,
            }
        )

    for key, value in defaults.items():
        config.setdefault(key, value)

    config["trainer_type"] = trainer_type

    if "max_completion_length" not in config and trainer_type in ["dpo", "orpo"]:
        if "max_length" in config and "max_prompt_length" in config:
            config["max_completion_length"] = config["max_length"] - config["max_prompt_length"]

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config.get("total_batch_size", 32)

    if "loss_config" in config:
        from easydel.infra.loss_utils import LossConfig

        config["loss_config"] = LossConfig(**config["loss_config"])
    return config


def get_trainer_class(trainer_type: str):
    """Get the appropriate trainer class based on type.

    Maps trainer type strings to their corresponding trainer class implementations.

    Args:
        trainer_type: Type of trainer (dpo, orpo, grpo, sft, reward, distillation, base)

    Returns:
        Trainer class corresponding to the specified type, defaults to base Trainer

    Example:
        >>> trainer_cls = get_trainer_class("dpo")
        >>> trainer_cls.__name__
        'DPOTrainer'
    """
    from easydel.utils import Registry

    return Registry.get_or_raise("trainer", trainer_type.lower())


def get_training_arguments_class(trainer_type: str):
    """Get the appropriate TrainingArguments class based on trainer type.

    Maps trainer type strings to their corresponding configuration classes.

    Args:
        trainer_type: Type of trainer (dpo, orpo, grpo, sft, reward, distillation, base)

    Returns:
        TrainingArguments class corresponding to the specified type, defaults to base TrainingArguments

    Example:
        >>> args_cls = get_training_arguments_class("sft")
        >>> args_cls.__name__
        'SFTConfig'
    """

    from easydel.utils import Registry

    return Registry.get_or_raise("trainer-arguments", trainer_type.lower())
