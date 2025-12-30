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
available in EasyDeL, including DPO, ORPO, GRPO, PPO, SFT, Reward, and Distillation trainers.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, NotRequired, TypedDict

_TRAINER_TYPE_ALIASES: dict[str, str] = {
    "nash_md": "nash-md",
}


def _normalize_trainer_type(trainer_type: str) -> str:
    """Normalize trainer type string to canonical form.

    Handles case normalization and known aliases (e.g., "nash_md" -> "nash-md").

    Args:
        trainer_type: Trainer type string to normalize.

    Returns:
        Normalized trainer type string in lowercase with aliases resolved.

    Example:
        >>> _normalize_trainer_type("DPO")
        'dpo'
        >>> _normalize_trainer_type("nash_md")
        'nash-md'
    """
    normalized = trainer_type.lower()
    return _TRAINER_TYPE_ALIASES.get(normalized, normalized)


class LossConfig(TypedDict, total=False):
    """Configuration for loss computation in training.

    Attributes:
        ignore_index: Token index to ignore in loss computation (default: -100 for padding).
        label_smoothing: Label smoothing factor (0.0 = no smoothing).
        z_loss: Z-loss regularization coefficient for router auxiliary loss.
        loss_normalizing_factor: How to normalize loss across tokens/sequences.
        num_labels: Number of labels for classification tasks.
        problem_type: Type of classification problem.
        divide_weight_sum: Whether to divide by sum of weights.
        shift_tokens: Whether to shift tokens for causal LM loss computation.
        break_on_nan: Whether to raise an error on NaN loss values.
        reduction: Loss reduction method ("none", "mean", or "sum").
        num_classification_labels: Number of classification labels.
        classification_problem_type: Type of classification problem for sequence classification.
        chunk_vocab_size: Chunk size for vocabulary-chunked cross entropy.
        chunk_token_size: Chunk size for token-chunked cross entropy.
        chunk_block_size: Block size for chunked computations.
        compute_dtype: Dtype for loss computation ("fp32" or "bf16").
    """

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

    trainer_type: NotRequired[
        Literal[
            "sft",
            "base",
            "dpo",
            "grpo",
            "gfpo",
            "gspo",
            "ppo",
            "orpo",
            "reward",
            "distillation",
            "bco",
            "cpo",
            "gkd",
            "kto",
            "nash_md",
            "nash-md",
            "xpo",
        ]
    ]
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
    max_length: NotRequired[int | None]
    # Deprecated alias (kept for backward compatibility).
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

    # Generation preview configuration
    generation_top_p: NotRequired[float | None]
    generation_top_k: NotRequired[int | None]
    generation_temperature: NotRequired[float | None]
    generation_do_sample: NotRequired[bool | None]
    generation_num_return_sequences: NotRequired[int | None]
    generation_max_new_tokens: NotRequired[int | None]
    generation_shard_inputs: NotRequired[bool]
    generation_interval: NotRequired[int | None]
    generation_prompts: NotRequired[list[str | dict[str, Any]]]
    generation_use_train_prompts: NotRequired[bool]
    generation_num_prompts: NotRequired[int]
    generation_dataset_prompt_field: NotRequired[str | None]
    generation_extra_kwargs: NotRequired[dict[str, Any] | None]
    generation_config_overrides: NotRequired[dict[str, Any] | None]
    generation_seed: NotRequired[int | None]
    generation_preview_print: NotRequired[bool]
    generation_log_to_wandb: NotRequired[bool]

    # eSurge integration for generation
    use_esurge_generation: NotRequired[bool]
    esurge_hbm_utilization: NotRequired[float | None]
    esurge_max_num_seqs: NotRequired[int | None]
    esurge_min_input_pad: NotRequired[int | None]
    esurge_page_size: NotRequired[int | None]
    esurge_silent_mode: NotRequired[bool]


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


class PPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Proximal Policy Optimization trainer (PPOConfig)."""

    max_prompt_length: NotRequired[int]
    max_completion_length: NotRequired[int]
    dataset_num_proc: NotRequired[int | None]
    reward_weights: NotRequired[list[float] | None]
    kl_coef: NotRequired[float]
    kl_estimator: NotRequired[Literal["k1", "k3"]]
    cliprange: NotRequired[float]
    vf_coef: NotRequired[float]
    cliprange_value: NotRequired[float]
    gamma: NotRequired[float]
    lam: NotRequired[float]
    whiten_rewards: NotRequired[bool]
    whiten_advantages: NotRequired[bool]
    entropy_coef: NotRequired[float]
    missing_eos_penalty: NotRequired[float | None]
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

    max_length: NotRequired[int | None]
    disable_dropout: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    center_rewards_coefficient: NotRequired[float | None]


class DistillationTrainerCfg(BaseTrainerCfg):
    """Configuration for Knowledge Distillation trainer (DistillationConfig)."""

    temperature: NotRequired[float]
    alpha: NotRequired[float]


class KTOTrainerCfg(BaseTrainerCfg):
    """Configuration for Kahneman-Tversky Optimization trainer (KTOConfig)."""

    beta: NotRequired[float]
    desirable_weight: NotRequired[float]
    undesirable_weight: NotRequired[float]
    loss_type: NotRequired[Literal["kto", "apo_zero_unpaired"]]
    label_pad_token_id: NotRequired[int]
    padding_value: NotRequired[int | None]
    max_length: NotRequired[int | None]
    max_prompt_length: NotRequired[int | None]
    max_completion_length: NotRequired[int | None]
    is_encoder_decoder: NotRequired[bool | None]
    disable_dropout: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    precompute_ref_log_probs: NotRequired[bool]


class BCOTrainerCfg(BaseTrainerCfg):
    """Configuration for Binary Classifier Optimization trainer (BCOConfig)."""

    beta: NotRequired[float]
    label_pad_token_id: NotRequired[int]
    padding_value: NotRequired[int | None]
    max_length: NotRequired[int | None]
    max_prompt_length: NotRequired[int | None]
    max_completion_length: NotRequired[int | None]
    disable_dropout: NotRequired[bool]
    generate_during_eval: NotRequired[bool]
    is_encoder_decoder: NotRequired[bool | None]
    precompute_ref_log_probs: NotRequired[bool]
    model_init_kwargs: NotRequired[dict[str, Any] | None]
    ref_model_init_kwargs: NotRequired[dict[str, Any] | None]
    dataset_num_proc: NotRequired[int | None]
    prompt_sample_size: NotRequired[int]
    min_density_ratio: NotRequired[float]
    max_density_ratio: NotRequired[float]


class CPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Contrastive Preference Optimization trainer (CPOConfig)."""

    beta: NotRequired[float]
    label_smoothing: NotRequired[float]
    loss_type: NotRequired[Literal["sigmoid", "hinge", "ipo", "simpo", "alphapo"]]
    disable_dropout: NotRequired[bool]
    cpo_alpha: NotRequired[float]
    simpo_gamma: NotRequired[float]
    alpha: NotRequired[float]
    label_pad_token_id: NotRequired[int]
    padding_value: NotRequired[int | None]
    max_length: NotRequired[int | None]
    max_prompt_length: NotRequired[int | None]
    max_completion_length: NotRequired[int | None]
    is_encoder_decoder: NotRequired[bool | None]
    dataset_num_proc: NotRequired[int | None]


class GKDTrainerCfg(SFTTrainerCfg):
    """Configuration for Generalized Knowledge Distillation trainer (GKDConfig)."""

    temperature: NotRequired[float]
    lmbda: NotRequired[float]
    beta: NotRequired[float]
    max_new_tokens: NotRequired[int]
    disable_dropout: NotRequired[bool]
    seq_kd: NotRequired[bool]


class NashMDTrainerCfg(GRPOTrainerCfg):
    """Configuration for Nash Mixture-of-Decoders trainer (NashMDConfig)."""

    beta: NotRequired[float | list[float]]
    mixture_coef: NotRequired[float | list[float]]
    missing_eos_penalty: NotRequired[float | None]


class XPOTrainerCfg(GRPOTrainerCfg):
    """Configuration for Exploratory Preference Optimization trainer (XPOConfig)."""

    loss_type: NotRequired[Literal["sigmoid", "ipo"]]
    beta: NotRequired[float | list[float]]
    alpha: NotRequired[float | list[float]]
    missing_eos_penalty: NotRequired[float | None]


class TrainerConfig(
    ORPOTrainerCfg,
    GRPOTrainerCfg,
    PPOTrainerCfg,
    SFTTrainerCfg,
    RewardTrainerCfg,
    DistillationTrainerCfg,
    KTOTrainerCfg,
    BCOTrainerCfg,
    CPOTrainerCfg,
    GKDTrainerCfg,
    NashMDTrainerCfg,
    XPOTrainerCfg,
    BaseTrainerCfg,
    DPOTrainerCfg,
): ...


BASE_TRAINER_DEFAULTS: BaseTrainerCfg = {
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
    "max_length": 4096,
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
    # Generation preview defaults
    "generation_shard_inputs": True,
    "generation_use_train_prompts": False,
    "generation_num_prompts": 1,
    "generation_dataset_prompt_field": "prompt",
    "generation_preview_print": False,
    "generation_log_to_wandb": True,
    # eSurge integration defaults
    "use_esurge_generation": True,
    "esurge_hbm_utilization": 0.45,
    "esurge_page_size": 32,
    "esurge_silent_mode": True,
}

# Trainer-specific defaults (only overrides, not full configs)
TRAINER_SPECIFIC_DEFAULTS: dict[str, TrainerConfig] = {
    "dpo": {
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
    },
    "orpo": {
        "trainer_prefix": "orpotrainer",
        "learning_rate": 1e-6,
        "beta": 0.1,
        "max_length": 1024,
        "max_prompt_length": 512,
        "disable_dropout": True,
        "label_pad_token_id": -100,
        "generate_during_eval": False,
    },
    "grpo": {
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
    },
    "ppo": {
        "trainer_prefix": "ppotrainer",
        "learning_rate": 1e-6,
        "remove_unused_columns": False,
        "max_length": 768,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "reward_weights": None,
        "kl_coef": 0.05,
        "kl_estimator": "k1",
        "cliprange": 0.2,
        "vf_coef": 0.1,
        "cliprange_value": 0.2,
        "gamma": 1.0,
        "lam": 0.95,
        "whiten_rewards": False,
        "whiten_advantages": True,
        "entropy_coef": 0.0,
        "missing_eos_penalty": None,
        "skip_apply_chat_template": False,
        "num_return_sequences": 1,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
    },
    "sft": {
        "trainer_prefix": "sfttrainer",
        "learning_rate": 2e-5,
        "add_special_tokens": False,
        "packing": False,
        "dataset_batch_size": 1000,
        "num_of_sequences": 1024,
    },
    "reward": {
        "trainer_prefix": "rewardtrainer",
        "max_length": 1024,
        "disable_dropout": True,
        "center_rewards_coefficient": 0.1,
        "remove_unused_columns": False,
    },
    "distillation": {
        "trainer_prefix": "distillationtrainer",
        "temperature": 2.0,
        "alpha": 0.9,
    },
    "kto": {
        "trainer_prefix": "ktotrainer",
        "learning_rate": 1e-6,
        "beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "loss_type": "kto",
        "label_pad_token_id": -100,
        "max_length": 1024,
        "max_prompt_length": 512,
        "disable_dropout": True,
        "precompute_ref_log_probs": False,
    },
    "bco": {
        "trainer_prefix": "bcotrainer",
        "learning_rate": 1e-6,
        "beta": 0.1,
        "label_pad_token_id": -100,
        "max_length": 1024,
        "max_prompt_length": 512,
        "disable_dropout": True,
        "generate_during_eval": False,
        "precompute_ref_log_probs": False,
        "prompt_sample_size": 1024,
        "min_density_ratio": 0.5,
        "max_density_ratio": 10.0,
    },
    "cpo": {
        "trainer_prefix": "cpotrainer",
        "learning_rate": 1e-6,
        "beta": 0.1,
        "label_smoothing": 0.0,
        "loss_type": "sigmoid",
        "disable_dropout": True,
        "cpo_alpha": 1.0,
        "simpo_gamma": 0.5,
        "alpha": 0.0,
        "label_pad_token_id": -100,
        "max_length": 1024,
        "max_prompt_length": 512,
    },
    "gkd": {
        "trainer_prefix": "gkdtrainer",
        "learning_rate": 2e-5,
        "temperature": 0.9,
        "lmbda": 0.5,
        "beta": 0.5,
        "max_new_tokens": 128,
        "disable_dropout": True,
        "seq_kd": False,
        "add_special_tokens": False,
        "packing": False,
        "dataset_batch_size": 1000,
        "num_of_sequences": 1024,
    },
    "nash_md": {
        "trainer_prefix": "nashmdtrainer",
        "learning_rate": 1e-6,
        "remove_unused_columns": False,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "beta": 0.1,
        "mixture_coef": 0.5,
        "sync_ref_model": False,
        "ref_model_mixup_alpha": 0.9,
        "ref_model_sync_steps": 64,
        "skip_apply_chat_template": False,
        "num_return_sequences": 1,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
    },
    "nash-md": {
        "trainer_prefix": "nashmdtrainer",
        "learning_rate": 1e-6,
        "remove_unused_columns": False,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "beta": 0.1,
        "mixture_coef": 0.5,
        "sync_ref_model": False,
        "ref_model_mixup_alpha": 0.9,
        "ref_model_sync_steps": 64,
        "skip_apply_chat_template": False,
        "num_return_sequences": 1,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
    },
    "xpo": {
        "trainer_prefix": "xpotrainer",
        "learning_rate": 1e-6,
        "remove_unused_columns": False,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "loss_type": "sigmoid",
        "beta": 0.1,
        "alpha": 1e-5,
        "sync_ref_model": False,
        "ref_model_mixup_alpha": 0.9,
        "ref_model_sync_steps": 64,
        "skip_apply_chat_template": False,
        "num_return_sequences": 1,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
    },
}

# Trainers that need max_completion_length auto-computed
_TRAINERS_WITH_COMPLETION_LENGTH = frozenset({"dpo", "orpo", "kto", "bco", "cpo", "ppo"})


def register_trainer_defaults(trainer_type: str, defaults: TrainerConfig) -> None:
    """Register default configuration for a trainer type.

    This allows external modules to register their own trainer defaults
    without modifying this module directly.

    Args:
        trainer_type: The trainer type identifier (lowercase)
        defaults: Dictionary of default values for this trainer

    Example:
        >>> register_trainer_defaults("my_trainer", {
        ...     "trainer_prefix": "mytrainer",
        ...     "learning_rate": 1e-5,
        ...     "custom_param": 42,
        ... })
    """
    TRAINER_SPECIFIC_DEFAULTS[_normalize_trainer_type(trainer_type)] = defaults


def get_trainer_defaults(trainer_type: str) -> TrainerConfig:
    """Get merged defaults for a trainer type.

    Merges base defaults with trainer-specific defaults.

    Args:
        trainer_type: The trainer type identifier

    Returns:
        Complete defaults dictionary for the trainer
    """
    trainer_type = _normalize_trainer_type(trainer_type)
    defaults = dict(BASE_TRAINER_DEFAULTS)
    if trainer_type in TRAINER_SPECIFIC_DEFAULTS:
        defaults.update(TRAINER_SPECIFIC_DEFAULTS[trainer_type])
    return defaults  # type: ignore[return-value]


def normalize_trainer_config(config: dict[str, Any]) -> TrainerConfig:
    """Normalize and validate trainer configuration.

    This function takes raw trainer configuration and applies appropriate defaults
    based on the trainer type, ensuring all required fields are present.

    Uses the modular `BASE_TRAINER_DEFAULTS` and `TRAINER_SPECIFIC_DEFAULTS` registry
    to apply defaults. New trainer types can be registered using `register_trainer_defaults`.

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
    trainer_type = _normalize_trainer_type(config.get("trainer_type", "sft"))

    if "max_sequence_length" in config and "max_length" not in config:
        warnings.warn(
            "`max_sequence_length` is deprecated; use `max_length` instead.",
            FutureWarning,
            stacklevel=2,
        )
        config["max_length"] = config.pop("max_sequence_length")
    elif (
        "max_sequence_length" in config
        and "max_length" in config
        and config["max_sequence_length"] != config["max_length"]
    ):
        warnings.warn(
            "Both `max_length` and `max_sequence_length` are set; ignoring `max_sequence_length`.",
            FutureWarning,
            stacklevel=2,
        )
        config.pop("max_sequence_length", None)

    # Get merged defaults from registry
    defaults = get_trainer_defaults(trainer_type)

    # Apply defaults (config values take precedence)
    for key, value in defaults.items():
        config.setdefault(key, value)

    config["trainer_type"] = trainer_type

    # Auto-compute max_completion_length for applicable trainers
    if "max_completion_length" not in config and trainer_type in _TRAINERS_WITH_COMPLETION_LENGTH:
        if "max_length" in config and "max_prompt_length" in config:
            config["max_completion_length"] = config["max_length"] - config["max_prompt_length"]

    # Default eval_batch_size to total_batch_size
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config.get("total_batch_size", 32)

    # Convert loss_config dict to LossConfig instance
    if "loss_config" in config and isinstance(config["loss_config"], dict):
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

    return Registry.get_or_raise("trainer", _normalize_trainer_type(trainer_type))


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

    return Registry.get_or_raise("trainer-arguments", _normalize_trainer_type(trainer_type))
