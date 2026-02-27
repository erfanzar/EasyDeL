# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Comprehensive trainer configuration support for ELM (EasyDeL Large Model).

This module provides TypedDict-based configuration classes for all trainer types
available in EasyDeL. It serves as the central configuration registry for training
workflows, supporting various optimization paradigms including:

- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Odds Ratio Preference Optimization (ORPO)
- Group Relative Policy Optimization (GRPO)
- Self-Distillation Policy Optimization (SDPO)
- Proximal Policy Optimization (PPO)
- Kahneman-Tversky Optimization (KTO)
- Binary Classifier Optimization (BCO)
- Contrastive Preference Optimization (CPO)
- Generalized Knowledge Distillation (GKD)
- Nash Mixture-of-Decoders (Nash-MD)
- Exploratory Preference Optimization (XPO)
- Reward Model Training
- Knowledge Distillation

The module uses TypedDict for type-safe configuration validation while maintaining
flexibility through NotRequired fields. Configuration classes follow an inheritance
hierarchy where specialized trainers extend base configurations.

Example:
    Basic usage with trainer configuration:

    >>> from easydel.infra.elarge_model.trainer_types import (
    ...     normalize_trainer_config,
    ...     get_trainer_class,
    ... )
    >>> config = {
    ...     "trainer_type": "dpo",
    ...     "learning_rate": 1e-6,
    ...     "beta": 0.1,
    ... }
    >>> normalized = normalize_trainer_config(config)
    >>> trainer_cls = get_trainer_class("dpo")

Note:
    All configuration classes use total=False to make all fields optional,
    allowing partial configuration with sensible defaults applied at runtime.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, NotRequired, TypedDict, cast

_TRAINER_TYPE_ALIASES: dict[str, str] = {
    "nash_md": "nash-md",
}
"""Mapping of trainer type aliases to their canonical forms.

This dictionary allows alternative naming conventions (e.g., underscore vs hyphen)
to be transparently normalized to canonical trainer type identifiers.
"""


def _normalize_trainer_type(trainer_type: str) -> str:
    """Normalize trainer type string to canonical form.

    Handles case normalization and known aliases (e.g., "nash_md" -> "nash-md").
    This ensures consistent trainer type identification regardless of how users
    specify the trainer type in their configuration.

    Args:
        trainer_type: Trainer type string to normalize. Can be in any case
            and may use alternative naming conventions.

    Returns:
        Normalized trainer type string in lowercase with aliases resolved
        to their canonical form.

    Example:
        >>> _normalize_trainer_type("DPO")
        'dpo'
        >>> _normalize_trainer_type("nash_md")
        'nash-md'
        >>> _normalize_trainer_type("GRPO")
        'grpo'
    """
    normalized = trainer_type.lower()
    return _TRAINER_TYPE_ALIASES.get(normalized, normalized)


class LossConfig(TypedDict, total=False):
    """Configuration for loss computation in training.

    This TypedDict defines all configurable parameters for loss calculation,
    supporting various loss types including cross-entropy, classification losses,
    and chunked computation strategies for memory efficiency.

    The loss configuration supports advanced features like label smoothing,
    Z-loss regularization for MoE models, and flexible reduction strategies.

    Attributes:
        ignore_index: Token index to ignore in loss computation.
            Defaults to -100, which is the standard PyTorch convention
            for padding tokens in language modeling.
        label_smoothing: Label smoothing factor for regularization.
            A value of 0.0 means no smoothing (hard labels), while
            positive values blend the target distribution with a
            uniform distribution over all labels.
        z_loss: Z-loss regularization coefficient for router auxiliary loss
            in Mixture-of-Experts models. Helps prevent router collapse.
        loss_normalizing_factor: Strategy for normalizing loss across tokens
            and sequences. Options include:
            - "NO_WEIGHT_NUM_REAL_TARGET_TOKENS": Normalize by real tokens without weights
            - "NUM_REAL_TARGET_TOKENS": Normalize by count of real (non-padding) tokens
            - "NUM_TOTAL_TARGET_TOKENS": Normalize by total token count
            - "AVERAGE_PER_SEQUENCE": Average loss per sequence
        num_labels: Number of labels for classification tasks.
            Used in sequence classification heads.
        problem_type: Type of classification problem being solved.
            Affects loss computation and output interpretation.
        divide_weight_sum: Whether to divide loss by the sum of sample weights
            when using weighted loss computation.
        shift_tokens: Whether to shift tokens for causal language model loss
            computation. When True, targets are shifted by one position
            to predict the next token.
        break_on_nan: Whether to raise an error when NaN loss values are
            detected during training. Useful for debugging numerical issues.
        reduction: Loss reduction method applied after computing per-sample
            losses. One of "none" (no reduction), "mean" (average), or
            "sum" (total).
        num_classification_labels: Number of classification labels for
            sequence classification tasks.
        classification_problem_type: Type of classification problem for
            sequence classification. One of "regression",
            "single_label_classification", or "multi_label_classification".
        chunk_vocab_size: Chunk size for vocabulary-chunked cross entropy
            computation. Reduces memory usage for large vocabularies.
        chunk_token_size: Chunk size for token-chunked cross entropy
            computation. Reduces memory usage for long sequences.
        chunk_block_size: Block size for chunked computations.
            Controls granularity of memory-efficient computation.
        compute_dtype: Data type for loss computation. Use "fp32" for
            numerical stability or "bf16" for faster computation.

    Example:
        >>> loss_config: LossConfig = {
        ...     "ignore_index": -100,
        ...     "label_smoothing": 0.1,
        ...     "reduction": "mean",
        ...     "shift_tokens": True,
        ... }
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
    """Base configuration shared by all trainers (TrainingArguments).

    This TypedDict serves as the foundation for all trainer configurations,
    providing common parameters for learning rate scheduling, optimization,
    data loading, checkpointing, logging, and distributed training.

    All specialized trainer configurations inherit from this base class,
    ensuring consistent parameter naming and behavior across different
    training paradigms.

    Attributes:
        trainer_type: Type of trainer to use. Determines which trainer class
            and configuration defaults are applied.
        learning_rate: Initial learning rate for optimization.
        learning_rate_end: Final learning rate for schedulers that support
            annealing to a specific value.
        num_train_epochs: Number of complete passes through the training dataset.
        max_training_steps: Maximum number of training steps. If set, overrides
            num_train_epochs.
        per_epoch_training_steps: Number of training steps per epoch.
            Used for custom epoch definitions.
        per_epoch_evaluation_steps: Number of evaluation steps per epoch.
        total_batch_size: Total batch size across all devices and accumulation steps.
        eval_batch_size: Batch size for evaluation. Defaults to total_batch_size.
        gradient_accumulation_steps: Number of gradient accumulation steps before
            performing an optimizer update.
        optimizer: Optimizer type string (e.g., "adamw", "lion", "sgd").
        scheduler: Learning rate scheduler type (e.g., "linear", "cosine", "none").
        warmup_steps: Number of warmup steps for learning rate scheduling.
        weight_decay: L2 regularization coefficient for optimizer.
        clip_grad: Maximum gradient norm for gradient clipping. None disables clipping.
        extra_optimizer_kwargs: Additional keyword arguments passed to the optimizer.
        custom_scheduler: Custom scheduler function or configuration.
        dataloader_num_workers: Number of worker processes for data loading.
        dataloader_pin_memory: Whether to pin memory in data loaders for faster
            GPU transfer.
        remove_unused_columns: Whether to remove columns not used by the model
            from the dataset.
        ids_to_pop_from_dataset: List of column names to explicitly remove from dataset.
        shuffle_train_dataset: Whether to shuffle the training dataset.
        shuffle_seed_train: Random seed for dataset shuffling.
        use_data_collactor: Whether to use a data collator for batch preparation.
        use_grain: Whether to use Grain for efficient data loading.
        grain_shard_index: Shard index for Grain-based data loading.
        grain_shard_count: Total number of shards for Grain-based data loading.
        offload_dataset: Whether to offload dataset to specified device.
        offload_device_type: Device type for dataset offloading (e.g., "cpu").
        offload_device_index: Device index for dataset offloading.
        do_train: Whether to run training.
        do_eval: Whether to run evaluation.
        do_last_save: Whether to save checkpoint at training completion.
        is_fine_tuning: Whether this is a fine-tuning run (vs pre-training).
        init_tx: Whether to initialize the optimizer (transformer) state.
        train_on_inputs: Whether to compute loss on input tokens in addition
            to target tokens.
        aux_loss_enabled: Whether to enable auxiliary losses (e.g., for MoE).
        training_time_limit: Maximum training time as string (e.g., "2h30m").
        step_start_point: Step number to resume training from.
        resume_if_possible: Whether to automatically resume from latest checkpoint.
        truncation_mode: How to truncate sequences that exceed max_length.
            "keep_end" preserves the end, "keep_start" preserves the beginning.
        max_length: Maximum sequence length for inputs.
        max_sequence_length: Deprecated alias for max_length.
        save_interval_minutes: Save checkpoint every N minutes.
        save_steps: Save checkpoint every N steps.
        save_total_limit: Maximum number of checkpoints to keep.
        save_directory: Directory path for saving checkpoints.
        save_optimizer_state: Whether to include optimizer state in checkpoints.
        remove_ckpt_after_load: Whether to remove checkpoint after loading.
        evaluation_steps: Run evaluation every N steps.
        max_evaluation_steps: Maximum number of evaluation steps per evaluation run.
        log_steps: Log training metrics every N steps.
        report_steps: Report metrics to external trackers every N steps.
        log_all_workers: Whether all workers should log, or only the main process.
        log_grad_norms: Whether to log gradient norms during training.
        report_metrics: Whether to report metrics to external trackers (e.g., WandB).
        metrics_to_show_in_rich_pbar: List of metric names to display in Rich progress bar.
        progress_bar_type: Type of progress bar ("tqdm", "rich", or "json").
        weight_distribution_pattern: Regex pattern for selecting weights to log distributions.
        weight_distribution_log_steps: Log weight distributions every N steps.
        verbose: Whether to enable verbose logging output.
        process_zero_is_admin: Whether process 0 has admin privileges for logging/saving.
        use_wandb: Whether to use Weights & Biases for experiment tracking.
        wandb_entity: WandB entity (team or user) for logging.
        wandb_name: Name for the WandB run.
        trainer_prefix: Prefix for the trainer name in logging and checkpoints.
        backend: JAX backend to use (e.g., "gpu", "tpu").
        auto_shard_states: Whether to automatically shard model states across devices.
        performance_mode: Whether to enable performance optimizations.
        track_memory: Whether to track memory usage. Can be bool or float threshold.
        low_mem_usage: Whether to enable low memory usage optimizations.
        quantization_mode: QAT/STE quantization mode for forward-pass emulation.
        quantization_group_size: Group size for group-wise quantizers.
        quantization_bits: Bit-width for configurable quantizers (e.g. affine).
        tensor_straight_through: Optional per-tensor STE transform callable.
        straight_through_emulator: Optional graphstate-level STE transform callable.
        model_name: Name of the model being trained.
        model_parameters: Dictionary of model configuration parameters.
        frozen_parameters: Regex pattern for parameters to freeze during training.
        loss_config: Configuration for loss computation.
        jax_distributed_config: Configuration for JAX distributed setup.
        step_partition_spec: PartitionSpec for step function sharding.
        state_apply_fn_kwarguments_to_model: Keyword arguments passed to model
            during state application.
        sparsify_module: Whether to enable module sparsification.
        sparse_module_type: Type of sparse representation (e.g., "bcoo").
        pruning_module: Pruning module or configuration.
        tx_mu_dtype: Data type for optimizer momentum terms.
        generation_top_p: Top-p (nucleus) sampling probability for generation preview.
        generation_top_k: Top-k sampling parameter for generation preview.
        generation_temperature: Temperature for generation preview sampling.
        generation_do_sample: Whether to use sampling for generation preview.
        generation_num_return_sequences: Number of sequences to generate in preview.
        generation_max_new_tokens: Maximum new tokens for generation preview.
        generation_shard_inputs: Whether to shard inputs for generation.
        generation_interval: Run generation preview every N steps.
        generation_prompts: List of prompts for generation preview.
        generation_use_train_prompts: Whether to use prompts from training data.
        generation_num_prompts: Number of prompts to use from dataset.
        generation_dataset_prompt_field: Field name containing prompts in dataset.
        generation_extra_kwargs: Extra keyword arguments for generation.
        generation_config_overrides: Overrides for generation configuration.
        generation_seed: Random seed for generation preview.
        generation_preview_print: Whether to print generation preview to console.
        generation_log_to_wandb: Whether to log generation preview to WandB.
        use_esurge_generation: Whether to use eSurge for optimized generation.
        esurge_use_tqdm: Whether to show tqdm progress for eSurge generation.
        esurge_hbm_utilization: HBM utilization target for eSurge.
        esurge_max_num_seqs: Maximum concurrent sequences for eSurge.
        esurge_min_input_pad: Minimum input padding for eSurge.
        esurge_page_size: Page size for eSurge paged attention.
        esurge_silent_mode: Whether to suppress eSurge output.
        esurge_max_num_batched_tokens: Max num tokens to batch together for eSurge generation.
        esurge_enable_prefix_caching: Enable/disable eSurge prefix caching.
        esurge_data_parallelism_axis: Mesh axis name for eSurge data-parallel KV pages.
        esurge_max_num_seq_buckets: Optional explicit sequence-capacity buckets for eSurge runner compilation.

    Example:
        >>> config: BaseTrainerCfg = {
        ...     "learning_rate": 5e-5,
        ...     "num_train_epochs": 3,
        ...     "total_batch_size": 32,
        ...     "optimizer": "adamw",
        ...     "warmup_steps": 100,
        ... }
    """

    trainer_type: NotRequired[
        Literal[
            "sft",
            "base",
            "dpo",
            "grpo",
            "sdpo",
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
    quantization_mode: NotRequired[Literal["nf4", "affine", "mxfp8", "nvfp8", "mxfp4", "nvfp4"] | None]
    quantization_group_size: NotRequired[int | None]
    quantization_bits: NotRequired[int | None]
    tensor_straight_through: NotRequired[Any | None]
    straight_through_emulator: NotRequired[Any | None]

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
    esurge_use_tqdm: NotRequired[bool]
    esurge_hbm_utilization: NotRequired[float | None]
    esurge_max_num_seqs: NotRequired[int | None]
    esurge_min_input_pad: NotRequired[int | None]
    esurge_page_size: NotRequired[int | None]
    esurge_silent_mode: NotRequired[bool]
    esurge_max_num_batched_tokens: NotRequired[int | None]
    esurge_enable_prefix_caching: NotRequired[bool | None]
    esurge_data_parallelism_axis: NotRequired[str | None]
    esurge_max_num_seq_buckets: NotRequired[list[int] | None]


class DPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Direct Preference Optimization trainer (DPOConfig).

    DPO is an algorithm for training language models from human preferences
    without explicit reward modeling. It directly optimizes the policy using
    a classification loss derived from the Bradley-Terry preference model.

    This configuration extends BaseTrainerCfg with DPO-specific parameters
    including the beta temperature, loss variants, and reference model settings.

    Attributes:
        beta: Temperature parameter controlling the deviation from the reference
            policy. Higher values mean stronger preference for preferred responses.
            Typical values range from 0.1 to 0.5.
        label_smoothing: Label smoothing factor for the preference loss.
            Helps prevent overconfident predictions.
        loss_type: Type of DPO loss to use. Options include:
            - "sigmoid": Standard DPO loss (default)
            - "hinge": Hinge loss variant
            - "ipo": Identity Policy Optimization
            - "exo_pair": Exo pair loss
            - "nca_pair": NCA pair loss
            - "robust": Robust DPO loss
            - "bco_pair": BCO pair loss
            - "sppo_hard": SPPO hard loss
            - "aot": AOT loss
            - "aot_pair": AOT pair loss
            - "apo_zero": APO zero loss
            - "apo_down": APO down loss
        use_weighting: Whether to use importance weighting in loss computation.
        label_pad_token_id: Token ID used for padding labels. Typically -100.
        padding_value: Value used for padding sequences.
        max_length: Maximum total sequence length (prompt + completion).
        max_prompt_length: Maximum length for the prompt portion.
        max_completion_length: Maximum length for the completion portion.
            Auto-computed as max_length - max_prompt_length if not specified.
        is_encoder_decoder: Whether the model is encoder-decoder architecture.
        disable_dropout: Whether to disable dropout during training for
            more stable preference learning.
        precompute_ref_log_probs: Whether to precompute reference model log
            probabilities before training for memory efficiency.
        dataset_num_proc: Number of processes for dataset preprocessing.
        reference_free: Whether to train without a reference model.
        force_use_ref_model: Force use of reference model even if not needed.
        sync_ref_model: Whether to synchronize reference model with policy
            during training.
        ref_model_mixup_alpha: Mixup alpha for reference model synchronization.
            Controls interpolation between old and new reference.
        ref_model_sync_steps: Number of steps between reference model syncs.
        rpo_alpha: Alpha parameter for RPO loss variant.
        tools: List of tool definitions for tool-use training scenarios.

    Example:
        >>> config: DPOTrainerCfg = {
        ...     "trainer_type": "dpo",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "loss_type": "sigmoid",
        ...     "max_length": 512,
        ...     "max_prompt_length": 256,
        ... }
    """

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
    """Configuration for Odds Ratio Preference Optimization trainer (ORPOConfig).

    ORPO is a preference optimization method that uses odds ratios to directly
    optimize the model without requiring a separate reference model. It combines
    supervised fine-tuning with preference optimization in a single training phase.

    This configuration extends BaseTrainerCfg with ORPO-specific parameters
    for controlling the preference learning behavior.

    Attributes:
        beta: Weight for the preference optimization term relative to the
            language modeling loss. Higher values prioritize preference learning.
        max_length: Maximum total sequence length (prompt + completion).
        max_prompt_length: Maximum length for the prompt portion.
        max_completion_length: Maximum length for the completion portion.
            Auto-computed as max_length - max_prompt_length if not specified.
        disable_dropout: Whether to disable dropout during training.
        label_pad_token_id: Token ID used for padding labels.
        padding_value: Value used for padding sequences.
        generate_during_eval: Whether to generate completions during evaluation
            for qualitative assessment.
        is_encoder_decoder: Whether the model is encoder-decoder architecture.
        dataset_num_proc: Number of processes for dataset preprocessing.

    Example:
        >>> config: ORPOTrainerCfg = {
        ...     "trainer_type": "orpo",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "max_length": 1024,
        ...     "max_prompt_length": 512,
        ... }
    """

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
    """Configuration for Group Relative Policy Optimization trainer (GRPOConfig).

    GRPO is a reinforcement learning algorithm that optimizes policies using
    group-relative rewards. It generates multiple completions per prompt and
    uses their relative rankings for policy updates.

    This configuration extends BaseTrainerCfg with GRPO-specific parameters
    for generation, reference model synchronization, and sampling.

    Attributes:
        beta: KL divergence coefficient controlling deviation from reference policy.
        max_prompt_length: Maximum length for input prompts.
        max_completion_length: Maximum length for generated completions.
        dataset_num_proc: Number of processes for dataset preprocessing.
        sync_ref_model: Whether to periodically synchronize the reference model
            with the current policy.
        ref_model_mixup_alpha: Mixup coefficient for reference model updates.
            Values closer to 1 mean slower reference updates.
        ref_model_sync_steps: Number of steps between reference model syncs.
        tools: List of tool definitions for tool-use scenarios.
        skip_apply_chat_template: Whether to skip applying chat template to prompts.
        num_return_sequences: Number of completions to generate per prompt.
        top_p: Top-p (nucleus) sampling probability threshold.
        top_k: Top-k sampling parameter.
        temperature: Sampling temperature for generation diversity.

    Example:
        >>> config: GRPOTrainerCfg = {
        ...     "trainer_type": "grpo",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.04,
        ...     "max_prompt_length": 512,
        ...     "max_completion_length": 256,
        ...     "num_return_sequences": 4,
        ... }
    """

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


class SDPOTrainerCfg(GRPOTrainerCfg):
    """Configuration for Self-Distillation Policy Optimization trainer (SDPOConfig).

    SDPO extends GRPO with token-level self-distillation signals derived from
    model-generated feedback contexts.

    Attributes:
        max_feedback_length: Maximum token budget reserved for textual feedback.
        distillation_type: Distillation objective variant ("kl" or "jsd").
        beta: KL regularization toward a reference model (defaults to 0.0 in SDPO).
    """

    max_feedback_length: NotRequired[int]
    distillation_type: NotRequired[Literal["kl", "jsd"]]
    beta: NotRequired[float]


class PPOTrainerCfg(BaseTrainerCfg):
    """Configuration for Proximal Policy Optimization trainer (PPOConfig).

    PPO is a policy gradient algorithm that uses clipped objective functions
    to ensure stable policy updates. It's widely used for RLHF (Reinforcement
    Learning from Human Feedback) in language model training.

    This configuration extends BaseTrainerCfg with PPO-specific parameters
    for reward processing, advantage estimation, and policy clipping.

    Attributes:
        max_prompt_length: Maximum length for input prompts.
        max_completion_length: Maximum length for generated completions.
        dataset_num_proc: Number of processes for dataset preprocessing.
        reward_weights: Weights for combining multiple reward signals.
        kl_coef: Coefficient for KL divergence penalty in the reward.
        kl_estimator: Estimator type for KL divergence ("k1" or "k3").
        cliprange: Clipping range for the policy ratio in PPO objective.
        vf_coef: Coefficient for value function loss.
        cliprange_value: Clipping range for value function updates.
        gamma: Discount factor for future rewards (0 to 1).
        lam: Lambda parameter for GAE (Generalized Advantage Estimation).
        whiten_rewards: Whether to normalize rewards across the batch.
        whiten_advantages: Whether to normalize advantages across the batch.
        entropy_coef: Coefficient for entropy bonus to encourage exploration.
        missing_eos_penalty: Penalty applied when completion lacks EOS token.
        tools: List of tool definitions for tool-use scenarios.
        skip_apply_chat_template: Whether to skip applying chat template.
        num_return_sequences: Number of completions to generate per prompt.
        top_p: Top-p (nucleus) sampling probability threshold.
        top_k: Top-k sampling parameter.
        temperature: Sampling temperature for generation.

    Example:
        >>> config: PPOTrainerCfg = {
        ...     "trainer_type": "ppo",
        ...     "learning_rate": 1e-6,
        ...     "kl_coef": 0.05,
        ...     "cliprange": 0.2,
        ...     "gamma": 0.99,
        ...     "lam": 0.95,
        ... }
    """

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
    """Configuration for Supervised Fine-Tuning trainer (SFTConfig).

    SFT is the standard approach for fine-tuning language models on
    instruction-following or task-specific datasets using next-token
    prediction loss.

    This configuration extends BaseTrainerCfg with SFT-specific parameters
    for dataset processing and sequence packing.

    Attributes:
        dataset_text_field: Name of the field containing text in the dataset.
            If None, the trainer will attempt to auto-detect.
        add_special_tokens: Whether to add special tokens (BOS/EOS) to sequences.
        packing: Whether to pack multiple sequences into a single training
            example for improved efficiency.
        dataset_num_proc: Number of processes for dataset preprocessing.
        dataset_batch_size: Batch size for dataset preprocessing operations.
        dataset_kwargs: Additional keyword arguments for dataset processing.
        eval_packing: Whether to use packing during evaluation.
        num_of_sequences: Target number of sequences per packed example.

    Example:
        >>> config: SFTTrainerCfg = {
        ...     "trainer_type": "sft",
        ...     "learning_rate": 2e-5,
        ...     "max_length": 2048,
        ...     "packing": True,
        ...     "dataset_text_field": "text",
        ... }
    """

    dataset_text_field: NotRequired[str | None]
    add_special_tokens: NotRequired[bool]
    packing: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    dataset_batch_size: NotRequired[int]
    dataset_kwargs: NotRequired[dict[str, Any] | None]
    eval_packing: NotRequired[bool | None]
    num_of_sequences: NotRequired[int]


class RewardTrainerCfg(BaseTrainerCfg):
    """Configuration for Reward Model trainer (RewardConfig).

    Reward training creates models that predict human preferences between
    response pairs. These models are used to provide reward signals for
    RLHF training methods like PPO.

    This configuration extends BaseTrainerCfg with reward model-specific
    parameters for preference learning.

    Attributes:
        max_length: Maximum sequence length for reward model inputs.
        disable_dropout: Whether to disable dropout during training for
            more deterministic reward predictions.
        dataset_num_proc: Number of processes for dataset preprocessing.
        center_rewards_coefficient: Coefficient for reward centering
            regularization. Helps prevent reward hacking.

    Example:
        >>> config: RewardTrainerCfg = {
        ...     "trainer_type": "reward",
        ...     "learning_rate": 1e-5,
        ...     "max_length": 1024,
        ...     "center_rewards_coefficient": 0.1,
        ... }
    """

    max_length: NotRequired[int | None]
    disable_dropout: NotRequired[bool]
    dataset_num_proc: NotRequired[int | None]
    center_rewards_coefficient: NotRequired[float | None]


class DistillationTrainerCfg(BaseTrainerCfg):
    """Configuration for Knowledge Distillation trainer (DistillationConfig).

    Knowledge distillation transfers knowledge from a larger teacher model
    to a smaller student model by training on soft probability distributions
    in addition to hard labels.

    This configuration extends BaseTrainerCfg with distillation-specific
    parameters for controlling the knowledge transfer process.

    Attributes:
        temperature: Temperature for softening probability distributions.
            Higher values create softer distributions that transfer more
            information about class relationships.
        alpha: Weight for distillation loss relative to task loss.
            alpha * distillation_loss + (1 - alpha) * task_loss.

    Example:
        >>> config: DistillationTrainerCfg = {
        ...     "trainer_type": "distillation",
        ...     "learning_rate": 2e-5,
        ...     "temperature": 2.0,
        ...     "alpha": 0.9,
        ... }
    """

    temperature: NotRequired[float]
    alpha: NotRequired[float]


class KTOTrainerCfg(BaseTrainerCfg):
    """Configuration for Kahneman-Tversky Optimization trainer (KTOConfig).

    KTO is a preference optimization method inspired by prospect theory that
    can learn from unpaired preference data (individual good/bad examples)
    rather than requiring paired comparisons.

    This configuration extends BaseTrainerCfg with KTO-specific parameters
    for controlling the preference learning with asymmetric weighting.

    Attributes:
        beta: Temperature parameter for the KTO loss function.
        desirable_weight: Weight for desirable (positive) examples.
            Increasing this emphasizes learning from good examples.
        undesirable_weight: Weight for undesirable (negative) examples.
            Increasing this emphasizes avoiding bad behaviors.
        loss_type: Type of KTO loss to use:
            - "kto": Standard KTO loss
            - "apo_zero_unpaired": APO zero unpaired variant
        label_pad_token_id: Token ID used for padding labels.
        padding_value: Value used for padding sequences.
        max_length: Maximum total sequence length.
        max_prompt_length: Maximum length for the prompt portion.
        max_completion_length: Maximum length for the completion portion.
        is_encoder_decoder: Whether the model is encoder-decoder architecture.
        disable_dropout: Whether to disable dropout during training.
        dataset_num_proc: Number of processes for dataset preprocessing.
        precompute_ref_log_probs: Whether to precompute reference log probs.

    Example:
        >>> config: KTOTrainerCfg = {
        ...     "trainer_type": "kto",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "desirable_weight": 1.0,
        ...     "undesirable_weight": 1.0,
        ... }
    """

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
    """Configuration for Binary Classifier Optimization trainer (BCOConfig).

    BCO treats preference optimization as a binary classification problem,
    using density ratio estimation to distinguish between preferred and
    non-preferred responses.

    This configuration extends BaseTrainerCfg with BCO-specific parameters
    for density ratio estimation and classifier training.

    Attributes:
        beta: Temperature parameter for the BCO loss.
        label_pad_token_id: Token ID used for padding labels.
        padding_value: Value used for padding sequences.
        max_length: Maximum total sequence length.
        max_prompt_length: Maximum length for the prompt portion.
        max_completion_length: Maximum length for the completion portion.
        disable_dropout: Whether to disable dropout during training.
        generate_during_eval: Whether to generate completions during evaluation.
        is_encoder_decoder: Whether the model is encoder-decoder architecture.
        precompute_ref_log_probs: Whether to precompute reference log probs.
        model_init_kwargs: Keyword arguments for model initialization.
        ref_model_init_kwargs: Keyword arguments for reference model initialization.
        dataset_num_proc: Number of processes for dataset preprocessing.
        prompt_sample_size: Number of prompts to sample for density estimation.
        min_density_ratio: Minimum value for clamping density ratios.
        max_density_ratio: Maximum value for clamping density ratios.

    Example:
        >>> config: BCOTrainerCfg = {
        ...     "trainer_type": "bco",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "prompt_sample_size": 1024,
        ...     "min_density_ratio": 0.5,
        ...     "max_density_ratio": 10.0,
        ... }
    """

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
    """Configuration for Contrastive Preference Optimization trainer (CPOConfig).

    CPO is a preference optimization method that uses contrastive learning
    objectives to align model outputs with human preferences without requiring
    a separate reference model.

    This configuration extends BaseTrainerCfg with CPO-specific parameters
    including multiple loss variants and SimPO extensions.

    Attributes:
        beta: Temperature parameter for the preference loss.
        label_smoothing: Label smoothing factor for regularization.
        loss_type: Type of CPO loss to use:
            - "sigmoid": Standard sigmoid loss
            - "hinge": Hinge loss variant
            - "ipo": Identity Policy Optimization
            - "simpo": Simple Preference Optimization
            - "alphapo": Alpha-weighted Preference Optimization
        disable_dropout: Whether to disable dropout during training.
        cpo_alpha: Alpha parameter for CPO loss weighting.
        simpo_gamma: Gamma parameter for SimPO loss variant.
        alpha: Additional alpha parameter for loss computation.
        label_pad_token_id: Token ID used for padding labels.
        padding_value: Value used for padding sequences.
        max_length: Maximum total sequence length.
        max_prompt_length: Maximum length for the prompt portion.
        max_completion_length: Maximum length for the completion portion.
        is_encoder_decoder: Whether the model is encoder-decoder architecture.
        dataset_num_proc: Number of processes for dataset preprocessing.

    Example:
        >>> config: CPOTrainerCfg = {
        ...     "trainer_type": "cpo",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "loss_type": "sigmoid",
        ...     "cpo_alpha": 1.0,
        ... }
    """

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
    """Configuration for Generalized Knowledge Distillation trainer (GKDConfig).

    GKD extends standard knowledge distillation with on-policy data generation,
    allowing the student model to learn from its own generations evaluated
    by the teacher model.

    This configuration extends SFTTrainerCfg with GKD-specific parameters
    for on-policy distillation and sequence-level knowledge distillation.

    Attributes:
        temperature: Temperature for softening teacher distributions.
        lmbda: Lambda parameter balancing on-policy and off-policy data.
        beta: Beta parameter for loss weighting.
        max_new_tokens: Maximum new tokens to generate for on-policy data.
        disable_dropout: Whether to disable dropout during training.
        seq_kd: Whether to use sequence-level knowledge distillation,
            training on teacher-generated sequences.

    Example:
        >>> config: GKDTrainerCfg = {
        ...     "trainer_type": "gkd",
        ...     "learning_rate": 2e-5,
        ...     "temperature": 0.9,
        ...     "lmbda": 0.5,
        ...     "seq_kd": False,
        ... }
    """

    temperature: NotRequired[float]
    lmbda: NotRequired[float]
    beta: NotRequired[float]
    max_new_tokens: NotRequired[int]
    disable_dropout: NotRequired[bool]
    seq_kd: NotRequired[bool]


class NashMDTrainerCfg(GRPOTrainerCfg):
    """Configuration for Nash Mixture-of-Decoders trainer (NashMDConfig).

    Nash-MD is an online RLHF algorithm that finds Nash equilibrium between
    multiple decoder policies, promoting diverse and high-quality responses
    through game-theoretic optimization.

    This configuration extends GRPOTrainerCfg with Nash-MD-specific parameters
    for mixture modeling and equilibrium computation.

    Attributes:
        beta: Temperature parameter(s) for KL divergence penalty.
            Can be a single float or list for multiple decoders.
        mixture_coef: Coefficient(s) for mixing decoder outputs.
            Can be a single float or list for multiple decoders.
        missing_eos_penalty: Penalty for completions lacking EOS token.

    Example:
        >>> config: NashMDTrainerCfg = {
        ...     "trainer_type": "nash-md",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "mixture_coef": 0.5,
        ... }
    """

    beta: NotRequired[float | list[float]]
    mixture_coef: NotRequired[float | list[float]]
    missing_eos_penalty: NotRequired[float | None]


class XPOTrainerCfg(GRPOTrainerCfg):
    """Configuration for Exploratory Preference Optimization trainer (XPOConfig).

    XPO extends preference optimization with exploration bonuses, encouraging
    the model to explore diverse responses while learning preferences.

    This configuration extends GRPOTrainerCfg with XPO-specific parameters
    for exploration-exploitation trade-off.

    Attributes:
        loss_type: Type of XPO loss ("sigmoid" or "ipo").
        beta: Temperature parameter(s) for preference loss.
            Can be a single float or list for multiple settings.
        alpha: Exploration bonus coefficient(s).
            Can be a single float or list for multiple settings.
        missing_eos_penalty: Penalty for completions lacking EOS token.

    Example:
        >>> config: XPOTrainerCfg = {
        ...     "trainer_type": "xpo",
        ...     "learning_rate": 1e-6,
        ...     "beta": 0.1,
        ...     "alpha": 1e-5,
        ...     "loss_type": "sigmoid",
        ... }
    """

    loss_type: NotRequired[Literal["sigmoid", "ipo"]]
    beta: NotRequired[float | list[float]]
    alpha: NotRequired[float | list[float]]
    missing_eos_penalty: NotRequired[float | None]


class TrainerConfig(
    ORPOTrainerCfg,
    GRPOTrainerCfg,
    SDPOTrainerCfg,
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
):
    """Unified trainer configuration combining all trainer-specific configs.

    This TypedDict combines all trainer configuration classes through multiple
    inheritance, providing a single type that can represent any trainer's
    configuration. This is primarily used for type annotations where the
    specific trainer type is not known at static analysis time.

    The class inherits from all specialized trainer configs, making it
    compatible with any trainer type's configuration dictionary.

    Note:
        Due to TypedDict inheritance limitations, some attributes may appear
        multiple times in the MRO with the same or compatible types. The
        actual validation of configuration values happens at runtime through
        the normalize_trainer_config function.

    Example:
        >>> def process_config(config: TrainerConfig) -> None:
        ...     trainer_type = config.get("trainer_type", "sft")
        ...     if trainer_type == "dpo":
        ...         beta = config.get("beta", 0.1)
        ...     # Works for any trainer type
    """

    ...


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
    "esurge_max_num_batched_tokens": None,
    "esurge_enable_prefix_caching": None,
    "esurge_data_parallelism_axis": None,
    "esurge_max_num_seq_buckets": None,
}
"""Default configuration values shared across all trainer types.

This dictionary provides sensible defaults for the BaseTrainerCfg parameters.
These values are applied first, then overridden by trainer-specific defaults
from TRAINER_SPECIFIC_DEFAULTS, and finally by user-provided configuration.

The defaults are designed to work well for most fine-tuning scenarios on
modern hardware, with conservative batch sizes and standard optimization
settings.
"""

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
    "sdpo": {
        "trainer_prefix": "sdpotrainer",
        "learning_rate": 1e-6,
        "remove_unused_columns": False,
        "max_prompt_length": 512,
        "max_completion_length": 256,
        "max_feedback_length": 256,
        "distillation_type": "jsd",
        "beta": 0.0,
        "sync_ref_model": False,
        "ref_model_mixup_alpha": 0.9,
        "ref_model_sync_steps": 64,
        "skip_apply_chat_template": False,
        "num_return_sequences": 4,
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
"""Trainer-specific default configuration overrides.

This dictionary maps trainer type identifiers to their specific default values.
These defaults override BASE_TRAINER_DEFAULTS for the corresponding trainer type.
Only parameters that differ from base defaults need to be specified.

New trainer types can be registered dynamically using register_trainer_defaults().
"""

# Trainers that need max_completion_length auto-computed
_TRAINERS_WITH_COMPLETION_LENGTH = frozenset({"dpo", "orpo", "kto", "bco", "cpo", "ppo", "sdpo"})
"""Set of trainer types that support automatic max_completion_length computation.

For these trainers, if max_completion_length is not explicitly set but max_length
and max_prompt_length are provided, max_completion_length will be automatically
computed as: max_length - max_prompt_length.
"""


def register_trainer_defaults(trainer_type: str, defaults: TrainerConfig) -> None:
    """Register default configuration for a trainer type.

    This function allows external modules to register their own trainer defaults
    without modifying this module directly. It enables extensibility for custom
    trainer implementations while maintaining compatibility with the configuration
    system.

    The registered defaults will be merged with BASE_TRAINER_DEFAULTS when
    get_trainer_defaults() or normalize_trainer_config() is called.

    Args:
        trainer_type: The trainer type identifier. Will be normalized to lowercase
            and have aliases resolved (e.g., "nash_md" -> "nash-md").
        defaults: Dictionary of default values for this trainer type. Only
            parameters that differ from BASE_TRAINER_DEFAULTS need to be specified.

    Example:
        >>> register_trainer_defaults("my_trainer", {
        ...     "trainer_prefix": "mytrainer",
        ...     "learning_rate": 1e-5,
        ...     "custom_param": 42,
        ... })
        >>> # Now "my_trainer" can be used as a trainer_type
        >>> config = normalize_trainer_config({"trainer_type": "my_trainer"})

    Note:
        Registering a trainer type that already exists will override the
        existing defaults entirely.
    """
    TRAINER_SPECIFIC_DEFAULTS[_normalize_trainer_type(trainer_type)] = defaults


def get_trainer_defaults(trainer_type: str) -> TrainerConfig:
    """Get merged defaults for a trainer type.

    This function retrieves the complete set of default configuration values
    for a specific trainer type by merging BASE_TRAINER_DEFAULTS with any
    trainer-specific overrides from TRAINER_SPECIFIC_DEFAULTS.

    Args:
        trainer_type: The trainer type identifier (e.g., "dpo", "sft", "grpo").
            Case-insensitive and aliases are automatically resolved.

    Returns:
        Complete defaults dictionary for the trainer with all base defaults
        and trainer-specific overrides applied.

    Example:
        >>> defaults = get_trainer_defaults("dpo")
        >>> defaults["beta"]
        0.1
        >>> defaults["learning_rate"]
        1e-06
        >>> defaults["optimizer"]  # From base defaults
        'adamw'
    """
    trainer_type = _normalize_trainer_type(trainer_type)
    defaults = dict(BASE_TRAINER_DEFAULTS)
    if trainer_type in TRAINER_SPECIFIC_DEFAULTS:
        defaults.update(TRAINER_SPECIFIC_DEFAULTS[trainer_type])
    return defaults  # type: ignore[return-value]


def normalize_trainer_config(config: dict[str, Any]) -> TrainerConfig:
    """Normalize and validate trainer configuration.

    This function takes raw trainer configuration and applies appropriate defaults
    based on the trainer type, ensuring all required fields are present with
    sensible values. It handles deprecated parameter names, auto-computes
    derived values, and converts nested configurations to appropriate types.

    The normalization process:
    1. Normalizes trainer_type to canonical form
    2. Handles deprecated max_sequence_length -> max_length migration
    3. Merges BASE_TRAINER_DEFAULTS with trainer-specific defaults
    4. Applies user configuration (user values take precedence)
    5. Auto-computes max_completion_length for applicable trainers
    6. Sets eval_batch_size to total_batch_size if not specified
    7. Converts loss_config dict to LossConfig instance

    Args:
        config: Raw trainer configuration dictionary. May contain partial
            configuration with missing values that will be filled from defaults.

    Returns:
        Normalized trainer configuration with proper type, all defaults applied,
        and derived values computed.

    Raises:
        FutureWarning: If deprecated parameter names are used.

    Example:
        >>> config = {"trainer_type": "dpo", "learning_rate": 2e-6}
        >>> normalized = normalize_trainer_config(config)
        >>> normalized["beta"]  # From DPO defaults
        0.1
        >>> normalized["optimizer"]  # From base defaults
        'adamw'
        >>> normalized["max_completion_length"]  # Auto-computed
        256

    Note:
        The input config is deep-copied to prevent mutation of the original.
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

    return cast(TrainerConfig, cast(object, config))


def get_trainer_class(trainer_type: str):
    """Get the appropriate trainer class based on type.

    This function maps trainer type strings to their corresponding trainer
    class implementations using the EasyDeL registry system. It provides
    a central point for trainer class resolution.

    Args:
        trainer_type: Type of trainer to retrieve. Supported values include:
            "sft", "dpo", "orpo", "grpo", "sdpo", "ppo", "reward", "distillation",
            "kto", "bco", "cpo", "gkd", "nash-md", "xpo", "base".
            Case-insensitive with alias support.

    Returns:
        Trainer class corresponding to the specified type. The class can be
        instantiated with appropriate configuration and model.

    Raises:
        KeyError: If the trainer type is not found in the registry.

    Example:
        >>> trainer_cls = get_trainer_class("dpo")
        >>> trainer_cls.__name__
        'DPOTrainer'
        >>> trainer = trainer_cls(model=model, args=args, ...)

    See Also:
        get_training_arguments_class: Get the corresponding config class.
        easydel.utils.Registry: The underlying registry system.
    """
    from easydel.utils import Registry

    return Registry.get_or_raise("trainer", _normalize_trainer_type(trainer_type))


def get_training_arguments_class(trainer_type: str):
    """Get the appropriate TrainingArguments class based on trainer type.

    This function maps trainer type strings to their corresponding configuration
    class implementations using the EasyDeL registry system. The returned class
    can be used to create type-safe configuration objects.

    Args:
        trainer_type: Type of trainer configuration to retrieve. Supported values
            include: "sft", "dpo", "orpo", "grpo", "sdpo", "ppo", "reward", "distillation",
            "kto", "bco", "cpo", "gkd", "nash-md", "xpo", "base".
            Case-insensitive with alias support.

    Returns:
        TrainingArguments class (or subclass) corresponding to the specified
        trainer type. Common classes include SFTConfig, DPOConfig, GRPOConfig, etc.

    Raises:
        KeyError: If the trainer type is not found in the registry.

    Example:
        >>> args_cls = get_training_arguments_class("sft")
        >>> args_cls.__name__
        'SFTConfig'
        >>> args = args_cls(learning_rate=2e-5, num_train_epochs=3)

    See Also:
        get_trainer_class: Get the corresponding trainer class.
        easydel.utils.Registry: The underlying registry system.
    """

    from easydel.utils import Registry

    return Registry.get_or_raise("trainer-arguments", _normalize_trainer_type(trainer_type))
