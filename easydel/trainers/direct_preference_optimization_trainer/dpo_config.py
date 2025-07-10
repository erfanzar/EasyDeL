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

import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments

LOSS_FN_VARIENTS = tp.Literal[
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


@auto_pytree
class DPOConfig(TrainingArguments):
    """Configuration class for Direct Preference Optimization (DPO) training.

    Inherits from TrainingArguments and adds parameters specific to DPO training
    as described in https://arxiv.org/abs/2305.18290. This configuration controls
    various aspects of the DPO training process including loss computation,
    model architecture, and dataset processing.

    Attributes:
        beta (float): Temperature parameter (β) controlling deviation from reference model.
            Higher values make training focus more on preference matching. Default: 0.1
        label_smoothing (float): Smoothing factor for labels in loss calculation.
            Helps prevent overconfidence. 0.0 means no smoothing. Default: 0.0
        loss_type (LOSS_FN_VARIENTS): Type of contrastive loss function to use.
            Valid options: 'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust',
            'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down'.
            Default: 'sigmoid'
        use_weighting (bool): Whether to apply example weighting in loss calculation.
            Default: False
        label_pad_token_id (int): Token ID used for padding labels. Default: -100
        padding_value (int | None): Value used for padding sequences. If None,
            uses model's default padding token. Default: None
        max_length (int | None): Maximum total sequence length (prompt + completion).
            Default: 512
        max_prompt_length (int | None): Maximum length for prompt sequences.
            Default: 256
        max_completion_length (int | None): Maximum length for completion sequences.
            Auto-calculated as max_length - max_prompt_length if None. Default: None
        is_encoder_decoder (bool | None): Explicitly set if model is encoder-decoder.
            Auto-detected if None. Default: None
        disable_dropout (bool): Whether to disable dropout during training for
            deterministic behavior. Default: True
        precompute_ref_log_probs (bool): Whether to precompute reference model
            log probabilities before training. Default: False
        dataset_num_proc (int | None): Number of processes for dataset preprocessing.
            Default: None (sequential processing)
        reference_free (bool): Whether to use reference-free variant of DPO.
            Default: False
        force_use_ref_model (bool): Force use reference model even when reference_free=True.
            Default: False
        sync_ref_model (bool): Whether to periodically sync reference model with
            training model. Default: False
        learning_rate (float): Optimizer learning rate. Default: 1e-6
        ref_model_mixup_alpha (float): Alpha parameter for mixup between policy
            and reference models. Default: 0.9
        ref_model_sync_steps (int): Number of steps between reference model syncs.
            Default: 64
        rpo_alpha (float | None): Alpha parameter for Relative Preference Optimization.
            None disables RPO. Default: None
        tools (list[dict | Callable] | None): Additional tools for training process

    Example:
        >>> config = DPOConfig(
        ...   beta=0.2, loss_type="ipo", max_length=1024, learning_rate=5e-6
        ... )
    """

    trainer_prefix: str | None = field(
        default="dpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Temperature parameter (β) controlling deviation from reference model. Higher values make training"
            " focus more on preference matching."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Smoothing factor for labels in loss calculation. Helps prevent overconfidence. "
            "0.0 means no smoothing."
        },
    )
    loss_type: LOSS_FN_VARIENTS = field(
        default="sigmoid",
        metadata={
            "help": "Type of contrastive loss function to use. Valid options: 'sigmoid', 'hinge', 'ipo', 'exo_pair', "
            "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'apo_zero', 'apo_down'."
        },
    )
    use_weighting: bool = field(
        default=False,
        metadata={"help": "Whether to apply example weighting in loss calculation."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Token ID used for padding labels."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Value used for padding sequences. If None, uses model's default padding token."},
    )
    max_length: int | None = field(
        default=512,
        metadata={"help": "Maximum total sequence length (prompt + completion)."},
    )
    max_prompt_length: int | None = field(
        default=256,
        metadata={"help": "Maximum length for prompt sequences."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum length for completion sequences. Auto-calculated as max_length - max_prompt_length if None."
        },
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Explicitly set if model is encoder-decoder. Auto-detected if None."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout during training for deterministic behavior."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute reference model log probabilities before training."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing. Default: None (sequential processing)"},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "Whether to use reference-free variant of DPO."},
    )
    force_use_ref_model: bool = field(
        default=False,
        metadata={"help": "Force use reference model even when reference_free=True."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync reference model with training model."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Optimizer learning rate."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "Alpha parameter for mixup between policy and reference models."},
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={"help": "Number of steps between reference model syncs."},
    )
    rpo_alpha: float | None = field(
        default=None,
        metadata={"help": "Alpha parameter for Relative Preference Optimization. None disables RPO."},
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for training process."},
    )

    def __post_init__(self):
        """Post-initialization processing to derive dependent parameters."""
        if self.max_completion_length is None:
            self.max_completion_length = self.max_length - self.max_prompt_length
        # chosen + rejected sequences
        self.max_sequence_length = self.max_length * 2
        # Call the post_init of the parent class if it exists. Important for inheritance
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    __hash__ = hash_fn
