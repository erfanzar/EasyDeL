# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from dataclasses import dataclass

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


@dataclass
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
	        (e.g., custom metrics, logging hooks). Default: None

	Example:
	    >>> config = DPOConfig(
	    ...   beta=0.2, loss_type="ipo", max_length=1024, learning_rate=5e-6
	    ... )
	"""

	model_name: str = "EasyDeL-DPOTrainer-Model"
	beta: float = 0.1
	label_smoothing: float = 0.0
	loss_type: LOSS_FN_VARIENTS = "sigmoid"
	use_weighting: bool = False
	label_pad_token_id: int = -100
	padding_value: tp.Optional[int] = None
	max_length: tp.Optional[int] = 512
	max_prompt_length: tp.Optional[int] = 256
	max_completion_length: tp.Optional[int] = None
	is_encoder_decoder: tp.Optional[bool] = None
	disable_dropout: bool = True
	precompute_ref_log_probs: bool = False
	dataset_num_proc: tp.Optional[int] = None
	reference_free: bool = False
	force_use_ref_model: bool = False
	sync_ref_model: bool = False
	learning_rate: float = 1e-6
	ref_model_mixup_alpha: float = 0.9
	ref_model_sync_steps: int = 64
	rpo_alpha: tp.Optional[float] = None
	tools: tp.Optional[tp.List[tp.Union[dict, tp.Callable]]] = None

	def __post_init__(self):
		"""Post-initialization processing to derive dependent parameters."""
		if self.max_completion_length is None:
			self.max_completion_length = self.max_length - self.max_prompt_length
		self.max_sequence_length = self.max_length * 2  # Chosen + Rejected sequences
		return super().__post_init__()

	__hash__ = hash_fn
