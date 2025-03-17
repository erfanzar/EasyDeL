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
from dataclasses import field

from easydel.utils import traversals as etr
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@etr.auto_pytree
class GRPOConfig(TrainingArguments):
	"""
	Configuration class for the GRPOTrainer.
	"""

	model_name: str = field(
		default="GRPOTrainer",
		metadata={"help": "The name of the model."},
	)
	remove_unused_columns: tp.Optional[bool] = field(
		default=False,
		metadata={"help": "Whether to remove unused columns from the dataset."},
	)
	max_prompt_length: int = field(
		default=512,
		metadata={"help": "The maximum length of the prompt."},
	)
	max_completion_length: int = field(
		default=256,
		metadata={"help": "The maximum length of the completion."},
	)
	dataset_num_proc: tp.Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for dataset processing."},
	)
	learning_rate: float = field(
		default=1e-6,
		metadata={"help": "The learning rate."},
	)
	beta: float = field(
		default=0.04,
		metadata={"help": "The beta parameter for GRPO."},
	)
	sync_ref_model: bool = field(
		default=False,
		metadata={
			"help": "Whether to periodically sync the reference model with the policy model."
		},
	)
	ref_model_mixup_alpha: float = field(
		default=0.9,
		metadata={
			"help": "The alpha parameter for mixing the reference model with the policy model."
		},
	)
	ref_model_sync_steps: int = field(
		default=64,
		metadata={"help": "The number of steps between syncing the reference model."},
	)
	tools: tp.Optional[tp.List[tp.Union[dict, tp.Callable]]] = field(
		default=None,
		metadata={"help": "Additional tools for training."},
	)
	skip_apply_chat_template: bool = field(
		default=False,
		metadata={"help": "whenever to skip extracting prompt from dataset."},
	)

	def __post_init__(self):
		"""Post initialization to set dependent parameters."""
		self.max_sequence_length = self.max_prompt_length + self.max_completion_length

		if hasattr(super(), "__post_init__"):
			super().__post_init__()

	__hash__ = hash_fn
