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

from ..training_configurations import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
	r"""
	Configuration class for the GRPOTrainer.
	"""

	remove_unused_columns: tp.Optional[bool] = False
	max_prompt_length: int = 512
	max_completion_length: int = 256
	dataset_num_proc: tp.Optional[int] = None
	learning_rate: float = 1e-6
	beta: float = 0.04
	sync_ref_model: bool = False
	ref_model_mixup_alpha: float = 0.9
	ref_model_sync_steps: int = 64
	tools: tp.Optional[list] = None

	def __post_init__(self):
		self.max_sequence_length = self.max_prompt_length + self.max_completion_length
		return super().__post_init__()
