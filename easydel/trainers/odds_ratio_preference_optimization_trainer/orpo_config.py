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


@dataclass
class ORPOConfig(TrainingArguments):
	beta: float = 0.1
	disable_dropout: bool = True
	is_encoder_decoder: bool = False
	apply_chat_template: bool = True
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
		if self.max_completion_length is None:
			self.max_completion_length = self.max_length - self.max_prompt_length
		self.max_sequence_length = self.max_length * 2  # Chosen - Rejected
		return super().__post_init__()

	__hash__ = hash_fn
