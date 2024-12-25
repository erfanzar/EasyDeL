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
import warnings
from dataclasses import dataclass

from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@dataclass
class ORPOConfig(TrainingArguments):
	max_length: tp.Optional[int] = None
	max_prompt_length: tp.Optional[int] = None
	max_completion_length: tp.Optional[int] = None
	beta: float = 0.1
	disable_dropout: bool = True
	label_pad_token_id: int = -100
	is_encoder_decoder: bool = False
	padding_value: int = None
	apply_chat_template: bool = False

	def __post_init__(self):
		if self.max_length is None:
			warnings.warn(
				"`max_length` is not set in the ORPOConfig init"
				" it will default to `512` by default, but you should do it yourself in the future.",
				UserWarning,
				stacklevel=1,
			)
			self.max_length = 512
		if self.max_prompt_length is None:
			warnings.warn(
				"`max_prompt_length` is not set in the ORPOConfig init"
				" it will default to `128` by default, but you should do it yourself in the future.",
				UserWarning,
				stacklevel=1,
			)
			self.max_prompt_length = 128

		if self.max_completion_length is None:
			warnings.warn(
				"When using an encoder decoder architecture, you should set `max_completion_length` in the "
				"ORPOTrainer's init it will default to `128` by default, but you should do it yourself in the future.",
				UserWarning,
				stacklevel=1,
			)
			self.max_completion_length = 128
		return super().__post_init__()

	__hash__ = hash_fn
