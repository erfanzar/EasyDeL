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
class RewardConfig(TrainingArguments):
	r"""
	Configuration class for the [`RewardTrainer`].

	Parameters:
	    max_length (`int` or `None`, *optional*, defaults to `1024`):
	        Maximum length of the sequences (prompt + completion) in the batch, filters out entries that exceed the
	        limit. This argument is required if you want to use the default data collator.
	    disable_dropout (`bool`, *optional*, defaults to `True`):
	        Whether to disable dropout in the model.
	    dataset_num_proc (`int`, *optional*, defaults to `None`):
	        Number of processes to use for processing the dataset.
	    center_rewards_coefficient (`float`, *optional*, defaults to `0.1`):
	        Coefficient to incentivize the reward model to output mean-zero rewards.
	    remove_unused_columns (`bool`, *optional*, defaults to `False`):
	        Whether to remove the columns that are not used by the model's forward pass. Can be `True` only if
	        the dataset is pretokenized.
	"""

	model_name: str = "EasyDeL-RewardTrainer-Model"
	max_length: tp.Optional[int] = 1024
	disable_dropout: bool = True
	dataset_num_proc: tp.Optional[int] = None
	center_rewards_coefficient: tp.Optional[float] = 0.1
	remove_unused_columns: bool = False
	__hash__ = hash_fn
