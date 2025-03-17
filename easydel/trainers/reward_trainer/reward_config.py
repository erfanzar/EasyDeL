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
class RewardConfig(TrainingArguments):
	r"""
	Configuration class for the [`RewardTrainer`].

	Parameters:
	    model_name (str): The name of the model. Defaults to "RewardTrainer".
	    max_length (int, optional): Maximum length of the sequences (prompt + completion) in the batch,
	        filters out entries that exceed the limit.  Defaults to 1024.
	    disable_dropout (bool, optional): Whether to disable dropout in the model. Defaults to True.
	    dataset_num_proc (int, optional): Number of processes to use for processing the dataset. Defaults to None.
	    center_rewards_coefficient (float, optional): Coefficient to incentivize the reward model to output
	        mean-zero rewards. Defaults to 0.1.
	    remove_unused_columns (bool, optional): Whether to remove the columns that are not used by the model's
	        forward pass. Can be `True` only if the dataset is pretokenized. Defaults to False.
	"""

	model_name: str = field(
		default="RewardTrainer",
		metadata={"help": "The name of the model."},
	)
	max_sequence_length: tp.Optional[int] = field(
		default=1024,
		metadata={
			"help": "Maximum length of the sequences (prompt + completion) in the batch, filters out entries that exceed the limit."
		},
	)
	disable_dropout: bool = field(
		default=True,
		metadata={"help": "Whether to disable dropout in the model."},
	)
	dataset_num_proc: tp.Optional[int] = field(
		default=None,
		metadata={"help": "Number of processes to use for processing the dataset."},
	)
	center_rewards_coefficient: tp.Optional[float] = field(
		default=0.1,
		metadata={
			"help": "Coefficient to incentivize the reward model to output mean-zero rewards."
		},
	)
	remove_unused_columns: bool = field(
		default=False,
		metadata={
			"help": "Whether to remove the columns that are not used by the model's forward pass. Can be `True` only if the dataset is pretokenized."
		},
	)

	__hash__ = hash_fn
