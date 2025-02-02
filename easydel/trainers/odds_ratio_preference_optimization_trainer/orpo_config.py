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
	"""
	Configuration class for ORPO training settings.

	This class inherits from TrainingArguments and holds configuration
	parameters specific to the ORPO model training. The dataclass automatically
	generates an initializer, and the __post_init__ method further processes
	some of the parameters after object initialization.

	Attributes:
	    learning_rate (float): The learning rate used during training.
	                          Default is 1e-6.
	    max_length (Optional[int]): The maximum allowed sequence length for the input.
	                                Default is 1024.
	    max_prompt_length (Optional[int]): The maximum allowed length of the prompt portion
	                                       of the input. Default is 512.
	    max_completion_length (Optional[int]): The maximum allowed length of the completion.
	                                           If not provided, it is set to max_length - max_prompt_length.
	    beta (float): A hyperparameter beta, with a default value of 0.1.
	    disable_dropout (bool): Flag to disable dropout during training.
	                            Default is True.
	    label_pad_token_id (int): The token id used for padding labels.
	                              Default is -100.
	    padding_value (Optional[int]): The value used for padding sequences.
	                                   Default is None.
	    generate_during_eval (bool): Flag indicating whether to generate sequences during evaluation.
	                                 Default is False.
	    is_encoder_decoder (Optional[bool]): Flag to indicate if the model is encoder-decoder.
	                                         Default is None.
	    model_init_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for model initialization.
	                                                  Default is None.
	    dataset_num_proc (Optional[int]): Number of processes to use for dataset processing.
	                                      Default is None.
	    max_sequence_length (int): Computed attribute representing the maximum sequence length
	                               used for training. It is set in the __post_init__ method.
	"""

	learning_rate: float = 1e-6
	max_length: tp.Optional[int] = 1024
	max_prompt_length: tp.Optional[int] = 512
	max_completion_length: tp.Optional[int] = None
	beta: float = 0.1
	disable_dropout: bool = True
	label_pad_token_id: int = -100
	padding_value: tp.Optional[int] = None
	generate_during_eval: bool = False
	is_encoder_decoder: tp.Optional[bool] = None
	model_init_kwargs: tp.Optional[dict[str, tp.Any]] = None
	dataset_num_proc: tp.Optional[int] = None

	def __post_init__(self):
		"""
		Post-initialization processing.

		This method is automatically called after the dataclass __init__ method.
		It sets the 'max_completion_length' if it is not provided by subtracting the
		'max_prompt_length' from 'max_length'. It also defines 'max_sequence_length'
		(here set to twice the max_length, based on a chosen/rejected policy).

		Returns:
		    The result of the superclass __post_init__ method.
		"""
		# If max_completion_length is not provided, derive it from max_length and max_prompt_length.
		if self.max_completion_length is None:
			self.max_completion_length = self.max_length - self.max_prompt_length

		# Set max_sequence_length based on a chosen policy.
		self.max_sequence_length = self.max_length * 2  # Chosen - Rejected
		return super().__post_init__()

	__hash__ = hash_fn
