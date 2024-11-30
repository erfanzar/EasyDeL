import warnings
from dataclasses import dataclass
from typing import Optional

from easydel.trainers.training_configurations import TrainingArguments
from easydel.utils.compiling_utils import hash_fn


@dataclass
class ORPOConfig(TrainingArguments):
	max_length: Optional[int] = None
	max_prompt_length: Optional[int] = None
	max_completion_length: Optional[int] = None
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
