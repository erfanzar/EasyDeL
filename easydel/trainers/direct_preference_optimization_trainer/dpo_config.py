import typing as tp
import warnings
from dataclasses import dataclass

from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@dataclass
class DPOConfig(TrainingArguments):
	beta: float = 0.1
	label_smoothing: float = 0.0
	loss_type: tp.Literal[
		"ipo",
		"kto",
		"hinge",
		"sigmoid",
		"robust",
		"exo_pair",
		"bco_pair",
		"sppo_hard",
		"nca_pair",
		"aot_pair",
		"aot",
		"apo_zero",
		"apo_down",
	] = "sigmoid"
	use_weighting: bool = False
	label_pad_token_id: int = -100
	padding_value: tp.Optional[int] = None
	truncation_mode: str = "keep_end"
	max_length: tp.Optional[int] = None
	max_prompt_length: tp.Optional[int] = None
	max_completion_length: tp.Optional[int] = None
	is_encoder_decoder: tp.Optional[bool] = None
	disable_dropout: bool = True
	generate_during_eval: bool = False
	precompute_ref_log_probs: bool = False
	dataset_num_proc: tp.Optional[int] = None
	model_adapter_name: tp.Optional[str] = None
	ref_adapter_name: tp.Optional[str] = None
	reference_free: bool = False
	force_use_ref_model: bool = False
	sync_ref_model: bool = False
	ref_model_mixup_alpha: float = 0.9
	ref_model_sync_steps: int = 64
	rpo_alpha: tp.Optional[float] = None

	def __post_init__(self):
		if self.max_length is None:
			warnings.warn(
				"`max_length` is not set in the DPOConfig init"
				" it will default to `512` by default, but you should do it yourself in the future.",
				UserWarning,
				stacklevel=1,
			)
			self.max_length = 512
		if self.max_prompt_length is None:
			warnings.warn(
				"`max_prompt_length` is not set in the DPOConfig init"
				" it will default to `128` by default, but you should do it yourself in the future.",
				UserWarning,
				stacklevel=1,
			)
			self.max_prompt_length = 128
		return super().__post_init__()

	__hash__ = hash_fn
