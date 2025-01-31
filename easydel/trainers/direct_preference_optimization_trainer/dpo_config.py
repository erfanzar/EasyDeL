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
	generate_during_eval: bool = False
	precompute_ref_log_probs: bool = False
	dataset_num_proc: tp.Optional[int] = None
	model_adapter_name: tp.Optional[str] = None
	ref_adapter_name: tp.Optional[str] = None
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
