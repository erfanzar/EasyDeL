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

from transformers import AutoTokenizer

from easydel.inference.utils import vInferenceConfig
from easydel.inference.vinference.vinference import vInference
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.helpers import get_logger
from easydel.utils.traversals import deepcopy_model

from ..trainer.trainer import Trainer
from .grpo_config import GRPOConfig

if tp.TYPE_CHECKING:
	from datasets import Dataset
	from tensorflow import data
	from transformers import PreTrainedTokenizerBase

	TFDataset = data.Dataset

else:
	Dataset = tp.Any
	PreTrainedTokenizerBase = tp.Any
	TFDataset = tp.Any

logger = get_logger(__name__)
RewardFunc = tp.Union[
	EasyDeLBaseModule,
	EasyDeLState,
	tp.Callable[[list, list], list[float]],
]


def _create_naive_generation_function(model: EasyDeLBaseModule, generation_config): ...


class GRPOTrainer(Trainer):
	arguments: GRPOConfig

	def __init__(
		self,
		arguments: GRPOConfig,
		model: tp.Optional[tp.Union[EasyDeLBaseModule, EasyDeLState]],
		reward_funcs: tp.Union[RewardFunc, list[RewardFunc]],
		train_dataset: tp.Optional[Dataset] = None,
		eval_dataset: tp.Optional[tp.Union[Dataset, tp.Dict[str, Dataset]]] = None,
		processing_class: tp.Optional[ProcessingClassType] = None,
		reward_processing_classes: tp.Optional[ProcessingClassType] = None,
	):
		# fmt:off
		# caused by OCD
		assert arguments is not None, "You Have to pass arguments that will be used for training but you have passed `arguments=None`"
		assert isinstance(arguments, GRPOConfig), f"arguments type must be `GRPOConfig` but got {type(arguments)}"
		assert processing_class is not None, "processing_class must be specified to tokenize a DPO dataset."
		# fmt:on
		self.arguments = arguments
		self.truncation_mode = arguments.truncation_mode
		self.processing_class = processing_class
		self.is_encoder_decoder = arguments.is_encoder_decoder

		if not isinstance(model, EasyDeLState):
			model = model.to_state()
		self.ref_state = deepcopy_model(model=model)
		if processing_class is None:
			processing_class = AutoTokenizer.from_pretrained(
				model.model.config._name_or_path,
				padding_side="left",
			)
		if not isinstance(reward_funcs, list):
			reward_funcs = [reward_funcs]
		self.reward_funcs = reward_funcs
		if reward_processing_classes is None:
			reward_processing_classes = [None] * len(reward_funcs)
		elif not isinstance(reward_processing_classes, list):
			reward_processing_classes = [reward_processing_classes]
		else:
			if len(reward_processing_classes) != len(reward_funcs):
				raise ValueError(
					"The number of reward processing classes must match the number of reward functions."
				)

		for i, (reward_processing_class, reward_func) in enumerate(
			zip(reward_processing_classes, reward_funcs)
		):
			if isinstance(reward_func, (EasyDeLBaseModule, EasyDeLState)):
				if isinstance(reward_func, EasyDeLBaseModule):
					reward_func = reward_func.to_state()
				if reward_processing_class is None:
					reward_processing_class = AutoTokenizer.from_pretrained(
						reward_func.model.config._name_or_path
					)
				if reward_processing_class.pad_token_id is None:
					reward_processing_class.pad_token = reward_processing_class.eos_token

				reward_func.model.config.pad_token_id = reward_processing_class.pad_token_id
				reward_processing_classes[i] = reward_processing_class
				reward_funcs[i] = reward_func
		self.vinference = None
		if arguments.use_vinference:
			if arguments.vinference_config is None:
				arguments.vinference_config = vInferenceConfig(
					max_new_tokens=arguments.max_completion_length,
					do_sample=True,
					temperature=arguments.temperature,
					pad_token_id=processing_class.pad_token_id,
					eos_token_id=processing_class.eos_token_id,
					streaming_chunks=arguments.max_completion_length,  # skip streaming step in order to get maximum speed.
				)
			self.vinference = vInference(
				model=model.model,
				processing_class=processing_class,
				generation_config=arguments.vinference_config,
				input_partition_spec=arguments.vinference_input_partition_spec,
				seed=arguments.vinference_seed,
			)
		self.reward_processing_classes = reward_processing_classes
		self.arguments = arguments
		self.processing_class = processing_class
		super().__init__(
			model_state=model,
			arguments=arguments,
			dataset_train=train_dataset,
			dataset_eval=eval_dataset,
			data_collator=None,
		)
