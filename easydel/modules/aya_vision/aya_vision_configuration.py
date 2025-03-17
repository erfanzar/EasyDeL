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


from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.factory import register_config, registry
from easydel.utils.helpers import get_logger

from ..auto import AutoEasyDeLConfig

logger = get_logger(__name__)


@register_config("aya_vision")
class AyaVisionConfig(EasyDeLBaseConfig):
	r"""
	This is the configuration class to store the configuration of a [`AyaVisionForConditionalGeneration`]. It is used to instantiate an
	AyaVision model according to the specified arguments, defining the model architecture. Instantiating a configuration
	with the defaults will yield a similar configuration to that of AyaVision.
	e.g. [CohereForAI/aya-vision-8b](https://huggingface.co/CohereForAI/aya-vision-8b)

	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
	documentation from [`PretrainedConfig`] for more information.

	Args:
	    vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
	        The config object or dictionary of the vision backbone.
	    text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
	        The config object or dictionary of the text backbone.
	    vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
	        The feature selection strategy used to select the vision feature from the vision backbone.
	        Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
	        If `"full"`, the full vision features are used.
	    vision_feature_layer (`int`, *optional*, defaults to -1):
	        The index of the layer to select the vision feature.
	    downsample_factor (`int`, *optional*, defaults to 2):
	        The downsample factor to apply to the vision features.
	    adapter_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
	        The epsilon value used for layer normalization in the adapter.
	    image_token_index (`int`, *optional*, defaults to 255036):
	        The image token index to encode the image prompt.
	"""

	model_type = "aya_vision"
	sub_configs = {"text_config": AutoEasyDeLConfig, "vision_config": AutoEasyDeLConfig}

	def __init__(
		self,
		vision_config=None,
		text_config=None,
		vision_feature_select_strategy="full",
		vision_feature_layer=-1,
		downsample_factor=2,
		adapter_layer_norm_eps=1e-6,
		image_token_index=255036,
		**kwargs,
	):
		self.image_token_index = image_token_index
		self.downsample_factor = downsample_factor
		self.adapter_layer_norm_eps = adapter_layer_norm_eps
		if vision_feature_select_strategy not in ["default", "full"]:
			raise ValueError(
				"vision_feature_select_strategy should be one of 'default', 'full'."
				f"Got: {vision_feature_select_strategy}"
			)

		self.vision_feature_select_strategy = vision_feature_select_strategy
		self.vision_feature_layer = vision_feature_layer

		if isinstance(vision_config, dict):
			vision_config["model_type"] = (
				vision_config["model_type"]
				if "model_type" in vision_config
				else "clip_vision_model"
			)
			vision_config = registry.get_config(vision_config["model_type"])(**vision_config)
		elif vision_config is None:
			from ..siglip import SiglipVisionConfig

			vision_config = SiglipVisionConfig(
				hidden_size=1152,
				intermediate_size=4304,
				patch_size=14,
				image_size=384,
				num_hidden_layers=26,
				num_attention_heads=14,
				vision_use_head=False,
			)

		self.vision_config = vision_config

		if isinstance(text_config, dict):
			text_config["model_type"] = (
				text_config["model_type"] if "model_type" in text_config else "llama"
			)
			text_config = registry.get_config(text_config["model_type"])(**text_config)
		elif text_config is None:
			from ..cohere2 import Cohere2Config

			text_config = Cohere2Config()

		self.text_config = text_config

		super().__init__(**kwargs)

	def get_partition_rules(self, *args, **kwargs):
		tp = self.text_config.get_partition_rules(*args, **kwargs)
		vp = self.vision_config.get_partition_rules(*args, **kwargs)
		return tp + vp
