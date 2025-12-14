# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""FalconMamba configuration for EasyDeL.

This module provides `FalconMambaConfig`, an EasyDeL-native configuration class
mirroring HuggingFace's `FalconMambaConfig`, while integrating EasyDeL sharding
metadata and generation/runtime knobs.

HuggingFace reference:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon_mamba/configuration_falcon_mamba.py
"""

import math

from eformer.common_types import ColumnWise, Replicated, RowWise

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("falcon_mamba")
class FalconMambaConfig(EasyDeLBaseConfig):
    """Configuration for the FalconMamba architecture.

    FalconMamba is a decoder-only language model that replaces attention with a
    Mamba-style selective state space mixer (SSM).

    Notes:
        - `intermediate_size` defaults to `expand * hidden_size` like HF.
        - `use_falcon_mambapy` exists for HF parity; EasyDeL always uses the JAX implementation.
    """

    model_type: str = "falcon_mamba"

    def __init__(
        self,
        vocab_size: int = 50280,
        hidden_size: int = 768,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        layer_norm_epsilon: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        expand: int = 2,
        conv_kernel: int = 4,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.1,
        residual_in_fp32: bool = True,
        time_step_rank: int | str = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        use_falcon_mambapy: bool = False,
        mixer_rms_eps: float = 1e-6,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.expand = expand
        self.conv_kernel = conv_kernel
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.residual_in_fp32 = residual_in_fp32

        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual

        self.use_cache = use_cache
        self.use_falcon_mambapy = use_falcon_mambapy
        self.mixer_rms_eps = mixer_rms_eps
        self.gradient_checkpointing = gradient_checkpointing

        self.intermediate_size = (
            int(expand * self.hidden_size)
            if kwargs.get("intermediate_size") is None
            else kwargs.get("intermediate_size")
        )
        super().__init__(**kwargs)

    def get_partition_rules(self, *args, **kwargs):
        """Return regex-based parameter partition rules.

        These follow the standard EasyDeL linear sharding convention:
        - Column-wise sharding for "expanding" projections (output dimension sharded).
        - Row-wise sharding for "contracting" projections (input dimension sharded).
        - Biases and norms replicated.
        """
        pmag = self.partition_manager
        return (
            (r"embeddings/embedding", pmag.resolve(ColumnWise)),
            (r"mixer/(in_proj|x_proj|dt_proj)/kernel", pmag.resolve(ColumnWise)),
            (r"mixer/out_proj/kernel", pmag.resolve(RowWise)),
            (r"mixer/.*proj/bias", pmag.resolve(Replicated)),
            (r"mixer/(A_log|D)", pmag.resolve(Replicated)),
            (r"mixer/conv1d/(kernel|bias)", pmag.resolve(Replicated)),
            (r"(norm|norm_f)/kernel", pmag.resolve(Replicated)),
            (r"lm_head/kernel", pmag.resolve(ColumnWise)),
            (r".*bias", pmag.resolve(Replicated)),
            (r".*", pmag.resolve(Replicated)),
        )
