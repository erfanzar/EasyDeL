# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
        use_associative_scan: bool = True,
        mixer_rms_eps: float = 1e-6,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ):
        """Initialize a :class:`FalconMambaConfig`.

        Args:
            vocab_size (int, optional): Token vocabulary size. Defaults to ``50280``.
            hidden_size (int, optional): Decoder hidden width. Defaults to ``768``.
            state_size (int, optional): Mamba SSM state dimension. Defaults to ``16``.
            num_hidden_layers (int, optional): Number of Mamba decoder layers.
                Defaults to ``32``.
            layer_norm_epsilon (float, optional): LayerNorm epsilon. Defaults to ``1e-5``.
            pad_token_id (int, optional): Padding token id. Defaults to ``0``.
            bos_token_id (int, optional): Beginning-of-sequence id. Defaults to ``0``.
            eos_token_id (int, optional): End-of-sequence id. Defaults to ``0``.
            expand (int, optional): Mamba expansion factor over ``hidden_size``.
                Defaults to ``2``.
            conv_kernel (int, optional): Causal convolution kernel width inside the
                mixer. Defaults to ``4``.
            use_bias (bool, optional): Bias on Mamba linear projections. Defaults to ``False``.
            use_conv_bias (bool, optional): Bias on the causal convolution.
                Defaults to ``True``.
            hidden_act (str, optional): Mixer activation. Defaults to ``"silu"``.
            initializer_range (float, optional): Weight init stddev. Defaults to ``0.1``.
            residual_in_fp32 (bool, optional): Keep residual additions in fp32.
                Defaults to ``True``.
            time_step_rank (int | str, optional): Rank of the dt projection. ``"auto"``
                uses ``ceil(hidden_size / 16)``. Defaults to ``"auto"``.
            time_step_scale (float, optional): Scale factor applied to dt.
                Defaults to ``1.0``.
            time_step_min (float, optional): Minimum dt before clamping. Defaults to ``0.001``.
            time_step_max (float, optional): Maximum dt before clamping. Defaults to ``0.1``.
            time_step_init_scheme (str, optional): Initialization scheme for the dt
                projection (``"random"`` or ``"constant"``). Defaults to ``"random"``.
            time_step_floor (float, optional): Numerical floor on the discretized dt.
                Defaults to ``1e-4``.
            rescale_prenorm_residual (bool, optional): Rescale the pre-norm residual
                using the layer-depth-aware factor. Defaults to ``False``.
            use_cache (bool, optional): Return SSM caches during forward.
                Defaults to ``True``.
            use_falcon_mambapy (bool, optional): HF parity flag (no effect in
                EasyDeL — JAX kernels are always used). Defaults to ``False``.
            use_associative_scan (bool, optional): Use associative-scan kernel for
                the SSM (vs. sequential scan). Defaults to ``True``.
            mixer_rms_eps (float, optional): Epsilon for the mixer's RMSNorm.
                Defaults to ``1e-6``.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional):
                Checkpointing strategy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            **kwargs: Forwarded to :class:`EasyDeLBaseConfig`. Recognizes an optional
                ``intermediate_size`` override; the default is ``expand * hidden_size``.
        """
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
        self.use_associative_scan = use_associative_scan
        self.mixer_rms_eps = mixer_rms_eps
        self.gradient_checkpointing = gradient_checkpointing

        self.intermediate_size = (
            int(expand * self.hidden_size)
            if kwargs.get("intermediate_size") is None
            else kwargs.get("intermediate_size")
        )
        super().__init__(**kwargs)

    @property
    def layer_types(self) -> list[str]:
        """Expose the recurrent-only layer layout for HF parity and cache helpers.

        Returns:
            List of ``"mamba"`` strings with length ``num_hidden_layers``.
        """
        return ["mamba"] * self.num_hidden_layers

    def get_mask_details(self):
        """Recurrent Mamba layers do not use attention-mask descriptors.

        Returns:
            Always ``None``, since Falcon Mamba layers have no attention masks.
        """
        return None
