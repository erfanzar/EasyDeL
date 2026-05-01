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


import math

from easydel.infra.base_module import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLGradientCheckPointers
from easydel.infra.factory import register_config


@register_config("mamba")
class MambaConfig(EasyDeLBaseConfig):
    """Configuration for the original Mamba selective state-space model (Mamba-1).

    Mamba replaces self-attention with a selective state-space recurrence

    .. math::
        h_t = \\bar{A}_t \\, h_{t-1} + \\bar{B}_t \\, x_t, \\qquad y_t = C_t \\, h_t + D \\, x_t

    where :math:`\\bar{A}, \\bar{B}` are obtained by zero-order-hold discretization of
    a continuous SSM with input-dependent step size :math:`\\Delta_t = \\text{softplus}
    (\\text{Linear}_{\\Delta}(x_t))`. ``B`` and ``C`` are projected from the input as
    well, which is what makes the SSM *selective* — it can let information through or
    drop it based on the token. A short causal depthwise 1-D convolution of width
    ``conv_kernel`` is applied before the SSM to give the model a learnable local
    receptive field; its rolling window is what is carried in ``conv_state`` during
    streaming decode.

    State carried per token across batch/seq:
        * ``conv_state``: rolling window of length ``conv_kernel`` over the
          ``intermediate_size`` channel axis.
        * ``ssm_state``: per-channel hidden state of shape
          ``(intermediate_size, state_size)`` accumulating the recurrence.

    Attributes:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the Mamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        state_size (`int`, *optional*, defaults to 16):
            State size of the Mamba model.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The index of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the *end-of-sequence* token.
        expand (`int`, *optional*, defaults to 2):
            Expansion factor for the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4):
            Kernel size of the convolution layer.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the linear layers.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the convolution layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) to use in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"swish"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to compute the residual connection in float32.
        time_step_rank (`str` or `int`, *optional*, defaults to `"auto"`):
            The rank of the time step embedding. If set to `"auto"`, the rank is calculated as
            `math.ceil(self.hidden_size / 16)`.
        time_step_scale (`float`, *optional*, defaults to 1.0):
            The scale factor for the time step embedding.
        time_step_min (`float`, *optional*, defaults to 0.001):
            The minimum value for the time step embedding.
        time_step_max (`float`, *optional*, defaults to 0.1):
            The maximum value for the time step embedding.
        time_step_init_scheme (`str`, *optional*, defaults to `"random"`):
            The initialization scheme for the time step embedding. Possible values are `"random"` and `"uniform"`.
        time_step_floor (`float`, *optional*, defaults to 1e-4):
            The floor value for the time step embedding.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether to rescale the pre-norm residual.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_associative_scan (`bool`, *optional*, defaults to `True`):
            Whether to use the associative-scan implementation when supported.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
    """

    model_type: str = "mamba"

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
        time_step_rank: str | int = "auto",
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_init_scheme: str = "random",
        time_step_floor: float = 1e-4,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        use_associative_scan: bool = True,
        tie_word_embeddings: bool = True,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        use_mambapy: bool = False,
        **kwargs,
    ):
        """Initialize the Mamba configuration.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 50280.
            hidden_size (int, optional): Hidden dimension. Defaults to 768.
            state_size (int, optional): SSM state dimension. Defaults to 16.
            num_hidden_layers (int, optional): Number of Mamba blocks. Defaults to 32.
            layer_norm_epsilon (float, optional): Epsilon for normalization layers.
                Defaults to 1e-5.
            pad_token_id (int, optional): Padding token id. Defaults to 0.
            bos_token_id (int, optional): Beginning-of-sequence token id. Defaults to 0.
            eos_token_id (int, optional): End-of-sequence token id. Defaults to 0.
            expand (int, optional): Intermediate-size expansion factor. Defaults to 2.
            conv_kernel (int, optional): 1D convolution kernel size. Defaults to 4.
            use_bias (bool, optional): Whether linear layers use bias. Defaults to False.
            use_conv_bias (bool, optional): Whether the conv layer uses bias.
                Defaults to True.
            hidden_act (str, optional): Activation function for the gated path.
                Defaults to "silu".
            initializer_range (float, optional): Initializer standard deviation.
                Defaults to 0.1.
            residual_in_fp32 (bool, optional): Whether to compute the residual
                connection in float32. Defaults to True.
            time_step_rank (str | int, optional): Rank of the time-step projection.
                Use ``"auto"`` to use ``ceil(hidden_size / 16)``. Defaults to "auto".
            time_step_scale (float, optional): Scale applied to the time-step
                projection. Defaults to 1.0.
            time_step_min (float, optional): Minimum bias value for the time step.
                Defaults to 0.001.
            time_step_max (float, optional): Maximum bias value for the time step.
                Defaults to 0.1.
            time_step_init_scheme (str, optional): Initialization scheme for the
                time-step bias (``"random"`` or ``"uniform"``). Defaults to "random".
            time_step_floor (float, optional): Floor applied to the time-step bias.
                Defaults to 1e-4.
            rescale_prenorm_residual (bool, optional): Whether to rescale the pre-norm
                residual. Defaults to False.
            use_cache (bool, optional): Whether to enable recurrent state caching.
                Defaults to True.
            use_associative_scan (bool, optional): Whether to use the associative-scan
                implementation when supported. Defaults to True.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to True.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient
                checkpointing policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            use_mambapy (bool, optional): Whether to use the ``mambapy`` Python
                reference path. Defaults to False.
            **kwargs: Additional keyword arguments forwarded to ``EasyDeLBaseConfig``.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.use_associative_scan = use_associative_scan
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mambapy = use_mambapy
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def layer_types(self) -> list[str]:
        """Layer-type schedule consumed by ``EasyDeLBaseModule`` for cache routing.

        Pure Mamba models advertise the recurrent ``"mamba"`` layer type for every
        block, which tells the cache machinery to allocate :class:`RecurrentCache`
        slots (rolling conv window + SSM state) instead of KV pages.

        Returns:
            list[str]: ``["mamba"] * num_hidden_layers``.
        """
        return ["mamba"] * self.num_hidden_layers

    def get_mask_details(self):
        """Mamba blocks consume no attention mask metadata.

        The selective scan is fully causal by construction (left-to-right
        recurrence); padding is handled by zeroing the channel inputs in
        :meth:`MambaMixer.forward`, not by an attention mask.

        Returns:
            None: Always — there is no mask schema to advertise.
        """
        return None
