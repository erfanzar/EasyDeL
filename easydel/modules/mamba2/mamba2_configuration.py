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


@register_config("mamba2")
class Mamba2Config(EasyDeLBaseConfig):
    """Configuration for Mamba-2 (the SSD / state-space dual formulation).

    Mamba-2 generalizes Mamba's selective recurrence to a *grouped* form
    (``num_heads`` heads × ``head_dim`` channels per head, sharing ``B``/``C``
    across ``n_groups`` head-groups) that is mathematically equivalent to a
    structured masked attention over chunks. The recurrence is

    .. math::
        h_t = a_t \\, h_{t-1} + B_t \\, x_t, \\quad y_t = C_t \\, h_t + D \\, x_t

    with :math:`a_t = \\exp(-\\Delta_t \\, \\text{softplus}(A))` a *scalar*
    decay per head — diagonal-of-scalar SSM — making the closed-form chunked
    matmul (the "state-space dual", SSD) cheap to run in O(L · chunk_size +
    L · state_size) instead of the full O(L · state_size · intermediate_size)
    of Mamba-1. ``chunk_size`` controls the chunked-matmul block size on
    hardware; ``conv_kernel`` is still a short causal depthwise conv giving
    each token a local window before the SSM.

    State carried per token across batch/seq:
        * ``conv_state``: rolling depthwise-conv window of length ``conv_kernel``.
        * ``ssm_state``: per-head state of shape ``(num_heads, head_dim, state_size)``.

    Attributes:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Mamba2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        state_size (`int`, *optional*, defaults to 128):
            State size of the Mamba2 model.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the Mamba2 encoder.
        num_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for the grouped selective scan.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each attention head.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups for the grouped selective scan.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            The index of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*, defaults to 2):
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
        time_step_min (`float`, *optional*, defaults to 0.001):
            The minimum value for the time step embedding.
        time_step_max (`float`, *optional*, defaults to 0.1):
            The maximum value for the time step embedding.
        time_step_floor (`float`, *optional*, defaults to 1e-4):
            The floor value for the time step embedding.
        time_step_limit (`tuple`, *optional*, defaults to (0.0, float("inf"))):
            The minimum and maximum limits for the time step.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether to rescale the pre-norm residual.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        norm_before_gate (`bool`, *optional*, defaults to `True`):
            Whether to apply normalization before the gate activation.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use root mean square normalization.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of chunks for processing long sequences.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embedding weights with the output projection weights.
        gradient_checkpointing (`str`, *optional*, defaults to `"nothing_saveable"`):
            The gradient checkpointing configuration.
    """

    model_type: str = "mamba2"

    def __init__(
        self,
        num_heads: int = 128,
        head_dim: int = 64,
        vocab_size: int = 32768,
        hidden_size: int = 4096,
        state_size: int = 128,
        num_hidden_layers: int = 64,
        layer_norm_epsilon: float = 1e-5,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 8,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.1,
        residual_in_fp32: bool = True,
        time_step_rank: str | int = "auto",
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit: tuple[float, float] | None = None,
        rescale_prenorm_residual: bool = False,
        use_cache: bool = True,
        norm_before_gate: bool = True,
        rms_norm: bool = True,
        chunk_size: int = 256,
        tie_word_embeddings: bool = False,
        gradient_checkpointing: EasyDeLGradientCheckPointers = EasyDeLGradientCheckPointers.NONE,
        **kwargs,
    ):
        """Initialize the Mamba2 configuration.

        Args:
            num_heads (int, optional): Number of selective-scan heads. Defaults to 128.
            head_dim (int, optional): Dimensionality per head. Defaults to 64.
            vocab_size (int, optional): Vocabulary size. Defaults to 32768.
            hidden_size (int, optional): Hidden dimension. Defaults to 4096.
            state_size (int, optional): SSM state dimension. Defaults to 128.
            num_hidden_layers (int, optional): Number of Mamba2 blocks. Defaults to 64.
            layer_norm_epsilon (float, optional): Epsilon for normalization layers.
                Defaults to 1e-5.
            pad_token_id (int, optional): Padding token id. Defaults to 1.
            bos_token_id (int, optional): Beginning-of-sequence token id. Defaults to 0.
            eos_token_id (int, optional): End-of-sequence token id. Defaults to 2.
            expand (int, optional): Intermediate-size expansion factor. Defaults to 2.
            conv_kernel (int, optional): 1D convolution kernel size. Defaults to 4.
            n_groups (int, optional): Number of groups in the grouped selective scan.
                Defaults to 8.
            use_bias (bool, optional): Whether linear layers use bias. Defaults to False.
            use_conv_bias (bool, optional): Whether the conv layer uses bias.
                Defaults to True.
            hidden_act (str, optional): Activation function. Defaults to "silu".
            initializer_range (float, optional): Initializer standard deviation.
                Defaults to 0.1.
            residual_in_fp32 (bool, optional): Whether to compute residuals in float32.
                Defaults to True.
            time_step_rank (str | int, optional): Rank of the time-step projection
                (``"auto"`` for ``ceil(hidden_size / 16)``). Defaults to "auto".
            time_step_min (float, optional): Minimum time-step bias. Defaults to 0.001.
            time_step_max (float, optional): Maximum time-step bias. Defaults to 0.1.
            time_step_floor (float, optional): Floor applied to the time-step bias.
                Defaults to 1e-4.
            time_step_limit (tuple[float, float] | None, optional): Inference clip
                limits ``(min, max)`` for the time step. Defaults to ``(0.0, inf)``.
            rescale_prenorm_residual (bool, optional): Rescale the pre-norm residual.
                Defaults to False.
            use_cache (bool, optional): Whether to enable recurrent state caching.
                Defaults to True.
            norm_before_gate (bool, optional): Apply RMSNorm before the gate activation.
                Defaults to True.
            rms_norm (bool, optional): Use RMSNorm rather than LayerNorm. Defaults to True.
            chunk_size (int, optional): Chunk size for chunked sequence processing.
                Defaults to 256.
            tie_word_embeddings (bool, optional): Tie input/output embeddings.
                Defaults to False.
            gradient_checkpointing (EasyDeLGradientCheckPointers, optional): Gradient
                checkpointing policy. Defaults to ``EasyDeLGradientCheckPointers.NONE``.
            **kwargs: Additional keyword arguments forwarded to ``EasyDeLBaseConfig``.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_before_gate = norm_before_gate
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit if time_step_limit is not None else (0.0, float("inf"))
        self.tie_word_embeddings = tie_word_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        self.intermediate_size = int(expand * hidden_size)
        super().__init__(**kwargs)
