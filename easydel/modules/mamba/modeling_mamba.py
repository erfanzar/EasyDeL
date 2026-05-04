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


import functools
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
from einops import repeat
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import RecurrentCache, RecurrentCacheConfig, RecurrentCacheView
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, Embed
from easydel.layers import RMSNorm as MambaRMSNorm
from easydel.modules._base import BaseCausalLMModule
from easydel.operations import OperationMetadata
from easydel.operations.kernels import SSM1Op

from .mamba_configuration import MambaConfig as MambaConfig


@auto_pytree
class MambaOutput(BaseModelOutput):
    """Output container for :class:`MambaModel`.

    Mirrors :class:`BaseModelOutput` but swaps the transformer KV cache for a
    :class:`RecurrentCache` carrying the per-layer SSM state and rolling conv
    window. ``last_hidden_state`` is the model trunk output after final RMSNorm
    of shape ``(batch, seq_len, hidden_size)``; ``hidden_states`` is the
    optional per-layer trace.
    """

    last_hidden_state: Array = None
    cache: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None


@auto_pytree
class MambaCausalLMOutput(BaseModelOutput):
    """Output container for :class:`MambaForCausalLM`.

    Adds the LM-head logits ``(batch, seq_len, vocab_size)`` on top of
    :class:`MambaOutput`. The ``cache`` field must be threaded back into the
    next call during streaming generation — without it the SSM recurrence
    restarts from zero state.
    """

    logits: Array = None
    cache: RecurrentCache | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None


class Lambda(spx.Module):
    """Convenience wrapper to insert callables into module pipelines.

    Wraps a Python callable as an ``spx.Module`` so it can participate in the
    Spectrax module tree (e.g. inside ``nn.ModuleList``).

    Attributes:
        fn (Callable): The callable invoked by ``forward``.
    """

    fn: tp.Callable

    def forward(self, x, **kwargs):
        """Invoke the wrapped callable.

        Args:
            x: Primary positional argument forwarded to ``fn``.
            **kwargs: Additional keyword arguments forwarded to ``fn``.

        Returns:
            The return value of ``fn(x, **kwargs)``.
        """
        return self.fn(x, **kwargs)


class MambaConv1D(spx.Module):
    """Causal depthwise 1-D convolution feeding the Mamba selective scan.

    Mamba prepends a small ``conv_kernel``-wide depthwise convolution
    (``feature_group_count = features``) on the channel-major layout
    ``(batch, features, seq_len)`` so that each token mixes a short local
    window before being fed into the otherwise *pointwise* recurrence. The
    convolution is left-padded with ``(conv_kernel - 1)`` zeros so it stays
    causal during prefill, and during decode the same effect is reproduced
    incrementally via the cached ``conv_state`` rolling buffer.

    Attributes:
        weight (ArrayParam): Kernel of shape ``(kernel_size, 1, features)``;
            transposed and broadcast over the depthwise group at apply-time.
        bias (ArrayParam): Optional per-channel bias of shape ``(features,)``.
        features (int): Number of channels (== ``intermediate_size`` in the mixer).
        kernel_size (int): Convolution kernel width / size of the rolling cache.
    """

    def __init__(
        self,
        features: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        num_spatial_dims: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mamba 1D convolution layer.

        Args:
            features (int): Number of output features (channels).
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding applied to input. Defaults to 0.
            dilation (int, optional): Dilation rate for the kernel. Defaults to 1.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            use_bias (bool, optional): Whether to include a bias term. Defaults to True.
            num_spatial_dims (int, optional): Number of spatial dimensions. Defaults to 1.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        kernel_shape = (kernel_size, 1, features)
        self.weight = ArrayParam.bound(
            shape=kernel_shape,
            dtype=param_dtype,
            init_method="lecun_normal",
            key=rngs.parameters,
        )

        if use_bias:
            self.bias = ArrayParam.bound(
                shape=(features,),
                dtype=param_dtype,
                init_method="zeros",
                key=rngs.parameters,
            )

        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.num_spatial_dims = num_spatial_dims
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

    def forward(self, x: Array) -> Array:
        """Apply 1D convolution to input tensor.

        Args:
            x (Array): Input tensor of shape (batch, features, sequence_length).

        Returns:
            Array: Convolved output tensor of shape (batch, features, output_length).

        Raises:
            ValueError: If input tensor rank does not match expected rank.
        """
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank}, but input has shape {x.shape}.",
            )
        org_x_dtype = x.dtype
        x = lax.conv_general_dilated(
            lhs=x.astype(self.dtype),
            rhs=jnp.asarray(jnp.swapaxes(self.weight.value, 0, 2), dtype=self.dtype),
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
        )

        if self.use_bias:
            x = x + jnp.asarray(self.bias.value.reshape(1, -1, 1), dtype=self.dtype)

        return x.astype(org_x_dtype)


class MambaMixer(spx.Module):
    """Selective state-space mixer ("S6") block — Mamba's attention replacement.

    Pipeline applied to ``(batch, seq_len, hidden_size)`` input :math:`x`:

    1. ``in_proj`` lifts ``hidden_size -> 2 * intermediate_size`` and splits
       into a *content* stream ``h`` and a *gate* stream ``g`` (the SiLU gate
       multiplied at the end).
    2. ``conv1d`` runs a causal depthwise convolution of width
       ``conv_kernel`` over ``h`` and applies the ``hidden_act`` activation.
       During decode the same window comes from the cached ``conv_state``.
    3. ``x_proj`` produces the *input-dependent* SSM parameters per token::

           [Δ_raw,  B,  C] = x_proj(h)               # rank/state-size split
           Δ = softplus(dt_proj(Δ_raw))              # positive step size
           Ā, B̄ = ZOH(A = -exp(A_log), Δ, B)         # discretization

       making the recurrence selective: ``A``, ``B``, ``C``, ``Δ`` are
       functions of ``x`` instead of fixed.
    4. The recurrence ``s_t = Ā_t s_{t-1} + B̄_t h_t``, ``y_t = C_t s_t + D h_t``
       is evaluated by :class:`SSM1Op` (parallel scan during prefill, single
       step during decode) which also returns the final ``ssm_state`` of shape
       ``(batch, intermediate_size, state_size)``.
    5. Result is gated by ``SiLU(g)`` and projected back with ``out_proj``
       down to ``hidden_size``.

    Attributes:
        in_proj, x_proj, dt_proj, out_proj: Linear projections.
        conv1d (MambaConv1D): Causal depthwise convolution feeding the SSM.
        A_log (ArrayParam): Log-parametrization of the negative diagonal SSM
            matrix; the actual ``A = -exp(A_log)`` is always negative-definite,
            guaranteeing stability of the recurrence.
        D (ArrayParam): Per-channel skip term added directly to the SSM output.
        ssm_state_size (int): SSM hidden dimension ``N`` per channel.
        intermediate_size (int): Channel dimension after ``in_proj``.
        conv_kernel_size (int): Width of the cached rolling conv window.
        time_step_rank (int): Rank of the low-rank ``Δ`` parameterization.
        ssm_op (SSM1Op): Fused kernel that runs the (selective) scan.
    """

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Mamba selective state space mixer.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = config.hidden_size
        ssm_state_size = config.state_size
        assert config.intermediate_size is not None, "intermediate_size must not be None"
        intermediate_size = config.intermediate_size
        time_step_rank = config.time_step_rank
        conv_kernel_size = config.conv_kernel

        self.conv1d = MambaConv1D(
            features=intermediate_size,
            use_bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=intermediate_size,
            padding=config.conv_kernel - 1,
            rngs=rngs,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        dt_init_std = time_step_rank**-0.5 * config.time_step_scale
        if config.time_step_init_scheme == "constant":
            init_kernel_dt = jax.nn.initializers.constant(dt_init_std, dtype=param_dtype)
        elif config.time_step_init_scheme == "random":

            def init_kernel_dt(key, shape, dtype):
                return (
                    jax.nn.initializers.uniform(scale=dt_init_std * 2, dtype=param_dtype)(key, shape, dtype)
                    - dt_init_std
                )

        else:
            init_kernel_dt = jax.nn.initializers.normal(config.initializer_range, param_dtype)

        def init_bias_dt(key, shape, dtype):
            dt = jax.lax.clamp(
                config.time_step_floor,
                jnp.exp(
                    jax.random.normal(
                        key=key,
                        shape=shape,
                        dtype=jnp.float32,
                    )
                    * (jnp.log(config.time_step_max) - jnp.log(config.time_step_min))
                    + jnp.log(config.time_step_min)
                ),
                config.time_step_max,
            )
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(dtype)

        linear_class = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.in_proj = linear_class(
            hidden_size,
            intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.x_proj = linear_class(
            intermediate_size,
            time_step_rank + ssm_state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.dt_proj = linear_class(
            time_step_rank,
            intermediate_size,
            use_bias=True,
            kernel_init=init_kernel_dt,
            bias_init=init_bias_dt,
            rngs=rngs,
        )
        self.out_proj = linear_class(
            intermediate_size,
            hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        A = repeat(jnp.arange(1, ssm_state_size + 1, dtype=jnp.float32), "n -> d n", d=intermediate_size)

        self.A_log = ArrayParam.bound(
            shape=A.shape,
            dtype=param_dtype,
            init_method="zeros",
            key=None,
            value=jnp.log(A).astype(param_dtype),
        )
        self.D = ArrayParam.bound(
            shape=(intermediate_size,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.time_step_rank = time_step_rank

        metadata = OperationMetadata(
            runtime_dtype=dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=config,
        )
        self.ssm_op = SSM1Op(metadata)

    def forward(
        self,
        input_states: Array,
        cache: RecurrentCacheView | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, RecurrentCacheView | None]:
        """Run the gated selective-scan over a chunk of tokens.

        Two execution paths share the same recurrence but differ in how
        the conv window is materialized:

        * **Prefill / training** (``seq_len > 1`` or no cache): the full
          ``conv1d`` is applied with left-padding of ``conv_kernel - 1`` and
          the trailing ``conv_kernel`` columns are saved into the cache for
          later decode steps. ``SSM1Op`` runs as a parallel associative scan
          across the whole sequence.
        * **Decode** (``seq_len == 1`` with cache): the cached ``conv_state``
          rolling buffer is updated with the new token, the convolution
          collapses to a single dot product, and ``SSM1Op`` performs one
          recurrent step per channel using the cached ``ssm_state``.

        Padding is handled in-stream by zeroing the channel inputs at masked
        positions (no attention mask needed by the recurrence itself).

        Args:
            input_states: ``(batch, seq_len, hidden_size)`` block input.
            cache: Per-layer recurrent cache view. ``None`` skips state
                threading entirely (training-only).
            position_ids: Unused by the recurrence; accepted for signature
                compatibility with attention layers.
            attention_mask: ``(batch, seq_len)`` boolean/0-1 mask. When
                provided, the gated channel stream is multiplied by the mask
                so padding tokens contribute neither to the conv window nor
                to the SSM state.

        Returns:
            tuple: ``(contextualized_states, cache_view)`` where
            ``contextualized_states`` has shape ``(batch, seq_len, hidden_size)``
            and ``cache_view`` carries the updated ``conv_state`` and
            ``ssm_state`` (or ``None`` if no cache was supplied).
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = checkpoint_name(self.in_proj(input_states), name="ssm_input_proj")
        projected_states = jnp.swapaxes(projected_states, 2, 1)
        hidden_states, gate = jnp.split(projected_states, 2, axis=1)

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        cache_view = cache
        if cache is not None and cache.recurrent_state is not None:
            ssm_state = cache.recurrent_state
        else:
            ssm_state = jnp.zeros((batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype)

        is_inference = seq_len == 1 and cache is not None

        if is_inference and cache is not None and cache.conv_state is not None:
            new_hidden = hidden_states[:, :, 0]
            conv_state, _, cache_view = cache.concatenate_to_cache(conv_state=new_hidden)

            kernel = jnp.swapaxes(self.conv1d.weight, 0, 2).astype(dtype)
            kernel = kernel[:, 0, :]
            hidden_states = jnp.sum(conv_state * kernel[None, :, :], axis=-1)
            if self.conv1d.use_bias:
                hidden_states = hidden_states + self.conv1d.bias.astype(dtype)[None, :]
            hidden_states = jnp.expand_dims(self.act(hidden_states).astype(dtype), -1)
        else:
            conv_input = hidden_states
            conv_out = self.conv1d(conv_input)[..., :seq_len]
            hidden_states = self.act(conv_out).astype(dtype)

            if cache is not None:
                if seq_len >= self.conv_kernel_size:
                    new_conv_state = conv_input[:, :, -self.conv_kernel_size :]
                else:
                    pad_width = self.conv_kernel_size - seq_len
                    new_conv_state = jnp.pad(conv_input, ((0, 0), (0, 0), (pad_width, 0)))

                cache_view = cache.replace(conv_state=new_conv_state) if cache_view is not None else cache_view

        if attention_mask is not None:
            hidden_states = hidden_states * jnp.expand_dims(attention_mask, 1)

        ssm_parameters = checkpoint_name(self.x_proj(jnp.swapaxes(hidden_states, 2, 1)), name="ssm_x_proj")
        time_step, B, C = jnp.split(
            ssm_parameters,
            [
                self.time_step_rank,
                self.ssm_state_size + self.time_step_rank,
            ],
            axis=-1,
        )

        discrete_time_step = checkpoint_name(self.dt_proj(time_step), name="ssm_dt_proj")
        discrete_time_step = jax.nn.softplus(discrete_time_step)

        hidden_states_t = jnp.swapaxes(hidden_states, 1, 2)
        gate_t = jnp.swapaxes(gate, 1, 2)

        hidden_states_t = apply_logical_sharding(
            hidden_states_t,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        ssm_output = self.ssm_op(
            hidden_states=hidden_states_t,
            A=self.A_log.value,
            B=B,
            C=C,
            D=self.D.value,
            discrete_time_step=discrete_time_step,
            gate=gate_t,
            ssm_state=ssm_state,
            activation=self.activation,
        )

        scan_output = jnp.swapaxes(ssm_output.attention_outputs, 1, 2)

        if cache_view is not None:
            cache_view = cache_view.replace(recurrent_state=ssm_output.ssm_state)

        scan_output_t = jnp.swapaxes(scan_output, 2, 1)
        scan_output_t = apply_logical_sharding(
            scan_output_t,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        contextualized_states = checkpoint_name(self.out_proj(scan_output_t), name="ssm_output_proj")
        return contextualized_states, cache_view


class MambaBlock(spx.Module):
    """Pre-norm wrapper around a :class:`MambaMixer`.

    A Mamba layer is just ``residual + mixer(RMSNorm(x))`` — there is no
    second feed-forward, the mixer's gated MLP-on-channels already plays
    that role. When ``config.residual_in_fp32`` is set, the residual stream
    is upcast to fp32 to bound numerical drift across many recurrent steps
    (Mamba accumulates information through the SSM rather than re-attending,
    so error in the residual compounds layer-to-layer).

    Attributes:
        norm (RMSNorm): Pre-mixer RMS normalization at ``layer_norm_epsilon``.
        mixer (MambaMixer): The selective-scan block, optionally rematerialized
            via :func:`auto_remat` according to the config's checkpointing policy.
        residual_in_fp32 (bool): Whether the residual is kept in float32.
    """

    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mamba block.

        Args:
            config (MambaConfig): Model configuration.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        block = auto_remat(
            MambaMixer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.mixer = block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: Array,
        cache: RecurrentCacheView | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, RecurrentCacheView | None]:
        """Apply the pre-norm Mamba block ``x = x + mixer(RMSNorm(x))``.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` block input.
            cache: Per-layer recurrent state (``conv_state`` + ``ssm_state``)
                threaded through during streaming decode; ``None`` for the
                training/non-cached path.
            position_ids: Forwarded to the mixer for signature parity; unused
                by the SSM recurrence itself.
            attention_mask: ``(batch, seq_len)`` mask used inside the mixer to
                zero out padding tokens before they enter the conv/SSM state.

        Returns:
            tuple: ``(hidden_states, cache_view)``. The output keeps the input
            shape; ``cache_view`` is the mixer's updated cache view (or ``None``).
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states, cache = self.mixer(
            hidden_states,
            cache,
            position_ids,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        return hidden_states, cache


@register_module(TaskType.BASE_MODULE, config=MambaConfig, model_type="mamba")
class MambaModel(EasyDeLBaseModule):
    """Mamba (S6) base trunk: token embedding + stack of :class:`MambaBlock` + final RMSNorm.

    There is no positional embedding and no attention — sequence ordering is
    encoded entirely by the causal SSM recurrence. Each block contributes a
    :class:`RecurrentCacheView` to the trunk's :class:`RecurrentCache` so that
    autoregressive decode runs in O(1) time per token by stepping the cached
    SSM/conv state instead of replaying the prefix.

    Attributes:
        embeddings (Embed): Token embedding ``(vocab_size, hidden_size)``.
        layers (nn.ModuleList[MambaBlock]): ``num_hidden_layers`` mixer blocks,
            assigned to pipeline stages via :func:`spx.assign_stage`.
        norm_f (RMSNorm): Final RMSNorm applied to the trunk output.
    """

    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize Mamba base model.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    MambaBlock(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        self.norm_f = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    def forward(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        cache: RecurrentCache | None = None,
        position_ids: Array | None = None,
        attention_mask: Array | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | MambaOutput:
        """Embed tokens and run them through every Mamba layer, threading the cache.

        The trunk is executed via :meth:`nn.ModuleList.scan` with
        ``trace=True`` so the layer body is JIT-traced once and replayed for
        each block, keeping compile time independent of ``num_hidden_layers``.
        At each step the per-layer :class:`RecurrentCacheView` is read from
        ``cache.views[idx]`` and the updated view is written back in-place;
        passing ``cache=None`` causes a fresh empty cache to be allocated, which
        is what training and one-shot prefill rely on.

        Args:
            input_ids: ``(batch, seq_len)`` int32 token ids. Mutually exclusive
                with ``inputs_embeds``.
            inputs_embeds: Pre-embedded ``(batch, seq_len, hidden_size)`` input
                used when token embeddings come from elsewhere.
            cache: Recurrent cache with one slot per layer; updated in place
                and returned on the output.
            position_ids: Forwarded to layers for signature compatibility but
                ignored by the recurrence. Reconstructed from the
                ``MaskInfo``'s ``q_position_ids`` when not provided.
            attention_mask: ``(batch, seq_len)`` 0/1 padding mask. Used by the
                mixer to gate out padding contributions to the SSM state.
            output_hidden_states: When true, return the per-layer trace
                including the post-final-norm output. Defaults to
                ``config.output_hidden_states``.
            **kwargs: Ignored (retained for cross-model call-site parity).

        Returns:
            MambaOutput: ``last_hidden_state`` of shape ``(batch, seq_len,
            hidden_size)``, the (possibly mutated) ``cache``, and an optional
            tuple of per-layer hidden states.

        Raises:
            ValueError: If exactly one of ``input_ids`` and ``inputs_embeds``
                is not provided.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        sequence_length = inputs_embeds.shape[1]
        if attention_mask is None:
            attention_mask = jnp.ones((inputs_embeds.shape[0], sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")

        mask_info = MaskInfo.dynamic_init(
            mask_info=None,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = mask_info.q_position_ids
        if cache is None:
            cache = RecurrentCache.init_empty(len(self.layers))

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        def _layer_loop(block, carry):
            hidden_states, all_hidden_states, idx = carry
            with self._layer_stage_context(idx, layers=self.layers):
                hidden_states, cache_view = block(
                    hidden_states=hidden_states,
                    cache=cache.views[idx],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            hidden_states = self._mark_layer_stage_boundary(hidden_states, idx, layers=self.layers)
            cache[idx] = cache_view
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = (*all_hidden_states, hidden_states)

            return hidden_states, all_hidden_states, idx + 1

        hidden_states, all_hidden_states, _ = self.layers.scan(
            _layer_loop,
            (hidden_states, all_hidden_states, 0),
            trace=True,
        )
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = (*all_hidden_states, hidden_states)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache=cache,
            hidden_states=all_hidden_states,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


@register_module(TaskType.CAUSAL_LM, config=MambaConfig, model_type="mamba")
class MambaForCausalLM(BaseCausalLMModule[MambaModel, MambaConfig]):  # type: ignore
    """Causal language model wrapper around :class:`MambaModel`.

    Adds an LM head on top of the SSM trunk. Because every layer is recurrent,
    autoregressive sampling is O(1) per token in sequence length once the cache
    is primed (cf. transformers, which still need to attend to a growing KV
    cache). The :class:`RecurrentCache` returned alongside the logits MUST be
    threaded into the next call — there is no recoverable state otherwise.

    Attributes:
        backbone (MambaModel): The SSM trunk; stored under the name
            ``"backbone"`` (not ``"model"``) to match the upstream Mamba
            checkpoint layout.
        lm_head: Tied or independent vocab projection produced by
            :class:`BaseCausalLMModule`.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mamba"
    _config_class = MambaConfig

    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | lax.Precision | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize Mamba model for causal language modeling.

        Args:
            config (MambaConfig): Model configuration with SSM parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=MambaModel,
            base_model_name="backbone",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )

    def forward(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        cache: RecurrentCache | None = None,
        position_ids: Array | None = None,
        apply_lm_head: bool = True,
        attention_mask: Array | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple | MambaCausalLMOutput:
        """Run the SSM trunk and (optionally) project to vocab logits.

        Args:
            input_ids: ``(batch, seq_len)`` int32 token ids. Mutually exclusive
                with ``inputs_embeds``.
            inputs_embeds: Pre-embedded ``(batch, seq_len, hidden_size)`` inputs.
            cache: Recurrent cache (per-layer ``conv_state`` + ``ssm_state``)
                threaded across decoding steps.
            position_ids: Accepted for signature parity; unused by the SSM.
            apply_lm_head: When ``False`` the LM projection is skipped and
                ``logits`` is left ``None`` — useful for distillation, value
                heads, or trace-only calls.
            attention_mask: ``(batch, seq_len)`` padding mask forwarded to the
                trunk.
            output_hidden_states: Optional per-layer hidden state trace.
            **kwargs: Ignored (consumed for cross-model call-site parity).

        Returns:
            MambaCausalLMOutput: ``logits`` of shape ``(batch, seq_len, vocab)``
            when ``apply_lm_head`` is true, the trunk's ``last_hidden_state``,
            the updated ``cache``, and the optional per-layer hidden states.
        """
        mamba_outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        logits = None
        if apply_lm_head:
            logits = self.compute_lm_logits(mamba_outputs.last_hidden_state)

        return MambaCausalLMOutput(
            logits=logits,
            cache=mamba_outputs.cache,
            hidden_states=mamba_outputs.hidden_states,
            last_hidden_state=mamba_outputs.last_hidden_state,
        )

    def update_inputs_for_generation(
        self,
        outputs: MambaOutput,
        model_kwargs: dict[str, tp.Any],
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Forward the updated recurrent cache into the next generation step.

        Unlike transformers, Mamba's ``cache`` is the only carrier of
        sequence context across decoding steps — token ids do *not* need to be
        re-prefilled. This hook simply lifts ``outputs.cache`` into
        ``model_kwargs`` so the generation loop's next ``forward`` keeps
        evolving the same SSM/conv state.

        Args:
            outputs: The :class:`MambaCausalLMOutput`/``MambaOutput`` returned
                by the previous step.
            model_kwargs: Mutable kwargs dict for the next call.
            **kwargs: Ignored.

        Returns:
            dict: ``model_kwargs`` with ``"cache"`` set to the latest state.
        """
        model_kwargs["cache"] = outputs.get("cache", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        **kwargs,
    ):
        """Allocate the per-layer recurrent cache for a new generation run.

        Builds a :class:`RecurrentCache` shaped for this model's
        ``intermediate_size``, ``state_size`` and ``conv_kernel`` if one was
        not supplied, then defers to :meth:`prepare_inputs_for_call` to wire
        up ``attention_mask`` / ``position_ids`` defaults. Unlike transformer
        models, ``max_length`` does not affect cache allocation: the cache
        size is constant per batch element regardless of generated length.

        Args:
            input_ids: Prompt tokens; only the batch dimension is read here.
            max_length: Generation budget; accepted for API parity but not
                used for cache shaping.
            pad_token_id: Padding id used to construct the default attention
                mask in the parent helper.
            starts: Optional starting offset, forwarded to
                :meth:`prepare_inputs_for_call`.
            **kwargs: May carry an existing ``cache``, ``attention_mask`` or
                ``position_ids`` to override the defaults.

        Returns:
            dict: Kwargs dict consumable by :meth:`forward`, containing the
            (possibly fresh) ``cache``, ``attention_mask`` and ``position_ids``.
        """
        from spectrax import PartitionAxis

        from easydel.caching import RecurrentCache

        cache = kwargs.get("cache", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if cache is None:
            partition_axis = getattr(self.config, "partition_axis", None) or PartitionAxis()
            cache_config = RecurrentCacheConfig.create_for_mamba(
                num_hidden_layers=self.config.num_hidden_layers,
                partition_axis=partition_axis,
                batch_size=int(input_ids.shape[0]),
                intermediate_size=int(self.config.intermediate_size),
                ssm_state_size=int(self.config.state_size),
                conv_kernel_size=int(self.config.conv_kernel),
            )
            cache = RecurrentCache.init_cache(cache_config, dtype=self.dtype)

        return self.prepare_inputs_for_call(
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.backbone.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.backbone.get_embedding()
