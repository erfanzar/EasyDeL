import functools
import itertools
import math
from typing import Optional, Tuple, Union, List, Dict, Any, Callable, Sequence, TypeVar

import fjformer
from jax.core import ShapedArray
import chex
from fjformer import linen as nn
import jax
import jax.numpy as jnp
from fjformer.linen import Linear
import numpy as np
from chex import PRNGKey, Shape, Array
from einops import einsum
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
    default_kernel_init,
    ConvGeneralDilatedT,
    PrecisionLike,
    Dtype,
    PaddingLike,
    canonicalize_padding,
    _conv_dimension_numbers
)
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax, eval_shape
import flax.struct
from transformers.modeling_flax_outputs import FlaxBaseModelOutput

from .mamba_configuration import MambaConfig
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from ..flax_modelling_utils import (
    get_gradient_checkpoint_policy,
    get_dot_general_by_bits,
    ACT2FN
)


def init_to_value(x, dtype):
    return lambda _: x.astype(dtype)


@flax.struct.dataclass
class MambaOutput(FlaxBaseModelOutput):
    last_hidden_state: chex.Array = None
    cache_params: Optional[List[chex.Array]] = None
    hidden_states: Optional[Tuple[chex.Array]] = None


@flax.struct.dataclass
class MambaCausalLMOutput(FlaxBaseModelOutput):
    logits: chex.Array = None
    cache_params: Optional[List[chex.Array]] = None
    hidden_states: Optional[Tuple[chex.Array]] = None


class FlaxMambaCache:
    def __init__(
            self,
            config: MambaConfig,
            batch_size: int,
            dtype=jnp.float16,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: jnp.zeros((batch_size, intermediate_size, conv_kernel_size), dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: jnp.zeros((batch_size, intermediate_size, ssm_state_size), dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


_T = TypeVar("_T")


def create_tuple_parser(n: int) -> Callable[[Union[_T, Sequence[_T]]], tuple[_T, ...]]:
    def parse(x: Union[_T, Sequence[_T]]) -> tuple[_T, ...]:
        if isinstance(x, Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"x!=n ({x}!=({n}))"
                )
        else:
            return tuple(itertools.repeat(x, n))

    return parse


class Conv1D(nn.Module):
    features: int
    kernel_size: int = 1
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    use_bias: bool = True
    num_spatial_dims: int = 1
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    @nn.compact
    def __call__(self, x):

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(dtype=self.param_dtype),
            (self.features, 1, self.kernel_size),
            self.param_dtype
        )
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},"
                f" but input has shape {x.shape}.",
            )

        # def rava_run(input_rava):
        #     input_rava = jnp.expand_dims(input_rava, 0)
        #     input_rava = lax.conv_general_dilated(
        #         lhs=input_rava,
        #         rhs=jnp.asarray(kernel, dtype=self.dtype),
        #         window_strides=stride,
        #         padding=padding,
        #         rhs_dilation=dilation,
        #         feature_group_count=self.groups,
        #     )
        #     input_rava = jnp.squeeze(input_rava, axis=0)
        #     if self.use_bias:
        #         bias = self.param(
        #             "bias",
        #             nn.initializers.zeros_init(),
        #             (self.features,) + (1,) * self.num_spatial_dims,
        #             self.param_dtype
        #         )
        #         input_rava = input_rava + jnp.asarray(bias, dtype=self.dtype)
        #     return input_rava

        # return nn.vmap(
        #     rava_run,
        #     in_axes=0,
        #     out_axes=0,
        #     variable_axes={"params": 0},
        #     split_rngs={"params": False}
        # )(x)

        # x = jnp.expand_dims(x, 0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=jnp.asarray(kernel, dtype=self.dtype),
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
        )
        if self.use_bias:
            bias = self.param(
                "bias",
                nn.initializers.zeros_init(),
                (self.features,),
                self.param_dtype
            )
            x = x + jnp.asarray(bias.reshape(1, -1, 1), dtype=self.dtype)
        return x


def mamba_ssm(
        u: jax.Array,
        delta: jax.Array,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: Optional[jax.Array] = None,
        delta_bias: Optional[jax.Array] = None,
        delta_softplus: bool = False,
        associative_scan: bool = True,
) -> jax.Array:
    if delta_bias is not None:
        raise NotImplementedError("delta_bias not implemented yet.")

    l, d_in = u.shape
    n = A.shape[1]

    delta = jnp.asarray(delta, dtype=jnp.float32)

    if delta_softplus:
        delta = jax.nn.softplus(delta)

    delta_A = jnp.exp(einsum(delta, A, "l d_in, d_in n -> l d_in n"))
    delta_B_u = einsum(delta, B, u, "l d_in, l n, l d_in -> l d_in n")

    x = jnp.zeros((d_in, n))

    def _scan_fn(x, params):
        d_A, d_Bu, C = params

        x = d_A * x + d_Bu
        return x, einsum(x, C, "d_in n, n -> d_in")

    def _associative_scan_fn(s, c):
        return tuple((c[0] * s[0], c[0] * s[1] + c[1]))

    if associative_scan:
        _, y = jax.lax.associative_scan(_associative_scan_fn, (delta_A, delta_B_u))
        y = einsum(y, C, "L d_in n, L n -> L d_in")
    else:
        _, y = jax.lax.scan(_scan_fn, init=x, xs=[delta_A, delta_B_u, C])

    y = y + u * D
    return y


class MambaRMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(fjformer.linen.linen.control_quantization(self.weight, self.dtype))
        return output * weight


class Conv(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros_init()
    conv_general_dilated: Optional[ConvGeneralDilatedT] = None
    conv_general_dilated_cls: Any = None

    @property
    def shared_weights(self) -> bool:
        return True

    @nn.compact
    def __call__(self, inputs: Array) -> Array:

        if isinstance(self.kernel_size, int):
            raise TypeError(
                'Expected Conv kernel_size to be a'
                ' tuple/list of integers (eg.: [3, 3]) but got'
                f' {self.kernel_size}.'
            )
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(
                x: Optional[Union[int, Sequence[int]]]
        ) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[
                                                     num_batch_dimensions:
                                                     ]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == 'CIRCULAR':
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            pads = (
                    zero_pad
                    + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                    + [(0, 0)]
            )
            inputs = jnp.pad(inputs, pads, mode='wrap')
            padding_lax = 'VALID'
        elif padding_lax == 'CAUSAL':
            if len(kernel_size) != 1:
                raise ValueError(
                    'Causal padding is only implemented for 1D convolutions.'
                )
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = 'VALID'

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.

            inf_f = in_features // self.feature_group_count
            # inf_f = 1
            kernel_shape = (self.features, inf_f,) + kernel_size

        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    '`lax.conv_general_dilated_local` does not support '
                    f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            if self.conv_general_dilated_cls is not None:
                conv_general_dilated = self.conv_general_dilated_cls()
            elif self.conv_general_dilated is not None:
                conv_general_dilated = self.conv_general_dilated
            else:
                conv_general_dilated = lax.conv_general_dilated
            conv_output_shape = eval_shape(
                lambda lhs, rhs: conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                    lhs_dilation=input_dilation,
                    rhs_dilation=kernel_dilation,
                ),
                inputs,
                ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (
                np.prod(kernel_size) * in_features,
                self.features,
            )

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                'Mask needs to have the same shape as weights. '
                f'Shapes are: {self.mask.shape}, {kernel_shape}'
            )

        kernel = self.param(
            'kernel', self.kernel_init, kernel_shape, self.param_dtype
        )
        kernel = jnp.asarray(kernel.transpose(2, 1, 0), self.dtype)
        if self.mask is not None:
            kernel *= self.mask

        if self.use_bias:
            if self.shared_weights:
                bias_shape = (self.features,)
            else:
                bias_shape = conv_output_shape[1:]

            bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if self.shared_weights:
            if self.conv_general_dilated_cls is not None:
                conv_general_dilated = self.conv_general_dilated_cls()
            elif self.conv_general_dilated is not None:
                conv_general_dilated = self.conv_general_dilated
            else:
                conv_general_dilated = lax.conv_general_dilated
            y = conv_general_dilated(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
            )
        else:
            y = lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=self.precision,
            )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


class FlaxMambaMixer(nn.Module):
    config: MambaConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        config = self.config
        hidden_size = config.hidden_size
        ssm_state_size = config.state_size
        intermediate_size = config.intermediate_size
        time_step_rank = config.time_step_rank

        self.conv1d = Conv1D(
            # features=config.intermediate_size,
            # kernel_size=(config.conv_kernel,),
            # feature_group_count=config.intermediate_size,
            # padding="SAME",
            # strides=(1,),
            # dtype=self.dtype,
            # param_dtype=self.param_dtype,
            # precision=self.precision,
            # use_bias=config.use_conv_bias,
            # #  ---- # #
            features=config.intermediate_size,
            kernel_size=config.conv_kernel,
            groups=config.intermediate_size,
            stride=1,
            padding=config.conv_kernel - 1,
            use_bias=config.use_conv_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        dt_init_std = self.config.time_step_rank ** -0.5 * self.config.time_step_scale
        if self.config.time_step_init_scheme == "constant":
            init_kernel_dt = nn.initializers.constant(dt_init_std, dtype=self.param_dtype)
        elif self.config.time_step_init_scheme == "random":
            # def init_kernel_dt():
            def init_kernel_dt(key, _shape, _dtype):
                return jax.nn.initializers.uniform(
                    scale=dt_init_std * 2, dtype=self.param_dtype
                )(key, _shape, _dtype) - dt_init_std

            # return init_r
        else:
            init_kernel_dt = nn.initializers.normal(self.config.initializer_range, self.param_dtype)

        dt = jax.lax.clamp(
            self.config.time_step_floor,
            jnp.exp(
                jax.random.normal(
                    key=self.make_rng("params"),
                    shape=(self.config.intermediate_size,),
                    dtype=self.param_dtype
                )
                * (jnp.log(self.config.time_step_max) - jnp.log(self.config.time_step_min))
                + jnp.log(self.config.time_step_min)
            ),
            self.config.time_step_max
        )
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))

        dense_class = functools.partial(
            Linear,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            **get_dot_general_by_bits(
                self.config.bits,
                self.config.easy_method
            )
        )
        self.in_proj = dense_class(
            intermediate_size * 2,
            use_bias=config.use_bias
        )
        self.x_proj = dense_class(
            time_step_rank + ssm_state_size * 2,
            use_bias=False
        )
        self.dt_proj = dense_class(
            intermediate_size,
            use_bias=True,
            kernel_init=init_kernel_dt,
            bias_init=lambda s1, s2, s3: inv_dt
        )
        self.out_proj = dense_class(
            hidden_size,
            use_bias=config.use_bias
        )

        self.A_log = self.param(
            "A_log",
            init_to_value(
                jnp.log(
                    jnp.broadcast_to(
                        jnp.arange(1, ssm_state_size + 1, dtype=jnp.float32)[None, :],
                        (intermediate_size, ssm_state_size)
                    )
                ),
                self.dtype
            )
        )
        self.D = self.param(
            "D", init_to_value(
                jnp.ones(
                    intermediate_size
                ),
                self.dtype
            )
        )
        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = self.config.conv_kernel
        self.time_step_rank = self.config.time_step_rank

    def __call__(self, input_states, cache_params=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(0, 2, 1)
        hidden_states, gate = jnp.split(projected_states, 2, axis=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx]
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]  # [batch, intermediate_size, conv_kernel_size]
                conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
                conv_state = conv_state.at[:, :, -1].set(hidden_states[:, :, 0])
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = jnp.sum(conv_state * self.conv1d.variables["kernel"][:, 0, :], axis=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.variables["bias"]
                hidden_states = jnp.expand_dims(self.act(hidden_states).astype(dtype), -1)
                # [batch, intermediate_size, 1] : decoding
            else:
                padding_amount = self.conv_kernel_size - hidden_states.shape[-1]
                conv_state = jnp.pad(hidden_states, ((0, 0), (0, padding_amount)), mode='constant')
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
                # [batch, intermediate_size, seq_len]
        else:
            ssm_state = jnp.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        ssm_parameters = self.x_proj(hidden_states.transpose(0, 2, 1))
        time_step, B, C = jnp.split(
            ssm_parameters,
            indices_or_sections=[self.time_step_rank, self.time_step_rank + self.ssm_state_size],
            axis=-1
        )
        discrete_time_step = self.dt_proj(time_step)
        # [batch, seq_len, intermediate_size]
        discrete_time_step = jax.nn.softplus(discrete_time_step).transpose(0, 2, 1)
        # [batch, intermediate_size, seq_len]
        A = -jnp.exp(self.A_log.astype(jnp.float32))
        # [intermediate_size, ssm_state_size]
        modified_a = jnp.expand_dims(jnp.expand_dims(A, axis=0), axis=2)
        modified_time_step = jnp.expand_dims(discrete_time_step, axis=-1)
        discrete_A = jnp.exp(modified_a * modified_time_step)
        # [batch, intermediate_size, seq_len, ssm_state_size]

        discrete_B = modified_time_step * B[:, jnp.newaxis, :, :].astype(jnp.float32)
        # [batch, intermediate_size, seq_len, ssm_state_size]

        deltaB_u = discrete_B * hidden_states[:, :, :, jnp.newaxis].astype(jnp.float32)

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            # [batch, intermediate_size, ssm_state]

            scan_output = jax.lax.batch_matmul(ssm_state.astype(dtype), jnp.expand_dims(C[:, i, :], -1))
            # [batch, intermediate_size, 1]

            scan_outputs.append(scan_output[:, :, 0])

        scan_output = jnp.stack(scan_outputs, axis=-1)
        # [batch, seq_len, intermediate_size]
        scan_output = scan_output + (hidden_states * self.D[jnp.newaxis, :, jnp.newaxis])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(0, 2, 1))
        # [batch, seq_len, hidden_size]
        return contextualized_states


class FlaxMambaBlock(nn.Module):
    config: MambaConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self):
        config = self.config
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        block = FlaxMambaMixer
        if self.config.gradient_checkpointing != "":
            block = nn_partitioning.remat(
                block,
                static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.gradient_checkpointing)
            )
        self.mixer = block(
            config,
            self.layer_idx,
            self.dtype,
            self.param_dtype,
            self.precision
        )

    def __call__(
            self,
            hidden_states: chex.Array,
            cache_params: Optional[FlaxMambaCache] = None
    ) -> chex.Array:
        residual = hidden_states
        hidden_states = self.norm(
            hidden_states
        )
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states = self.mixer(
            hidden_states,
            cache_params
        )
        hidden_states = residual + hidden_states
        return hidden_states


class FlaxMambaLayerCollection(nn.Module):
    config: MambaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        self.blocks = [
            FlaxMambaBlock(
                config=self.config,
                layer_idx=layer_idx,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(layer_idx)
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]

    def __call__(
            self,
            hidden_states: chex.Array,
            cache_params: Optional[FlaxMambaCache] = None,
            output_hidden_states: bool = False
    ) -> Tuple[chex.Array, Union[None, Tuple[chex.Array, ...]]]:
        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cache_params
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


class FlaxMambaModule(nn.Module):
    config: MambaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        config = self.config
        self.embeddings = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.layers = FlaxMambaLayerCollection(
            config=config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.norm_f = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            cache_params: Optional[chex.Array] = None,
            deterministic: bool = True,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, MambaOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not deterministic else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if deterministic and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = FlaxMambaCache(
                self.config, inputs_embeds.shape[0], dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        hidden_states, all_hidden_states = self.layers(hidden_states, cache_params=cache_params)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class FlaxMambaForCausalLMModule(nn.Module):
    config: MambaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        self.backbone = FlaxMambaModule(
            self.config,
            self.dtype,
            self.param_dtype,
            self.precision
        )
        self.lm_head = Linear(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            cache_params: Optional[chex.Array] = None,
            deterministic: bool = True,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, MambaCausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input_ids: Optional[chex.Array] = None,
        # inputs_embeds: Optional[chex.Array] = None,
        # deterministic: bool = True,
        # cache_params: Optional[List[chex.Array]] = None,
        # use_cache: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

        mamba_outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
            cache_params=cache_params,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = mamba_outputs[0]

        logits = self.lm_head(hidden_states).astype(jnp.float32)

        if not return_dict:
            return (logits,) + mamba_outputs[1:]

        return MambaCausalLMOutput(
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )


class FlaxMambaPretrainedModel(EasyDeLFlaxPretrainedModel):
    config_class = MambaConfig
    base_model_prefix = "backbone"
    module_class: nn.Module = None

    def __init__(
            self,
            config: MambaConfig,
            input_shape: Tuple = (1, 1),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[Union[str, lax.Precision]] = None,
            _do_init: bool = True,
            **kwargs,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines what happens when it's created.
        The __init__ function can take arguments, but self is always required (it refers to the instance of the object).


        :param self: Refer to the object itself
        :param config: MambaConfig: Pass the configuration to the module
        :param input_shape: Tuple: Specify the shape of the input to the model
        :param seed: int: Set the seed for random number generation
        :param dtype: jnp.dtype: Specify the data type of the model ra
        :param param_dtype: jnp.dtype: Specify the data type of the param_dtype
        :param precision: Optional[Union[str, lax.Precision]]: precision for model operations
        :param _do_init: bool: Control whether the module is initialized or not
        :param kwargs: Pass in any additional parameters that the module_class might need
        :param : Specify the number of layers in the network
        :return: The super() of the class

        """
        module = self.module_class(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            **kwargs
        )
        super().__init__(
            config,
            module,
            input_shape=(input_shape[0], 1),
            seed=seed,
            dtype=dtype,
            _do_init=_do_init
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        """
        The init_weights function is used to initialize the weights of a model.

        :param self: Access variables that belong to the class
        :param rng: jax.random.PRNGKey: Initialize the weights of the model
        :param input_shape: Tuple: Specify the shape of the input tensor
        :param params: FrozenDict: Pass in the parameters of a pre-trained model
        :return: A frozendict of parameters

        """
        input_ids = jnp.zeros(input_shape, dtype="i4")
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            input_ids,
            return_dict=False
        )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        return None

    def __call__(
            self,
            input_ids: Optional[chex.Array] = None,
            inputs_embeds: Optional[chex.Array] = None,
            cache_params: dict = None,
            deterministic: bool = True,
            params: dict = None,
            dropout_rng: jax.random.PRNGKey = None,
            train: bool = False,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
            add_params_field: bool = False,
            attention_mask: Optional[chex.Array] = None,  # Ignored(we are using an SSM model not attention)
            use_cache: bool = False,
            **kwargs
    ):
        """
        The __call__ function is the main function of a JAX module.

        :param self: Represent the instance of the class
        :param input_ids: Optional[chex.Array]: Pass in the input tokens
        :param inputs_embeds: Optional[chex.Array]: Pass in the embedded tokens
        :param cache_params: dict: Pass in the past cache_params from a previous call to __call__
        :param params: dict: Pass in the parameters of the model
        :param dropout_rng: jax.random.PRNGKey: Make sure that the dropout is applied in a random way
        :param train: bool: Determine whether to use dropout or not
        :param output_hidden_states: Optional[bool]: Return the hidden states of all layers
        :param return_dict: Optional[bool]: Determine whether to return a dictionary or not
        :param extra_embedding: Optional[Union[jnp.ndarray,None]]: Pass in the embedding for the input_ids
        :param add_params_field: bool: Add the params field to the inputs dictionary
        :return: A tuple of the following:

        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        assert sequence_length <= self.config.max_position_embeddings, "Maximum Position Embedding Reached !"
        if cache_params is not None:
            assert isinstance(cache_params, FlaxMambaCache), f"Wrong cache input_type of {type(cache_params)}"
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        rngs["params"] = jax.random.key(0)

        inputs = {
            "params": params or self.params
        } if add_params_field else params or self.params

        # input_ids: Optional[chex.Array] = None,
        # inputs_embeds: Optional[chex.Array] = None,
        # cache_params: Optional[chex.Array] = None,
        # deterministic: bool = True,
        # use_cache: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,

        return self.module.apply(
            inputs,
            input_ids,
            inputs_embeds,
            cache_params,
            train,
            use_cache,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=False,
        )


class FlaxMambaModel(FlaxMambaPretrainedModel):
    module_class = FlaxMambaModule


class FlaxMambaForCausalLM(FlaxMambaPretrainedModel):
    module_class = FlaxMambaForCausalLMModule

    def update_inputs_for_generation(
            self,
            outputs: MambaOutput,
            model_kwargs: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            max_length,
            **kwargs
    ):
        return {
            "cache_params": kwargs.get("cache_params", None)
        }
