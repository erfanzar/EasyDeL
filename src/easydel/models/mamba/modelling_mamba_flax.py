import functools
import itertools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
from einops import einsum
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from flax import nnx
from easydel.models.common import RMSNorm as MambaRMSNorm
from easydel.models.flax_modelling_utils import ACT2FN
from easydel.models.mamba.mamba_configuration import MambaConfig as MambaConfig
from easydel.models.modelling_utils import BaseNNXModule
from easydel.models.flax_modelling_utils import MambaOutput, MambaCausalLMOutput


def init_to_value(x, dtype):
    return lambda _: x.astype(dtype)


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
                raise ValueError(f"x!=n ({x}!=({n}))")
        else:
            return tuple(itertools.repeat(x, n))

    return parse


class Conv1D(nnx.Module):
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
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.precision = precision
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.stride = stride
        self.num_spatial_dims = num_spatial_dims
        self.dtype = dtype

        self.kernel = nnx.Param(
            nnx.initializers.lecun_normal(dtype=param_dtype)(
                rngs.params(),
                (features, 1, kernel_size),
                param_dtype,
            )
        )
        self.bias = nnx.Param(
            nnx.initializers.zeros_init()(rngs.params(), (features,), param_dtype)
            if use_bias
            else None
        )

    def __call__(self, x):
        kernel = self.kernel.value
        bias = self.bias.value
        unbatched_rank = self.num_spatial_dims + 2
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},"
                f" but input has shape {x.shape}.",
            )
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=jnp.asarray(kernel, dtype=self.dtype),
            window_strides=(self.stride,),
            padding=((self.padding, self.padding),),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups,
        )
        if bias is not None:
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

    _, d_in = u.shape
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


class MambaMixer(nnx.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        hidden_size = config.hidden_size
        ssm_state_size = config.state_size
        intermediate_size = config.intermediate_size
        time_step_rank = config.time_step_rank

        self.conv1d = Conv1D(
            features=config.intermediate_size,
            kernel_size=config.conv_kernel,
            groups=config.intermediate_size,
            stride=1,
            padding=config.conv_kernel - 1,
            use_bias=config.use_conv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
        if self.config.time_step_init_scheme == "constant":
            init_kernel_dt = nnx.initializers.constant(
                dt_init_std,
                dtype=self.param_dtype,
            )
        elif self.config.time_step_init_scheme == "random":
            # def init_kernel_dt():
            def init_kernel_dt(key, _shape, _dtype):
                return (
                    nnx.initializers.uniform(
                        scale=dt_init_std * 2,
                        dtype=self.param_dtype,
                    )(
                        key,
                        _shape,
                        _dtype,
                    )
                    - dt_init_std
                )

            # return init_r
        else:
            init_kernel_dt = nnx.initializers.normal(
                self.config.initializer_range,
                self.param_dtype,
            )

        dt = jax.lax.clamp(
            self.config.time_step_floor,
            jnp.exp(
                jax.random.normal(
                    key=self.make_rng("params"),
                    shape=(self.config.intermediate_size,),
                    dtype=self.param_dtype,
                )
                * (
                    jnp.log(self.config.time_step_max)
                    - jnp.log(self.config.time_step_min)
                )
                + jnp.log(self.config.time_step_min)
            ),
            self.config.time_step_max,
        )
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))

        dense_class = functools.partial(
            nnx.Linear,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.in_proj = dense_class(
            hidden_size,
            intermediate_size * 2,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.x_proj = dense_class(
            hidden_size,
            time_step_rank + ssm_state_size * 2,
            use_bias=False,
            rngs=rngs,
        )
        self.dt_proj = dense_class(
            hidden_size,
            intermediate_size,
            use_bias=True,
            kernel_init=init_kernel_dt,
            bias_init=lambda s1, s2, s3: inv_dt,
            rngs=rngs,
        )
        self.out_proj = dense_class(
            hidden_size,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.A_log = nnx.Param(
            init_to_value(
                jnp.log(
                    jnp.broadcast_to(
                        jnp.arange(1, ssm_state_size + 1, dtype=jnp.float32)[None, :],
                        (intermediate_size, ssm_state_size),
                    )
                ),
                self.dtype,
            ),
        )
        self.D = nnx.Param(init_to_value(jnp.ones(intermediate_size), self.dtype))
        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = self.config.conv_kernel
        self.time_step_rank = self.config.time_step_rank

    def __call__(
        self,
        input_states,
        cache_params=None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(0, 2, 1)
        hidden_states, gate = jnp.split(projected_states, 2, axis=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx]
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[
                    self.layer_idx
                ]  # [batch, intermediate_size, conv_kernel_size]
                conv_state = jnp.roll(
                    conv_state,
                    shift=-1,
                    axis=-1,
                )
                conv_state = conv_state.at[:, :, -1].set(hidden_states[:, :, 0])
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = jnp.sum(
                    conv_state * self.conv1d.kernel.value[:, 0, :],
                    axis=-1,
                )
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias.value
                hidden_states = jnp.expand_dims(
                    self.act(hidden_states).astype(dtype), -1
                )
                # [batch, intermediate_size, 1] : decoding
            else:
                padding_amount = self.conv_kernel_size - hidden_states.shape[-1]
                conv_state = jnp.pad(
                    hidden_states, ((0, 0), (0, padding_amount)), mode="constant"
                )
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
            indices_or_sections=[
                self.time_step_rank,
                self.time_step_rank + self.ssm_state_size,
            ],
            axis=-1,
        )
        discrete_time_step = self.dt_proj(time_step)
        # [batch, seq_len, intermediate_size]
        discrete_time_step = jax.nn.softplus(discrete_time_step).transpose(0, 2, 1)
        # [batch, intermediate_size, seq_len]
        A = -jnp.exp(self.A_log.value.astype(jnp.float32))
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

            scan_output = jax.lax.batch_matmul(
                ssm_state.astype(dtype), jnp.expand_dims(C[:, i, :], -1)
            )
            # [batch, intermediate_size, 1]

            scan_outputs.append(scan_output[:, :, 0])

        scan_output = jnp.stack(scan_outputs, axis=-1)
        # [batch, seq_len, intermediate_size]
        scan_output = scan_output + (
            hidden_states * self.D.value[jnp.newaxis, :, jnp.newaxis]
        )
        scan_output = scan_output * self.act(gate)

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(0, 2, 1))
        # [batch, seq_len, hidden_size]
        return contextualized_states


class MambaBlock(nnx.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        block = MambaMixer
        # if self.config.gradient_checkpointing != "":
        #     block = nn_partitioning.remat(
        #         block,
        #         static_argnums=(1,),
        #         policy=get_gradient_checkpoint_policy(
        #             self.config.gradient_checkpointing
        #         ),
        #     )
        self.mixer = block(
            config=config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        cache_params: Optional[FlaxMambaCache] = None,
    ) -> chex.Array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hidden_states = self.mixer(hidden_states, cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaModel(BaseNNXModule):
    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.embeddings = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
        self.layers = [
            MambaBlock(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        self.norm_f = MambaRMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        cache_params: Optional[chex.Array] = None,
        deterministic: bool = True,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not deterministic else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and input_embeds at the same time, and must specify either one"
            )

        if input_embeds is None:
            input_embeds = self.embeddings(input_ids)

        if deterministic and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = FlaxMambaCache(
                self.config, input_embeds.shape[0], dtype=input_embeds.dtype
            )

        hidden_states = input_embeds

        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(hidden_states, cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if use_cache:
            cache_params.seqlen_offset += input_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, cache_params, all_hidden_states]
                if v is not None
            )

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MambaForCausalLM(BaseNNXModule):
    def __init__(
        self,
        config: MambaConfig,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: Optional[Union[str, lax.Precision]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.backbone = MambaModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Optional[chex.Array] = None,
        input_embeds: Optional[chex.Array] = None,
        cache_params: Optional[chex.Array] = None,
        deterministic: bool = True,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MambaCausalLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        mamba_outputs = self.backbone(
            input_ids=input_ids,
            input_embeds=input_embeds,
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

    def update_inputs_for_generation(
        self,
        outputs: MambaOutput,
        model_kwargs: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        **kwargs,
    ):
        return {"cache_params": kwargs.get("cache_params", None)}

    @property
    def can_generate(self):
        return True

    @property
    def model_architecure_type(self):
        return "mamba1"
