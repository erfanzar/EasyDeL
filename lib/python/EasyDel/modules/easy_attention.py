import warnings
from functools import partial

import jax
from chex import Array
from fjformer import with_sharding_constraint
from flax.linen import dot_product_attention_weights
from jax import numpy as jnp, lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, Mesh
from fjformer.pallas_operations.flash_attention import gpu as flash_attn_gpu
from fjformer.pallas_operations.flash_attention import tpu as flash_attn_tpu
from fjformer.pallas_operations.ring_attention import (
    ring_flash_attention_tpu,
    ring_attention_standard,
    ring_attention
)
from typing import Tuple, Callable, Type, Any, Optional, Literal
from dataclasses import dataclass

from .flax_modelling_utils import get_gradient_checkpoint_policy


@dataclass
class AttentionOutput:
    attention_weights: Optional[Array] = None
    attention_outputs: Optional[Array] = None


def get_flash_attention() -> Tuple[Callable, bool, bool]:
    """
    return: FlashAttention FN, Upcast Needed to float32,do_shard_map
    """
    platform = jax.lib.xla_bridge.get_backend().platform
    if platform == "gpu":
        float32_logits = False
        ring_attention_fn = flash_attn_gpu.mha
        do_shard_map = True
    elif platform == "tpu":
        float32_logits = True
        ring_attention_fn = flash_attn_tpu.flash_attention
        do_shard_map = False
    else:
        raise ValueError(f"Unsupported platform {platform}")

    return ring_attention_fn, float32_logits, do_shard_map


class EasyAttention:
    def __init__(
            self,
            attn_type: Literal["normal", "alibi"],
            attn_mechanism: Literal[
                "normal", "flash", "splash", "ring"
            ],
            block_k: int,
            block_q: int,
            block_b: int,
            block_k_major: int,
            block_q_major_dkv: int,
            block_k_major_dkv: int,
            block_k_dkv: int,
            block_q_dkv: int,
            block_k_major_dq: int,
            block_k_dq: int,
            block_q_dq: int,
            sm_scale: float,
            num_attention_heads: int,
            head_dims: int,
            mesh: Mesh,
            query_partition_spec: PartitionSpec,
            key_partition_spec: PartitionSpec,
            value_partition_spec: PartitionSpec,
            bias_partition_spec: PartitionSpec,
            attention_partition_spec: PartitionSpec,
            scan_ring_attention: bool = True,
            scan_attention_layers: bool = False,
            attention_dropout: float = 0.0,
            dtype: jnp.dtype = jnp.float32,
            precision: lax.Precision = lax.Precision("fastest"),
            force_float32_tpu: bool = True,
            use_shard_map: bool = False

    ):
        platform = jax.lib.xla_bridge.get_backend().platform
        if attn_mechanism == "splash":
            raise NotImplementedError("Splash Attention is not Supported YET !")
        if attn_mechanism == "flash" and platform not in ["gpu", "tpu"]:
            raise NotImplementedError("Flash Attention is only supported for GPU/TPU.")
        self.platform = platform
        self.attn_type = attn_type
        self.attn_mechanism = attn_mechanism
        self.block_k = block_k
        self.block_q = block_q
        self.block_b = block_b
        self.block_k_major = block_k_major
        self.block_q_major_dkv = block_q_major_dkv
        self.block_k_major_dkv = block_k_major_dkv
        self.block_k_dkv = block_k_dkv
        self.block_q_dkv = block_q_dkv
        self.block_k_major_dq = block_k_major_dq
        self.block_k_dq = block_k_dq
        self.block_q_dq = block_q_dq
        self.num_attention_heads = num_attention_heads
        self.head_dims = head_dims
        self.sm_scale = sm_scale
        self.mesh = mesh
        self.query_partition_spec = query_partition_spec
        self.key_partition_spec = key_partition_spec
        self.value_partition_spec = value_partition_spec
        self.bias_partition_spec = bias_partition_spec
        self.attention_partition_spec = attention_partition_spec
        self.attention_dropout = attention_dropout
        self.dtype = dtype
        self.precision = precision
        self.force_float32_tpu = force_float32_tpu
        self.use_shard_map = use_shard_map
        self.scan_ring_attention = scan_ring_attention
        self.scan_attention_layers = scan_attention_layers
        self.assertion_mkv_err = f"""
query_states, key_states, value_states and bias shapes must be like
query_states Shape : [batch_size, q_seq_len , num_attention_heads({self.num_attention_heads}), head_dims({self.head_dims})]
key_states   Shape : [batch_size, kv_seq_len, num_attention_heads({self.num_attention_heads}), head_dims({self.head_dims})]
value_states Shape : [batch_size, kv_seq_len, num_attention_heads({self.num_attention_heads}), head_dims({self.head_dims})]
bias         Shape : [batch_size, num_attention_heads({self.num_attention_heads}), q_seq_len , kv_seq_len]
    """

    def _qkv_check(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
    ):
        ...

    def __call__(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            segment_ids: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            use_pjit_attention_force: bool = False,
            uses_cache: bool = False
    ):
        with self.mesh:
            batch_size = query_states.shape[0]
            assert batch_size == key_states.shape[0] == value_states.shape[0], "Batch Size for q,k,v wont match"
            k_v_req_shape = (
                batch_size,
                key_value_sequence_length,
                self.num_attention_heads,
                self.head_dims
            )
            q_shape = (
                batch_size,
                query_sequence_length,
                self.num_attention_heads,
                self.head_dims
            )
            assert query_states.shape == q_shape, self.assertion_mkv_err + (
                f"\nMiss Match {query_states.shape} and "
                f"required Shape {q_shape}"
            )
            assert key_states.shape == k_v_req_shape, self.assertion_mkv_err + (
                f"\nMiss Match {key_states.shape} and "
                f"required Shape {k_v_req_shape}"
            )
            assert value_states.shape == k_v_req_shape, self.assertion_mkv_err + (
                f"\nMiss Match {value_states.shape} and "
                f"required Shape {k_v_req_shape}"
            )

            if self.attn_type == "normal":

                if self.attn_mechanism == "flash":

                    query_states = query_states.transpose(0, 2, 1, 3)
                    key_states = key_states.transpose(0, 2, 1, 3)
                    value_states = value_states.transpose(0, 2, 1, 3)

                    attentions = self._qkv_normal_flash_op(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        bias=bias,
                        dropout_rng=dropout_rng,
                        use_pjit_attention_force=use_pjit_attention_force,
                        causal=causal,
                        deterministic=deterministic,
                        query_sequence_length=query_sequence_length,
                        key_value_sequence_length=key_value_sequence_length,
                    )

                    attentions.attention_outputs = attentions.attention_outputs.transpose(0, 2, 1, 3)

                elif self.attn_mechanism == "normal":

                    attentions = self._qkv_normal_op(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        bias=bias,
                        dropout_rng=dropout_rng,
                        use_pjit_attention_force=use_pjit_attention_force,
                        causal=causal,
                        deterministic=deterministic,
                        query_sequence_length=query_sequence_length,
                        key_value_sequence_length=key_value_sequence_length,
                    )
                elif self.attn_mechanism == "ring":
                    attentions = self._qkv_ring_op(
                        query_states=query_states,
                        key_states=key_states,
                        value_states=value_states,
                        bias=bias,
                        dropout_rng=dropout_rng,
                        use_pjit_attention_force=use_pjit_attention_force,
                        causal=causal,
                        deterministic=deterministic,
                        query_sequence_length=query_sequence_length,
                        key_value_sequence_length=key_value_sequence_length,
                        segment_ids=segment_ids,
                    )

                elif self.attn_mechanism == "splash":
                    raise NotImplementedError("Splash Attention is not Implemented YET!")
                else:
                    raise ValueError(f"Unknown Attention mechanism of {self.attn_mechanism}")
                return attentions
            elif self.attn_type == "alibi":
                raise NotImplementedError("Not Implemented Yet i guess!")
            else:
                raise ValueError(f"Unknown Attention Type of {self.attn_type}")

    def _qkv_ring_op(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            use_pjit_attention_force: bool = False,
            segment_ids: Optional[Array] = None
    ):
        if segment_ids is None:
            segment_ids = jnp.zeros((query_states.shape[0], query_sequence_length), dtype="i4")
        if self.scan_ring_attention and query_states.shape[1] > max(
                self.block_q,
                self.block_k
        ):
            if self.platform == "tpu":
                ring_attention_fn = ring_flash_attention_tpu
            else:
                ring_attention_fn = ring_attention
            ring_attention_sharded = shard_map(
                partial(
                    ring_attention_fn,
                    axis_name="sp",
                    float32_logits=True,
                    blockwise_kwargs=dict(
                        deterministic=deterministic,
                        dropout_rng=dropout_rng,
                        attn_pdrop=self.attention_dropout,
                        causal=True,
                        query_chunk_size=self.block_q,
                        key_chunk_size=self.block_k,
                        dtype=self.dtype,
                        policy=get_gradient_checkpoint_policy('nothing_saveable'),
                        precision=self.precision,
                        prevent_cse=not self.scan_attention_layers,
                    )
                ),
                mesh=self.mesh,
                in_specs=(
                    self.query_partition_spec,
                    self.key_partition_spec,
                    self.value_partition_spec,
                    self.bias_partition_spec,
                    PartitionSpec(("dp", "fsdp"), None),
                ),
                out_specs=self.attention_partition_spec,
                check_rep=False
            )
            attn_output = ring_attention_sharded(query_states, key_states, value_states, bias, segment_ids)
            attn_output = with_sharding_constraint(attn_output, self.attention_partition_spec)
        else:

            warnings.warn(
                "Using Ring attention on CPUs or GPUs are not recommended due to miss computations at the moment. "
                "please refer to other types of attention mechanism.your are bing fell back on `ring_attention_sharded`"
                f" Usage conditions was\nscan_ring_attention = {self.scan_ring_attention} [MUST BE TRUE]"
                f"\nquery_states.shape[1]({query_states.shape[1]}) > max({self.block_q},{self.block_k})"
                f"({max(self.block_q, self.block_k)})"
            )
            query_sequence_partition = None if query_states.shape[1] == 1 else "sp"
            ring_attention_sharded = shard_map(
                partial(ring_attention_standard, axis_name="sp"),
                mesh=self.mesh,
                in_specs=(
                    PartitionSpec(("dp", "fsdp"), query_sequence_partition, "tp", None),
                    PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                    PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                    PartitionSpec(("dp", "fsdp"), None, query_sequence_partition, None)
                ),
                out_specs=PartitionSpec(("dp", "fsdp"), query_sequence_partition, "tp", None),
                check_rep=False
            )
            attn_output = ring_attention_sharded(
                query_states, key_states, value_states, bias
            )
        return AttentionOutput(
            attention_weights=None,
            attention_outputs=attn_output
        )

    def _qkv_normal_op(
            self,
            *,  # it's Kwarg Only
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            use_pjit_attention_force: bool = False
    ) -> AttentionOutput:

        attn_weights = None
        dtype_c = jnp.promote_types(self.dtype, jnp.float32)
        if self.use_shard_map:
            attn_weights = shard_map(
                partial(
                    dot_product_attention_weights,
                    dtype=dtype_c,
                    deterministic=deterministic,
                    dropout_rate=self.attention_dropout,
                    precision=self.precision,
                    dropout_rng=dropout_rng
                ),
                mesh=self.mesh,
                in_specs=(
                    self.query_partition_spec,
                    self.key_partition_spec,
                    self.bias_partition_spec
                ),
                out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                check_rep=False
            )(
                query_states, key_states, bias
            )
        else:
            attn_weights = dot_product_attention_weights(
                query=query_states,
                key=key_states,
                bias=bias,
                dtype=dtype_c,
                deterministic=deterministic,
                dropout_rate=self.attention_dropout,
                precision=self.precision,
                dropout_rng=dropout_rng
            )

        if use_pjit_attention_force:
            attn_weights = with_sharding_constraint(
                attn_weights, self.attention_partition_spec
            )

        attn_output = jnp.einsum(
            "...hqk,...khd->...qhd",
            attn_weights.astype(dtype_c),
            value_states.astype(dtype_c),
            precision=self.precision
        ).astype(dtype_c)

        return AttentionOutput(
            attention_outputs=attn_output,
            attention_weights=attn_weights
        )

    def _qkv_normal_flash_op(
            self,
            query_states: Array,
            key_states: Array,
            value_states: Array,
            query_sequence_length: int,
            key_value_sequence_length: int,
            bias: Optional[Array] = None,
            causal: bool = False,
            deterministic: bool = False,
            dropout_rng: Optional[random.PRNGKey] = None,
            use_pjit_attention_force: bool = False
    ) -> AttentionOutput:

        batch_size, num_attention_heads, query_sequence_length, head_dims = query_states.shape
        if bias is not None:
            if bias.shape[1] != num_attention_heads:
                es = bias.shape
                bias = bias.repeat(
                    num_attention_heads, 1,
                )
        assert bias.shape == (
            batch_size,
            self.num_attention_heads,
            query_sequence_length,
            key_value_sequence_length
        ), self.assertion_mkv_err
        flash_func, float32_logits, _ = get_flash_attention()
        if float32_logits:
            query_states, key_states, value_states = map(
                lambda s: s.astype(jnp.float32),
                (query_states, key_states, value_states)
            )
        attention_o = shard_map(
            partial(
                flash_func,
                causal=causal,
                sm_scale=self.sm_scale,
                block_sizes=flash_attn_tpu.BlockSizes(
                    block_b=self.block_b,
                    block_k=self.block_k,
                    block_q=self.block_q,
                    block_k_major=self.block_k_major,
                    block_k_dq=self.block_k_dq,
                    block_q_dq=self.block_q_dq,
                    block_k_dkv=self.block_k_dkv,
                    block_q_dkv=self.block_q_dkv,
                    block_k_major_dq=self.block_k_major_dq,
                    block_k_major_dkv=self.block_k_major_dkv,
                    block_q_major_dkv=self.block_q_major_dkv,
                ),
                debug=False
            ),
            in_specs=(
                self.query_partition_spec,
                self.key_partition_spec,
                self.value_partition_spec,
                self.bias_partition_spec
            ),
            out_specs=(
                self.attention_partition_spec
            ),
            mesh=self.mesh,
            check_rep=False,
        )(
            query_states,
            key_states,
            value_states,
            bias,
        )
        return AttentionOutput(
            attention_outputs=attention_o,
            attention_weights=None
        )
