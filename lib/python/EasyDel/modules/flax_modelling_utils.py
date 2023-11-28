import fjformer.attention
from jax.interpreters import pxla
from jax.experimental.pjit import with_sharding_constraint as wsc
import jax
from flax import linen as nn
from functools import partial
import chex
from typing import Sequence, Optional
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),

}


def get_names_from_partition_spec(partition_specs):
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


def get_gradient_checkpoint_policy(name):
    gradients = dict(
        everything_saveable=jax.checkpoint_policies.everything_saveable,
        nothing_saveable=jax.checkpoint_policies.nothing_saveable,
        dots_saveable=jax.checkpoint_policies.dots_saveable,
        checkpoint_dots=jax.checkpoint_policies.checkpoint_dots,
        dots_with_no_batch_dims_saveable=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        checkpoint_dots_with_no_batch_dims=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        save_anything_except_these_names=jax.checkpoint_policies.save_anything_except_these_names,
        save_any_names_but_these=jax.checkpoint_policies.save_any_names_but_these,
        save_only_these_names=jax.checkpoint_policies.save_only_these_names,
        save_from_both_policies=jax.checkpoint_policies.save_from_both_policies
    )
    return gradients[name]


def repeat_kv_bnsh(x: chex.Array, n_rep: int) -> chex.Array:
    bs, n_kv_heads, s, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    return x.reshape(bs, n_kv_heads * n_rep, s, head_dim)


def repeat_kv_bsnh(x: chex.Array, n_rep: int) -> chex.Array:
    bs, s, n_kv_heads, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    x = x.transpose(0, 2, 1, 3)

    return x.reshape(bs, s, n_kv_heads * n_rep, head_dim)


def precompute_freq_cis(max_position_embedding, head_dim):
    inv_freq = 1.0 / (10000 ** (jax.numpy.arange(0, head_dim, 2, dtype=jax.numpy.float32) / head_dim))
    freq = jax.numpy.einsum("i , j -> i j", jax.numpy.arange(max_position_embedding), inv_freq).astype("float32")

    embed = jax.numpy.concatenate((freq, freq), axis=-1)
    return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jax.numpy.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(tensor, sin_, cos_):
    return (tensor * cos_) + (rotate_half(tensor) * sin_)


def get_ranks_and_size(mesh):
    out = dict(mesh=mesh)
    mp_size = mesh.shape["tp"] * mesh.shape["mp"]
    mp_node_size = max(1, mp_size // jax.local_device_count())
    dp_node_size = jax.process_count() // mp_node_size
    out.update(mp_node_size=mp_node_size,
               dp_node_size=dp_node_size)

    dp_node_rank = jax.process_index() // mp_node_size
    mp_node_rank = jax.process_index() % mp_node_size
    out.update(dp_node_rank=dp_node_rank,
               mp_node_rank=mp_node_rank)
    return out


def get_flash_attention():
    """
    return: FlashAttention FN, Upcast Needed to float32,do_shard_map
    """
    platform = jax.lib.xla_bridge.get_backend().platform
    if platform == "gpu":
        float32_logits = False
        ring_attention_fn = fjformer.attention.ring_flash_attention_gpu
        do_shard_map = True
    elif platform == "tpu":
        float32_logits = True
        ring_attention_fn = fjformer.attention.tpu_flash_attention
        do_shard_map = False
    else:
        raise ValueError(f"Unsupported platform {platform}")

    return ring_attention_fn, float32_logits, do_shard_map


def smart_flash_attention(
        q: chex.Array,
        k: chex.Array,
        v: chex.Array,
        bias: chex.Array,
        block_k: int,
        block_q: int,
        block_b: int,
        q_seq_len: int,
        kv_seq_len: int,
        num_attention_heads: int,
        head_dims: int,
        causal: bool,
        attn_pdrop: float,
        mesh: jax.sharding.Mesh = None,
        dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: jax.lax.Precision = jax.lax.Precision('fastest'),
        dropout_rng: jax.random.PRNGKey = None,
        force_float32_tpu: bool = True,
        deterministic: bool = False
):
    """
    Smart Flash Attention mechanism for efficient attention computation.

    Args:
    - q: Query tensor with shape [batch_size, num_attention_heads, q_seq_len, head_dims].
    - k: Key tensor with shape [batch_size, num_attention_heads, kv_seq_len, head_dims].
    - v: Value tensor with shape [batch_size, num_attention_heads, kv_seq_len, head_dims].
    - bias: Bias tensor with shape [batch_size, num_attention_heads, q_seq_len, kv_seq_len].
    - block_k: Block size for key tensor reshaping.
    - block_q: Block size for query tensor reshaping.
    - block_b: Block size for bias tensor reshaping.
    - q_seq_len: Length of the query sequence.
    - kv_seq_len: Length of the key-value sequence.
    - num_attention_heads: Number of attention heads.
    - head_dims: Dimensionality of each attention head.
    - causal: If True, applies causal masking to the attention scores.
    - attn_pdrop: Dropout probability for attention weights.
    - mesh: Mesh specifying the data distribution for parallel computation.
    - dtype: Data type of the tensors.
    - precision: Precision mode for computation (default is 'fastest').
    - dropout_rng: Random number generator key for dropout.
    - force_float32_tpu: If True, forces computation to use float32 on TPU.
    - deterministic: If True, ensures deterministic computation.

    Returns:
    - Output tensor with the same shape as the input value tensor v.

    Raises:
    - ValueError: If the shapes of input tensors are not compatible for attention computation.

    """
    assertion_mkv_err = """
    Q,K,V and bias shapes must be like
    Q Shape : [batch_size, num_attention_heads, q_seq_len, head_dims]
    K Shape : [batch_size, num_attention_heads, kv_seq_len, head_dims]
    V Shape : [batch_size, num_attention_heads, kv_seq_len, head_dims]
    bias Shape : [batch_size, num_attention_heads, q_seq_len, kv_seq_len]
    """
    batch_size = q.shape[0]
    assert batch_size == k.shape[0] == v.shape[0], 'Batch Size for q,k,v wont match'

    assert q.shape == (batch_size, num_attention_heads, q_seq_len, head_dims), assertion_mkv_err
    assert k.shape == (batch_size, num_attention_heads, kv_seq_len, head_dims), assertion_mkv_err
    assert v.shape == (batch_size, num_attention_heads, kv_seq_len, head_dims), assertion_mkv_err
    assert bias.shape == (batch_size, num_attention_heads, q_seq_len, kv_seq_len), assertion_mkv_err

    flash_attn_fn, f32_upcast, do_shard_map = get_flash_attention()

    if do_shard_map:
        q, k, v = map(lambda x: jax.numpy.transpose(x, (0, 2, 1, 3)), [q, k, v])
        assert mesh is not None, 'For Using Shard Map on GPUs you have to pass Mesh'
        ring_attention_sharded = shard_map(
            partial(
                flash_attn_fn,
                axis_name="mp",
                float32_logits=f32_upcast,
                blockwise_kwargs=dict(
                    deterministic=deterministic,
                    dropout_rng=dropout_rng,
                    attn_pdrop=attn_pdrop,
                    causal=causal,
                    query_chunk_size=block_q,
                    key_chunk_size=block_k,
                    dtype=dtype,
                    policy=jax.checkpoint_policies.nothing_saveable,
                    precision=precision,
                    prevent_cse=False,
                )
            ),
            mesh=mesh,
            in_specs=(
                PS(("dp", "fsdp"), "mp", "tp", None),
                PS(("dp", "fsdp"), "mp", "tp", None),
                PS(("dp", "fsdp"), "mp", "tp", None),
                PS(("dp", "fsdp"), None, None, None)
            ),
            out_specs=PS(("dp", "fsdp"), "mp", "tp", None),
            check_rep=False
        )
        attn_output = ring_attention_sharded(q, k, v, bias)
        attn_output = with_sharding_constraint(attn_output, PS(("dp", "fsdp"), "mp", "tp", None))
    else:
        if force_float32_tpu or f32_upcast:
            q, k, v = map(lambda x: x.astype(jax.numpy.float32), [q, k, v])
        attn_output = fjformer.attention.jax_flash_attn_tpu.flash_attention(
            q,
            k,
            v,
            bias,
            None,
            causal=False,
            sm_scale=1.0,
            block_sizes=fjformer.attention.jax_flash_attn_tpu.BlockSizes(
                block_b=block_b,
                block_k=block_k,
                block_q=block_q,
                block_k_major=block_k
            ),
            debug=False,
        )
    attn_output = attn_output.astype(dtype)
    return attn_output


def create_mesh(
        axis_dims: Sequence[int] = (1, -1, 1, 1), axis_names: Sequence[str] = ("dp", "fsdp", "tp", "mp"), backend=""
):
    array_devices = jax.numpy.ones((len(jax.devices() if backend == "" else jax.devices(backend)), 1))
    resh = array_devices.reshape(axis_dims).shape

    return jax.sharding.Mesh(
        create_device_mesh(resh), axis_names
    )


class JaxBaseClassModel:
    def __init__(
            self,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "mp"),
            backend: Optional[None] = None
    ):
        self.axis_dims = axis_dims
        self.axis_names = axis_names
        self.backend = backend if backend is not None else ""

    def jax_mesh(self) -> jax.sharding.Mesh:
        return create_mesh(
            axis_dims=self.axis_dims,
            axis_names=self.axis_names,
            backend=(self.backend if self.backend is not None else "") if hasattr(self, 'backend') else ""
        )

    def get_axis_dims(self) -> Sequence[int]:
        return self.axis_dims

    def get_axis_names(self) -> Sequence[str]:
        return self.axis_names

    def get_backend(self) -> str:
        return self.backend if not self.backend == "" else jax.lib.xla_bridge.get_backend().platform

    @staticmethod
    def get_flash_attention():
        return get_flash_attention()


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator
