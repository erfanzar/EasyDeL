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

"""Chunk-scanned fused-MoE dispatch (``moe_chunk_size``).

The fused MoE dispatch materializes a ``[local_tokens * k, H]`` permuted
buffer per core; at production shapes (v5p-2048, 131k buckets) that buffer is
4-17 GB per core with ~3 copies live across fwd+bwd. ``moe_chunk_size`` opts
into processing the local token stream in fixed-size chunks under
``jax.lax.scan`` + ``jax.checkpoint``, capping every live dispatch buffer at
``chunk * k * H`` regardless of batch/sequence size.

Chunking over tokens is mathematically exact for token-choice top-k routing:
each token's output depends only on its own k experts, so splitting the
stream can only change float accumulation order. These tests pin:

* default-off is bit-identical (``moe_chunk_size=None`` equals a config that
  predates the knob entirely),
* chunked outputs and input-gradients match the single-pass dispatch on the
  ep=1 all-to-all mesh and the ring fsdp2/ep2/tp2 mesh, including
  non-dividing chunk sizes (tail padding) and chunk > tokens (single padded
  chunk),
* the shard_map body really does see chunk-sized buffers,
* the non-ring ep>1 chunked branch traces (per-chunk ``ragged_all_to_all``
  parameter plumbing) — XLA:CPU cannot *execute* ragged_all_to_all, so that
  branch's runtime numerics remain TPU-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
TOL = 1e-5


def _require_fake_mesh():
    """Skip unless the fake 8-device CPU host mesh is available."""
    if jax.device_count() < 8:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def _build_gptoss_block(
    *,
    sharding_axis_dims,
    ring: bool,
    moe_chunk_size: int | None = None,
    num_experts: int = 8,
    seed: int = 123,
):
    """Instantiate a GptOssMLP (fused MoE) under a mesh.

    Same scaffold as ``test_moe_fsdp_batch_sharding``; identical seeds give
    identical weights across meshes and chunk settings, so chunked-vs-single
    comparisons isolate the dispatch pipeline.

    Args:
        sharding_axis_dims: 6-tuple of mesh dims in ``(pp,dp,fsdp,ep,tp,sp)``
            order.
        ring: Whether to enable ring-of-experts (the ep>1 variant that lowers
            on XLA:CPU).
        moe_chunk_size: Chunk size for the scanned dispatch (``None``
            disables chunking).
        num_experts: Number of routed experts.
        seed: Parameter init seed.

    Returns:
        A ``GptOssMLP`` module built under the requested mesh.
    """
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=num_experts,
        num_experts_per_tok=2,
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        use_ring_of_experts=ring,
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_chunk_size=moe_chunk_size,
    )
    return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _input(hidden_size: int = 32):
    """Deterministic (8, 4, hidden) test batch shared by all meshes."""
    return jax.random.normal(jax.random.PRNGKey(0), (8, 4, hidden_size), dtype=jnp.float32)


def test_chunk_none_is_bit_identical_to_knob_absent():
    """moe_chunk_size=None must equal a config that predates the knob, bitwise.

    ``delattr`` simulates configs serialized before ``moe_chunk_size``
    existed: the module reads the knob via ``getattr(..., None)``, so both
    blocks must take the identical single-pass dispatch trace.
    """
    _require_fake_mesh()

    explicit_none = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, moe_chunk_size=None)
    legacy = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, moe_chunk_size=None)
    delattr(legacy.config, "moe_chunk_size")
    x = _input()

    with explicit_none.config.mesh:
        out_none, _ = explicit_none(x)
    with legacy.config.mesh:
        out_legacy, _ = legacy(x)

    assert out_none.shape == x.shape
    assert bool(jnp.array_equal(out_none, out_legacy)), (
        "moe_chunk_size=None must be bit-identical to a config without the attribute"
    )


def test_chunked_shard_map_body_sees_chunk_sized_buffers():
    """The permute inside the scan must observe (chunk, hidden) local buffers.

    ``permute`` invokes ``refine_inputs_hook`` on the flattened local tokens;
    under the chunk scan the hook is traced once with the chunk-shaped
    buffer, proving the chunked path is active and its working set is capped
    at ``chunk`` rows (times k after replication) instead of the full local
    stream.
    """
    from easydel.layers.moe._communication_utils import MoeFusedHooks

    _require_fake_mesh()

    chunk = 4
    block = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, moe_chunk_size=chunk)
    seen: list[tuple[int, ...]] = []

    def spy(inputs_2d, weights, inputs_shape):
        seen.append(tuple(inputs_2d.shape))
        return inputs_2d

    block.moe_hooks = MoeFusedHooks(refine_inputs_hook=spy)
    x = _input()

    with block.config.mesh:
        out, _ = block(x)

    assert out.shape == x.shape
    # Local stream is (8/(1*4))*4 = 8 tokens; the hook must never see it.
    assert set(seen) == {(chunk, x.shape[-1])}, (
        f"chunked dispatch must permute (chunk={chunk}, H) buffers, saw {set(seen)}"
    )


@pytest.mark.parametrize(
    ("candidate_dims", "ring", "chunk"),
    [
        # ep=1 all-to-all-free mesh: local stream is 8 tokens per shard.
        pytest.param((1, 1, 4, 1, 2, 1), False, 4, id="noring-fsdp4-tp2-chunk4"),
        pytest.param((1, 1, 4, 1, 2, 1), False, 3, id="noring-fsdp4-tp2-chunk3-tailpad"),
        pytest.param((1, 1, 4, 1, 2, 1), False, 999, id="noring-fsdp4-tp2-chunk-gt-T"),
        # ring mesh: post-gather stream is (8/2)*4*2 = 32 tokens per shard.
        pytest.param((1, 1, 2, 2, 2, 1), True, 8, id="ring-fsdp2-ep2-tp2-chunk8"),
        pytest.param((1, 1, 2, 2, 2, 1), True, 5, id="ring-fsdp2-ep2-tp2-chunk5-tailpad"),
        pytest.param((1, 1, 2, 2, 2, 1), True, 999, id="ring-fsdp2-ep2-tp2-chunk-gt-T"),
    ],
)
def test_chunked_matches_unchunked_outputs_and_grads(candidate_dims, ring, chunk):
    """Chunked dispatch must reproduce single-pass outputs AND input grads.

    Identical weights (same seed), identical mesh — only the dispatch
    pipeline differs. Non-dividing chunks exercise the zero-padded tail
    (padded rows carry zero combine weights); chunk > tokens exercises the
    degenerate single padded chunk. Gradients flow through the scan +
    ``jax.checkpoint`` body, the tail slice, and (in ring mode) the
    expert-axis ``psum_scatter`` combine.
    """
    _require_fake_mesh()

    single = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring, moe_chunk_size=None)
    chunked = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring, moe_chunk_size=chunk)
    x = _input()

    with single.config.mesh:
        out_single, _ = single(x)
    with chunked.config.mesh:
        out_chunked, _ = chunked(x)

    assert out_chunked.shape == out_single.shape
    assert bool(jnp.all(jnp.isfinite(out_chunked)))
    max_abs_diff = float(jnp.max(jnp.abs(out_chunked - out_single)))
    assert max_abs_diff < TOL, f"chunked forward diverges on {candidate_dims} chunk={chunk}: {max_abs_diff:.3e}"

    def loss(block, xin):
        out, _ = block(xin)
        return jnp.sum(out * out)

    with single.config.mesh:
        grad_single = jax.grad(lambda xin: loss(single, xin))(x)
    with chunked.config.mesh:
        grad_chunked = jax.grad(lambda xin: loss(chunked, xin))(x)

    assert grad_chunked.shape == grad_single.shape
    assert bool(jnp.all(jnp.isfinite(grad_chunked)))
    grad_diff = float(jnp.max(jnp.abs(grad_chunked - grad_single)))
    assert grad_diff < TOL, f"chunked input-grad diverges on {candidate_dims} chunk={chunk}: {grad_diff:.3e}"


def test_chunked_nonring_ep2_traces():
    """The chunked non-ring ep>1 branch (per-chunk ragged a2a) must trace.

    XLA:CPU cannot execute ``ragged_all_to_all`` (unsupported by the CPU
    thunk emitter), so runtime numerics for this branch are TPU-only;
    abstract evaluation still validates the per-chunk all-to-all parameter
    plumbing, static chunk count, and output shapes.
    """
    _require_fake_mesh()

    block = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ring=False, moe_chunk_size=4)
    x = _input()

    with block.config.mesh:
        out_struct = jax.eval_shape(lambda xin: block(xin)[0], x)

    assert tuple(out_struct.shape) == tuple(x.shape)


def test_chunk_size_must_be_positive():
    """A non-positive chunk size must fail loudly at config or dispatch time."""
    _require_fake_mesh()

    block = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, moe_chunk_size=None)
    block.config.moe_chunk_size = 0  # bypass config validation on purpose
    x = _input()

    with block.config.mesh, pytest.raises(ValueError, match="moe_chunk_size"):
        block(x)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
