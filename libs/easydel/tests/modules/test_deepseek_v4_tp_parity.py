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

"""DeepSeek-V4 tensor-parallel parity, through the real load path.

Serving V4-Flash at A16W4 on a 4-chip v5p-8 produced coherent text at
``ep=4, tp=1`` but token salad at ``ep=2, tp=2`` and ``ep=1, tp=4``. These
cases pin ``tp>1`` against ``tp=1`` so a regression in the tensor-parallel
weight path is caught on CPU.

**The comparison must save once and load twice.** Comparing two independently
*initialized* models across meshes is invalid here: fused projections
(``gate_up_proj`` and friends) are stored TP-interleaved, so at ``tp=2`` the
physical column order is ``[gate_0, up_0, gate_1, up_1]`` while at ``tp=1`` it
is ``[gate, up]``. Two freshly-initialized models therefore hold *identical raw
values under different conventions* -- i.e. different logical weights -- and
diverge by ~1e-2 with nothing actually broken. Saving once and loading under
each mesh runs ``retp_fused_state``'s re-interleave, which is what production
does and what makes the comparison meaningful.

That re-interleave is a **no-op at tp=1** (``saved_tp == target_tp``), so
``tp=1`` alone never exercises it -- which is exactly why these cases matter.
Both quantization states are covered because the serving path loads
CHANNELWISE bits=4 while the rest of the CPU suite runs unquantized, and a
fused quantized param is codes *plus* per-output-channel scales that have to
be re-interleaved together.
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from jax import numpy as jnp

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2,
    reason="tensor-parallel parity needs >=2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)",
)

HIDDEN = 128
BATCH, SEQ = 2, 24
NO_TP = (1, 1, 1, 1, 1, 1)
TP = (1, 1, 1, 1, 2, 1)

# fp32 reassociation only: splitting a contraction across 2 shards reorders the
# sum. Anything above this is a real divergence, not arithmetic noise.
TOL = 5e-4


def _config(sharding_axis_dims):
    """Small V4 spanning a sliding and a compressed-sparse layer."""
    import easydel as ed

    config = ed.DeepseekV4Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "compressed_sparse_attention"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=8,
        index_n_heads=2,
        index_head_dim=16,
        # Deterministic indexer regime: with random init many relu'd scores are
        # exactly zero and a truncating top-k orders ties backend-dependently.
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        tie_word_embeddings=False,
    )
    config.sharding_axis_dims = sharding_axis_dims
    # XLA:CPU cannot lower ragged-all-to-all (the expert-parallel collective).
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _inputs():
    rng = np.random.default_rng(7)
    ids = jnp.asarray(rng.integers(0, 128, size=(BATCH, SEQ)), jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(SEQ)[None, :], (BATCH, SEQ))
    return ids, positions


def _logits(model, config):
    ids, positions = _inputs()
    with config.mesh:
        return np.asarray(model(input_ids=ids, position_ids=positions).logits, np.float32)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tp=1 checkpoint every case loads from, so weights are common."""
    import easydel as ed

    path = tmp_path_factory.mktemp("v4_tp_parity") / "ckpt"
    config = _config(NO_TP)
    with config.mesh:
        model = ed.DeepseekV4ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        model.eval()
        model = model.shard_model()
    model.save_pretrained(str(path))
    return str(path)


def _load(path, sharding_axis_dims, *, quantize=False):
    import easydel as ed

    kwargs = {}
    if quantize:
        from easydel.layers.quantization import QuantizationType

        # Exactly what the v5p-8 serving path uses (A16W4).
        kwargs = dict(
            quantization_config=ed.EasyDeLQuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=4),
            apply_quantization=True,
        )
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        sharding_axis_dims=sharding_axis_dims,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        **kwargs,
    )
    model.eval()
    return model


def test_tp_matches_single_device(checkpoint):
    """tp=2 must reproduce tp=1 logits: sharding may not change the math."""
    ref = _load(checkpoint, NO_TP)
    tp = _load(checkpoint, TP)

    ref_logits = _logits(ref, ref.config)
    tp_logits = _logits(tp, tp.config)

    assert ref_logits.shape == tp_logits.shape
    worst = float(np.max(np.abs(ref_logits - tp_logits)))
    assert worst < TOL, f"tp=2 diverged from tp=1 (unquantized): max|diff|={worst}"


def test_tp_matches_single_device_channelwise_int4(checkpoint):
    """The serving configuration: A16W4 weights under tensor parallelism.

    Both meshes quantize the same checkpoint, so quantization error is
    common-mode; what remains is whether the packed codes and their
    per-channel scales are re-interleaved together for the live ``tp``.
    """
    ref = _load(checkpoint, NO_TP, quantize=True)
    tp = _load(checkpoint, TP, quantize=True)

    ref_logits = _logits(ref, ref.config)
    tp_logits = _logits(tp, tp.config)

    assert ref_logits.shape == tp_logits.shape
    worst = float(np.max(np.abs(ref_logits - tp_logits)))
    assert worst < TOL, f"tp=2 diverged from tp=1 (CHANNELWISE bits=4): max|diff|={worst}"


def test_hyper_stream_spec_leaves_stream_axis_replicated():
    """The rank-4 residual stream must not put a mesh axis on ``hc_mult``.

    V4 carries ``hc_mult`` parallel residual streams, so the stream tensor is
    ``[B, T, hc, D]`` -- rank 4. ``common_types.HiddenStateSharding`` is the
    rank-3 ``[B, T, D]`` spec, and passing it here does not raise: the sanitizer
    only collapses a spec when ``len(spec) > len(shape)``, and 3 <= 4, so it was
    applied positionally and put ``TP`` on the stream axis while leaving the
    hidden feature axis replicated.

    That is invisible at tp=1 and silently relayouts the stream at tp>1, which
    is how ``DeepseekV4HyperMix``'s Sinkhorn ``shard_map`` -- which declares its
    input replicated and runs with ``check_vma=False`` -- came to be handed a
    sharded ``comb`` on TPU. Asserting the resolved spec directly is the only
    CPU-visible half of that failure: the ``shard_map`` itself is gated on
    ``jax.default_backend() == "tpu"``.
    """
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import _HyperStreamSharding

    config = _config(TP)
    hc = config.hc_mult
    spec = config.runtime_sharding_resolver.resolve(
        dynamic_axes=_HyperStreamSharding,
        shape=(1, SEQ, hc, HIDDEN),
    )

    assert len(spec) == 4, f"spec must match the rank-4 stream tensor, got {spec}"
    assert spec[2] is None, f"stream (hc_mult) axis must stay replicated, got {spec}"
    # and the feature axis must be the one carrying tensor parallelism
    assert spec[3] is not None, f"hidden feature axis should be sharded, got {spec}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
