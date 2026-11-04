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

"""``kda_serving_ep_split``: eSurge KDA heads split over (tp, ep) must reproduce the unsplit layer.

On a 4-device mesh (ep2 x tp2) the split cache view (``head_shard_axes == ("tp", "ep")``, one head pair per device)
and the default view (heads over tp only) start from the same pools. A multi-step eSurge sequence (two fresh prompts,
a mixed step with a decode row, a continuation and a new prompt, then decode-only steps; explicit pool slots and a
padding row) must leave bitwise-equal conv windows and recurrent states, and layer outputs that differ only by the
rounding of o_proj's partial sums.
"""

from types import SimpleNamespace

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import KDACacheView
from easydel.caching.kda.cache import KDACacheConfig
from easydel.layers.linear_attention._kda import kda_serving_head_axes
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

HIDDEN, HEAD_DIM, SLOTS = 256, 128, 8
EP2_TP2 = (1, 1, 1, 2, 2, 1)
LEAVES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
# (token bucket, tokens per row, context lens per row, pool slot per row); rows without tokens name slot 0 like eSurge.
PLAN = (
    (128, (50, 37, 0, 0), (50, 37, 0, 0), (5, 2, 0, 0)),
    (128, (1, 40, 30, 0), (51, 77, 30, 0), (5, 2, 7, 0)),
    (4, (1, 1, 1, 0), (52, 78, 31, 0), (5, 2, 7, 0)),
    (4, (1, 1, 1, 0), (53, 79, 32, 0), (5, 2, 7, 0)),
    (4, (1, 1, 1, 0), (54, 80, 33, 0), (5, 2, 7, 0)),
)


def _need_four_devices():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices for the ep2 x tp2 mesh")


def _config(dims=EP2_TP2, split=True, heads=8):
    return ed.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=4,
        index_kpool=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_head_dim=HEAD_DIM,
        linear_num_heads=heads,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        sharding_axis_dims=dims,
        gather_shared_projection_input=True,
        kda_serving_ep_split=split,
    )


def _model(config):
    model = ed.Glm5NextForCausalLM(config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    return model, model.model.layers[0].self_attn


def _jax_mesh(model):
    return getattr(model.mesh, "jax_mesh", model.mesh)


def _auto(config):
    """The automatic setting (``None``); this tree's constructor still stores ``False`` for an omitted field."""
    config.kda_serving_ep_split = None
    return config


def _cache_configs(model):
    split = model.create_kda_cache_config(batch_size=SLOTS, serving=True)
    plain = KDACacheConfig.create(
        num_hidden_layers=split.num_hidden_layers,
        partition_axis=split.partition_axis,
        batch_size=SLOTS,
        key_dim=split.key_dim,
        value_dim=split.value_dim,
        d_conv=split.d_conv,
        recurrent_state_shape=split.recurrent_state_shape,
    )
    return split, plain


def _random_pools(model, cfg, seed):
    with model.mesh:
        view = KDACacheView.init(config=cfg, layer_index=0, dtype=jnp.bfloat16)
    rng = np.random.default_rng(seed)
    pools = []
    for name in LEAVES:
        leaf = getattr(view, name)
        scale = 0.1 if name == "recurrent_state" else 0.5
        pools.append(jax.device_put(jnp.asarray(scale * rng.standard_normal(leaf.shape), jnp.bfloat16), leaf.sharding))
    return tuple(pools)


def _compiled_step(model, layer, cfg, pools):
    graphdef, state = spx.export(layer)
    hidden_sharding = NamedSharding(_jax_mesh(model), P(None, None, "tp"))

    def step(state_, hidden, pools_, qsl, ctx, slots):
        view = KDACacheView(
            q_conv_state=pools_[0],
            k_conv_state=pools_[1],
            v_conv_state=pools_[2],
            recurrent_state=pools_[3],
            positions=jnp.zeros((pools_[3].shape[0],), jnp.int32),
            metadata=cfg,
            layer_index=0,
        )
        meta = SimpleNamespace(query_start_loc=qsl, context_lens=ctx, recurrent_state_indices=slots)
        out = spx.bind(graphdef, state_)(hidden_states=hidden, cache_view=view, cache_metadata=meta)
        return out.attention_output, tuple(getattr(out.cache_view, name) for name in LEAVES)

    shardings = tuple(pool.sharding for pool in pools)
    return jax.jit(step, donate_argnums=(2,), out_shardings=(hidden_sharding, shardings)), state, hidden_sharding


def test_split_axes_follow_the_setting_the_mesh_and_the_cache_kind():
    _need_four_devices()
    assert kda_serving_head_axes(_config(), serving=True) == ("tp", "ep")
    # Caches created outside eSurge (``init_cache``) never split, whatever the setting.
    assert kda_serving_head_axes(_config()) is None
    # ``False`` is the escape hatch.
    assert kda_serving_head_axes(_config(split=False), serving=True) is None
    # ``None`` (the automatic rule) splits every eSurge cache whose mesh allows it.
    assert kda_serving_head_axes(_auto(_config(split=False)), serving=True) == ("tp", "ep")
    assert kda_serving_head_axes(_auto(_config(split=False))) is None
    # No expert-parallel devices (ep1 x tp4): nothing to split over.
    assert kda_serving_head_axes(_auto(_config(dims=(1, 1, 1, 1, 4, 1))), serving=True) is None
    # 2 heads do not split over 4 devices: today's head-axis layout is kept.
    assert kda_serving_head_axes(_auto(_config(heads=2)), serving=True) is None
    assert kda_serving_head_axes(_config(heads=2), serving=True) is None


def test_only_esurge_cache_configs_split_under_the_automatic_rule():
    _need_four_devices()
    model, _ = _model(_config(split=False))
    _auto(model.config.get_text_config())
    # ``init_cache`` builds its views without a paged-cache config: the KDA pools keep the head axis.
    views = model.init_operations_cache_config(batch_size=SLOTS, max_length=128)
    assert [cfg.head_shard_axes for cfg in views] == [None]
    assert model.create_kda_cache_config(batch_size=SLOTS).head_shard_axes is None
    assert model.create_kda_cache_config(batch_size=SLOTS, serving=True).head_shard_axes == ("tp", "ep")


def test_split_cache_pools_shard_heads_over_tp_and_ep():
    _need_four_devices()
    model, _ = _model(_config())
    cfg = model.create_kda_cache_config(batch_size=SLOTS, serving=True)
    assert cfg.head_shard_axes == ("tp", "ep")
    with model.mesh:
        view = KDACacheView.init(config=cfg, layer_index=0, dtype=jnp.bfloat16)
    for name in LEAVES:
        leaf = getattr(view, name)
        assert tuple(leaf.sharding.spec[1]) == ("tp", "ep"), (name, leaf.sharding.spec)
        assert leaf.addressable_shards[0].data.shape[1] * 4 == leaf.shape[1], name


@pytest.mark.parametrize("wire", [False, True], ids=["plain-linears", "row-parallel-engines"])
def test_split_layer_matches_unsplit_over_prefill_mixed_and_decode_steps(wire):
    _need_four_devices()
    model, layer = _model(_config())
    if wire:
        # eSurge's loader wires the row-parallel distributed_matmul engines (auto_modeling.py); the split o_proj
        # must give the same result through them.
        model.wire_distributed_matmul()
    split_cfg, plain_cfg = _cache_configs(model)
    configs = {"split": split_cfg, "plain": plain_cfg}
    pools = {kind: _random_pools(model, cfg, seed=7) for kind, cfg in configs.items()}
    replicated = NamedSharding(_jax_mesh(model), P())
    steps = {}
    rng = np.random.default_rng(0)
    with model.mesh:
        for bucket, lens, ctx, slots in PLAN:
            hidden = rng.standard_normal((1, bucket, HIDDEN))
            qsl = np.concatenate([[0], np.cumsum(lens)])
            tokens = int(qsl[-1])
            outs = {}
            for kind, cfg in configs.items():
                if (kind, bucket) not in steps:
                    steps[(kind, bucket)] = _compiled_step(model, layer, cfg, pools[kind])
                fn, state, hidden_sharding = steps[(kind, bucket)]
                out, pools[kind] = fn(
                    state,
                    jax.device_put(jnp.asarray(hidden, jnp.bfloat16), hidden_sharding),
                    pools[kind],
                    jax.device_put(jnp.asarray(qsl, jnp.int32), replicated),
                    jax.device_put(jnp.asarray(ctx, jnp.int32), replicated),
                    jax.device_put(jnp.asarray(slots, jnp.int32), replicated),
                )
                outs[kind] = np.asarray(out, np.float32)
            for name, a, b in zip(LEAVES, pools["split"], pools["plain"], strict=True):
                np.testing.assert_array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16), name)
            live_split, live_plain = outs["split"][0, :tokens], outs["plain"][0, :tokens]
            assert np.linalg.norm(live_split - live_plain) / np.linalg.norm(live_plain) < 1e-2
            np.testing.assert_array_equal(outs["split"][0, tokens:], outs["plain"][0, tokens:])
    assert tuple(pools["split"][3].sharding.spec[1]) == ("tp", "ep")


def _has_all_to_all(hlo_text: str) -> bool:
    return " all-to-all(" in hlo_text or " all-to-all-start(" in hlo_text


def test_split_prefill_trades_tokens_for_heads_and_decode_keeps_its_o_proj():
    """A head-split prefill step reaches ``o_proj`` through one all-to-all over ep (tokens for heads), also with the
    row-parallel engines eSurge's loader wires; a decode bucket (replicated residual under ``decode_mode_specs``)
    keeps today's ``o_proj`` and has no all-to-all."""
    from easydel.infra.sharding import decode_mode_specs

    _need_four_devices()
    model, layer = _model(_config())
    model.wire_distributed_matmul()
    split_cfg, _ = _cache_configs(model)
    pools = _random_pools(model, split_cfg, seed=7)
    replicated = NamedSharding(_jax_mesh(model), P())
    texts = {}
    with model.mesh:
        for name, bucket, lens in (("prefill", 128, (50, 37, 0, 0)), ("decode", 4, (1, 1, 1, 0))):
            fn, state, hidden_sharding = _compiled_step(model, layer, split_cfg, pools)
            qsl = np.concatenate([[0], np.cumsum(lens)])
            args = (
                state,
                jax.device_put(jnp.zeros((1, bucket, HIDDEN), jnp.bfloat16), hidden_sharding),
                pools,
                jax.device_put(jnp.asarray(qsl, jnp.int32), replicated),
                jax.device_put(jnp.asarray(lens, jnp.int32), replicated),
                jax.device_put(jnp.asarray((5, 2, 7, 0), jnp.int32), replicated),
            )
            with decode_mode_specs(name == "decode"):
                texts[name] = fn.lower(*args).compile().as_text()
    assert _has_all_to_all(texts["prefill"])
    assert not _has_all_to_all(texts["decode"])


def test_flag_leaves_the_cacheless_forward_unchanged():
    _need_four_devices()
    hidden = jnp.asarray(np.random.default_rng(1).standard_normal((2, 16, HIDDEN)), jnp.bfloat16)
    outs = []
    for split in (True, False):
        model, layer = _model(_config(split=split))
        with model.mesh:
            outs.append(np.asarray(layer(hidden_states=hidden).attention_output, np.float32))
    np.testing.assert_array_equal(outs[0], outs[1])
