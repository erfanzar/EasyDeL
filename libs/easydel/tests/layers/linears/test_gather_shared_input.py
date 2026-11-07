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

"""``gather_shared_input`` and the automatic rule of ``gather_shared_projection_input``.

- On a tp mesh the projections of the gathered input give the plain projections' values and gradients (the custom VJP
  only re-lays-out the summed cotangent).
- Without tensor parallelism, or for a non-3-D input, it returns the input unchanged.
- ``resolve_gather_shared_projection_input``: ``True`` and ``False`` win; the default ``None`` gathers only in inference
  mode and only when the hidden axis of the input is sharded under the active layout (not in eSurge's pure-decode
  layout, not without tensor parallelism, not for a 2-D input); a config without the field takes the automatic rule.
- A GLM-5-Next KDA stack with the default ``None`` lowers like ``False`` outside inference mode (training unchanged)
  and like ``True`` in inference mode, with bitwise-equal logits.
"""

from __future__ import annotations

import collections
from types import SimpleNamespace

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import decode_mode_specs
from easydel.layers.linears import ColumnParallelLinear, gather_shared_input, resolve_gather_shared_projection_input
from easydel.utils import set_inference_mode
from easydel.utils.traversals import iter_module_search
from jax.sharding import NamedSharding
from spectrax import common_types as ct

HI = jax.lax.Precision.HIGHEST
F32 = jnp.float32


def _config(tp_size):
    return ed.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        sharding_axis_dims=(1, 1, 4 // tp_size, 1, tp_size, 1),
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _projections():
    rngs = spx.Rngs(0)
    return tuple(
        ColumnParallelLinear(64, out, use_bias=True, dtype=F32, param_dtype=F32, precision=HI, rngs=rngs)
        for out in (64, 32, 8)
    )


def _jax_mesh(config):
    mesh = config.mesh
    return getattr(mesh, "jax_mesh", mesh)


def _sharded_input(config, shape=(4, 16, 64)):
    """A ``[B, S, D]`` hidden state laid out the way the decoder layers constrain it (``HiddenStateSharding``)."""
    spec = config.runtime_sharding_resolver.resolve(dynamic_axes=ct.HiddenStateSharding, shape=shape)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    return jax.device_put(x, NamedSharding(_jax_mesh(config), spec))


def _decision(config, x, *, inference, decode=False):
    """The resolver's answer while ``x`` is traced under the given inference-mode and decode-layout scopes."""
    seen = []

    def trace(a):
        seen.append(resolve_gather_shared_projection_input(config, a))
        return a

    with config.mesh, set_inference_mode(inference), decode_mode_specs(decode):
        jax.make_jaxpr(trace)(x)
    return seen[-1]


def test_gathered_projections_match_plain_values_and_gradients():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    config = _config(tp_size=2)
    lins = _projections()
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((4, 16, 64)), F32)
    cots = [jnp.asarray(rng.standard_normal((4, 16, lin.out_features)), F32) for lin in lins]
    kernels = [lin.weight.value for lin in lins]

    def loss(x_, ks, gather):
        h = gather_shared_input(x_, config) if gather else x_
        return sum(jnp.sum(lin(h, w=k) * c) for lin, k, c in zip(lins, ks, cots, strict=True))

    with config.mesh:
        plain = jax.jit(jax.value_and_grad(lambda a, b: loss(a, b, False), argnums=(0, 1)))(x, kernels)
        gathered = jax.jit(jax.value_and_grad(lambda a, b: loss(a, b, True), argnums=(0, 1)))(x, kernels)
    (v_p, (dx_p, dk_p)), (v_g, (dx_g, dk_g)) = plain, gathered
    assert abs(float(v_g) - float(v_p)) <= 1e-5 * abs(float(v_p))
    assert dx_g.shape == x.shape and _rel(dx_g, dx_p) < 1e-5
    for a, b in zip(dk_g, dk_p, strict=True):
        assert a.shape == b.shape and _rel(a, b) < 1e-5


def test_no_tensor_parallelism_or_non_3d_input_is_a_no_op():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    x3 = jnp.ones((2, 8, 64), F32)
    x2 = jnp.ones((8, 64), F32)
    assert gather_shared_input(x3, _config(tp_size=1)) is x3
    assert gather_shared_input(x2, _config(tp_size=2)) is x2


def test_explicit_settings_win_over_the_automatic_rule():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    config = _config(tp_size=2)
    x = _sharded_input(config)
    for value in (True, False):
        config.gather_shared_projection_input = value
        for inference in (False, True):
            assert _decision(config, x, inference=inference) is value


def test_default_gathers_a_feature_sharded_input_in_inference_mode_only():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    config = _config(tp_size=2)
    config.gather_shared_projection_input = None
    x = _sharded_input(config)
    assert _decision(config, x, inference=False) is False  # training and every other non-inference trace: as today
    assert _decision(config, x, inference=True) is True  # serving prefill / mixed steps: hidden sharded over tp
    assert _decision(config, x, inference=True, decode=True) is False  # eSurge pure-decode layout: hidden replicated
    assert _decision(config, jnp.ones((16, 64), F32), inference=True) is False  # not a [B, S, D] input
    no_tp = _config(tp_size=1)
    no_tp.gather_shared_projection_input = None
    assert _decision(no_tp, _sharded_input(no_tp), inference=True) is False  # hidden not sharded: nothing to share


def test_a_config_without_the_field_takes_the_automatic_rule():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    config = _config(tp_size=2)
    bare = SimpleNamespace(mesh=config.mesh, runtime_sharding_resolver=config.runtime_sharding_resolver)
    x = _sharded_input(config)
    assert _decision(bare, x, inference=False) is False
    assert _decision(bare, x, inference=True) is True


def _glm_kda_model(mesh_dims):
    config = ed.Glm5NextTextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, num_experts_per_tok=2,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8,
        index_topk=8, index_head_dim=8, index_n_heads=2, index_kpool=4, layer_types=["linear_attention"] * 2,
        mlp_layer_types=["dense"] * 2, linear_head_dim=8, linear_num_heads=2, linear_conv_kernel_dim=4, hc_mult=2,
        kda_chunk_impl="kda_chunk", sharding_axis_dims=mesh_dims,
    )  # fmt: skip
    with config.mesh:
        return ed.Glm5NextForCausalLM(config=config, dtype=F32, param_dtype=F32, precision=HI, rngs=spx.Rngs(0))


def _set_gather(model, value):
    """Set the field on the model config and on every config object a sub-module holds."""
    configs = {id(model.config): model.config}
    for _, module in iter_module_search(model, spx.Module):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "gather_shared_projection_input"):
            configs[id(cfg)] = cfg
    for cfg in configs.values():
        cfg.gather_shared_projection_input = value


def _primitive_counts(fn, *args):
    counts = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts.update([eqn.primitive.name])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


def test_default_kda_stack_lowers_like_false_in_training_and_like_true_in_inference():
    """The KDA layers' consumption site follows the resolver (needs 4 devices: fsdp=2 x tp=2)."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices (fsdp=2 x tp=2)")
    model = _glm_kda_model((1, 1, 2, 1, 2, 1))
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 128, (2, 128)), jnp.int32)

    def lowered(value, inference):
        _set_gather(model, value)
        with model.config.mesh, set_inference_mode(inference):
            return _primitive_counts(lambda i: model(input_ids=i).logits, ids)

    def logits(value):
        _set_gather(model, value)
        with model.config.mesh, set_inference_mode(True):
            return np.asarray(jax.jit(lambda i: model(input_ids=i).logits)(ids))

    assert lowered(True, False) != lowered(False, False)  # the setting reaches the KDA layers
    assert lowered(None, False) == lowered(False, False)  # outside inference mode: today's program
    assert lowered(None, True) == lowered(True, True)  # inference mode: the gathered program
    assert lowered(None, True) != lowered(False, True)
    np.testing.assert_array_equal(logits(None), logits(True))
