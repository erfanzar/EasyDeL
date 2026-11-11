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

"""W8A8 DSA query/KV-latent and dense-MLP projections contract a tensor-parallel-gathered input in inference mode.

With the layer input sharded over ``tp`` on its feature axis, the partitioner splits the int8 contraction of these
projections over the idle expert axis and all-reduces int32 partial sums across it. Under the automatic
``gather_shared_projection_input`` rule (inference mode, feature-sharded input) the input is gathered over ``tp`` once
and each projection contracts locally: fewer int32 all-reduces, each projection's output unchanged bit for bit (int32
sums are exact), and training and decode traces unchanged. Model logits are not compared bit for bit: with the
gathered layout the float ops around the projections are partitioned differently, and their rounding changes.
"""

from __future__ import annotations

import collections
import re

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import decode_mode_specs
from easydel.layers.linears import ParallelLinearQuantized, gather_shared_input
from easydel.layers.quantization import QuantizationConfig
from easydel.utils import set_inference_mode
from easydel.utils.traversals import iter_module_search
from jax.sharding import NamedSharding
from spectrax import common_types as ct

DSA = "deepseek_sparse_attention"
VOCAB, SEQ = 128, 128
# The projections the W8A8 serving recipe quantizes in these layers: DSA query / KV-latent and the dense MLP.
PATTERN = r"(?:q_a_proj|kv_a_proj_with_mqa|gate_up_proj|down_proj)$"


def _model():
    config = ed.Glm5NextTextConfig(
        vocab_size=VOCAB, hidden_size=256, intermediate_size=512, moe_intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, num_experts_per_tok=2,
        kv_lora_rank=128, q_lora_rank=128, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8,
        index_topk=8, index_head_dim=8, index_n_heads=2, index_kpool=4, layer_types=[DSA, DSA],
        mlp_layer_types=["dense", "dense"], linear_head_dim=8, linear_num_heads=2, linear_conv_kernel_dim=4,
        hc_mult=2, sharding_axis_dims=(1, 1, 1, 2, 2, 1),
    )  # fmt: skip
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, rngs=spx.Rngs(0)
        )
        model = model.quantize(quantization_config=QuantizationConfig(dtype="w8a8", pattern=PATTERN), verbose=False)
    return model


def _set_gather(model, value):
    """Set the field on the model config and on every config object a sub-module holds."""
    configs = {id(model.config): model.config}
    for _, module in iter_module_search(model, spx.Module):
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "gather_shared_projection_input"):
            configs[id(cfg)] = cfg
    for cfg in configs.values():
        cfg.gather_shared_projection_input = value


def _ids():
    # One packed row, as eSurge's model step feeds the model.
    return jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (1, SEQ)), jnp.int32)


def _s32_all_reduces(hlo_text: str) -> int:
    count = 0
    for line in hlo_text.splitlines():
        match = re.search(r"=\s*(.*?)\s+all-reduce(?:-start)?\(", line)
        if match and "s32[" in match.group(1):
            count += 1
    return count


def _primitive_counts(model, ids, *, inference, decode=False):
    counts = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts.update([eqn.primitive.name])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    with model.config.mesh, set_inference_mode(inference), decode_mode_specs(decode):
        walk(jax.make_jaxpr(lambda i: model(input_ids=i).logits)(ids).jaxpr)
    return counts


def _compiled(model, ids):
    with model.config.mesh, set_inference_mode(True):
        compiled = jax.jit(lambda i: model(input_ids=i).logits).lower(ids).compile()
        return np.asarray(compiled(ids)), compiled.as_text()


@pytest.fixture(scope="module")
def model():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices (ep=2 x tp=2)")
    return _model()


def test_recipe_projections_are_w8a8(model):
    quantized = [path for path, module in iter_module_search(model, ParallelLinearQuantized)]
    assert len(quantized) == 8, quantized  # q_a, kv_a, gate_up, down in each of the two layers


def test_inference_prefill_drops_int32_all_reduces(model):
    ids = _ids()
    _set_gather(model, False)
    _, plain_hlo = _compiled(model, ids)
    _set_gather(model, None)
    gathered_logits, gathered_hlo = _compiled(model, ids)
    plain_n, gathered_n = _s32_all_reduces(plain_hlo), _s32_all_reduces(gathered_hlo)
    assert gathered_n < plain_n, (plain_n, gathered_n)
    assert np.isfinite(gathered_logits).all()


def test_gathered_input_leaves_each_dsa_w8a8_projection_bit_identical(model):
    """The int8 x int8 -> int32 contraction is exact: contracting the gathered input locally gives the same bits."""
    cfg = model.config
    mesh = getattr(cfg.mesh, "jax_mesh", cfg.mesh)
    shape = (1, SEQ, cfg.hidden_size)
    spec = cfg.runtime_sharding_resolver.resolve(dynamic_axes=ct.HiddenStateSharding, shape=shape)
    x = jax.device_put(
        jnp.asarray(np.random.default_rng(1).standard_normal(shape) * 2.0, jnp.bfloat16), NamedSharding(mesh, spec)
    )
    checked = []
    for path, linear in iter_module_search(model, ParallelLinearQuantized):
        name = ".".join(str(part) for part in path)
        if not name.endswith(("q_a_proj", "kv_a_proj_with_mqa")):
            continue
        with cfg.mesh, set_inference_mode(True):
            plain = np.asarray(jax.jit(lambda a, linear=linear: linear(a))(x))
            gathered = np.asarray(jax.jit(lambda a, linear=linear: linear(gather_shared_input(a, cfg)))(x))
        np.testing.assert_array_equal(gathered, plain, err_msg=name)
        checked.append(name)
    assert len(checked) == 4, checked  # q_a and kv_a in each of the two layers


def test_training_and_decode_traces_are_unchanged(model):
    ids = _ids()
    for inference, decode in ((False, False), (True, True)):
        _set_gather(model, False)
        off = _primitive_counts(model, ids, inference=inference, decode=decode)
        _set_gather(model, None)
        auto = _primitive_counts(model, ids, inference=inference, decode=decode)
        assert auto == off, (inference, decode)
