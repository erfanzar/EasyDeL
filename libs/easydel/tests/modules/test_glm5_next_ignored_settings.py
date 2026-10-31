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

"""GLM-5-Next settings reach the code they configure.

- ``scan_layers``: the scanned stack gives the Python-traced stack's loss and gradients and lowers one body per run of
  identical layers (``layer_idx`` does not split scan segments).
- ``kda_chunk_impl`` / ``kda_chunk_precision`` reach ``KernelDeltaAttnOp``.
- Layers hold an RNG only when attention dropout draws from it: a model-wide ``Rngs`` held by every layer would force
  the scan onto its sequential-carry lowering.
"""

from __future__ import annotations

import collections

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx

KDA, DSA = "linear_attention", "deepseek_sparse_attention"
VOCAB, SEQ = 128, 128


def _config(layer_types, mlp_layer_types=None, **overrides):
    n = len(layer_types)
    overrides.setdefault("sharding_axis_dims", (1, 1, 1, 1, 1, 1))
    return ed.Glm5NextTextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=n,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=4, num_experts_per_tok=2,
        kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8,
        index_topk=8, index_head_dim=8, index_n_heads=2, index_kpool=4, layer_types=list(layer_types),
        mlp_layer_types=list(mlp_layer_types or ["dense"] * n), linear_head_dim=8, linear_num_heads=2,
        linear_conv_kernel_dim=4, hc_mult=2, **overrides,
    )  # fmt: skip


def _model(config):
    with config.mesh:
        return ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


def _ids():
    return jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (2, SEQ)), jnp.int32)


def _loss_and_grads(model, ids):
    gdef, state = spx.export(model)

    def loss(st):
        logits = spx.bind(gdef, st)(input_ids=ids).logits.astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:]).mean()

    with model.config.mesh, jax.default_matmul_precision("highest"):
        value, grads = jax.jit(jax.value_and_grad(loss, allow_int=True))(state)
    leaves = [np.asarray(g, np.float64) for g in jax.tree.leaves(grads) if jnp.issubdtype(g.dtype, jnp.inexact)]
    return float(value), leaves


def _assert_same_training_signal(got, want, *, loss_rtol, grad_rtol):
    (loss_got, grads_got), (loss_want, grads_want) = got, want
    assert np.isfinite(loss_got) and abs(loss_got - loss_want) <= loss_rtol * abs(loss_want)
    assert len(grads_got) == len(grads_want)
    for a, b in zip(grads_got, grads_want, strict=True):
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - b) <= grad_rtol * np.linalg.norm(b) + 1e-7


def _walk(jaxpr, visit):
    for eqn in jaxpr.eqns:
        visit(eqn)
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    _walk(sub, visit)


def _forward_jaxpr(model):
    with model.config.mesh:
        return jax.make_jaxpr(lambda i: model(input_ids=i).logits)(_ids()).jaxpr


def _primitive_counts(model):
    counts = collections.Counter()
    _walk(_forward_jaxpr(model), lambda eqn: counts.update([eqn.primitive.name]))
    return counts


@pytest.mark.parametrize(
    "layer_types,mlp_layer_types",
    [
        ([KDA] * 4, None),
        ([KDA, KDA, DSA] * 2, ["dense", "sparse", "sparse", "sparse", "sparse", "sparse"]),
    ],
    ids=["kda", "kda_dsa_moe"],
)
def test_scan_layers_matches_the_python_trace(layer_types, mlp_layer_types):
    model = _model(_config(layer_types, mlp_layer_types))
    ids = _ids()
    model.config.scan_layers = False
    traced = _loss_and_grads(model, ids)
    model.config.scan_layers = True
    scanned = _loss_and_grads(model, ids)
    _assert_same_training_signal(scanned, traced, loss_rtol=1e-5, grad_rtol=1e-4)


def test_scan_layers_lowers_one_body_for_identical_layers():
    """Scanned, the program size does not grow with depth (``layer_idx`` does not split segments); traced, it does."""
    sizes = {}
    for scan in (False, True):
        for depth in (2, 6):
            model = _model(_config([KDA] * depth))
            model.config.scan_layers = scan
            sizes[scan, depth] = sum(_primitive_counts(model).values())
    assert sizes[True, 2] == sizes[True, 6], sizes
    assert sizes[False, 6] > sizes[False, 2], sizes
    assert sizes[True, 6] < sizes[False, 6], sizes


def test_kda_chunk_impl_reaches_the_kda_operation():
    ids = _ids()
    scan = _model(_config([KDA, KDA]))
    fused = _model(_config([KDA, KDA], kda_chunk_impl="kda_chunk"))
    for a, b in zip(jax.tree.leaves(spx.export(scan)[1]), jax.tree.leaves(spx.export(fused)[1]), strict=True):
        if jnp.issubdtype(a.dtype, jnp.inexact):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    _assert_same_training_signal(_loss_and_grads(fused, ids), _loss_and_grads(scan, ids), loss_rtol=1e-4, grad_rtol=1e-3)
    # The setting changes the lowered program (the fused kda_chunk path replaces the per-channel chunk scan; how
    # many Pallas launches each lowers to depends on the device count, so compare whole programs).
    assert _primitive_counts(fused) != _primitive_counts(scan)


def test_kda_chunk_precision_reaches_the_kernel():
    def dot_precisions(model):
        found = collections.Counter()

        def visit(eqn):
            if eqn.primitive.name == "dot_general":
                found.update([repr(eqn.params.get("precision"))])

        _walk(_forward_jaxpr(model), visit)
        return found

    highest = _model(_config([KDA], kda_chunk_impl="kda_chunk"))
    default = _model(_config([KDA], kda_chunk_impl="kda_chunk", kda_chunk_precision="default"))
    assert dot_precisions(highest) != dot_precisions(default)
    ids = _ids()
    _assert_same_training_signal(
        _loss_and_grads(default, ids), _loss_and_grads(highest, ids), loss_rtol=1e-2, grad_rtol=5e-2
    )


def test_kda_gate_lower_bound_reaches_the_kernel():
    """The safe gate's ``linear_lower_bound`` switches the fused kernel to the factored (MXU) pairwise decay.

    ``operation_configs["kda_chunk"]["pairwise"] = "dense"`` opts out, so the two models share weights and differ
    only in whether the bound is used: the lowered programs differ and the training signal agrees.
    """
    ids = _ids()
    factored = _model(_config([KDA, KDA], kda_chunk_impl="kda_chunk"))
    dense = _model(
        _config([KDA, KDA], kda_chunk_impl="kda_chunk", operation_configs={"kda_chunk": {"pairwise": "dense"}})
    )
    assert factored.config.linear_lower_bound == -5.0
    assert _primitive_counts(factored) != _primitive_counts(dense)
    _assert_same_training_signal(
        _loss_and_grads(factored, ids), _loss_and_grads(dense, ids), loss_rtol=1e-5, grad_rtol=1e-4
    )


def test_gather_shared_projection_input_reaches_the_kda_layer():
    """``gather_shared_projection_input`` gathers the KDA layer input over tensor parallelism once.

    On a tp=2 mesh: same weights (the flag adds no parameter), the lowered forward changes (the gather's custom VJP
    appears once per KDA layer) and the training signal matches the plain layer. The six projections keep their own
    dots (same ``dot_general`` count).
    """
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices (fsdp=2 x tp=2)")
    mesh_dims = (1, 1, 2, 1, 2, 1)
    ids = _ids()
    plain = _model(_config([KDA, KDA], kda_chunk_impl="kda_chunk", sharding_axis_dims=mesh_dims))
    gathered = _model(
        _config(
            [KDA, KDA], kda_chunk_impl="kda_chunk", sharding_axis_dims=mesh_dims, gather_shared_projection_input=True
        )
    )
    assert gathered.config.gather_shared_projection_input is True
    for a, b in zip(jax.tree.leaves(spx.export(plain)[1]), jax.tree.leaves(spx.export(gathered)[1]), strict=True):
        if jnp.issubdtype(a.dtype, jnp.inexact):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    counts_plain, counts_gathered = _primitive_counts(plain), _primitive_counts(gathered)
    assert counts_gathered != counts_plain
    assert counts_gathered["dot_general"] == counts_plain["dot_general"]
    _assert_same_training_signal(
        _loss_and_grads(gathered, ids), _loss_and_grads(plain, ids), loss_rtol=1e-5, grad_rtol=1e-4
    )


@pytest.mark.parametrize("field,value", [("kda_chunk_impl", "twophase"), ("kda_chunk_precision", "bf16")])
def test_invalid_kda_chunk_settings_raise(field, value):
    with pytest.raises(ValueError, match=field):
        _config([KDA], **{field: value})


def test_layers_hold_an_rng_only_when_attention_dropout_draws_from_it():
    def rng_paths(model):
        return [path for path, var in spx.iter_variables(model.model.layers) if getattr(var, "kind", None) == "rng"]

    assert rng_paths(_model(_config([KDA, DSA]))) == []
    with_dropout = rng_paths(_model(_config([KDA, DSA], attention_dropout=0.1)))
    assert with_dropout, "attention dropout needs the RNG on the DSA attention performer"
    assert all("attention_performer" in str(path) for path in with_dropout), with_dropout
