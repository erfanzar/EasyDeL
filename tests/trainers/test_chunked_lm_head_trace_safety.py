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

"""Regression tests: chunked LM-head projection inside nested JAX traced regions.

Validates that calling ``model.apply_lm_head`` (and related paths) from inside
``jax.lax.scan``, ``jax.lax.fori_loop``, and ``jax.checkpoint``, regardless of the gradient
checkpointing policy or tied-embedding configuration.

The root cause was that ``BaseCausalLMModule.__init__`` wrapped the LM-head
class with ``auto_remat``.  SpecTrax's ``nn.remat`` performs variable mutation
(``update_from_state``) in its ``split_inputs`` protocol, which fails when
called from a different JAX trace level (e.g., inside a ``lax.scan`` body
that runs under ``jax.grad``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

# Import easydel first so jax.distributed.initialize() runs before jax.devices()
import easydel  # noqa: F401

_MESH = jax.sharding.Mesh(jax.devices(), ("dp",))


def _make_model(tie: bool, gradient_checkpointing: str):
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=32,
        tie_word_embeddings=tie,
        gradient_checkpointing=gradient_checkpointing,
    )
    with _MESH:
        model = LlamaForCausalLM(
            config=cfg,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )
    return model


def _split(model):
    return model.split_module()


# ---------------------------------------------------------------------------
# Helpers for building loss functions that exercise the chunked paths
# ---------------------------------------------------------------------------

B, L, CHUNK = 2, 8, 4


def _loss_make_lm_head_fn_scan(graphdef, graphother, model):
    """make_lm_head_fn inside lax.scan + jax.checkpoint + jax.grad."""
    H = model.config.hidden_size

    def loss_fn(params):
        other = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
            graphother,
        )
        mdl = spx.bind(graphdef, params.merge(other, copy=True))
        outputs = mdl(input_ids=jnp.ones((B, L), dtype=jnp.int32), apply_lm_head=False)
        hidden = outputs.last_hidden_state

        n_chunks = L // CHUNK
        h_chunks = hidden.reshape(B, n_chunks, CHUNK, H).transpose(1, 0, 2, 3)

        # Obtain the trace-safe fn ONCE before the scan.
        lm_head_fn = mdl.make_lm_head_fn()

        @jax.checkpoint
        def _chunk_fn(h):
            return lm_head_fn(h)

        def _scan_body(carry, chunk_h):
            return carry + jnp.sum(_chunk_fn(chunk_h)), None

        total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), h_chunks)
        return total

    return loss_fn


def _loss_distillation_chunked(graphdef, graphother, model):
    """chunked_distillation_loss with model.make_lm_head_fn."""
    from easydel.trainers.distillation_trainer._fn import chunked_distillation_loss

    def loss_fn(params):
        other = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
            graphother,
        )
        mdl = spx.bind(graphdef, params.merge(other, copy=True))
        outputs = mdl(input_ids=jnp.ones((B, L), dtype=jnp.int32), apply_lm_head=False)
        student_h = outputs.last_hidden_state
        teacher_h = jax.lax.stop_gradient(student_h)

        total, _ = chunked_distillation_loss(
            student_hidden=student_h,
            teacher_hidden=teacher_h,
            student_lm_head_fn=mdl.make_lm_head_fn(),
            teacher_lm_head_fn=mdl.make_lm_head_fn(),
            attention_mask=jnp.ones((B, L), dtype=jnp.int32),
            labels=jnp.ones((B, L), dtype=jnp.int32),
            use_hard_labels=True,
            temperature=2.0,
            alpha=0.9,
            chunk_size=CHUNK,
        )
        return total

    return loss_fn


def _loss_logprob_utils(graphdef, graphother, model):
    """compute_sequence_scores_from_hidden_states path."""
    from easydel.trainers._logprob_utils import compute_sequence_scores_from_hidden_states

    def loss_fn(params):
        other = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
            graphother,
        )
        mdl = spx.bind(graphdef, params.merge(other, copy=True))
        outputs = mdl(input_ids=jnp.ones((B, L), dtype=jnp.int32), apply_lm_head=False)
        hidden = outputs.last_hidden_state
        logp_sums, _, _ = compute_sequence_scores_from_hidden_states(
            model=mdl,
            hidden_states=hidden,
            labels=jnp.ones((B, L), dtype=jnp.int32),
            loss_mask=jnp.ones((B, L), dtype=jnp.bool_),
            token_chunk_size=CHUNK,
            vocab_chunk_size=None,
        )
        return jnp.sum(logp_sums)

    return loss_fn


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

_LOSS_FNS = [
    ("make_lm_head_fn_scan", _loss_make_lm_head_fn_scan),
    ("distillation_chunked", _loss_distillation_chunked),
    ("logprob_utils", _loss_logprob_utils),
]


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("gc", ["mlp_notsaveable", "nothing_saveable"], ids=["mlp_notsave", "nothing_save"])
@pytest.mark.parametrize("path_name,factory", _LOSS_FNS, ids=[n for n, _ in _LOSS_FNS])
def test_chunked_lm_head_no_trace_error(tie, gc, path_name, factory):
    """Chunked LM-head projection must not raise TraceContextError."""
    model = _make_model(tie=tie, gradient_checkpointing=gc)
    graphdef, graphstate, graphother = _split(model)
    loss_fn = factory(graphdef, graphother, model)

    with _MESH:
        grads = jax.grad(loss_fn)(graphstate)

    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    assert jnp.isfinite(grad_norm), f"Non-finite grad norm: {float(grad_norm)}"
    assert float(grad_norm) > 0.0, "Zero gradients — projection is likely broken"


def test_make_lm_head_fn_matches_compute_lm_logits():
    """make_lm_head_fn must produce the same logits as compute_lm_logits."""
    model = _make_model(tie=True, gradient_checkpointing="mlp_notsaveable")
    hidden = jnp.ones((B, L, model.config.hidden_size), dtype=jnp.float32)

    with _MESH:
        lm_fn = model.make_lm_head_fn()
        logits_via_fn = lm_fn(hidden)
        logits_via_compute = model.compute_lm_logits(hidden)

    assert jnp.allclose(logits_via_fn, logits_via_compute, atol=1e-5), (
        f"make_lm_head_fn and compute_lm_logits diverge: "
        f"max diff = {float(jnp.max(jnp.abs(logits_via_fn - logits_via_compute)))}"
    )


def test_lm_head_is_wrapped_with_remat():
    """LM head should still be wrapped with nn.remat for memory savings."""
    model = _make_model(tie=False, gradient_checkpointing="mlp_notsaveable")
    head = model.get_task_head()
    assert getattr(head.forward, "_easydel_auto_remat_wrapped", False), (
        "LM head forward should be wrapped with auto_remat"
    )


def test_make_lm_head_fn_bypasses_remat():
    """make_lm_head_fn must use native_forward, not __call__."""
    model = _make_model(tie=False, gradient_checkpointing="mlp_notsaveable")
    model.make_lm_head_fn()
    # The closure should reference native_forward, not the wrapped __call__
    hidden = jnp.ones((B, L, model.config.hidden_size), dtype=jnp.float32)
    with _MESH:
        # If this succeeds inside grad + scan, native_forward bypass works
        graphdef, graphstate, graphother = _split(model)

        def loss_fn(params):
            other = jax.tree_util.tree_map(
                lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                graphother,
            )
            mdl = spx.bind(graphdef, params.merge(other, copy=True))
            lm_fn = mdl.make_lm_head_fn()

            def _scan_body(carry, _):
                return carry + jnp.sum(lm_fn(hidden)), None

            total, _ = jax.lax.scan(_scan_body, jnp.float32(0.0), None, length=2)
            return total

        grads = jax.grad(loss_fn)(graphstate)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    assert jnp.isfinite(grad_norm) and float(grad_norm) > 0
