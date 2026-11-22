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

"""Non-mocked regression tests for Subsection 2 review fixes (S2-C1..S2-L5).

Covers sampling/speculative primitives, logits processors, sampling parameters,
eSurge evaluation adapter, and (integrated from subsystem suites) distributed
serving and HTTP/API server regressions.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import RaggedPagesMetadata, TransformerCache
from easydel.inference.evaluations.esurge_eval import (
    _strip_empty_reasoning_scaffold,
    eSurgeLMEvalAdapter,
)
from easydel.inference.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    FrequencyPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
from easydel.inference.sampling_funcs import (
    dynamic_sample_tokens,
    sample_top_p_efficient,
    vmaped_sample_top_p_efficient,
)
from easydel.inference.sampling_params import (
    GuidedDecodingParams,
    JitableSamplingParams,
    SamplingParams,
    SamplingType,
)
from easydel.inference.sampling_processors import (
    apply_sampling_constraints,
    apply_sparse_sampling_constraints,
)
from easydel.inference.speculative import (
    Gemma4AssistantDrafter,
    Qwen3_5MTPDrafter,
    accept_or_reject,
    resample_rejected,
)

pytestmark = pytest.mark.cpu_ok


# ---------------------------------------------------------------------------
# Minimal real SpectraX modules for Qwen3_5MTPDrafter & Gemma4AssistantDrafter
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class _DummyPagedCacheView:
    """Minimal JAX-pytree paged cache view tracking write steps without dense views."""

    def __init__(self, kv_pages: jax.Array, write_count: jax.Array, *, num_pages: int = 8, page_size: int = 4):
        self.kv_pages = kv_pages
        self.write_count = write_count
        self.metadata = SimpleNamespace(num_pages=num_pages, page_size=page_size)

    def tree_flatten(self):
        return (self.kv_pages, self.write_count), (self.metadata.num_pages, self.metadata.page_size)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        num_pages, page_size = aux_data
        return cls(children[0], children[1], num_pages=num_pages, page_size=page_size)


class _TinyEmbed(spx.Module):
    def __init__(self, vocab: int, hidden: int, *, seed: int):
        self.weight = spx.Parameter(jax.random.normal(jax.random.PRNGKey(seed), (vocab, hidden), dtype=jnp.float32) * 0.25)

    def forward(self, ids: jax.Array) -> jax.Array:
        return self.weight.value[ids]


class _TinyLinear(spx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, seed: int):
        self.weight = spx.Parameter(jax.random.normal(jax.random.PRNGKey(seed), (in_dim, out_dim), dtype=jnp.float32) * 0.25)

    def forward(self, x: jax.Array, w: jax.Array | None = None) -> jax.Array:
        kernel = self.weight.value if w is None else w
        return x @ kernel


class _TinyNorm(spx.Module):
    def __init__(self, hidden: int):
        self.scale = spx.Parameter(jnp.ones((hidden,), dtype=jnp.float32))

    def forward(self, x: jax.Array) -> jax.Array:
        return x * self.scale.value


class _TinyMTPLayer(spx.Module):
    def __init__(self, hidden: int, *, paged_only: bool = False, seed: int):
        self.layer_idx = 0
        self.paged_only = paged_only
        self.proj = _TinyLinear(hidden, hidden, seed=seed)

    def forward(self, hidden_states, mask_info, pos, mode, cache_view, cache_metadata, frequencies):
        del mask_info, mode, frequencies
        if self.paged_only and not isinstance(cache_view, _DummyPagedCacheView):
            raise AssertionError("Paged-only MTP layer must not be invoked with dense/None cache_view")
        out = hidden_states + 0.1 * self.proj(hidden_states)
        if isinstance(cache_view, _DummyPagedCacheView):
            # Incorporate metadata context_lens into output slightly and advance write_count
            # to prove multi-step paged cache_view + cache_metadata threading.
            ctx_bias = 0.0
            if cache_metadata is not None and getattr(cache_metadata, "context_lens", None) is not None:
                ctx_bias = jnp.mean(cache_metadata.context_lens.astype(jnp.float32)) * 1e-4
            out = out + ctx_bias
            new_view = _DummyPagedCacheView(
                cache_view.kv_pages + 1.0,
                cache_view.write_count + 1,
                num_pages=cache_view.metadata.num_pages,
                page_size=cache_view.metadata.page_size,
            )
            return out, new_view
        if cache_view is not None:
            new_idx = cache_view.indexes + hidden_states.shape[1]
            return out, cache_view.replace(indexes=new_idx)
        return out, cache_view


class _TinyMTPBlock(spx.Module):
    def __init__(self, hidden: int, *, paged_only: bool = False, seed: int):
        self.pre_fc_norm_embedding = _TinyNorm(hidden)
        self.pre_fc_norm_hidden = _TinyNorm(hidden)
        self.fc = _TinyLinear(2 * hidden, hidden, seed=seed + 10)
        self.layers = spx.nn.ModuleList([_TinyMTPLayer(hidden, paged_only=paged_only, seed=seed + 20)])
        self.norm = _TinyNorm(hidden)


class _TinyTargetModel(spx.Module):
    def __init__(
        self,
        *,
        vocab: int = 16,
        hidden: int = 8,
        paged_only: bool = False,
        tie_via_feature: bool = False,
        seed: int = 0,
    ):
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("dp",))
        self.mesh = mesh
        self.dtype = jnp.float32
        self.config = SimpleNamespace(
            vocab_size=vocab,
            hidden_size=hidden,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=hidden,
            mtp_num_hidden_layers=1,
            max_position_embeddings=64,
            mesh=mesh,
        )
        self.embed = _TinyEmbed(vocab, hidden, seed=seed + 1)
        self.lm_head = _TinyLinear(hidden, vocab, seed=seed + 2)
        self.mtp = _TinyMTPBlock(hidden, paged_only=paged_only, seed=seed + 3)
        self.model = SimpleNamespace(frequencies=None, get_embedding=lambda: self.embed)
        if tie_via_feature:
            self._tie_embeddings_feature = SimpleNamespace(tie=True)

    def has_mtp(self) -> bool:
        return True

    def get_input_embeddings(self):
        return self.embed

    def get_lm_head(self):
        return self.lm_head

    def apply_lm_head(self, h: jax.Array) -> jax.Array:
        feature = getattr(self, "_tie_embeddings_feature", None)
        w = self.embed.weight.value.T if getattr(feature, "tie", False) else None
        return self.lm_head(h, w=w)


# ---------------------------------------------------------------------------
# S2-C5, S2-C6, S2-H1, S2-M8, S2-L5: Speculative Qwen3_5MTPDrafter & Gemma4
# ---------------------------------------------------------------------------


def test_s2_c5_and_s2_m8_spec_step_arg_order_constraints_sampled_and_k_mismatch():
    vocab, hidden, n_rows = 16, 8, 2
    model = _TinyTargetModel(vocab=vocab, hidden=hidden, paged_only=False, seed=0)
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=2, use_cache=True, cache_length=32)

    # Constant logits from project_fn so we know exact unconstrained vs constrained picks:
    # row 0 (rows 0..3) prefers token 3 (logit 5.0) over token 7 (logit 4.0);
    # row 1 (rows 4..7) prefers token 2 (logit 5.0) over token 9 (logit 4.0).
    base_logits = jnp.zeros((8, vocab), dtype=jnp.float32)
    base_logits = base_logits.at[0:4, 3].set(5.0).at[0:4, 7].set(4.0)
    base_logits = base_logits.at[4:8, 2].set(5.0).at[4:8, 9].set(4.0)

    def project_fn(h):
        return base_logits[: h.shape[0]]

    gathered_hidden = jnp.ones((8, hidden), dtype=jnp.float32)
    row_offsets = np.array([0, 4], dtype=np.int32)
    draft_tokens = np.array([[7, 7], [9, 9]], dtype=np.int32)
    draft_counts = np.array([2, 2], dtype=np.int32)
    seed_pos_base = np.array([4, 4], dtype=np.int32)
    committed_base = np.array([0, 0], dtype=np.int32)
    fresh_mask = np.array([True, True], dtype=bool)
    collected_mask = np.array([True, True], dtype=bool)

    # Ban token 3 on row 0 and token 2 on row 1; also pass a paged cache_view/metadata
    # simultaneously to exercise the previously transposed trailing 5 args (S2-C5).
    paged_view = _DummyPagedCacheView(jnp.zeros((2, 4), dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32))
    paged_meta = RaggedPagesMetadata(
        pages_tables=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        context_lens=jnp.asarray([4, 4], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        num_seqs=jnp.asarray([2], dtype=jnp.int32),
    )
    bad_words = np.zeros((n_rows, vocab), dtype=bool)
    bad_words[0, 3] = True
    bad_words[1, 2] = True
    allowed = np.zeros((n_rows, vocab), dtype=bool)
    allowed[0, [3, 7]] = True
    allowed[1, [2, 9]] = True

    # S2-C5 + S2-M8: constraints apply to arg_mat, bonus, resolve_rows, AND next drafts.
    arg_mat, accepted, _seed_pos, spec_out = drafter.spec_step(
        gathered_hidden=gathered_hidden,
        row_offsets=row_offsets,
        draft_tokens=draft_tokens,
        draft_counts=draft_counts,
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=2,
        cache_view=paged_view,
        cache_metadata=paged_meta,
        allowed_token_ids_mask=jnp.asarray(allowed),
        bad_words_token_ids=jnp.asarray(bad_words),
        project_fn=project_fn,
    )
    # With token 3 banned on row 0 and token 2 banned on row 1, constrained argmax is 7 and 9!
    np.testing.assert_array_equal(np.asarray(arg_mat[0]), [7, 7, 7])
    np.testing.assert_array_equal(np.asarray(arg_mat[1]), [9, 9, 9])
    np.testing.assert_array_equal(np.asarray(accepted), [2, 2])
    # Next drafts in spec_out[:, k+2:] must ALSO respect allowed & bad_words -> only 7 on row 0, 9 on row 1!
    next_drafts = np.asarray(spec_out[:, 4:])
    np.testing.assert_array_equal(next_drafts[0], [7, 7])
    np.testing.assert_array_equal(next_drafts[1], [9, 9])

    # S2-M8: explicit non-greedy params + sampled bonus honors sampled instead of hardcoding greedy=1.
    params = np.zeros((n_rows, 4), dtype=np.float32)
    params[0, 0] = 0.0  # row 0 greedy
    params[1, 0] = 1.0  # row 1 sampled
    params[:, 1] = 1.0
    sampled = np.array([15, 14], dtype=np.int32)
    _, accepted_s, _, spec_out_s = drafter.spec_step(
        gathered_hidden=gathered_hidden,
        row_offsets=row_offsets,
        draft_tokens=np.array([[7, 7], [9, 9]], dtype=np.int32),
        draft_counts=np.array([0, 0], dtype=np.int32),  # 0 drafts -> emits only bonus token!
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=2,
        allowed_token_ids_mask=jnp.asarray(allowed),
        bad_words_token_ids=jnp.asarray(bad_words),
        sampled=jnp.asarray(sampled),
        params=jnp.asarray(params),
        rng_key=jax.random.PRNGKey(42),
        project_fn=project_fn,
    )
    np.testing.assert_array_equal(np.asarray(accepted_s), [0, 0])
    assert int(spec_out_s[0, 0]) == 7  # row 0 greedy bonus
    assert int(spec_out_s[1, 0]) == 14  # row 1 sampled bonus

    # S2-M8: k_pad != k_live must not crash pack_rows (both k_pad < k_live and k_pad > k_live).
    _, _, _, spec_grow = drafter.spec_step(
        gathered_hidden=gathered_hidden,
        row_offsets=row_offsets,
        draft_tokens=np.array([[7], [9]], dtype=np.int32),  # k_pad = 1
        draft_counts=np.array([1, 1], dtype=np.int32),
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=3,  # k_live = 3
        project_fn=project_fn,
    )
    assert spec_grow.shape == (n_rows, 2 * 3 + 2)

    _, _, _, spec_shrink = drafter.spec_step(
        gathered_hidden=gathered_hidden,
        row_offsets=row_offsets,
        draft_tokens=np.array([[3, 3, 3], [2, 2, 2]], dtype=np.int32),  # k_pad = 3
        draft_counts=np.array([3, 3], dtype=np.int32),
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=1,  # k_live = 1
        project_fn=project_fn,
    )
    assert spec_shrink.shape == (n_rows, 2 * 3 + 2)
    np.testing.assert_array_equal(np.asarray(spec_shrink[:, :4]), [[3, 3, 3, 3], [2, 2, 2, 2]])


def test_s2_c6_paged_cache_view_multistep_and_warmup_no_dense_crash():
    """Paged-only MTP head must run multi-step k>=2 draft_batched & spec_step without touching dense cache."""
    vocab, hidden, n_rows = 16, 8, 2
    model = _TinyTargetModel(vocab=vocab, hidden=hidden, paged_only=True, seed=7)
    paged_view = _DummyPagedCacheView(jnp.zeros((2, 4), dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32))
    paged_meta = RaggedPagesMetadata(
        pages_tables=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        context_lens=jnp.asarray([3, 5], dtype=jnp.int32),
        query_start_loc=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        num_seqs=jnp.asarray([2], dtype=jnp.int32),
    )

    for use_cache in (True, False):
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=3, use_cache=use_cache, cache_length=32)
        drafts = drafter.draft_batched(
            seed_tokens=np.array([1, 2], dtype=np.int32),
            seed_hidden=jnp.ones((n_rows, 1, hidden), dtype=jnp.float32),
            seed_positions=np.array([3, 5], dtype=np.int32),
            committed_lens=np.array([0, 0], dtype=np.int32),
            collected_mask=np.array([True, True], dtype=bool),
            num_draft_tokens=3,
            cache_view=paged_view,
            cache_metadata=paged_meta,
        )
        assert drafts.shape == (n_rows, 3)
        if use_cache:
            assert isinstance(drafter._mtp_cache, TransformerCache)
        else:
            assert drafter._mtp_cache is None

        arg_mat, accepted, seed_pos, spec_out = drafter.spec_step(
            gathered_hidden=jnp.ones((8, hidden), dtype=jnp.float32),
            row_offsets=np.array([0, 4], dtype=np.int32),
            draft_tokens=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            draft_counts=np.array([3, 3], dtype=np.int32),
            seed_pos_base=np.array([3, 5], dtype=np.int32),
            committed_base=np.array([0, 0], dtype=np.int32),
            fresh_mask=np.array([True, True], dtype=bool),
            collected_mask=np.array([True, True], dtype=bool),
            num_draft_tokens=3,
            cache_view=paged_view,
            cache_metadata=paged_meta,
        )
        assert arg_mat.shape == (n_rows, 4)
        assert accepted.shape == (n_rows,)
        assert seed_pos.shape == (n_rows,)
        assert spec_out.shape == (n_rows, 2 * 3 + 2)
        if use_cache:
            assert isinstance(drafter._mtp_cache, TransformerCache)


def test_s2_h1_adaptive_and_batched_constraints_traced_and_2d_safe():
    vocab, hidden = 16, 8
    model = _TinyTargetModel(vocab=vocab, hidden=hidden, seed=3)
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=2, use_cache=True, cache_length=32)

    seed_hidden = jnp.ones((1, 1, hidden), dtype=jnp.float32)

    # First call with NO constraints, then call with Mask A (only token 5), then Mask B (only token 11).
    d0, c0 = drafter.draft_adaptive(1, seed_hidden, 0, k_max=2, conf_threshold=0.0)
    assert int(c0) == 2 and d0.shape == (2,)

    allow_a = jnp.zeros((1, vocab), dtype=jnp.bool_).at[0, 5].set(True)
    allow_b = jnp.zeros((1, vocab), dtype=jnp.bool_).at[0, 11].set(True)
    da, ca = drafter.draft_adaptive(1, seed_hidden, 0, k_max=2, conf_threshold=0.0, allowed_token_ids_mask=allow_a)
    db, cb = drafter.draft_adaptive(1, seed_hidden, 0, k_max=2, conf_threshold=0.0, allowed_token_ids_mask=allow_b)
    assert int(ca) == 2 and int(cb) == 2
    np.testing.assert_array_equal(np.asarray(da), [5, 5])
    np.testing.assert_array_equal(np.asarray(db), [11, 11])

    # Batched (B=3 > 1) with 2D (3, vocab) constraints: must not raise 3D/2D broadcast ValueError.
    allow_batch = jnp.zeros((3, vocab), dtype=jnp.bool_)
    allow_batch = allow_batch.at[0, 4].set(True).at[1, 8].set(True).at[2, 12].set(True)
    drafts = drafter.draft_batched(
        seed_tokens=np.array([1, 2, 3], dtype=np.int32),
        seed_hidden=jnp.ones((3, 1, hidden), dtype=jnp.float32),
        seed_positions=np.array([0, 0, 0], dtype=np.int32),
        committed_lens=np.array([0, 0, 0], dtype=np.int32),
        collected_mask=np.array([True, True, True], dtype=bool),
        num_draft_tokens=2,
        allowed_token_ids_mask=allow_batch,
    )
    np.testing.assert_array_equal(np.asarray(drafts), [[4, 4], [8, 8], [12, 12]])


def test_s2_l5_speculative_polish_and_target_lifecycle_hooks():
    # _tie_embeddings_feature detection in _batched_bundle_graph_and_states
    tied_model = _TinyTargetModel(vocab=16, hidden=8, tie_via_feature=True, seed=11)
    drafter = Qwen3_5MTPDrafter(tied_model, num_draft_tokens=2, use_cache=True, cache_length=16)
    _, _, _, tie_flag = drafter._batched_bundle_graph_and_states()
    assert tie_flag is True

    # k == 0 in draft_batched & verify_draft_batched does not crash jnp.stack
    d_zero = drafter.draft_batched(
        seed_tokens=np.array([1, 2], dtype=np.int32),
        seed_hidden=jnp.ones((2, 1, 8), dtype=jnp.float32),
        seed_positions=np.array([0, 0], dtype=np.int32),
        committed_lens=np.array([0, 0], dtype=np.int32),
        collected_mask=np.array([True, True], dtype=bool),
        num_draft_tokens=0,
    )
    assert d_zero.shape == (2, 0)
    _, _, _, vd_zero = drafter.verify_draft_batched(
        gathered_hidden=jnp.ones((4, 8), dtype=jnp.float32),
        row_offsets=np.array([0, 2], dtype=np.int32),
        draft_tokens=np.array([[1], [2]], dtype=np.int32),
        draft_counts=np.array([1, 1], dtype=np.int32),
        seed_pos_base=np.array([0, 0], dtype=np.int32),
        committed_base=np.array([0, 0], dtype=np.int32),
        fresh_mask=np.array([True, True], dtype=bool),
        collected_mask=np.array([True, True], dtype=bool),
        num_draft_tokens=0,
    )
    assert vd_zero.shape == (2, 0)

    # Lifecycle hooks: update_target_model & release_target_model
    updated_model = _TinyTargetModel(vocab=16, hidden=8, tie_via_feature=True, seed=99)
    drafter.update_target_model(updated_model)
    assert drafter.model is updated_model
    drafter.release_target_model()
    assert drafter.model is None
    assert drafter._mtp_cache is None

    # Gemma4AssistantDrafter num_draft_tokens default/kwarg, greedy skips log_softmax, lifecycle hooks
    from typing import NamedTuple

    class _AssistantOut(NamedTuple):
        top_logits: jax.Array | None
        top_token_ids: jax.Array | None
        logits: jax.Array
        backbone_hidden_state: jax.Array

    class _TinyAssistant(spx.Module):
        def __init__(self):
            self.config = SimpleNamespace(backbone_hidden_size=8)
            self.proj = _TinyLinear(8, 16, seed=5)

        def forward(self, backbone_hidden_states, target_token_embeds, **_kw):
            logits = self.proj(backbone_hidden_states + target_token_embeds)
            return _AssistantOut(
                top_logits=None,
                top_token_ids=None,
                logits=logits,
                backbone_hidden_state=backbone_hidden_states,
            )

    embed = _TinyEmbed(16, 8, seed=6)
    gemma_drafter = Gemma4AssistantDrafter(_TinyAssistant(), embed, num_draft_tokens=3)
    assert gemma_drafter.num_draft_tokens == 3
    step = gemma_drafter.draft(
        input_ids=jnp.asarray([[1]], dtype=jnp.int32),
        target_hidden_states=jnp.ones((1, 1, 8), dtype=jnp.float32),
        return_full_log_probs=False,
        sample=False,
    )
    assert step.log_probs is None and step.full_log_probs is None
    gemma_drafter.update_target_model(updated_model)
    assert gemma_drafter.target_embed is updated_model.embed
    gemma_drafter.release_target_model()
    assert gemma_drafter.target_embed is None


# ---------------------------------------------------------------------------
# S2-H17: accept_or_reject & resample_rejected
# ---------------------------------------------------------------------------


def test_s2_h17_accept_or_reject_and_resample_rejected():
    key = jax.random.PRNGKey(0)
    # (-inf, -inf), (-1e10, -1e10), (-inf, -1.0) must reject (0) with no NaN;
    # (-1.0, -inf) and huge positive ratio (-0.1, -500.0) must accept (1).
    draft_lp = jnp.asarray([-jnp.inf, -1e10, -1.0, -jnp.inf, -500.0], dtype=jnp.float32)
    target_lp = jnp.asarray([-jnp.inf, -1e10, -jnp.inf, -1.0, -0.1], dtype=jnp.float32)
    accepted = accept_or_reject(draft_lp, target_lp, key)
    np.testing.assert_array_equal(np.asarray(accepted), [0, 0, 0, 1, 1])

    # resample_rejected must NEVER leak probability mass onto masked (-inf) or zero-residual tokens.
    vocab = 2048
    t_lp = jnp.full((1, vocab), -jnp.inf, dtype=jnp.float32).at[0, 3].set(jnp.log(0.7)).at[0, 5].set(jnp.log(0.3))
    d_lp = jnp.full((1, vocab), -jnp.inf, dtype=jnp.float32).at[0, 3].set(jnp.log(0.2)).at[0, 5].set(jnp.log(0.8))
    # Residual is strictly positive ONLY on token 3 (0.5), zero on token 5 and all masked tokens.
    t_batch = jnp.repeat(t_lp, 256, axis=0)
    d_batch = jnp.repeat(d_lp, 256, axis=0)
    draws = resample_rejected(t_batch, d_batch, jax.random.PRNGKey(123))
    assert set(np.asarray(draws).tolist()) == {3}

    # Zero-residual fallback: target == draft concentrated on token 7 -> must fall back to target (7),
    # NOT uniform garbage over all 2048 tokens!
    same_lp = jnp.full((64, vocab), -jnp.inf, dtype=jnp.float32).at[:, 7].set(0.0)
    fallback_draws = resample_rejected(same_lp, same_lp, jax.random.PRNGKey(7))
    assert set(np.asarray(fallback_draws).tolist()) == {7}


# ---------------------------------------------------------------------------
# S2-H18, S2-M3, S2-M9, S2-M11, S3-C4: Logits Processors & Sampling Processors
# ---------------------------------------------------------------------------


def test_s2_h18_penalties_sparse_and_bf16_frequency_above_256():
    vocab = 64
    # 300 occurrences of token 5 (> 256 bf16 integer precision limit) plus negative padding -100.
    ids = jnp.asarray([[5] * 300 + [-100, -1, 999]], dtype=jnp.int32)
    scores_bf16 = jnp.full((1, vocab), 10.0, dtype=jnp.bfloat16)

    freq = FrequencyPenaltyLogitsProcessor(0.5)(ids, scores_bf16, 303)
    assert freq.dtype == jnp.bfloat16
    # Exact count is 300 -> 10.0 - 150.0 = -140.0 (would be 10.0 - 128.0 = -118.0 if saturated at 256).
    assert float(freq[0, 5]) == pytest.approx(-140.0, abs=1.0)
    # Tail token (vocab - 1 = 63) must NOT be hit by negative token IDs -1 / -100!
    assert float(freq[0, vocab - 1]) == pytest.approx(10.0, abs=0.01)

    pres = PresencePenaltyLogitsProcessor(2.0)(ids, scores_bf16, 303)
    assert pres.dtype == jnp.bfloat16
    assert float(pres[0, 5]) == pytest.approx(8.0, abs=0.1)
    assert float(pres[0, vocab - 1]) == pytest.approx(10.0, abs=0.01)

    rep_scores = jnp.asarray([[4.0, -4.0] + [1.0] * (vocab - 2)], dtype=jnp.bfloat16)
    rep_ids = jnp.asarray([[0, 1, -1]], dtype=jnp.int32)
    rep = RepetitionPenaltyLogitsProcessor(2.0)(rep_ids, rep_scores, 3)
    assert rep.dtype == jnp.bfloat16
    assert float(rep[0, 0]) == pytest.approx(2.0, abs=0.05)
    assert float(rep[0, 1]) == pytest.approx(-8.0, abs=0.05)
    assert float(rep[0, vocab - 1]) == pytest.approx(1.0, abs=0.05)


def test_s2_m3_temperature_zero_and_batched_jitable_sampling_params():
    scores = jnp.asarray([[1.0, 3.0, 2.0, 0.5], [2.0, 0.0, 4.0, 1.0]], dtype=jnp.float32)
    ids = jnp.asarray([[1, 1], [2, 0]], dtype=jnp.int32)

    # Temperature 0.0 and -1.0 must not divide by zero.
    out_zero = TemperatureLogitsWarper(jnp.asarray(0.0))(ids, scores, 2)
    out_neg1 = TemperatureLogitsWarper(jnp.asarray(-1.0))(ids, scores, 2)
    np.testing.assert_allclose(np.asarray(out_zero), np.asarray(scores))
    np.testing.assert_allclose(np.asarray(out_neg1), np.asarray(scores))

    # JitableSamplingParams neutral init_empty, insert with 0-D scalars + stop/bad_words/allowed, view_1d/2d
    batch = JitableSamplingParams.init_empty(2)
    np.testing.assert_allclose(np.asarray(batch.temperature), [-1.0, -1.0])
    np.testing.assert_allclose(np.asarray(batch.top_p), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(batch.repetition_penalty), [1.0, 1.0])

    sp0 = SamplingParams(temperature=0.0, stop_token_ids=[3], allowed_token_ids=[0, 1])
    sp0._bad_words_token_ids = [[1, 2]]
    sp1 = SamplingParams(
        temperature=0.5, top_k=2, top_p=0.9, min_p=0.1, presence_penalty=1.0, frequency_penalty=0.5, repetition_penalty=2.0
    )
    batch = batch.insert(JitableSamplingParams.from_host_params(sp0), 0)
    batch = batch.insert(JitableSamplingParams.from_host_params(sp1), 1)

    assert batch.bad_words_token_ids.ndim == 3
    assert batch.view_1d().bad_words_token_ids.shape == batch.bad_words_token_ids.shape
    assert batch.view_2d().bad_words_token_ids.shape == batch.bad_words_token_ids.shape
    assert int(batch.all_stop_token_ids[0, 0]) == 3
    assert batch.allowed_token_ids is not None
    np.testing.assert_array_equal(np.asarray(batch.allowed_token_ids[0]), [0, 1])

    # All 7 warpers/processors work with batched JitableSamplingParams under jax.jit without lax.cond crash.
    @jax.jit
    def apply_all(jp: JitableSamplingParams, input_ids, logits):
        x = jp.get_logits_processor()(input_ids, logits, 2)
        return jp.get_logits_warper()(input_ids, x, 2)

    warped = apply_all(batch, ids, scores)
    assert warped.shape == scores.shape
    assert not bool(jnp.any(jnp.isnan(warped)))
    # Row 0 is greedy with neutral penalties -> completely unchanged!
    np.testing.assert_allclose(np.asarray(warped[0]), np.asarray(scores[0]))
    # Row 1 has top_k=2 -> at most 2 finite entries remain.
    assert int(jnp.sum(jnp.isfinite(warped[1]))) <= 2


def test_s2_m9_negative_scatter_wraparound_and_fp16_masked_floor():
    # apply_sparse_sampling_constraints: negative indices must drop, not wrap onto last row/token.
    logits = jnp.ones((2, 8), dtype=jnp.float16)
    out = apply_sparse_sampling_constraints(
        logits,
        bias_rows=jnp.asarray([-1, 0], dtype=jnp.int32),
        bias_tokens=jnp.asarray([-1, 1], dtype=jnp.int32),
        bias_values=jnp.asarray([5.0, 2.0], dtype=jnp.float32),
        restricted=jnp.asarray([False, True], dtype=jnp.bool_),
        allowed_rows=jnp.asarray([-1], dtype=jnp.int32),
        allowed_tokens=jnp.asarray([-1], dtype=jnp.int32),
        banned_rows=jnp.asarray([-1], dtype=jnp.int32),
        banned_tokens=jnp.asarray([-2], dtype=jnp.int32),
    )
    assert out.dtype == jnp.float16
    assert float(out[0, 1]) == pytest.approx(3.0, abs=1e-2)
    assert float(out[0, 7]) == pytest.approx(1.0, abs=1e-2)  # not corrupted by -1
    # Row 1 is restricted with no valid allowed tokens -> finite float16 floor (no -inf, no NaN softmax!).
    assert bool(jnp.all(jnp.isfinite(out[1])))
    probs = jax.nn.softmax(out[1].astype(jnp.float32))
    assert not bool(jnp.any(jnp.isnan(probs)))

    # Dense apply_sampling_constraints in float16 stays finite.
    all_banned = jnp.ones((1, 8), dtype=jnp.bool_)
    dense_out = apply_sampling_constraints(jnp.ones((1, 8), dtype=jnp.float16), bad_words_token_ids=all_banned)
    assert dense_out.dtype == jnp.float16
    assert bool(jnp.all(jnp.isfinite(dense_out)))

    # NoRepeatNGramLogitsProcessor with negative completion (-2) must not wrap to ban vocab-1.
    ngram_ids = jnp.asarray([[1, -2, 1]], dtype=jnp.int32)
    ngram_scores = jnp.zeros((1, 8), dtype=jnp.float32)
    ngram_out = NoRepeatNGramLogitsProcessor(ngram_size=2)(ngram_ids, ngram_scores, 3)
    assert bool(jnp.isfinite(ngram_out[0, 7]))


def test_s2_m11_and_s3_c4_logits_process_bf16_and_empty_inputs():
    # TopPLogitsWarper bf16 precision (512 uniform logits: bf16 cumsum saturates at 1.0 early, fp32 does not).
    bf16_uniform = jnp.zeros((1, 512), dtype=jnp.bfloat16)
    topp_out = TopPLogitsWarper(top_p=0.5)(jnp.zeros((1, 1), dtype=jnp.int32), bf16_uniform, 1)
    assert topp_out.dtype == jnp.bfloat16
    kept = int(jnp.sum(jnp.isfinite(topp_out[0])))
    assert 250 <= kept <= 260

    # ForcedBOS & ForcedEOS preserve input dtype (bf16 / fp16).
    for dt in (jnp.bfloat16, jnp.float16):
        s = jnp.zeros((1, 8), dtype=dt)
        assert ForcedBOSTokenLogitsProcessor(bos_token_id=2)(jnp.zeros((1, 1), dtype=jnp.int32), s, 1).dtype == dt
        assert (
            ForcedEOSTokenLogitsProcessor(max_length=5, eos_token_id=3)(jnp.zeros((1, 1), dtype=jnp.int32), s, 4).dtype
            == dt
        )

    # SuppressTokensAtBeginLogitsProcessor([]) must not raise float32 index TypeError.
    s_f32 = jnp.ones((1, 8), dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(SuppressTokensAtBeginLogitsProcessor([], begin_index=0)(jnp.zeros((1, 1), dtype=jnp.int32), s_f32, 0)),
        np.asarray(s_f32),
    )

    # S3-C4 / S2-M11: ForceTokensLogitsProcessor({}) and negative cur_len must not crash or wrap.
    empty_forcer = ForceTokensLogitsProcessor({})
    np.testing.assert_array_equal(
        np.asarray(empty_forcer(jnp.zeros((1, 1), dtype=jnp.int32), s_f32, 0)),
        np.asarray(s_f32),
    )
    forcer = ForceTokensLogitsProcessor({1: 4})
    neg_out = forcer(jnp.zeros((1, 1), dtype=jnp.int32), s_f32, -1)
    np.testing.assert_array_equal(np.asarray(neg_out), np.asarray(s_f32))
    pos_out = forcer(jnp.zeros((1, 1), dtype=jnp.int32), s_f32, 1)
    assert float(pos_out[0, 4]) == 0.0 and bool(jnp.isneginf(pos_out[0, 0]))


# ---------------------------------------------------------------------------
# S2-M2, S2-M11, S2-L5: sampling_funcs.py
# ---------------------------------------------------------------------------


def test_s2_m2_m11_l5_sampling_funcs_prng_independence_greedy_penalties_and_bf16():
    # PRNG independence across batch rows for vmaped_sample_top_p_efficient & dynamic_sample_tokens
    batch_size, vocab = 32, 16
    uniform_logits = jnp.zeros((batch_size, vocab), dtype=jnp.float32)
    vmap_tokens = vmaped_sample_top_p_efficient(
        uniform_logits,
        jnp.ones((batch_size,), dtype=jnp.float32),
        jnp.ones((batch_size,), dtype=jnp.float32),
        jax.random.PRNGKey(0),
        vocab,
    )
    assert len(set(np.asarray(vmap_tokens).tolist())) > 1

    dyn_tokens = dynamic_sample_tokens(
        tokens=jnp.zeros((batch_size, 1), dtype=jnp.int32),
        length=jnp.zeros((batch_size,), dtype=jnp.int32),
        logits=uniform_logits,
        top_p=jnp.ones((batch_size,), dtype=jnp.float32),
        temperature=jnp.ones((batch_size,), dtype=jnp.float32),
        random_sampling=jnp.ones((batch_size,), dtype=jnp.bool_),
        presence_penalty=jnp.zeros((batch_size,), dtype=jnp.float32),
        frequency_penalty=jnp.zeros((batch_size,), dtype=jnp.float32),
        repetition_penalty=jnp.ones((batch_size,), dtype=jnp.float32),
        rngs=jax.random.PRNGKey(0),
    )
    assert len(set(np.asarray(dyn_tokens[:, 0, 0]).tolist())) > 1

    # Greedy branch in dynamic_sample_tokens respects presence/frequency/repetition penalties
    greedy_pen = dynamic_sample_tokens(
        tokens=jnp.asarray([[0, 0], [0, 0]], dtype=jnp.int32),
        length=jnp.asarray([2, 2], dtype=jnp.int32),
        logits=jnp.asarray([[4.0, 3.0, 0.0], [4.0, 3.0, 0.0]], dtype=jnp.float32),
        top_p=jnp.asarray([1.0, 1.0], dtype=jnp.float32),
        temperature=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        random_sampling=jnp.asarray([False, False], dtype=jnp.bool_),
        presence_penalty=jnp.asarray([2.0, 0.0], dtype=jnp.float32),
        frequency_penalty=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        repetition_penalty=jnp.asarray([2.0, 1.0], dtype=jnp.float32),
        rngs=jax.random.PRNGKey(0),
    )
    np.testing.assert_array_equal(np.asarray(greedy_pen[:, 0, 0]), [1, 0])

    # sample_top_p_efficient with temperature == 0.0 must deterministically return argmax (not T=1.0 random)
    close_logits = jnp.broadcast_to(jnp.asarray([1.0, 1.05, 1.0, 1.0], dtype=jnp.bfloat16), (64, 4))
    greedy_t0 = sample_top_p_efficient(
        close_logits,
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jax.random.PRNGKey(7),
        top_k_for_computation=4,
    )
    assert set(np.asarray(greedy_t0).tolist()) == {1}


# ---------------------------------------------------------------------------
# S2-M10: SamplingParams & GuidedDecodingParams
# ---------------------------------------------------------------------------


def test_s2_m10_sampling_params_and_guided_decoding_validation():
    # ignore_eos=True must not add model_eos_token_id; list model_eos_token_id must not crash
    sp_ign = SamplingParams(ignore_eos=True, stop_token_ids=[99])
    sp_ign.update_with_generation_config({"eos_token_id": [1, 2]}, model_eos_token_id=[1, 3])
    assert sp_ign.all_stop_token_ids == {99}

    sp_use = SamplingParams(ignore_eos=False)
    sp_use.update_with_generation_config({"eos_token_id": [1, 2]}, model_eos_token_id=[1, 3])
    assert sp_use.all_stop_token_ids == {1, 2, 3}

    # sampling_type is a dynamic @property (never stale after mutating temperature)
    sp_mut = SamplingParams(temperature=0.8)
    assert sp_mut.sampling_type == SamplingType.RANDOM
    sp_mut.temperature = 0.0
    assert sp_mut.sampling_type == SamplingType.GREEDY

    # _verify_args rejects invalid frequency_penalty, min_p, min_tokens, truncate_prompt_tokens, and NaN
    for bad_kwargs in (
        {"frequency_penalty": 2.5},
        {"frequency_penalty": float("nan")},
        {"presence_penalty": float("nan")},
        {"repetition_penalty": float("nan")},
        {"temperature": float("nan")},
        {"top_p": float("nan")},
        {"min_p": -0.1},
        {"min_p": 1.2},
        {"min_p": float("nan")},
        {"min_tokens": -1},
        {"truncate_prompt_tokens": 0},
    ):
        with pytest.raises(ValueError):
            SamplingParams(**bad_kwargs)

    # update_with_tokenizer honors len(tokenizer) > vocab_size (added special tokens)
    class _AddedTokenTokenizer:
        vocab_size = 100
        model_max_length = 512

        def __len__(self):
            return 105

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [102]

    sp_bw = SamplingParams(bad_words=["<extra>"])
    sp_bw.update_with_tokenizer(_AddedTokenTokenizer())
    assert sp_bw.bad_words_token_ids == [[102]]

    # stop=[""] clamps buffer length to >= 0 (never -1)
    assert SamplingParams(stop=[""])._output_text_buffer_length == 0

    # GuidedDecodingParams allows json_object=False combined with another mode
    gd = GuidedDecodingParams(json_object=False, regex=r"[0-9]+")
    assert gd.regex == r"[0-9]+"


# ---------------------------------------------------------------------------
# S2-H19, S2-M12 (eval), S2-L5 (eval): eSurgeLMEvalAdapter
# ---------------------------------------------------------------------------


def test_s2_h19_m12_l5_esurge_eval_adapter():
    # _strip_empty_reasoning_scaffold strips ONLY the trailing (last) scaffold, keeping earlier few-shot turns.
    few_shot = "user: q1\nassistant: <think></think>\na1\nuser: q2\nassistant: <think></think>\ndef f():"
    stripped = _strip_empty_reasoning_scaffold(few_shot, "<think>", "</think>")
    assert "assistant: <think></think>\na1" in stripped
    assert stripped.endswith("assistant: def f():")

    class _HFStyleTokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = [2, 3]
            self.bos_token_id = 1
            self.padding_side = "right"
            self.name_or_path = "hf-test"
            self.last_template_kwargs = None
            self.decode_calls = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            self.last_template_kwargs = dict(kwargs)
            return f"rendered:{len(messages)}:think={kwargs.get('enable_thinking')}"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [10 + (ord(c) % 20) for c in text]

        def decode(self, ids, skip_special_tokens=False, spaces_between_special_tokens=False):
            del skip_special_tokens, spaces_between_special_tokens
            self.decode_calls += 1
            return f"decoded-{len(ids)}"

    class _ScoringModel:
        def __init__(self, attn_mech: str):
            self.config = SimpleNamespace(
                attn_mechanism=attn_mech,
                decode_attn_mechanism=attn_mech,
                get_text_config=lambda: self.config,
            )
            self.graphstate = {}
            self.graphother = {}
            self.rebuilt_with = None
            self.seen_shapes: list[tuple[int, int]] = []

        def new_graphdef(self, **kwargs):
            self.rebuilt_with = kwargs
            return "vanilla_graphdef"

        def merge_module(self, graphdef, gs, go):
            del graphdef, gs, go
            return self

        def __call__(self, input_ids, attention_mask):
            del attention_mask
            self.seen_shapes.append(tuple(input_ids.shape))
            b, s = input_ids.shape
            logits = jnp.zeros((b, s, 64), dtype=jnp.float32)
            return SimpleNamespace(logits=logits)

    class _FakeSurge:
        _scheduler_running = True
        max_num_seqs = 4
        truncate_mode = "left"

        def __init__(self, model):
            self.runner = SimpleNamespace(model=model, mesh=None)
            self.captured_calls: list[tuple[list[str], SamplingParams]] = []

        def generate(self, prompts, sampling_params, request_id, use_tqdm=False):
            del use_tqdm
            self.captured_calls.append((list(prompts), sampling_params.clone()))
            return [
                SimpleNamespace(request_id=rid, accumulated_text="ok", get_text=lambda: "ok") for rid in request_id
            ]

    # S2-H19(b): both multi_latent_ragged_page_attention_v2 and unified_attention trigger vanilla rebuild!
    for mech in ("multi_latent_ragged_page_attention_v2", "unified_attention"):
        tok = _HFStyleTokenizer()
        mdl = _ScoringModel(mech)
        adapter = eSurgeLMEvalAdapter(
            _FakeSurge(mdl),
            tok,
            max_length=64,
            max_new_tokens=128,
            enable_thinking=True,
            sampling_params=SamplingParams(temperature=0.7),
            add_bos_token=True,
        )
        # S2-H19(c): list eos_token_id normalized to scalar int on pad_token_id and eot_token_id
        assert tok.pad_token_id == 2
        assert adapter.eot_token_id == 2
        adapter._build_scoring_model()
        assert mdl.rebuilt_with is not None
        assert mdl.rebuilt_with["attn_mechanism"] == "vanilla"

    # S2-H19(a): apply_chat_template forwards enable_thinking through **kwargs (VAR_KEYWORD)
    tok = _HFStyleTokenizer()
    mdl = _ScoringModel("vanilla")
    surge = _FakeSurge(mdl)
    adapter = eSurgeLMEvalAdapter(
        surge,
        tok,
        max_length=256,
        max_new_tokens=128,
        enable_thinking=True,
        sampling_params=SamplingParams(temperature=0.7),
        add_bos_token=True,
    )
    rendered = adapter.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert tok.last_template_kwargs["enable_thinking"] is True
    assert "think=True" in rendered

    # S2-H19(c) + S2-M12: .args support, disable_tqdm=True, default max_tokens=16 does NOT cap to 16,
    # untruncated prompt is NOT re-decoded, and greedy_until keeps temperature=0.0 despite template!
    @dataclasses.dataclass
    class _V04Instance:
        args: tuple

    tok.decode_calls = 0
    adapter.generate_until([_V04Instance(args=("hello world", {}))], disable_tqdm=True)
    prompt_sent, sp_sent = surge.captured_calls[-1]
    assert prompt_sent == ["hello world"]  # untruncated prompt kept verbatim!
    assert sp_sent.max_tokens == 128  # NOT silently capped to SamplingParams default 16!
    assert sp_sent.temperature == pytest.approx(0.7)

    adapter.greedy_until([_V04Instance(args=("greedy prompt", {}))], disable_tqdm=True)
    _, sp_greedy = surge.captured_calls[-1]
    assert sp_greedy.temperature == 0.0

    # S2-H19(c) + S2-L5: loglikelihood_rolling with list eos, .args, disable_tqdm=True,
    # plus length bucketing (5 & 9 both pad to 16) and add_bos_token=True in _encode_pair.
    ctx_enc, _ = adapter._encode_pair("abc", "def")
    assert ctx_enc[0] == 1  # bos_token_id prepended

    adapter._scoring_logits_fn = lambda inp, mask: mdl(inp, mask).logits
    rolling = adapter.loglikelihood_rolling([_V04Instance(args=("hello",)), _V04Instance(args=("hello world",))], disable_tqdm=True)
    assert len(rolling) == 2
    assert all(isinstance(v, float) and np.isfinite(v) for v in rolling)

    mdl.seen_shapes.clear()
    adapter._loglikelihood_token_ids([[1, 2, 3]], [[4, 5]])  # len 4 -> bucket 16
    adapter._loglikelihood_token_ids([[1, 2, 3, 4, 5, 6]], [[7, 8, 9]])  # len 8 -> bucket 16
    assert mdl.seen_shapes == [(1, 16), (1, 16)]


# ---------------------------------------------------------------------------
# Audit #28 (sampling_params.py): JitableSamplingParams.insert trims padding
# ---------------------------------------------------------------------------


def test_leak_fix_28_jitable_sampling_params_insert_shrinks_padding_widths():
    batch = JitableSamplingParams.init_empty(2)

    wide = SamplingParams(
        temperature=0.5,
        stop_token_ids=list(range(10, 26)),
        allowed_token_ids=list(range(100, 132)),
    )
    wide._bad_words_token_ids = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [11]]

    narrow = SamplingParams(
        temperature=0.2,
        stop_token_ids=[42, 43],
        allowed_token_ids=[7, 8, 9],
    )
    narrow._bad_words_token_ids = [[55, 56]]

    plain = SamplingParams(temperature=0.0)

    # 1. Insert wide sample at slot 0, narrow sample at slot 1.
    batch = batch.insert(JitableSamplingParams.from_host_params(wide), 0)
    batch = batch.insert(JitableSamplingParams.from_host_params(narrow), 1)
    assert batch.all_stop_token_ids.shape == (2, 16)
    assert batch.allowed_token_ids is not None and batch.allowed_token_ids.shape == (2, 32)
    assert batch.bad_words_token_ids.shape == (2, 4, 5)
    assert batch.bad_words_lengths.shape == (2, 4)

    # 2. Overwrite wide slot 0 with plain sample -> widths MUST shrink to slot 1's active bounds.
    batch = batch.insert(JitableSamplingParams.from_host_params(plain), 0)
    assert batch.all_stop_token_ids.shape == (2, 2)
    assert batch.allowed_token_ids is not None and batch.allowed_token_ids.shape == (2, 3)
    assert batch.bad_words_token_ids.shape == (2, 1, 2)
    assert batch.bad_words_lengths.shape == (2, 1)
    np.testing.assert_array_equal(np.asarray(batch.all_stop_token_ids[1]), [42, 43])
    np.testing.assert_array_equal(np.asarray(batch.allowed_token_ids[1]), [7, 8, 9])
    np.testing.assert_array_equal(np.asarray(batch.bad_words_token_ids[1]), [[55, 56]])
    np.testing.assert_array_equal(np.asarray(batch.bad_words_lengths[1]), [2])

    # 3. Overwrite slot 1 with plain sample -> all widths collapse back to 0 / None.
    batch = batch.insert(JitableSamplingParams.from_host_params(plain), 1)
    assert batch.all_stop_token_ids.shape == (2, 0)
    assert batch.allowed_token_ids is None
    assert batch.bad_words_token_ids.shape == (2, 0, 0)
    assert batch.bad_words_lengths.shape == (2, 0)

    # 4. insert() remains JIT-traceable when called inside jax.jit.
    base_jit = JitableSamplingParams.init_empty(2).insert(JitableSamplingParams.from_host_params(narrow), 1)
    sample_jit = JitableSamplingParams.from_host_params(narrow)
    inserted_jit = jax.jit(lambda b, s: b.insert(s, 0))(base_jit, sample_jit)
    assert inserted_jit.all_stop_token_ids.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(inserted_jit.all_stop_token_ids[0]), [42, 43])
