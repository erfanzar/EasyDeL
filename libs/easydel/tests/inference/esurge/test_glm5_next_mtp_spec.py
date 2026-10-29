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

"""GLM-5-Next nextn (MTP) head: reference math, cached drafting and eSurge greedy parity (computation test).

A tiny fp32 GLM-5-Next hybrid (3 KDA + 1 DSA layer, HIGHEST precision) gets a randomly initialised
:class:`Glm5NextMTPHead` attached as ``model.mtp``:

- the stateless head forward equals an independent composition of the DeepSeek-V3 / GLM nextn equation
  ``norm(block(fc(concat[enorm(embed(t)), hnorm(h)])))`` (vLLM ``glm4_moe_mtp``);
- the inline-MTP drafter's cached path (prefill then one token per step through the drafter-local cache sized by
  ``cache_spec``) reproduces the stateless forward;
- eSurge serving with the drafter is greedy-identical to plain serving (KDA rollback on rejected drafts).
"""

from __future__ import annotations

import gc

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter
from easydel.modules.glm5_next import Glm5NextMTPHead, attach_glm5_next_mtp
from ejkernel.types import MaskInfo
from spectrax import common_types

VOCAB = 128
K = 2
STEP_BUDGET = 256
WAVE = [
    {"id": "a", "prompt_token_ids": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], "max_tokens": 12},
    {"id": "b", "prompt_token_ids": [10, 11, 12, 13, 14], "max_tokens": 10},
]


@pytest.fixture(autouse=True)
def _default_fast_recurrent_path(monkeypatch):
    monkeypatch.delenv("EASYDEL_SPEC_RECURRENT_REPLAY", raising=False)


def _make_model_with_head():
    config = ed.Glm5NextTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=64,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=4,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 4,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        max_position_embeddings=512,
        initializer_range=0.2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.kvdtype = jnp.float32
    kwargs = dict(dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST)
    with config.mesh:
        model = ed.Glm5NextForCausalLM(config=config, rngs=spx.Rngs(0), **kwargs)
        head = Glm5NextMTPHead(config, rngs=spx.Rngs(1), max_cache_length=128, **kwargs)
    model.eval()
    return attach_glm5_next_mtp(model, head)


@pytest.fixture(scope="module")
def model():
    return _make_model_with_head()


def _rms(x, w, eps):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def test_head_forward_matches_nextn_equation(model):
    head = model.mtp
    eps = head.config.rms_norm_eps
    rng = np.random.default_rng(0)
    n = 9
    hidden = jnp.asarray(rng.standard_normal((1, n, 32)), jnp.float32)
    ids = jnp.asarray(rng.integers(0, VOCAB, (1, n)), jnp.int32)
    with model.mesh, jax.default_matmul_precision("highest"):
        embeds = model.get_embedding()(ids)
        got = head(hidden, embeds)
        fused = jnp.concatenate(
            [
                _rms(embeds, head.pre_fc_norm_embedding.weight.value, eps),
                _rms(hidden, head.pre_fc_norm_hidden.weight.value, eps),
            ],
            axis=-1,
        )
        h = fused @ head.fc.weight.value  # EasyDeL kernels are [in, out]
        layer = head.layers[0]
        mask_info = MaskInfo.dynamic_init(mask_info=None, input_ids=None, inputs_embeds=h, attention_mask=None)
        pos = jnp.arange(n, dtype=jnp.int32)[None]
        attn = layer.self_attn(
            _rms(h, layer.input_layernorm.weight.value, eps),
            mask_info,
            pos,
            common_types.MODE_TRAIN,
            None,
            None,
            False,
            None,
        )
        h = h + attn.attention_output
        mlp_out, _ = layer.mlp(_rms(h, layer.post_attention_layernorm.weight.value, eps))
        want = _rms(h + mlp_out, head.norm.weight.value, eps)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)


def test_cached_drafter_matches_stateless_head(model):
    head = model.mtp
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=1)
    spec = head.cache_spec()
    assert (spec["key_dim"], spec["value_dim"], spec["num_heads"]) == (8, 8, 2)
    rng = np.random.default_rng(1)
    n = 7
    hidden = jnp.asarray(rng.standard_normal((1, n, 32)), jnp.float32)
    ids = jnp.asarray(rng.integers(0, VOCAB, (1, n)), jnp.int32)
    pos = jnp.arange(n, dtype=jnp.int32)[None]
    with model.mesh, jax.default_matmul_precision("highest"):
        want = model.apply_lm_head(head(hidden, model.get_embedding()(ids)))
        drafter.reset(1)
        cache = drafter._ensure_mtp_cache(1)
        view = cache.views[0]
        assert view.key.shape == (1, 128, 2, 8), view.key.shape
        split = 4
        _, logits_pre, cache = drafter._mtp_hidden_and_logits(ids[:, :split], hidden[:, :split], pos[:, :split], cache)
        got = [logits_pre]
        for t in range(split, n):
            _, lg, cache = drafter._mtp_hidden_and_logits(
                ids[:, t : t + 1], hidden[:, t : t + 1], pos[:, t : t + 1], cache
            )
            got.append(lg)
    got = jnp.concatenate(got, axis=1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2e-4, atol=2e-4)


class _Harness:
    def __init__(self, model, *, drafter=None):
        self.runner = eSurgeRunner(
            model=model,
            hbm_utilization=0.02,
            page_size=16,
            max_cache_tokens=2048,
            max_model_len=128,
            max_num_batched_tokens=64,
            min_input_pad=1,
            min_token_pad=16,
            max_num_seqs=2,
            max_num_seq_buckets=[2],
            async_scheduling=False,
            use_aot_forward=True,
            verbose=False,
            enable_overlap_execution=False,
            enable_sampler_metrics=False,
            drafter=drafter,
        )
        self.runner.compile(max_num_batched_tokens=64)
        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
            num_draft_tokens=self.runner.num_draft_tokens,
        )

    def generate(self, specs):
        requests = []
        for spec in specs:
            req = EngineRequest(
                request_id=spec["id"],
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=int(spec["max_tokens"]), temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            requests.append(req)
            self.scheduler.add_request(req)
        for _ in range(STEP_BUDGET):
            out = self.runner.execute_model(so := self.scheduler.schedule())
            self.scheduler.update_from_output(so, out)
            if all(r.is_finished() for r in requests):
                break
        else:
            raise AssertionError("generation did not finish")
        return {s["id"]: list(r.output_token_ids) for s, r in zip(specs, requests, strict=True)}

    def close(self):
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        gc.collect()


def test_nextn_drafter_serving_matches_plain_greedy(model):
    base = _Harness(model)
    try:
        baseline = base.generate(WAVE)
    finally:
        base.close()
    harness = _Harness(model, drafter=Qwen3_5MTPDrafter(model, num_draft_tokens=K))
    try:
        served = harness.generate(WAVE)
        generated = harness.runner.spec_decode_num_drafts_generated
        accepted = harness.runner.spec_decode_num_drafts_accepted
        print(f"[glm nextn spec] generated={generated} accepted={accepted}")
        assert generated > 0, "the nextn drafter never ran"
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}
