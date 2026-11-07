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
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter
from easydel.modules.glm5_next import Glm5NextMTPHead, attach_glm5_next_mtp
from ejkernel.types import MaskInfo
from spectrax import common_types

try:
    from ._common import make_core_for_runner, served_request
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request

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


def _make_model():
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
    model.eval()
    return model


def _make_model_with_head():
    model = _make_model()
    kwargs = dict(dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST)
    with model.config.mesh:
        head = Glm5NextMTPHead(model.config, rngs=spx.Rngs(1), max_cache_length=128, **kwargs)
    return attach_glm5_next_mtp(model, head)


@pytest.fixture(scope="module")
def model():
    return _make_model_with_head()


_HF_OWN = {
    "pre_fc_norm_embedding.weight": "enorm.weight",
    "pre_fc_norm_hidden.weight": "hnorm.weight",
    "fc.weight": "eh_proj.weight",
    "norm.weight": "shared_head.norm.weight",
}


def _head_to_hf(head, layer: int) -> dict[str, np.ndarray]:
    """Independent inverse of the nextn checkpoint layout (HF orientation as in the published shards).

    HF stores 2-D linears as ``[out, in]`` (EasyDeL ``[in, out]``), routed experts one tensor per expert, and the
    shared expert's gate and up projections separately; the indexer's raw ``index_kpool_compress_*`` parameters and
    1-D tensors keep their layout.
    """
    _gd, state = spx.export(head)
    out: dict[str, np.ndarray] = {}
    prefix = f"model.language_model.layers.{layer}."
    for _collection, path, leaf in state.items():
        path = str(path)
        arr = np.asarray(jax.device_get(leaf))
        if path in _HF_OWN:
            name = prefix + _HF_OWN[path]
            out[name] = arr.T if arr.ndim == 2 else arr
            continue
        rest = path[len("layers.0.") :]
        if ".experts." in path and "shared" not in path and arr.ndim == 3:
            base, proj, weight = rest.rsplit(".", 2)
            for i in range(arr.shape[0]):
                out[f"{prefix}{base}.{i}.{proj}.{weight}"] = arr[i].T
        elif rest.endswith("shared_experts.gate_up_proj.weight"):
            half = arr.shape[1] // 2
            out[prefix + "mlp.shared_experts.gate_proj.weight"] = arr[:, :half].T
            out[prefix + "mlp.shared_experts.up_proj.weight"] = arr[:, half:].T
        elif arr.ndim == 2 and "index_kpool_compress" not in rest:
            out[prefix + rest] = arr.T
        else:
            out[prefix + rest] = arr
    return out


def _write_nextn_checkpoint(directory, tensors: dict, fp8_key: str) -> None:
    """Two shards + index + config; ``fp8_key`` is stored as block-fp8 with a ``weight_scale_inv`` sibling."""
    import json

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    hf = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in tensors.items()}
    hf[fp8_key] = hf[fp8_key].to(torch.float8_e4m3fn)
    hf[fp8_key + "_scale_inv"] = torch.tensor([[0.5]], dtype=torch.float32)
    keys = sorted(hf)
    shards = {"model-00001-of-00002.safetensors": keys[::2], "model-00002-of-00002.safetensors": keys[1::2]}
    weight_map = {}
    for name, part in shards.items():
        save_file({k: hf[k] for k in part}, str(directory / name))
        weight_map.update(dict.fromkeys(part, name))
    weight_map["model.language_model.layers.0.input_layernorm.weight"] = "model-00009-of-00009.safetensors"
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    quant = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    (directory / "config.json").write_text(json.dumps({"quantization_config": quant, "text_config": {}}))


def test_drafter_hook_loads_nextn_checkpoint(tmp_path):
    """``model.drafter(nextn_checkpoint=...)`` loads every nextn tensor (block-fp8 dequantized) and drafts with it."""
    reference = _make_model_with_head().mtp
    layer = 4
    tensors = _head_to_hf(reference, layer)
    fp8_key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
    # fp8-representable values at scale 0.5: the dequantized weight is exactly code * 0.5.
    codes = np.round(np.clip(tensors[fp8_key] * 8, -8, 8)) / 4
    tensors[fp8_key] = codes.astype(np.float32)
    _write_nextn_checkpoint(tmp_path, tensors, fp8_key)

    target = _make_model()
    assert not target.has_mtp()
    drafter = target.drafter(method="mtp", num_draft_tokens=K, nextn_checkpoint=str(tmp_path), nextn_cache_length=96)
    assert target.has_mtp() and isinstance(drafter, Qwen3_5MTPDrafter)
    assert target.mtp.cache_spec()["max_length"] == 96

    _gd, want = spx.export(reference)
    _gd, got = spx.export(target.mtp)
    got_leaves = {str(p): leaf for _c, p, leaf in got.items()}
    for _c, path, leaf in want.items():
        expected = np.asarray(jax.device_get(leaf))
        if str(path) == "layers.0.self_attn.o_proj.weight":
            expected = (codes * 0.5).T
        actual = np.asarray(jax.device_get(got_leaves.pop(str(path))))
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        np.testing.assert_array_equal(actual, expected, err_msg=str(path))
    assert not got_leaves

    drafter.ensure_batch_size(2)
    assert drafter.mtp_cache_capacity() == 96
    with pytest.raises(ValueError, match="already attached"):
        target.drafter(method="mtp", nextn_checkpoint=str(tmp_path))


def test_drafter_hook_rejects_incomplete_nextn_checkpoint(tmp_path):
    reference = _make_model_with_head().mtp
    tensors = _head_to_hf(reference, 4)
    fp8_key = "model.language_model.layers.4.self_attn.o_proj.weight"
    tensors[fp8_key] = np.zeros_like(tensors[fp8_key])
    tensors.pop("model.language_model.layers.4.hnorm.weight")
    _write_nextn_checkpoint(tmp_path, tensors, fp8_key)
    with pytest.raises(ValueError, match="missing"):
        _make_model().drafter(method="mtp", nextn_checkpoint=str(tmp_path))


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
        self.scheduler = make_core_for_runner(
            self.runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
            num_draft_tokens=self.runner.num_draft_tokens,
        )

    def generate(self, specs):
        requests = []
        for spec in specs:
            req = served_request(
                request_id=spec["id"],
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=int(spec["max_tokens"]), temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            requests.append(req)
            self.scheduler.add_request(req)
        for _ in range(STEP_BUDGET):
            out = self.runner.execute_model(so := self.scheduler.schedule())
            self.scheduler.commit(so, out)
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
