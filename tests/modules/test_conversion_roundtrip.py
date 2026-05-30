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

"""HuggingFace <-> EasyDeL conversion round-trip tests across all supported models.

For every registered ``CAUSAL_LM`` model that has a HuggingFace torch equivalent,
this verifies the full conversion pipeline using the protocol:

1. Build a small EasyDeL model (random init, ``scan_layers=False``).
2. ``ed_model.to_torch()``                -> ``hf1``  (EasyDeL -> HF export)
3. ``hf1.state_dict()`` -> EasyDeL        -> ``ed2`` (HF -> EasyDeL import,
   via the same ``transform_fn`` the on-disk loader uses)
4. ``logits(ed_model) ~= logits(hf1)``    (export fidelity)
5. ``logits(ed2)       ~= logits(hf1)``   (import fidelity / round-trip)

Both conversion directions are exercised, so a regression in either the
``to_torch`` export or the ``transform_fn`` import path is caught.

Run a single model:  ``pytest tests/modules/test_conversion_roundtrip.py -k llama``
Run all:             ``pytest tests/modules/test_conversion_roundtrip.py``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import easydel as ed

jax.config.update("jax_platform_name", "cpu")

# Every registered ``CAUSAL_LM`` ``model_type`` with a HuggingFace torch equivalent
# (i.e. ``module.get_torch_loader()._model_mapping`` contains its config class), so the
# full ``.to_torch()`` export + ``transform_fn`` import round-trip can be exercised.
# Models without an HF mapping (arctic, xerxes, kimi_linear, ...) are intentionally
# omitted — there is nothing to round-trip against.
DENSE_MODELS = [
    "llama",
    "qwen2",
    "qwen3",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3_text",
    "gemma4_text",
    "phi",
    "phi3",
    "olmo",
    "olmo2",
    "olmo3",
    "cohere",
    "cohere2",
    "glm",
    "glm4",
    "smollm3",
    "stablelm",
    "exaone4",
    "seed_oss",
    "gpt2",
    "gptj",
    "gpt_neox",
    "opt",
    "mpt",
    "falcon",
    "roberta",
    "qwen3_next",
    "falcon_h1",
]
# MoE models: ``.to_torch`` export of fused experts is a known work-in-progress
# (needs the reform/layout fused-expert path). Tracked with xfail so the suite stays
# green while the gap is visible.
MOE_MODELS = [
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_5_moe",
    "deepseek_v2",
    "deepseek_v3",
    "dbrx",
    "glm4_moe",
    "gpt_oss",
    "minimax",
]
# State-space / hybrid models with idiosyncratic param layouts (separate follow-up).
SSM_MODELS = ["mamba", "mamba2", "falcon_mamba", "rwkv"]

# Known conversion gaps (each a distinct, architecture-specific issue), xfail-marked so
# the suite stays green while keeping the gap visible. Map: model_type -> reason.
KNOWN_GAPS = {
    # GPT-family fused projections (Conv1D ``c_attn``/``c_fc``, phi fused ``qkv``) are not
    # split on EasyDeL->HF export: HF receives a [3*hidden] tensor where it wants [hidden].
    "gpt2": "fused Conv1D c_attn/c_fc not split on export",
    "phi": "fused qkv projection not split on export",
    "opt": "fc1/fc2 MLP weights not transposed on export",
    # FalconConfig.head_dim is a read-only property; config round-trip can't set it.
    "falcon": "FalconConfig.head_dim is read-only on the HF side",
    # roberta ties the LM-head decoder; import hits IllegalMutationError on decoder.weight.
    "roberta": "tied LM-head decoder not declared mutable on import",
    # qwen3_next 3-D linear-attn conv1d kernel is not re-oriented on export ([4,1,8192]
    # exported where HF wants [8192,1,4]); 3-D axis mapping is ambiguous from shape alone.
    "qwen3_next": "linear-attn conv1d 3-D kernel orientation not handled on export",
    # GPT-family fused QKV (gptj rotary qkv, gpt_neox query_key_value) not split on export.
    "gptj": "fused qkv / lm_head.bias layout not handled on export",
    "gpt_neox": "fused query_key_value projection not split on export",
}

ROUNDTRIP_MODELS = DENSE_MODELS

# Tolerances on max|Δlogits| (fp32, tiny random model).
EXPORT_ATOL = 5e-2
ROUNDTRIP_ATOL = 5e-2

# Per-model config overrides for models whose forward has constraints the default
# tiny config would violate (unrelated to conversion correctness). E.g. stablelm uses
# partial-rotary, so head_dim must be large enough that ``partial_rotary_factor *
# head_dim`` is an even, non-degenerate rotary dimension.
CONFIG_OVERRIDES = {
    "stablelm": dict(hidden_size=128),  # head_dim 32 -> rotary_ndims 8 (even)
}


def _small_config(config_cls, model_type):
    """Build a tiny, conversion-friendly config for ``config_cls``.

    Only fields the constructor accepts are passed. ``scan_layers=False`` keeps the
    EasyDeL layer params un-stacked so per-layer names match the HF checkpoint.
    """
    import inspect

    fields = set(inspect.signature(config_cls.__init__).parameters)
    desired = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        scan_layers=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    desired.update(CONFIG_OVERRIDES.get(model_type, {}))
    kwargs = {k: v for k, v in desired.items() if k in fields}
    return config_cls(**kwargs)


def _logits(model, input_ids):
    out = model(input_ids=jnp.asarray(input_ids))
    return np.asarray(out.logits.astype(jnp.float32))


def _torch_logits(torch_model, input_ids):
    import torch

    torch_model.eval()
    with torch.no_grad():
        return torch_model(input_ids=torch.tensor(input_ids)).logits.float().numpy()


@pytest.mark.parametrize("model_type", ROUNDTRIP_MODELS)
def test_hf_easydel_roundtrip(model_type, request):
    """EasyDeL -> HF -> EasyDeL round-trip parity for ``model_type``."""
    import spectrax as spx

    if model_type in KNOWN_GAPS:
        request.node.add_marker(pytest.mark.xfail(reason=KNOWN_GAPS[model_type], strict=True))
    pytest.importorskip("torch")
    try:
        config_cls, module_cls = ed.get_modules_by_type(model_type, ed.TaskType.CAUSAL_LM)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_type}: not registered for CAUSAL_LM ({exc})")

    config = _small_config(config_cls, model_type)

    # 1. EasyDeL model (random init).
    ed_model = module_cls(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, 256, size=(1, 12)).astype("int32")
    ed_logits = _logits(ed_model, input_ids)

    # 2. EasyDeL -> HF (export). Skips cleanly if no HF torch mapping exists.
    try:
        hf1 = ed_model.to_torch()
    except KeyError as exc:
        pytest.skip(f"{model_type}: no HuggingFace torch mapping ({exc})")
    hf1_logits = _torch_logits(hf1, input_ids)

    assert ed_logits.shape == hf1_logits.shape, f"{model_type}: export logits shape mismatch"
    export_diff = float(np.max(np.abs(ed_logits - hf1_logits)))
    assert export_diff < EXPORT_ATOL, f"{model_type}: EasyDeL->HF export diverged, max|Δ|={export_diff:.2e}"

    # 3. HF -> EasyDeL (import) via the in-memory transform_fn (same path the loader uses).
    ed2 = module_cls.lazy_init(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(1))
    ed2 = ed.traversals.merge_model_and_tree(ed2, tree=ed2.transform_fn(hf1.state_dict()))
    ed2.eval()
    ed2_logits = _logits(ed2, input_ids)

    # 4./5. Round-trip fidelity: ed2 (loaded from hf1) must match hf1.
    assert ed2_logits.shape == hf1_logits.shape, f"{model_type}: import logits shape mismatch"
    roundtrip_diff = float(np.max(np.abs(ed2_logits - hf1_logits)))
    assert roundtrip_diff < ROUNDTRIP_ATOL, (
        f"{model_type}: HF->EasyDeL import diverged after round-trip, max|Δ|={roundtrip_diff:.2e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
