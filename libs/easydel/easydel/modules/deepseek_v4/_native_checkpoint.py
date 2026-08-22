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

"""Key mapping for DeepSeek-V4's *native* checkpoint layout.

DeepSeek publishes V4 (Flash and Pro, dated and undated) only in the naming
its own inference stack uses -- ``layers.N.attn.wq_a``, ``layers.N.ffn.experts.
<i>.w1`` -- not in transformers naming. The repo's ``inference/convert.py``
runs HF -> native, so the transformers-format weights are its *input* and were
never released; native is the only form on the hub. This module is the inverse
of that script's renames, so :meth:`from_pretrained` can consume the published
checkpoint directly.

The mapping is data, not control flow: :data:`LEAF_ALIASES` and
:data:`PREFIX_ALIASES` are the whole contract, and
:func:`native_key_to_easydel` is a pure string transform over them. Structural
regrouping (256 per-expert tensors -> one stacked ``[E, H, I]`` parameter,
gate/up fusion) is deliberately NOT done here -- that belongs to the
``reform_param`` layout descriptors, which already express it.
"""

from __future__ import annotations

import re

#: Trailing module-name renames, applied to the second-to-last path segment
#: (or the last, for the bare ``hc_*`` tensors which carry no ``.weight``).
LEAF_ALIASES: dict[str, str] = {
    # MLA attention projections
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "wkv": "kv_proj",
    "wo_a": "o_a_proj",
    "wo_b": "o_b_proj",
    "q_norm": "q_a_norm",
    # the compressor's own sub-projections
    "wgate": "gate_proj",
    "norm": "kv_norm",
    "ape": "position_bias",
    # MoE router
    "bias": "e_score_correction_bias",
    # per-expert SwiGLU triple (w1 = gate, w3 = up, w2 = down)
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}

#: Whole-key renames for the model-level tensors.
ROOT_ALIASES: dict[str, str] = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_scale": "model.hc_head.hc_scale",
}

#: Per-layer path-segment renames applied before the leaf pass.
PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    (".attn.", ".self_attn."),
    (".ffn.", ".mlp."),
    (".attn_norm.", ".input_layernorm."),
    (".ffn_norm.", ".post_attention_layernorm."),
)

#: Bare per-layer tensors (no ``.weight``) that move into a submodule.
HC_ALIASES: tuple[tuple[str, str], ...] = (
    ("hc_attn_base", "attn_hc.base"),
    ("hc_attn_fn", "attn_hc.fn"),
    ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_base", "ffn_hc.base"),
    ("hc_ffn_fn", "ffn_hc.fn"),
    ("hc_ffn_scale", "ffn_hc.scale"),
)

#: Attention sink: native carries it under the attention module, EasyDeL as a
#: plain parameter on the attention module.
_SINK = ("attn.attn_sink", "self_attn.sinks")

#: The lightning indexer nests the OTHER way round in the two layouts: the
#: checkpoint puts the compressor inside the indexer (``attn.indexer.
#: compressor.wkv``) while EasyDeL puts the indexer inside the compressor
#: (``self_attn.compressor.indexer.kv_proj``), and routes the scoring
#: projection through a ``scorer`` submodule. Rewrite the indexer paths before
#: the generic ``.attn.`` -> ``.self_attn.`` pass so the inversion is applied
#: exactly once and the leaf pass still sees a normal trailing module name.
INDEXER_ALIASES: tuple[tuple[str, str], ...] = (
    ("attn.indexer.compressor.", "self_attn.compressor.indexer."),
    ("attn.indexer.weights_proj", "self_attn.compressor.indexer.scorer.weights_proj"),
    ("attn.indexer.", "self_attn.compressor.indexer."),
)

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def native_key_to_easydel(key: str) -> str | None:
    """Rewrite one native DeepSeek-V4 checkpoint key to its EasyDeL name.

    Args:
        key: Key as published in the checkpoint, e.g.
            ``layers.3.attn.wq_a.weight`` or
            ``layers.3.ffn.experts.17.w1.weight``.

    Returns:
        The EasyDeL parameter name, or ``None`` for tensors the runtime does
        not own (the ``mtp.*`` multi-token-prediction stack, which is a
        separate optional head and is not part of the causal-LM parameter
        tree). Returning ``None`` rather than raising keeps the loader's
        "unused checkpoint key" reporting meaningful.
    """
    if key.startswith("mtp."):
        return None
    if key in ROOT_ALIASES:
        return ROOT_ALIASES[key]

    m = _LAYER_RE.match(key)
    if m is None:
        return None
    idx, rest = m.group(1), key[m.end() :]

    for native, target in HC_ALIASES:
        if rest == native:
            return f"model.layers.{idx}.{target}"
    if rest == _SINK[0]:
        return f"model.layers.{idx}.{_SINK[1]}"

    for src, dst in INDEXER_ALIASES:
        if rest.startswith(src):
            rest = dst + rest[len(src) :]
            break

    rest = f".{rest}"
    for src, dst in PREFIX_ALIASES:
        rest = rest.replace(src, dst)
    rest = rest.lstrip(".")

    parts = rest.split(".")
    # `.weight` / `.scale` keep their suffix; the module name sits before it.
    if parts[-1] in ("weight", "scale") and len(parts) >= 2:
        parts[-2] = LEAF_ALIASES.get(parts[-2], parts[-2])
    else:
        parts[-1] = LEAF_ALIASES.get(parts[-1], parts[-1])
    return f"model.layers.{idx}." + ".".join(parts)
