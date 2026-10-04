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

"""Key mapping for DeepSeek-V4.1's native checkpoint layout.

V4.1 ships in DeepSeek's inference naming like V4, so the V4 normalizer handles attention, compressor, MoE, norms
and hyper-connection keys. The one structural difference is the indexer: V4 nests a compressor inside the indexer
(``attn.indexer.compressor.*``) and EasyDeL mirrors that inside ``self_attn.compressor``, whereas V4.1's indexer
has no compressor of its own (keys come from the KV source's latent through ``wk``) and lives directly on the
attention module. Its keys are therefore mapped here before falling back to the V4 rules.

Vision (``vision.*``, ``aligner.*``, image delimiters) and image-token router biases (``ffn.gate.bias_vl``) are not
owned by the text-only model and map to ``None`` in :func:`native_key_to_easydel_v41`; the vision-language model uses
:func:`native_key_to_easydel_v41_vl`, which maps them. The DSpark MTP stack (``mtp.*``) is unowned by both; :func:`native_key_to_easydel_v41_mtp` maps it for the
DSpark draft model (``modeling_deepseek_v41_dspark``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from easydel.modules.deepseek_v4._native_checkpoint import LEAF_ALIASES, native_key_to_easydel

_INDEXER_RE = re.compile(r"^layers\.(\d+)\.attn\.indexer\.([A-Za-z0-9_]+)\.(weight|scale)$")
_UNOWNED_ROOTS = ("image_start", "image_end", "image_newline")
_UNOWNED_PREFIXES = ("mtp.", "model.mtp.", "vision.", "aligner.")
_ENGRAM_TABLE_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")


def engram_table_checkpoint_keys(keys: Iterable[str]) -> dict[int, dict[str, str]]:
    """Find the engram table tensors (fp8 ``embed.weight`` and e8m0 ``embed.scale``) in a native key list.

    These are mapped to ``None`` by :func:`native_key_to_easydel_v41` so the streaming conversion never reads them
    (they are ~100 GB each and must not reach the fp32 quant decoder); the engram loader fills them instead.

    Args:
        keys: Native checkpoint keys, e.g. the ``weight_map`` of ``model.safetensors.index.json``.

    Returns:
        ``{layer_id: {"weight": key, "scale": key}}`` for every layer that has table keys.
    """
    found: dict[int, dict[str, str]] = {}
    for key in keys:
        m = _ENGRAM_TABLE_RE.match(key)
        if m is not None:
            found.setdefault(int(m.group(1)), {})[m.group(2)] = key
    return found


def native_key_to_easydel_v41(key: str) -> str | None:
    """Normalize a native (or already-normalized) DeepSeek-V4.1 checkpoint key.

    Args:
        key: Checkpoint key, e.g. ``layers.24.attn.indexer.wq_b.weight``.

    Returns:
        The EasyDeL parameter name, or ``None`` for keys the text model does not own.
    """
    if key.startswith(_UNOWNED_PREFIXES) or key in _UNOWNED_ROOTS or key.endswith(".ffn.gate.bias_vl"):
        return None
    if _ENGRAM_TABLE_RE.match(key) is not None:
        return None
    m = _INDEXER_RE.match(key)
    if m is not None:
        idx, leaf, suffix = m.groups()
        return f"model.layers.{idx}.self_attn.indexer.{LEAF_ALIASES.get(leaf, leaf)}.{suffix}"
    return native_key_to_easydel(key)


_VL_ROOTS = ("image_start", "image_end", "image_newline")
_VL_PREFIXES = ("vision.", "aligner.")
_VL_BIAS_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias_vl$")


def native_key_to_easydel_v41_vl(key: str) -> str | None:
    """Normalize a native DeepSeek-V4.1 key for the vision-language model.

    Everything the text normalizer maps is mapped identically. In addition ``vision.*`` / ``aligner.*`` and the
    image delimiters live under the backbone (``model.vision.*``, ``model.aligner.*``, ``model.image_start`` ...)
    and ``layers.N.ffn.gate.bias_vl`` becomes ``model.layers.N.mlp.gate.bias_vl``. ``mtp.*`` stays unowned.

    Args:
        key: Checkpoint key, e.g. ``vision.blocks.3.attn.wqkv.weight``.

    Returns:
        The EasyDeL parameter name, or ``None`` for keys the model does not own.
    """
    if key.startswith(("mtp.", "model.mtp.")):
        return None
    if key.startswith(_VL_PREFIXES) or key in _VL_ROOTS:
        return f"model.{key}"
    m = _VL_BIAS_RE.match(key)
    if m is not None:
        return f"model.layers.{m.group(1)}.mlp.gate.bias_vl"
    return native_key_to_easydel_v41(key)


_MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
_MTP_OWN_RE = re.compile(
    r"^(main_proj\.(weight|scale)|main_norm\.weight|norm\.weight|markov_head\.(embed|head)\.weight"
    r"|confidence_head\.proj\.weight)$"
)


def native_key_to_easydel_v41_mtp(key: str) -> str | None:
    """Normalize a native DeepSeek-V4.1 DSpark key (``mtp.N.*``) to the draft model's parameter names.

    Names are relative to :class:`~easydel.modules.deepseek_v41.DeepseekV41DSpark` (``stages.N.*``). Backbone-style
    block keys reuse the V4 rules. The tied ``mtp.N.embed`` / ``mtp.N.head`` (shared with the backbone, skipped by
    DeepSeek's ``convert.py``) and ``mtp.N.ffn.gate.bias_vl`` (never read by text drafts) are unowned.

    Args:
        key: Checkpoint key, e.g. ``mtp.0.attn.wq_a.weight`` or an already normalized ``stages.0.*`` key.

    Returns:
        The draft-model parameter name, or ``None`` for keys the draft model does not own.
    """
    if key.startswith("stages."):
        return key
    m = _MTP_RE.match(key)
    if m is None:
        return None
    stage, rest = m.groups()
    if rest == "ffn.gate.bias_vl" or rest.startswith(("embed.", "head.")):
        return None
    if _MTP_OWN_RE.match(rest):
        return f"stages.{stage}.{rest}"
    mapped = native_key_to_easydel(f"layers.{stage}.{rest}")
    prefix = f"model.layers.{stage}."
    if mapped is None or not mapped.startswith(prefix):
        return None
    return f"stages.{stage}.{mapped[len(prefix) :]}"


__all__ = [
    "engram_table_checkpoint_keys",
    "native_key_to_easydel_v41",
    "native_key_to_easydel_v41_mtp",
    "native_key_to_easydel_v41_vl",
]
