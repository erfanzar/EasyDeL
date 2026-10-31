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

"""Independent torch oracle for EasyDeL's DeepSeek-V4.1 model (pure torch, no JAX).

The oracle is DeepSeek's own reference ``model.py`` (``.xerxes/projects/deepseek-v41/ref``), loaded by file path
with its CUDA/tilelang dependencies replaced by pure-torch CPU fp32 stubs:

* ``kernel.sparse_attn`` follows ``kernel.py:311-403`` (gather ``topk_idxs`` from ``kv``, ``-1`` ignored, per-head sink
  in the softmax denominator only, scale applied to the scores, running max floored at ``-1e30``).
* ``kernel.hc_split_sinkhorn`` is a line-by-line port of the tilelang kernel (``kernel.py:406-480``).
* ``kernel.act_quant`` / ``fp4_act_quant`` are IDENTITIES. The reference fake-quantizes the window KV (fp8), the
  compressed KV (fp4, E4M3 scales) and the indexer q/k (fp4); EasyDeL does not emulate that, so the oracle compares the
  unquantized math. ``fp8_gemm`` / ``fp4_gemm`` raise: every weight here is fp32.
* ``engram`` is the real ``engram.py`` except ``NgramHashState`` (needs a tokenizer): hash ids are passed in.
* ``image_processor`` is a stub holding only the token-type constants of ``ref_dl/inference/image_processor.py``
  (``TEXT = -1``; ``IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)``; PIL is not installed). ``vision`` is the
  real ``ref/vision.py``. DSpark and MTP are disabled; vision is built only when the config enables it.

Deviations from the reference forward, all deliberate:

* ``ParallelEngramEmbedding`` stores fp8 rows + e8m0 scales and casts the looked-up rows to bf16. Here the table is
  fp32 with unit scales and the lookup stays fp32 (otherwise a bf16 rounding of ~4e-3 enters the stream).
* The reference indexer runs under ``torch.no_grad`` (its top-k is a hard selection; EasyDeL stop-gradients it too).
* ``reference_logits`` re-implements the loop of ``Transformer.forward`` (``start_pos=0``, all positions) because the
  original is ``inference_mode`` and returns last-position logits only. Reference state (KV caches, compressor
  state, ``shared_attn``) is reset on every call. With ``token_types`` / ``images`` it follows the VL branch of that
  forward (``image_mask = token_types >= 0``, ``engram_mask = ~image_mask``, ``merge_image_embeddings``).

The module imports no JAX and no EasyDeL: ``build_reference`` accepts any object with the ``DeepseekV41Config``
attributes.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib.util
import math
import pathlib
import re
import sys
import types
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REF_DIR = pathlib.Path(__file__).resolve().parents[4] / ".xerxes" / "projects" / "deepseek-v41" / "ref"

VOCAB = 256
CASE_BATCH = 2
MAX_SEQ_LEN = 64
CASES = ("plain", "engram", "yarn")
#: Parameter-draw seed per case (chosen so every top-k in the reference has a clear margin, see ``topk_margins``).
CASE_SEEDS: dict[str, int] = {"plain": 4, "engram": 35, "yarn": 39}
CASE_SEQ_LEN: dict[str, int] = {"plain": 31, "engram": 32, "yarn": 36}

#: Token types of ``image_processor.py``: text is ``-1``, every other type is a position inside an image span.
TOKEN_TEXT = -1
TOKEN_IMAGE_START, TOKEN_IMAGE, TOKEN_IMAGE_NEW_LINE, TOKEN_IMAGE_END = range(4)

_UNOWNED_PREFIXES = ("mtp.",)


# ----------------------------------------------------------------------------------------------------------------------
# Pure-torch stand-ins for the tilelang kernels
# ----------------------------------------------------------------------------------------------------------------------


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Sparse shared-KV attention with a per-head sink (``kernel.py:311-403``) in fp32.

    Args:
        q: ``[b, m, h, d]`` queries.
        kv: ``[b, n, d]`` keys == values.
        attn_sink: ``[h]`` sink logit; only enters the denominator and is not scaled.
        topk_idxs: ``[b, m, topk]`` int indices into ``kv``; ``-1`` slots are ignored.
        softmax_scale: Applied to the scores.

    Returns:
        ``[b, m, h, d]``; a query with no valid index gets zeros.
    """
    batch, _, heads, _ = q.shape
    idx = topk_idxs.long()
    valid = idx != -1
    safe = idx.clamp_min(0)
    rows = torch.arange(batch, device=q.device).view(batch, 1, 1)
    gathered = kv.float()[rows, safe]
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -math.inf)
    floor = torch.full_like(scores[..., :1], -1e30)
    peak = torch.maximum(scores.amax(dim=-1, keepdim=True).detach(), floor)
    weights = torch.exp(scores - peak)
    denom = weights.sum(dim=-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, heads, 1) - peak)
    out = torch.einsum("bmhk,bmkd->bmhd", weights, gathered) / denom
    return out.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of ``hc_split_sinkhorn_kernel`` (``kernel.py:406-480``) in fp32.

    ``mixes`` is ``[b, s, (2 + hc) * hc]``; returns ``pre [b, s, hc]``, ``post [b, s, hc]``, ``comb [b, s, hc, hc]``.
    """
    hc = hc_mult
    mixes = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)  # comb[j, k] = mixes[j * hc + k + 2 * hc]
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype=None, inplace: bool = False):
    """Identity: fp8 fake-quantization is not emulated (in-place form only, as the reference uses it)."""
    del block_size, scale_fmt, scale_dtype
    if not inplace:
        raise NotImplementedError("oracle act_quant only supports the in-place (fake-quant) form")
    return x


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False, scale_dtype=None):
    """Identity: fp4 fake-quantization is not emulated (in-place form only)."""
    del block_size, scale_dtype
    if not inplace:
        raise NotImplementedError("oracle fp4_act_quant only supports the in-place (fake-quant) form")
    return x


def fp8_gemm(*args, **kwargs):
    """Not available: the oracle keeps every weight in fp32."""
    raise NotImplementedError("fp8_gemm: oracle weights are fp32")


def fp4_gemm(*args, **kwargs):
    """Not available: the oracle keeps every weight in fp32."""
    raise NotImplementedError("fp4_gemm: oracle weights are fp32")


class _PrecomputedHashes(nn.Module):
    """Placeholder for ``NgramHashState`` (needs a tokenizer); hash ids are supplied by the caller."""

    def __init__(self, *args, **kwargs):
        super().__init__()


# ----------------------------------------------------------------------------------------------------------------------
# Loading the reference module
# ----------------------------------------------------------------------------------------------------------------------


def _load_by_path(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _engram_embedding_forward(self, indices: torch.Tensor) -> torch.Tensor:
    """``ParallelEngramEmbedding.forward`` without the bf16 cast (single rank: every row is local)."""
    values = F.embedding(indices, self.weight)
    scales = F.embedding(indices, self.scale)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    return values.flatten(-2)


@functools.lru_cache(maxsize=1)
def load_reference_module() -> types.ModuleType:
    """Import ``ref/model.py`` with stub ``kernel`` / ``image_processor`` modules, the real ``vision``, and patch it."""
    if not hasattr(torch, "float4_e2m1fn_x2"):  # referenced (never matched) by ``linear()`` on older torch
        torch.float4_e2m1fn_x2 = "float4_e2m1fn_x2"
    names = ("kernel", "engram", "image_processor", "vision")
    saved = {name: sys.modules.get(name) for name in names}
    previous_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # do not litter the reference directory with bytecode
    try:
        kernel = types.ModuleType("kernel")
        for fn in (sparse_attn, hc_split_sinkhorn, act_quant, fp4_act_quant, fp8_gemm, fp4_gemm):
            setattr(kernel, fn.__name__, fn)
        image_processor = types.ModuleType("image_processor")
        image_processor.TEXT = TOKEN_TEXT
        image_processor.IMAGE_START = TOKEN_IMAGE_START
        image_processor.IMAGE = TOKEN_IMAGE
        image_processor.IMAGE_NEW_LINE = TOKEN_IMAGE_NEW_LINE
        image_processor.IMAGE_END = TOKEN_IMAGE_END
        sys.modules.update(kernel=kernel, image_processor=image_processor)
        _load_by_path("engram", REF_DIR / "engram.py")
        _load_by_path("vision", REF_DIR / "vision.py")
        module = _load_by_path("_deepseek_v41_reference_model", REF_DIR / "model.py")
    finally:
        sys.dont_write_bytecode = previous_flag
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    module.NgramHashState = _PrecomputedHashes
    module.ParallelEngramEmbedding.forward = _engram_embedding_forward
    original_indexer_forward = module.Indexer.forward

    @functools.wraps(original_indexer_forward)
    def indexer_forward(self, *args, **kwargs):
        with torch.no_grad():  # hard top-k selection: no gradient, as in EasyDeL
            return original_indexer_forward(self, *args, **kwargs)

    module.Indexer.forward = indexer_forward
    return module


# ----------------------------------------------------------------------------------------------------------------------
# Case definitions (shared by the TPU test and the CPU probe)
# ----------------------------------------------------------------------------------------------------------------------


def case_config_kwargs(case: str) -> dict[str, tp.Any]:
    """``DeepseekV41Config`` kwargs of a test case.

    Schedule (9 backbone layers): ratios ``[0, 2, 2, 2, 1, 1, 1, 1, 0]``; KV sources 1 (ratio 2) and 4 (ratio 1);
    index sources 1, 3, 4, 6, 7 (layer 3 scores against layer 1's index keys, layer 2 reuses layer 1's top-k, layer 5
    reuses layer 4's); candidate source 4, candidate-using index sources 6 and 7. ``index_topk=5`` is below the
    reachable entry count and ``4 * 2 - 2 + 1 = 7 >= index_topk`` keeps every selection tie-free (see
    ``check_selection_schedule``).
    """
    if case not in CASES:
        raise KeyError(f"unknown case {case!r}; expected one of {CASES}")
    kwargs: dict[str, tp.Any] = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=9,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        compress_ratios=[0, 2, 2, 2, 1, 1, 1, 1, 0],
        kv_source_layer_ids=[1, 4],
        index_source_layer_ids=[1, 3, 4, 6, 7],
        candidate_source_layer_id=4,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        index_n_heads=8,
        index_head_dim=16,
        index_topk=5,
        hc_mult=2,
        hc_sinkhorn_iters=6,
        sliding_window=4,
        rms_norm_eps=1e-6,
        engram_layer_ids=[],
        dspark_target_layer_ids=[6, 7, 8],
        dspark_noise_token_id=5,
        image_token_id=7,
        max_position_embeddings=256,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    if case == "engram":
        kwargs.update(
            engram_layer_ids=[1, 5],
            engram_vocab_size=50,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_max_ngram_size=3,
            engram_compressed_vocab_size=VOCAB,
            engram_pad_token_id=2,
        )
    if case == "yarn":
        kwargs["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        }
    return kwargs


def case_input_ids(config, case: str) -> np.ndarray:
    """Deterministic ``int64[CASE_BATCH, CASE_SEQ_LEN[case]]`` token ids of a case."""
    rng = np.random.default_rng(1000)  # fixed: the CASE_SEEDS margin search ran with these ids
    return rng.integers(0, int(config.vocab_size), size=(CASE_BATCH, CASE_SEQ_LEN[case]), dtype=np.int64)


def check_selection_schedule(config, seq_len: int) -> list[str]:
    """Static preconditions that make the oracle comparison meaningful; returns a list of problems."""
    problems: list[str] = []
    n = config.num_hidden_layers
    ratios = [int(config.layer_compress_ratio(i)) for i in range(n)]
    cand = config.candidate_source_layer_id
    if cand is not None and cand >= 0:
        cover = config.candidate_topk_blocks * config.candidate_block_size - config.candidate_block_size + 1
        if cover < config.index_topk:
            problems.append(
                f"candidate coverage {cover} < index_topk {config.index_topk}: the reference would pick reachable "
                "non-candidate (-inf) entries through top-k ties"
            )
        if not any(config.uses_candidates(i) for i in range(n)):
            problems.append("no index source consumes the candidate blocks")
    for ratio in sorted({r for r in ratios if r > 0}):
        entries = seq_len // ratio
        if config.index_topk >= entries:
            problems.append(f"ratio {ratio}: index_topk {config.index_topk} >= {entries} entries, nothing is selected")
    if len({r for r in ratios if r > 0}) < 2:
        problems.append("need at least two distinct compression ratios")
    if not any(r > 0 and not config.is_kv_source(i) for i, r in enumerate(ratios)):
        problems.append("no compressed layer reuses another layer's KV")
    if not any(r > 0 and not config.is_index_source(i) for i, r in enumerate(ratios)):
        problems.append("no compressed layer reuses another layer's top-k")
    return problems


# ----------------------------------------------------------------------------------------------------------------------
# Vision-language case (token types and image spans follow ``ref_dl/inference/image_processor.py``)
# ----------------------------------------------------------------------------------------------------------------------

#: Parameter-draw seed of the VL case (chosen so every top-k, image-token routing included, has a clear margin).
VL_SEED: int = 0
VL_SEQ_LEN = 32
#: Patch grid ``(n_vit_h, n_vit_w)`` of the test image; neither side divides by the aligner's 3, so both are padded.
VL_GRID = (5, 4)
#: Position of ``IMAGE_START`` in batch row 0 (row 1 is text only).
VL_IMAGE_START = 3


class ImageInput(tp.NamedTuple):
    """One image of a sample: the attributes ``Transformer.merge_image_embeddings`` reads."""

    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


class VLInputs(tp.NamedTuple):
    """Inputs of the VL case: ids / token types ``[B, S]``, patches ``[P, 3, patch, patch]`` and the patch grid."""

    ids: np.ndarray
    token_types: np.ndarray
    patches: np.ndarray
    grid: tuple[int, int]


def vl_config_kwargs() -> dict[str, tp.Any]:
    """``DeepseekV41Config`` kwargs of the VL case: the engram text case plus a tiny vision tower."""
    kwargs = case_config_kwargs("engram")
    kwargs["vision_config"] = {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "intermediate_size": 96,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
    }
    return kwargs


def vl_inputs(config) -> VLInputs:
    """Deterministic VL inputs: row 0 holds one image span, row 1 is text only.

    The span is ``[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]`` (every position
    carries ``image_token_id``) at ``VL_IMAGE_START``; patches are uniform in ``[-1, 1]`` in row-major patch order.
    """
    rng = np.random.default_rng(2000)
    params = config.vision_params()
    ratio, patch = params["downsample_ratio"], params["patch_size"]
    n_h, n_w = VL_GRID
    n_llm_h, n_llm_w = -(-n_h // ratio), -(-n_w // ratio)
    span = [TOKEN_IMAGE_START] + ([TOKEN_IMAGE] * n_llm_w + [TOKEN_IMAGE_NEW_LINE]) * n_llm_h + [TOKEN_IMAGE_END]
    ids = rng.integers(0, int(config.vocab_size), size=(CASE_BATCH, VL_SEQ_LEN), dtype=np.int64)
    token_types = np.full(ids.shape, TOKEN_TEXT, dtype=np.int64)
    token_types[0, VL_IMAGE_START : VL_IMAGE_START + len(span)] = span
    ids[token_types >= 0] = int(config.image_token_id)
    patches = rng.uniform(-1.0, 1.0, size=(n_h * n_w, 3, patch, patch)).astype(np.float32)
    return VLInputs(ids, token_types, patches, (n_h, n_w))


def vl_reference_images(inputs: VLInputs) -> list[list[ImageInput] | None]:
    """Per-sample image lists for ``merge_image_embeddings`` (row 0: one image, row 1: none)."""
    types = inputs.token_types[0]
    span = np.flatnonzero(types >= 0)
    image = ImageInput(
        int(span[0]), torch.from_numpy(inputs.patches), inputs.grid[0], inputs.grid[1], torch.from_numpy(types[span])
    )
    return [[image], None]


# ----------------------------------------------------------------------------------------------------------------------
# EasyDeL parameter table, random draws and the EasyDeL -> reference mapping
# ----------------------------------------------------------------------------------------------------------------------


def easydel_param_shapes(config) -> dict[str, tuple[int, ...]]:
    """Names and shapes of every ``DeepseekV41ForCausalLM`` parameter as EasyDeL exports them (kernels are [in, out])."""
    d, hd = config.hidden_size, config.head_dim
    heads, hc, m, e = config.num_attention_heads, config.hc_mult, config.moe_intermediate_size, config.n_routed_experts
    mix = (2 + hc) * hc
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, d),
        "model.norm.weight": (d,),
        "lm_head.weight": (d, config.vocab_size),
    }
    vision = bool(getattr(config, "vision_enabled", False))
    engram_ids = list(config.engram_layer_ids)
    rows = config.engram_table_rows() if engram_ids else []
    cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        shapes[p + "input_layernorm.weight"] = (d,)
        shapes[p + "post_attention_layernorm.weight"] = (d,)
        for hc_name in ("attn_hc", "ffn_hc"):
            shapes[f"{p}{hc_name}.fn"] = (mix, hc * d)
            shapes[f"{p}{hc_name}.base"] = (mix,)
            shapes[f"{p}{hc_name}.scale"] = (3,)
        a = p + "self_attn."
        shapes[a + "q_a_proj.weight"] = (d, config.q_lora_rank)
        shapes[a + "q_a_norm.weight"] = (config.q_lora_rank,)
        shapes[a + "q_b_proj.weight"] = (config.q_lora_rank, heads * hd)
        shapes[a + "kv_proj.weight"] = (d, hd)
        shapes[a + "kv_norm.weight"] = (hd,)
        shapes[a + "o_a_proj.weight"] = (heads * hd // config.o_groups, config.o_groups * config.o_lora_rank)
        shapes[a + "o_b_proj.weight"] = (config.o_groups * config.o_lora_rank, d)
        shapes[a + "sinks"] = (heads,)
        if config.is_kv_source(i):
            shapes[a + "compressor.kv_proj.weight"] = (d, hd)
            if config.layer_compress_ratio(i) > 1:
                shapes[a + "compressor.gate_proj.weight"] = (d, hd)
            shapes[a + "compressor.kv_norm.weight"] = (hd,)
        if config.is_index_source(i):
            shapes[a + "indexer.q_b_proj.weight"] = (config.q_lora_rank, config.index_n_heads * config.index_head_dim)
            shapes[a + "indexer.weights_proj.weight"] = (d, config.index_n_heads)
            if config.owns_index_keys(i):
                shapes[a + "indexer.wk.weight"] = (hd, config.index_head_dim)
                shapes[a + "indexer.k_norm.weight"] = (config.index_head_dim,)
        mlp = p + "mlp."
        shapes[mlp + "gate.weight"] = (d, e)
        shapes[mlp + "gate.e_score_correction_bias"] = (e,)
        if vision:
            shapes[mlp + "gate.bias_vl"] = (e,)
        shapes[mlp + "experts.gate_proj.weight"] = (e, d, m)
        shapes[mlp + "experts.up_proj.weight"] = (e, d, m)
        shapes[mlp + "experts.down_proj.weight"] = (e, m, d)
        shapes[mlp + "shared_experts.gate_up_proj.weight"] = (d, 2 * m)
        shapes[mlp + "shared_experts.down_proj.weight"] = (m, d)
        if i in engram_ids:
            slot = engram_ids.index(i)
            shapes[p + "engram.embed.weight"] = (rows[slot], config.engram_head_dim)
            shapes[p + "engram.kv_proj.weight"] = (cols * config.engram_head_dim, d * (hc + 1))
            shapes[p + "engram.q_weight"] = (hc, d)
            shapes[p + "engram.k_weight"] = (hc, d)
    if vision:
        shapes.update(_vision_param_shapes(config))
    return shapes


def _vision_param_shapes(config) -> dict[str, tuple[int, ...]]:
    """Names and shapes of the vision tower, aligner and image delimiters of ``DeepseekV41ForConditionalGeneration``."""
    vp = config.vision_params()
    d, hid, inter, patch = config.hidden_size, vp["hidden_size"], vp["intermediate_size"], vp["patch_size"]
    fold = vp["downsample_ratio"] ** 2
    shapes: dict[str, tuple[int, ...]] = {
        "model.vision.patch_embed.proj.weight": (3 * patch * patch, hid),
        "model.vision.patch_embed.proj.bias": (hid,),
        "model.vision.norm.weight": (hid,),
        "model.aligner.w1.weight": (hid * fold, d),
        "model.aligner.w1.bias": (d,),
        "model.aligner.w2.weight": (d, d),
        "model.aligner.w2.bias": (d,),
        "model.image_start": (d,),
        "model.image_end": (d,),
        "model.image_newline": (d,),
    }
    for i in range(vp["num_hidden_layers"]):
        b = f"model.vision.blocks.{i}."
        shapes[b + "norm1.weight"] = (hid,)
        shapes[b + "norm2.weight"] = (hid,)
        shapes[b + "attn.wqkv.weight"] = (hid, 3 * hid)
        shapes[b + "attn.wqkv.bias"] = (3 * hid,)
        shapes[b + "attn.wo.weight"] = (hid, hid)
        shapes[b + "attn.wo.bias"] = (hid,)
        shapes[b + "mlp.w1.weight"] = (hid, 2 * inter)
        shapes[b + "mlp.w2.weight"] = (inter, hid)
    return shapes


def _draw(rng: np.random.Generator, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """One random parameter, scaled so activations are O(1) and every selection is sharp."""
    if name.endswith(("norm.weight", "layernorm.weight", "norm1.weight", "norm2.weight")):
        value = 1.0 + 0.2 * rng.standard_normal(shape)
    elif name.endswith(".sinks"):
        value = rng.standard_normal(shape)
    elif name.endswith(("e_score_correction_bias", "bias_vl")):
        value = 0.5 * rng.standard_normal(shape)
    elif name.endswith(("attn_hc.fn", "ffn_hc.fn")):
        value = rng.standard_normal(shape) / math.sqrt(shape[1])
    elif name.endswith(("attn_hc.base", "ffn_hc.base")):
        value = 0.5 * rng.standard_normal(shape)
    elif name.endswith(("attn_hc.scale", "ffn_hc.scale")):
        value = 1.0 + 0.2 * rng.standard_normal(shape)
    elif name.endswith(("engram.q_weight", "engram.k_weight")):
        value = 1.0 + 0.3 * rng.standard_normal(shape)
    elif name.endswith(("embed_tokens.weight", "engram.embed.weight")):
        value = rng.standard_normal(shape)
    elif name.endswith(("indexer.q_b_proj.weight", "indexer.weights_proj.weight")):
        value = 4.0 * rng.standard_normal(shape) / math.sqrt(shape[0])  # widen the indexer score spread
    elif name.endswith("mlp.gate.weight"):
        value = 3.0 * rng.standard_normal(shape) / math.sqrt(shape[0])  # widen the router score spread
    elif name.endswith(".bias"):  # vision / aligner linear biases
        value = 0.1 * rng.standard_normal(shape)
    elif name.endswith(("image_start", "image_end", "image_newline")):
        value = rng.standard_normal(shape)
    elif len(shape) == 3:  # stacked experts [E, in, out]
        value = rng.standard_normal(shape) / math.sqrt(shape[1])
    else:  # [in, out] kernels
        value = rng.standard_normal(shape) / math.sqrt(shape[0])
    return value.astype(np.float32)


def random_easydel_params(config, seed: int) -> dict[str, np.ndarray]:
    """Deterministic random fp32 values for every EasyDeL parameter (overrides EasyDeL's near-zero init)."""
    rng = np.random.default_rng(seed)
    return {name: _draw(rng, name, shape) for name, shape in sorted(easydel_param_shapes(config).items())}


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

#: EasyDeL leaf (relative to the layer) -> reference leaf, same orientation.
_DIRECT = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn",
    "attn_hc.base": "hc_attn_base",
    "attn_hc.scale": "hc_attn_scale",
    "ffn_hc.fn": "hc_ffn_fn",
    "ffn_hc.base": "hc_ffn_base",
    "ffn_hc.scale": "hc_ffn_scale",
    "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
    "self_attn.sinks": "attn.attn_sink",
    "self_attn.compressor.kv_norm.weight": "attn.compressor.norm.weight",
    "self_attn.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
    "mlp.gate.bias_vl": "ffn.gate.bias_vl",
    "engram.embed.weight": "engram.embed.weight",
    "engram.q_weight": "engram.q_weight",
    "engram.k_weight": "engram.k_weight",
}
#: EasyDeL [in, out] kernel -> reference [out, in] ``Linear.weight`` (transpose).
_TRANSPOSED = {
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_proj.weight": "attn.wkv.weight",
    "self_attn.o_a_proj.weight": "attn.wo_a.weight",
    "self_attn.o_b_proj.weight": "attn.wo_b.weight",
    "self_attn.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
    "self_attn.compressor.gate_proj.weight": "attn.compressor.wgate.weight",
    "self_attn.indexer.q_b_proj.weight": "attn.indexer.wq_b.weight",
    "self_attn.indexer.weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight": "attn.indexer.wk.weight",
    "mlp.gate.weight": "ffn.gate.weight",
    "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    "engram.kv_proj.weight": "engram.wkv.weight",
}
_EXPERT_LEAF = {
    "mlp.experts.gate_proj.weight": "w1",
    "mlp.experts.up_proj.weight": "w3",
    "mlp.experts.down_proj.weight": "w2",
}


def easydel_to_reference(name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Map one EasyDeL parameter (or a same-shaped gradient) to reference ``(name, array)`` pairs.

    Raises:
        KeyError: For an EasyDeL name the oracle does not know.
    """
    array = np.asarray(array)
    if name == "model.embed_tokens.weight":
        return [("embed.weight", array)]
    if name == "model.norm.weight":
        return [("norm.weight", array)]
    if name == "lm_head.weight":
        return [("head.weight", array.T)]
    if name in ("model.image_start", "model.image_end", "model.image_newline"):
        return [(name[len("model.") :], array)]
    if name.startswith(("model.vision.", "model.aligner.")):
        ref_name = name[len("model.") :]
        if ref_name.endswith((".bias", "norm.weight", "norm1.weight", "norm2.weight")):
            return [(ref_name, array)]
        return [(ref_name, array.T)]  # [in, out] kernel -> [out, in] Linear.weight
    match = _LAYER_RE.match(name)
    if match is None:
        raise KeyError(f"unmapped EasyDeL parameter {name!r}")
    prefix, rest = f"layers.{int(match.group(1))}.", match.group(2)
    if rest in _DIRECT:
        return [(prefix + _DIRECT[rest], array)]
    if rest in _TRANSPOSED:
        return [(prefix + _TRANSPOSED[rest], array.T)]
    if rest == "mlp.shared_experts.gate_up_proj.weight":  # [D, 2M]: gate columns first (tp=1 split), then up
        half = array.shape[1] // 2
        return [
            (prefix + "ffn.shared_experts.w1.weight", array[:, :half].T),
            (prefix + "ffn.shared_experts.w3.weight", array[:, half:].T),
        ]
    if rest in _EXPERT_LEAF:  # [E, in, out] -> per-expert [out, in]
        leaf = _EXPERT_LEAF[rest]
        return [(f"{prefix}ffn.experts.{e}.{leaf}.weight", array[e].T) for e in range(array.shape[0])]
    raise KeyError(f"unmapped EasyDeL parameter {name!r}")


# ----------------------------------------------------------------------------------------------------------------------
# Building and running the reference
# ----------------------------------------------------------------------------------------------------------------------


def model_args_from_config(
    config, *, max_batch_size: int = CASE_BATCH, max_seq_len: int = MAX_SEQ_LEN, dspark: bool = False
):
    """Reference ``ModelArgs`` built from a ``DeepseekV41Config`` (vision only if enabled).

    DSpark and MTP are disabled unless ``dspark=True``: then the config's ``num_nextn_predict_layers`` MTP stages are
    built with its ``dspark_*`` fields and ``temperature=0`` (greedy drafting).
    """
    module = load_reference_module()
    rope = config.v4_rope_params("compress")
    yarn = rope.get("rope_type", "default") == "yarn"
    if rope.get("attention_factor", 1.0) != 1.0 or not rope.get("truncate", True):
        raise NotImplementedError("the reference only supports yarn without mscale and with truncation")
    original = int(rope.get("original_max_position_embeddings") or config.max_position_embeddings) if yarn else 0
    cand = config.candidate_source_layer_id
    engram_ids = tuple(int(i) for i in config.engram_layer_ids)
    n_mtp = 0
    dspark_kwargs: dict[str, tp.Any] = {}
    if dspark:
        n_mtp = int(config.num_nextn_predict_layers)
        dspark_kwargs = dict(
            temperature=0.0,
            dspark_block_size=int(config.dspark_block_size),
            dspark_noise_token_id=int(config.dspark_noise_token_id),
            dspark_target_layer_ids=tuple(int(i) for i in config.dspark_target_layer_ids),
            dspark_markov_rank=int(config.dspark_markov_rank),
            dspark_n_routed_experts=int(config.dspark_n_routed_experts),
            dspark_n_activated_experts=int(config.dspark_num_experts_per_tok),
        )
    vision_kwargs: dict[str, tp.Any] = {}
    if getattr(config, "vision_enabled", False):
        vp = config.vision_params()
        vision_kwargs = dict(
            vision_n_layers=vp["num_hidden_layers"],
            vision_dim=vp["hidden_size"],
            vision_n_heads=vp["num_attention_heads"],
            vision_inter_dim=vp["intermediate_size"],
            vision_patch_size=vp["patch_size"],
            vision_rope_theta=vp["rope_theta"],
            vision_downsample_ratio=vp["downsample_ratio"],
            vision_max_n_token=vp["max_image_tokens"],
            vision_min_pixels=vp["min_pixels"],
            vision_max_wh_ratio=vp["max_wh_ratio"],
        )
    return module.ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_layers=config.num_hidden_layers,
        n_mtp_layers=n_mtp,
        n_heads=config.num_attention_heads,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        score_func=config.scoring_func,
        gate_temp=float(config.gate_temp),
        norm_topk_prob=bool(config.norm_topk_prob),
        route_scale=float(config.routed_scaling_factor),
        swiglu_limit=float(config.swiglu_limit),
        q_lora_rank=config.q_lora_rank,
        head_dim=config.head_dim,
        rope_head_dim=config.qk_rope_head_dim,
        norm_eps=float(config.rms_norm_eps),
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        window_size=config.sliding_window,
        compress_ratios=tuple(int(r) for r in config.compress_ratios),
        kv_source_layers=tuple(int(i) for i in config.kv_source_layer_ids),
        index_source_layers=tuple(int(i) for i in config.index_source_layer_ids),
        compress_rope_theta=float(config.compress_rope_theta),
        original_seq_len=original,
        rope_theta=float(config.rope_theta),
        rope_factor=float(rope.get("factor") or 1.0) if yarn else 40,
        beta_fast=int(rope.get("beta_fast") or 32),
        beta_slow=int(rope.get("beta_slow") or 1),
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        candidate_source_layer=-1 if cand is None else int(cand),
        candidate_topk_blocks=config.candidate_topk_blocks,
        candidate_block_size=config.candidate_block_size,
        hc_mult=config.hc_mult,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=float(config.hc_eps),
        engram_layer_ids=engram_ids,
        engram_num_embeddings=tuple(config.engram_table_rows()) if engram_ids else (),
        engram_max_ngram_size=config.engram_max_ngram_size,
        engram_vocab_size=config.engram_vocab_size,
        engram_n_heads=config.engram_n_heads,
        engram_head_dim=config.engram_head_dim,
        engram_pad_id=config.engram_pad_token_id,
        engram_compressed_vocab_size=config.engram_compressed_vocab_size,
        image_token_id=config.image_token_id,
        **vision_kwargs,
        **dspark_kwargs,
    )


def build_reference(config, *, max_batch_size: int = CASE_BATCH, max_seq_len: int = MAX_SEQ_LEN, dspark: bool = False):
    """Build the reference ``Transformer`` with fp32 parameters (uninitialized until ``load_easydel_params``).

    With ``dspark=True`` the MTP stages (``mtp.N``) are built too; load them with ``load_dspark_params``.
    """
    module = load_reference_module()
    args = model_args_from_config(config, max_batch_size=max_batch_size, max_seq_len=max_seq_len, dspark=dspark)
    model = module.Transformer(args)
    prewritten: set[str] = set()
    for index, layer in enumerate(model.layers):
        if layer.engram is None:
            continue
        embed = layer.engram.embed
        rows, dim = embed.weight.shape
        embed.weight = nn.Parameter(torch.zeros(rows, dim, dtype=torch.float32))
        embed.scale = nn.Parameter(torch.ones(rows, dim // embed.block_size, dtype=torch.float32), requires_grad=False)
        prewritten.add(f"layers.{index}.engram.embed.scale")  # dequant scales are unit, not an EasyDeL parameter
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    model._harness_prewritten = prewritten
    model.eval()
    return model


def load_easydel_params(ref_model, flat_easydel_params: dict[str, np.ndarray]) -> list[str]:
    """Copy every EasyDeL parameter into its reference tensor and prove nothing was left unwritten.

    Raises:
        KeyError: If an EasyDeL name has no reference counterpart.
        ValueError: On a shape mismatch.
        AssertionError: If a reference parameter (vision/MTP excluded) was written zero or more than one time.
    """
    params = dict(ref_model.named_parameters())
    written: collections.Counter[str] = collections.Counter()
    for name in sorted(flat_easydel_params):
        for ref_name, value in easydel_to_reference(name, np.asarray(flat_easydel_params[name])):
            if ref_name not in params:
                raise KeyError(f"EasyDeL parameter {name!r} maps to {ref_name!r}, absent from the reference model")
            target = params[ref_name]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if tuple(target.shape) != value.shape:
                raise ValueError(f"{name!r} -> {ref_name!r}: shape {value.shape} != reference {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(torch.from_numpy(value))
            written[ref_name] += 1
    expected = {n for n in params if not n.startswith(_UNOWNED_PREFIXES)} - set(
        getattr(ref_model, "_harness_prewritten", ())
    )
    missing = sorted(expected - set(written))
    repeated = sorted(n for n, count in written.items() if count != 1)
    if missing or repeated:
        raise AssertionError(
            f"reference parameters never written: {missing[:8]}; written more than once: {repeated[:8]}"
        )
    return sorted(written)


def _reset_state(ref_model, module) -> None:
    """Fresh caches / compressor state / shared-attention slots (the reference is stateful)."""
    for sub in ref_model.modules():
        for name, buffer in list(sub._buffers.items()):
            if buffer is None or name == "freqs_cis":
                continue
            fill = -math.inf if name == "score_state" else 0.0
            sub._buffers[name] = torch.full_like(buffer.detach(), fill)
    shared = module.shared_attn
    shared.compress_kv = shared.index_k = shared.topk_idxs = shared.candidates = None


class TopkRecord(tp.NamedTuple):
    """One ``Tensor.topk`` call made by the reference indexer / router / candidate-block selector."""

    kind: str
    k: int
    scores: torch.Tensor


@contextlib.contextmanager
def _record_topk(records: list[TopkRecord]):
    original = torch.Tensor.topk

    def topk(self, *args, **kwargs):
        frame = sys._getframe(1)
        owner, fn = frame.f_locals.get("self"), frame.f_code.co_name
        kind = None
        if fn == "forward" and type(owner).__name__ == "Indexer":
            kind = "index"
        elif fn == "forward" and type(owner).__name__ == "Gate":
            kind = "router"
        elif fn == "select_candidate_blocks":
            kind = "candidate_block"
        result = original(self, *args, **kwargs)
        if kind is not None:
            k = args[0] if args else kwargs["k"]
            records.append(TopkRecord(kind, int(k), self.detach().clone()))
        return result

    torch.Tensor.topk = topk
    try:
        yield
    finally:
        torch.Tensor.topk = original


def reference_logits(
    ref_model,
    input_ids,
    engram_hash_ids=None,
    *,
    requires_grad: bool = False,
    capture: list[TopkRecord] | None = None,
    token_types=None,
    images=None,
) -> torch.Tensor:
    """fp32 logits ``[B, S, V]`` for ALL positions (prefill, ``start_pos=0``), as ``Transformer.forward`` computes them.

    Args:
        ref_model: Model from ``build_reference`` + ``load_easydel_params``.
        input_ids: ``[B, S]`` token ids.
        engram_hash_ids: ``[B, S, n_engram_layers, n_cols]`` hash ids (``EngramHasher.hash``) or ``None``.
        requires_grad: Keep the autograd graph (for ``loss.backward()``); otherwise run under ``no_grad``.
        capture: If given, receives a ``TopkRecord`` for every indexer / router / candidate-block top-k.
        token_types: Optional ``[B, S]`` int token types (``-1`` text). ``image_mask = token_types >= 0`` selects
            ``bias_vl`` in every router and ``~image_mask`` is the engram ``token_mask``.
        images: Optional per-sample lists of ``ImageInput`` merged into the embeddings (``merge_image_embeddings``).
    """
    module = load_reference_module()
    _reset_state(ref_model, module)
    ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long)
    hashes = None if engram_hash_ids is None else torch.as_tensor(np.asarray(engram_hash_ids), dtype=torch.long)
    grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
    record_ctx = contextlib.nullcontext() if capture is None else _record_topk(capture)
    image_mask = None if token_types is None else torch.as_tensor(np.asarray(token_types), dtype=torch.long) >= 0
    engram_mask = None if image_mask is None else ~image_mask
    with grad_ctx, record_ctx:
        h = ref_model.embed(ids)
        if images is not None:
            ref_model.merge_image_embeddings(images, h)
        h = h.unsqueeze(2).repeat(1, 1, ref_model.hc_mult, 1)
        pre_mix = module.make_identity_pre_mix(h, ref_model.hc_mult)
        layer = None
        for layer in ref_model.layers:
            if layer.engram is not None:
                if hashes is None:
                    raise ValueError("engram layers need engram_hash_ids")
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
            h, pre_mix = layer(h, 0, pre_mix, image_mask)
        h = layer.hc_pre(h, pre_mix)
        return ref_model.head(ref_model.norm(h), full_logits=True)


def reference_loss_and_grads(ref_model, input_ids, engram_hash_ids=None) -> tuple[float, dict[str, np.ndarray]]:
    """Next-token cross-entropy (mean over ``[:, :-1]`` vs ``ids[:, 1:]``) and torch-autograd parameter gradients."""
    ref_model.zero_grad(set_to_none=True)
    logits = reference_logits(ref_model, input_ids, engram_hash_ids, requires_grad=True)
    ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    grads = {n: p.grad.detach().numpy().copy() for n, p in ref_model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def topk_margins(records: list[TopkRecord]) -> dict[str, dict[str, float]]:
    """Per kind: smallest gap between the k-th and (k+1)-th score, and how many rows really select.

    ``min_gap`` is ``inf`` when no call had more than ``k`` candidates (no real selection). Rows whose k-th and
    (k+1)-th scores are both ``-inf`` (unreachable entries) do not count as ties. ``selecting_rows`` counts rows with
    more than ``k`` finite candidates, i.e. rows where the top-k really discards something.
    """
    summary: dict[str, dict[str, float]] = {}
    for record in records:
        scores = record.scores.reshape(-1, record.scores.shape[-1]).double()
        entry = summary.setdefault(record.kind, {"min_gap": math.inf, "selecting_rows": 0, "rows": 0, "calls": 0})
        entry["calls"] += 1
        entry["rows"] += scores.shape[0]
        if record.k >= scores.shape[-1]:
            continue
        top = torch.sort(scores, dim=-1, descending=True).values
        gap = top[:, record.k - 1] - top[:, record.k]
        gap = torch.where(torch.isnan(gap), torch.full_like(gap, math.inf), gap)
        entry["min_gap"] = min(entry["min_gap"], float(gap.min()))
        entry["selecting_rows"] += int((top[:, record.k] > -math.inf).sum())
    return summary


# ----------------------------------------------------------------------------------------------------------------------
# DSpark (MTP draft model): parameter table, mapping onto ``mtp.N.*``, loader and ``forward_spec`` runner
# ----------------------------------------------------------------------------------------------------------------------

#: Parameter-draw seed of the DSpark case (chosen so router top-k and greedy draft arg-max have clear margins).
DSPARK_SEED: int = 4
#: Prefill lengths of the DSpark cases: 2 < window (partially valid ring) and 6 > window 4 (ring wraps at seeding).
DSPARK_PREFILL_LENS = (2, 6)
DSPARK_NUM_STEPS = 2


def dspark_config_kwargs() -> dict[str, tp.Any]:
    """``DeepseekV41Config`` kwargs of the DSpark case: the plain text case with 2 stages, block 3, 2 target layers.

    Targets are backbone layers 6 and 8 (``main_hidden`` width ``2 * hidden_size``); the draft MoE has 4 routed experts
    / top-2 (the backbone has 8 / top-2) so ``dspark_n_routed_experts`` is exercised; markov rank is 8.
    """
    kwargs = case_config_kwargs("plain")
    kwargs.update(
        num_nextn_predict_layers=2,
        dspark_block_size=3,
        dspark_target_layer_ids=[6, 8],
        dspark_noise_token_id=5,
        dspark_markov_rank=8,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    return kwargs


def dspark_param_shapes(config) -> dict[str, tuple[int, ...]]:
    """Names and shapes of every ``DeepseekV41DSpark`` parameter as EasyDeL exports them (kernels are [in, out])."""
    d, hd = config.hidden_size, config.head_dim
    heads, hc, m = config.num_attention_heads, config.hc_mult, config.moe_intermediate_size
    e = int(config.dspark_n_routed_experts or config.n_routed_experts)
    rank = int(config.dspark_markov_rank)
    mix = (2 + hc) * hc
    stages = int(config.num_nextn_predict_layers)
    shapes: dict[str, tuple[int, ...]] = {}
    for s in range(stages):
        p = f"stages.{s}."
        shapes[p + "input_layernorm.weight"] = (d,)
        shapes[p + "post_attention_layernorm.weight"] = (d,)
        for hc_name in ("attn_hc", "ffn_hc"):
            shapes[f"{p}{hc_name}.fn"] = (mix, hc * d)
            shapes[f"{p}{hc_name}.base"] = (mix,)
            shapes[f"{p}{hc_name}.scale"] = (3,)
        a = p + "self_attn."
        shapes[a + "q_a_proj.weight"] = (d, config.q_lora_rank)
        shapes[a + "q_a_norm.weight"] = (config.q_lora_rank,)
        shapes[a + "q_b_proj.weight"] = (config.q_lora_rank, heads * hd)
        shapes[a + "kv_proj.weight"] = (d, hd)
        shapes[a + "kv_norm.weight"] = (hd,)
        shapes[a + "o_a_proj.weight"] = (heads * hd // config.o_groups, config.o_groups * config.o_lora_rank)
        shapes[a + "o_b_proj.weight"] = (config.o_groups * config.o_lora_rank, d)
        shapes[a + "sinks"] = (heads,)
        mlp = p + "mlp."
        shapes[mlp + "gate.weight"] = (d, e)
        shapes[mlp + "gate.e_score_correction_bias"] = (e,)
        shapes[mlp + "experts.gate_proj.weight"] = (e, d, m)
        shapes[mlp + "experts.up_proj.weight"] = (e, d, m)
        shapes[mlp + "experts.down_proj.weight"] = (e, m, d)
        shapes[mlp + "shared_experts.gate_up_proj.weight"] = (d, 2 * m)
        shapes[mlp + "shared_experts.down_proj.weight"] = (m, d)
        if s == 0:
            shapes[p + "main_proj.weight"] = (len(config.dspark_target_layer_ids) * d, d)
            shapes[p + "main_norm.weight"] = (d,)
        if s == stages - 1:
            shapes[p + "norm.weight"] = (d,)
            shapes[p + "markov_head.embed.weight"] = (config.vocab_size, rank)
            shapes[p + "markov_head.head.weight"] = (rank, config.vocab_size)
            shapes[p + "confidence_head.proj.weight"] = (d + rank, 1)
    return shapes


def random_dspark_params(config, seed: int) -> dict[str, np.ndarray]:
    """Deterministic random fp32 values for every DSpark parameter (markov embedding ~N(0, 1) so its bias is O(1))."""
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for name, shape in sorted(dspark_param_shapes(config).items()):
        if name.endswith("markov_head.embed.weight"):
            params[name] = rng.standard_normal(shape).astype(np.float32)
        else:
            params[name] = _draw(rng, name, shape)
    return params


_DSPARK_NAME_RE = re.compile(r"^stages\.(\d+)\.(.+)$")


def dspark_to_reference(name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Map one DeepseekV41DSpark parameter to reference ``mtp.N.*`` ``(name, array)`` pairs.

    Raises:
        KeyError: For a name the oracle does not know.
    """
    match = _DSPARK_NAME_RE.match(name)
    if match is None:
        raise KeyError(f"unmapped DSpark parameter {name!r}")
    stage, rest = int(match.group(1)), match.group(2)
    prefix = f"mtp.{stage}."
    array = np.asarray(array)
    if rest in ("main_norm.weight", "norm.weight", "markov_head.embed.weight"):
        return [(prefix + rest, array)]
    if rest in ("main_proj.weight", "markov_head.head.weight", "confidence_head.proj.weight"):
        return [(prefix + rest, array.T)]  # [in, out] kernel -> [out, in] weight
    stand_in = f"layers.{stage}."  # block parameters share the backbone block layout
    return [
        (prefix + ref_name[len(stand_in) :], value)
        for ref_name, value in easydel_to_reference(f"model.{stand_in}{rest}", array)
    ]


def load_dspark_params(ref_model, flat_dspark_params: dict[str, np.ndarray]) -> list[str]:
    """Copy every DSpark parameter into its ``mtp.N.*`` reference tensor and prove each one was written exactly once.

    The tied ``mtp.N.embed`` / ``mtp.N.head`` are the backbone's (``named_parameters`` lists them once, under the
    backbone), so they are loaded by ``load_easydel_params``.

    Raises:
        KeyError: If a name maps to a tensor absent from the reference model.
        ValueError: On a shape mismatch.
        AssertionError: If an ``mtp.*`` parameter was written zero or more than one time.
    """
    params = dict(ref_model.named_parameters())
    written: collections.Counter[str] = collections.Counter()
    for name in sorted(flat_dspark_params):
        for ref_name, value in dspark_to_reference(name, flat_dspark_params[name]):
            if ref_name not in params:
                raise KeyError(f"DSpark parameter {name!r} maps to {ref_name!r}, absent from the reference model")
            target = params[ref_name]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if tuple(target.shape) != value.shape:
                raise ValueError(f"{name!r} -> {ref_name!r}: shape {value.shape} != reference {tuple(target.shape)}")
            with torch.no_grad():
                target.copy_(torch.from_numpy(value))
            written[ref_name] += 1
    expected = {n for n in params if n.startswith("mtp.")}
    missing = sorted(expected - set(written))
    repeated = sorted(n for n, count in written.items() if count != 1)
    if missing or repeated or not expected:
        raise AssertionError(
            f"mtp parameters never written: {missing[:8]}; written more than once: {repeated[:8]}; "
            f"expected {len(expected)} (reference built without dspark=True?)"
        )
    return sorted(written)


class DSparkInputs(tp.NamedTuple):
    """Inputs of the DSpark case: prefill main hidden ``[B, S, T*D]``, step tokens ``[steps, B]``, step main hidden."""

    prefill_hidden: np.ndarray
    tokens: np.ndarray
    step_hidden: np.ndarray


class DSparkOutputs(tp.NamedTuple):
    """One ``forward_spec`` decode result: ids ``[B, block + 1]``, logits ``[B, block, V]``, confidence ``[B, block]``."""

    ids: np.ndarray
    logits: np.ndarray
    confidence: np.ndarray


def dspark_inputs(config, prefill_len: int, num_steps: int = DSPARK_NUM_STEPS) -> DSparkInputs:
    """Deterministic DSpark inputs: random main hidden states and accepted tokens (positions ``S .. S + steps - 1``)."""
    rng = np.random.default_rng(3000 + prefill_len)
    width = len(config.dspark_target_layer_ids) * config.hidden_size
    prefill = rng.standard_normal((CASE_BATCH, prefill_len, width)).astype(np.float32)
    tokens = rng.integers(0, int(config.vocab_size), size=(num_steps, CASE_BATCH), dtype=np.int64)
    step_hidden = rng.standard_normal((num_steps, CASE_BATCH, 1, width)).astype(np.float32)
    return DSparkInputs(prefill, tokens, step_hidden)


def reference_dspark(ref_model, inputs: DSparkInputs, *, capture: list[TopkRecord] | None = None) -> list[DSparkOutputs]:
    """Run the reference ``Transformer.forward_spec``: one prefill (``start_pos=0``) then one decode call per step.

    Args:
        ref_model: Model from ``build_reference(dspark=True)`` with backbone and DSpark parameters loaded.
        inputs: Prefill main hidden, accepted tokens and per-step main hidden states.
        capture: If given, receives a ``TopkRecord`` for every router top-k.

    Returns:
        One ``DSparkOutputs`` per decode step (greedy: the reference is built with ``temperature=0``).
    """
    module = load_reference_module()
    _reset_state(ref_model, module)
    batch, prefill_len, _ = inputs.prefill_hidden.shape
    record_ctx = contextlib.nullcontext() if capture is None else _record_topk(capture)
    outputs: list[DSparkOutputs] = []
    with torch.no_grad(), record_ctx:
        prefill = ref_model.forward_spec(
            torch.zeros(batch, dtype=torch.long), torch.from_numpy(np.ascontiguousarray(inputs.prefill_hidden)), 0
        )
        assert prefill is None, "the prefill call only seeds the window rings"
        start_pos = prefill_len
        for tokens, hidden in zip(inputs.tokens, inputs.step_hidden, strict=True):
            ids, logits, confidence = ref_model.forward_spec(
                torch.as_tensor(np.asarray(tokens), dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(hidden)),
                start_pos,
            )
            outputs.append(
                DSparkOutputs(ids.numpy().copy().astype(np.int64), logits.numpy().copy(), confidence.numpy().copy())
            )
            start_pos += 1
    return outputs


__all__ = [
    "CASES",
    "CASE_BATCH",
    "CASE_SEEDS",
    "CASE_SEQ_LEN",
    "DSPARK_NUM_STEPS",
    "DSPARK_PREFILL_LENS",
    "DSPARK_SEED",
    "MAX_SEQ_LEN",
    "TOKEN_IMAGE",
    "TOKEN_IMAGE_END",
    "TOKEN_IMAGE_NEW_LINE",
    "TOKEN_IMAGE_START",
    "TOKEN_TEXT",
    "VL_GRID",
    "VL_IMAGE_START",
    "VL_SEED",
    "VL_SEQ_LEN",
    "VOCAB",
    "DSparkInputs",
    "DSparkOutputs",
    "ImageInput",
    "TopkRecord",
    "VLInputs",
    "build_reference",
    "case_config_kwargs",
    "case_input_ids",
    "check_selection_schedule",
    "dspark_config_kwargs",
    "dspark_inputs",
    "dspark_param_shapes",
    "dspark_to_reference",
    "easydel_param_shapes",
    "easydel_to_reference",
    "hc_split_sinkhorn",
    "load_dspark_params",
    "load_easydel_params",
    "load_reference_module",
    "model_args_from_config",
    "random_dspark_params",
    "random_easydel_params",
    "reference_dspark",
    "reference_logits",
    "reference_loss_and_grads",
    "sparse_attn",
    "topk_margins",
    "vl_config_kwargs",
    "vl_inputs",
    "vl_reference_images",
]
