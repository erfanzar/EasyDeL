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

"""Contract tests for the precomputed "embeds-only" VLM data pack.

QAT-distillation of the Qwen3.5-VL teacher consumes parquet rows that already carry
POST-vision-tower image embeddings (bf16). Training scatters those embeddings into the
``<image>`` placeholder positions and SKIPS the vision tower entirely. These tests lock
that contract:

1. ``test_data_contract``           - per-row byte/shape/count invariants on real parquet.
2. ``test_scatter_places_real_embeds`` - ``merge_multimodal_embeddings`` puts the k-th
   decoded embed at the k-th placeholder and leaves text positions untouched.
3. ``test_collate_embeds_pack``     - a batch collator (in ``easydel.data.transforms.collators``) that pads ids/labels
   and concatenates+zero-pads embeds keeps per-sample placeholder/embed alignment.
4. ``test_embeds_pack_e2e_no_vision_tower`` (slow) - merged embeddings flow through the
   real Qwen3.5 causal-LM stack via ``inputs_embeds`` (no vision tower) to finite logits.
5. ``test_compute_embedding_scatters_precomputed_embeds`` (slow) - the model's embedding
   entrypoint scatters precomputed ``image_embeds`` identically to the standalone helper.
6. ``test_forward_inject_equivalence`` (slow) - the standard call ``model(input_ids=,
   image_embeds=, image_grid_thw=)`` equals the explicit two-step (pre-merged
   ``inputs_embeds`` + grid-derived mRoPE ``position_ids``), so training can inject
   precomputed embeds through the ordinary forward signature.
7. ``test_forward_mrope_uses_image_grid`` (slow) - the image grid drives genuine 3-D mRoPE
   positions (distinct from the 1-D text fallback), and the forward routes the grid for
   precomputed-embed batches (the Phase-1 gating fix).
8. ``test_for_conditional_generation_standard_call`` (slow) - the LM-head wrapper consumes
   the standard precomputed-embed call and returns finite vocab logits.

Tests 6-8 exercise the real decoder, so they remap the pack's special/text token ids into a
tiny vocab (see ``_remap_ids``): the real ids (~248k) are out of range for a small embedding
table and would embed to NaN at text positions. The remap is structure-preserving — it keeps
the vision_start -> placeholder-run -> grid alignment that ``get_rope_index`` relies on — and
the config's special-token ids are pointed at the remapped values.

Data lives under ``local_testdata/`` (override with ``EMBEDS_PACK_GLOB``).
"""

from __future__ import annotations

import glob
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from easydel.data.transforms.collators import (  # noqa: E402
    HIDDEN,
    IMAGE_PLACEHOLDER_ID,
    _decode_row_embeds,
    collate_bucket_static,
    collate_embeds_pack,
    collate_packed_embeds,
)

# Vision span special-token ids; IMAGE_PLACEHOLDER_ID + HIDDEN are imported from the collators module above.
VISION_START_ID = 248053
VISION_END_ID = 248054

_DATA_GLOB = os.environ.get(
    "EMBEDS_PACK_GLOB",
    os.path.join(os.path.dirname(__file__), "..", "..", "local_testdata", "**", "*.parquet"),
)


def _parquet_paths() -> list[str]:
    return sorted(glob.glob(_DATA_GLOB, recursive=True))


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    import pyarrow.parquet as pq

    paths = _parquet_paths()
    if not paths:
        pytest.skip(f"no staged embed-pack parquet found at {_DATA_GLOB}")
    out: list[dict] = []
    for p in paths:
        out.extend(pq.read_table(p).to_pylist())
    return out


def test_data_contract(rows):
    """Every row: bf16 blob bytes == n_tok*dim*2, decoded shape & finiteness, placeholder/embed
    count parity, and seq_len == len(input_ids) == len(labels) == len(attention_mask)."""
    import ml_dtypes

    assert rows, "no rows loaded"
    for ri, r in enumerate(rows):
        ed = int(r["embed_dim"])
        assert ed == HIDDEN, f"row {ri}: embed_dim {ed} != {HIDDEN}"
        ent = list(r["embed_n_tok"])
        blobs = r["image_embeds"]
        assert len(blobs) == len(ent) == int(r["n_images"]), f"row {ri}: n_images/blobs/embed_n_tok length mismatch"
        for i, b in enumerate(blobs):
            assert len(b) == ent[i] * ed * 2, f"row {ri} img {i}: blob bytes {len(b)} != {ent[i]}*{ed}*2"
            dec = np.frombuffer(b, dtype=ml_dtypes.bfloat16).reshape(ent[i], ed)
            assert dec.shape == (ent[i], ed)
            assert np.isfinite(dec.astype(np.float32)).all(), f"row {ri} img {i}: non-finite embed"
        ids = list(r["input_ids"])
        n_place = sum(1 for x in ids if x == IMAGE_PLACEHOLDER_ID)
        assert n_place == sum(ent), f"row {ri}: placeholders {n_place} != sum(embed_n_tok) {sum(ent)}"
        sl = int(r["seq_len"])
        assert sl == len(ids) == len(r["labels"]) == len(r["attention_mask"]), f"row {ri}: seq_len/length mismatch"


def test_scatter_places_real_embeds(rows):
    """``merge_multimodal_embeddings`` replaces the k-th placeholder with the k-th decoded embed
    (left-to-right) and leaves every non-placeholder position equal to the original text embed."""
    import jax.numpy as jnp

    from easydel.modules.qwen3_vl.modeling_qwen3_vl import merge_multimodal_embeddings

    row = next(r for r in rows if int(r["n_images"]) > 0)
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    seq = ids.shape[0]
    real = _decode_row_embeds(row)
    place_pos = np.where(ids == IMAGE_PLACEHOLDER_ID)[0]
    assert place_pos.shape[0] == real.shape[0], "contract precondition: #placeholders == #embed rows"

    # Position p -> a constant row of value p, distinguishable from arbitrary real embeds.
    synth = np.broadcast_to(np.arange(seq, dtype=np.float32)[:, None], (seq, HIDDEN)).copy()
    merged = merge_multimodal_embeddings(
        jnp.asarray(ids)[None], jnp.asarray(synth)[None], jnp.asarray(real), IMAGE_PLACEHOLDER_ID
    )
    merged = np.asarray(merged[0])

    for k, p in enumerate(place_pos):
        assert np.array_equal(merged[p], real[k]), f"placeholder #{k} at pos {p} != real_embeds[{k}]"
    for p in np.setdiff1d(np.arange(seq), place_pos):
        assert np.array_equal(merged[p], synth[p]), f"non-placeholder pos {p} was mutated"


def test_collate_embeds_pack(rows):
    """The batch collator preserves global placeholder<->embed parity and per-sample alignment,
    zero-pads embeds past the real count, and masks the padded tail (labels=-100, attn=0)."""
    batch = rows[:4]
    pad_id = 0  # any non-placeholder id
    total_real = sum(_decode_row_embeds(r).shape[0] for r in batch)
    max_total = total_real + 7  # exercise zero-padding past the real rows
    out = collate_embeds_pack(batch, pad_id=pad_id, max_total=max_total)

    ids = np.asarray(out["input_ids"])
    emb = np.asarray(out["image_embeds"])
    labels = np.asarray(out["labels"])
    attn = np.asarray(out["attention_mask"])

    assert int((ids == IMAGE_PLACEHOLDER_ID).sum()) == total_real == out["n_real_embeds"]
    assert emb.shape == (max_total, HIDDEN)
    assert np.array_equal(emb[total_real:], np.zeros((max_total - total_real, HIDDEN), np.float32))

    cursor = 0
    for bi, r in enumerate(batch):
        dec = _decode_row_embeds(r)
        length = len(r["input_ids"])
        place_pos = np.where(ids[bi, :length] == IMAGE_PLACEHOLDER_ID)[0]
        assert place_pos.shape[0] == dec.shape[0], f"sample {bi}: placeholder count != decoded embeds"
        assert np.array_equal(emb[cursor : cursor + dec.shape[0]], dec), f"sample {bi}: embed slice misaligned"
        assert not (ids[bi, length:] == IMAGE_PLACEHOLDER_ID).any(), f"sample {bi}: placeholder in padded tail"
        assert np.all(labels[bi, length:] == -100), f"sample {bi}: padded labels not -100"
        assert np.all(attn[bi, length:] == 0), f"sample {bi}: padded attention not 0"
        cursor += dec.shape[0]

    total_imgs = sum(int(r["n_images"]) for r in batch)
    assert np.asarray(out["image_grid_thw"]).shape == (total_imgs, 3)


def test_collate_bucket_static_fixed_shape_and_overflow_guard(rows):
    """The bucketed static collator emits a caller-fixed ``(B, seq_len)`` / ``(max_embed_rows, H)``
    shape regardless of the batch's actual row lengths (so a bucket compiles once), preserves
    placeholder<->embed parity, masks the padded tail, and HARD-RAISES when any row would overflow
    the static window (the seq_len overflow guard the producer does not enforce)."""
    batch = rows[:4]
    longest = max(len(r["input_ids"]) for r in batch)
    S = longest + 16  # fixed bucket window strictly above the longest row
    total_real = sum(_decode_row_embeds(r).shape[0] for r in batch)
    cap = total_real + 9  # exercise zero-padded embed tail past the real rows

    out = collate_bucket_static(batch, pad_id=0, seq_len=S, max_embed_rows=cap)
    ids = np.asarray(out["input_ids"])
    emb = np.asarray(out["image_embeds"])
    labels = np.asarray(out["labels"])
    attn = np.asarray(out["attention_mask"])

    assert ids.shape == (len(batch), S), "input_ids not padded to the fixed bucket window"
    assert emb.shape == (cap, HIDDEN), "image_embeds not padded to the fixed embed-row cap"
    assert int((ids == IMAGE_PLACEHOLDER_ID).sum()) == total_real == out["n_real_embeds"]
    assert np.array_equal(emb[total_real:], np.zeros((cap - total_real, HIDDEN), np.float32))

    for bi, r in enumerate(batch):
        length = len(r["input_ids"])
        assert not (ids[bi, length:] == IMAGE_PLACEHOLDER_ID).any(), f"sample {bi}: placeholder in padded tail"
        assert np.all(labels[bi, length:] == -100), f"sample {bi}: padded labels not -100"
        assert np.all(attn[bi, length:] == 0), f"sample {bi}: padded attention not 0"

    # A DIFFERENT batch padded to the SAME window keeps the identical static shape (the point of bucketing).
    other = rows[4:7] if len(rows) >= 7 else batch[:3]
    out2 = collate_bucket_static(other, pad_id=0, seq_len=S, max_embed_rows=cap)
    assert np.asarray(out2["input_ids"]).shape == (len(other), S)
    assert np.asarray(out2["image_embeds"]).shape == (cap, HIDDEN)

    # Overflow guard: a window shorter than the longest row must raise (never silently truncate).
    with pytest.raises(AssertionError, match="overflow guard"):
        collate_bucket_static(batch, pad_id=0, seq_len=longest - 1, max_embed_rows=cap)


def test_collate_bucket_static_partition_homogeneity_guard(rows):
    """Defense-in-depth for the 3-level (source x area_bucket x slen_band) static-S design: a batch
    that mixes ANY hive-partition key HARD-RAISES and names the offending key + values (so a
    spanning-glob / cross-partition-shuffle misconfig fails legibly instead of forcing a wrong static
    window), the static window must equal the partition's ``slen_band`` upper edge, and rows lacking a
    key skip that level (no KeyError for the wrong reason). Copies fixture rows before mutating."""
    batch = [dict(r) for r in rows[:2]]
    longest = max(len(r["input_ids"]) for r in batch)
    S = longest + 16
    cap = sum(_decode_row_embeds(r).shape[0] for r in batch) + 4

    # Homogeneous on every partition key -> passes (real rows already share source + area_bucket).
    collate_bucket_static([dict(r) for r in batch], pad_id=0, seq_len=S, max_embed_rows=cap)

    # Each partition key, mixed in isolation -> raises naming that key + BOTH offending values.
    for key, a, b, pat in [
        ("source", "src_a", "src_b", r"heterogeneous batch.*source.*src_a.*src_b"),
        ("area_bucket", 1024, 1536, r"heterogeneous batch.*area_bucket.*1024.*1536"),
        ("slen_band", S, S + 64, r"heterogeneous batch.*slen_band"),
    ]:
        mixed = [dict(r) for r in batch]
        mixed[0][key], mixed[1][key] = a, b
        with pytest.raises(AssertionError, match=pat):
            collate_bucket_static(mixed, pad_id=0, seq_len=S, max_embed_rows=cap)

    # Homogeneous slen_band but static-S != band edge -> raises (window must equal the band upper edge);
    # the matched window passes.
    banded = [dict(r) for r in batch]
    for r in banded:
        r["slen_band"] = S
    with pytest.raises(AssertionError, match="static-S mismatch"):
        collate_bucket_static([dict(r) for r in banded], pad_id=0, seq_len=S + 64, max_embed_rows=cap)
    collate_bucket_static([dict(r) for r in banded], pad_id=0, seq_len=S, max_embed_rows=cap)

    # Tolerant: rows lacking the partition keys skip the checks (no KeyError, no raise).
    no_label = [dict(r) for r in rows[:2]]
    for r in no_label:
        for k in ("source", "area_bucket", "slen_band"):
            r.pop(k, None)
    out = collate_bucket_static(no_label, pad_id=0, seq_len=S, max_embed_rows=cap)
    assert np.asarray(out["input_ids"]).shape == (len(no_label), S)


# PROVISIONAL DP-optimal k=8 slen_band upper edges (fit on the smoke pack; the build re-fits on the
# FULL pack and is the canonical source). Shared representation with sink.py: slen_band partition value
# == the band's upper edge (int, 64-aligned) == the collator's static-S; band(n) = smallest edge >= n.
_PROVISIONAL_SLEN_EDGES = [192, 448, 576, 768, 1088, 1408, 2368, 4096]


def test_collate_bucket_static_three_level_real_cell(rows):
    """The 3-level (source x area_bucket x slen_band) layout on REAL rows: assign each row its band via
    the provisional ladder, take one homogeneous (source, area_bucket, slen_band) cell, and collate
    with static-S = the band's upper edge. Proves a sub-banded partition collates to a fixed
    (B, band_edge) window, every row fits its band, and the band==seq_len net accepts the matched
    window. Edge list is PROVISIONAL (re-fit on the full pack at build)."""

    def band_of(n: int) -> int:
        return next(e for e in _PROVISIONAL_SLEN_EDGES if e >= n)

    cells: dict = {}
    for r in rows:
        n = len(r["input_ids"])
        if n > _PROVISIONAL_SLEN_EDGES[-1]:
            continue  # > cap rows are dropped at build, never collated
        cells.setdefault((r.get("source"), r.get("area_bucket"), band_of(n)), []).append(r)
    (_src, _ab, band), cell_rows = max(cells.items(), key=lambda kv: len(kv[1]))
    assert len(cell_rows) >= 2, "need a 3-level cell with >=2 rows to build a static batch"

    batch = [dict(r) for r in cell_rows[:4]]
    for r in batch:
        r["slen_band"] = band
    cap = sum(_decode_row_embeds(r).shape[0] for r in batch) + 4

    out = collate_bucket_static(batch, pad_id=0, seq_len=band, max_embed_rows=cap)
    ids = np.asarray(out["input_ids"])
    assert ids.shape == (len(batch), band), "sub-banded batch not padded to the band upper edge"
    assert all(len(r["input_ids"]) <= band for r in batch), "row exceeds its assigned band (mis-banded)"
    assert int((ids == IMAGE_PLACEHOLDER_ID).sum()) == out["n_real_embeds"], "placeholder<->embed parity broke"


def _build_tiny_text_model(seq_cap: int):
    """A minimal CPU Qwen3.5 causal-LM whose hidden size matches the pack's embed_dim (5120).

    Dense (Qwen3_5TextConfig forces dense MLPs / no MoE), 1 full-attention layer, tiny vocab and
    intermediate size — enough to exercise the real forward through ``inputs_embeds`` without the
    vision tower, cheaply on CPU.
    """
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=640,
        max_position_embeddings=max(2048, seq_cap + 8),
        layer_types=["full_attention"],
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        scan_layers=False,
    )
    return Qwen3_5ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


@pytest.mark.slow
def test_embeds_pack_e2e_no_vision_tower(rows):
    """End-to-end: scatter real embeds into synthetic text embeds, push the merged tensor through
    the real Qwen3.5 stack via ``inputs_embeds`` (vision tower skipped) and get finite logits."""
    import jax.numpy as jnp

    from easydel.modules.qwen3_vl.modeling_qwen3_vl import merge_multimodal_embeddings

    row = next(r for r in rows if int(r["n_images"]) > 0)
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    seq = ids.shape[0]
    real = _decode_row_embeds(row)
    assert real.shape[0] == int((ids == IMAGE_PLACEHOLDER_ID).sum())

    model = _build_tiny_text_model(seq)
    synth_text = jnp.asarray((np.random.default_rng(0).standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32))
    merged = merge_multimodal_embeddings(jnp.asarray(ids)[None], synth_text, jnp.asarray(real), IMAGE_PLACEHOLDER_ID)

    out = model(inputs_embeds=merged, attention_mask=jnp.ones((1, seq), dtype=jnp.int32), apply_lm_head=True)
    logits = np.asarray(out.logits.astype(jnp.float32))
    assert logits.shape == (1, seq, 256)
    assert np.isfinite(logits).all()


def _build_tiny_vl_model(seq_cap: int):
    """A minimal CPU Qwen3.5 VL model: tiny 5120-hidden text backbone + a deliberately tiny vision
    tower (never executed here — we pass precomputed ``image_embeds``, so the tower is bypassed).
    """
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    text = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=640,
        max_position_embeddings=max(2048, seq_cap + 8),
        layer_types=["full_attention"],
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        scan_layers=False,
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN)
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=IMAGE_PLACEHOLDER_ID)
    model = Qwen3_5Model(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    return model, config


@pytest.mark.slow
def test_compute_embedding_scatters_precomputed_embeds(rows):
    """The model's own embedding entrypoint, given precomputed ``image_embeds``, splices them at the
    ``image_token_id`` placeholders and leaves text positions untouched — no vision tower run, and
    identical to the standalone ``merge_multimodal_embeddings`` helper.

    Note: the tiny test vocab (256) cannot embed the real token ids (~248k), so text positions come
    out non-finite; that is irrelevant to this contract (training feeds precomputed ``inputs_embeds``)
    and we compare those positions bit-for-bit (``equal_nan``) against the no-image baseline.
    """
    import jax.numpy as jnp

    from easydel.modules.qwen3_vl.modeling_qwen3_vl import merge_multimodal_embeddings

    row = next(r for r in rows if int(r["n_images"]) > 0)
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    real = _decode_row_embeds(row)
    place_pos = np.where(ids == IMAGE_PLACEHOLDER_ID)[0]

    model, config = _build_tiny_vl_model(ids.shape[0])
    assert config.image_token_id == IMAGE_PLACEHOLDER_ID

    ids_j = jnp.asarray(ids)[None]
    real_j = jnp.asarray(real)
    merged = np.asarray(model.compute_embedding(ids_j, image_embeds=real_j))
    base = np.asarray(model.compute_embedding(ids_j))  # text-only baseline (no image_embeds)
    reference = np.asarray(merge_multimodal_embeddings(ids_j, jnp.asarray(base), real_j, IMAGE_PLACEHOLDER_ID))

    assert np.array_equal(merged[0, place_pos], real), "placeholders not replaced by exact precomputed embeds"
    nonp = np.setdiff1d(np.arange(ids.shape[0]), place_pos)
    assert np.array_equal(merged[0, nonp], base[0, nonp], equal_nan=True), "text positions were modified by the scatter"
    assert np.array_equal(merged, reference, equal_nan=True), "model embedding entrypoint != standalone merge helper"


# --- Remapped tiny-vocab scheme for the decoder-running tests (6-8) -------------------------
# Real ids (~248k) are out of range for a tiny embedding table -> NaN at text positions. We map
# the structure-bearing tokens into a small vocab and leave everything else as one text filler id.
_RM_VSTART, _RM_PLACE, _RM_VEND, _RM_VIDEO, _RM_TEXT = 1, 2, 3, 4, 5
_RM_VOCAB = 16


def _remap_ids(ids: np.ndarray) -> np.ndarray:
    """Map real token ids to a tiny vocab, preserving vision_start / image-placeholder / vision_end
    positions so ``get_rope_index`` still detects the image span and the scatter still hits it."""
    out = np.full(ids.shape[0], _RM_TEXT, dtype=np.int32)
    out[ids == VISION_START_ID] = _RM_VSTART
    out[ids == IMAGE_PLACEHOLDER_ID] = _RM_PLACE
    out[ids == VISION_END_ID] = _RM_VEND
    return out


def _remapped_vl_config(seq_cap: int):
    """Qwen3.5 VL config for the decoder tests: tiny 5120-hidden text backbone + tiny (unused)
    vision tower, with special-token ids pointed at the remapped scheme. ``spatial_merge_size``
    stays at its default (2): with the pack's raw ``image_grid_thw`` that yields exactly the
    placeholder count, which ``get_rope_index`` needs for a correctly-aligned 3-D layout."""
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    text = Qwen3_5TextConfig(
        vocab_size=_RM_VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=640,
        max_position_embeddings=max(2048, seq_cap + 8),
        layer_types=["full_attention"],
        mtp_num_hidden_layers=0,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
        scan_layers=False,
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=64, intermediate_size=64, num_heads=2, out_hidden_size=HIDDEN)
    return Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=_RM_PLACE,
        video_token_id=_RM_VIDEO,
        vision_start_token_id=_RM_VSTART,
        vision_end_token_id=_RM_VEND,
    )


@pytest.fixture(scope="module")
def vl_remapped(rows):
    """Tiny remapped-vocab VL model + one real single-image row's tensors (ids remapped, embeds
    decoded, raw grid, attention mask). Module-scoped so the ~20s model build happens once and is
    shared by the forward-equivalence and mRoPE tests."""
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    row = next(r for r in rows if int(r["n_images"]) > 0)
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    ids_rm = _remap_ids(ids)
    real = _decode_row_embeds(row)
    grid = np.asarray(row["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
    am = np.asarray(row["attention_mask"], dtype=np.int32)
    config = _remapped_vl_config(ids.shape[0])
    model = Qwen3_5Model(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    # precondition: remapped placeholder run still matches the decoded embed count
    assert int((ids_rm == _RM_PLACE).sum()) == real.shape[0]
    return {
        "model": model,
        "config": config,
        "ids_rm": ids_rm,
        "ids_j": jnp.asarray(ids_rm)[None],
        "real_j": jnp.asarray(real),
        "grid_j": jnp.asarray(grid),
        "am_j": jnp.asarray(am)[None],
        "place_pos": np.where(ids_rm == _RM_PLACE)[0],
        "seq": ids.shape[0],
    }


@pytest.mark.slow
def test_forward_inject_equivalence(vl_remapped):
    """The standard inject call ``model(input_ids=, image_embeds=, image_grid_thw=)`` is equivalent
    to the explicit two-step: pre-merge the embeds via ``compute_embedding`` and feed the resulting
    ``inputs_embeds`` together with the grid-derived mRoPE ``position_ids``. This is the core
    Phase-1 contract: precomputed embeds can ride the ordinary forward signature, vision tower
    skipped, without changing the decoder's output."""
    import jax.numpy as jnp

    m = vl_remapped["model"]
    ids_j, real_j, grid_j, am_j = (vl_remapped["ids_j"], vl_remapped["real_j"], vl_remapped["grid_j"], vl_remapped["am_j"])

    out_std = m(input_ids=ids_j, image_embeds=real_j, image_grid_thw=grid_j, attention_mask=am_j)
    std = np.asarray(out_std.last_hidden_state.astype(jnp.float32))

    merged = m.compute_embedding(ids_j, image_embeds=real_j)
    pos, _ = m.get_rope_index(ids_j, image_grid_thw=grid_j, attention_mask=am_j)
    out_manual = m(inputs_embeds=merged, attention_mask=am_j, position_ids=pos)
    manual = np.asarray(out_manual.last_hidden_state.astype(jnp.float32))

    assert std.shape == (1, vl_remapped["seq"], HIDDEN)
    assert np.isfinite(std).all(), "standard inject forward produced non-finite hidden states"
    assert np.allclose(std, manual, rtol=1e-5, atol=1e-5), "inject path != pre-merged inputs_embeds + grid mRoPE"


@pytest.mark.slow
def test_forward_mrope_uses_image_grid(vl_remapped):
    """mRoPE correctness: ``image_grid_thw`` must drive a genuine 3-D position layout (distinct t/h/w
    axes over the image span), unlike the 1-D text fallback where all three axes coincide. And the
    forward must actually route the grid for a precomputed-embed batch (no ``pixel_values``) — the
    Phase-1 gating fix — verified by ``rope_deltas`` matching the grid path and differing from the
    grid-less path."""
    m = vl_remapped["model"]
    ids_j, real_j, grid_j, am_j = (vl_remapped["ids_j"], vl_remapped["real_j"], vl_remapped["grid_j"], vl_remapped["am_j"])
    place_pos = vl_remapped["place_pos"]

    pos_g, dl_g = m.get_rope_index(ids_j, image_grid_thw=grid_j, attention_mask=am_j)
    pos_n, dl_n = m.get_rope_index(ids_j, image_grid_thw=None, attention_mask=am_j)
    pos_g, pos_n = np.asarray(pos_g), np.asarray(pos_n)
    dl_g_v, dl_n_v = int(np.asarray(dl_g).ravel()[0]), int(np.asarray(dl_n).ravel()[0])

    # grid path: the 3 mRoPE axes are NOT all identical over the image span (true 3-D layout)
    g_t, g_h, g_w = pos_g[0, 0, place_pos], pos_g[1, 0, place_pos], pos_g[2, 0, place_pos]
    assert not (np.array_equal(g_t, g_h) and np.array_equal(g_h, g_w)), "grid path did not produce a 3-D layout"
    # grid-less fallback: all 3 axes identical (plain 1-D positions broadcast to 3)
    assert np.array_equal(pos_n[0], pos_n[1]) and np.array_equal(pos_n[1], pos_n[2]), "fallback was not 1-D"
    assert dl_g_v != dl_n_v, "grid vs grid-less rope_deltas coincide; grid had no effect"

    out = m(input_ids=ids_j, image_embeds=real_j, image_grid_thw=grid_j, attention_mask=am_j)
    assert int(np.asarray(out.rope_deltas).ravel()[0]) == dl_g_v, "forward did not route image_grid_thw into mRoPE"


@pytest.mark.slow
def test_for_conditional_generation_standard_call(rows):
    """The LM-head wrapper (``Qwen3_5ForConditionalGeneration``) accepts the same precomputed-embed
    standard call and returns finite vocab logits — confirming ``image_embeds`` propagates through
    the wrapper's ``**kwargs`` into the base model and out through the head."""
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    row = next(r for r in rows if int(r["n_images"]) > 0)
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    ids_rm = _remap_ids(ids)
    real = _decode_row_embeds(row)
    grid = np.asarray(row["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
    am = np.asarray(row["attention_mask"], dtype=np.int32)

    config = _remapped_vl_config(ids.shape[0])
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)

    out = model(
        input_ids=jnp.asarray(ids_rm)[None],
        image_embeds=jnp.asarray(real),
        image_grid_thw=jnp.asarray(grid),
        attention_mask=jnp.asarray(am)[None],
        apply_lm_head=True,
    )
    logits = np.asarray(out.logits.astype(jnp.float32))
    assert logits.shape == (1, ids.shape[0], _RM_VOCAB)
    assert np.isfinite(logits).all(), "wrapper produced non-finite logits for the precomputed-embed call"


def _path_keys(path) -> list[str]:
    """Flatten a jax tree path into its string key/index parts (for partitioning grads by submodule)."""
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return parts


def _remapped_pack_batch(row: dict, pad_tail: int = 0) -> tuple[dict, int]:
    """Single-row ``collate_embeds_pack`` batch with input_ids/labels remapped into the tiny vocab.

    The decoder runs over token ids, so the 248k real ids are remapped (structure-preserving so the
    image span is still detected and the scatter still hits the placeholders). Labels keep their -100
    mask and map every supervised position to one in-vocab filler — target identity is irrelevant to a
    finiteness/grad proof, only that the loss is well-defined. ``pad_tail`` appends that many zero embed
    rows past the real count so the scatter/padding inertness can be exercised. Returns ``(batch, n_real)``.
    """
    import jax.numpy as jnp

    n_real = _decode_row_embeds(row).shape[0]
    batch = collate_embeds_pack([row], pad_id=0, max_total=n_real + pad_tail)
    assert batch["n_real_embeds"] == n_real
    ids_real = np.asarray(batch["input_ids"][0])
    labels_real = np.asarray(batch["labels"][0])
    assert int((labels_real != -100).sum()) > 0, "row has no supervised label positions"
    batch["input_ids"] = jnp.asarray(_remap_ids(ids_real))[None]
    batch["labels"] = jnp.asarray(np.where(labels_real == -100, -100, _RM_TEXT).astype(np.int32))[None]
    return batch, n_real


@pytest.mark.slow
def test_compute_loss_and_grads_train_the_pack(rows):
    """Trainability proof on a real pack row, through the model's canonical training entry.

    Builds a ``collate_embeds_pack`` batch (ids/labels remapped into the tiny vocab so the
    248k-id text positions don't NaN a small embedding table), then calls the SAME
    ``compute_loss`` the trainers call with ``image_embeds`` + ``image_grid_thw`` + ``labels``.
    Replicates the trainer's exact gradient computation (``split_module`` -> ``merge_module`` ->
    ``compute_loss`` -> ``outputs.loss`` -> ``jax.value_and_grad`` w.r.t. params) and asserts:

      1. a finite scalar loss;
      2. finite, non-zero param gradients overall AND specifically on the LM head + decoder
         layers (the path the scattered image embeds flow through to the loss);
      3. d(loss)/d(image_embeds) is finite and non-zero on the REAL (scattered) embed rows and
         EXACTLY zero on the zero-padded tail -- proving the precomputed embeds are live in the
         differentiable loss graph (not dropped/detached) and the padded rows are inert.
    """
    import jax
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    row = next(r for r in rows if int(r["n_images"]) > 0)
    batch, n_real = _remapped_pack_batch(row, pad_tail=5)  # pad_tail exercises inert zero-padded rows
    ids_j = batch["input_ids"]
    labels_j = batch["labels"]
    am_j = batch["attention_mask"]
    grid_j = batch["image_grid_thw"]
    emb_j = batch["image_embeds"]

    config = _remapped_vl_config(int(ids_j.shape[1]))
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    gdef, gstate, gother = model.split_module()

    def loss_fn(gs):
        m = model.merge_module(gdef, gs, gother)
        outputs, _metrics = m.compute_loss(
            input_ids=ids_j, attention_mask=am_j, image_embeds=emb_j, image_grid_thw=grid_j, labels=labels_j
        )
        return outputs.loss

    loss, grads = jax.value_and_grad(loss_fn)(gstate)
    loss_v = float(loss)
    assert np.isfinite(loss_v), f"compute_loss returned non-finite loss {loss_v}"

    total_sq, head_sq, layer_sq = 0.0, 0.0, 0.0
    n_nonfinite = 0
    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        ga = np.asarray(jnp.asarray(g, jnp.float32))
        if not np.isfinite(ga).all():
            n_nonfinite += 1
        gsq = float(np.sum(ga**2))
        total_sq += gsq
        parts = _path_keys(path)
        if any("lm_head" in p or p == "lm_head" for p in parts):
            head_sq += gsq
        if any(p == "layers" for p in parts):
            layer_sq += gsq
    assert n_nonfinite == 0, f"{n_nonfinite} param-grad leaves were non-finite"
    assert total_sq**0.5 > 1e-6, "model received zero gradient on the pack batch — it would not train"
    assert head_sq**0.5 > 1e-6, "LM head received zero gradient — the embed-derived logits are not in the loss graph"
    assert layer_sq**0.5 > 1e-6, "decoder layers received zero gradient — the embed-derived hidden states don't train"

    # Decisive embed-injection proof: differentiate the loss w.r.t. the precomputed embeds INPUT.
    def loss_wrt_embeds(emb):
        m = model.merge_module(gdef, gstate, gother)
        outputs, _metrics = m.compute_loss(
            input_ids=ids_j, attention_mask=am_j, image_embeds=emb, image_grid_thw=grid_j, labels=labels_j
        )
        return outputs.loss

    g_emb = np.asarray(jax.grad(loss_wrt_embeds)(emb_j).astype(jnp.float32))
    assert np.isfinite(g_emb).all(), "d(loss)/d(image_embeds) was non-finite"
    real_norm = float(np.linalg.norm(g_emb[:n_real]))
    tail_norm = float(np.linalg.norm(g_emb[n_real:]))
    assert real_norm > 1e-6, "real image-embed rows got zero gradient — embeds are not in the differentiable loss graph"
    assert tail_norm == 0.0, f"zero-padded embed rows got non-zero gradient {tail_norm} — padding leaked into the scatter"


@pytest.mark.slow
def test_zero_image_batch_trains_with_none_and_empty_embeds():
    """Regression for the zero-image (coords-only / text-only, n_images=0) lane.

    The bucket collator emits a LITERAL empty ``(0, HIDDEN)`` ``image_embeds`` for an all-text batch.
    The forward gate is ``if image_embeds is not None`` -- so that empty array (not None) ENTERS the
    cumsum merge, which must be a true no-op. The merge pads a dummy row; built as
    ``zeros_like(mm[0:1])`` it was itself ``(0, HIDDEN)`` on 0 rows, leaving no index-0 slot for the
    cumsum gather (gather OOB / TypeError at trace). The fix pads an explicit ``(1, HIDDEN)`` row.
    This locks the contract end-to-end:

      * collate builds the all-text batch -- ``image_embeds=(0, HIDDEN)``, ``n_real_embeds=0``, zero placeholders;
      * training ``compute_loss`` is finite for BOTH ``image_embeds=None`` AND the empty ``(0, HIDDEN)`` array,
        and the two losses are identical (the empty-embed merge changed nothing);
      * grads are finite and non-zero (the all-text batch trains; nothing depends on an absent embed).

    ``position_ids`` are host-precomputed (text mRoPE, grid=None), isolating the merge no-op from
    ``get_rope_index`` -- the same way the real training step precomputes (3, B, S) positions.
    """
    import jax
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    S = 64

    def _zero_image_row(n_sup: int) -> dict:
        labels = [-100] * S
        labels[1 : 1 + n_sup] = [_RM_TEXT] * n_sup  # >=1 supervised label -> meaningful loss/grad
        return {
            "input_ids": [_RM_TEXT] * S,  # all text, NO placeholder -> is_multimodal all-False
            "attention_mask": [1] * S,
            "labels": labels,
            "embed_dim": HIDDEN,
            "embed_n_tok": [],
            "image_embeds": [],
            "image_grid_thw": [],
            "source": "GUI__os-atlas",
            "area_bucket": 1024,
            "slen_band": S,
        }

    batch = collate_bucket_static([_zero_image_row(8), _zero_image_row(5)], pad_id=0, seq_len=S, max_embed_rows=0)
    emb = np.asarray(batch["image_embeds"])
    ids_j, am_j, labels_j = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert emb.shape == (0, HIDDEN) and int(batch["n_real_embeds"]) == 0, "collate did not emit an empty all-text batch"
    assert int((np.asarray(ids_j) == _RM_PLACE).sum()) == 0, "an all-text batch must carry no visual placeholders"

    config = _remapped_vl_config(S)
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    pos, _ = model.base_model.get_rope_index(ids_j, image_grid_thw=None, attention_mask=am_j)
    empty_j = jnp.asarray(emb, dtype=jnp.float32)  # the collator's literal (0, HIDDEN) output

    def _loss(image_embeds):
        outputs, _metrics = model.compute_loss(
            input_ids=ids_j, attention_mask=am_j, position_ids=pos, image_embeds=image_embeds, labels=labels_j
        )
        return outputs.loss

    loss_none = float(_loss(None))
    loss_empty = float(_loss(empty_j))  # exercises the fixed 0-row merge (pre-fix: gather OOB)
    assert np.isfinite(loss_none) and np.isfinite(loss_empty), f"non-finite loss: None={loss_none} empty={loss_empty}"
    assert abs(loss_none - loss_empty) < 1e-5, (
        f"empty (0,H) image_embeds is not a no-op: None={loss_none} empty={loss_empty} "
        "-- the zero-image merge altered the text embeddings"
    )

    # the all-text batch trains: finite, non-zero param grads through the fixed merge
    gdef, gstate, gother = model.split_module()

    def loss_fn(gs):
        m = model.merge_module(gdef, gs, gother)
        outputs, _metrics = m.compute_loss(
            input_ids=ids_j, attention_mask=am_j, position_ids=pos, image_embeds=empty_j, labels=labels_j
        )
        return outputs.loss

    loss, grads = jax.value_and_grad(loss_fn)(gstate)
    assert np.isfinite(float(loss)), "non-finite loss under value_and_grad on the zero-image batch"
    total_sq, n_nonfinite = 0.0, 0
    for _path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        ga = np.asarray(jnp.asarray(g, jnp.float32))
        n_nonfinite += int(not np.isfinite(ga).all())
        total_sq += float(np.sum(ga**2))
    assert n_nonfinite == 0, f"{n_nonfinite} param-grad leaves non-finite on the zero-image batch"
    assert total_sq**0.5 > 1e-6, "zero gradient on the all-text batch -- it would not train"


@pytest.mark.slow
def test_get_rope_index_is_eager_only_under_jit():
    """Guards the Phase-2 scaling finding: ``get_rope_index`` is host/eager-only and CANNOT be traced.

    The Qwen3-VL ``get_rope_index`` is pure Python/numpy (``.tolist()``, ``input_tokens.index(...)``,
    per-image Python loops, ``np.array(image_grid_thw)``). It works in the eager standard call
    ``model(input_ids, image_embeds, image_grid_thw)`` because the grid is concrete, but the moment it
    runs under a JAX trace (``jax.jit``/``jax.checkpoint`` — e.g. the distillation step's remat'd
    teacher forward) ``image_grid_thw`` is a tracer and the ``np.array(tracer)`` conversion raises.
    This test pins that contract so a future "make it jittable" change is a conscious decision: a
    jitted forward that must compute mRoPE from a traced ``image_grid_thw`` raises
    ``TracerArrayConversionError``. The supported training path is to precompute ``position_ids`` on
    host (next test) so the jitted forward skips ``get_rope_index`` entirely.
    """
    import jax
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    # Minimal grid-detected input: vision_start, then 4 placeholders (1x4x2 grid, merge 2 -> 1*2*1=2)...
    # use a 2x2 grid so merge=2 -> 1*1*1=1 placeholder for a tiny, fast trace.
    ids = np.array([_RM_TEXT, _RM_VSTART, _RM_PLACE, _RM_VEND, _RM_TEXT], dtype=np.int32)
    grid = np.array([[1, 2, 2]], dtype=np.int32)  # t*(h//2)*(w//2) = 1 == #placeholders
    model = Qwen3_5Model(config=_remapped_vl_config(ids.shape[0]), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    ids_j, grid_j = jnp.asarray(ids)[None], jnp.asarray(grid)

    # Eager: concrete grid -> works.
    pos_eager, _ = model.get_rope_index(ids_j, image_grid_thw=grid_j)
    assert np.asarray(pos_eager).shape == (3, 1, ids.shape[0])

    # Under trace: grid becomes a tracer -> np.array(tracer) raises.
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(lambda g: model.get_rope_index(ids_j, image_grid_thw=g)[0])(grid_j)


@pytest.mark.slow
def test_live_teacher_distillation_step_trains_student(rows):
    """End-to-end trainability through the REAL distillation step on a pack row (live teacher).

    Builds a tiny student and a differently-initialized teacher VL model, wraps them in
    ``EasyDeLState`` (student with an optimizer), and calls the SAME ``distillation_step`` the
    ``DistillationTrainer`` compiles: ``gradient_accumulation_steps=1``, one optimizer step. The pack
    carries no teacher logits, so the teacher MODEL is run live — exactly the path the embeds-only
    pack hits in training.

    Scaling contract (the Phase-2 finding): ``get_rope_index`` is host/eager-only (see the test
    above), so the jitted/remat'd trainer step must NOT receive ``image_grid_thw`` to turn into
    positions under trace. Instead we precompute the mRoPE ``position_ids`` on host (concrete grid) and
    pass them in the batch; the forward then skips ``get_rope_index`` (``if position_ids is None``).
    The batch carries precomputed ``image_embeds`` + ``position_ids`` + ``labels``; the trainer's
    kwargs filter (``__call__`` has VAR_KEYWORD) passes them to BOTH the teacher and student forwards.
    Asserts a finite positive loss, finite non-zero student gradients (folded into
    ``metrics.max_grad_norm``), an applied optimizer step, and at least one moved param.
    """
    import jax
    import jax.numpy as jnp
    import optax
    import spectrax as spx
    from jax.sharding import PartitionSpec

    from easydel.infra.base_state import EasyDeLState
    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    from easydel.trainers.distillation_trainer._fn import distillation_step

    row = next(r for r in rows if int(r["n_images"]) > 0)
    batch, _n_real = _remapped_pack_batch(row, pad_tail=0)
    batch.pop("n_real_embeds", None)  # metadata, not a model input
    seq = int(batch["input_ids"].shape[1])

    # Student and teacher share the arch but differ in init seed, so the soft-KL term is non-degenerate.
    student_model = Qwen3_5ForConditionalGeneration(
        config=_remapped_vl_config(seq), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32
    )
    teacher_model = Qwen3_5ForConditionalGeneration(
        config=_remapped_vl_config(seq), rngs=spx.Rngs(7), dtype=jnp.float32, param_dtype=jnp.float32
    )
    student_state = EasyDeLState.create(model=student_model, tx=optax.adam(1e-3), init_opt_state=True)
    teacher_state = EasyDeLState.create(model=teacher_model, tx=None)  # inference-only; teacher_forward only reads .model

    # Host-precompute mRoPE positions from the (concrete) grid, then drop the grid so the jitted step
    # never traces get_rope_index. This is the actual scale path for the embeds-only pack.
    position_ids, _ = student_model.base_model.get_rope_index(
        batch["input_ids"], image_grid_thw=batch["image_grid_thw"], attention_mask=batch["attention_mask"]
    )
    batch["position_ids"] = position_ids
    batch.pop("image_grid_thw", None)

    with student_state.model.mesh:
        new_state, metrics = distillation_step(
            student_state,
            batch,
            teacher_state,
            partition_spec=PartitionSpec(),  # single-device CPU: replicate every leaf (no axis-name dependency)
            gradient_accumulation_steps=1,
            temperature=4.0,
            alpha=0.9,
            is_training=True,
        )

    loss_v = float(metrics.loss)
    assert np.isfinite(loss_v) and loss_v > 0.0, f"distillation step produced a non-finite/zero loss {loss_v}"
    mgn = float(metrics.max_grad_norm)
    assert np.isfinite(mgn) and mgn > 1e-6, f"student received non-finite/zero gradient (max_grad_norm={mgn})"
    assert int(new_state.step) == int(student_state.step) + 1, "optimizer step was not applied"

    moved = any(
        not np.allclose(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32)))
        for a, b in zip(
            jax.tree_util.tree_leaves(student_state.graphstate),
            jax.tree_util.tree_leaves(new_state.graphstate),
            strict=False,
        )
    )
    assert moved, "no student parameter changed after the distillation step — training is a no-op"


@pytest.mark.slow
def test_bucketed_static_batch_jit_stable_no_rope_trace(rows):
    """BUCKETED static-shape scaling proof. With mRoPE ``position_ids`` precomputed on host (eager,
    concrete grid) and passed into the batch, a JITTED forward consumes a fixed-shape precomputed-embed
    batch WITHOUT tracing the eager-only ``get_rope_index`` (which would raise — see
    ``test_get_rope_index_is_eager_only_under_jit``). Two batches of the SAME ``(B, S)`` window reuse the
    one compiled program (no retrace); a DIFFERENT ``S`` forces a recompile. This is the concrete
    rationale for bucketing: one compiled step per fixed bucket window instead of a recompile per
    distinct batch length.
    """
    import jax
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    # Two SHORTEST real image rows -> a small, fast CPU forward; the proof is shape-driven, not size-driven.
    # The fixture concatenates every local partition, so pick the two shortest image rows FROM A SINGLE
    # (source, area_bucket) partition: a static-S batch is one partition, which collate_bucket_static asserts.
    img_by_partition: dict = {}
    for r in rows:
        if int(r["n_images"]) > 0:
            img_by_partition.setdefault((r.get("source"), r.get("area_bucket")), []).append(r)
    pairs = [sorted(g, key=lambda r: len(r["input_ids"]))[:2] for g in img_by_partition.values() if len(g) >= 2]
    assert pairs, "need a partition with >=2 image rows to build a static batch"
    img_rows = min(pairs, key=lambda pair: max(len(r["input_ids"]) for r in pair))
    S = max(len(r["input_ids"]) for r in img_rows) + 6  # fixed bucket window above the longest row
    cap = sum(_decode_row_embeds(r).shape[0] for r in img_rows) + 4  # fixed embed-row cap (constant across S)

    config = _remapped_vl_config(S + 64)
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    gdef, gstate, gother = model.split_module()

    def make_batch(seq_len: int) -> dict:
        b = collate_bucket_static(img_rows, pad_id=0, seq_len=seq_len, max_embed_rows=cap)
        ids_rm = np.stack([_remap_ids(np.asarray(b["input_ids"][i])) for i in range(len(img_rows))], axis=0)
        b["input_ids"] = jnp.asarray(ids_rm)
        # Host-precompute (3,B,S) mRoPE positions from the concrete grid (eager); the jitted fwd never does.
        pos, _ = model.base_model.get_rope_index(
            b["input_ids"], image_grid_thw=b["image_grid_thw"], attention_mask=b["attention_mask"]
        )
        b["position_ids"] = pos
        return b

    n_traces = {"n": 0}

    @jax.jit
    def fwd(gs, input_ids, attention_mask, image_embeds, position_ids):
        n_traces["n"] += 1  # side effect runs once per COMPILE (trace), not per call
        m = model.merge_module(gdef, gs, gother)
        return m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            position_ids=position_ids,
            apply_lm_head=True,
        ).logits

    b1 = make_batch(S)
    logits1 = np.asarray(fwd(gstate, b1["input_ids"], b1["attention_mask"], b1["image_embeds"], b1["position_ids"]).astype(jnp.float32))
    assert logits1.shape == (2, S, _RM_VOCAB), "unexpected logits shape from the static bucket forward"
    assert np.isfinite(logits1).all(), "jitted host-position_ids forward produced non-finite logits"
    assert n_traces["n"] == 1, "first call should compile exactly once"

    # Same (B, S) window -> reuse the compiled program (no recompile).
    b1b = make_batch(S)
    _ = fwd(gstate, b1b["input_ids"], b1b["attention_mask"], b1b["image_embeds"], b1b["position_ids"])
    assert n_traces["n"] == 1, "a same-shape batch unexpectedly recompiled (bucketing would not amortize)"

    # Different S (same fixed embed cap) -> a new compiled program: the recompile bucketing avoids.
    b2 = make_batch(S + 32)
    _ = fwd(gstate, b2["input_ids"], b2["attention_mask"], b2["image_embeds"], b2["position_ids"])
    assert n_traces["n"] == 2, "changing the sequence window did not recompile (cache should have grown)"


@pytest.mark.slow
def test_two_image_row_both_images_inject_and_train(rows):
    """A REAL multi-image row (mimic_cgd: 2 images/row): both images' precomputed embeds scatter to
    their own contiguous placeholder runs AND both carry non-zero gradient into the loss. Proves
    multi-image rows train EVERY image (not just the first), and that the per-image blob -> placeholder
    -run alignment survives concatenation + scatter + the real decoder. Falls back to skip only if no
    2-image row is staged."""
    import jax
    import jax.numpy as jnp
    import spectrax as spx

    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    row = next((r for r in rows if int(r["n_images"]) == 2), None)
    if row is None:
        pytest.skip("no 2-image row staged (pull a mimic_cgd partition to exercise multi-image)")

    ent = list(row["embed_n_tok"])
    assert len(ent) == 2 and len(row["image_embeds"]) == 2, "expected exactly 2 per-image embed blobs"
    ids = np.asarray(row["input_ids"], dtype=np.int32)
    place_pos = np.where(ids == IMAGE_PLACEHOLDER_ID)[0]
    assert place_pos.shape[0] == sum(ent), "placeholder count != total embed rows for the 2-image row"
    runs = 1 + int((np.diff(place_pos) != 1).sum())
    assert runs == 2, f"expected 2 contiguous placeholder runs (one per image), got {runs}"

    # Remapped single-row batch (concatenates both images' embeds: rows [0:ent0] then [ent0:ent0+ent1]).
    batch, n_real = _remapped_pack_batch(row, pad_tail=0)
    assert n_real == sum(ent)
    config = _remapped_vl_config(int(batch["input_ids"].shape[1]))
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    gdef, gstate, gother = model.split_module()
    ids_j, am_j, grid_j, labels_j = (batch["input_ids"], batch["attention_mask"], batch["image_grid_thw"], batch["labels"])
    assert np.asarray(grid_j).shape == (2, 3), "two images should yield a (2,3) grid"

    def loss_wrt_embeds(emb):
        m = model.merge_module(gdef, gstate, gother)
        outputs, _metrics = m.compute_loss(
            input_ids=ids_j, attention_mask=am_j, image_embeds=emb, image_grid_thw=grid_j, labels=labels_j
        )
        return outputs.loss

    g_emb = np.asarray(jax.grad(loss_wrt_embeds)(batch["image_embeds"]).astype(jnp.float32))
    assert np.isfinite(g_emb).all(), "d(loss)/d(image_embeds) was non-finite for the 2-image row"
    n0 = ent[0]
    img0_norm = float(np.linalg.norm(g_emb[:n0]))
    img1_norm = float(np.linalg.norm(g_emb[n0 : n0 + ent[1]]))
    assert img0_norm > 1e-6, "image-0 embeds got zero gradient — first image is not in the differentiable loss"
    assert img1_norm > 1e-6, "image-1 embeds got zero gradient — second image is not in the differentiable loss"


@pytest.mark.slow
def test_real_loader_carries_bucket_collator_into_jitted_step():
    """END-TO-END through the REAL EasyDeL data path: a ``ParquetShardedSource`` over a staged pack
    parquet, wrapped in the REAL ``AsyncDataLoader`` with ``collate_fn=collate_bucket_static`` (the
    new hook), yields a fixed static-shape precomputed-embed batch that flows into a jitted forward
    and produces finite logits.

    Two things are proven beyond the proof-only harness above:
      1. The ``collate_fn`` hook is actually wired through ``LoadStageConfig`` -> ``AsyncDataLoader`` ->
         ``batch_iterator``: the loader emits the bucket collator's static ``(B,S)`` / ``(cap,H)`` layout
         and its ``n_real_embeds`` field, whereas the DEFAULT loader (``collate_fn=None``) emits neither
         (ragged per-row stack, no ``n_real_embeds``) — so the batch shape is the hook's doing, not a
         pre-existing behaviour.
      2. That real-loader batch, with host-precomputed mRoPE ``position_ids``, runs through the jitted
         decoder to finite logits — i.e. the producer parquet -> real loader -> bucket collator ->
         jitted step path is closed.
    """
    import functools

    import jax
    import jax.numpy as jnp
    import pyarrow.parquet as pq
    import spectrax as spx

    from easydel.data.execution.loader import AsyncDataLoader
    from easydel.data.sources.base import ParquetShardedSource
    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    paths = [p for p in _parquet_paths() if "mimic_cgd" in p]
    if not paths:
        pytest.skip("no mimic_cgd parquet staged for the real-loader end-to-end path")
    path = paths[0]

    # Pre-scan the shard to size the static bucket window (S) and embed-row cap from real maxima.
    scan = pq.read_table(path).to_pylist()
    B = 2
    S = max(len(r["input_ids"]) for r in scan) + 8
    cap = B * max(sum(r["embed_n_tok"]) for r in scan) + 4

    bucket_collate = functools.partial(collate_bucket_static, pad_id=0, seq_len=S, max_embed_rows=cap)
    source = ParquetShardedSource(data_files=path)
    loader = AsyncDataLoader(
        source=source, batch_size=B, prefetch_enabled=False, shuffle_buffer_size=None, drop_last=True, collate_fn=bucket_collate
    )
    batch = next(iter(loader))

    # (1) the hook ran: static padded shape + scatter layout + the collator-only metadata field.
    assert np.asarray(batch["input_ids"]).shape == (B, S), "loader did not apply the bucket collator's fixed window"
    assert np.asarray(batch["image_embeds"]).shape == (cap, HIDDEN), "loader did not apply the fixed embed-row cap"
    assert "n_real_embeds" in batch, "bucket collator's n_real_embeds missing — collate_fn was not used"
    assert int((np.asarray(batch["input_ids"]) == IMAGE_PLACEHOLDER_ID).sum()) == batch["n_real_embeds"]

    # Contrast: the DEFAULT loader (collate_fn=None) produces neither the static layout nor n_real_embeds.
    default_loader = AsyncDataLoader(
        source=source, batch_size=B, prefetch_enabled=False, shuffle_buffer_size=None, drop_last=True
    )
    default_batch = next(iter(default_loader))
    assert "n_real_embeds" not in default_batch, "default collator unexpectedly produced n_real_embeds (hook not isolated)"
    assert np.asarray(default_batch["input_ids"]).shape != (B, S), "default collator unexpectedly matched the static window"

    # (2) end-to-end: host position_ids + the real-loader batch -> jitted forward -> finite logits.
    ids_rm = np.stack([_remap_ids(np.asarray(batch["input_ids"][i])) for i in range(B)], axis=0)
    config = _remapped_vl_config(S + 8)
    model = Qwen3_5ForConditionalGeneration(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    pos, _ = model.base_model.get_rope_index(
        jnp.asarray(ids_rm), image_grid_thw=batch["image_grid_thw"], attention_mask=batch["attention_mask"]
    )
    gdef, gstate, gother = model.split_module()

    @jax.jit
    def fwd(gs, input_ids, attention_mask, image_embeds, position_ids):
        m = model.merge_module(gdef, gs, gother)
        return m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            position_ids=position_ids,
            apply_lm_head=True,
        ).logits

    logits = np.asarray(
        fwd(gstate, jnp.asarray(ids_rm), batch["attention_mask"], batch["image_embeds"], pos).astype(jnp.float32)
    )
    assert logits.shape == (B, S, _RM_VOCAB)
    assert np.isfinite(logits).all(), "real-loader bucket batch did not flow through the jitted step to finite logits"


# --- packed-sequence collator (collate_packed_embeds) -------------------------------------------
def _packable_row(text_len: int, imgs: list[int], source: str = "pixmo-points", area_bucket: int = 1024,
                  slen_band: int | None = None) -> dict:
    """Synthetic pack row for the packing collator. ``imgs`` = per-image post-merge token counts; each
    image is laid down as a contiguous placeholder run followed by a single text token so it forms its
    own modality group, with ``image_grid_thw = (1, 2*nt, 2)`` which under spatial_merge_size=2 yields
    exactly ``nt`` LLM image tokens (matching the placeholder run). Then ``text_len`` trailing text
    tokens. embed blobs are deterministic bf16."""
    import ml_dtypes

    ent = list(imgs)
    blobs: list[bytes] = []
    grid: list[int] = []
    ids: list[int] = []
    for nt in ent:
        arr = (np.arange(nt * HIDDEN, dtype=np.float32).reshape(nt, HIDDEN) % 7).astype(ml_dtypes.bfloat16)
        blobs.append(arr.tobytes())
        ids += [IMAGE_PLACEHOLDER_ID] * nt + [5]  # trailing text token splits consecutive images into groups
        grid += [1, 2 * nt, 2]
    ids += [5] * text_len
    row = {
        "embed_dim": HIDDEN,
        "embed_n_tok": ent,
        "image_embeds": blobs,
        "n_images": len(ent),
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids,
        "image_grid_thw": grid,
        "seq_len": len(ids),
        "source": source,
        "area_bucket": area_bucket,
    }
    if slen_band is not None:
        row["slen_band"] = slen_band
    return row


def _text_row(tokens: list[int]) -> dict:
    """All-text pack row (no images) from explicit token ids."""
    return {
        "embed_dim": HIDDEN,
        "embed_n_tok": [],
        "image_embeds": [],
        "n_images": 0,
        "input_ids": list(tokens),
        "attention_mask": [1] * len(tokens),
        "labels": list(tokens),
        "image_grid_thw": [],
        "seq_len": len(tokens),
        "source": "pixmo-points",
        "area_bucket": 1024,
    }


def test_collate_packed_embeds_contract():
    """Greedy packing into a fixed ``seq_len`` window: static shapes, example-order embed concat,
    placeholder<->embed parity, contiguous per-window ``segment_ids`` (pad tail = -1), masked tail,
    and per-segment-reset 3D M-RoPE positions (each segment == its standalone-example layout)."""
    from easydel.modules.qwen3_5.modeling_qwen3_5 import _get_rope_index_from_mm_token_types

    rows = [_packable_row(text_len=6, imgs=[]), _packable_row(text_len=3, imgs=[2]), _packable_row(text_len=4, imgs=[3])]
    filled = sum(len(r["input_ids"]) for r in rows)
    seq_len = filled + 5  # all rows fit -> a single window with a padded tail
    total_real = sum(sum(r["embed_n_tok"]) for r in rows)  # 0 + 2 + 3 = 5
    cap = total_real + 4  # exercise the zero-padded embed tail

    out = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=cap, n_windows=1)
    ids = np.asarray(out["input_ids"])
    emb = np.asarray(out["image_embeds"])
    seg = np.asarray(out["segment_ids"])
    pos = np.asarray(out["position_ids"])
    attn = np.asarray(out["attention_mask"])
    labels = np.asarray(out["labels"])

    assert ids.shape == (1, seq_len), "short rows should pack into one fixed-width window"
    assert emb.shape == (cap, HIDDEN)
    assert seg.shape == (1, seq_len)
    assert pos.shape == (3, 1, seq_len), "positions must be 3D M-RoPE (3, windows, seq_len)"

    # (a) placeholder <-> embed parity, embeds concatenated in EXAMPLE order, zero-padded tail.
    assert int((ids == IMAGE_PLACEHOLDER_ID).sum()) == total_real == out["n_real_embeds"]
    assert np.array_equal(emb[total_real:], np.zeros((cap - total_real, HIDDEN), np.float32))
    cursor = 0
    for r in rows:
        dec = _decode_row_embeds(r)
        assert np.array_equal(emb[cursor : cursor + dec.shape[0]], dec), "embeds not concatenated in example order"
        cursor += dec.shape[0]
    assert np.asarray(out["image_grid_thw"]).shape == (sum(int(r["n_images"]) for r in rows), 3)

    # segment_ids: contiguous per-example index over the filled span, -1 over the padded tail.
    expected_seg = [k for k, r in enumerate(rows) for _ in range(len(r["input_ids"]))] + [-1] * (seq_len - filled)
    assert np.array_equal(seg[0], np.asarray(expected_seg, dtype=np.int32))
    assert np.all(attn[0, :filled] == 1) and np.all(attn[0, filled:] == 0), "padded tail not attention-masked"
    assert np.all(labels[0, filled:] == -100), "padded tail labels not -100"
    assert not (ids[0, filled:] == IMAGE_PLACEHOLDER_ID).any(), "placeholder leaked into the padded tail"

    # (c) per-segment-reset positions: each segment's columns equal the helper's output for that
    # example ALONE (current_pos starts at 0) -> no cumulative drift, every segment resets to 0.
    offset = 0
    for k, r in enumerate(rows):
        ids_e = np.asarray(r["input_ids"], dtype=np.int32)
        length = ids_e.shape[0]
        grid = np.asarray(r["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
        mm = (ids_e == IMAGE_PLACEHOLDER_ID).astype(np.int32).reshape(1, -1)
        pos_e, _ = _get_rope_index_from_mm_token_types(
            input_ids=ids_e.reshape(1, -1),
            mm_token_type_ids=mm,
            image_grid_thw=grid if grid.shape[0] else None,
            attention_mask=None,
            spatial_merge_size=2,
        )
        pos_e = np.asarray(pos_e)[:, 0, :]
        seg_slice = slice(offset, offset + length)
        assert np.array_equal(pos[:, 0, seg_slice], pos_e), f"segment {k}: positions are not the per-example reset layout"
        assert int(pos[:, 0, seg_slice].min()) == 0, f"segment {k}: positions did not reset to 0"
        offset += length


def test_collate_packed_embeds_greedy_window_count():
    """FCFS greedy fill: rows that overflow the current window start the next one. With the leading
    dim pinned to ``n_windows`` the batch is ALWAYS ``(n_windows, seq_len)``; per-window
    ``segment_ids`` restart at 0. A row-list that packs into more than ``n_windows`` windows RAISES."""
    rows = [_packable_row(text_len=8, imgs=[]) for _ in range(3)]  # each row is length 8
    seq_len = 20  # two rows (16) fit; the third (24) spills into a second window -> M=2
    out = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=0, n_windows=2)
    ids = np.asarray(out["input_ids"])
    seg = np.asarray(out["segment_ids"])

    assert ids.shape == (2, seq_len), "leading dim must equal n_windows (=2) here"
    assert np.array_equal(seg[0, :16], np.asarray([0] * 8 + [1] * 8, dtype=np.int32))
    assert np.all(seg[0, 16:] == -1)
    assert np.array_equal(seg[1, :8], np.asarray([0] * 8, dtype=np.int32)), "second window must restart segment ids at 0"
    assert np.all(seg[1, 8:] == -1)

    # M (=2) > n_windows (=1) -> raise, never silently drop the spilled rows.
    with pytest.raises(AssertionError, match="window overflow"):
        collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=0, n_windows=1)


def test_collate_packed_embeds_underfill():
    """Static leading dim: when greedy packing yields fewer windows than ``n_windows`` the batch is
    STILL ``(n_windows, seq_len)`` and the trailing ``n_windows - M`` windows are emitted whole-padded
    -- attention 0, ``segment_ids`` -1, ``labels`` -100, no image placeholders -- so they contribute
    zero loss and zero attention. The real window's content is unchanged by the padding rows."""
    rows = [_packable_row(text_len=6, imgs=[2]), _packable_row(text_len=4, imgs=[3])]
    filled = sum(len(r["input_ids"]) for r in rows)
    seq_len = filled + 5  # both rows fit one window -> M=1
    total_real = sum(sum(r["embed_n_tok"]) for r in rows)
    n_windows = 3  # ask for 3 -> windows 1 and 2 are all-pad

    out = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=total_real, n_windows=n_windows)
    ids = np.asarray(out["input_ids"])
    seg = np.asarray(out["segment_ids"])
    attn = np.asarray(out["attention_mask"])
    labels = np.asarray(out["labels"])

    # leading dim == n_windows regardless of the (smaller) real window count.
    assert ids.shape == (n_windows, seq_len) and seg.shape == (n_windows, seq_len)
    assert np.asarray(out["position_ids"]).shape == (3, n_windows, seq_len)
    # placeholder<->embed parity counts only the real window (trailing pad windows carry none).
    assert int((ids == IMAGE_PLACEHOLDER_ID).sum()) == total_real == out["n_real_embeds"]

    # window 0 is the real packed window; windows 1..2 are fully inert.
    assert np.any(attn[0] == 1), "the real window must have live tokens"
    for w in range(1, n_windows):
        assert np.all(attn[w] == 0), f"pad window {w} must be fully attention-masked"
        assert np.all(seg[w] == -1), f"pad window {w} segment_ids must all be -1"
        assert np.all(labels[w] == -100), f"pad window {w} labels must all be ignore (-100)"
        assert np.all(ids[w] == 0), f"pad window {w} must be pure pad_id"
        assert not (ids[w] == IMAGE_PLACEHOLDER_ID).any(), f"pad window {w} must carry no placeholders"

    # Equivalence on the real segments under underfill: the real window (row 0) is BYTE-IDENTICAL to
    # the same rows collated with n_windows == M. Combined with batch-row independence in the forward,
    # the trailing pad windows cannot change the real window's logits -- so the model-level equivalence
    # proven in test_collate_packed_embeds_equivalence carries over unchanged. (Asserting it this way
    # avoids running the forward on an all-masked pad row, whose softmax behaviour is out of scope.)
    base = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=total_real, n_windows=1)
    for key in ("input_ids", "attention_mask", "labels", "segment_ids"):
        assert np.array_equal(np.asarray(out[key])[0], np.asarray(base[key])[0]), f"real window {key} changed under underfill"
    assert np.array_equal(np.asarray(out["position_ids"])[:, 0], np.asarray(base["position_ids"])[:, 0]), "real window positions changed under underfill"
    assert np.array_equal(np.asarray(out["image_embeds"]), np.asarray(base["image_embeds"])), "embed side-channel changed under underfill"


def test_collate_packed_embeds_guards():
    """Guard suite: pad/placeholder collision, single-row overflow, and source/area_bucket
    homogeneity all RAISE; ``slen_band`` is DELIBERATELY not checked (packing mixes lengths by design)."""
    a = _packable_row(text_len=4, imgs=[2])
    b = _packable_row(text_len=3, imgs=[2])
    seq_len, cap = 64, 8

    with pytest.raises(AssertionError, match="collide"):
        collate_packed_embeds([a], pad_id=IMAGE_PLACEHOLDER_ID, seq_len=seq_len, max_embed_rows=cap, n_windows=1)

    big = _packable_row(text_len=seq_len + 1, imgs=[])  # single row longer than the window
    with pytest.raises(AssertionError, match="overflow guard"):
        collate_packed_embeds([big], pad_id=0, seq_len=seq_len, max_embed_rows=cap, n_windows=1)

    a_src, b_src = dict(a), dict(b)
    a_src["source"], b_src["source"] = "src_a", "src_b"
    with pytest.raises(AssertionError, match=r"heterogeneous batch.*source"):
        collate_packed_embeds([a_src, b_src], pad_id=0, seq_len=seq_len, max_embed_rows=cap, n_windows=2)

    a_ab, b_ab = dict(a), dict(b)
    a_ab["area_bucket"], b_ab["area_bucket"] = 1024, 1536
    with pytest.raises(AssertionError, match=r"heterogeneous batch.*area_bucket"):
        collate_packed_embeds([a_ab, b_ab], pad_id=0, seq_len=seq_len, max_embed_rows=cap, n_windows=2)

    # mixed slen_band is allowed: packing deliberately concatenates examples of different lengths.
    a_sb, b_sb = dict(a), dict(b)
    a_sb["slen_band"], b_sb["slen_band"] = 192, 448
    out = collate_packed_embeds([a_sb, b_sb], pad_id=0, seq_len=seq_len, max_embed_rows=cap, n_windows=2)
    assert np.asarray(out["input_ids"]).shape == (2, seq_len), "leading dim must equal n_windows"


@pytest.mark.slow
def test_collate_packed_embeds_equivalence():
    """(b)+(d): a packed batch of N short examples gives the SAME per-example logits as running each
    example unpacked. ``segment_ids`` (-> block-diagonal ``mask_info``) isolates the documents
    (cross-example attention masked) and the per-segment-reset ``position_ids`` reproduce each
    example's own positions, so packed per-segment logits match the unpacked references within tol.

    This is the collator-side equivalence check. The MODEL side -- that ``MaskInfo.from_segments``
    truly yields block-diagonal full attention AND resets the GDR linear-attention recurrence, forward
    and backward, across full/hybrid/linear-only layer mixes -- is proven in
    tests/modules/test_qwen3_5_packing.py; we cite it rather than re-prove it here. Underfill
    (M < n_windows) equivalence on the real window is covered structurally in
    ``test_collate_packed_embeds_underfill`` (trailing pad windows are byte-inert and, by batch-row
    independence, cannot perturb the real window's logits)."""
    import jax.numpy as jnp

    from ejkernel.types import MaskInfo

    docs = [[5, 9, 2, 7, 1], [3, 8, 4], [6, 2, 9, 5]]
    rows = [_text_row(d) for d in docs]
    seq_len = sum(len(d) for d in docs)  # exact fit -> one fully-packed window, no padding
    model = _build_tiny_text_model(seq_len)

    # unpacked references: each document run on its own.
    refs = [
        np.asarray(model(input_ids=jnp.asarray([d], dtype=jnp.int32), apply_lm_head=True).logits.astype(jnp.float32))[0]
        for d in docs
    ]

    out = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=0, n_windows=1)
    assert np.asarray(out["input_ids"]).shape == (1, seq_len), "the docs should fully pack into one window"
    seg = np.asarray(out["segment_ids"])
    pos = np.asarray(out["position_ids"])  # (3, 1, seq_len); text -> all 3 axes equal, feed the 1D row
    mask_info = MaskInfo.from_segments(q_segment_ids=jnp.asarray(seg, dtype=jnp.int32))

    packed = model(
        input_ids=out["input_ids"],
        mask_info=mask_info,
        position_ids=jnp.asarray(pos[0]),
        apply_lm_head=True,
    )
    lp = np.asarray(packed.logits.astype(jnp.float32))[0]

    for k, d in enumerate(docs):
        cols = np.where(seg[0] == k)[0]
        assert cols.shape[0] == len(d)
        max_abs = float(np.max(np.abs(lp[cols] - refs[k])))
        assert max_abs < 1e-3, f"packed segment {k} logits diverge from the unpacked run, max|Δ|={max_abs:.2e}"


@pytest.mark.slow
def test_collate_packed_embeds_underfill_backward_finite():
    """Training feeds ALL ``n_windows`` rows through forward+BACKWARD -- the inert all-pad windows
    that underfill produces CANNOT be skipped (that is the entire point of the static leading dim).
    An all-pad window (attention 0 -> ``segment_ids`` all -1) is the classic all-masked-softmax NaN
    footgun: softmax over a fully-masked row -> 0/0 -> NaN in the FORWARD, and ``labels`` -100 does
    NOT save the backward -- the NaN is upstream in the forward graph and backprops into NaN grads on
    the SHARED params (a where-masked loss can't undo an already-NaN activation). Per the standing
    "empirically backstop degenerate shapes" rule this is RUN, not argued: real trainer entry
    (``compute_loss`` folds ``segment_ids`` -> ``mask_info``) on a batch with a trailing all-pad
    window, asserting the scalar loss AND every param gradient are finite."""
    import jax

    rows = [_text_row([5, 9, 2, 7, 1]), _text_row([3, 8, 4])]
    seq_len = sum(len(r["input_ids"]) for r in rows)  # 8 -> both rows pack into ONE window (M=1)
    n_windows = 2  # ask for 2 -> exactly one trailing all-pad window
    batch = collate_packed_embeds(rows, pad_id=0, seq_len=seq_len, max_embed_rows=0, n_windows=n_windows)
    attn = np.asarray(batch["attention_mask"])
    assert attn.shape == (n_windows, seq_len) and np.all(attn[1] == 0), "need a fully-masked pad window"

    model = _build_tiny_text_model(seq_len)
    # Mirror the trainer's grad path (trainers/trainer/_fn.py): differentiate the scalar loss w.r.t.
    # the trainable graphstate, re-binding the module each call via merge_module (== state.merge(tree)).
    gdef, params, others = model.split_module()

    def loss_fn(params):
        m = model.merge_module(gdef, params, others)
        loss_out, _ = m.compute_loss(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            labels=batch["labels"],
        )
        return loss_out.loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    loss = float(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert np.isfinite(loss), f"scalar loss non-finite ({loss}) with a trailing all-pad window present"
    assert leaves, "expected differentiable param gradients"
    bad = sum(1 for g in leaves if not bool(np.all(np.isfinite(np.asarray(g)))))
    assert not bad, (
        f"{bad}/{len(leaves)} param-grad tensors non-finite with a trailing all-pad window -- "
        "all-masked-softmax NaN backprops into shared params; pad windows must self-attend"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
