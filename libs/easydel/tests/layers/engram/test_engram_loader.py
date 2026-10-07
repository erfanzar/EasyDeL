# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Engram loader CPU contracts: header scan and row-chunked raw reads on a tiny synthetic safetensors file, plus the
native key mapper's engram-table handling against the REAL released key index. No array computation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from easydel.layers.engram import engram_table_checkpoint_keys, read_engram_rows, scan_engram_tensors
from easydel.modules.deepseek_v41._native_checkpoint import native_key_to_easydel_v41, native_key_to_easydel_v41_vl

pytestmark = pytest.mark.cpu_ok

_INDEX = Path(__file__).resolve().parents[5] / ".xerxes/projects/deepseek-v41/ref/model.safetensors.index.json"


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Hand-rolled safetensors writer (the one-byte F8 dtypes are not writable through numpy)."""
    header, offset, blobs = {}, 0, []
    for name, (dtype, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


@pytest.fixture()
def synthetic(tmp_path: Path):
    rng = np.random.default_rng(0)
    weight = rng.integers(0, 256, size=(11, 64), dtype=np.uint8)
    scale = rng.integers(100, 140, size=(11, 2), dtype=np.uint8)
    other = rng.integers(0, 256, size=(3, 5), dtype=np.uint8)
    _write_safetensors(
        tmp_path / "a.safetensors",
        {"layers.3.engram.embed.weight": ("F8_E4M3", weight), "layers.0.attn.wq_a.weight": ("U8", other)},
    )
    _write_safetensors(tmp_path / "b.safetensors", {"layers.3.engram.embed.scale": ("F8_E8M0", scale)})
    return tmp_path, weight, scale


def test_scan_finds_table_tensors_across_shards_and_ignores_others(synthetic) -> None:
    root, _, _ = synthetic
    found = scan_engram_tensors([root / "a.safetensors", root / "b.safetensors"])
    assert set(found) == {3}
    assert found[3]["weight"].shape == (11, 64) and found[3]["weight"].dtype == "F8_E4M3"
    assert found[3]["scale"].shape == (11, 2) and found[3]["scale"].dtype == "F8_E8M0"
    assert found[3]["weight"].path.endswith("a.safetensors") and found[3]["scale"].path.endswith("b.safetensors")


def test_chunked_reads_reassemble_the_tensor_exactly(synthetic) -> None:
    root, weight, scale = synthetic
    found = scan_engram_tensors([root / "a.safetensors", root / "b.safetensors"])[3]
    for ref, expected in ((found["weight"], weight), (found["scale"], scale)):
        pieces = [read_engram_rows(ref, s, min(s + 4, 11)) for s in range(0, 11, 4)]
        np.testing.assert_array_equal(np.concatenate(pieces), expected)


def test_padding_rows_follow_the_reference_convention(synthetic) -> None:
    root, weight, scale = synthetic
    found = scan_engram_tensors([root / "a.safetensors", root / "b.safetensors"])[3]
    w = read_engram_rows(found["weight"], 8, 16, padded_rows=16, pad_value=0)
    s = read_engram_rows(found["scale"], 8, 16, padded_rows=16, pad_value=1)
    np.testing.assert_array_equal(w[:3], weight[8:])
    assert (w[3:] == 0).all() and w.shape == (8, 64)
    np.testing.assert_array_equal(s[:3], scale[8:])
    assert (s[3:] == 1).all()
    with pytest.raises(ValueError):
        read_engram_rows(found["weight"], 10, 17, padded_rows=16)


def test_unexpected_dtype_is_rejected(tmp_path: Path) -> None:
    _write_safetensors(
        tmp_path / "c.safetensors", {"layers.1.engram.embed.weight": ("BF16", np.zeros((2, 64), np.uint16))}
    )
    with pytest.raises(ValueError, match="expected 2-D"):
        scan_engram_tensors([tmp_path / "c.safetensors"])


def test_native_mapper_keeps_engram_tables_out_of_the_conversion_path() -> None:
    for layer in (1, 14):
        for leaf in ("weight", "scale"):
            key = f"layers.{layer}.engram.embed.{leaf}"
            assert native_key_to_easydel_v41(key) is None
            assert native_key_to_easydel_v41_vl(key) is None
        assert native_key_to_easydel_v41(f"layers.{layer}.engram.wkv.weight") == (
            f"model.layers.{layer}.engram.kv_proj.weight"
        )
        assert native_key_to_easydel_v41(f"layers.{layer}.engram.q_weight") == f"model.layers.{layer}.engram.q_weight"


def test_released_index_lists_the_engram_tables_in_the_last_two_shards() -> None:
    if not _INDEX.is_file():
        pytest.skip(f"released key index not found at {_INDEX}")
    weight_map = json.loads(_INDEX.read_text())["weight_map"]
    tables = engram_table_checkpoint_keys(weight_map)
    assert set(tables) == {1, 14}
    assert {weight_map[k] for k in tables[1].values()} == {"model-00047-of-00048.safetensors"}
    assert {weight_map[k] for k in tables[14].values()} == {"model-00048-of-00048.safetensors"}
    assert set(tables[1]) == {"weight", "scale"}
