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

"""Bit-exact parity of the NumPy engram hasher against the official DeepSeek-V4.1 reference ``engram.py``.

The reference (``.xerxes/projects/deepseek-v41/ref/engram.py``) is loaded by file path and run on CPU torch; nothing
here does accelerator array computation.
"""

from __future__ import annotations

import importlib.util
import sys
import typing as tp
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easydel.layers.engram import (
    DEAD,
    EngramHasher,
    EngramHashState,
    EngramWindowHasher,
    build_engram_token_map,
    engram_bucket_primes,
)
from easydel.modules.deepseek_v41.deepseek_v41_configuration import DeepseekV41Config

pytestmark = pytest.mark.cpu_ok

_REF_PATH = Path(__file__).resolve().parents[5] / ".xerxes" / "projects" / "deepseek-v41" / "ref" / "engram.py"

_PAD_ID = 2

# (layer_ids, max_ngram_size, n_heads, engram_vocab_size)
_LAYOUTS = [
    ((1, 3), 4, 2, 1000),
    ((0,), 2, 1, 50),
    ((5, 2, 9), 3, 3, 97),
]


@pytest.fixture(scope="module")
def ref() -> tp.Any:
    if not _REF_PATH.exists():
        pytest.skip(f"reference engram.py not found at {_REF_PATH}")
    name = "_deepseek_v41_ref_engram"
    spec = importlib.util.spec_from_file_location(name, _REF_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as err:  # sympy / torch missing
        pytest.skip(f"reference engram.py dependencies unavailable: {err}")
    return module


class _FakeHFTokenizer:
    """Exactly the surface the reference touches: ``len()`` and ``backend_tokenizer``."""

    def __init__(self, backend: tp.Any):
        self.backend_tokenizer = backend

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size()


_WORDS = [
    "[UNK]",
    "hello",
    "HELLO",  # id 2 == pad id; compresses with "hello"
    "Hello",
    "h\u00e9llo",
    "H\u00c9LLO",
    "he\u0301llo",  # decomposed accent
    " ",
    "  ",
    "\t",
    "\n",
    " the",
    "The",
    "THE",
    "the",
    "\uff41",  # fullwidth a -> a (NFKC)
    "a",
    "A",
    "\ufb01",  # fi ligature -> fi (NFKC)
    "fi",
    "\u0301",  # lone combining accent: normalises to the empty string
    "\ufffd",  # partial UTF-8 byte tokens are keyed by their raw form
    "x\ufffd",
    "y\ufffd",
    *[f"t{i}" for i in range(40)],
]


@pytest.fixture(scope="module")
def tokenizer() -> _FakeHFTokenizer:
    from tokenizers import Tokenizer, models

    vocab = {word: idx for idx, word in enumerate(_WORDS)}
    assert len(vocab) == len(_WORDS), "duplicate words in the fake vocab"
    return _FakeHFTokenizer(Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")))


@pytest.fixture(scope="module")
def token_map(tokenizer: _FakeHFTokenizer) -> np.ndarray:
    return build_engram_token_map(tokenizer)


def _ref_args(layer_ids, max_ngram_size, n_heads, vocab_size, compressed_size) -> SimpleNamespace:
    return SimpleNamespace(
        engram_layer_ids=tuple(layer_ids),
        engram_max_ngram_size=max_ngram_size,
        engram_n_heads=n_heads,
        engram_vocab_size=vocab_size,
        engram_num_embeddings=(0,) * len(layer_ids),  # stored only; the hash never reads it
        engram_head_dim=8,
        engram_pad_id=_PAD_ID,
        engram_compressed_vocab_size=compressed_size,
        max_batch_size=4,
        max_seq_len=64,
    )


def _make_pair(ref, tokenizer, token_map, layout):
    layer_ids, max_ngram_size, n_heads, vocab_size = layout
    compressed = int(token_map.max()) + 1
    args = _ref_args(layer_ids, max_ngram_size, n_heads, vocab_size, compressed)
    ref_layout = ref.EngramLayout.from_args(args)
    ref_state = ref.NgramHashState(args, ref_layout, tokenizer)
    hasher = EngramHasher(token_map, layer_ids, max_ngram_size, n_heads, vocab_size, _PAD_ID)
    return ref_layout, ref_state, hasher


def _ref_hash(ref_state, ids: np.ndarray, start_pos: int, dead: np.ndarray | None) -> np.ndarray:
    token_mask = None if dead is None else torch.from_numpy(~dead)
    return ref_state(torch.from_numpy(ids.astype(np.int64)), start_pos, token_mask).numpy()


def _random_batch(seed: int, vocab: int, dead_p: float = 0.0) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    batch = int(rng.integers(1, 5))
    length = int(rng.integers(6, 25))
    ids = rng.integers(0, vocab, size=(batch, length))
    dead = (rng.random((batch, length)) < dead_p) if dead_p > 0 else None
    return ids, dead


def test_token_map_matches_reference(ref, tokenizer, token_map):
    ref_lookup, ref_size = ref.build_compressed_token_map(tokenizer)
    assert token_map.shape == (len(_WORDS),)
    assert np.issubdtype(token_map.dtype, np.integer)
    np.testing.assert_array_equal(token_map, np.asarray(ref_lookup))
    assert int(token_map.max()) + 1 == ref_size

    ids = {word: int(token_map[i]) for i, word in enumerate(_WORDS)}
    # case, accents, decomposed accents, NFKC compatibility forms and a leading space all collapse together
    assert len({ids[w] for w in ("hello", "HELLO", "Hello", "h\u00e9llo", "H\u00c9LLO", "he\u0301llo")}) == 1
    assert len({ids[w] for w in (" the", "The", "THE", "the")}) == 1
    assert len({ids[w] for w in ("\uff41", "a", "A")}) == 1
    assert ids["\ufb01"] == ids["fi"]
    # every whitespace-only token collapses to one id, which is not the empty-string / other-token id
    assert len({ids[w] for w in (" ", "  ", "\t", "\n")}) == 1
    assert ids[" "] not in {ids["hello"], ids["the"], ids["a"], ids["\u0301"]}
    # distinct texts stay distinct; byte-fragment tokens are keyed by raw token, so they do not merge
    assert len({ids["hello"], ids["the"], ids["a"], ids["fi"]}) == 4
    assert len({ids["\ufffd"], ids["x\ufffd"], ids["y\ufffd"]}) == 3
    assert len({ids[f"t{i}"] for i in range(40)}) == 40
    # compression actually happened
    assert int(token_map.max()) + 1 < len(_WORDS)


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_primes_match_reference_layout(ref, layout):
    layer_ids, max_ngram_size, n_heads, vocab_size = layout
    args = _ref_args(layer_ids, max_ngram_size, n_heads, vocab_size, 10)
    ref_layout = ref.EngramLayout.from_args(args)
    got = engram_bucket_primes(len(layer_ids), max_ngram_size, n_heads, vocab_size)
    assert got == ref_layout.primes


def test_primes_match_reference_released_layout(ref):
    """The released V4.1 layout (16M buckets, 8 heads, 4-grams, 2 layers) against the reference and SPEC values."""
    args = _ref_args((1, 14), 4, 8, 16_000_000, 99092)
    ref_layout = ref.EngramLayout.from_args(args)
    got = engram_bucket_primes(2, 4, 8, 16_000_000)
    assert got == ref_layout.primes
    assert got[0][0][:3] == (16000057, 16000079, 16000081)
    assert got[1][0][:3] == (16000477, 16000487, 16000499)
    totals = [sum(p for per_ngram in layer for p in per_ngram) for layer in got]
    assert totals == [384006168, 384016682]


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_tables_and_multipliers_match_reference(ref, tokenizer, token_map, layout):
    ref_layout, ref_state, hasher = _make_pair(ref, tokenizer, token_map, layout)
    layer_ids, max_ngram_size, n_heads, _ = layout
    assert hasher.primes == ref_layout.primes
    np.testing.assert_array_equal(hasher.offsets, ref_state.offsets.numpy())
    np.testing.assert_array_equal(hasher.multipliers, ref_state.multipliers.numpy())
    assert hasher.pad_id == int(ref_state.pad_id)
    assert hasher.n_columns == (max_ngram_size - 1) * n_heads
    assert hasher.layer_ids == tuple(layer_ids)
    assert all(int(m) % 2 == 1 for m in hasher.multipliers.ravel())

    # independent derivation of the multipliers: seeded with the GLOBAL layer id, bounded by the compressed vocab
    compressed = int(token_map.max()) + 1
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    for row, layer_id in enumerate(layer_ids):
        expected = np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram_size,), dtype=np.int64)
        np.testing.assert_array_equal(hasher.multipliers[row], expected * 2 + 1)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stateless_hash_matches_reference(ref, tokenizer, token_map, layout, seed):
    _, ref_state, hasher = _make_pair(ref, tokenizer, token_map, layout)
    ids, _ = _random_batch(seed, len(_WORDS))
    expected = _ref_hash(ref_state, ids, 0, None)
    got = hasher.hash(ids)
    assert got.dtype == np.int32
    assert got.shape == (*ids.shape, len(layout[0]), (layout[1] - 1) * layout[2])
    np.testing.assert_array_equal(got.astype(np.int64), expected)
    np.testing.assert_array_equal(hasher(ids), got)  # __call__ is the same function
    for row in range(len(layout[0])):
        assert int(got[:, :, row].max()) < hasher.table_rows[row]
        assert int(got[:, :, row].min()) >= 0


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_dead_mask_matches_reference(ref, tokenizer, token_map, layout, seed):
    _, ref_state, hasher = _make_pair(ref, tokenizer, token_map, layout)
    ids, dead = _random_batch(seed, len(_WORDS), dead_p=0.3)
    dead[0, 0] = True  # a dead first position
    dead[-1] = False
    dead[-1, 3] = True  # a dead position in the middle of an otherwise live row
    expected = _ref_hash(ref_state, ids, 0, dead)
    got = hasher.hash(ids, dead)
    np.testing.assert_array_equal(got.astype(np.int64), expected)
    # the mask must matter: ignoring it gives different ids
    assert not np.array_equal(got, hasher.hash(ids))


def test_all_dead_row_matches_reference(ref, tokenizer, token_map):
    layout = _LAYOUTS[0]
    _, ref_state, hasher = _make_pair(ref, tokenizer, token_map, layout)
    ids = np.random.default_rng(7).integers(0, len(_WORDS), size=(2, 9))
    dead = np.zeros(ids.shape, dtype=bool)
    dead[1] = True
    expected = _ref_hash(ref_state, ids, 0, dead)
    got = hasher.hash(ids, dead)
    np.testing.assert_array_equal(got.astype(np.int64), expected)
    # a dead position hashes pad tokens only, so every dead position of row 1 has the same ids
    assert (got[1] == got[1, :1]).all()


def test_blocked_shift_blocks_all_later_shifts(token_map):
    """A dead token at shift s blanks shifts s..N-1 even if live tokens sit behind it."""
    hasher = EngramHasher(token_map, (1, 3), 4, 2, 1000, _PAD_ID)
    base = np.array([[5, 6, 7, 8, 9, 10, 11]])
    dead = np.zeros(base.shape, dtype=bool)
    dead[0, 3] = True
    other = base.copy()
    other[0, :3] = [20, 21, 22]  # change only what lies behind the dead token
    # position 5 sees shifts 0,1 live, shift 2 = position 3 dead -> shifts 2 and 3 are pad regardless of ids 0..2
    np.testing.assert_array_equal(hasher.hash(base, dead)[0, 5], hasher.hash(other, dead)[0, 5])
    # position 4 sees shift 1 dead -> only shift 0 is live: it depends only on its own token
    a, b = hasher.hash(base, dead)[0, 4], hasher.hash(other, dead)[0, 4]
    np.testing.assert_array_equal(a, b)


def test_first_position_matches_hand_computation(token_map):
    """Position 0 sees only its own token; the rest is pad. Recomputed with Python ints from independent inputs."""
    layer_ids, n, heads, vocab = (1, 3), 4, 2, 1000
    hasher = EngramHasher(token_map, layer_ids, n, heads, vocab, _PAD_ID)
    ids = np.array([[3, 11, 4]])
    got = hasher.hash(ids)

    compressed = int(token_map.max()) + 1
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    primes = engram_bucket_primes(len(layer_ids), n, heads, vocab)
    pad = int(token_map[_PAD_ID])
    first = int(token_map[ids[0, 0]])
    for row, layer_id in enumerate(layer_ids):
        mult = [
            int(v) * 2 + 1 for v in np.random.default_rng(10007 * layer_id).integers(0, bound, size=(n,), dtype=np.int64)
        ]
        flat = [p for per_ngram in primes[row] for p in per_ngram]
        rolling = first * mult[0]
        columns = []
        for i in range(1, n):
            rolling ^= pad * mult[i]
            columns.extend(rolling % p for p in primes[row][i - 1])
        offsets = [sum(flat[:j]) for j in range(len(flat))]
        expected = [c + o for c, o in zip(columns, offsets, strict=True)]
        assert got[0, 0, row].tolist() == expected


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("with_dead", [False, True])
@pytest.mark.parametrize("seed", [20, 21, 22])
def test_prefill_then_incremental_equals_full_and_reference(ref, tokenizer, token_map, layout, with_dead, seed):
    _, ref_state, hasher = _make_pair(ref, tokenizer, token_map, layout)
    ids, dead = _random_batch(seed, len(_WORDS), dead_p=0.25 if with_dead else 0.0)
    batch, length = ids.shape
    prefill = max(1, length // 2)

    full = hasher.hash(ids, dead)

    state = hasher.init_state(batch)
    assert state.position == 0
    chunk, state = hasher.step(ids[:, :prefill], state, None if dead is None else dead[:, :prefill])
    assert state.position == prefill
    pieces = [chunk]
    for t in range(prefill, length):
        chunk, new_state = hasher.step(ids[:, t : t + 1], state, None if dead is None else dead[:, t : t + 1])
        assert state.position == t  # the consumed state is not mutated
        state = new_state
        pieces.append(chunk)
    assert state.position == length
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), full)

    # the reference run through the same prefill + decode split gives the same ids
    ref_pieces = [_ref_hash(ref_state, ids[:, :prefill], 0, None if dead is None else dead[:, :prefill])]
    for t in range(prefill, length):
        ref_pieces.append(_ref_hash(ref_state, ids[:, t : t + 1], t, None if dead is None else dead[:, t : t + 1]))
    np.testing.assert_array_equal(np.concatenate(ref_pieces, axis=1), full.astype(np.int64))


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_uneven_chunking_equals_full(token_map, seed):
    layout = _LAYOUTS[0]
    hasher = EngramHasher(token_map, layout[0], layout[1], layout[2], layout[3], _PAD_ID)
    ids, dead = _random_batch(seed, len(_WORDS), dead_p=0.2)
    batch, length = ids.shape
    rng = np.random.default_rng(seed)
    state = hasher.init_state(batch)
    pieces, start = [], 0
    while start < length:
        size = int(rng.integers(1, 6))  # includes chunks shorter than the lookback
        chunk, state = hasher.step(ids[:, start : start + size], state, dead[:, start : start + size])
        pieces.append(chunk)
        start += size
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), hasher.hash(ids, dead))


def test_history_state_layout(token_map):
    hasher = EngramHasher(token_map, (1, 3), 4, 2, 1000, _PAD_ID)
    state = hasher.init_state(2)
    assert state.history.shape == (2, 3)
    assert (state.history == DEAD).all()
    ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    dead = np.zeros(ids.shape, dtype=bool)
    dead[0, 4] = True
    _, state = hasher.step(ids, state, dead)
    np.testing.assert_array_equal(state.history[1], token_map[[8, 9, 10]])
    np.testing.assert_array_equal(state.history[0], [token_map[3], token_map[4], DEAD])


def test_from_config_matches_explicit_construction(token_map):
    compressed = int(token_map.max()) + 1
    config = DeepseekV41Config(
        engram_layer_ids=[1, 3],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_vocab_size=1000,
        engram_pad_token_id=_PAD_ID,
        engram_compressed_vocab_size=compressed,
        engram_head_dim=8,
    )
    hasher = EngramHasher.from_config(config, token_map)
    explicit = EngramHasher(token_map, (1, 3), 4, 2, 1000, _PAD_ID)
    ids = np.random.default_rng(3).integers(0, len(_WORDS), size=(3, 12))
    np.testing.assert_array_equal(hasher.hash(ids), explicit.hash(ids))
    assert list(hasher.table_rows) == config.engram_table_rows()

    config.engram_compressed_vocab_size = compressed + 1
    with pytest.raises(ValueError, match="compressed vocab size"):
        EngramHasher.from_config(config, token_map)
    config.engram_compressed_vocab_size = compressed
    config.engram_num_embeddings = [1, 2]
    with pytest.raises(ValueError, match="disagrees"):
        EngramHasher.from_config(config, token_map)


def test_input_validation(token_map):
    hasher = EngramHasher(token_map, (1, 3), 4, 2, 1000, _PAD_ID)
    with pytest.raises(ValueError, match="input_ids"):
        hasher.hash(np.array([len(_WORDS)]))
    with pytest.raises(ValueError, match="input_ids"):
        hasher.hash(np.array([[len(_WORDS)]]))
    with pytest.raises(ValueError, match="dead_mask"):
        hasher.hash(np.array([[1, 2]]), np.array([[True]]))
    with pytest.raises(ValueError, match="state history"):
        hasher.step(np.array([[1, 2]]), hasher.init_state(2))
    with pytest.raises(ValueError, match="state position"):
        bad = EngramHashState(history=hasher.init_state(2).history, position=np.zeros(3, np.int64))
        hasher.step(np.array([[1, 2], [3, 4]]), bad)


@pytest.mark.parametrize("seed", [40, 41, 42])
def test_batched_windows_equal_single_windows_and_full_hash(token_map, seed):
    """``hash_windows`` (one step per window length, per-row start positions) is bit-identical to hashing each window
    alone, and both equal the stateless full-sequence hash sliced to the window."""
    layout = _LAYOUTS[0]
    hasher = EngramHasher(token_map, layout[0], layout[1], layout[2], layout[3], _PAD_ID)
    windower = EngramWindowHasher(hasher)
    rng = np.random.default_rng(seed)
    windows, expected = [], []
    for _ in range(12):
        length = int(rng.integers(1, 20))
        row = rng.integers(0, len(_WORDS), size=(length,))
        types = None
        if rng.random() < 0.5:  # an image span somewhere in the prompt (dead positions)
            types = np.full((length,), -1, dtype=np.int64)
            a = int(rng.integers(0, length))
            types[a : a + int(rng.integers(1, 4))] = 0
        # decode-shaped (1 token) windows dominate, plus chunked-prefill and from-zero windows
        start = int(rng.integers(0, length))
        end = start + 1 if rng.random() < 0.7 else int(rng.integers(start, length + 1))
        windows.append((row, types, start, end))
        dead = None if types is None else (types >= 0)[None]
        expected.append(hasher.hash(row[None], dead)[0, start:end])
    batched = windower.hash_windows(windows)
    for (row, types, start, end), got, full in zip(windows, batched, expected, strict=True):
        np.testing.assert_array_equal(got, windower.hash_window(row, types, start, end))
        np.testing.assert_array_equal(got, full)
        assert got.dtype == np.int32 and got.shape == (end - start, hasher.n_layers, hasher.n_columns)
