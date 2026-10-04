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

"""Host-side n-gram hashing for DeepSeek-V4.1 engram layers (pure Python + NumPy).

This is a bit-exact port of ``EngramLayout`` / ``NgramHashState`` / ``build_compressed_token_map`` from the official
reference ``engram.py``. The hash ids are computed on the host in int64 (awkward on TPU) and then index a per-layer
table of ``sum(primes)`` rows. No JAX is imported here.

Two pieces:

* :func:`build_engram_token_map` maps every tokenizer id onto a compressed id space in which tokens that normalise
  alike (NFKC, NFD, strip accents, lowercase, whitespace collapse) share an id.
* :class:`EngramHasher` turns ``[B, L]`` token ids into ``[B, L, n_engram_layers, (max_ngram_size - 1) * n_heads]``
  int32 hash ids, statelessly (:meth:`EngramHasher.hash`) or incrementally with an explicit
  :class:`EngramHashState` (:meth:`EngramHasher.step`).
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np

from .deepseek_v41_configuration import engram_bucket_primes

if tp.TYPE_CHECKING:
    from .deepseek_v41_configuration import DeepseekV41Config

#: History marker of a position that takes no part in any n-gram (reference ``NgramHashState.DEAD``).
DEAD = -1

# A private-use character so that a token that is exactly one space survives ``Strip()`` instead of collapsing to
# the empty string and merging with unrelated tokens (reference ``build_compressed_token_map``).
_SPACE_SENTINEL = "\ue000"


def _engram_normalizer() -> tp.Any:
    """Build the reference text normaliser (NFKC, NFD, strip accents, lowercase, whitespace collapse)."""
    from tokenizers import Regex, normalizers

    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), _SPACE_SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(_SPACE_SENTINEL, " "),
        ]
    )


def build_engram_token_map(tokenizer: tp.Any) -> np.ndarray:
    """Map every token id onto the compressed id space used for n-gram hashing.

    Reproduces ``build_compressed_token_map`` of the reference ``engram.py`` exactly: each token is decoded on its
    own with the raw Rust tokenizer (``backend_tokenizer.decode([id], skip_special_tokens=False)``), normalised, and
    keyed by the normalised text (or by the raw token string for a partial UTF-8 byte token that decodes to U+FFFD,
    or by the decoded text when normalisation empties it). Compressed ids are handed out in first-seen order.

    Args:
        tokenizer: A Hugging Face fast tokenizer: ``len(tokenizer)`` is the vocab size including added tokens and
            ``tokenizer.backend_tokenizer`` is the underlying ``tokenizers.Tokenizer``.

    Returns:
        ``int64[len(tokenizer)]`` compressed ids. The compressed vocab size (which every hash multiplier is derived
        from) is ``token_map.max() + 1``.
    """
    normalizer = _engram_normalizer()
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    vocab = len(tokenizer)
    lookup = np.zeros((vocab,), dtype=np.int64)
    for token_id in range(vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalise, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup


@dataclasses.dataclass(frozen=True)
class EngramHashState:
    """Incremental-decode state of :class:`EngramHasher` (what the reference keeps in its ``cache`` buffer).

    Attributes:
        history: ``int64[B, max_ngram_size - 1]`` compressed ids of the last ``max_ngram_size - 1`` positions seen
            (oldest first). A dead position (image span) is stored as :data:`DEAD`; positions before the start of the
            sequence are also :data:`DEAD`.
        position: Number of positions consumed so far (the reference ``start_pos`` of the next call).
    """

    history: np.ndarray
    position: int


class EngramHasher:
    """Maps each position to the hash ids of the n-grams ending there (reference ``NgramHashState``).

    Ids go through the compressed token map, then each position is hashed with the ``max_ngram_size - 1`` tokens
    before it. Look-back stops at the start of the sequence and at any dead token, so an n-gram never spans one: once
    a shift is blocked, every later shift is blocked too, and a blocked slot reads ``pad_id``. The running XOR of the
    per-lookback ``token * multiplier`` products after step ``i`` is the hash of the ``(i + 1)``-gram; it is reduced
    modulo that (n-gram size, head) prime and shifted by the cumulative prime sizes so the ids index directly into a
    per-layer table of ``sum(primes)`` rows.

    Attributes:
        token_map: ``int64[vocab]`` compressed token map.
        layer_ids: Global engram layer ids (multipliers are seeded with these, not with the engram-layer index).
        max_ngram_size: Largest n-gram size.
        n_heads: Hash heads per n-gram size.
        pad_id: Compressed id that fills slots with no history (``token_map[pad_token_id]``).
        primes: ``[layer][ngram_size - 2][head]`` bucket moduli.
        table_rows: Rows of each layer's table (sum of its primes).
        offsets: ``int64[n_layers, n_columns]`` cumulative prime offsets added to the hashes.
        multipliers: ``int64[n_layers, max_ngram_size]`` odd per-layer, per-lookback hash multipliers.
    """

    def __init__(
        self,
        token_map: np.ndarray,
        layer_ids: tp.Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        vocab_size: int,
        pad_token_id: int,
        *,
        compressed_vocab_size: int | None = None,
    ):
        """Build a hasher.

        Args:
            token_map: ``[vocab]`` compressed ids from :func:`build_engram_token_map`.
            layer_ids: Global layer ids carrying an engram (``engram_layer_ids``).
            max_ngram_size: Largest n-gram size (``engram_max_ngram_size``), at least 2.
            n_heads: Hash heads per n-gram size (``engram_n_heads``).
            vocab_size: ``engram_vocab_size``; primes are drawn upward from ``vocab_size - 1``.
            pad_token_id: Raw token id whose compressed id fills empty history slots (``engram_pad_token_id``).
            compressed_vocab_size: Size of the compressed vocab that every multiplier bound derives from. Defaults to
                ``token_map.max() + 1`` (the reference uses the number of distinct keys, which is the same).

        Raises:
            ValueError: On an empty or invalid layout, an invalid token map, or an out-of-range pad id.
        """
        token_map = np.asarray(token_map)
        if token_map.ndim != 1 or token_map.size == 0 or not np.issubdtype(token_map.dtype, np.integer):
            raise ValueError("token_map must be a non-empty 1-D integer array.")
        if int(token_map.min()) < 0:
            raise ValueError("token_map must be non-negative.")
        self.token_map = token_map.astype(np.int64)
        self.layer_ids = tuple(int(i) for i in layer_ids)
        if not self.layer_ids:
            raise ValueError("layer_ids must not be empty.")
        if max_ngram_size < 2:
            raise ValueError(f"max_ngram_size must be at least 2; got {max_ngram_size}.")
        if n_heads < 1:
            raise ValueError(f"n_heads must be at least 1; got {n_heads}.")
        if not 0 <= pad_token_id < self.token_map.size:
            raise ValueError(f"pad_token_id {pad_token_id} is outside the token map (size {self.token_map.size}).")
        derived_size = int(self.token_map.max()) + 1
        if compressed_vocab_size is None:
            compressed_vocab_size = derived_size
        if compressed_vocab_size < derived_size:
            raise ValueError(
                f"compressed_vocab_size {compressed_vocab_size} is below token_map.max() + 1 = {derived_size}."
            )
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.pad_id = int(self.token_map[pad_token_id])

        self.primes = engram_bucket_primes(len(self.layer_ids), self.max_ngram_size, self.n_heads, int(vocab_size))
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in self.primes]
        self.table_rows = tuple(sum(sizes) for sizes in flat)
        self.offsets = np.asarray([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
        self._primes = np.asarray(self.primes, dtype=np.int64)  # [layers, ngram sizes, heads]

        # one multiplier per (layer, lookback) from a per-layer RNG seeded with the GLOBAL layer id; odd, and bounded
        # so that ``token_id * multiplier`` cannot overflow int64
        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // self.compressed_vocab_size) // 2)
        rows = []
        for layer_id in self.layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64)
            rows.append(values * 2 + 1)
        self.multipliers = np.stack(rows)

    @classmethod
    def from_config(cls, config: DeepseekV41Config, token_map: np.ndarray) -> EngramHasher:
        """Build a hasher from a :class:`DeepseekV41Config` and a compressed token map.

        Args:
            config: Config providing the ``engram_*`` fields.
            token_map: Compressed token map from :func:`build_engram_token_map`.

        Returns:
            The configured hasher.

        Raises:
            ValueError: If the map's compressed vocab size differs from ``config.engram_compressed_vocab_size`` (every
                multiplier derives from it, so a mismatch would silently rehash the whole table), or if
                ``config.engram_num_embeddings`` disagrees with the derived prime layout.
        """
        token_map = np.asarray(token_map)
        size = int(token_map.max()) + 1
        if size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"token map has compressed vocab size {size}, config.engram_compressed_vocab_size is "
                f"{config.engram_compressed_vocab_size}."
            )
        config.engram_table_rows()  # validates engram_num_embeddings against the prime layout
        return cls(
            token_map,
            config.engram_layer_ids,
            config.engram_max_ngram_size,
            config.engram_n_heads,
            config.engram_vocab_size,
            config.engram_pad_token_id,
            compressed_vocab_size=size,
        )

    @property
    def n_layers(self) -> int:
        """Number of engram layers."""
        return len(self.layer_ids)

    @property
    def n_columns(self) -> int:
        """Hash columns per (position, layer): ``(max_ngram_size - 1) * n_heads``."""
        return (self.max_ngram_size - 1) * self.n_heads

    def init_state(self, batch_size: int) -> EngramHashState:
        """Fresh incremental state for a new batch of sequences (nothing consumed yet).

        Args:
            batch_size: Number of sequences.

        Returns:
            A state with an all-dead history and ``position == 0``.
        """
        history = np.full((int(batch_size), self.max_ngram_size - 1), DEAD, dtype=np.int64)
        return EngramHashState(history=history, position=0)

    def step(
        self,
        input_ids: np.ndarray,
        state: EngramHashState,
        dead_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, EngramHashState]:
        """Hash the next chunk of tokens, continuing from ``state`` (reference ``start_pos`` path).

        Hashing a sequence in any number of consecutive chunks gives the same ids as hashing it at once. All chunks of
        one sequence must keep the same batch size and rows; ``state`` is never mutated.

        Args:
            input_ids: ``[B, L]`` integer raw token ids of the next ``L`` positions.
            state: State returned by :meth:`init_state` or a previous :meth:`step`.
            dead_mask: Optional ``[B, L]`` bool, True for positions that take no part in any n-gram (image spans).
                Note this is the inverse of the reference ``token_mask`` (False there means dead).

        Returns:
            ``(hash_ids, new_state)`` where ``hash_ids`` is ``int32[B, L, n_layers, n_columns]``.

        Raises:
            ValueError: On malformed ids / mask / state or an id outside the token map.
            AssertionError: If an id would not fit int32 (the table would exceed 2**31 rows).
        """
        ids = np.asarray(input_ids)
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"input_ids must be an integer [B, L] array; got shape {ids.shape} dtype {ids.dtype}.")
        batch, seqlen = ids.shape
        lookback = self.max_ngram_size - 1
        if state.history.shape != (batch, lookback):
            raise ValueError(f"state history has shape {state.history.shape}; expected {(batch, lookback)}.")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.token_map.size):
            raise ValueError(f"input_ids must lie in [0, {self.token_map.size}).")

        compressed = self.token_map[ids]
        if dead_mask is not None:
            dead = np.asarray(dead_mask, dtype=bool)
            if dead.shape != ids.shape:
                raise ValueError(f"dead_mask shape {dead.shape} does not match input_ids {ids.shape}.")
            compressed = np.where(dead, DEAD, compressed)

        # window[:, lookback + j] is the new position j; shift s reads window[:, lookback + j - s]
        window = np.concatenate([state.history, compressed], axis=1)
        positions = state.position + np.arange(seqlen)
        blocked = np.zeros((batch, seqlen), dtype=bool)
        columns = []
        for shift in range(self.max_ngram_size):
            source = window[:, lookback - shift : lookback - shift + seqlen]
            blocked = blocked | (positions < shift)[None, :] | (source == DEAD)
            columns.append(np.where(blocked, self.pad_id, source))
        tokens = np.stack(columns, axis=-1)  # [B, L, max_ngram_size]

        # XOR the multiplied ids together one lookback at a time: the running value after step i is the hash of the
        # (i + 1)-gram; each lands in its own prime-sized bucket range
        products = tokens[:, :, None, :] * self.multipliers[None, None]  # [B, L, n_layers, max_ngram_size]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, self.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self._primes[:, i - 1])
        out = np.concatenate(hashes, axis=-1) + self.offsets

        if out.size:
            assert int(out.min()) >= 0 and int(out.max()) <= np.iinfo(np.int32).max, "engram hash ids exceed int32"
        new_state = EngramHashState(
            history=window[:, window.shape[1] - lookback :].copy(), position=state.position + seqlen
        )
        return out.astype(np.int32), new_state

    def hash(self, input_ids: np.ndarray, dead_mask: np.ndarray | None = None) -> np.ndarray:
        """Hash whole sequences from position 0 (the reference ``start_pos == 0`` call).

        Args:
            input_ids: ``[B, L]`` integer raw token ids.
            dead_mask: Optional ``[B, L]`` bool, True for dead positions (image spans); inverse of the reference
                ``token_mask``.

        Returns:
            ``int32[B, L, n_layers, (max_ngram_size - 1) * n_heads]`` ids indexing a per-layer table of
            ``table_rows[layer]`` rows.
        """
        ids = np.asarray(input_ids)
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L]; got shape {ids.shape}.")
        return self.step(ids, self.init_state(ids.shape[0]), dead_mask)[0]

    def __call__(self, input_ids: np.ndarray, dead_mask: np.ndarray | None = None) -> np.ndarray:
        """Alias of :meth:`hash`."""
        return self.hash(input_ids, dead_mask)


__all__ = ["DEAD", "EngramHashState", "EngramHasher", "build_engram_token_map"]
