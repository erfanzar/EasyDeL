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

"""Host-side n-gram hashing for engram layers (pure Python + NumPy, no JAX).

Bit-exact with ``EngramLayout`` / ``NgramHashState`` / ``build_compressed_token_map`` of DeepSeek's reference
``engram.py``. Hash ids are int64 arithmetic (awkward on TPU), so they are computed on the host and index a per-layer
table of ``sum(primes)`` rows.

* :func:`engram_bucket_primes` / :func:`engram_table_rows`: the prime-bucket table layout.
* :func:`build_engram_token_map`: maps every tokenizer id onto a compressed id space in which tokens that normalise
  alike (NFKC, NFD, strip accents, lowercase, whitespace collapse) share an id.
* :class:`EngramHasher`: ``[B, L]`` token ids -> ``[B, L, n_engram_layers, (max_ngram_size - 1) * n_heads]`` int32
  hash ids, statelessly (:meth:`EngramHasher.hash`) or incrementally (:meth:`EngramHasher.step`).
* :class:`EngramWindowHasher`: hash ids of one request's scheduled window ``[start, end)``, rebuilt from its token
  history, so chunked prefill, decode and preemption hash exactly like one pass over the sequence.

A model describes its engram layout with these config fields: ``engram_layer_ids``, ``engram_max_ngram_size``,
``engram_n_heads``, ``engram_vocab_size``, ``engram_pad_token_id``, ``engram_compressed_vocab_size``, ``vocab_size``
and optionally ``engram_num_embeddings``.
"""

from __future__ import annotations

import dataclasses
import functools
import typing as tp

import numpy as np


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test (exact for ``n < 3.3e24``)."""
    if n < 2:
        return False
    witnesses = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in witnesses:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in witnesses:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@functools.lru_cache(maxsize=16)
def engram_bucket_primes(
    num_layers: int,
    max_ngram_size: int,
    n_heads: int,
    vocab_size: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Prime bucket moduli ``[layer][ngram_size - 2][head]`` of the engram hash tables.

    Mirrors ``EngramLayout.from_args`` in the reference ``engram.py``: for every (layer, n-gram size, head) the
    search starts again at ``vocab_size - 1`` and takes the next prime not handed out before, so bucket ranges
    never repeat across the whole model.

    Args:
        num_layers: Number of engram layers.
        max_ngram_size: Largest n-gram size (n-gram sizes ``2..max_ngram_size`` are hashed).
        n_heads: Hash heads per n-gram size.
        vocab_size: ``engram_vocab_size``; primes are drawn upward from ``vocab_size - 1``.

    Returns:
        Nested tuple of primes.
    """
    seen: set[int] = set()
    layers = []
    for _ in range(num_layers):
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        layers.append(tuple(per_ngram))
    return tuple(layers)


def engram_table_rows(config: tp.Any) -> list[int]:
    """Rows of each engram layer's table (the sum of its prime buckets) for a model config.

    Args:
        config: Config with the ``engram_*`` fields (see the module docstring).

    Returns:
        One row count per entry of ``config.engram_layer_ids``.

    Raises:
        ValueError: If ``config.engram_num_embeddings`` is set and disagrees with the derived layout (the hash offsets
            would then address the wrong rows).
    """
    primes = engram_bucket_primes(
        len(config.engram_layer_ids),
        int(config.engram_max_ngram_size),
        int(config.engram_n_heads),
        int(config.engram_vocab_size),
    )
    derived = [sum(p for per_ngram in layer for p in per_ngram) for layer in primes]
    declared = getattr(config, "engram_num_embeddings", None)
    if declared is not None and list(declared) != derived:
        raise ValueError(f"engram_num_embeddings {list(declared)} disagrees with the prime-bucket layout {derived}.")
    return derived


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
        position: Number of positions consumed so far (the reference ``start_pos`` of the next call): one int shared
            by every row, or ``int64[B]`` per row (rows of different requests hashed in one call).
    """

    history: np.ndarray
    position: int | np.ndarray


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
    def from_config(cls, config: tp.Any, token_map: np.ndarray | None = None) -> EngramHasher:
        """Build a hasher from a model config's ``engram_*`` fields and a compressed token map.

        Args:
            config: Config providing the ``engram_*`` fields (see the module docstring).
            token_map: Compressed token map from :func:`build_engram_token_map`. ``None`` uses the identity map,
                which is only valid when ``engram_compressed_vocab_size == vocab_size``.

        Returns:
            The configured hasher.

        Raises:
            ValueError: If no map is given and the identity map does not fit; if the map's compressed vocab size
                differs from ``config.engram_compressed_vocab_size`` (every multiplier derives from it, so a mismatch
                would silently rehash the whole table); or if ``config.engram_num_embeddings`` disagrees with the
                derived prime layout.
        """
        if token_map is None:
            vocab = int(config.vocab_size)
            if int(config.engram_compressed_vocab_size) != vocab:
                raise ValueError(
                    "This model hashes engram n-grams over the tokenizer's compressed vocabulary "
                    f"(engram_compressed_vocab_size={config.engram_compressed_vocab_size} != vocab_size={vocab}); "
                    "build it with easydel.layers.engram.build_engram_token_map(tokenizer)."
                )
            token_map = np.arange(vocab, dtype=np.int64)
        token_map = np.asarray(token_map)
        size = int(token_map.max()) + 1
        if size != int(config.engram_compressed_vocab_size):
            raise ValueError(
                f"token map has compressed vocab size {size}, config.engram_compressed_vocab_size is "
                f"{config.engram_compressed_vocab_size}."
            )
        engram_table_rows(config)  # validates engram_num_embeddings against the prime layout
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
        start = np.asarray(state.position, dtype=np.int64)
        if start.ndim > 1 or (start.ndim == 1 and start.shape != (batch,)):
            raise ValueError(f"state position must be a scalar or [{batch}]; got shape {start.shape}.")
        positions = start.reshape(-1, 1) + np.arange(seqlen)[None, :]  # [1 or B, L]
        blocked = np.zeros((batch, seqlen), dtype=bool)
        columns = []
        for shift in range(self.max_ngram_size):
            source = window[:, lookback - shift : lookback - shift + seqlen]
            blocked = blocked | (positions < shift) | (source == DEAD)
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


class EngramWindowHasher:
    """Hash ids of one request's scheduled token window, rebuilt from its token history.

    The :class:`EngramHasher` state is reconstructed from the last ``max_ngram_size - 1`` tokens before the window
    (image-span positions dead), which is exactly the state an incremental hasher would carry. Chunked prefill,
    decode, preemption/resume and speculative rollback therefore hash identically to one pass over the sequence, with
    no per-request hasher state to keep or reset.
    """

    def __init__(self, hasher: EngramHasher) -> None:
        """Wrap a configured hasher.

        Args:
            hasher: The hasher.
        """
        self.hasher = hasher
        self.lookback = int(hasher.max_ngram_size) - 1

    @classmethod
    def from_config(cls, config: tp.Any, token_map: np.ndarray | None = None) -> EngramWindowHasher:
        """Shorthand for ``EngramWindowHasher(EngramHasher.from_config(config, token_map))``."""
        return cls(EngramHasher.from_config(config, token_map))

    @property
    def n_layers(self) -> int:
        """Number of engram layers."""
        return int(self.hasher.n_layers)

    @property
    def n_columns(self) -> int:
        """Hash columns per (position, layer)."""
        return int(self.hasher.n_columns)

    def hash_window(
        self,
        token_ids_row: np.ndarray,
        token_types: np.ndarray | None,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Hash positions ``[start, end)`` of one request.

        Args:
            token_ids_row: The request's token ids (prompt followed by generated tokens); positions
                ``max(0, start - lookback) .. end`` must be real tokens.
            token_types: Prompt token types (``>= 0`` marks an image-span position, which is dead), or ``None`` for a
                text-only request. Positions past its end are text.
            start: First position of the window.
            end: One past the last position of the window.

        Returns:
            ``int32 [end - start, n_layers, n_columns]``.
        """
        history, ids, dead = self._window_inputs(token_ids_row, token_types, start, end)
        state = EngramHashState(history=history[None], position=int(start))
        out, _ = self.hasher.step(ids[None], state, dead[None])
        return out[0]

    def hash_windows(self, windows: tp.Sequence[tuple[np.ndarray, np.ndarray | None, int, int]]) -> list[np.ndarray]:
        """Hash many requests' windows, one :meth:`EngramHasher.step` call per distinct window length.

        Bit-identical to calling :meth:`hash_window` on each window; batching removes the per-call NumPy overhead
        that dominates decode (every window is one token long).

        Args:
            windows: ``(token_ids_row, token_types, start, end)`` per request, as for :meth:`hash_window`.

        Returns:
            ``int32 [end - start, n_layers, n_columns]`` per window, in input order (empty for empty windows).
        """
        out: list[np.ndarray | None] = [None] * len(windows)
        groups: dict[int, list[int]] = {}
        prepared = []
        for i, (row, types, start, end) in enumerate(windows):
            n = int(end) - int(start)
            if n <= 0:
                out[i] = np.zeros((0, self.n_layers, self.n_columns), dtype=np.int32)
                prepared.append(None)
                continue
            prepared.append(self._window_inputs(row, types, start, end))
            groups.setdefault(n, []).append(i)
        for members in groups.values():
            if len(members) == 1:  # no batching win; skip the stack/gather overhead (same computation as hash_window)
                i = members[0]
                history, ids, dead = prepared[i]
                state = EngramHashState(history=history[None], position=int(windows[i][2]))
                out[i] = self.hasher.step(ids[None], state, dead[None])[0][0]
                continue
            state = EngramHashState(
                history=np.stack([prepared[i][0] for i in members]),
                position=np.asarray([int(windows[i][2]) for i in members], dtype=np.int64),
            )
            ids = np.stack([prepared[i][1] for i in members])
            dead = np.stack([prepared[i][2] for i in members])
            hashed, _ = self.hasher.step(ids, state, dead)
            for slot, i in enumerate(members):
                out[i] = hashed[slot]
        return out  # type: ignore[return-value]

    def _window_inputs(
        self, token_ids_row: np.ndarray, token_types: np.ndarray | None, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(history [lookback], ids [end - start], dead [end - start])`` of one window."""
        lo = max(0, int(start) - self.lookback)
        # negative ids are speculative-draft placeholders; their rows are replaced downstream, hash them as id 0
        ids = np.maximum(np.asarray(token_ids_row[lo:end], dtype=np.int64), 0)
        dead = np.zeros(ids.shape, dtype=bool)
        if token_types is not None:
            hi = min(int(end), int(token_types.shape[0]))
            if hi > lo:
                dead[: hi - lo] = np.asarray(token_types[lo:hi]) >= 0
        history = np.full((self.lookback,), DEAD, dtype=np.int64)
        n_history = int(start) - lo
        if n_history:
            history[self.lookback - n_history :] = np.where(
                dead[:n_history], DEAD, self.hasher.token_map[ids[:n_history]]
            )
        return history, ids[n_history:], dead[n_history:]


__all__ = [
    "DEAD",
    "EngramHashState",
    "EngramHasher",
    "EngramWindowHasher",
    "build_engram_token_map",
    "engram_bucket_primes",
    "engram_table_rows",
]
