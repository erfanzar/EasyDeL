# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Hashed n-gram embeddings (Qwen4 / Qwen3.8-Flash-Next PLE).

Each token is described not only by its own id but by the n-grams ending at it.
Those n-grams are hashed into a large table: for every n in ``2..ngram_size``
and every one of ``heads_per_ngram`` heads, the shifted token ids are multiplied
by odd 64-bit multipliers, XOR-folded, and reduced modulo a head-specific prime.
The per-head lookups are concatenated to form the token's n-gram vector.

Two properties drive the implementation:

**The hash needs true 64-bit width.** Multipliers are drawn just below
``2**63 / vocab_size`` (~3.7e13 for a 248k vocab), so ``token * multiplier``
reaches ~9.2e18 against an ``int64`` ceiling of 9.22e18. JAX runs with
``jax_enable_x64`` disabled and silently truncates to ``int32``, where those
products wrap and every n-gram resolves to the wrong row -- with nothing
raising. The arithmetic therefore goes through :mod:`._u64`, which emulates the
width on ``uint32`` limb pairs.

**The table is enormous.** ``ngram_heads`` primes of ~20,000,000 rows each at
``embedding_dim // ngram_heads`` columns is ~51B parameters -- larger than the
rest of the model. The reference checkpoint stores it split into
``split_ngram_parts`` shards and pre-quantized; this layer keeps the same split
so the shards map onto the checkpoint directly and can be sharded along the
vocabulary axis.

The multipliers, per-head vocabulary sizes and offsets are deterministic
functions of ``(vocab_size, ngram_size, ple_layer_index, seed)``, so they are
recomputed here rather than restored, and are asserted against the checkpoint's
own buffers by the tests.
"""

from __future__ import annotations

import math
import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Int

from ._u64 import mul_by_scalar, u64_mod_small, u64_xor

# Constants are taken verbatim from the reference implementation; ``_PRIME_1``
# is a small prime (10007), NOT the SplitMix gamma -- it only shifts the seed
# per PLE layer, so getting it wrong silently mis-hashes every layer past the
# first rather than failing.
_MASK64 = (1 << 64) - 1
_PRIME_1 = 10007
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB

__all__ = ("NGramEmbed", "build_layer_multipliers", "find_nth_prime_after")


def _splitmix64(value: int) -> int:
    """One round of the SplitMix64 finaliser."""
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """Return the ``count``-th prime strictly greater than ``start``."""
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    """Derive the odd 64-bit multipliers mixed into the n-gram hash.

    Args:
        unigram_vocab_size: Token vocabulary size; bounds the multiplier so the
            product stays inside 63 bits.
        ngram_size: Widest n-gram; one multiplier per position.
        ple_layer_index: Index of this PLE layer among all PLE layers.
        seed: Model-level hash seed.

    Returns:
        ``ngram_size`` odd multipliers below ``2**63 / unigram_vocab_size``.
    """
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return multipliers


def _identity_split(tensor: tp.Any) -> tp.Any:
    """Pass a checkpoint tensor through unchanged (layout already matches)."""
    return tensor


def _vocab_shard_splitter(index: int, rows: int) -> tp.Callable[[tp.Any], tp.Any]:
    """Build a loader slice for one vocab-axis shard of a flat table.

    Args:
        index: Shard ordinal along the vocabulary axis.
        rows: Rows per shard.

    Returns:
        Callable slicing ``tensor[index * rows : (index + 1) * rows]``.
    """

    def split(tensor: tp.Any) -> tp.Any:
        return tensor[index * rows : (index + 1) * rows]

    return split


class NGramEmbed(spx.Module):
    """Hashed n-gram lookup producing one dense vector per token.

    Attributes:
        shards: ``split_ngram_parts`` slices of the embedding table, matching
            the checkpoint's ``ngram_embedding.shard_{i}.weight`` layout.
        head_vocab_sizes: Per-head prime moduli.
        head_offsets: Per-head base row into the concatenated table.
        layer_multipliers: Odd multipliers folded into the hash.
    """

    def __init__(
        self,
        config: tp.Any,
        embedding_dim: int,
        ple_layer_index: int = 0,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: spx.Rngs,
    ) -> None:
        """Build the table and the deterministic hash constants.

        Args:
            config: Provides ``ngram_size``, ``heads_per_ngram``, ``vocab_size``,
                ``ngram_vocab_size_base``, ``make_ngram_vocab_size_divisible_by``,
                ``split_ngram_parts``, ``seed`` and ``eos_token_id``.
            embedding_dim: Width of the concatenated per-token output; must
                divide evenly by the number of hash heads.
            ple_layer_index: Which PLE layer this is; shifts the head indices so
                different layers hash to different primes.
            dtype: Activation dtype.
            param_dtype: Table storage dtype.
            rngs: Random number generators.
        """
        self.ngram_size = int(config.ngram_size)
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.dtype = dtype
        self.param_dtype = param_dtype

        eos = config.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        if embedding_dim % self.ngram_heads:
            raise ValueError(f"embedding_dim ({embedding_dim}) must divide by ngram_heads ({self.ngram_heads}).")
        self.head_dim = embedding_dim // self.ngram_heads

        base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = find_nth_prime_after(base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size

        divisor = int(config.make_ngram_vocab_size_divisible_by)
        self.total_vocab_size = total
        self.padded_vocab_size = math.ceil(total / divisor) * divisor

        # Stored as host NumPy on purpose: jnp arrays here would bind as
        # tracers during spectrax lazy init and leak when the trace closes.
        import numpy as np

        self.head_vocab_sizes = np.asarray(sizes, np.uint32)
        self.head_offsets = np.asarray(offsets, np.uint32)
        self.layer_multipliers = build_layer_multipliers(
            int(config.vocab_size), self.ngram_size, self.ple_layer_index, int(config.seed)
        )

        # Kept split so the shards line up with the checkpoint and so the
        # vocabulary axis can be sharded without reshaping a 51B-parameter
        # tensor as one piece.
        self.num_shards = int(config.split_ngram_parts)
        rows = math.ceil(self.padded_vocab_size / self.num_shards)
        self.shard_rows = rows
        # Deferred import: easydel.infra.utils imports the layers package at
        # module level, so a top-level import here would be circular.
        from easydel.infra.utils import ArrayParam

        keys = jax.random.split(rngs.parameters, self.num_shards)
        feature_sharding_axis = getattr(config, "ngram_sharding_axis", "tp")
        # ParameterList, not a bare python list: plain-list attributes are
        # treated as opaque graph structure, so tracer-valued Parameters
        # stored in one leak out of the lazy-init trace. The container
        # registers every shard as a real state leaf (path ``shards/{i}``).
        self.shards = spx.nn.ParameterList(
            [
                ArrayParam.bound(
                    shape=(rows, self.head_dim),
                    dtype=param_dtype,
                    init_method="normal",
                    init_kwargs={"stddev": 0.02},
                    # Feature-axis tensor parallelism, matching the vocab
                    # embedding contract: every rank owns all rows of its
                    # head_dim slice, so :meth:`lookup`'s plain ``jnp.take``
                    # stays communication-free and the gathered output is
                    # feature-sharded for the PLE projection to consume.
                    sharding=(None, feature_sharding_axis),
                    key=keys[i],
                )
                for i in range(self.num_shards)
            ]
        )
        # Dequantization scale for narrow (fp8/int8) storage only: quantized
        # checkpoints carry one per-tensor scalar alongside the codes. Wide
        # (bf16/fp32) tables have no scale -- creating the parameter anyway
        # would leave it unmaterialized on the merge-only load path and change
        # the tree shape for every existing bf16 checkpoint.
        self.narrow_storage = jnp.dtype(param_dtype).itemsize == 1
        if self.narrow_storage:
            self.weight_scale = ArrayParam.bound(
                shape=(1,),
                dtype=jnp.bfloat16,
                init_method="ones",
                key=rngs.parameters,
            )

    @property
    def reform_param(self) -> dict[str, dict[str, tp.Any]]:
        """Checkpoint reform rules mapping published n-gram tables onto shards.

        Two published layouts exist and both are accepted on load:

        * the Qwen3.8-Flash-Next release checkpoint splits the table along
          the vocabulary axis into ``split_ngram_parts`` tensors named
          ``ngram_embedding.shard_{i}.weight``;
        * the transformers reference implementation stores one flat
          ``ngram_embedding.weight`` ``[padded_vocab, head_dim]`` table.

        Rules are keyed on local names; the EasyDeL collector prefixes them
        with the owning module's path. On export the sharded rules run first
        (and consume the shard leaves), so EasyDeL models always re-export in
        the release checkpoint's sharded layout. Quantized fp8 checkpoints
        additionally carry ``ngram_embedding.weight_scale``, which belongs to
        the quantization pipeline rather than to these rules.
        """
        rules: dict[str, dict[str, tp.Any]] = {}
        for i in range(self.num_shards):
            rules[f"ngram_embedding.shard_{i}.weight$"] = {
                "splits": [{"name": f"shards.{i}", "spliter": _identity_split}],
                "inverse_spliter": lambda shard: shard,
                # fp8-stored shards must survive the loader param-dtype cast.
                "preserve_dtype": True,
            }
        if self.narrow_storage:
            rules["ngram_embedding.weight_scale$"] = {
                "splits": [{"name": "weight_scale", "spliter": _identity_split}],
                "inverse_spliter": lambda scale: scale,
                "preserve_dtype": True,
            }
        rules["ngram_embedding.weight$"] = {
            "splits": [
                {"name": f"shards.{i}", "spliter": _vocab_shard_splitter(i, self.shard_rows)}
                for i in range(self.num_shards)
            ],
            "inverse_spliter": lambda torch, *shards: torch.cat(list(shards), dim=0),
        }
        return rules

    def _shift_right_ignore_eos(
        self,
        token_ids: Int[Array, "batch seq"],
        shift: int,
        segment_ids: Int[Array, "batch seq"] | None = None,
    ) -> Int[Array, "batch seq"]:
        """Shift right by ``shift``, never reading across an EOS boundary.

        Positions whose source would fall in a previous document are replaced by
        the EOS id, so n-grams never straddle two sequences.
        """
        if shift == 0:
            return token_ids
        batch, seq_len = token_ids.shape
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        source_positions = positions - shift
        gather = jnp.clip(source_positions, 0, None)[None, :]
        shifted = jnp.take_along_axis(token_ids, jnp.broadcast_to(gather, token_ids.shape), axis=1)
        if segment_ids is not None:
            source_segments = jnp.take_along_axis(segment_ids, jnp.broadcast_to(gather, segment_ids.shape), axis=1)
            valid = (source_positions[None, :] >= 0) & (segment_ids >= 0) & (source_segments == segment_ids)
            return jnp.where(valid, shifted, jnp.full_like(token_ids, self.eos_token_id))

        eos_positions = jnp.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = jax.lax.cummax(eos_positions, axis=1)
        previous_eos = jnp.concatenate(
            [jnp.full((batch, 1), -1, eos_positions.dtype), previous_eos_inclusive[:, :-1]], axis=1
        )
        position_in_segment = positions[None, :] - (previous_eos + 1)
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return jnp.where(valid, shifted, jnp.full_like(token_ids, self.eos_token_id))

    def hash_ids(
        self,
        token_history: Int[Array, "batch total"],
        segment_ids: Int[Array, "batch total"] | None = None,
    ) -> Int[Array, "batch total heads"]:
        """Hash a token history into per-head table rows.

        Args:
            token_history: Token ids including the carried context.

        Returns:
            Row indices ``[batch, total, ngram_heads]`` into the flat table.
        """
        shifted = [
            self._shift_right_ignore_eos(token_history, shift, segment_ids=segment_ids).astype(jnp.uint32)
            for shift in range(self.ngram_size)
        ]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            stop = start + self.heads_per_ngram

            mixed = mul_by_scalar(shifted[0], self.layer_multipliers[0])
            for position in range(1, ngram):
                mixed = u64_xor(mixed, mul_by_scalar(shifted[position], self.layer_multipliers[position]))

            sizes = self.head_vocab_sizes[start:stop]
            offsets = self.head_offsets[start:stop]
            # Broadcast the head axis in: [B, T, 1] against [heads]
            wide = (mixed[0][..., None], mixed[1][..., None])
            rows = u64_mod_small(wide, jnp.broadcast_to(sizes, (*token_history.shape, stop - start)))
            blocks.append(rows + offsets)

        return jnp.concatenate(blocks, axis=-1)

    def lookup(self, rows: Int[Array, "batch seq heads"]) -> Array:
        """Gather from the sharded table and concatenate the heads.

        Args:
            rows: Flat row indices produced by :meth:`hash_ids`.

        Returns:
            ``[batch, seq, ngram_heads * head_dim]``.
        """
        rows = rows.astype(jnp.int32)
        shard_idx = rows // self.shard_rows
        within = rows % self.shard_rows

        out = jnp.zeros((*rows.shape, self.head_dim), self.dtype)
        scale = self.weight_scale.value.astype(self.dtype) if self.narrow_storage else 1.0
        for i, shard in enumerate(self.shards):
            table = shard.value
            take_idx = jnp.clip(within, 0, self.shard_rows - 1)
            belongs_to_shard = shard_idx == i

            def gather_present_rows(current, table=table, take_idx=take_idx, belongs_to_shard=belongs_to_shard):
                if jnp.dtype(table.dtype).itemsize == 1:
                    # fp8 (or int8) storage: gather through a same-width
                    # integer view, then restore the storage dtype.
                    gathered = jnp.take(table.view(jnp.uint8), take_idx, axis=0).view(table.dtype)
                else:
                    gathered = jnp.take(table, take_idx, axis=0)
                gathered = gathered.astype(self.dtype) * scale
                return jnp.where(belongs_to_shard[..., None], gathered, current)

            # A decode step touches at most one shard per n-gram head, while
            # release checkpoints have 128 table shards.  Avoid issuing a
            # large gather for every absent shard.  Keeping the condition
            # scalar is important: vmapping it would lower back to eager
            # branch selects and execute all gathers.
            out = jax.lax.cond(jnp.any(belongs_to_shard), gather_present_rows, lambda current: current, out)
        return out.reshape(*rows.shape[:-1], self.ngram_heads * self.head_dim)

    def forward(
        self,
        input_ids: Int[Array, "batch seq"],
        context: Int[Array, "batch ctx"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        context_segment_ids: Int[Array, "batch ctx"] | None = None,
    ) -> Array:
        """Embed the n-grams ending at each position.

        Args:
            input_ids: Token ids ``[batch, seq]``.
            context: The previous ``ngram_size - 1`` tokens, carried across
                decode steps. ``None`` starts a fresh stream (EOS padding).

        Returns:
            ``[batch, seq, ngram_heads * head_dim]``.
        """
        if context is None:
            context = jnp.full((input_ids.shape[0], self.context_len), self.eos_token_id, input_ids.dtype)
        history = jnp.concatenate([context, input_ids], axis=-1)
        history_segments = None
        if segment_ids is not None:
            if context_segment_ids is None:
                context_segment_ids = jnp.full((input_ids.shape[0], self.context_len), -1, segment_ids.dtype)
            history_segments = jnp.concatenate([context_segment_ids, segment_ids], axis=-1)
        rows = self.hash_ids(history, segment_ids=history_segments)[:, -input_ids.shape[1] :]
        return self.lookup(rows)
