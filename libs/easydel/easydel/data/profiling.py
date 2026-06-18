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

"""Dataset profiling and health-check utilities for EasyData.

This module provides a lightweight, non-mutating way to inspect a
:class:`~easydel.data.core.protocols.ShardedDataSource` before launching a
training run. It reports sequence-length distributions, field presence,
truncation/padding rates, source mix, and estimated token-packing efficiency
for one or more packing strategies.

Example:
    >>> from easydel.data import profile_dataset, JsonShardedSource
    >>> source = JsonShardedSource("data/*.jsonl")
    >>> profile = profile_dataset(source, seq_length=2048, max_rows=5000)
    DatasetProfile(
      num_rows_sampled=5000,
      ...
    )
"""

from __future__ import annotations

import logging
import typing as tp
from collections import Counter
from dataclasses import dataclass

from .core.protocols import ShardedDataSource
from .transforms.pack import FirstFitPacker, GreedyPacker, PoolPacker

if tp.TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_LENGTH_FIELDS = ("input_ids", "attention_mask", "labels")


@dataclass
class DatasetProfile:
    """Aggregate health/efficiency report for a sampled data source.

    Attributes:
        num_rows_sampled: Number of upstream rows consumed by the profiler.
        field_counts: How many sampled rows contained each field name.
        length_histogram: Per-field mapping ``length -> row_count`` for the
            configured length fields (e.g. ``input_ids``).
        truncation_rate: Fraction of rows with an explicit truthy
            ``"truncated"`` field. ``0.0`` when the field is absent.
        padding_rate: Fraction of rows where ``attention_mask`` sum is less
            than ``len(input_ids)``. ``0.0`` when the mask is absent.
        source_distribution: Count of rows per ``"__source__"`` value.
        source_token_counts: Sum of ``len(input_ids)`` per
            ``"__source__"`` value.
        total_input_tokens: Sum of ``len(input_ids)`` across all sampled
            rows.
        length_percentiles: Per-field length percentiles
            (``p50``, ``p90``, ``p99``) for the configured length fields.
        chat_template_fallback_rate: Fraction of rows with an explicit
            chat-template fallback marker (``chat_template_applied=False``
            or ``template_fallback=True``). ``0.0`` when no markers are
            present.
        estimated_packing_efficiency: Mapping ``strategy -> efficiency`` for
            the requested ``seq_length``. Empty when ``seq_length`` is None.
        packing_efficiency_gap: Mapping of smarter-packer efficiency minus
            greedy efficiency (``pool_vs_greedy``, ``first_fit_vs_greedy``).
            Empty when ``seq_length`` is None.
        seq_length: Window length used for packing estimation, if any.
    """

    num_rows_sampled: int
    field_counts: dict[str, int]
    length_histogram: dict[str, dict[int, int]]
    length_percentiles: dict[str, dict[str, float]]
    truncation_rate: float
    padding_rate: float
    chat_template_fallback_rate: float
    source_distribution: dict[str, int]
    source_token_counts: dict[str, int]
    total_input_tokens: int
    estimated_packing_efficiency: dict[str, float]
    packing_efficiency_gap: dict[str, float]
    seq_length: int | None = None

    def to_dict(self) -> dict[str, tp.Any]:
        """Render the profile as a JSON-friendly dict."""
        return {
            "num_rows_sampled": self.num_rows_sampled,
            "field_counts": dict(self.field_counts),
            "length_histogram": {field: dict(sorted(hist.items())) for field, hist in self.length_histogram.items()},
            "truncation_rate": self.truncation_rate,
            "padding_rate": self.padding_rate,
            "chat_template_fallback_rate": self.chat_template_fallback_rate,
            "source_distribution": dict(self.source_distribution),
            "source_token_counts": dict(self.source_token_counts),
            "total_input_tokens": self.total_input_tokens,
            "length_percentiles": {field: dict(percs) for field, percs in self.length_percentiles.items()},
            "estimated_packing_efficiency": dict(self.estimated_packing_efficiency),
            "packing_efficiency_gap": dict(self.packing_efficiency_gap),
            "seq_length": self.seq_length,
        }

    def __str__(self) -> str:
        lines = [
            "DatasetProfile(",
            f"  num_rows_sampled={self.num_rows_sampled}",
            f"  field_counts={self.field_counts}",
            f"  length_histogram={self.length_histogram}",
            f"  truncation_rate={self.truncation_rate:.4f}",
            f"  padding_rate={self.padding_rate:.4f}",
            f"  chat_template_fallback_rate={self.chat_template_fallback_rate:.4f}",
            f"  source_distribution={self.source_distribution}",
            f"  source_token_counts={self.source_token_counts}",
            f"  total_input_tokens={self.total_input_tokens}",
            f"  length_percentiles={self.length_percentiles}",
            f"  estimated_packing_efficiency={self.estimated_packing_efficiency}",
            f"  packing_efficiency_gap={self.packing_efficiency_gap}",
            f"  seq_length={self.seq_length}",
            ")",
        ]
        return "\n".join(lines)


class DatasetProfiler:
    """Stream rows from a sharded source and build a :class:`DatasetProfile`.

    The profiler is intentionally lightweight: it consumes at most
    ``max_rows`` examples, collects simple counts, and optionally runs a
    fast in-memory packing simulation using the same packer classes that
    :class:`~easydel.data.transforms.pack.PackedShardedSource` uses.

    Args:
        max_rows: Maximum number of upstream rows to sample.
        seq_length: If provided, run greedy/pool/first-fit packing
            simulators on the sampled ``input_ids`` lengths and report
            estimated efficiencies.
        length_fields: Fields for which to collect length histograms.
            Defaults to ``("input_ids", "attention_mask", "labels")``.
        eos_token_id: EOS id passed to the packing simulators. The actual
            token value does not affect the efficiency estimate.
        pad_token_id: PAD id passed to the packing simulators.
    """

    def __init__(
        self,
        max_rows: int = 10_000,
        seq_length: int | None = None,
        length_fields: Sequence[str] | None = None,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
    ):
        self.max_rows = max_rows
        self.seq_length = seq_length
        self.length_fields = list(length_fields or DEFAULT_LENGTH_FIELDS)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def profile(self, source: ShardedDataSource) -> DatasetProfile:
        """Collect a profile for ``source`` without mutating it."""
        field_counts: Counter[str] = Counter()
        length_histogram: dict[str, Counter] = {field: Counter() for field in self.length_fields}
        source_distribution: Counter[str] = Counter()
        source_token_counts: Counter[str] = Counter()

        truncated_rows = 0
        padded_rows = 0
        chat_template_fallback_rows = 0
        total_rows = 0
        total_input_tokens = 0
        input_id_lengths: list[int] = []
        length_values: dict[str, list[int]] = {field: [] for field in self.length_fields}

        for row in source.iter_shards():
            total_rows += 1
            if total_rows > self.max_rows:
                total_rows = self.max_rows
                break

            if not isinstance(row, dict):
                continue

            for key in row.keys():
                field_counts[key] += 1

            for field in self.length_fields:
                length = self._get_length(row.get(field))
                if length is not None:
                    length_histogram[field][length] += 1
                    length_values[field].append(length)

            input_length = self._get_length(row.get("input_ids"))
            if input_length is not None:
                input_id_lengths.append(input_length)
                total_input_tokens += input_length

            if row.get("truncated"):
                truncated_rows += 1

            if self._is_padded(row):
                padded_rows += 1

            if self._is_chat_template_fallback(row):
                chat_template_fallback_rows += 1

            source_id = row.get("__source__")
            if source_id is not None:
                source_distribution[str(source_id)] += 1
                if input_length is not None:
                    source_token_counts[str(source_id)] += input_length

        estimated: dict[str, float] = {}
        gaps: dict[str, float] = {}
        if self.seq_length is not None and input_id_lengths:
            estimated = self._estimate_packing_efficiency(input_id_lengths)
            greedy_eff = estimated.get("greedy", 0.0)
            gaps = {
                "pool_vs_greedy": estimated.get("pool", 0.0) - greedy_eff,
                "first_fit_vs_greedy": estimated.get("first_fit", 0.0) - greedy_eff,
            }

        return DatasetProfile(
            num_rows_sampled=total_rows,
            field_counts=dict(field_counts),
            length_histogram={field: dict(hist) for field, hist in length_histogram.items()},
            length_percentiles=self._compute_percentiles(length_values),
            truncation_rate=truncated_rows / total_rows if total_rows else 0.0,
            padding_rate=padded_rows / total_rows if total_rows else 0.0,
            chat_template_fallback_rate=chat_template_fallback_rows / total_rows if total_rows else 0.0,
            source_distribution=dict(source_distribution),
            source_token_counts=dict(source_token_counts),
            total_input_tokens=total_input_tokens,
            estimated_packing_efficiency=estimated,
            packing_efficiency_gap=gaps,
            seq_length=self.seq_length,
        )

    @staticmethod
    def _get_length(value: tp.Any) -> int | None:
        """Return the length of a list/array-like field, or None."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        if isinstance(value, (str, bytes)):
            return None
        try:
            return len(value)
        except Exception:
            return None

    @staticmethod
    def _sum_mask(value: tp.Any) -> int | None:
        """Return the sum of an attention-mask-like field, or None."""
        if value is None:
            return None
        if isinstance(value, list):
            return sum(1 for x in value if x)
        if hasattr(value, "sum"):
            try:
                return int(value.sum())
            except Exception:
                return None
        return None

    def _is_padded(self, row: dict[str, tp.Any]) -> bool:
        """Detect a padded row from attention_mask vs input_ids lengths."""
        input_ids = row.get("input_ids")
        attention_mask = row.get("attention_mask")
        input_len = self._get_length(input_ids)
        mask_len = self._get_length(attention_mask)
        if input_len is None or mask_len is None or input_len != mask_len:
            return False
        mask_sum = self._sum_mask(attention_mask)
        return mask_sum is not None and mask_sum < input_len

    @staticmethod
    def _is_chat_template_fallback(row: dict[str, tp.Any]) -> bool:
        """Detect a row where chat-template application fell back."""
        if row.get("chat_template_applied") is False:
            return True
        if row.get("template_fallback") is True:
            return True
        return False

    @staticmethod
    def _compute_percentiles(
        length_values: dict[str, list[int]],
    ) -> dict[str, dict[str, float]]:
        """Compute p50/p90/p99 length percentiles per field."""
        import numpy as np

        result: dict[str, dict[str, float]] = {}
        for field, values in length_values.items():
            if not values:
                result[field] = {"p50": 0.0, "p90": 0.0, "p99": 0.0}
                continue
            arr = np.asarray(values, dtype=np.float64)
            result[field] = {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }
        return result

    def _estimate_packing_efficiency(self, lengths: list[int]) -> dict[str, float]:
        """Run lightweight packing simulators on a list of sequence lengths."""
        assert self.seq_length is not None
        seq_length = self.seq_length
        dummy_tokens = [self.eos_token_id]

        # Greedy
        greedy = GreedyPacker(seq_length, self.eos_token_id, self.pad_token_id)
        for length in lengths:
            greedy.add(dummy_tokens * length)
        greedy.flush_final()
        result: dict[str, float] = {"greedy": greedy.stats.efficiency}

        # Pool
        pool = PoolPacker(seq_length, self.eos_token_id, self.pad_token_id, num_packers=4)
        for length in lengths:
            pool.add(dummy_tokens * length)
        pool.flush_all()
        result["pool"] = pool.stats.efficiency

        # First-fit
        buffer_size = max(1, min(len(lengths), 1000))
        first_fit = FirstFitPacker(
            seq_length,
            self.eos_token_id,
            self.pad_token_id,
            buffer_size=buffer_size,
        )
        for length in lengths:
            first_fit.add(dummy_tokens * length)
        first_fit.flush_all()
        result["first_fit"] = first_fit.stats.efficiency

        return result


def profile_dataset(
    source: ShardedDataSource,
    max_rows: int = 10_000,
    seq_length: int | None = None,
    length_fields: Sequence[str] | None = None,
    eos_token_id: int = 1,
    pad_token_id: int = 0,
) -> DatasetProfile:
    """Profile a source and print the result.

    This is a convenience wrapper around :class:`DatasetProfiler` that
    both returns and prints the profile, useful for one-off diagnostics
    in notebooks or scripts.

    Args:
        source: Sharded source to sample.
        max_rows: Maximum rows to consume.
        seq_length: Window length for packing-efficiency estimates.
        length_fields: Fields for length histograms.
        eos_token_id: EOS id for the packing simulator.
        pad_token_id: PAD id for the packing simulator.

    Returns:
        DatasetProfile: The collected profile.
    """
    profiler = DatasetProfiler(
        max_rows=max_rows,
        seq_length=seq_length,
        length_fields=length_fields,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    profile = profiler.profile(source)
    print(profile)
    return profile
