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
"""Grain random-access data lanes for the trainers: raw parquet ("parquetra") and ArrayRecord.

Both lanes read each dataset as a ``grain.MapDataset`` (random access), shuffle it by a global index
permutation, mix datasets by weight, shard per host, and hand the trainer's ``data_collator`` LISTS of
row dicts. Resume is index arithmetic, never a replay of consumed batches.

* **Map lanes** (:func:`build_map_lane`): rows are batched per source and the batched streams are mixed,
  so every batch comes from one dataset (proportions hold in expectation). Seek to batch ``n`` is a
  ``slice`` -- exact and O(1).
* **Packed parquet lane** (:class:`PackedParquetLoader`): rows are mixed at ROW level and pulled until
  ``batch_size`` windows of ``pack_seq_len`` tokens are full (first fit, whole documents; rows that do
  not fit carry to the next window-set, bounded by ``misfit_patience``). Window boundaries depend on the
  data, so the loader records a cursor (stream position + carried rows) after every emitted window-set;
  the trainer persists it with each checkpoint and resume continues EXACTLY from it. Without a recorded
  cursor (no checkpoint state, or a different fleet/config) resume falls back to a seed-shifted stream:
  the stream a fresh build with ``step_start_point=n`` would produce, still without materializing the
  skipped window-sets.

Every row is stamped with the per-dataset sideband carriers (:data:`~easydel.data.ASSISTANT_ONLY_FIELD`,
:data:`~easydel.data.DATASET_ID_FIELD`) AFTER tokenization, so the tokenizer's purify allow-list cannot
strip them. Dataset ids index ``sorted(datasets)``.

``grain`` (a declared dependency) and ``msgpack`` / ``array_record`` (pulled in by grain) are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import fnmatch
import importlib
import json
import threading
import time
import typing as tp

import numpy as np
from eformer.loggings import get_logger

from easydel.data.transforms.mixture import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD

logger = get_logger(__name__)

DATA_LANE_STATE_DIR = "data_lane_state"

_PER_TOKEN_FIELDS = ("input_ids", "attention_mask", "labels", "completion_mask", "assistant_masks")

_PASSTHROUGH_FIELDS = (
    "input_ids",
    "attention_mask",
    "labels",
    "image_grid_thw",
    "image_embeds",
    "embed_n_tok",
    "embed_dim",
    "n_images",
    "seq_len",
)


class GrainLaneLoader:
    """Re-iterable loader with a known length and an O(1) ``iter_from_batch`` seek.

    Args:
        factory: Returns a fresh iterator from batch 0.
        length: Number of batches per pass.
        skip_factory: Returns an iterator positioned at batch ``n`` without materializing the skipped
            batches.
    """

    def __init__(
        self,
        factory: tp.Callable[[], collections.abc.Iterator],
        length: int,
        skip_factory: tp.Callable[[int], collections.abc.Iterator],
    ):
        self._factory = factory
        self._length = int(length)
        self._skip_factory = skip_factory

    def __iter__(self) -> collections.abc.Iterator:
        return self._factory()

    def __len__(self) -> int:
        return self._length

    def iter_from_batch(self, num_batches: int) -> collections.abc.Iterator:
        """Return an iterator positioned at batch ``num_batches``."""
        return self._skip_factory(max(int(num_batches), 0))


class StampSourceSideband:
    """Stamp the per-dataset sideband carriers on every row of one source.

    Args:
        assistant_only: Tri-state choice of the dataset: ``True`` opt in, ``False`` opt out, ``None``
            inherit the trainer's global ``assistant_only_loss``.
        dataset_id: The dataset's position in ``sorted(datasets)``.
    """

    def __init__(self, assistant_only: bool | None, dataset_id: int):
        self.flag = -1 if assistant_only is None else int(bool(assistant_only))
        self.dataset_id = int(dataset_id)

    def __call__(self, row: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
        out = dict(row)
        out[ASSISTANT_ONLY_FIELD] = self.flag
        out[DATASET_ID_FIELD] = self.dataset_id
        return out


class MsgpackDecode:
    """Decode one msgpack-packed ArrayRecord record (``bytes -> dict``); ``msgpack`` is imported lazily."""

    def __init__(self):
        self._unpackb = None

    def __getstate__(self) -> dict:
        return {"_unpackb": None}

    def __call__(self, record: bytes) -> dict:
        if self._unpackb is None:
            import msgpack

            self._unpackb = msgpack.unpackb
        return self._unpackb(record, raw=False)


class ReadRateLimiter:
    """Byte token bucket shared by every read thread of one process.

    The bucket starts empty so hosts that start together do not release a full burst at once.

    Args:
        bytes_per_second: Sustained rate.
        burst_seconds: Capacity in seconds of the rate.

    Raises:
        ValueError: On a non-positive rate or burst window.
    """

    def __init__(self, bytes_per_second: float, burst_seconds: float):
        self._lock = threading.Lock()
        self._tokens = 0.0
        self._updated_at = time.monotonic()
        self._bytes_per_second = 0.0
        self._capacity = 1.0
        self.configure(bytes_per_second, burst_seconds)

    @property
    def bytes_per_second(self) -> float:
        """The configured sustained rate."""
        return self._bytes_per_second

    @property
    def burst_seconds(self) -> float:
        """The configured burst window."""
        return self._burst_seconds

    def configure(self, bytes_per_second: float, burst_seconds: float) -> None:
        """Change the rate, crediting tokens accrued under the previous rate first."""
        bytes_per_second = float(bytes_per_second)
        burst_seconds = float(burst_seconds)
        if bytes_per_second <= 0:
            raise ValueError("read rate limiter needs a positive bytes/second rate.")
        if burst_seconds <= 0:
            raise ValueError("read rate limiter needs a positive burst window.")
        with self._lock:
            now = time.monotonic()
            if self._bytes_per_second > 0:
                elapsed = max(now - self._updated_at, 0.0)
                self._tokens = min(self._capacity, self._tokens + elapsed * self._bytes_per_second)
            self._bytes_per_second = bytes_per_second
            self._burst_seconds = burst_seconds
            self._capacity = max(bytes_per_second * burst_seconds, 1.0)
            self._tokens = min(self._tokens, self._capacity)
            self._updated_at = now

    def acquire(self, num_bytes: int) -> float:
        """Charge ``num_bytes``; sleep until the bucket is out of debt.

        Returns:
            float: Seconds slept.
        """
        num_bytes = int(num_bytes)
        if num_bytes <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            elapsed = max(now - self._updated_at, 0.0)
            self._tokens = min(self._capacity, self._tokens + elapsed * self._bytes_per_second)
            self._updated_at = now
            self._tokens -= float(num_bytes)
            wait = max(-self._tokens / self._bytes_per_second, 0.0)
        if wait > 0:
            time.sleep(wait)
        return wait


_READ_RATE_LIMITER: ReadRateLimiter | None = None
_READ_RATE_LIMITER_LOCK = threading.Lock()


def get_read_rate_limiter(bytes_per_second: float, burst_seconds: float) -> ReadRateLimiter:
    """Return the single per-process limiter, reconfiguring it when the settings changed."""
    global _READ_RATE_LIMITER
    with _READ_RATE_LIMITER_LOCK:
        if _READ_RATE_LIMITER is None:
            _READ_RATE_LIMITER = ReadRateLimiter(bytes_per_second, burst_seconds)
        elif (_READ_RATE_LIMITER.bytes_per_second, _READ_RATE_LIMITER.burst_seconds) != (
            float(bytes_per_second),
            float(burst_seconds),
        ):
            _READ_RATE_LIMITER.configure(bytes_per_second, burst_seconds)
        return _READ_RATE_LIMITER


def read_rate_bytes_per_second(mb_per_second: float | int | None) -> float | None:
    """Convert a MB/s cap to bytes/s; ``None`` or ``0`` disables the limiter.

    Raises:
        ValueError: On a bool, negative, or non-finite value.
    """
    if mb_per_second is None:
        return None
    if isinstance(mb_per_second, bool):
        raise ValueError("`arrayrecord_read_rate_limit_mb_per_sec` must be a number, None, or 0.")
    rate = float(mb_per_second)
    if rate == 0:
        return None
    if rate < 0 or not np.isfinite(rate):
        raise ValueError("`arrayrecord_read_rate_limit_mb_per_sec` must be finite and non-negative.")
    return rate * 1_000_000.0


class MeterReadBytes:
    """Charge each record's bytes (or a fixed per-record weight) to the process-wide read limiter.

    Args:
        bytes_per_second: Limiter rate.
        burst_seconds: Limiter burst window.
        bytes_per_record: Fixed charge per record (e.g. compressed bytes per row from a manifest);
            ``None`` charges ``len(record)``.
    """

    def __init__(self, bytes_per_second: float, burst_seconds: float, *, bytes_per_record: float | None = None):
        self.bytes_per_second = float(bytes_per_second)
        self.burst_seconds = float(burst_seconds)
        self.bytes_per_record = None if bytes_per_record is None else int(np.ceil(float(bytes_per_record)))

    def __call__(self, record: bytes) -> bytes:
        charge = len(record) if self.bytes_per_record is None else self.bytes_per_record
        get_read_rate_limiter(self.bytes_per_second, self.burst_seconds).acquire(charge)
        return record


def _strip_padding(out: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Drop padded positions (``attention_mask == 0``) from every per-token list of a tokenized row."""
    ids = list(out.get("input_ids") or [])
    attn = out.get("attention_mask")
    keep = list(range(len(ids))) if attn is None else [i for i, a in enumerate(attn) if int(a)]
    row: dict[str, tp.Any] = {}
    for key in _PER_TOKEN_FIELDS:
        value = out.get(key)
        if value is not None and len(value) == len(ids):
            row[key] = [int(value[i]) for i in keep]
    row["attention_mask"] = [1] * len(row["input_ids"])
    return row


def _finalize_row(row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Make ``labels`` explicit: masked positions of an existing completion/assistant mask become ``-100``."""
    if "labels" not in row:
        mask = row.get("completion_mask", row.get("assistant_masks"))
        ids = row["input_ids"]
        row["labels"] = list(ids) if mask is None else [t if m else -100 for t, m in zip(ids, mask, strict=True)]
    row.pop("assistant_masks", None)
    return row


def load_fast_encoder_factory(spec: str) -> tp.Callable[[tp.Any], tp.Callable[[str], list[int]]]:
    """Import a ``module:attr`` fast-tokenizer factory.

    Raises:
        ValueError: When ``spec`` is not of the form ``module:attr``.
    """
    module_name, sep, attr = str(spec).partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"`parquetra_fast_tokenizer` must be a 'module:attr' import path, got {spec!r}.")
    target: tp.Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


class ParquetRowTokenize:
    """Tokenize one raw parquet row with the trainer's preprocess transform.

    Emits ``input_ids``, ``attention_mask`` (all ones) and ``labels`` (``-100`` where the transform's
    completion/assistant mask excluded a token), with padding removed so rows can be packed. A row that
    tokenizes to nothing raises: these lanes form batches by index arithmetic and cannot skip rows
    without desynchronizing hosts. Pre-embedded multimodal rows (``input_ids`` + ``image_embeds``) pass
    through.

    An optional fast encoder renders conversational rows with the chat template and encodes them with
    ``fast_encoder_factory(hf_tokenizer)``. Each process checks the first ``verify_rows`` conversational
    rows against the full transform and raises on any difference; only then does the fast path run alone.

    Args:
        transform: Callable ``row -> tokenized row`` (e.g. ``SFTPreprocessTransform``).
        hf_tokenizer: Tokenizer used for the fast path's chat-template render.
        fast_encoder_factory: Optional ``factory(hf_tokenizer) -> encode(text) -> list[int]``.
        messages_field: Conversation column for the fast path.
        max_length: Truncation length of the fast path.
        truncation_mode: ``"keep_end"`` or ``"keep_start"``; must match the transform.
        verify_rows: Rows to verify before trusting the fast path.

    Raises:
        ValueError: On an unknown ``truncation_mode``.
    """

    def __init__(
        self,
        transform: tp.Callable[[dict], dict],
        *,
        hf_tokenizer: tp.Any = None,
        fast_encoder_factory: tp.Callable[[tp.Any], tp.Callable[[str], list[int]]] | None = None,
        messages_field: str = "messages",
        max_length: int | None = None,
        truncation_mode: str = "keep_end",
        verify_rows: int = 64,
    ):
        if truncation_mode not in ("keep_end", "keep_start"):
            raise ValueError(f"truncation_mode must be keep_end|keep_start, got {truncation_mode!r}")
        self.transform = transform
        self.hf_tokenizer = hf_tokenizer
        self.fast_encoder_factory = fast_encoder_factory
        self.messages_field = messages_field
        self.max_length = max_length
        self.truncation_mode = truncation_mode
        self.verify_rows = int(verify_rows)
        self.fast_verified = 0
        self._encode = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_encode"] = None  # built lazily per worker process
        return state

    def _fast_ids(self, row: tp.Mapping[str, tp.Any]) -> list[int] | None:
        """Fast-path ids, or ``None`` when the row is outside the fast common case."""
        from .prompt_utils import maybe_convert_to_chatml, normalize_message_payload, resolve_example_tools

        if self._encode is None:
            self._encode = self.fast_encoder_factory(self.hf_tokenizer)
        result = maybe_convert_to_chatml(dict(row))
        raw = result.get(self.messages_field)
        if raw is None and self.messages_field != "messages":
            raw = result.get("messages")
        messages = normalize_message_payload(raw, allow_plain_text=True)
        if not messages:
            return None
        try:
            text = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, tools=resolve_example_tools(result))
        except Exception:
            return None
        ids = [int(t) for t in self._encode(text)]
        if self.max_length and len(ids) > self.max_length:
            ids = ids[: self.max_length] if self.truncation_mode == "keep_start" else ids[-self.max_length :]
        return ids

    def _slow(self, row: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
        out = self.transform(dict(row))
        if not out or not out.get("input_ids"):
            keys = sorted(row) if isinstance(row, collections.abc.Mapping) else type(row).__name__
            raise ValueError(
                f"in-loop tokenization produced no input_ids for a row (keys={keys}); fix the row or the "
                "text/messages field mapping -- this lane cannot skip rows without desynchronizing hosts."
            )
        return _finalize_row(_strip_padding(out))

    def __call__(self, row: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
        if isinstance(row, collections.abc.Mapping) and row.get("input_ids") is not None:
            if row.get("image_embeds") is not None:
                return {k: row[k] for k in _PASSTHROUGH_FIELDS if row.get(k) is not None}
            return _finalize_row(_strip_padding(row))
        if self.fast_encoder_factory is None:
            return self._slow(row)
        if self.fast_verified >= self.verify_rows:
            ids = self._fast_ids(row)
            if ids is None:
                return self._slow(row)
            if not ids:
                raise ValueError("fast tokenization produced no input_ids for a row")
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}
        slow = self._slow(row)
        fast = self._fast_ids(row)
        if fast is None:
            return slow
        if fast != slow["input_ids"] or slow["labels"] != slow["input_ids"]:
            div = next(
                (i for i, (a, b) in enumerate(zip(slow["input_ids"], fast, strict=False)) if a != b),
                min(len(slow["input_ids"]), len(fast)),
            )
            raise ValueError(
                "parquetra fast tokenizer does not reproduce the preprocess transform for this row (template or "
                f"prompt-masking mismatch): slow_len={len(slow['input_ids'])} fast_len={len(fast)} "
                f"first_divergence_at={div} labels_equal_ids={slow['labels'] == slow['input_ids']}. Unset "
                "`parquetra_fast_tokenizer` for this run."
            )
        self.fast_verified += 1
        return slow


class CpuOnlyWorkerInit:
    """``mp_prefetch`` worker initializer that makes a spawned data worker CPU-only.

    grain spawns worker processes (``multiprocessing`` ``spawn`` context) and runs this initializer in
    the CHILD before it unpickles the pipeline. It edits only that child's environment so a worker that
    imports JAX cannot try to claim the accelerators (or the multi-host runtime) its parent holds; the
    trainer process's environment is never touched.

    Args:
        drop_prefixes: Variable-name prefixes removed in the child.
        overrides: Variables set in the child.
    """

    def __init__(
        self,
        drop_prefixes: tuple[str, ...] = ("TPU_", "MEGASCALE_", "CLOUD_TPU_"),
        overrides: tp.Mapping[str, str] | None = None,
    ):
        self.drop_prefixes = tuple(drop_prefixes)
        self.overrides = dict(overrides or {"JAX_PLATFORMS": "cpu", "ENABLE_DISTRIBUTED_INIT": "0"})

    def __call__(self, worker_index: int | None = None, worker_count: int | None = None) -> None:
        import os
        import sys

        del worker_index, worker_count
        for key in [k for k in os.environ if k.startswith(self.drop_prefixes)]:
            os.environ.pop(key, None)
        os.environ.update(self.overrides)
        if "jax" in sys.modules and "JAX_PLATFORMS" in self.overrides:
            sys.modules["jax"].config.update("jax_platforms", self.overrides["JAX_PLATFORMS"])


def _mark_absl_flags_parsed() -> None:
    """grain workers read absl flags; outside ``absl.app.run`` mark them parsed (no-op under a real app)."""
    try:
        from absl import flags

        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
    except Exception:  # absl missing or already configured: grain surfaces real problems itself
        pass


def resolve_lane_assistant_only(
    datasets: tp.Mapping[str, tp.Any], choice: tp.Mapping[str, tp.Any] | None, field_name: str
) -> dict[str, bool]:
    """Validate a per-dataset assistant-only dict (partial dicts inherit; unknown keys and non-bools raise)."""
    choice = dict(choice or {})
    unknown = sorted(set(choice) - set(datasets))
    if unknown:
        raise ValueError(f"`{field_name}` names unknown dataset(s) {unknown}; known: {sorted(datasets)}.")
    for key, value in choice.items():
        if not isinstance(value, bool):
            raise TypeError(f"`{field_name}[{key!r}]` must be a bool, got {type(value).__name__}: {value!r}")
    return choice


def check_lane_weights(
    datasets: tp.Mapping[str, tp.Any], weights: tp.Mapping[str, float] | None, field_name: str
) -> None:
    """Raise unless ``weights`` keys equal the dataset names and every weight is finite and >= 0."""
    if weights is None:
        return
    missing, extra = set(datasets) - set(weights), set(weights) - set(datasets)
    if missing or extra:
        raise ValueError(
            f"{field_name} keys must match the dataset names; missing {sorted(missing)}, unknown {sorted(extra)}."
        )
    bad = {k: v for k, v in weights.items() if not np.isfinite(float(v)) or float(v) < 0}
    if bad or sum(float(v) for v in weights.values()) <= 0:
        raise ValueError(f"{field_name} must be finite, non-negative and sum > 0; got {dict(weights)!r}.")


def _weights_list(names: list[str], weights: tp.Mapping[str, float] | None, rows: tp.Mapping[str, int]) -> list:
    return [float(weights[n]) if weights else float(rows[n]) for n in names]


def build_map_lane(
    per_source: list,
    names: list[str],
    rows_by_name: dict[str, int],
    weights: tp.Mapping[str, float] | None,
    *,
    batch_size: int,
    max_steps: int | None,
    num_epochs: int | None,
    shard_index: int,
    shard_count: int,
    read_threads: int,
    prefetch_buffer: int,
    lane: str,
) -> GrainLaneLoader:
    """Mix per-source batched ``MapDataset`` streams, shard per host and return a seekable loader.

    With several sources, several hosts and an explicit ``max_steps``, each source is strided per host
    BEFORE the mix: striding the mixed stream samples one residue class of grain's periodic mix layout and
    can lock a host to one source. The step count is ``max_steps`` on every host in that case, so the
    per-host length difference of a per-source stride never reaches a loop bound. Otherwise the mixed
    stream is strided (the step count is the uniform floor over hosts).

    Args:
        per_source: One ``MapDataset`` of LIST batches per source, aligned with ``names``.
        names: Source names.
        rows_by_name: Row count per source.
        weights: Optional mixture weights (``None`` = row-count proportional).
        batch_size: Rows per batch.
        max_steps: Explicit step count, or ``None`` to derive it from the data.
        num_epochs: Epochs when the step count is derived.
        shard_index: This host's index.
        shard_count: Number of hosts.
        read_threads: grain read threads.
        prefetch_buffer: grain prefetch buffer.
        lane: Lane name for messages.

    Returns:
        GrainLaneLoader: ``length`` batches per pass, O(1) seek.

    Raises:
        ValueError: On an unbounded mixture without ``max_steps`` or a source too small for the fleet.
    """
    import grain.python as grain

    shard_count = max(int(shard_count), 1)
    mixture = len(per_source) > 1
    target = int(max_steps) if max_steps and int(max_steps) > 0 else None
    if target is None and mixture:
        raise ValueError(f"{lane} weighted mixtures need an explicit max_training_steps.")
    ws = _weights_list(names, weights, rows_by_name)
    shard_first = mixture and shard_count > 1 and target is not None
    if shard_first:
        strided = []
        for name, ds in zip(names, per_source, strict=True):
            if len(ds) < shard_count:
                raise ValueError(
                    f"{lane} source {name!r} yields {len(ds)} batches of {batch_size} rows, fewer than the "
                    f"{shard_count} hosts; some host would never draw it."
                )
            strided.append(ds.slice(slice(shard_index, None, shard_count)))
        mixed = grain.MapDataset.mix(strided, weights=ws)
    else:
        mixed = grain.MapDataset.mix(per_source, weights=ws) if mixture else per_source[0]
    if target is not None:
        base = len(mixed)
        if base <= 0:
            raise ValueError(f"{lane} mixture has no full batch; lower the batch size or add rows.")
        needed = target if shard_first else target * shard_count
        mixed = mixed.repeat(-(-needed // base) + 1)
    elif num_epochs and int(num_epochs) > 1:
        mixed = mixed.repeat(int(num_epochs))
    steps = target if target is not None else len(mixed) // shard_count
    if steps <= 0:
        raise ValueError(f"{lane} loader resolved 0 steps; check dataset sizes and shard count.")
    sharded = mixed if (shard_first or shard_count == 1) else mixed.slice(slice(shard_index, None, shard_count))
    sharded = sharded.slice(slice(0, steps))
    read_options = grain.ReadOptions(num_threads=int(read_threads), prefetch_buffer_size=int(prefetch_buffer))
    logger.info(
        "%s lane: %d steps x %d rows on host %d/%d, %s, sources %s",
        lane,
        steps,
        batch_size,
        shard_index,
        shard_count,
        "per-source host sharding" if shard_first else "mixed-stream host sharding",
        dict(zip(names, ws, strict=True)),
    )

    def _from(n: int) -> collections.abc.Iterator:
        view = sharded if n == 0 else sharded.slice(slice(int(n), None))
        return iter(view.to_iter_dataset(read_options=read_options))

    return GrainLaneLoader(factory=lambda: _from(0), length=steps, skip_factory=_from)


@dataclasses.dataclass(frozen=True)
class PackedLaneCursor:
    """Exact position of the packed lane after ``step`` window-sets.

    Attributes:
        step: Window-sets consumed.
        shift: Seed shift of the stream the cursor indexes.
        cursor: Index of the next unread row of that stream.
        carry: Stream indices of rows carried into the next window-set, in order.
        fingerprint: Configuration fingerprint; a cursor from another configuration is ignored.
    """

    step: int
    shift: int
    cursor: int
    carry: tuple[int, ...]
    fingerprint: str

    def to_json(self) -> dict:
        """JSON view."""
        return {**dataclasses.asdict(self), "carry": list(self.carry)}

    @classmethod
    def from_json(cls, data: tp.Mapping[str, tp.Any]) -> PackedLaneCursor:
        """Inverse of :meth:`to_json`."""
        return cls(
            step=int(data["step"]),
            shift=int(data["shift"]),
            cursor=int(data["cursor"]),
            carry=tuple(int(i) for i in data["carry"]),
            fingerprint=str(data["fingerprint"]),
        )


class PackedParquetLoader:
    """Demand-driven packed loader over weight-mixed, per-host-sharded random-access row streams.

    Each call of the iterator yields one window-set: a flat list of row dicts, window-major (all rows of
    window 0, then window 1, ...), whose rows fit ``batch_size`` windows of ``pack_seq_len`` tokens. Rows
    are placed first fit; a window-set closes once ``misfit_patience`` rows failed to place, and those
    rows carry to the next set.

    Args:
        build_stream: ``shift -> MapDataset`` of tokenized, stamped rows (this host's stream).
        batch_size: Windows per set.
        pack_seq_len: Tokens per window.
        max_steps: Window-sets per pass.
        misfit_patience: Consecutive misfits that close a window-set.
        embed_cap: Max ``sum(embed_n_tok)`` per set (``0`` = uncapped).
        read_threads: grain read threads.
        prefetch_buffer: grain prefetch buffer.
        mp_workers: grain ``mp_prefetch`` worker processes (``0`` = threads only).
        worker_init: Optional per-worker initializer for ``mp_prefetch``.
        base_shift: Seed shift used for a fresh pass (the run's ``step_start_point``).
        fingerprint: Configuration fingerprint stored in cursors.
        resume_cursor: Cursor restored from a checkpoint, used by :meth:`iter_from_batch`.
        cursor_history: Recorded cursors kept per loader (bounded).
    """

    def __init__(
        self,
        build_stream: tp.Callable[[int], tp.Any],
        *,
        batch_size: int,
        pack_seq_len: int,
        max_steps: int,
        misfit_patience: int,
        embed_cap: int = 0,
        read_threads: int = 1,
        prefetch_buffer: int = 2,
        mp_workers: int = 0,
        worker_init: tp.Callable[..., None] | None = None,
        base_shift: int = 0,
        fingerprint: str = "",
        resume_cursor: PackedLaneCursor | None = None,
        cursor_history: int = 4096,
    ):
        self._build_stream = build_stream
        self.batch_size = int(batch_size)
        self.pack_seq_len = int(pack_seq_len)
        self.max_steps = int(max_steps)
        self.misfit_patience = int(misfit_patience)
        self.embed_cap = int(embed_cap) if int(embed_cap) > 0 else self.batch_size * self.pack_seq_len
        self.read_threads = int(read_threads)
        self.prefetch_buffer = int(prefetch_buffer)
        self.mp_workers = int(mp_workers)
        self.worker_init = worker_init
        self.base_shift = int(base_shift)
        self.fingerprint = fingerprint
        self.resume_cursor = resume_cursor
        self._streams: dict[int, tp.Any] = {}
        self._cursors: collections.OrderedDict[int, PackedLaneCursor] = collections.OrderedDict()
        self._cursor_history = int(cursor_history)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self.max_steps

    def _stream(self, shift: int):
        if shift not in self._streams:
            self._streams[shift] = self._build_stream(shift)
        return self._streams[shift]

    def _row_iter(self, stream, start: int) -> collections.abc.Iterator:
        import grain.python as grain

        view = stream if start == 0 else stream.slice(slice(int(start), None))
        it_ds = view.to_iter_dataset(
            read_options=grain.ReadOptions(num_threads=self.read_threads, prefetch_buffer_size=self.prefetch_buffer)
        )
        if self.mp_workers > 0:
            _mark_absl_flags_parsed()
            it_ds = it_ds.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=self.mp_workers, per_worker_buffer_size=8),
                worker_init_fn=self.worker_init,
            )
        return iter(it_ds)

    def _record(self, cursor: PackedLaneCursor) -> None:
        with self._lock:
            self._cursors[cursor.step] = cursor
            while len(self._cursors) > self._cursor_history:
                self._cursors.popitem(last=False)

    def cursor_at(self, step: int) -> PackedLaneCursor | None:
        """Return the recorded cursor after ``step`` window-sets, if this process produced it."""
        with self._lock:
            return self._cursors.get(int(step))

    def _iterate(self, *, step: int, shift: int, cursor: int, carry: tuple[int, ...]):
        stream = self._stream(shift)
        rows = self._row_iter(stream, cursor)
        pending: list[tuple[int, dict]] = [(i, stream[i]) for i in carry]
        while step < self.max_steps:
            windows: list[list[dict]] = [[] for _ in range(self.batch_size)]
            remaining = [self.pack_seq_len] * self.batch_size
            embed_left = self.embed_cap
            misfits: list[tuple[int, dict]] = []
            queue, pending = pending, []
            while True:
                if queue:
                    index, row = queue.pop(0)
                else:
                    index, row = cursor, next(rows)
                    cursor += 1
                length = len(row["input_ids"])
                if length > self.pack_seq_len:
                    raise ValueError(
                        f"packed lane: a row of {length} tokens exceeds the {self.pack_seq_len}-token window; the "
                        "packer never truncates -- set max_length <= parquetra_pack_seq_len."
                    )
                embeds = row.get("embed_n_tok")
                embed_rows = int(sum(int(v) for v in embeds)) if embeds is not None and len(embeds) else 0
                placed = False
                if embed_rows <= embed_left:
                    for w in range(self.batch_size):
                        if remaining[w] >= length:
                            windows[w].append(row)
                            remaining[w] -= length
                            embed_left -= embed_rows
                            placed = True
                            break
                if not placed:
                    misfits.append((index, row))
                    if len(misfits) >= self.misfit_patience:
                        break
            pending = misfits + queue
            step += 1
            self._record(
                PackedLaneCursor(
                    step=step,
                    shift=shift,
                    cursor=cursor,
                    carry=tuple(i for i, _ in pending),
                    fingerprint=self.fingerprint,
                )
            )
            yield [r for window in windows for r in window]

    def __iter__(self) -> collections.abc.Iterator:
        return self._iterate(step=0, shift=self.base_shift, cursor=0, carry=())

    def iter_from_batch(self, num_batches: int) -> collections.abc.Iterator:
        """Position at window-set ``num_batches`` without materializing the skipped sets.

        Uses (in order) a cursor recorded by this loader, the checkpoint-restored cursor, and otherwise
        the seed-shifted stream (``shift = num_batches``), i.e. what a fresh build at
        ``step_start_point = num_batches`` produces.
        """
        n = max(int(num_batches), 0)
        if n == 0:
            return iter(self)
        cursor = self.cursor_at(n)
        if cursor is None and self.resume_cursor is not None:
            if self.resume_cursor.step == n and self.resume_cursor.fingerprint == self.fingerprint:
                cursor = self.resume_cursor
            else:
                logger.warning(
                    "packed lane: restored cursor (step %d) does not match the resume step %d or this "
                    "configuration; falling back to a seed-shifted stream.",
                    self.resume_cursor.step,
                    n,
                )
        if cursor is not None:
            logger.info(
                "packed lane: exact resume at window-set %d (stream row %d, %d carried rows).",
                n,
                cursor.cursor,
                len(cursor.carry),
            )
            return self._iterate(step=n, shift=cursor.shift, cursor=cursor.cursor, carry=cursor.carry)
        return self._iterate(step=n, shift=n, cursor=0, carry=())


def build_packed_parquet_stream_factory(
    sources: dict[str, tp.Any],
    tokenizers: dict[str, tp.Callable[[dict], dict]],
    assistant_only: tp.Mapping[str, bool],
    weights: tp.Mapping[str, float] | None,
    *,
    shard_index: int,
    shard_count: int,
    seed: int,
    shuffle: bool,
    rows_needed: int,
) -> tp.Callable[[int], tp.Any]:
    """Return ``shift -> MapDataset`` producing this host's tokenized, stamped, weight-mixed row stream.

    Each source is shuffled (seed ``seed + i + shift``), strided per host BEFORE the mix, tokenized and
    stamped; the mixed stream is repeated so ``rows_needed`` rows are always available.

    Raises:
        ValueError: When a source has fewer rows than hosts.
    """
    import grain.python as grain

    names = sorted(sources)
    rows = {name: len(sources[name]) for name in names}
    shard_count = max(int(shard_count), 1)

    def build(shift: int):
        per_source = []
        for i, name in enumerate(names):
            ds = grain.MapDataset.source(sources[name])
            if shuffle:
                ds = ds.shuffle(seed=int(seed) + i + int(shift))
            if shard_count > 1:
                ds = ds.slice(slice(int(shard_index), None, shard_count))
                if len(ds) == 0:
                    raise ValueError(f"packed lane: source {name!r} has fewer rows than the {shard_count} hosts.")
            ds = ds.map(tokenizers[name]).map(StampSourceSideband(assistant_only.get(name), i))
            per_source.append(ds)
        mixed = (
            grain.MapDataset.mix(per_source, weights=_weights_list(names, weights, rows))
            if len(per_source) > 1
            else per_source[0]
        )
        return mixed.repeat(-(-int(rows_needed) // max(len(mixed), 1)) + 1)

    return build


def lane_fingerprint(**parts: tp.Any) -> str:
    """Stable fingerprint of a lane configuration (JSON of the sorted parts)."""
    import hashlib

    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclasses.dataclass
class ArrayRecordFileInstruction:
    """``array_record`` ``FileInstruction`` (read ``take`` records of ``filename`` after ``skip``)."""

    filename: str
    skip: int
    take: int
    examples_in_shard: int


def arrayrecord_manifest_instructions(
    pattern: str, *, source_name: str
) -> tuple[list[ArrayRecordFileInstruction], float] | None:
    """Read ``_MANIFEST.tsv`` next to a set's shards, if present.

    Each manifest line is ``relpath<TAB>nbytes<TAB>nrows`` with ``relpath`` relative to the set directory's
    parent; the glob's file pattern filters the entries. Using it skips the per-file footer reads.

    Args:
        pattern: The set's shard glob (``<root>/<set>/<shard-pattern>``).
        source_name: Dataset name for messages.

    Returns:
        ``(instructions, average bytes per row)``, or ``None`` when the set has no manifest.

    Raises:
        ValueError: On malformed lines or a manifest that selects no shard.
    """
    from eformer.paths import ePath

    set_dir, _, shard_pattern = str(pattern).rpartition("/")
    root = set_dir.rpartition("/")[0]
    manifest = ePath(f"{set_dir}/_MANIFEST.tsv")
    if not manifest.exists():
        return None
    instructions: list[ArrayRecordFileInstruction] = []
    total_bytes = total_rows = 0
    for line in manifest.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 3 or not parts[0].endswith(".array_record"):
            continue
        if not fnmatch.fnmatch(parts[0].rpartition("/")[2], shard_pattern):
            continue
        try:
            nbytes, nrows = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ValueError(f"{manifest}: invalid byte/row counts in line {line!r}") from exc
        if nbytes <= 0 or nrows <= 0:
            raise ValueError(f"{manifest}: non-positive byte/row counts in line {line!r}")
        total_bytes += nbytes
        total_rows += nrows
        instructions.append(
            ArrayRecordFileInstruction(filename=f"{root}/{parts[0]}", skip=0, take=nrows, examples_in_shard=nrows)
        )
    if not instructions:
        raise ValueError(f"{manifest} selects no shard for source {source_name!r} (pattern {shard_pattern!r}).")
    return instructions, total_bytes / total_rows


def _host_shard(arguments: tp.Any) -> tuple[int, int]:
    import jax

    index = arguments.grain_shard_index if arguments.grain_shard_index is not None else jax.process_index()
    count = arguments.grain_shard_count if arguments.grain_shard_count is not None else jax.process_count()
    return int(index), max(int(count), 1)


def _require_user_collator(trainer: tp.Any, field_name: str) -> None:
    if not getattr(trainer, "_user_data_collator", False):
        raise ValueError(
            f"`{field_name}` requires an explicit `data_collator`: the lane yields uncollated LIST batches of row "
            "dicts and the trainer applies your collator to each."
        )


def lane_state_path(checkpoint_dir: str, process_index: int) -> str:
    """Path of one host's packed-lane cursor file inside a checkpoint directory."""
    return f"{str(checkpoint_dir).rstrip('/')}/{DATA_LANE_STATE_DIR}/host-{int(process_index):04d}.json"


def read_lane_cursor(checkpoint_dir: str | None, process_index: int) -> PackedLaneCursor | None:
    """Read this host's packed-lane cursor from a checkpoint directory, or ``None`` when absent."""
    if not checkpoint_dir:
        return None
    from eformer.paths import ePath

    path = ePath(lane_state_path(checkpoint_dir, process_index))
    try:
        if not path.exists():
            return None
        return PackedLaneCursor.from_json(json.loads(path.read_text()))
    except Exception as exc:
        logger.warning("packed lane: unreadable cursor %s (%s); resume will use the seed-shifted stream.", path, exc)
        return None


def write_lane_cursor(checkpoint_dir: str, process_index: int, cursor: PackedLaneCursor) -> str:
    """Write this host's packed-lane cursor next to a checkpoint and return the path."""
    from eformer.paths import ePath

    path = ePath(lane_state_path(checkpoint_dir, process_index))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor.to_json()))
    return str(path)


def _parquetra_tokenizers(trainer: tp.Any, names: list[str]) -> dict[str, ParquetRowTokenize]:
    """Build one :class:`ParquetRowTokenize` per source from the trainer's own preprocess transform."""
    from .prompt_transforms import SFTPreprocessTransform

    arguments = trainer.arguments
    processor = trainer._get_processing_class()
    if processor is None:
        if not arguments.parquetra_tokenizer_id:
            raise ValueError(
                "the parquetra lane tokenizes in the loader but found no tokenizer: attach `processing_class` or "
                "set `parquetra_tokenizer_id`."
            )
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingTypeStubs]

        processor = AutoTokenizer.from_pretrained(arguments.parquetra_tokenizer_id)
    max_length = int(arguments.max_length or 0)
    if max_length <= 0:
        raise ValueError("the parquetra lane needs a positive `max_length` (rows are truncated in the loader).")
    masking = bool(getattr(arguments, "assistant_only_loss", False)) or any(
        bool(v) for v in (arguments.parquetra_assistant_only or {}).values()
    )
    # keep_end would cut assistant headers off over-long rows, and the span scan then zeroes those documents.
    truncation_mode = "keep_start" if masking else str(arguments.truncation_mode)
    transform = trainer._get_preprocess_transform()
    if transform is None:
        transform = SFTPreprocessTransform(
            tokenizer=processor,
            max_length=max_length,
            text_field="messages",
            mask_prompt=False,
            padding=False,
            truncation_mode=truncation_mode,
        )
    factory = None
    if arguments.parquetra_fast_tokenizer:
        factory = load_fast_encoder_factory(arguments.parquetra_fast_tokenizer)
    hf_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return {
        name: ParquetRowTokenize(
            transform,
            hf_tokenizer=hf_tokenizer,
            fast_encoder_factory=factory,
            max_length=max_length,
            truncation_mode=getattr(transform, "_truncation_mode", truncation_mode),
            verify_rows=int(arguments.parquetra_fast_tokenizer_verify_rows),
        )
        for name in names
    }


def configure_parquetra_lane(trainer: tp.Any) -> tuple[tp.Any, int]:
    """Build the random-access parquet lane (map or packed) for ``trainer``.

    Returns:
        ``(loader, steps)``.

    Raises:
        ValueError: On a missing collator, bad weights / assistant-only keys, or (packed) no ``max_training_steps``.
    """
    import grain.python as grain
    import jax

    from easydel.data.sources.parquet_random_access import ParquetRandomAccessSource

    arguments = trainer.arguments
    _require_user_collator(trainer, "parquetra_train_datasets")
    datasets = dict(arguments.parquetra_train_datasets)
    weights = arguments.parquetra_mixture_weights
    check_lane_weights(datasets, weights, "parquetra_mixture_weights")
    ao = resolve_lane_assistant_only(datasets, arguments.parquetra_assistant_only, "parquetra_assistant_only")
    names = sorted(datasets)
    trainer._dataset_id_names = dict(enumerate(names))
    shard_index, shard_count = _host_shard(arguments)
    sources = {
        name: ParquetRandomAccessSource(
            datasets[name],
            metadata_cache_dir=arguments.parquet_metadata_cache_dir,
            process_index=jax.process_index(),
        )
        for name in names
    }
    tokenizers = _parquetra_tokenizers(trainer, names)
    batch_size = int(trainer.training_batch_size)
    seed = int(arguments.shuffle_seed_train)
    shuffle = bool(arguments.shuffle_train_dataset)
    pack_seq_len = int(arguments.parquetra_pack_seq_len)

    if pack_seq_len > 0:
        max_steps = int(arguments.max_training_steps or 0)
        if max_steps <= 0:
            raise ValueError("`parquetra_pack_seq_len > 0` requires an explicit `max_training_steps`.")
        patience = int(arguments.parquetra_pack_misfit_patience)
        build = build_packed_parquet_stream_factory(
            sources,
            tokenizers,
            ao,
            weights,
            shard_index=shard_index,
            shard_count=shard_count,
            seed=seed,
            shuffle=shuffle,
            rows_needed=max_steps * (batch_size * pack_seq_len + patience),
        )
        fingerprint = lane_fingerprint(
            datasets=datasets,
            weights=weights,
            assistant_only=ao,
            seed=seed,
            shuffle=shuffle,
            shard=(shard_index, shard_count),
            batch_size=batch_size,
            pack_seq_len=pack_seq_len,
            patience=patience,
            embed_cap=int(arguments.parquetra_pack_embed_cap),
            max_length=int(arguments.max_length or 0),
        )
        worker_init = CpuOnlyWorkerInit() if arguments.parquetra_mp_worker_cpu_only else None
        loader = PackedParquetLoader(
            build,
            batch_size=batch_size,
            pack_seq_len=pack_seq_len,
            max_steps=max_steps,
            misfit_patience=patience,
            embed_cap=int(arguments.parquetra_pack_embed_cap),
            read_threads=int(arguments.arrayrecord_num_threads),
            prefetch_buffer=int(arguments.arrayrecord_prefetch_buffer),
            mp_workers=int(arguments.parquetra_mp_workers),
            worker_init=worker_init,
            base_shift=max(int(arguments.step_start_point or 0), 0),
            fingerprint=fingerprint,
            resume_cursor=getattr(trainer, "_data_lane_resume_cursor", None),
        )
        return loader, max_steps

    per_source, rows = [], {}
    for i, name in enumerate(names):
        ds = grain.MapDataset.source(sources[name])
        rows[name] = len(ds)
        if shuffle:
            ds = ds.shuffle(seed=seed + i)
        ds = ds.map(tokenizers[name]).map(StampSourceSideband(ao.get(name), i))
        per_source.append(ds.batch(batch_size=batch_size, drop_remainder=True, batch_fn=list))
    loader = build_map_lane(
        per_source,
        names,
        rows,
        weights,
        batch_size=batch_size,
        max_steps=arguments.max_training_steps,
        num_epochs=arguments.num_train_epochs,
        shard_index=shard_index,
        shard_count=shard_count,
        read_threads=int(arguments.arrayrecord_num_threads),
        prefetch_buffer=int(arguments.arrayrecord_prefetch_buffer),
        lane="parquetra",
    )
    return loader, len(loader)


def configure_arrayrecord_lane(trainer: tp.Any) -> tuple[tp.Any, int]:
    """Build the ArrayRecord (msgpack rows) map lane for ``trainer``.

    Returns:
        ``(loader, steps)``.
    """
    import grain.python as grain

    from easydel.data.sources.base import expand_data_files

    arguments = trainer.arguments
    _require_user_collator(trainer, "arrayrecord_train_datasets / arrayrecord_train_files")
    if arguments.arrayrecord_train_datasets:
        datasets = dict(arguments.arrayrecord_train_datasets)
        weights = arguments.arrayrecord_mixture_weights
    else:
        datasets = {"_default": arguments.arrayrecord_train_files}
        weights = None
    check_lane_weights(datasets, weights, "arrayrecord_mixture_weights")
    ao = resolve_lane_assistant_only(datasets, arguments.arrayrecord_assistant_only, "arrayrecord_assistant_only")
    names = sorted(datasets)
    trainer._dataset_id_names = dict(enumerate(names))
    rate = read_rate_bytes_per_second(arguments.arrayrecord_read_rate_limit_mb_per_sec)
    burst = float(arguments.arrayrecord_read_rate_burst_seconds)
    shard_index, shard_count = _host_shard(arguments)
    batch_size = int(trainer.training_batch_size)
    seed = int(arguments.shuffle_seed_train)
    per_source, rows = [], {}
    for i, name in enumerate(names):
        spec = datasets[name]
        manifest = arrayrecord_manifest_instructions(spec, source_name=name) if isinstance(spec, str) else None
        paths, per_record = (manifest[0], manifest[1]) if manifest is not None else (expand_data_files(spec), None)
        source = grain.ArrayRecordDataSource(paths)
        rows[name] = len(source)
        ds = grain.MapDataset.source(source)
        if arguments.shuffle_train_dataset:
            ds = ds.shuffle(seed=seed + i)
        if rate is not None:
            ds = ds.map(MeterReadBytes(rate, burst, bytes_per_record=per_record))
        ds = ds.map(MsgpackDecode()).map(StampSourceSideband(ao.get(name), i))
        per_source.append(ds.batch(batch_size=batch_size, drop_remainder=True, batch_fn=list))
    loader = build_map_lane(
        per_source,
        names,
        rows,
        weights,
        batch_size=batch_size,
        max_steps=arguments.max_training_steps,
        num_epochs=arguments.num_train_epochs,
        shard_index=shard_index,
        shard_count=shard_count,
        read_threads=int(arguments.arrayrecord_num_threads),
        prefetch_buffer=int(arguments.arrayrecord_prefetch_buffer),
        lane="arrayrecord",
    )
    return loader, len(loader)


__all__ = (
    "DATA_LANE_STATE_DIR",
    "ArrayRecordFileInstruction",
    "CpuOnlyWorkerInit",
    "GrainLaneLoader",
    "MeterReadBytes",
    "MsgpackDecode",
    "PackedLaneCursor",
    "PackedParquetLoader",
    "ParquetRowTokenize",
    "ReadRateLimiter",
    "StampSourceSideband",
    "arrayrecord_manifest_instructions",
    "build_map_lane",
    "build_packed_parquet_stream_factory",
    "check_lane_weights",
    "configure_arrayrecord_lane",
    "configure_parquetra_lane",
    "get_read_rate_limiter",
    "lane_fingerprint",
    "lane_state_path",
    "load_fast_encoder_factory",
    "read_lane_cursor",
    "read_rate_bytes_per_second",
    "resolve_lane_assistant_only",
    "write_lane_cursor",
)
