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

"""Offline teacher cache: content-keyed teacher top-k next-token distributions per training row.

A dump pass runs a frozen teacher once over the training rows and stores, per tokenized row, the
teacher's top-k token ids and log-probabilities (temperature 1, normalised over the FULL vocabulary)
at every position plus the log tail mass ``log(1 - sum_k p_k)``. Distillation then reads the cache
instead of running the teacher. Records are keyed by the blake2b-128 digest of the row's ``input_ids``
(post chat template, post truncation -- exactly the token list the packed collator materialises), so
a training run joins the cache by content, independent of order, sharding or mixing.

The tail is computed as ``lse(non-top-k) - lse(all)`` rather than ``log(1 - sum p)``, which stays
exact when the top-k mass is within float32 epsilon of 1.

Layout on disk (local path or ``gs://``, through :data:`eformer.paths.ePath`)::

    <root>/MANIFEST.json                 written LAST by one process; a reader refuses a root without it
    <root>/host-<hhhh>/part-<nnnnn>.npz  one writer per host prefix -> disjoint objects, no coordination
    <root>/host-<hhhh>/DONE.json         per-host totals, written on close

A part is an ``.npz`` whose small members (``keys``, ``n_tokens``, ``row_offsets``, ``layout``) are the
index and whose flat token-major members (``ids``, ``supervised``, ``topk_idx``, ``topk_logp``,
``tail_logp``) are the payload; ``np.load`` reads zip members lazily, so building the key index touches
only the small members. ``layout`` records where the dump saw each row ``(step, window, offset)``.

Pieces:

* :func:`row_key` -- the record key.
* :class:`TeacherCacheWriter` -- the dump side (one per host; parts rotate at ``part_bytes``) and
  :func:`write_manifest` to seal a finished dump.
* :class:`TeacherCacheReader` -- the training side: manifest contract check, key index, LRU of decoded
  parts, :meth:`TeacherCacheReader.fill_windows` to place records on a ``(windows, seq_len)`` grid. A
  missing row raises :class:`TeacherCacheMiss`; there is no fallback.
* :func:`layout_sidecar` / :func:`records_from_windows` -- the per-window ``(key, offset, n_tokens)``
  sidecar a packed collator attaches, and its inverse for the dump.
* :func:`teacher_topk_from_logits` / :func:`chunked_teacher_topk` -- the device-side dump math, including
  the vocabulary-parallel (``shard_map``) variant.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import typing as tp
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
from eformer.paths import ePath

MANIFEST_NAME = "MANIFEST.json"
DONE_NAME = "DONE.json"
KEY_BYTES = 16
CACHE_FORMAT_VERSION = 1

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = ("teacher_uri", "tokenizer", "k", "segment_align")

LAYOUT_KEYS_FIELD = "teacher_layout_keys"
LAYOUT_SPAN_FIELD = "teacher_layout_span"

EMPTY_TAIL_LOGP = -88.0


class TeacherCacheMiss(KeyError):
    """A row the cache does not hold. Fatal by design: training must not silently continue without it."""


class TeacherCacheMismatch(ValueError):
    """The cache was produced under different assumptions than this run (teacher, tokenizer, alignment, k)."""


def row_key(input_ids: tp.Sequence[int] | np.ndarray) -> bytes:
    """Return the record key of a row: blake2b-128 of its tokens as little-endian int32 bytes.

    Args:
        input_ids: The row's 1-D token ids.

    Returns:
        bytes: 16-byte digest. Depends on tokens only (not labels, not the source dataset).

    Raises:
        ValueError: If ``input_ids`` is not 1-D.
    """
    ids = np.ascontiguousarray(np.asarray(input_ids, dtype="<i4"))
    if ids.ndim != 1:
        raise ValueError(f"row_key expects a 1-D token list, got shape {ids.shape}")
    return hashlib.blake2b(ids.tobytes(), digest_size=KEY_BYTES).digest()


def key_to_array(key: bytes) -> np.ndarray:
    """Return ``key`` as a ``uint8[16]`` array (the on-disk / sidecar representation)."""
    return np.frombuffer(key, dtype=np.uint8).copy()


def key_hex(key: bytes | np.ndarray) -> str:
    """Return the hex string of a key given as bytes or a ``uint8`` array."""
    if isinstance(key, np.ndarray):
        key = key.astype(np.uint8).tobytes()
    return key.hex()


def _join(root: str, *parts: str) -> str:
    return str(root).rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)


def _write_bytes(path: str, data: bytes) -> None:
    ePath(path).write_bytes(data)


def _read_bytes(path: str) -> bytes:
    return ePath(path).read_bytes()


def _exists(path: str) -> bool:
    return bool(ePath(path).exists())


def host_prefix(host_index: int) -> str:
    """Directory name of one host's records (``host-0003``)."""
    return f"host-{int(host_index):04d}"


def part_name(part_index: int) -> str:
    """File name of one part (``part-00012.npz``)."""
    return f"part-{int(part_index):05d}.npz"


class TeacherCacheWriter:
    """Append records for ONE host under ``<root>/host-<h>/``; parts rotate at ``part_bytes`` of payload.

    Writers of different hosts touch disjoint objects, so a multi-host dump needs no coordination until
    :func:`write_manifest`.

    Args:
        root: Cache root (local path or ``gs://`` URI).
        host_index: This process's index; the only thing that keeps writers disjoint.
        k: Top-k width every record must carry.
        part_bytes: Buffered payload bytes per part before it is flushed (default 256 MiB).
    """

    def __init__(self, root: str, *, host_index: int, k: int, part_bytes: int = 256 << 20):
        if int(k) <= 0:
            raise ValueError("k must be positive")
        if int(part_bytes) <= 0:
            raise ValueError("part_bytes must be positive")
        self.root = str(root)
        self.host_index = int(host_index)
        self.k = int(k)
        self.part_bytes = int(part_bytes)
        self._part_index = 0
        self.rows_written = 0
        self.tokens_written = 0
        self.parts_written: list[str] = []
        self._closed = False
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._keys: list[np.ndarray] = []
        self._n_tokens: list[int] = []
        self._layout: list[tuple[int, int, int]] = []
        self._ids: list[np.ndarray] = []
        self._sup: list[np.ndarray] = []
        self._idx: list[np.ndarray] = []
        self._logp: list[np.ndarray] = []
        self._tail: list[np.ndarray] = []
        self._buffered_bytes = 0

    def append(
        self,
        input_ids: np.ndarray,
        supervised: np.ndarray,
        topk_idx: np.ndarray,
        topk_logp: np.ndarray,
        tail_logp: np.ndarray,
        *,
        step: int = -1,
        window: int = -1,
        offset: int = -1,
    ) -> bytes:
        """Buffer one row's record and return its key.

        Args:
            input_ids: ``[n]`` row tokens.
            supervised: ``[n]`` supervised mask (stored as ``uint8``).
            topk_idx: ``[n, k]`` teacher top-k token ids.
            topk_logp: ``[n, k]`` full-vocabulary log-probs of those ids (stored as ``float16``).
            tail_logp: ``[n]`` log tail mass (stored as ``float16``).
            step: Optional dump step the row was seen at.
            window: Optional window index inside that step.
            offset: Optional token offset inside that window.

        Returns:
            bytes: The record key (:func:`row_key` of ``input_ids``).

        Raises:
            RuntimeError: If the writer is closed.
            ValueError: On an empty row or disagreeing shapes.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        ids = np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32))
        n = int(ids.shape[0]) if ids.ndim == 1 else -1
        if n <= 0:
            raise ValueError(f"expected a non-empty 1-D row, got shape {ids.shape}")
        sup = np.ascontiguousarray(np.asarray(supervised).astype(np.uint8))
        idx = np.ascontiguousarray(np.asarray(topk_idx, dtype=np.int32))
        logp = np.ascontiguousarray(np.asarray(topk_logp).astype(np.float16))
        tail = np.ascontiguousarray(np.asarray(tail_logp).astype(np.float16))
        if sup.shape != (n,) or idx.shape != (n, self.k) or logp.shape != (n, self.k) or tail.shape != (n,):
            raise ValueError(
                f"record shapes disagree: ids {ids.shape}, supervised {sup.shape}, topk_idx {idx.shape}, "
                f"topk_logp {logp.shape}, tail {tail.shape}; expected n={n}, k={self.k}"
            )
        key = row_key(ids)
        self._keys.append(key_to_array(key))
        self._n_tokens.append(n)
        self._layout.append((int(step), int(window), int(offset)))
        self._ids.append(ids)
        self._sup.append(sup)
        self._idx.append(idx)
        self._logp.append(logp)
        self._tail.append(tail)
        self._buffered_bytes += ids.nbytes + sup.nbytes + idx.nbytes + logp.nbytes + tail.nbytes
        self.rows_written += 1
        self.tokens_written += n
        if self._buffered_bytes >= self.part_bytes:
            self.flush()
        return key

    def flush(self) -> str | None:
        """Write the buffered records as one part.

        Returns:
            str | None: The part path, or ``None`` when nothing was buffered.
        """
        if not self._keys:
            return None
        n_tokens = np.asarray(self._n_tokens, dtype=np.int32)
        row_offsets = np.zeros(len(n_tokens) + 1, dtype=np.int64)
        np.cumsum(n_tokens, out=row_offsets[1:])
        payload = {
            "format_version": np.asarray(CACHE_FORMAT_VERSION, dtype=np.int32),
            "k": np.asarray(self.k, dtype=np.int32),
            "host_index": np.asarray(self.host_index, dtype=np.int32),
            "keys": np.stack(self._keys).astype(np.uint8),
            "n_tokens": n_tokens,
            "row_offsets": row_offsets,
            "layout": np.asarray(self._layout, dtype=np.int32).reshape(-1, 3),
            "ids": np.concatenate(self._ids),
            "supervised": np.concatenate(self._sup),
            "topk_idx": np.concatenate(self._idx, axis=0),
            "topk_logp": np.concatenate(self._logp, axis=0),
            "tail_logp": np.concatenate(self._tail),
        }
        buf = io.BytesIO()
        np.savez(buf, **payload)
        rel = _join(host_prefix(self.host_index), part_name(self._part_index)).lstrip("/")
        path = _join(self.root, rel)
        _write_bytes(path, buf.getvalue())
        self.parts_written.append(rel)
        self._part_index += 1
        self._reset_buffers()
        return path

    def close(self) -> dict:
        """Flush and write this host's ``DONE.json``.

        Returns:
            dict: The ``DONE.json`` contents (host index, k, rows, tokens, part list).

        Raises:
            RuntimeError: If the writer was already closed.
        """
        if self._closed:
            raise RuntimeError("writer already closed")
        self.flush()
        done = {
            "host_index": self.host_index,
            "k": self.k,
            "rows": int(self.rows_written),
            "tokens": int(self.tokens_written),
            "parts": list(self.parts_written),
            "closed_at_unix": time.time(),
        }
        _write_bytes(_join(self.root, host_prefix(self.host_index), DONE_NAME), json.dumps(done, indent=1).encode())
        self._closed = True
        return done


def write_manifest(
    root: str,
    *,
    meta: dict,
    host_count: int,
    wait_seconds: float = 0.0,
    poll_seconds: float = 5.0,
) -> dict:
    """Collect every host's ``DONE.json`` and write ``MANIFEST.json`` (the commit point of a dump).

    Args:
        root: Cache root.
        meta: Reader contract fields (:data:`MANIFEST_REQUIRED_FIELDS`); anything else is stored verbatim.
        host_count: Number of writer hosts expected.
        wait_seconds: How long to wait for missing hosts before refusing.
        poll_seconds: Upper bound of the sleep between polls while waiting.

    Returns:
        dict: The manifest written.

    Raises:
        ValueError: If ``meta`` lacks a contract field or a host's ``k`` disagrees with ``meta["k"]``.
        RuntimeError: If some host never wrote ``DONE.json`` within ``wait_seconds``.
    """
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in meta:
            raise ValueError(f"manifest meta lacks {field!r}")
    deadline = time.time() + float(wait_seconds)
    hosts: dict[int, dict] = {}
    while True:
        missing = []
        for h in range(int(host_count)):
            if h in hosts:
                continue
            path = _join(root, host_prefix(h), DONE_NAME)
            if not _exists(path):
                missing.append(h)
                continue
            hosts[h] = json.loads(_read_bytes(path).decode())
        if not missing:
            break
        if time.time() >= deadline:
            more = "..." if len(missing) > 8 else ""
            raise RuntimeError(f"teacher cache {root}: hosts {missing[:8]}{more} never wrote {DONE_NAME}")
        time.sleep(min(float(poll_seconds), max(0.05, deadline - time.time())))
    bad_k = {h: d["k"] for h, d in hosts.items() if int(d["k"]) != int(meta["k"])}
    if bad_k:
        raise ValueError(f"hosts wrote k={bad_k}, manifest meta says k={meta['k']}")
    manifest = {
        **meta,
        "format_version": CACHE_FORMAT_VERSION,
        "host_count": int(host_count),
        "rows": int(sum(d["rows"] for d in hosts.values())),
        "tokens": int(sum(d["tokens"] for d in hosts.values())),
        "parts": [p for h in sorted(hosts) for p in hosts[h]["parts"]],
        "written_at_unix": time.time(),
    }
    _write_bytes(_join(root, MANIFEST_NAME), json.dumps(manifest, indent=1).encode())
    return manifest


class TeacherCacheReader:
    """Key -> record lookup over a finished cache, with the manifest contract checked up front.

    Args:
        root: Cache root holding ``MANIFEST.json``.
        expect: Contract fields that must equal the manifest's (e.g. ``teacher_uri``, ``tokenizer``,
            ``segment_align``); compared as strings.
        topk: The k this run trains on (``0`` = the cache's k). Must be ``<=`` the cache's k.
        part_cache: Number of decoded parts kept in memory (LRU).

    Raises:
        TeacherCacheMismatch: When the manifest is missing, has another format version or k, or an
            ``expect`` field differs.
    """

    def __init__(self, root: str, *, expect: dict | None = None, topk: int = 0, part_cache: int = 2):
        self.root = str(root)
        mpath = _join(self.root, MANIFEST_NAME)
        if not _exists(mpath):
            raise TeacherCacheMismatch(
                f"teacher cache {self.root} has no {MANIFEST_NAME}: the dump did not finish (or never ran)"
            )
        self.manifest = json.loads(_read_bytes(mpath).decode())
        if int(self.manifest.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise TeacherCacheMismatch(
                f"cache format_version {self.manifest.get('format_version')} != {CACHE_FORMAT_VERSION}"
            )
        self.k_cache = int(self.manifest["k"])
        self.topk = int(topk) if int(topk) > 0 else self.k_cache
        if self.topk > self.k_cache:
            raise TeacherCacheMismatch(f"run wants top-{self.topk} but the cache holds k={self.k_cache}")
        for field, want in (expect or {}).items():
            have = self.manifest.get(field)
            if str(have) != str(want):
                raise TeacherCacheMismatch(f"cache {field}={have!r} != run {field}={want!r} (root {self.root})")
        self.parts: list[str] = list(self.manifest["parts"])
        self._index: dict[bytes, tuple[int, int]] = {}
        self._decoded: OrderedDict[int, dict] = OrderedDict()
        self._part_cache = max(int(part_cache), 1)
        self._build_index()

    def _open_part(self, part_i: int):
        return np.load(io.BytesIO(_read_bytes(_join(self.root, self.parts[part_i]))), allow_pickle=False)

    def _build_index(self) -> None:
        for i in range(len(self.parts)):
            npz = self._open_part(i)
            if int(npz["k"]) != self.k_cache:
                raise TeacherCacheMismatch(f"part {self.parts[i]} has k={int(npz['k'])}, manifest says {self.k_cache}")
            keys = np.asarray(npz["keys"], dtype=np.uint8)
            for r in range(keys.shape[0]):
                # Duplicates (identical rows) resolve to the FIRST record: identical tokens, identical teacher.
                self._index.setdefault(keys[r].tobytes(), (i, r))

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: bytes) -> bool:
        return key in self._index

    def _part(self, part_i: int) -> dict:
        got = self._decoded.get(part_i)
        if got is not None:
            self._decoded.move_to_end(part_i)
            return got
        npz = self._open_part(part_i)
        names = ("row_offsets", "ids", "supervised", "topk_idx", "topk_logp", "tail_logp")
        got = {name: np.asarray(npz[name]) for name in names}
        self._decoded[part_i] = got
        while len(self._decoded) > self._part_cache:
            self._decoded.popitem(last=False)
        return got

    def get(self, key: bytes, *, input_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Return the record for ``key``, verifying the stored tokens re-hash to it.

        Args:
            key: Record key (:func:`row_key`).
            input_ids: Optional tokens the record must equal (guards hash collisions and truncation changes).

        Returns:
            dict[str, np.ndarray]: ``input_ids``, ``supervised``, ``topk_idx`` ``[n, topk]``, ``topk_logp``
            ``[n, topk]`` and ``tail_logp`` ``[n]``. When ``topk < k_cache`` the tail is re-derived from the
            kept log-probs.

        Raises:
            TeacherCacheMiss: When the cache holds no record for ``key``.
            TeacherCacheMismatch: On a corrupt record or tokens differing from ``input_ids``.
        """
        hit = self._index.get(key)
        if hit is None:
            detail = ""
            if input_ids is not None:
                detail = f" (row of {len(input_ids)} tokens, first {list(np.asarray(input_ids)[:8])})"
            raise TeacherCacheMiss(f"teacher cache {self.root} has no record for key {key.hex()}{detail}")
        part_i, r = hit
        part = self._part(part_i)
        a, b = int(part["row_offsets"][r]), int(part["row_offsets"][r + 1])
        ids = part["ids"][a:b]
        if row_key(ids) != key:
            raise TeacherCacheMismatch(
                f"corrupt record {key.hex()} in {self.parts[part_i]}: stored tokens do not re-hash to the key"
            )
        if input_ids is not None and (
            len(input_ids) != len(ids) or not np.array_equal(np.asarray(input_ids, np.int32), ids)
        ):
            raise TeacherCacheMismatch(f"record {key.hex()} tokens differ from the row (hash collision or truncation)")
        idx = part["topk_idx"][a:b, : self.topk]
        logp = part["topk_logp"][a:b, : self.topk]
        if self.topk == self.k_cache:
            tail = part["tail_logp"][a:b]
        else:
            # The stored tail belongs to the full k; re-derive it in fp32 from the kept log-probs.
            mass = np.sum(np.exp(logp.astype(np.float32)), axis=-1)
            tail = np.log(np.clip(1.0 - mass, 1e-7, 1.0)).astype(np.float16)
        return {
            "input_ids": ids,
            "supervised": part["supervised"][a:b],
            "topk_idx": idx,
            "topk_logp": logp,
            "tail_logp": tail,
        }

    def fill_windows(
        self,
        layout_keys: np.ndarray,
        layout_span: np.ndarray,
        *,
        seq_len: int,
        input_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Place records on a ``(windows, seq_len)`` grid from a collator's layout sidecar.

        Args:
            layout_keys: ``uint8[W, D, 16]`` record keys per window / document slot (unused slots carry
                ``layout_span[..., 1] == 0``).
            layout_span: ``int32[W, D, 2]`` ``(offset, n_tokens)`` per slot.
            seq_len: Grid width.
            input_ids: Optional ``[W, seq_len]`` tokens to verify each record against its cells.

        Returns:
            dict[str, np.ndarray]: ``teacher_topk_idx int32[W, S, k]``, ``teacher_topk_logp float32[W, S, k]``
            and ``teacher_tail_logp float32[W, S]``; pad cells are zero (they carry no loss mask).

        Raises:
            TeacherCacheMiss: When any slot's record is missing.
            TeacherCacheMismatch: When a record's length disagrees with its slot.
        """
        layout_keys = np.asarray(layout_keys, dtype=np.uint8)
        layout_span = np.asarray(layout_span, dtype=np.int32)
        n_windows = int(layout_keys.shape[0])
        idx = np.zeros((n_windows, seq_len, self.topk), dtype=np.int32)
        logp = np.zeros((n_windows, seq_len, self.topk), dtype=np.float32)
        tail = np.zeros((n_windows, seq_len), dtype=np.float32)
        for w in range(n_windows):
            for d in range(layout_keys.shape[1]):
                off, n = int(layout_span[w, d, 0]), int(layout_span[w, d, 1])
                if n <= 0:
                    continue
                if off < 0 or off + n > seq_len:
                    raise TeacherCacheMismatch(f"window {w} slot {d} span ({off}, {n}) exceeds seq_len={seq_len}")
                key = layout_keys[w, d].tobytes()
                cells = None if input_ids is None else np.asarray(input_ids[w, off : off + n])
                rec = self.get(key, input_ids=cells)
                if rec["input_ids"].shape[0] != n:
                    raise TeacherCacheMismatch(
                        f"record {key.hex()} has {rec['input_ids'].shape[0]} tokens, the window slot says {n}"
                    )
                idx[w, off : off + n] = rec["topk_idx"]
                logp[w, off : off + n] = rec["topk_logp"].astype(np.float32)
                tail[w, off : off + n] = rec["tail_logp"].astype(np.float32)
        return {"teacher_topk_idx": idx, "teacher_topk_logp": logp, "teacher_tail_logp": tail}


def layout_sidecar(
    rows: tp.Sequence[tp.Mapping[str, tp.Any]],
    placement: tp.Sequence[tp.Sequence[int]],
    *,
    n_windows: int,
    align: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(layout_keys, layout_span)`` sidecar of a collated window set.

    Args:
        rows: The rows of the batch.
        placement: ``placement[w]`` lists row indices in window order.
        n_windows: Static window count (windows beyond ``len(placement)`` stay empty).
        align: Each row's footprint is its length rounded up to this multiple, so offsets match a
            materialiser that aligns document starts.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``uint8[W, D, 16]`` keys and ``int32[W, D, 2]`` ``(offset, n)`` spans.
    """
    if int(align) < 1:
        raise ValueError("align must be >= 1")
    if len(placement) > n_windows:
        raise ValueError(f"{len(placement)} windows placed but n_windows={n_windows}")
    slots = max(max((len(p) for p in placement), default=0), 1)
    keys = np.zeros((n_windows, slots, KEY_BYTES), dtype=np.uint8)
    span = np.zeros((n_windows, slots, 2), dtype=np.int32)
    for w, indices in enumerate(placement):
        offset = 0
        for d, i in enumerate(indices):
            ids = np.asarray(rows[i]["input_ids"], dtype=np.int32)
            n = int(ids.shape[0])
            keys[w, d] = key_to_array(row_key(ids))
            span[w, d] = (offset, n)
            offset += -(-n // int(align)) * int(align)
    return keys, span


def records_from_windows(
    batch: tp.Mapping[str, np.ndarray],
    topk_idx: np.ndarray,
    topk_logp: np.ndarray,
    tail_logp: np.ndarray,
    *,
    window_indices: tp.Iterable[int],
    step: int = -1,
) -> tp.Iterator[dict]:
    """Cut a window set's dump output back into per-row records using the batch's layout sidecar.

    Args:
        batch: Collated batch carrying ``input_ids``, ``labels`` and the two layout sidecar fields.
        topk_idx: ``[W, S, k]`` dump output.
        topk_logp: ``[W, S, k]`` dump output.
        tail_logp: ``[W, S]`` dump output.
        window_indices: Windows to emit (e.g. this host's addressable windows).
        step: Dump step recorded in each record's layout.

    Yields:
        dict: Keyword arguments for :meth:`TeacherCacheWriter.append` (``supervised`` from
        ``labels != -100``).

    Raises:
        ValueError: When a sidecar key does not match the grid tokens at its span.
    """
    keys = np.asarray(batch[LAYOUT_KEYS_FIELD])
    span = np.asarray(batch[LAYOUT_SPAN_FIELD])
    ids = np.asarray(batch["input_ids"])
    labels = np.asarray(batch["labels"])
    for w in window_indices:
        for d in range(span.shape[1]):
            off, n = int(span[w, d, 0]), int(span[w, d, 1])
            if n <= 0:
                continue
            row_ids = ids[w, off : off + n]
            if row_key(row_ids) != keys[w, d].tobytes():
                raise ValueError(f"layout key at window {w} slot {d} does not match the grid tokens")
            yield {
                "input_ids": row_ids,
                "supervised": (labels[w, off : off + n] != -100).astype(np.uint8),
                "topk_idx": np.asarray(topk_idx[w, off : off + n]),
                "topk_logp": np.asarray(topk_logp[w, off : off + n]),
                "tail_logp": np.asarray(tail_logp[w, off : off + n]),
                "step": step,
                "window": int(w),
                "offset": off,
            }


def teacher_topk_from_logits(
    logits: jax.Array,
    k: int,
    *,
    vocab_offset: jax.Array | int = 0,
    vocab_axis_name: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Top-k ids, their full-vocabulary log-probs and the log tail mass from (possibly sharded) logits.

    Inside ``shard_map`` (``vocab_axis_name`` set) each member holds a vocabulary slice starting at
    ``vocab_offset``: a local ``top_k``, an ``all_gather`` of the candidates, a second ``top_k`` over them,
    the logsumexp via ``pmax``/``psum``, and the tail as the logsumexp of every NON-selected local entry,
    ``psum``-combined. ``log(1 - sum p)`` would cancel to 0 in fp32 once the top-k mass passes ``1 - 1e-7``;
    this formulation does not.

    Args:
        logits: ``[..., V_local]`` logits.
        k: Top-k width.
        vocab_offset: Global id of this member's first vocabulary entry.
        vocab_axis_name: Mesh axis the vocabulary is sharded over, or ``None`` for unsharded logits.

    Returns:
        tuple: ``(idx int32[..., k], logp float32[..., k], tail_logp float32[...])``. An empty tail is
        reported as :data:`EMPTY_TAIL_LOGP`.

    Raises:
        ValueError: When ``k`` exceeds an unsharded vocabulary.
    """
    x = logits.astype(jnp.float32)
    v_local = int(x.shape[-1])
    kk = min(int(k), v_local)
    if vocab_axis_name is None and kk < int(k):
        raise ValueError(f"k={k} exceeds the vocabulary {v_local}")
    offset = jnp.asarray(vocab_offset, jnp.int32)
    local_vals, local_idx = jax.lax.top_k(x, kk)
    local_idx = local_idx.astype(jnp.int32) + offset
    m = jnp.max(x, axis=-1)
    if vocab_axis_name is not None:
        m = jax.lax.pmax(m, axis_name=vocab_axis_name)
    z = jnp.sum(jnp.exp(x - m[..., None]), axis=-1)
    if vocab_axis_name is not None:
        z = jax.lax.psum(z, axis_name=vocab_axis_name)
    lse = m + jnp.log(z)
    if vocab_axis_name is not None:
        cand_vals = jax.lax.all_gather(local_vals, vocab_axis_name, axis=-1, tiled=True)
        cand_idx = jax.lax.all_gather(local_idx, vocab_axis_name, axis=-1, tiled=True)
        vals, pos = jax.lax.top_k(cand_vals, int(k))
        idx = jnp.take_along_axis(cand_idx, pos, axis=-1)
    else:
        vals, idx = local_vals, local_idx
    # Tail: mask the selected ids living on this member, logsumexp the rest, combine over the vocab axis.
    loc = idx - offset
    hit = (loc >= 0) & (loc < v_local)
    onehot = jax.nn.one_hot(jnp.where(hit, loc, 0), v_local, dtype=jnp.float32) * hit[..., None].astype(jnp.float32)
    selected = jnp.sum(onehot, axis=-2) > 0.5
    rest = jnp.where(selected, -jnp.inf, x)
    m_rest = jnp.max(rest, axis=-1)
    if vocab_axis_name is not None:
        m_rest = jax.lax.pmax(m_rest, axis_name=vocab_axis_name)
    m_rest_safe = jnp.where(jnp.isfinite(m_rest), m_rest, 0.0)
    z_rest = jnp.sum(jnp.where(selected, 0.0, jnp.exp(rest - m_rest_safe[..., None])), axis=-1)
    if vocab_axis_name is not None:
        z_rest = jax.lax.psum(z_rest, axis_name=vocab_axis_name)
    tail_lse = m_rest_safe + jnp.log(jnp.maximum(z_rest, 1e-38))
    tail = jnp.where(z_rest > 0.0, tail_lse - lse, jnp.float32(EMPTY_TAIL_LOGP))
    return idx, vals - lse[..., None], tail


def chunked_teacher_topk(
    hidden: jax.Array,
    lm_head_fn: tp.Callable[[jax.Array], jax.Array],
    k: int,
    *,
    chunk_size: int,
    vp_mesh: tp.Any = None,
    vp_logit_spec: tp.Any = None,
    vp_token_spec: tp.Any = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dump pass over one window set: ``[B, S, H]`` hidden -> ``(idx int32, logp f16, tail f16)``.

    Runs ``lm_head_fn`` per ``chunk_size`` positions inside ``lax.scan`` so no ``[S, V]`` tensor is
    materialised. With a vocabulary-sharded logit spec (``vp_logit_spec[-1]`` names a mesh axis) the top-k
    runs under ``jax.shard_map`` via :func:`teacher_topk_from_logits`.

    Args:
        hidden: ``[B, S, H]`` final hidden states.
        lm_head_fn: Maps ``[B, chunk, H]`` hidden to ``[B, chunk, V]`` logits.
        k: Top-k width.
        chunk_size: Positions per scan step.
        vp_mesh: Mesh for the vocabulary-parallel path (``None`` = unsharded).
        vp_logit_spec: ``PartitionSpec`` of the ``[B, chunk, V]`` logits.
        vp_token_spec: ``PartitionSpec`` whose first two entries shard ``[B, chunk]``.

    Returns:
        tuple: ``(idx int32[B, S, k], logp float16[B, S, k], tail float16[B, S])``.
    """
    from jax.sharding import PartitionSpec

    batch, seq = int(hidden.shape[0]), int(hidden.shape[1])
    pad = (-seq) % int(chunk_size)
    if pad:
        hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
    n_chunks = (seq + pad) // int(chunk_size)
    chunks = hidden.reshape(batch, n_chunks, int(chunk_size), -1).transpose(1, 0, 2, 3)
    use_vp = vp_mesh is not None and vp_logit_spec is not None and isinstance(vp_logit_spec[-1], str)

    def _one(h):
        logits = lm_head_fn(h)
        if not use_vp:
            idx, logp, tail = teacher_topk_from_logits(logits, k)
        else:
            vocab_axis = vp_logit_spec[-1]
            token_spec = PartitionSpec(*tuple(vp_token_spec)[:2])
            topk_spec = PartitionSpec(*tuple(token_spec), None)
            v_local = int(logits.shape[-1]) // int(vp_mesh.shape[vocab_axis])

            def body(x):
                offset = jax.lax.axis_index(vocab_axis) * v_local
                return teacher_topk_from_logits(x, k, vocab_offset=offset, vocab_axis_name=vocab_axis)

            # Outputs are replicated after the all_gather, but VMA inference still marks them varying.
            idx, logp, tail = jax.shard_map(
                body,
                mesh=vp_mesh,
                in_specs=(vp_logit_spec,),
                out_specs=(topk_spec, topk_spec, token_spec),
                check_vma=False,
            )(logits)
        return idx.astype(jnp.int32), logp.astype(jnp.float16), tail.astype(jnp.float16)

    _, (idx, logp, tail) = jax.lax.scan(lambda c, h: (c, _one(h)), 0, chunks)
    idx = idx.transpose(1, 0, 2, 3).reshape(batch, seq + pad, -1)[:, :seq]
    logp = logp.transpose(1, 0, 2, 3).reshape(batch, seq + pad, -1)[:, :seq]
    tail = tail.transpose(1, 0, 2).reshape(batch, seq + pad)[:, :seq]
    return idx, logp, tail


__all__ = (
    "CACHE_FORMAT_VERSION",
    "DONE_NAME",
    "EMPTY_TAIL_LOGP",
    "KEY_BYTES",
    "LAYOUT_KEYS_FIELD",
    "LAYOUT_SPAN_FIELD",
    "MANIFEST_NAME",
    "MANIFEST_REQUIRED_FIELDS",
    "TeacherCacheMismatch",
    "TeacherCacheMiss",
    "TeacherCacheReader",
    "TeacherCacheWriter",
    "chunked_teacher_topk",
    "host_prefix",
    "key_hex",
    "key_to_array",
    "layout_sidecar",
    "part_name",
    "records_from_windows",
    "row_key",
    "teacher_topk_from_logits",
    "write_manifest",
)
