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
"""Per-dataset assistant-only supervision mask assembly shared by SFT-style trainers.

A mixture may choose assistant-only supervision per dataset (``mixture.informs[].assistant_only`` on
the eLarge sharded lane, ``parquetra_assistant_only`` / ``arrayrecord_assistant_only`` on the grain
lanes). The choice is stamped on every row as the tri-state column
:data:`~easydel.data.ASSISTANT_ONLY_FIELD` (``1`` opt in, ``0`` opt out, ``-1`` inherit the trainer's
global ``assistant_only_loss``). Two carriers reach the trainer:

* ``_assistant_only`` -- one value per ROW (unpacked batches);
* ``assistant_only_mask`` -- one value per TOKEN, emitted by
  :func:`~easydel.data.transforms.collators.collate_packed_embeds` so co-packed documents keep their own
  choices.

Dataset attribution (``_dataset_id`` per row, ``dataset_ids`` per token) rides the same two lanes and is
used only for the zero-supervision census. All four carriers are popped before the batch reaches the
model: model forwards accept ``**kwargs``, so a leaked carrier would be silent.

Mask sources, in priority order: an explicit ``completion_mask`` column, then supervision-bearing
``-100`` labels (segment-aware under packing), then assistant spans scanned from the token ids. The
span markers come from :attr:`TrainingArguments.assistant_header_text` /
:attr:`TrainingArguments.assistant_end_text` when set, otherwise they are derived from the tokenizer's
chat template (the generation prompt is the header, the text the template appends after an assistant
turn is the terminator).

When no batch carries a per-dataset choice the legacy path
(:func:`~easydel.trainers.training_utils.normalize_completion_masks_and_labels`) runs unchanged.
"""

from __future__ import annotations

import dataclasses
import time
import typing as tp

import numpy as np
from eformer.loggings import get_logger

from easydel.data.transforms.mixture import (
    ASSISTANT_ONLY_FIELD,
    ASSISTANT_ONLY_TOKEN_FIELD,
    DATASET_ID_FIELD,
    DATASET_ID_TOKEN_FIELD,
)

from .training_utils import normalize_completion_masks_and_labels

logger = get_logger(__name__)

SIDEBAND_CARRIER_FIELDS: tuple[str, ...] = (
    ASSISTANT_ONLY_FIELD,
    ASSISTANT_ONLY_TOKEN_FIELD,
    DATASET_ID_FIELD,
    DATASET_ID_TOKEN_FIELD,
)

_HEADER_SENTINEL_USER = "EASYDEL_USER_SENTINEL_7f3a"
_HEADER_SENTINEL_ASSISTANT = "EASYDEL_ASSISTANT_SENTINEL_7f3a"


@dataclasses.dataclass(frozen=True)
class AssistantSpanIds:
    """Token-id signature of an assistant span.

    Attributes:
        header_ids: Ids of the assistant header (the text right before assistant content).
        end_id: Id that closes an assistant turn (included in the span).
        newline_family: Alternate ids accepted at the header's final position when the header ends
            with a newline that BPE may fuse with newline-leading content.
    """

    header_ids: tuple[int, ...]
    end_id: int
    newline_family: tuple[int, ...] = ()


def _encode(tokenizer: tp.Any, text: str) -> list[int]:
    """Encode ``text`` without special tokens, retrying the fast-tokenizer borrow race."""
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
        except Exception as exc:  # Rust tokenizers raise "Already borrowed" under thread contention.
            last_exc = exc
            if "borrowed" in str(exc).lower() and attempt < 4:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    raise RuntimeError("tokenizer encode kept hitting the borrow race") from last_exc


def _template_header_and_end(tokenizer: tp.Any) -> tuple[str, str]:
    """Derive ``(header_text, end_text)`` from the tokenizer's chat template.

    Raises:
        ValueError: When the template does not expose a generation prompt or the terminator cannot be
            located.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("the tokenizer has no chat template")
    user = [{"role": "user", "content": _HEADER_SENTINEL_USER}]
    without = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    if not (isinstance(without, str) and isinstance(with_prompt, str)) or not with_prompt.startswith(without):
        raise ValueError("the generation prompt is not a suffix of the rendered conversation")
    header = with_prompt[len(without) :]
    if not header.strip():
        raise ValueError("the chat template renders an empty generation prompt")
    full = tokenizer.apply_chat_template(
        [*user, {"role": "assistant", "content": _HEADER_SENTINEL_ASSISTANT}],
        tokenize=False,
        add_generation_prompt=False,
    )
    pos = full.rfind(_HEADER_SENTINEL_ASSISTANT)
    if pos < 0:
        raise ValueError("the assistant turn content does not appear in the rendered template")
    end = full[pos + len(_HEADER_SENTINEL_ASSISTANT) :].strip()
    if not end:
        end = getattr(tokenizer, "eos_token", None) or ""
    if not end:
        raise ValueError("the chat template appends no terminator after an assistant turn and no eos_token is set")
    return header, end


def derive_assistant_span_ids(
    tokenizer: tp.Any,
    *,
    header_text: str | None = None,
    end_text: str | None = None,
) -> AssistantSpanIds:
    """Resolve the assistant-span token signature for ``tokenizer``.

    Args:
        tokenizer: HF tokenizer (or processor exposing ``.tokenizer``).
        header_text: Explicit assistant header text; ``None`` derives it from the chat template.
        end_text: Explicit assistant terminator text; ``None`` derives it from the chat template.

    Returns:
        AssistantSpanIds: Header ids, terminator id and newline family.

    Raises:
        RuntimeError: When the signature cannot be derived. A requested mask must never silently
            degrade to full-sequence supervision.
    """
    if tokenizer is not None and hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "encode"):
        tokenizer = tokenizer.tokenizer
    if tokenizer is None:
        raise RuntimeError("assistant-span ids need a tokenizer; none is attached to the trainer.")
    try:
        if header_text is None or end_text is None:
            derived_header, derived_end = _template_header_and_end(tokenizer)
            header_text = derived_header if header_text is None else header_text
            end_text = derived_end if end_text is None else end_text
        header = tuple(_encode(tokenizer, header_text))
        end = _encode(tokenizer, end_text)
    except Exception as exc:
        raise RuntimeError(
            "assistant-only masking needs the assistant-span markers, but they could not be derived from the "
            f"tokenizer ({type(exc).__name__}: {exc}). Set `assistant_header_text` / `assistant_end_text` in the "
            "training arguments, or provide an explicit completion_mask / -100 labels."
        ) from exc
    if not header or not end:
        raise RuntimeError(
            f"assistant-span markers encode to nothing (header={header_text!r}, end={end_text!r}); set "
            "`assistant_header_text` / `assistant_end_text` explicitly."
        )
    family: set[int] = set()
    if header_text.endswith("\n"):
        for repeat in range(1, 9):
            try:
                toks = _encode(tokenizer, "\n" * repeat)
            except Exception:
                break
            if toks:
                family.add(int(toks[0]))
    return AssistantSpanIds(header_ids=header, end_id=int(end[0]), newline_family=tuple(sorted(family)))


def assistant_mask_from_token_ids(
    input_ids: np.ndarray,
    span_ids: AssistantSpanIds,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Build a ``[batch, seq]`` assistant-span mask by scanning token ids.

    A span starts right after each assistant header and runs through the next terminator id
    (inclusive); a span cut by its row/segment edge runs to that edge. Rules apply per contiguous
    segment when ``segment_ids`` is given, so spans never cross document boundaries:

    * a segment with no role-marker structure (neither the header's first id nor the terminator) is
      plain text and is fully supervised;
    * a segment with structure but no complete header (e.g. a truncated window that lost its
      header) fails safe to all zeros;
    * when the header ends with a newline, a fused newline id at that position is accepted and the
      span starts AT the fused token (it carries content).

    Args:
        input_ids: ``[batch, seq]`` token ids.
        span_ids: The span signature.
        segment_ids: Optional ``[batch, seq]`` packed-document ids.

    Returns:
        np.ndarray: ``int32[batch, seq]`` mask.
    """
    ids = np.asarray(input_ids)
    bsz, seqlen = ids.shape
    header = span_ids.header_ids
    k = len(header)
    prefix, last_id = header[:-1], header[-1]
    last_ok = np.asarray(sorted(set(span_ids.newline_family) | {last_id}))
    structure_id = header[0]
    end_id = span_ids.end_id
    mask = np.zeros((bsz, seqlen), dtype=np.int32)

    def _scan(row: int, lo: int, hi: int) -> None:
        seg = ids[row, lo:hi]
        if not bool((seg == structure_id).any() or (seg == end_id).any()):
            mask[row, lo:hi] = 1
            return
        n = hi - lo
        m = len(prefix)
        starts: list[int] = []
        if n > m:
            hit = np.ones(n - m, dtype=bool)
            for offset, tok in enumerate(prefix):
                hit &= seg[offset : n - m + offset] == tok
            hit &= np.isin(seg[m:n], last_ok)
            for p in np.flatnonzero(hit):
                starts.append(int(p) + k if seg[p + m] == last_id else int(p) + m)
        if not starts:
            return
        ends = np.flatnonzero(seg == end_id)
        for s in starts:
            j = np.searchsorted(ends, s)
            stop = ends[j] if j < len(ends) else n - 1
            mask[row, lo + s : lo + stop + 1] = 1

    for row in range(bsz):
        if segment_ids is None:
            _scan(row, 0, seqlen)
            continue
        seg_row = np.asarray(segment_ids)[row]
        bounds = np.flatnonzero(np.diff(seg_row)) + 1
        lo = 0
        for hi in [*bounds.tolist(), seqlen]:
            _scan(row, int(lo), int(hi))
            lo = hi
    return mask


@dataclasses.dataclass(frozen=True)
class ZeroSupervisionStats:
    """Per-batch census of documents that ended with no supervised token.

    A document is a ``(window, segment)`` pair on packed batches and a row otherwise; padding
    (``segment_ids < 0`` or ``attention_mask == 0``) is excluded.

    Attributes:
        n_docs: Documents in the batch.
        n_zero: Documents with zero supervised tokens.
        n_zero_with_header: Zero documents that still contain an assistant header (content truncated).
        n_zero_without_header: Zero documents with no assistant header at all (no assistant turn).
        per_dataset: ``dataset_id -> (documents, zero documents)``; ``-1`` is unattributed.
        dataset_names: Optional ``dataset_id -> name`` map used for metric keys.
        samples: A few offending documents as ``(dataset_id, token_ids, has_header)``.
    """

    n_docs: int
    n_zero: int
    n_zero_with_header: int
    n_zero_without_header: int
    per_dataset: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)
    dataset_names: dict[int, str] | None = None
    samples: tuple[tuple[int, tuple[int, ...], bool], ...] = ()

    @property
    def frac(self) -> float:
        """Fraction of documents with zero supervised tokens."""
        return (self.n_zero / self.n_docs) if self.n_docs else 0.0

    def name(self, dataset_id: int) -> str:
        """Return the display name of ``dataset_id``."""
        if self.dataset_names and dataset_id in self.dataset_names:
            return str(self.dataset_names[dataset_id])
        return f"dataset_{dataset_id}" if dataset_id >= 0 else "unattributed"

    def as_metrics(self) -> dict[str, float]:
        """Return flat scalar metrics (``data/...`` keys) for the training logger."""
        out: dict[str, float] = {
            "data/documents": float(self.n_docs),
            "data/zero_supervision_docs": float(self.n_zero),
            "data/zero_supervision_frac": self.frac,
            "data/zero_no_assistant_turn": float(self.n_zero_without_header),
            "data/zero_truncated_header": float(self.n_zero_with_header),
        }
        for dataset_id, (n, z) in sorted(self.per_dataset.items()):
            label = self.name(dataset_id)
            out[f"data/zero_supervision_frac/{label}"] = (z / n) if n else 0.0
            out[f"data/documents/{label}"] = float(n)
        return out

    def diagnosis(self) -> str:
        """Describe the observed zero-supervision regime and the worst datasets."""
        if self.n_zero == 0:
            return "No zero-supervision documents."
        bits = []
        if self.n_zero_without_header:
            bits.append(f"{self.n_zero_without_header} carry no assistant header (no assistant turn in the row)")
        if self.n_zero_with_header:
            bits.append(
                f"{self.n_zero_with_header} carry an assistant header whose content is missing (truncation; check "
                "truncation_mode / max_length)"
            )
        ranked = sorted(
            ((z / n if n else 0.0, self.name(i), z, n) for i, (n, z) in self.per_dataset.items()),
            reverse=True,
        )
        top = [f"{label} {z}/{n} ({f:.1%})" for f, label, z, n in ranked[:3] if z]
        worst = (" Worst datasets: " + ", ".join(top) + ".") if top else ""
        return "Of these, " + "; ".join(bits) + "." + worst


def _contains_subsequence(tokens: np.ndarray, pattern: tuple[int, ...]) -> bool:
    if not pattern or tokens.shape[0] < len(pattern):
        return False
    windows = np.lib.stride_tricks.sliding_window_view(tokens, len(pattern))
    return bool((windows == np.asarray(pattern)).all(axis=-1).any())


def zero_supervision_stats(
    input_ids: np.ndarray,
    completion_mask: np.ndarray,
    *,
    attention_mask: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    dataset_ids_per_token: np.ndarray | None = None,
    dataset_ids_per_row: np.ndarray | None = None,
    header_ids: tuple[int, ...] = (),
    dataset_names: dict[int, str] | None = None,
    sample_limit: int = 3,
) -> ZeroSupervisionStats:
    """Census the documents of a batch that ended with zero supervised tokens.

    Args:
        input_ids: ``[B, S]`` token ids.
        completion_mask: ``[B, S]`` final loss mask.
        attention_mask: ``[B, S]`` real-token mask (unpacked batches).
        segment_ids: ``[B, S]`` packed document ids (``< 0`` = padding).
        dataset_ids_per_token: ``[B, S]`` per-token dataset ids (packed lane).
        dataset_ids_per_row: ``[B]`` per-row dataset ids (unpacked lane).
        header_ids: Assistant header ids, used to split zero documents into "no assistant turn" and
            "truncated"; empty disables the split (every zero counts as no-header).
        dataset_names: Optional id -> name map.
        sample_limit: Offending documents retained for the one-shot dump.

    Returns:
        ZeroSupervisionStats: The census.
    """
    ids = np.asarray(input_ids)
    cm = np.asarray(completion_mask)
    docs: list[tuple[np.ndarray, np.ndarray, int]] = []
    if segment_ids is not None:
        seg = np.asarray(segment_ids)
        for w in range(seg.shape[0]):
            valid = seg[w] >= 0
            for s in np.unique(seg[w][valid]):
                sel = seg[w] == s
                d_id = -1
                if dataset_ids_per_token is not None:
                    vals = np.asarray(dataset_ids_per_token)[w][sel]
                    d_id = int(vals[0]) if vals.size else -1
                docs.append((ids[w][sel], cm[w][sel], d_id))
    else:
        attn = np.ones_like(ids) if attention_mask is None else np.asarray(attention_mask)
        for r in range(ids.shape[0]):
            sel = attn[r] > 0
            if not sel.any():
                continue
            d_id = int(np.asarray(dataset_ids_per_row).reshape(-1)[r]) if dataset_ids_per_row is not None else -1
            docs.append((ids[r][sel], cm[r][sel], d_id))

    n_zero = with_header = without_header = 0
    per_dataset: dict[int, list[int]] = {}
    samples: list[tuple[int, tuple[int, ...], bool]] = []
    for tokens, doc_mask, d_id in docs:
        bucket = per_dataset.setdefault(d_id, [0, 0])
        bucket[0] += 1
        if int(np.asarray(doc_mask).sum()) != 0:
            continue
        n_zero += 1
        bucket[1] += 1
        has_header = _contains_subsequence(tokens, header_ids)
        if has_header:
            with_header += 1
        else:
            without_header += 1
        if len(samples) < max(int(sample_limit), 0):
            samples.append((d_id, tuple(int(t) for t in tokens), has_header))
    return ZeroSupervisionStats(
        n_docs=len(docs),
        n_zero=n_zero,
        n_zero_with_header=with_header,
        n_zero_without_header=without_header,
        per_dataset={key: (v[0], v[1]) for key, v in per_dataset.items()},
        dataset_names=dataset_names,
        samples=tuple(samples),
    )


def _segment_aware_label_mask(
    labels: np.ndarray,
    attention: np.ndarray | None,
    segment_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Mask source 2: supervision-bearing ``-100`` labels.

    Returns ``(mask, pending_rows, pending_tokens)``. A row (or packed document) WITHOUT any ``-100``
    label carries no supervision information; its positions are reported as pending so a later
    source (the span scan) or the full-sequence fallback can claim them. Under packing the rule runs
    per document, so one ``-100``-labelled document cannot zero its window neighbours.
    """
    neg = labels == -100
    if attention is not None:
        neg = neg & (attention == 1)
    if segment_ids is None:
        row_has = neg.any(axis=-1)
        if not bool(row_has.any()):
            return None, None, None
        mask = (labels != -100).astype(np.int32)
        if attention is not None:
            mask = mask * attention
        pending = None if bool(row_has.all()) else ~row_has
        return mask, pending, None
    if not bool(neg.any()):
        return None, None, None
    seg = np.asarray(segment_ids)
    valid = seg >= 0
    window = np.arange(seg.shape[0])[:, None]
    stride = int(seg.max()) + 2
    group = window * stride + np.where(valid, seg + 1, 0)
    has_any = np.bincount(group[neg].ravel(), minlength=int(group.max()) + 1) > 0
    has_tok = has_any[group] & valid
    mask = np.where(has_tok, (labels != -100).astype(np.int32), 0)
    if attention is not None:
        mask = mask * attention
    pending = valid & ~has_tok
    return mask, None, (pending if bool(pending.any()) else None)


class AssistantOnlyMaskMixin:
    """Assemble the per-dataset assistant-only loss mask on a host-side batch.

    Mixed into trainers whose ``_preprocess_batch_input`` sees host NumPy batches (SFT, distillation,
    speculative decoding). Requires ``self.arguments`` and, when the span scan runs, a tokenizer via
    ``self.processing_class``. Subclasses set :attr:`_assistant_mask_fallback_on_empty` to choose the
    legacy ``assistant_masks`` folding rule.
    """

    _assistant_mask_fallback_on_empty: tp.ClassVar[bool] = True
    consumes_assistant_only_carriers: tp.ClassVar[bool] = True

    def _ao_arg(self, name: str, default: tp.Any = None) -> tp.Any:
        return getattr(getattr(self, "arguments", None), name, default)

    def _assistant_span_ids(self) -> AssistantSpanIds:
        """Return (and cache) the assistant-span signature for this trainer's tokenizer."""
        cached = getattr(self, "_cached_assistant_span_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(self, "processing_class", None)
        span = derive_assistant_span_ids(
            tokenizer,
            header_text=self._ao_arg("assistant_header_text"),
            end_text=self._ao_arg("assistant_end_text"),
        )
        self._cached_assistant_span_ids = span
        return span

    def _apply_assistant_supervision(self, batch: dict[str, tp.Any], infos: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Pop the sideband carriers and normalize ``completion_mask`` / ``labels`` in place.

        Args:
            batch: Host-side batch, mutated in place.
            infos: Preprocessing metrics dict; the zero-supervision census is added when it runs.

        Returns:
            The same batch. Without a per-dataset choice this is exactly the legacy normalization.

        Raises:
            ValueError: On incoherent carriers, or when the zero-supervision fraction exceeds
                ``zero_supervision_abort_fraction`` while per-dataset masking is active.
        """
        if not isinstance(batch, dict):
            return batch
        row_ao = batch.pop(ASSISTANT_ONLY_FIELD, None)
        tok_ao = batch.pop(ASSISTANT_ONLY_TOKEN_FIELD, None)
        row_ds = batch.pop(DATASET_ID_FIELD, None)
        tok_ds = batch.pop(DATASET_ID_TOKEN_FIELD, None)
        if row_ao is not None and tok_ao is not None:
            raise ValueError("both per-row `_assistant_only` and per-token `assistant_only_mask` reached one batch.")
        segment_ids = batch.get("segment_ids")
        if row_ao is not None and segment_ids is not None:
            raise ValueError(
                "per-row `_assistant_only` reached a packed batch (segment_ids present); the packing collator must "
                "emit the per-token `assistant_only_mask` instead."
            )

        ao = None
        if tok_ao is not None:
            ao = np.asarray(tok_ao, dtype=np.int32)
        elif row_ao is not None and "input_ids" in batch:
            flags = np.asarray(row_ao).reshape(-1).astype(np.int32)
            ao = np.repeat(flags[:, None], int(np.asarray(batch["input_ids"]).shape[-1]), axis=1)
        if ao is not None and not np.isin(ao, (-1, 0, 1)).all():
            raise ValueError(f"assistant-only carrier values outside -1/0/1: {np.unique(ao)!r}")

        global_flag = bool(self._ao_arg("assistant_only_loss", False))
        per_dataset = ao is not None and bool((ao != -1).any())
        if not per_dataset:
            batch = normalize_completion_masks_and_labels(
                batch, assistant_mask_fallback_on_empty=self._assistant_mask_fallback_on_empty
            )
            carriers_present = any(c is not None for c in (row_ao, tok_ao, row_ds, tok_ds))
            if carriers_present and "completion_mask" in batch:
                self._run_zero_supervision_census(batch, infos, tok_ds=tok_ds, row_ds=row_ds, enforce=False)
            return batch

        if "assistant_masks" in batch:
            if "completion_mask" not in batch or (
                self._assistant_mask_fallback_on_empty and not np.asarray(batch["completion_mask"]).any()
            ):
                batch["completion_mask"] = batch["assistant_masks"]
            batch.pop("assistant_masks", None)
        self._assemble_per_dataset_mask(batch, ao, global_flag)
        if "completion_mask" in batch:
            self._run_zero_supervision_census(batch, infos, tok_ds=tok_ds, row_ds=row_ds, enforce=True)
        return batch

    def _assemble_per_dataset_mask(self, batch: dict[str, tp.Any], ao: np.ndarray, global_flag: bool) -> None:
        """Assemble ``completion_mask`` per token from the mask sources and the per-dataset choices.

        ``ao == 0`` tokens train on every real token; ``ao == 1`` tokens take the masked result (source
        1/2 when present, else the span scan); ``ao == -1`` tokens follow the global flag.
        """
        attention = batch.get("attention_mask")
        attention = None if attention is None else np.asarray(attention).astype(np.int32)
        segment_ids = batch.get("segment_ids")
        segment_ids = None if segment_ids is None else np.asarray(segment_ids)
        input_ids = None if "input_ids" not in batch else np.asarray(batch["input_ids"])
        ones = attention if attention is not None else (None if input_ids is None else np.ones_like(input_ids))

        base = None
        pending_rows = pending_tok = None
        if batch.get("completion_mask") is not None:
            base = np.asarray(batch["completion_mask"]).astype(np.int32)
            if attention is not None:
                base = base * attention
        elif "labels" in batch:
            base, pending_rows, pending_tok = _segment_aware_label_mask(
                np.asarray(batch["labels"]), attention, segment_ids
            )

        need_scan = input_ids is not None and (
            bool((ao == 1).any())
            or (global_flag and (base is None or pending_rows is not None or pending_tok is not None))
        )
        scan = None
        if need_scan:
            scan = assistant_mask_from_token_ids(input_ids, self._assistant_span_ids(), segment_ids=segment_ids)
            if attention is not None:
                scan = scan * attention

        def _masked() -> np.ndarray | None:
            """The masked result ``M``: sources 1/2 with pending positions filled by the scan."""
            if base is None:
                return None if scan is None else scan.copy()
            out = base.copy()
            if scan is not None and pending_rows is not None:
                out[pending_rows] = scan[pending_rows]
            elif scan is not None and pending_tok is not None:
                out = np.where(pending_tok, scan, out)
            return out

        if ones is None:
            return
        # Opted-in tokens take the masked result M (explicit mask > -100 labels > span scan).
        opt_in = _masked()
        if global_flag:
            inherit = opt_in if opt_in is not None else ones
        else:
            # Global off: inheriting tokens keep the pre-scan mask (or train fully); documents whose labels carry
            # no -100 keep full supervision, exactly like the legacy path.
            inherit = ones.copy() if base is None else base.copy()
            if pending_tok is not None:
                inherit = np.where(pending_tok, ones, inherit)
            if pending_rows is not None:
                inherit[pending_rows] = ones[pending_rows]
        cm = np.where(ao == 0, ones, np.where(ao == 1, opt_in if opt_in is not None else ones, inherit))
        cm = cm.astype(np.int32, copy=False)
        if attention is not None:
            cm = cm * attention
        batch["completion_mask"] = cm
        if "labels" not in batch and input_ids is not None:
            labels = input_ids.astype(np.int32, copy=True)
            labels[cm == 0] = -100
            batch["labels"] = labels

    def _run_zero_supervision_census(
        self,
        batch: dict[str, tp.Any],
        infos: dict[str, tp.Any],
        *,
        tok_ds: tp.Any,
        row_ds: tp.Any,
        enforce: bool,
    ) -> ZeroSupervisionStats | None:
        """Measure zero-supervision documents, log them, and abort past the configured fraction."""
        if "input_ids" not in batch:
            return None
        header: tuple[int, ...] = ()
        if not getattr(self, "_assistant_span_ids_unavailable", False):
            try:
                header = self._assistant_span_ids().header_ids
            except Exception:
                # Diagnostic only: remember the failure so the derivation is not retried every batch.
                self._assistant_span_ids_unavailable = True
        stats = zero_supervision_stats(
            np.asarray(batch["input_ids"]),
            np.asarray(batch["completion_mask"]),
            attention_mask=batch.get("attention_mask"),
            segment_ids=batch.get("segment_ids"),
            dataset_ids_per_token=tok_ds,
            dataset_ids_per_row=row_ds,
            header_ids=header,
            dataset_names=getattr(self, "_dataset_id_names", None),
            sample_limit=int(self._ao_arg("zero_supervision_dump_limit", 3) or 0),
        )
        self._last_zero_supervision_stats = stats
        infos.update(stats.as_metrics())
        abort = self._ao_arg("zero_supervision_abort_fraction", 0.5)
        report = self._ao_arg("zero_supervision_report_fraction", 0.05)
        if enforce and abort is not None and stats.frac > float(abort):
            raise ValueError(
                f"assistant-only masking left {stats.frac:.1%} of documents with zero supervised tokens "
                f"({stats.n_zero}/{stats.n_docs}), above zero_supervision_abort_fraction={float(abort):.0%}: this "
                f"is a configuration problem, not data noise. {stats.diagnosis()}"
            )
        if report is not None and stats.frac > float(report):
            logger.warning(
                "assistant-only masking: %.2f%% of documents in this batch have zero supervised tokens (%d/%d), "
                "above the %.0f%% report threshold. %s They contribute no loss; watch `data/zero_supervision_frac`.",
                stats.frac * 100.0,
                stats.n_zero,
                stats.n_docs,
                float(report) * 100.0,
                stats.diagnosis(),
            )
            self._dump_zero_supervision_documents(stats)
        return stats

    def _dump_zero_supervision_documents(self, stats: ZeroSupervisionStats) -> None:
        """Log a few zero-supervision documents once per process (diagnostics never raise)."""
        if getattr(self, "_zero_dump_done", False):
            return
        self._zero_dump_done = True
        try:
            tokenizer = getattr(self, "processing_class", None)
            if tokenizer is not None and hasattr(tokenizer, "tokenizer") and not hasattr(tokenizer, "decode"):
                tokenizer = tokenizer.tokenizer
            for i, (d_id, tokens, has_header) in enumerate(stats.samples):
                ids = list(tokens)[:4096]
                text = ""
                if tokenizer is not None:
                    try:
                        text = tokenizer.decode(ids, skip_special_tokens=False)
                    except Exception as exc:
                        text = f"<decode failed: {type(exc).__name__}>"
                logger.warning(
                    "[zero-supervision-dump] %d/%d dataset=%s tokens=%d has_header=%s first_ids=%s text=%r",
                    i + 1,
                    len(stats.samples),
                    stats.name(d_id),
                    len(ids),
                    has_header,
                    ids[:24],
                    text.replace("\n", "\\n")[:700],
                )
        except Exception as exc:
            logger.warning("[zero-supervision-dump] failed (%s); training continues.", type(exc).__name__)


__all__ = (
    "SIDEBAND_CARRIER_FIELDS",
    "AssistantOnlyMaskMixin",
    "AssistantSpanIds",
    "ZeroSupervisionStats",
    "assistant_mask_from_token_ids",
    "derive_assistant_span_ids",
    "zero_supervision_stats",
)
