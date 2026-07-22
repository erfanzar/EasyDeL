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

"""Shared building blocks for eSurge serving benchmarks.

The benchmark scripts (``scripts/bench_esurge.py``, ``scripts/bench_spec_ab.py``,
the spec-decode bench, ...) each rebuilt the same prompt generators and
scheduler-loop predicates. These pure helpers now live here so every harness
draws from one source.

The continuous-batching *driver* (``run_batch``) is intentionally left in each
script: the copies genuinely diverge (speculative-token handling, async-dispatch
gating, and the shape of the returned profile), and their throughput output is
only meaningful on accelerator hardware, so unifying the driver needs a
device-side before/after comparison rather than a CPU refactor.
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

__all__ = [
    "all_finished",
    "make_prompts",
    "make_real_prompts",
    "zero_schedule_needs_update",
]


def make_prompts(*, count: int, prompt_len: int, vocab_size: int, seed: int = 0) -> list[list[int]]:
    """Generate ``count`` random token-ID prompts of exactly ``prompt_len`` tokens.

    Samples integer tokens uniformly from ``[1000, min(vocab_size-1, 100000))``
    to avoid special tokens at the very low end of the vocabulary while keeping
    prompts well inside the model's regular vocab range.

    Args:
        count: Number of prompts to produce.
        prompt_len: Token length of each prompt.
        vocab_size: Vocabulary size of the target model.
        seed: NumPy PRNG seed for reproducibility.

    Returns:
        list[list[int]]: ``count`` lists of ``prompt_len`` Python ints.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    lo = 1000
    hi = max(lo + 1, min(int(vocab_size) - 1, 100000))
    return [[int(tok) for tok in rng.integers(lo, hi, size=prompt_len, dtype=np.int32)] for _ in range(count)]


_REAL_PROMPT_CACHE: dict[tuple, list[list[int]]] = {}


def make_real_prompts(
    *,
    count: int,
    prompt_len: int,
    tokenizer_name: str,
    text_file: str | None = None,
    doc_paths: tp.Sequence[str | Path] | None = None,
) -> list[list[int]]:
    """Build deterministic REAL chat-templated prompts, each exactly ``prompt_len`` tokens.

    Slices long real text at ``count`` different offsets and wraps each slice in
    the model's chat template with a generation prompt. In-distribution text
    gives realistic MTP acceptance, unlike random-token prompts. Results are
    cached by ``(count, prompt_len, tokenizer_name, text_file)``.

    Args:
        count: Number of prompts to produce.
        prompt_len: Exact token length of each prompt.
        tokenizer_name: HF tokenizer id/path for encoding + chat templating.
        text_file: Optional path to a source text; when omitted, ``doc_paths``
            (or the caller's default docs) are concatenated as the source.
        doc_paths: Fallback source documents used when ``text_file`` is None.

    Returns:
        list[list[int]]: ``count`` prompts, each exactly ``prompt_len`` ints.

    Raises:
        AssertionError: If the source text is too short for the requested prompts.
    """
    key = (count, prompt_len, tokenizer_name, text_file)
    if key in _REAL_PROMPT_CACHE:
        return _REAL_PROMPT_CACHE[key]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if text_file:
        doc = Path(text_file).read_text()
    else:
        texts = []
        for p in doc_paths or ():
            try:
                texts.append(Path(p).read_text())
            except Exception:
                pass
        doc = "\n\n".join(texts)
    doc_ids = tok.encode(doc, add_special_tokens=False)
    assert len(doc_ids) > prompt_len * (count + 2), "source text too short for these prompts"

    prompts = []
    stride = max(1, (len(doc_ids) - prompt_len - 64) // max(count, 1))
    for i in range(count):
        off = i * stride
        content_ids = doc_ids[off : off + prompt_len]  # upper bound; trimmed below
        take = prompt_len - 32
        ids = None
        for _ in range(6):
            content = tok.decode(content_ids[:take])
            msgs = [{"role": "user", "content": "Continue writing this documentation:\n\n" + content}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids["input_ids"]
            if hasattr(ids, "ids"):
                ids = ids.ids
            ids = list(ids)
            if ids and not isinstance(ids[0], int):
                ids = list(ids[0].ids if hasattr(ids[0], "ids") else ids[0])
            if len(ids) == prompt_len:
                break
            take += prompt_len - len(ids)
            take = max(8, min(take, len(content_ids)))
        if len(ids) > prompt_len:
            ids = ids[:prompt_len]
        elif len(ids) < prompt_len:
            ids = ids + doc_ids[off + take : off + take + (prompt_len - len(ids))]
        prompts.append([int(t) for t in ids])
        assert len(prompts[-1]) == prompt_len
    _REAL_PROMPT_CACHE[key] = prompts
    return prompts


def all_finished(requests) -> bool:
    """Return ``True`` when every request in ``requests`` has finished.

    Args:
        requests: Iterable of :class:`EngineRequest` objects.

    Returns:
        bool: ``True`` if every request is finished.
    """
    return all(req.is_finished() for req in requests)


def zero_schedule_needs_update(scheduler_output) -> bool:
    """Return ``True`` if a zero-token schedule still requires a model update.

    A schedule with zero scheduled tokens can still carry side effects
    (finished requests to release, preempted requests to spill, new requests
    to admit, or cached requests that need a no-op pass). When any of those
    bookkeeping fields are non-empty the model still needs to be invoked.

    Args:
        scheduler_output: Output of :meth:`Scheduler.schedule`.

    Returns:
        bool: ``True`` when a follow-up ``execute_model`` is required.
    """
    return bool(
        scheduler_output.finished_req_ids
        or scheduler_output.preempted_req_ids
        or scheduler_output.scheduled_new_reqs
        or scheduler_output.scheduled_cached_reqs.num_reqs
    )
