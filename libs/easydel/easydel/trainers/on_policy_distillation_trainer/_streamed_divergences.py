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

"""Vocabulary-streamed distillation divergences for on-policy distillation.

The functions here compute every divergence the on-policy distillation trainer supports (forward KL,
reverse KL, generalized JSD, confidence-aware KL, teacher-top-k KL) and the trust-region distillation
objective directly from hidden states and the LM-head projections, without materializing ``[B, T, V]``
logits.

Two projection flavours are supported:

* :class:`VocabChunkedProjection` streams the vocabulary: each scan step projects the hidden states onto
  one ``chunk_size``-wide slice of the LM-head weight. Log-normalizers are built with an online
  max/logsumexp scan and teacher top-k support with a running top-k, so peak logit memory is
  ``O(N * chunk_size)``.
* :class:`FullVocabProjection` wraps an opaque ``hidden -> logits`` callable (e.g. ``make_lm_head_fn()``)
  for heads whose weight cannot be sliced (quantized, fused collective matmul, family-specific heads). It is
  combined with sequence chunking so peak logit memory is ``O(B * sequence_chunk_size * V)``.

Every per-chunk scan body is wrapped in :func:`jax.checkpoint`, so the backward pass recomputes one chunk of
logits at a time instead of keeping them all alive.
"""

from __future__ import annotations

import math
import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array

from ..distillation_trainer.loss_utils import resolve_kl_direction

# Finite fill for vocabulary padding columns. Finite (not -inf) so ``p * (log p - log q)`` stays 0 * finite
# on padded columns and no NaN reaches the backward pass through a ``where``.
_PAD_LOGIT = -1e9


class VocabChunkedProjection:
    """LM-head projection evaluated one vocabulary slice at a time.

    Reproduces ``ParallelLinear.forward`` (dtype promotion, ``precision``, output scale, bias) on a column
    slice of the ``[hidden, vocab]`` kernel, followed by an optional elementwise logit post-process (logit
    clipping, soft-capping, output scaling). Elementwise post-processes commute with vocabulary slicing.

    Args:
        weight: ``[hidden, vocab]`` kernel.
        bias: Optional ``[vocab]`` bias.
        chunk_size: Vocabulary columns per scan step.
        output_scale: Scalar multiplier applied before the bias (``ParallelLinear.output_scale``).
        postprocess: Optional elementwise ``logits -> logits`` callable.
        dtype: Compute dtype of the projection; ``None`` keeps the input dtypes.
        precision: Matmul precision forwarded to ``jnp.einsum``.
    """

    def __init__(
        self,
        weight: Array,
        bias: Array | None = None,
        *,
        chunk_size: int,
        output_scale: float = 1.0,
        postprocess: tp.Callable[[Array], Array] | None = None,
        dtype: tp.Any = None,
        precision: tp.Any = None,
    ):
        if weight.ndim != 2:
            raise ValueError(f"VocabChunkedProjection expects a [hidden, vocab] kernel, got shape {weight.shape}.")
        self.vocab_size = int(weight.shape[-1])
        self.chunk_size = max(1, min(int(chunk_size), self.vocab_size))
        self.num_chunks = -(-self.vocab_size // self.chunk_size)
        pad = self.num_chunks * self.chunk_size - self.vocab_size
        self._weight = jnp.pad(weight, ((0, 0), (0, pad))) if pad else weight
        self._bias = None if bias is None else (jnp.pad(bias, ((0, pad),)) if pad else bias)
        self.output_scale = float(output_scale)
        self.postprocess = postprocess
        self.dtype = dtype
        self.precision = precision

    def layout(self, hidden: Array) -> tuple[int, int, int]:
        """Return ``(vocab_size, num_chunks, chunk_size)``."""
        del hidden
        return self.vocab_size, self.num_chunks, self.chunk_size

    def logits(self, hidden: Array, chunk_index: Array) -> Array:
        """Project ``[N, hidden]`` states onto vocabulary chunk ``chunk_index``; returns ``[N, chunk_size]``."""
        start = chunk_index * self.chunk_size
        w = jax.lax.dynamic_slice_in_dim(self._weight, start, self.chunk_size, axis=1)
        h = hidden
        b = None if self._bias is None else jax.lax.dynamic_slice_in_dim(self._bias, start, self.chunk_size, axis=0)
        if self.dtype is not None:
            h = h.astype(self.dtype)
            w = w.astype(self.dtype)
            b = None if b is None else b.astype(self.dtype)
        y = jnp.einsum("nh,hc->nc", h, w, precision=self.precision)
        if self.output_scale != 1.0:
            y = y * self.output_scale
        if b is not None:
            y = y + b
        if self.postprocess is not None:
            y = self.postprocess(y)
        return y


class FullVocabProjection:
    """Opaque ``hidden -> logits`` projection treated as a single vocabulary chunk.

    Args:
        lm_head_fn: Trace-safe callable mapping ``[..., hidden]`` to ``[..., vocab]`` logits.
    """

    num_chunks = 1

    def __init__(self, lm_head_fn: tp.Callable[[Array], Array]):
        self.lm_head_fn = lm_head_fn
        self._vocab: int | None = None

    def layout(self, hidden: Array) -> tuple[int, int, int]:
        """Return ``(vocab_size, 1, vocab_size)``; the vocabulary is read from a one-row projection's shape."""
        if self._vocab is None:
            # Only the static shape is used; the one-row matmul is dead code after tracing.
            self._vocab = int(self.lm_head_fn(hidden.reshape(-1, hidden.shape[-1])[:1]).shape[-1])
        return self._vocab, 1, self._vocab

    def logits(self, hidden: Array, chunk_index: Array) -> Array:
        """Project ``[N, hidden]`` states to the full ``[N, vocab]`` logits."""
        del chunk_index
        return self.lm_head_fn(hidden)


Projection = VocabChunkedProjection | FullVocabProjection


def resolve_vocab_chunked_projection(module: tp.Any, chunk_size: int) -> VocabChunkedProjection | None:
    """Build a :class:`VocabChunkedProjection` from a model's LM head, or ``None`` when it cannot be sliced.

    The head must be a plain (non-quantized, no fused collective matmul) ``ParallelLinear``. Tied embeddings
    are resolved through the module's tied-weight helper and the complete post-projection transform comes from
    ``get_lm_logit_postprocess`` (the same contract split LM heads use). Families exporting a custom
    ``get_lm_head_projection`` are not sliced.

    Args:
        module: The (merged) EasyDeL model.
        chunk_size: Vocabulary columns per scan step.

    Returns:
        The projection, or ``None`` so the caller falls back to :class:`FullVocabProjection`.
    """
    from easydel.layers.linears import ParallelLinear

    if getattr(module, "get_lm_head_projection", None) is not None:
        return None
    # The sliced projection rebuilds the head from ``get_lm_logit_postprocess``. That is only faithful when the
    # class that last overrode ``make_lm_head_fn`` also owns (or is overridden by) the postprocess hook;
    # otherwise the family added projection semantics the hook does not describe.
    mro = type(module).__mro__
    head_owner = next((c for c in mro if "make_lm_head_fn" in vars(c)), None)
    post_owner = next((c for c in mro if "get_lm_logit_postprocess" in vars(c)), None)
    if head_owner is not None and head_owner.__name__ not in ("EasyDeLBaseModule", "BaseTaskModule"):
        if post_owner is None or not issubclass(post_owner, head_owner):
            return None
    get_head = getattr(module, "get_lm_head", None)
    if not callable(get_head):
        return None
    try:
        head = get_head()
    except (NotImplementedError, AttributeError):
        return None
    if not isinstance(head, ParallelLinear) or getattr(head, "distributed_matmul", None) is not None:
        return None
    # Subclasses that replace the matmul (adapters, fused heads) are not reproduced by the sliced projection.
    if type(head).forward is not ParallelLinear.forward:
        return None
    if spx.quantization.rule_for(head, "dot_general") is not None:
        return None
    tied_fn = getattr(module, "_tied_lm_head_weight", None)
    weight = tied_fn(head) if callable(tied_fn) else None
    if weight is None:
        weight = head.weight.value
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return None
    bias = head.bias.value if head.use_bias and head.bias is not None else None
    postprocess_fn = getattr(module, "get_lm_logit_postprocess", None)
    postprocess = postprocess_fn() if callable(postprocess_fn) else None
    return VocabChunkedProjection(
        weight,
        bias,
        chunk_size=chunk_size,
        output_scale=float(getattr(head, "output_scale", 1.0)),
        postprocess=postprocess,
        dtype=head.dtype,
        precision=head.precision,
    )


def build_distillation_projections(
    student_module: tp.Any,
    teacher_module: tp.Any,
    chunk_size: int | None,
    *,
    prefer_sequence_chunks: bool = False,
) -> tuple[Projection, Projection, int | None]:
    """Pick the student/teacher projections and the sequence chunk size for the streamed losses.

    With ``chunk_size > 0`` and sliceable LM heads of equal vocabulary, both sides stream the vocabulary in
    ``chunk_size`` columns. Otherwise the opaque ``make_lm_head_fn`` heads are used with ``chunk_size``
    sequence positions per scan step (or no chunking when ``chunk_size`` is falsy).

    Args:
        student_module: Student model (differentiable).
        teacher_module: Teacher model.
        chunk_size: ``logits_chunk_size`` from the trainer config.
        prefer_sequence_chunks: Use sequence chunking even when the heads could be sliced, e.g. when the
            vocabulary is tensor-parallel sharded and slicing it per scan step would gather the weight.

    Returns:
        ``(student_projection, teacher_projection, sequence_chunk_size)``.
    """
    if chunk_size and not prefer_sequence_chunks:
        student = resolve_vocab_chunked_projection(student_module, chunk_size)
        teacher = resolve_vocab_chunked_projection(teacher_module, chunk_size)
        if student is not None and teacher is not None and student.vocab_size == teacher.vocab_size:
            return student, teacher, None
    student_full = FullVocabProjection(student_module.make_lm_head_fn())
    teacher_full = FullVocabProjection(teacher_module.make_lm_head_fn())
    return student_full, teacher_full, (int(chunk_size) if chunk_size else None)


def _online_lse_update(m: Array, s: Array, x: Array) -> tuple[Array, Array]:
    """One step of the online logsumexp: running max ``m`` (no gradient) and rescaled sum ``s``."""
    # The running max only re-centres the exponentials; ``lse = m + log(s)`` is invariant to it, so it carries
    # no gradient (and the ``-inf`` initial value never reaches a backward product).
    m_new = jax.lax.stop_gradient(jnp.maximum(m, jnp.max(x, axis=-1)))
    s_new = s * jnp.exp(m - m_new) + jnp.sum(jnp.exp(x - m_new[:, None]), axis=-1)
    return m_new, s_new


def _token_statistics(
    h_s: Array,
    h_t: Array,
    targets: Array | None,
    s_proj: Projection,
    t_proj: Projection,
    *,
    temperature: float,
    top_k: int,
    divergences: tuple[str, ...],
    beta: float,
) -> dict[str, Array]:
    """Per-position divergence statistics for ``[N, hidden]`` states, streamed over vocabulary chunks.

    Pass 1 builds both log-normalizers online, keeps the teacher's running top-k (values and ids), and gathers
    the sampled-token logits. Pass 2 (only when needed) accumulates the full-vocabulary divergence sums and
    gathers the student logits on the teacher's top-k ids. All math is float32 at temperature ``T``.

    Returns:
        Mapping with any of ``forward``/``reverse``/``jsd``/``entropy`` (``[N]``), ``topk_teacher_logp`` /
        ``topk_student_logp`` (``[N, k]``, normalized over the full vocabulary), ``target_student_logp`` /
        ``target_teacher_logp`` (``[N]``).
    """
    vocab, n_chunks, chunk = s_proj.layout(h_s)
    if t_proj.layout(h_t) != (vocab, n_chunks, chunk):
        raise ValueError(f"student/teacher LM-head layouts differ: {(vocab, n_chunks, chunk)} vs {t_proj.layout(h_t)}.")
    inv_t = 1.0 / float(temperature)
    n = h_s.shape[0]
    k = min(int(top_k), vocab) if top_k else 0
    col = jnp.arange(chunk, dtype=jnp.int32)
    f32 = jnp.float32

    def _scaled(proj: Projection, h: Array, j: Array) -> tuple[Array, Array, Array]:
        start = (j * chunk).astype(jnp.int32)
        valid = (start + col) < vocab
        x = proj.logits(h, j).astype(f32) * inv_t
        return jnp.where(valid[None, :], x, _PAD_LOGIT), valid, start

    def _gather(x: Array, idx: Array, start: Array, prev: Array) -> Array:
        local = idx - start
        in_range = (local >= 0) & (local < chunk)
        got = jnp.take_along_axis(x, jnp.clip(local, 0, chunk - 1), axis=-1)
        return jnp.where(in_range, got, prev)

    def _pass_one(carry: dict[str, Array], j: Array) -> tuple[dict[str, Array], None]:
        s, _, start = _scaled(s_proj, h_s, j)
        t = jax.lax.stop_gradient(_scaled(t_proj, h_t, j)[0])
        out = dict(carry)
        out["s_m"], out["s_s"] = _online_lse_update(carry["s_m"], carry["s_s"], s)
        out["t_m"], out["t_s"] = _online_lse_update(carry["t_m"], carry["t_s"], t)
        if k:
            cand_v = jnp.concatenate([carry["tk_v"], t], axis=-1)
            cand_i = jnp.concatenate([carry["tk_i"], jnp.broadcast_to(start + col, (n, chunk))], axis=-1)
            top_v, pos = jax.lax.top_k(cand_v, k)
            out["tk_v"] = top_v
            out["tk_i"] = jnp.take_along_axis(cand_i, pos, axis=-1)
        if targets is not None:
            tgt = targets[:, None]
            out["s_tgt"] = _gather(s, tgt, start, carry["s_tgt"][:, None])[:, 0]
            out["t_tgt"] = _gather(t, tgt, start, carry["t_tgt"][:, None])[:, 0]
        return out, None

    zero = jnp.zeros((n,), f32)
    init: dict[str, Array] = {
        "s_m": jnp.full((n,), -jnp.inf, f32),
        "s_s": zero,
        "t_m": jnp.full((n,), -jnp.inf, f32),
        "t_s": zero,
    }
    if k:
        init["tk_v"] = jnp.full((n, k), -jnp.inf, f32)
        init["tk_i"] = jnp.zeros((n, k), jnp.int32)
    if targets is not None:
        init["s_tgt"] = zero
        init["t_tgt"] = zero
    chunk_ids = jnp.arange(n_chunks, dtype=jnp.int32)
    stats, _ = jax.lax.scan(jax.checkpoint(_pass_one), init, chunk_ids)
    lse_s = stats["s_m"] + jnp.log(stats["s_s"])
    lse_t = jax.lax.stop_gradient(stats["t_m"] + jnp.log(stats["t_s"]))

    result: dict[str, Array] = {}
    if targets is not None:
        result["target_student_logp"] = stats["s_tgt"] - lse_s
        result["target_teacher_logp"] = jax.lax.stop_gradient(stats["t_tgt"]) - lse_t

    if divergences or k:
        log_b = math.log(beta) if "jsd" in divergences else 0.0
        log_1mb = math.log1p(-beta) if "jsd" in divergences else 0.0
        topk_ids = stats["tk_i"] if k else None

        def _pass_two(carry: dict[str, Array], j: Array) -> tuple[dict[str, Array], None]:
            s, valid, start = _scaled(s_proj, h_s, j)
            out = dict(carry)
            if divergences:
                t = jax.lax.stop_gradient(_scaled(t_proj, h_t, j)[0])
                vm = valid[None, :]
                lp = t - lse_t[:, None]
                lq = s - lse_s[:, None]
                p = jnp.exp(lp)
                if "forward" in divergences:
                    out["forward"] = carry["forward"] + jnp.sum(jnp.where(vm, p * (lp - lq), 0.0), axis=-1)
                if "entropy" in divergences:
                    out["entropy"] = carry["entropy"] + jnp.sum(jnp.where(vm, -p * lp, 0.0), axis=-1)
                if "reverse" in divergences:
                    q = jnp.exp(lq)
                    out["reverse"] = carry["reverse"] + jnp.sum(jnp.where(vm, q * (lq - lp), 0.0), axis=-1)
                if "jsd" in divergences:
                    q = jnp.exp(lq)
                    lm = jnp.logaddexp(lp + log_b, lq + log_1mb)
                    term = beta * p * (lp - lm) + (1.0 - beta) * q * (lq - lm)
                    out["jsd"] = carry["jsd"] + jnp.sum(jnp.where(vm, term, 0.0), axis=-1)
            if k:
                out["tk_s"] = _gather(s, topk_ids, start, carry["tk_s"])
            return out, None

        init_two: dict[str, Array] = {name: zero for name in divergences}
        if k:
            init_two["tk_s"] = jnp.zeros((n, k), f32)
        sums, _ = jax.lax.scan(jax.checkpoint(_pass_two), init_two, chunk_ids)
        for name in divergences:
            result[name] = sums[name]
        if k:
            result["topk_teacher_logp"] = jax.lax.stop_gradient(stats["tk_v"]) - lse_t[:, None]
            result["topk_student_logp"] = sums["tk_s"] - lse_s[:, None]
    return result


def _map_positions(
    fn: tp.Callable[[Array, Array, Array | None], dict[str, Array]],
    h_s: Array,
    h_t: Array,
    targets: Array | None,
    sequence_chunk_size: int | None,
) -> dict[str, Array]:
    """Apply ``fn`` to flattened positions, optionally scanning ``sequence_chunk_size`` positions at a time.

    Returns ``fn``'s outputs reshaped to ``[B, T, ...]``.
    """
    b, t = h_s.shape[:2]
    if not sequence_chunk_size or sequence_chunk_size >= t:
        flat_t = None if targets is None else targets.reshape(b * t)
        out = fn(h_s.reshape(b * t, h_s.shape[-1]), h_t.reshape(b * t, h_t.shape[-1]), flat_t)
        return jax.tree_util.tree_map(lambda x: x.reshape(b, t, *x.shape[1:]), out)
    c = int(sequence_chunk_size)
    pad = (-t) % c
    n_s = (t + pad) // c

    def _split(x: Array) -> Array:
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2))
        x = x.reshape(b, n_s, c, *x.shape[2:])
        return jnp.moveaxis(x, 1, 0).reshape(n_s, b * c, *x.shape[3:])

    xs = (_split(h_s), _split(h_t), None if targets is None else _split(targets))

    def _body(carry: None, chunk_xs: tuple[Array, Array, Array | None]) -> tuple[None, dict[str, Array]]:
        return carry, fn(*chunk_xs)

    _, ys = jax.lax.scan(_body, None, xs)

    def _merge(y: Array) -> Array:
        y = y.reshape(n_s, b, c, *y.shape[2:])
        return jnp.moveaxis(y, 0, 1).reshape(b, n_s * c, *y.shape[3:])[:, :t]

    return jax.tree_util.tree_map(_merge, ys)


def _shift_and_mask(
    student_hidden: Array,
    teacher_hidden: Array,
    attention_mask: Array | None,
    loss_mask: Array | None,
    input_ids: Array | None,
    shift_tokens: bool,
) -> tuple[Array, Array, Array, Array | None]:
    """Apply the causal shift and build the float32 ``loss_mask > attention_mask > ones`` token mask."""
    targets = input_ids
    if shift_tokens:
        student_hidden = student_hidden[:, :-1]
        teacher_hidden = teacher_hidden[:, :-1]
        attention_mask = None if attention_mask is None else attention_mask[:, 1:]
        loss_mask = None if loss_mask is None else loss_mask[:, 1:]
        targets = None if input_ids is None else input_ids[:, 1:]
    if loss_mask is not None:
        mask = loss_mask.astype(jnp.float32)
    elif attention_mask is not None:
        mask = attention_mask.astype(jnp.float32)
    else:
        mask = jnp.ones(student_hidden.shape[:2], jnp.float32)
    return student_hidden, jax.lax.stop_gradient(teacher_hidden), mask, targets


def _topk_xent_entropy(lt: Array, ls: Array, *, add_tail: bool) -> tuple[Array, Array]:
    """Teacher-top-k cross-entropy / entropy from full-vocabulary log-probs on the support (dense semantics)."""
    if add_tail:
        eps = jnp.float32(1e-7)
        pt = jnp.exp(lt)
        teacher_tail = jnp.clip(1.0 - jnp.sum(pt, axis=-1), eps, 1.0)
        student_tail = jnp.clip(1.0 - jnp.sum(jnp.exp(ls), axis=-1), eps, 1.0)
        xent = -(jnp.sum(pt * ls, axis=-1) + teacher_tail * jnp.log(student_tail))
        entropy = -(jnp.sum(pt * lt, axis=-1) + teacher_tail * jnp.log(teacher_tail))
        return xent, entropy
    t_support = jax.nn.log_softmax(lt, axis=-1)
    s_support = jax.nn.log_softmax(ls, axis=-1)
    pt = jnp.exp(t_support)
    return -jnp.sum(pt * s_support, axis=-1), -jnp.sum(pt * t_support, axis=-1)


def _masked_mean(values: Array, mask: Array, normalizer: Array) -> Array:
    return jnp.sum(values.astype(jnp.float32) * mask) / normalizer


def streamed_distillation_loss(
    student_hidden: Array,
    teacher_hidden: Array,
    student_projection: Projection,
    teacher_projection: Projection,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    *,
    temperature: float = 4.0,
    alpha: float = 0.9,
    beta: float | None = None,
    cakld_gamma: float | None = None,
    loss_top_k: int = 0,
    loss_add_tail: bool = False,
    shift_tokens: bool = True,
    sequence_chunk_size: int | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Memory-bounded distillation loss for every divergence selector, from hidden states.

    Objective dispatch and metrics match the dense ``distillation_loss`` exactly:

    * ``loss_top_k > 0``: teacher-top-k KL (tail-bucket form with ``loss_add_tail`` when ``k < V``, otherwise
      both distributions renormalized over the support). The support is the teacher's top-k at temperature T,
      found with a running top-k across vocabulary chunks.
    * ``cakld_gamma``: ``gamma * KL(q || p) + (1 - gamma) * KL(p || q)``.
    * ``beta``: ``None``/``<= 0`` forward KL ``KL(p || q)``, ``>= 1`` reverse KL ``KL(q || p)``, otherwise
      generalized JSD ``beta * KL(p || m) + (1 - beta) * KL(q || m)`` with ``m = beta * p + (1 - beta) * q``.

    Here ``p`` is the teacher and ``q`` the student distribution, both at ``logits / temperature``. The result
    is the masked token mean times ``temperature**2``, scaled by ``alpha``.

    Args:
        student_hidden: ``[B, L, H]`` student last hidden states.
        teacher_hidden: ``[B, L, H_t]`` teacher last hidden states (gradient is stopped).
        student_projection: Student LM-head projection.
        teacher_projection: Teacher LM-head projection with the same vocabulary layout.
        attention_mask: Optional ``[B, L]`` padding mask.
        loss_mask: Optional ``[B, L]`` token mask; takes precedence over ``attention_mask``.
        temperature: Softmax temperature.
        alpha: Weight applied to the divergence.
        beta: Generalized JSD coefficient (see above).
        cakld_gamma: Confidence-aware KL coefficient; mutually exclusive with ``beta``.
        loss_top_k: Teacher top-k support size; ``0`` uses the full vocabulary.
        loss_add_tail: Add one bucket for all non-top-k mass.
        shift_tokens: Train position ``t`` against token ``t + 1``.
        sequence_chunk_size: Optional number of sequence positions per outer scan step.

    Returns:
        ``(total_loss, metrics)`` with ``kl_loss``, ``distill_xent_loss``, ``teacher_entropy_loss`` and
        ``ce_loss`` (always zero; there are no hard labels).

    Raises:
        ValueError: On incompatible selector combinations.
    """
    if cakld_gamma is not None:
        if not 0.0 <= float(cakld_gamma) <= 1.0:
            raise ValueError("`cakld_gamma` must be within [0, 1] when set.")
        if beta is not None:
            raise ValueError("`cakld_gamma` is mutually exclusive with GJSD `beta`.")
        if loss_top_k > 0 or loss_add_tail:
            raise ValueError("`cakld_gamma` is not supported with `loss_top_k` / `loss_add_tail`.")
    h_s, h_t, mask, _ = _shift_and_mask(student_hidden, teacher_hidden, attention_mask, loss_mask, None, shift_tokens)

    top_k = int(loss_top_k) if loss_top_k > 0 else 0
    direction = None
    if top_k:
        divergences: tuple[str, ...] = ()
    elif cakld_gamma is not None:
        divergences = ("forward", "reverse")
    else:
        direction = resolve_kl_direction(beta)
        if direction == "forward":
            divergences = ("forward", "entropy") if beta is None else ("forward",)
        else:
            divergences = (direction,)
    beta_value = 0.5 if beta is None else float(beta)

    def _fn(hs: Array, ht: Array, tg: Array | None) -> dict[str, Array]:
        return _token_statistics(
            hs,
            ht,
            tg,
            student_projection,
            teacher_projection,
            temperature=temperature,
            top_k=top_k,
            divergences=divergences,
            beta=beta_value,
        )

    stats = _map_positions(_fn, h_s, h_t, None, sequence_chunk_size)
    zero_tokens = jnp.zeros(mask.shape, jnp.float32)
    if top_k:
        vocab = student_projection.layout(h_s.reshape(-1, h_s.shape[-1]))[0]
        xent, entropy = _topk_xent_entropy(
            stats["topk_teacher_logp"],
            stats["topk_student_logp"],
            add_tail=bool(loss_add_tail) and min(top_k, vocab) < vocab,
        )
        kl = xent - entropy
    elif cakld_gamma is not None:
        gamma = float(cakld_gamma)
        kl = gamma * stats["reverse"] + (1.0 - gamma) * stats["forward"]
        xent, entropy = kl, zero_tokens
    else:
        kl = stats[direction]
        entropy = stats["entropy"] if "entropy" in stats else zero_tokens
        xent = kl + entropy

    temp_sq = jnp.float32(float(temperature) ** 2)
    normalizer = jnp.maximum(jnp.sum(mask), jnp.float32(1.0))
    kl_loss = _masked_mean(kl, mask, normalizer) * temp_sq
    metrics = {
        "kl_loss": kl_loss,
        "distill_xent_loss": _masked_mean(xent, mask, normalizer) * temp_sq,
        "teacher_entropy_loss": _masked_mean(entropy, mask, normalizer) * temp_sq,
        "ce_loss": jnp.zeros((), jnp.float32),
    }
    return jnp.float32(alpha) * kl_loss, metrics


def trust_region_distillation_loss(
    student_hidden: Array,
    teacher_hidden: Array,
    student_projection: Projection,
    teacher_projection: Projection,
    input_ids: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    *,
    temperature: float = 1.0,
    top_k: int = 64,
    rng: Array | None = None,
    acceptance_mask: Array | None = None,
    acceptance_uniform: Array | None = None,
    shift_tokens: bool = True,
    sequence_chunk_size: int | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Trust-region distillation with speculative-decoding-style acceptance of sampled tokens.

    Each sampled (completion) token is treated like a drafted token under speculative decoding: the student
    proposed it and the teacher "verifies" it. With sampled-token log-probs ``s_lp`` (student) and ``t_lp``
    (teacher), the acceptance probability is ``P = min(exp(sg(t_lp - s_lp)), 1)`` and ``M = [U < P]`` with
    ``U ~ Uniform(0, 1)``.

    * accepted tokens (``M = 1``) contribute the reverse-KL policy-gradient term ``sg(s_lp - t_lp) * s_lp``,
      whose gradient is ``(s_lp - t_lp) * grad(s_lp)``;
    * rejected tokens (``M = 0``) contribute the teacher-top-k forward KL
      ``sum_{v in topk(p)} p_v (log p_v - log q_v)`` with ``p``/``q`` normalized over the full
      vocabulary, pulling the student toward the teacher where it strayed outside the teacher's trust region.

    The loss is ``sum(mask * per_token) / max(sum(mask), 1)``. All log-probs use the full-vocabulary
    ``softmax(logits / temperature)``, where ``temperature`` must be the temperature the tokens were *sampled* at.
    ``P`` is the target/proposal ratio of the sampling distributions only when the student itself sampled the
    tokens from that untruncated, unpenalised distribution: no top-k / top-p / min-p truncation, no presence,
    frequency or repetition penalty, and no teacher-generated completions
    (:class:`OnPolicyDistillationConfig` refuses those settings in trust-region mode). Neither branch carries a
    ``T**2`` factor: both are per-token divergence estimates of the same pair of distributions, so they share one scale (the ``T**2``
    Hinton correction only compensates for softening by a separate distillation temperature, which this mode
    does not use).

    Args:
        student_hidden: ``[B, L, H]`` student last hidden states.
        teacher_hidden: ``[B, L, H_t]`` teacher last hidden states.
        student_projection: Student LM-head projection.
        teacher_projection: Teacher LM-head projection.
        input_ids: ``[B, L]`` sampled sequences; with ``shift_tokens`` position ``t`` scores token ``t + 1``.
        attention_mask: Optional ``[B, L]`` padding mask.
        loss_mask: Optional ``[B, L]`` completion mask; takes precedence over ``attention_mask``.
        temperature: Sampling temperature of the scored tokens (softmax temperature of every log-prob).
        top_k: Teacher support size of the rejected-token forward KL.
        rng: PRNG key for the acceptance draw (used when neither ``acceptance_mask`` nor
            ``acceptance_uniform`` is given).
        acceptance_mask: Optional fixed ``M`` aligned with the shifted positions (``[B, L - 1]`` when
            ``shift_tokens``); overrides the random draw.
        acceptance_uniform: Optional precomputed ``U`` with the same layout as ``acceptance_mask``; the mask
            is ``U < P``. Callers that split a batch into microbatches draw ``U`` once for the full batch and
            slice it, so every row gets an independent draw regardless of the accumulation split.
        shift_tokens: Train position ``t`` against token ``t + 1``.
        sequence_chunk_size: Optional number of sequence positions per outer scan step.

    Returns:
        ``(loss, metrics)`` with ``kl_loss`` (sampled reverse-KL estimate), ``acceptance_rate``,
        ``accepted_pg_loss`` and ``rejected_topk_kl``.

    Raises:
        ValueError: If no acceptance source (``rng``, ``acceptance_mask``, ``acceptance_uniform``) is given,
            or ``top_k < 1``.
    """
    if int(top_k) < 1:
        raise ValueError("`top_k` must be >= 1 for trust-region distillation.")
    h_s, h_t, mask, targets = _shift_and_mask(
        student_hidden, teacher_hidden, attention_mask, loss_mask, input_ids, shift_tokens
    )
    targets = targets.astype(jnp.int32)

    def _fn(hs: Array, ht: Array, tg: Array | None) -> dict[str, Array]:
        return _token_statistics(
            hs,
            ht,
            tg,
            student_projection,
            teacher_projection,
            temperature=temperature,
            top_k=int(top_k),
            divergences=(),
            beta=0.5,
        )

    stats = _map_positions(_fn, h_s, h_t, targets, sequence_chunk_size)
    s_lp = stats["target_student_logp"]
    t_lp = jax.lax.stop_gradient(stats["target_teacher_logp"])
    log_ratio = jax.lax.stop_gradient(t_lp - s_lp)
    accept_prob = jnp.exp(jnp.minimum(log_ratio, 0.0))
    if acceptance_mask is not None:
        accepted = jax.lax.stop_gradient(acceptance_mask.astype(jnp.float32))
    elif acceptance_uniform is not None:
        if acceptance_uniform.shape != accept_prob.shape:
            raise ValueError(
                f"`acceptance_uniform` shape {acceptance_uniform.shape} must match the scored positions "
                f"{accept_prob.shape}."
            )
        uniform = jax.lax.stop_gradient(acceptance_uniform.astype(jnp.float32))
        accepted = (uniform < accept_prob).astype(jnp.float32)
    elif rng is not None:
        accepted = (jax.random.uniform(rng, accept_prob.shape, jnp.float32) < accept_prob).astype(jnp.float32)
    else:
        raise ValueError("trust_region_distillation_loss needs `rng`, `acceptance_mask` or `acceptance_uniform`.")

    pg = jax.lax.stop_gradient(s_lp - t_lp) * s_lp
    lt = stats["topk_teacher_logp"]
    topk_kl = jnp.sum(jnp.exp(lt) * (lt - stats["topk_student_logp"]), axis=-1)
    per_token = accepted * pg + (1.0 - accepted) * topk_kl
    normalizer = jnp.maximum(jnp.sum(mask), jnp.float32(1.0))
    loss = _masked_mean(per_token, mask, normalizer)
    metrics = {
        "kl_loss": _masked_mean(jax.lax.stop_gradient(s_lp - t_lp), mask, normalizer),
        "acceptance_rate": _masked_mean(accepted, mask, normalizer),
        "accepted_pg_loss": _masked_mean(accepted * pg, mask, normalizer),
        "rejected_topk_kl": _masked_mean((1.0 - accepted) * topk_kl, mask, normalizer),
    }
    return loss, metrics
