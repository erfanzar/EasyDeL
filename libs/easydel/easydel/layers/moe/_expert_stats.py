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

"""Expert-load recording for any fused-MoE model.

Expert-parallel shards own a contiguous block of the expert axis
(``roll_to_expert_id = experts_per_shard * expert_shard_id`` in
``_moe_module.py``), so a shard's grouped-matmul work is the token count of its
block. Nothing in the runtime measured that, which made it impossible to say
whether routing skew was costing anything — or what an expert-placement change
would be worth.

This records the per-expert token histogram that the fused path *already*
computes (``group_size`` in :func:`~._communication_utils.permute`), so
recording adds no arithmetic to the model. It is family-agnostic: every MoE
model routes through that one function, so the recorder sees deepseek-v2/v3/v4,
qwen3-moe, mixtral, gpt-oss, glm4-moe and anything else registered, with no
per-model code.

Two deliberate design points:

* **Activation is a ContextVar scope, not an environment variable.** Outside
  :func:`record_expert_load` the hook is a single ``is None`` check at trace
  time and emits nothing into the graph.
* **Transport is a debug callback, which makes this a CALIBRATION tool, not a
  production counter.** ``jax.debug.callback`` serializes the step. For
  always-on accounting the right vehicle is an ``spx.Buffer`` accumulator
  (see :class:`easydel.infra.utils.ModuleCaches` for the pattern), which rides
  the existing graphstate and so does not disturb a compiled runner's pinned
  ``in_shardings``/``out_shardings``.

Example:
    >>> from easydel.layers.moe import record_expert_load
    >>> with record_expert_load() as rec:
    ...     model(input_ids=ids)
    >>> rec.summary()["per_layer_max_over_mean"]
"""

from __future__ import annotations

import contextvars
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from eformer.loggings import get_logger

logger = get_logger(__name__)

__all__ = [
    "ExpertLoadRecord",
    "ExpertLoadRecorder",
    "active_expert_load_recorder",
    "balancedness",
    "optimal_shard_loads",
    "record_expert_load",
    "shard_active_experts",
    "shard_loads",
]


@dataclass(frozen=True)
class ExpertLoadRecord:
    """One router histogram: tokens routed to each expert in one call.

    Attributes:
        counts: Per-expert token counts, global expert order. Shape ``[E]``.
        layer_idx: Layer that produced it, or ``None`` when the caller did not
            supply one (then :meth:`ExpertLoadRecorder.stack` falls back to
            call order).
        regime: ``"decode"`` or ``"prefill"``, from the ambient
            :func:`~easydel.infra.sharding.decode_mode_specs` scope.
        step: Monotonic index of the forward pass within the recording scope.
    """

    counts: np.ndarray
    layer_idx: int | None
    regime: str
    step: int


@dataclass
class ExpertLoadRecorder:
    """Collects :class:`ExpertLoadRecord`s and reduces them to placement stats.

    Attributes:
        ep_size: Expert-parallel shard count the statistics are reduced over.
            Placement quality is only meaningful relative to a shard count.
        max_records: Ring capacity. Long calibration runs would otherwise grow
            without bound; the newest records win.
        records: Collected histograms.
    """

    ep_size: int = 1
    max_records: int = 200_000
    records: list[ExpertLoadRecord] = field(default_factory=list)
    _step: int = 0
    _dropped: int = 0

    def add(self, counts: np.ndarray, layer_idx: int | None, regime: str) -> None:
        """Append one histogram, dropping the oldest when at capacity."""
        if len(self.records) >= self.max_records:
            self.records.pop(0)
            self._dropped += 1
        self.records.append(ExpertLoadRecord(np.asarray(counts), layer_idx, regime, self._step))

    def mark_step(self) -> None:
        """Advance the forward-pass counter (optional; for per-step stats)."""
        self._step += 1

    def stack(self, regime: str | None = None) -> dict[int, np.ndarray]:
        """Sum histograms per layer.

        Records are duplicated across EP shards (every shard traces the same
        ``permute`` over replicated tokens), and identical duplicates would
        inflate every count by ``ep_size`` uniformly — harmless for ratios, but
        it makes absolute counts misleading, so they are averaged out.

        Args:
            regime: Keep only records from this regime, or ``None`` for all.

        Returns:
            ``{layer_idx: counts}`` with counts summed over steps.
        """
        chosen = [r for r in self.records if regime is None or r.regime == regime]
        if not chosen:
            return {}
        n_experts = int(chosen[0].counts.shape[0])
        labelled = all(r.layer_idx is not None for r in chosen)
        per_layer: dict[int, np.ndarray] = {}
        dupes: dict[int, int] = {}
        for i, rec in enumerate(chosen):
            key = int(rec.layer_idx) if labelled else i
            acc = per_layer.get(key)
            per_layer[key] = rec.counts.astype(np.int64) if acc is None else acc + rec.counts
            dupes[key] = dupes.get(key, 0) + 1
        if labelled:
            steps = max(1, len({r.step for r in chosen}))
            for key in per_layer:
                reps = max(1, dupes[key] // max(1, steps))
                if reps > 1:
                    per_layer[key] = per_layer[key] // reps
        del n_experts
        return per_layer

    def summary(self, regime: str | None = None) -> dict[str, tp.Any]:
        """Placement statistics for the recorded traffic.

        Returns:
            Dict with per-layer and aggregate shard imbalance
            (``max/mean``), balancedness, the imbalance an optimal placement
            would reach, and the implied reduction in MoE time.
        """
        per_layer = self.stack(regime)
        if not per_layer:
            return {"n_layers": 0, "regime": regime}
        cur, opt, act = [], [], []
        for counts in per_layer.values():
            if counts.sum() <= 0:
                continue
            loads = shard_loads(counts, self.ep_size)
            best = optimal_shard_loads(counts, self.ep_size)
            tiles = shard_active_experts(counts, self.ep_size)
            cur.append(float(loads.max() / loads.mean()))
            opt.append(float(best.max() / best.mean()))
            act.append(tiles)
        if not cur:
            return {"n_layers": 0, "regime": regime}
        cur_a, opt_a = np.asarray(cur), np.asarray(opt)
        act_a = np.asarray(act)  # [layers, ep]
        total = np.sum(np.stack(list(per_layer.values())), axis=0)
        agg = shard_loads(total, self.ep_size)
        return {
            "regime": regime,
            "n_layers": len(cur),
            "n_experts": int(total.shape[0]),
            "ep_size": self.ep_size,
            "per_layer_max_over_mean": float(cur_a.mean()),
            "per_layer_max_over_mean_worst": float(cur_a.max()),
            "per_layer_balanced_max_over_mean": float(opt_a.mean()),
            "aggregate_max_over_mean": float(agg.max() / agg.mean()),
            "balancedness": float(np.mean([balancedness(c, self.ep_size) for c in per_layer.values()])),
            "moe_time_reduction_pct_if_token_bound": float((1.0 - (opt_a / cur_a).mean()) * 100.0),
            "active_experts_per_shard_mean": float(act_a.mean()),
            "active_experts_per_shard_max_over_mean": float(
                np.mean(act_a.max(axis=1) / np.maximum(act_a.mean(axis=1), 1e-9))
            ),
            "dropped_records": self._dropped,
        }

    def to_arrays(self, regime: str | None = None) -> dict[str, np.ndarray]:
        """Export in EPLB interchange format.

        ``logical_count`` is ``[steps, layers, experts]`` and
        ``physical_to_logical_map`` is ``[layers, physical_experts]``. Keeping
        the step axis matters: aggregate counts understate per-step decode
        skew, and any placement solver worth running consumes the steps.
        """
        chosen = [r for r in self.records if regime is None or r.regime == regime]
        if not chosen:
            return {}
        steps = sorted({r.step for r in chosen})
        layers = sorted({int(r.layer_idx) for r in chosen if r.layer_idx is not None}) or [0]
        n_e = int(chosen[0].counts.shape[0])
        out = np.zeros((len(steps), len(layers), n_e), dtype=np.int64)
        s_ix = {s: i for i, s in enumerate(steps)}
        l_ix = {v: i for i, v in enumerate(layers)}
        for rec in chosen:
            li = l_ix.get(int(rec.layer_idx)) if rec.layer_idx is not None else 0
            if li is None:
                continue
            out[s_ix[rec.step], li] += rec.counts.astype(np.int64)
        phys = np.tile(np.arange(n_e, dtype=np.int64), (len(layers), 1))
        return {"logical_count": out, "physical_to_logical_map": phys}

    def save(self, path: str, regime: str | None = None) -> str:
        """Write :meth:`to_arrays` to ``path`` as ``.npz`` (GCS-aware)."""
        import io

        from eformer.paths import ePath

        arrays = self.to_arrays(regime)
        if not arrays:
            raise ValueError("no records to save")
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        target = ePath(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(buf.getvalue())
        return str(target)


def shard_loads(counts: np.ndarray, ep_size: int) -> np.ndarray:
    """Token load per EP shard under the runtime's contiguous-block placement.

    Shard ``d`` owns experts ``[d*E/ep, (d+1)*E/ep)`` — see
    ``build_ep_traffic_matrix`` and ``local_permute``, both of which derive
    ownership by integer division on the expert index.
    """
    counts = np.asarray(counts, dtype=np.float64)
    ep_size = max(1, int(ep_size))
    per = counts.shape[0] // ep_size
    if per == 0:
        return counts.sum(keepdims=True)
    return counts[: per * ep_size].reshape(ep_size, per).sum(axis=1)


def shard_active_experts(counts: np.ndarray, ep_size: int) -> np.ndarray:
    """Non-empty experts per EP shard — the quantity that sets grouped-matmul cost.

    Measured on v5p (int4 expert path, DeepSeek-V4 shapes): grouped-matmul time
    is flat in a shard's TOKEN count (0.85x-2.0x load at fixed buffer height all
    within 0.1%) and near-linear in the number of non-empty experts (4/8/16/32/48
    experts -> 0.039/0.041/0.062/0.116/0.170 ms). Each expert's group is far
    smaller than one m-tile, so every touched expert costs a full tile whether it
    holds 1 row or 100.

    Consequence: balancing *tokens* across shards changes nothing. Report this
    alongside :func:`shard_loads` so the distinction stays visible.
    """
    counts = np.asarray(counts)
    ep_size = max(1, int(ep_size))
    per = counts.shape[0] // ep_size
    if per == 0:
        return np.asarray([(counts > 0).sum()], dtype=np.float64)
    return (counts[: per * ep_size].reshape(ep_size, per) > 0).sum(axis=1).astype(np.float64)


def balancedness(counts: np.ndarray, ep_size: int) -> float:
    """``mean/max`` over shard loads: 1.0 is perfect, lower is worse.

    Same definition reports, so numbers are comparable against theirs.
    """
    loads = shard_loads(counts, ep_size)
    return float((loads.mean() + 1e-5) / (loads.max() + 1e-5))


def optimal_shard_loads(counts: np.ndarray, ep_size: int) -> np.ndarray:
    """Shard loads under the best equal-cardinality placement (greedy LPT).

    Every shard must keep exactly ``E/ep`` experts or the shard shapes stop
    matching, so this is number partitioning with a cardinality constraint,
    not free bin packing. Greedy longest-processing-time lands within a
    fraction of a percent of optimal at these sizes and is the same heuristic
    ``balanced_packing`` uses.
    """
    counts = np.asarray(counts, dtype=np.float64)
    ep_size = max(1, int(ep_size))
    per = counts.shape[0] // ep_size
    if per == 0:
        return counts.sum(keepdims=True)
    loads = np.zeros(ep_size)
    held = np.zeros(ep_size, dtype=int)
    for e in np.argsort(-counts[: per * ep_size]):
        d = min((d for d in range(ep_size) if held[d] < per), key=lambda d: loads[d])
        loads[d] += counts[e]
        held[d] += 1
    return loads


_recorder_var: contextvars.ContextVar[ExpertLoadRecorder | None] = contextvars.ContextVar(
    "easydel_expert_load_recorder", default=None
)


def active_expert_load_recorder() -> ExpertLoadRecorder | None:
    """The recorder for the enclosing :func:`record_expert_load` scope, if any."""
    return _recorder_var.get()


@contextmanager
def record_expert_load(ep_size: int = 1, max_records: int = 200_000) -> tp.Iterator[ExpertLoadRecorder]:
    """Scope that captures every fused-MoE router histogram.

    Calibration only — the histograms leave the device through
    ``jax.debug.callback``, which serializes the step. Do not wrap a
    throughput measurement in this.

    Args:
        ep_size: Expert-parallel shard count to reduce statistics over. Read
            the model's mesh (``config.mesh.shape["ep"]``) to pass the value
            the runtime actually uses.
        max_records: Ring capacity before the oldest records are dropped.

    Yields:
        The :class:`ExpertLoadRecorder` collecting for this scope.
    """
    recorder = ExpertLoadRecorder(ep_size=int(ep_size), max_records=int(max_records))
    token = _recorder_var.set(recorder)
    try:
        yield recorder
    finally:
        _recorder_var.reset(token)


def maybe_record_group_sizes(group_size, num_experts: int, roll_to_expert_id=None, layer_idx: int | None = None) -> None:
    """Record ``group_size`` when a recording scope is active; else do nothing.

    Called from :func:`~._communication_utils.permute`. When no scope is open
    this returns before touching the traced value, so nothing is added to the
    graph.

    Args:
        group_size: Traced per-expert token counts, shape ``[num_experts]``.
        num_experts: Expert count, used to validate the histogram width.
        roll_to_expert_id: Rotation applied to expert ids before counting
            (ring-of-experts). Undone here so records are always in GLOBAL
            expert order regardless of dispatch path.
        layer_idx: Owning layer, when the caller knows it.
    """
    recorder = _recorder_var.get()
    if recorder is None:
        return
    import jax

    from easydel.infra.sharding import inference_mode_forces_decode

    regime = "decode" if inference_mode_forces_decode() else "prefill"
    roll = int(roll_to_expert_id) if isinstance(roll_to_expert_id, int) else 0

    def _sink(counts):
        counts = np.asarray(counts)
        if counts.ndim != 1 or counts.shape[0] != num_experts:
            return
        if roll:
            counts = np.roll(counts, roll)
        recorder.add(counts, layer_idx, regime)

    jax.debug.callback(_sink, group_size)
