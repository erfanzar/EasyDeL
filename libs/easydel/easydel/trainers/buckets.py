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

"""Training buckets.

A *training bucket* is a (model-config variant, dataloader) pair selected per
optimizer step. The motivating use case is distillation where every Nth step
uses a short-sequence / dense-attention config while the rest use a
long-sequence / sparse-attention config::

    buckets = [
        TrainingBucket(name="vanilla_8k",   max_length=8192,   config={"attn_mechanism": "vanilla"}),
        TrainingBucket(name="splash_131k",  max_length=131072, config={"attn_mechanism": "blocksparse"}),
    ]
    rule = ModBucketRule(mod=5, on_bucket=0, off_bucket=1)

Because the attention kernel is resolved once at module construction and baked
into the model ``graphdef`` (it is ``pytree_node=False``), a single compiled
step function cannot switch kernels at runtime. Instead the trainer builds one
``graphdef`` per bucket (all sharing an identical parameter-tree structure) and
swaps ``state.graphdef`` per step before invoking the bucket's own compiled
function. The shared ``graphstate`` / ``opt_state`` buffers are updated in
place by whichever function runs.
"""

from __future__ import annotations

import abc
import bisect
import copy
import typing as tp
from dataclasses import dataclass

if tp.TYPE_CHECKING:
    from jax.sharding import PartitionSpec

    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.loss_utils import LossConfig


class BucketRule(abc.ABC):
    """Maps an optimizer step to a bucket index in ``[0, len(buckets))``.

    Subclasses implement :meth:`select`. Concrete rules are serializable via
    :meth:`to_dict` / :meth:`from_dict` so they can live in an eLarge config.
    """

    @abc.abstractmethod
    def select(self, step: int) -> int:
        """Return the bucket index for ``step``."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize to a discriminator-tagged dict (``{"kind": ...}``)."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, tp.Any]) -> "BucketRule":
        """Reconstruct a rule from its serialized form.

        Raises:
                ValueError: if the ``"kind"`` discriminator is unknown.
        """
        kind = d.get("kind")
        if kind == "mod":
            return ModBucketRule(
                mod=int(d["mod"]),
                offset=int(d.get("offset", 0)),
                on_bucket=int(d.get("on_bucket", 0)),
                off_bucket=int(d.get("off_bucket", 1)),
            )
        if kind == "step":
            return StepThresholdRule(thresholds=[int(t) for t in d["thresholds"]])
        if kind == "cycle":
            return CycleBucketRule(
                period=int(d["period"]),
                num_buckets=int(d.get("num_buckets", 2)),
            )
        if kind == "piecewise":
            return PiecewiseBucketRule(
                boundaries=[int(b) for b in d["boundaries"]],
                rules=[r if isinstance(r, BucketRule) else cls.from_dict(r) for r in d["rules"]],
            )
        if kind == "pattern":
            return PatternBucketRule(
                pattern=[int(b) for b in d["pattern"]],
                offset=int(d.get("offset", 0)),
            )
        raise ValueError(f"Unknown BucketRule kind: {kind!r}")


@dataclass
class ModBucketRule(BucketRule):
    """Two-bucket modulo rule.

    ``select(step) == on_bucket`` iff ``(step - offset) % mod == 0``, else
    ``off_bucket``. The distillation case (vanilla every 5th step) is::

        ModBucketRule(mod=5, on_bucket=0, off_bucket=1)

    Attributes:
            mod: Modulus.
            offset: Step offset; ``offset=0`` means step 0 falls in ``on_bucket``.
            on_bucket: Bucket index returned on the modulus match.
            off_bucket: Bucket index returned otherwise.
    """

    mod: int
    offset: int = 0
    on_bucket: int = 0
    off_bucket: int = 1

    def __post_init__(self) -> None:
        if self.mod <= 0:
            raise ValueError(f"ModBucketRule.mod must be > 0, got {self.mod}.")

    def select(self, step: int) -> int:
        return self.on_bucket if ((step - self.offset) % self.mod) == 0 else self.off_bucket

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "kind": "mod",
            "mod": self.mod,
            "offset": self.offset,
            "on_bucket": self.on_bucket,
            "off_bucket": self.off_bucket,
        }


@dataclass
class StepThresholdRule(BucketRule):
    """N-bucket curriculum by sorted step thresholds.

    ``thresholds=[100, 500]`` -> bucket 0 for ``step < 100``, 1 for
    ``100 <= step < 500``, 2 for ``step >= 500``.

    Attributes:
            thresholds: Ascending step cutoffs. ``len(thresholds) + 1`` buckets.
    """

    thresholds: list[int]

    def __post_init__(self) -> None:
        self.thresholds = sorted(self.thresholds)
        if any(b <= a for a, b in zip(self.thresholds[:-1], self.thresholds[1:], strict=False)):
            raise ValueError(f"StepThresholdRule.thresholds must be strictly increasing, got {self.thresholds}.")

    def select(self, step: int) -> int:
        idx = 0
        for t in self.thresholds:
            if step >= t:
                idx += 1
            else:
                break
        return idx

    def to_dict(self) -> dict[str, tp.Any]:
        return {"kind": "step", "thresholds": list(self.thresholds)}


@dataclass
class CycleBucketRule(BucketRule):
    """Round-robin blocks of ``period`` steps across ``num_buckets`` buckets.

    ``select(step) == (step // period) % num_buckets``: steps 0..period-1 run
    bucket 0, the next ``period`` steps bucket 1, and so on, wrapping around.
    The alternating case (5 steps of bucket 0, then 5 of bucket 1, repeat) is::

        CycleBucketRule(period=5, num_buckets=2)

    Attributes:
            period: Consecutive steps spent in each bucket before advancing.
            num_buckets: Number of buckets cycled through.
    """

    period: int
    num_buckets: int = 2

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"CycleBucketRule.period must be > 0, got {self.period}.")
        if self.num_buckets <= 0:
            raise ValueError(f"CycleBucketRule.num_buckets must be > 0, got {self.num_buckets}.")

    def select(self, step: int) -> int:
        return (step // self.period) % self.num_buckets

    def to_dict(self) -> dict[str, tp.Any]:
        return {"kind": "cycle", "period": self.period, "num_buckets": self.num_buckets}


@dataclass
class PiecewiseBucketRule(BucketRule):
    """Compose serializable rules over disjoint step ranges.

    ``boundaries=[b0]`` with ``rules=[r0, r1]`` selects via ``r0`` for
    ``step < b0`` and via ``r1`` for ``step >= b0`` (same boundary semantics
    as :class:`StepThresholdRule`). Sub-rules receive the GLOBAL step, not a
    segment-local one, so a ``ModBucketRule`` segment keeps its absolute
    offset anchoring. The motivating case is a cadence change late in a run
    (e.g. a long-context bucket every 11th step for most of training, then
    every 4th step during a final cooldown)::

        PiecewiseBucketRule(
            boundaries=[17_900],
            rules=[
                ModBucketRule(mod=11, offset=10, on_bucket=1, off_bucket=0),
                ModBucketRule(mod=4, offset=3, on_bucket=1, off_bucket=0),
            ],
        )

    Attributes:
            boundaries: Strictly increasing step cutoffs; segment ``i`` covers
                    ``boundaries[i-1] <= step < boundaries[i]``.
            rules: One rule per segment; ``len(rules) == len(boundaries) + 1``.
    """

    boundaries: list[int]
    rules: list[BucketRule]

    def __post_init__(self) -> None:
        self.boundaries = [int(b) for b in self.boundaries]
        if any(b <= a for a, b in zip(self.boundaries[:-1], self.boundaries[1:], strict=False)):
            raise ValueError(f"PiecewiseBucketRule.boundaries must be strictly increasing, got {self.boundaries}.")
        if len(self.rules) != len(self.boundaries) + 1:
            raise ValueError(
                f"PiecewiseBucketRule needs len(boundaries) + 1 rules, got {len(self.rules)} rule(s) "
                f"for {len(self.boundaries)} boundary(ies)."
            )
        for rule in self.rules:
            if not isinstance(rule, BucketRule):
                raise TypeError(f"PiecewiseBucketRule.rules entries must be BucketRule, got {type(rule).__name__}.")

    def select(self, step: int) -> int:
        return self.rules[bisect.bisect_right(self.boundaries, step)].select(step)

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "kind": "piecewise",
            "boundaries": list(self.boundaries),
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass
class PatternBucketRule(BucketRule):
    """Repeating explicit bucket pattern — arbitrary N-way ratios.

    ``select(step) == pattern[(step - offset) % len(pattern)]``. This is the
    general repeating-ratio rule: :class:`ModBucketRule` covers only two
    buckets, :class:`CycleBucketRule` only equal-width blocks, and
    :class:`PiecewiseBucketRule` / :class:`StepThresholdRule` partition absolute
    step *ranges* rather than repeating phases. A three-bucket 2:2:1 cadence
    (two medium-context steps, two short steps, then one long-context step) is::

        PatternBucketRule(pattern=[0, 0, 1, 1, 2])

    Unlike :class:`CallableBucketRule` this is serializable, so it round-trips
    through an eLarge config and through ``TrainingArguments.to_dict`` (which
    every checkpoint save writes).

    Attributes:
            pattern: Bucket index per phase, one entry per step of the cycle.
                    Repeats forever; ``len(pattern)`` is the cadence modulus.
            offset: Step offset; ``offset=0`` means ``pattern[0]`` applies at
                    step 0 (and at every step divisible by ``len(pattern)``).
    """

    pattern: list[int]
    offset: int = 0

    def __post_init__(self) -> None:
        self.pattern = [int(b) for b in self.pattern]
        if not self.pattern:
            raise ValueError("PatternBucketRule.pattern must be non-empty.")
        if any(b < 0 for b in self.pattern):
            raise ValueError(f"PatternBucketRule.pattern entries must be >= 0, got {self.pattern}.")
        self.offset = int(self.offset)

    def select(self, step: int) -> int:
        return self.pattern[(step - self.offset) % len(self.pattern)]

    def to_dict(self) -> dict[str, tp.Any]:
        return {"kind": "pattern", "pattern": list(self.pattern), "offset": self.offset}


@dataclass
class CallableBucketRule(BucketRule):
    """Escape hatch wrapping any ``(int) -> int`` callable.

    Not serializable for eLarge configs; :meth:`to_dict` raises ``TypeError``.
    Because ``TrainingArguments.to_dict`` is called on every checkpoint save,
    a callable rule also breaks saving mid-run — prefer
    :class:`PatternBucketRule` for any fixed repeating cadence.
    """

    fn: tp.Callable[[int], int]

    def select(self, step: int) -> int:
        return int(self.fn(step))

    def to_dict(self) -> dict[str, tp.Any]:
        raise TypeError("CallableBucketRule is not serializable; use ModBucketRule or StepThresholdRule for eLarge.")


@dataclass
class TrainingBucket:
    """One per-step training configuration variant.

    The model config is *not* an attention field; it is a general override of
    the base model config. ``attn_mechanism`` is just one of the knobs you can
    put in ``config``.

    Attributes:
            name: Human-readable identifier (used in logs/metrics).
            max_length: Packing/collation target for this bucket's data.
            config: Model-config variant.

                    * ``None`` -> inherit the trainer's base model config unchanged.
                    * ``dict`` -> deep-copy the base config and ``setattr`` each key.
                    * ``EasyDeLBaseConfig`` -> used as-is (caller must keep it
                      structure-compatible with the base config).
            gradient_accumulation_steps: Per-bucket grad-accum; ``None`` inherits
                    from ``TrainingArguments``.
            loss_config: Per-bucket loss config; ``None`` inherits.
            step_partition_spec: Per-bucket batch sharding spec; ``None`` inherits.
            total_batch_size: Per-bucket global batch size for this bucket's
                    dataloader and compiled step; ``None`` inherits the trainer's
                    ``training_batch_size``.
            data_collator: Per-bucket batch collator. ``None`` uses the
                    trainer-built default (stack + bucket max_length checks).
                    When set, it REPLACES the default and receives the raw list
                    of rows the bucket dataloader produced — the same contract
                    as the trainer-level ``data_collator`` (e.g. a packed-embeds
                    VLM collator for one bucket while another bucket streams
                    plain packed text).
    """

    name: str
    max_length: int
    config: "EasyDeLBaseConfig | dict[str, tp.Any] | None" = None
    gradient_accumulation_steps: int | None = None
    loss_config: "LossConfig | None" = None
    step_partition_spec: PartitionSpec | None = None
    total_batch_size: int | None = None
    data_collator: tp.Callable | None = None
    # dataset is attached separately on the trainer (one source per bucket),
    # not on this dataclass, to keep it free of dataset/dataloader imports.


def apply_config_overrides(
    base_config: "EasyDeLBaseConfig",
    overrides: dict[str, tp.Any],
    *,
    label: str,
) -> "EasyDeLBaseConfig":
    """Return a deep copy of ``base_config`` with ``overrides`` applied via ``setattr``.

    Shared by per-bucket config resolution and the trainer-level
    ``eval_config_overrides`` path so both validate keys identically.

    Args:
            base_config: The config to copy and override; never mutated.
            overrides: Field-name -> value overrides.
            label: Human-readable origin used in error messages
                    (e.g. a bucket name or ``"eval_config_overrides"``).

    Returns:
            The overridden deep copy.

    Raises:
            AttributeError: if an override key is not a field of the base config.
    """
    cfg = copy.deepcopy(base_config)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"{label}: base config has no field {key!r} to override.")
        setattr(cfg, key, value)
    return cfg


def resolve_bucket_config(
    base_config: "EasyDeLBaseConfig",
    bucket: TrainingBucket,
) -> "EasyDeLBaseConfig":
    """Resolve a bucket's model config against a base config.

    Args:
            base_config: The trainer's base model config.
            bucket: The bucket whose ``config`` field is resolved.

    Returns:
            The resolved :class:`EasyDeLBaseConfig`. When ``bucket.config`` is a
            dict, a deep copy of ``base_config`` is returned with each key applied
            via ``setattr``; the base config is never mutated.

    Raises:
            AttributeError: if a dict override key is not a field of the base
                    config.
    """
    if bucket.config is None:
        return base_config
    if isinstance(bucket.config, dict):
        return apply_config_overrides(base_config, bucket.config, label=f"Bucket {bucket.name!r}")
    # Already an EasyDeLBaseConfig (or compatible): use as-is.
    return bucket.config
