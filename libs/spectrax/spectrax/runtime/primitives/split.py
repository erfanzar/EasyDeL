# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto-split a single :class:`spectrax.Module` into pipeline stages.

The user writes one Module — typically ``embed`` + ``blocks: ModuleList``
+ ``head`` — and :func:`auto_split` slices it into ``n_pp`` per-rank
stages without any explicit pipeline annotations.

**Default behavior** (no annotations):

* Stage 0 gets everything declared before ``blocks`` in ``__init__``
  (``embed``, etc.).
* Stage ``n_pp - 1`` gets everything declared after ``blocks``
  (``head``, ``norm_f``, etc.).
* ``blocks`` is evenly sliced across all stages.

**Manual stage assignment** via ``pp_stage``:

Any :class:`Module` child, including individual blocks inside the repeated
``ModuleList``, can carry a ``pp_stage`` attribute to override automatic
placement::

    model.embed.pp_stage = 0        # explicit: rank 0
    model.blocks[0].pp_stage = 0    # explicit block placement
    model.head.pp_stage = "last"    # explicit: last rank
    model.aux_head.pp_stage = 2     # put on rank 2

Supported values:

* ``int`` — stage index (0-based). ``-1`` means last stage.
* ``"first"`` — alias for 0.
* ``"last"`` — alias for ``n_pp - 1``.

When ``pp_stage`` is set, that module is placed on the requested
stage regardless of whether it appears before or after ``blocks``
in ``__init__`` order. Modules without ``pp_stage`` fall back to
the default pre/post auto-detection.
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from inspect import Parameter

from ...core.containers import ModuleList
from ...core.module import Module

__all__ = ["auto_split", "split_block_stack"]


def _resolve_stage(pp_stage: int | str, n_pp: int) -> int:
    """Resolve a ``pp_stage`` annotation to a concrete stage index.

    Accepts the strings ``"first"`` / ``"last"`` and any integer
    (negative indices count from the end, mirroring Python list
    semantics). The resulting index is bounds-checked against
    ``n_pp``.

    Args:
        pp_stage: User-supplied annotation. Either an ``int`` or one
            of the string aliases ``"first"`` / ``"last"``.
        n_pp: Total number of pipeline stages, used to interpret
            negative indices and the ``"last"`` alias.

    Returns:
        A non-negative integer in ``[0, n_pp)``.

    Raises:
        ValueError: If ``pp_stage`` is an unrecognised string or an
            integer that resolves outside ``[0, n_pp)``.
    """
    if isinstance(pp_stage, str):
        if pp_stage == "first":
            return 0
        if pp_stage == "last":
            return n_pp - 1
        raise ValueError(f"pp_stage must be an int, 'first', or 'last'; got {pp_stage!r}.")
    idx = int(pp_stage)
    if idx < 0:
        idx = n_pp + idx
    if not 0 <= idx < n_pp:
        raise ValueError(f"pp_stage={pp_stage} resolves to index {idx}, out of range for n_pp={n_pp}.")
    return idx


def _default_block_stage(index: int, n_blocks: int, n_pp: int) -> int:
    """Map block ``index`` to a balanced contiguous default pipeline rank.

    Used when a block carries no explicit ``pp_stage``. The mapping
    keeps blocks contiguous within a stage (block ``i`` and ``i+1``
    land on the same rank when possible) so cross-rank transport only
    happens at stage boundaries, not inside a stage.

    Args:
        index: Position of the block in the original ``ModuleList``.
        n_blocks: Total number of blocks.
        n_pp: Number of pipeline stages.

    Returns:
        The rank index this block defaults to.
    """
    return min(n_pp - 1, (index * n_pp) // n_blocks)


def _block_stage_indices(blocks: list[Module], n_pp: int) -> list[int]:
    """Resolve per-block stage ownership for an entire block list.

    Walks ``blocks`` in order, asking :func:`_resolve_stage` for any
    block carrying an explicit ``pp_stage`` annotation and falling
    back to :func:`_default_block_stage` otherwise. The final
    sequence must be non-decreasing — block ``i+1`` cannot land on a
    rank earlier than block ``i`` because that would let an
    activation flow backwards through the pipeline mid-forward.

    Args:
        blocks: The original ``ModuleList`` contents.
        n_pp: Number of pipeline stages.

    Returns:
        A list of length ``len(blocks)`` whose entries are stage
        indices in ``[0, n_pp)``.

    Raises:
        ValueError: If the resolved sequence is not monotonic, i.e.
            a user annotation puts a later block on an earlier stage.
    """
    n_blocks = len(blocks)
    stages: list[int] = []
    for i, block in enumerate(blocks):
        pp_stage = getattr(block, "pp_stage", None)
        stages.append(
            _resolve_stage(pp_stage, n_pp) if pp_stage is not None else _default_block_stage(i, n_blocks, n_pp)
        )
    for i, (prev, cur) in enumerate(itertools.pairwise(stages), start=1):
        if cur < prev:
            raise ValueError(
                "Block-level pp_stage annotations must be non-decreasing in block order; "
                f"block {i - 1} maps to stage {prev}, but block {i} maps to stage {cur}."
            )
    return stages


def _forward_positional_params(module: Module) -> list[inspect.Parameter]:
    """Return the meaningful positional params from ``module.forward``.

    Used to detect blocks whose ``forward`` accepts more than one
    positional argument so we can emit a ``block_carry`` warning.
    Wrappers such as :func:`spx.remat` often expose
    ``forward(*args, **kwargs)``; those generic varargs would falsely
    trigger the warning, so they are filtered out by restricting to
    ``POSITIONAL_ONLY`` and ``POSITIONAL_OR_KEYWORD`` kinds.

    Args:
        module: A :class:`Module` whose ``forward`` is to be
            inspected.

    Returns:
        The subset of ``inspect.signature(module.forward).parameters``
        that represent real positional arguments (not ``self`` and
        not ``*args``).
    """
    sig = inspect.signature(module.forward)
    return [
        p
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]


class _StageWrapper(Module):
    """One pipeline stage's worth of submodules, chained ``pre -> blocks -> post``.

    Constructed by :func:`split_block_stack` to repackage a slice of
    a user's model into a single :class:`Module` that the runtime can
    treat as one pipeline stage. The wrapper preserves enough metadata
    (``pre_names``, ``post_names``, ``carry_indices``) for
    :meth:`forward` to know which positional arguments to feed each
    inner block.

    Supports both single-arg and multi-arg block signatures. When a
    block returns a tuple, the tuple is unpacked as positional args
    for the next block. When it returns a single value, it is wrapped
    in a 1-tuple. This lets blocks like::

        def forward(self, hidden, mask, pos_ids, kv_cache=None):
            ...
            return hidden, mask, pos_ids, updated_kv_cache

    chain naturally through the pipeline — all args flow through, only
    the ones the block modifies change between stages.
    """

    def __init__(
        self,
        pre: list[tuple[str, Module]],
        blocks: list[Module],
        post: list[tuple[str, Module]],
        carry_indices: tuple[int, ...] = (0,),
    ):
        """Materialise pre/blocks/post submodules as attributes of this stage.

        The pre/post modules are reattached under their original
        attribute names so the final module's pytree structure mirrors
        the user's hand-written model. The repeated blocks are wrapped
        in a fresh :class:`ModuleList` named ``blocks`` regardless of
        the source name.

        Args:
            pre: ``(name, module)`` pairs to install before the block
                run (e.g. embedding, norm).
            blocks: The slice of original ``ModuleList`` blocks
                assigned to this stage.
            post: ``(name, module)`` pairs to install after the block
                run (e.g. final norm, lm_head).
            carry_indices: Indices into the positional-args tuple that
                identify which arguments are *carried* by each block
                (i.e. updated and forwarded). Other arguments are
                broadcast to every block unchanged. See
                :meth:`forward` for the semantics.
        """
        super().__init__()
        for name, m in pre:
            setattr(self, name, m)
        self.blocks = ModuleList(blocks)
        for name, m in post:
            setattr(self, name, m)
        self.pre_names = tuple(name for name, _ in pre)
        self.post_names = tuple(name for name, _ in post)
        self.carry_indices = carry_indices

    def forward(self, *args, **kwargs):
        """Run pre -> blocks -> post, threading carry + broadcast args.

        Args are split into **carry** (change each block) and
        **broadcast** (same for all blocks) by ``carry_indices``.
        Blocks receive ``(*carry, *broadcast)`` and return updated
        carry values. Pre/post modules receive only the first carry
        arg (the canonical "hidden state" position).

        Example: ``carry_indices=(0, 3)`` with args
        ``(hidden, mask, pos_ids, kv_cache)`` means carry is
        ``(hidden, kv_cache)`` and broadcast is ``(mask, pos_ids)``.
        Block called as ``blk(hidden, kv_cache, mask, pos_ids)``
        returns ``(new_hidden, new_kv_cache)`` which replaces carry
        for the next block.

        Args:
            *args: Positional arguments to the stage. The first arg is
                always treated as the primary carry; other carries are
                determined by :attr:`carry_indices`.
            **kwargs: Keyword arguments. Forwarded only to the *first*
                pre-module call; subsequent calls drop them so blocks
                don't receive accidental keyword leakage.

        Returns:
            Either the single primary output (when the stage produces
            one positional value) or the full updated argument tuple
            (when the stage produces multiple). Matches what the
            downstream stage will receive as its ``*args``.
        """
        ci = self.carry_indices
        n_args = len(args)
        max_idx = max(ci) + 1 if ci else 1
        if n_args < max_idx:
            args = args + (None,) * (max_idx - n_args)

        for name in self.pre_names:
            first = getattr(self, name)(args[0], **kwargs)
            args = (first, *args[1:])
            kwargs = {}
        for blk in self.blocks:
            result = blk(*args)
            returned = result if isinstance(result, tuple) else (result,)
            args_list = list(args)
            for carry_pos, ret_val in zip(ci, returned, strict=False):
                args_list[carry_pos] = ret_val
            args = tuple(args_list)
        for name in self.post_names:
            first = getattr(self, name)(args[0])
            args = (first, *args[1:])

        return args if len(args) > 1 else args[0]


def split_block_stack(
    model: Module,
    n_pp: int,
    *,
    blocks_attr: str = "blocks",
    pre_attrs: list[str] | None = None,
    post_attrs: list[str] | None = None,
) -> list[Module]:
    """Split ``model`` into ``n_pp`` :class:`_StageWrapper` stages.

    The split has three pieces: pre-block modules (e.g. ``embed``),
    the contents of the ``blocks`` :class:`ModuleList`, and post-block
    modules (e.g. ``norm_f`` / ``head``). Each block is assigned to a
    stage by :func:`_block_stage_indices`; pre/post modules respect
    explicit ``pp_stage`` annotations and otherwise default to stage
    0 / stage ``n_pp - 1`` respectively.

    Block signatures with multiple positional arguments (e.g. a
    ``forward(hidden, mask, pos_ids)``) are handled via the
    ``block_carry`` class attribute on the block: it lists which
    positional indices are *carried* (mutated and forwarded) vs
    broadcast unchanged across blocks. Without ``block_carry`` and
    with multi-arg blocks the split emits a :class:`UserWarning`
    suggesting a setting.

    Respects ``pp_stage`` annotations on :class:`Module` children:
    children with ``pp_stage`` set are placed on the specified stage;
    children without it fall back to auto-detection (before ``blocks``
    -> stage 0, after ``blocks`` -> stage ``n_pp - 1``).

    Args:
        model: The full model with a ``blocks: ModuleList``.
        n_pp: Number of pipeline stages. Should divide ``len(blocks)``
            for balanced splits, but uneven splits are also accepted.
        blocks_attr: Name of the :class:`ModuleList` attribute on
            ``model``. Defaults to ``"blocks"``.
        pre_attrs: Override the auto-detected list of pre-block
            children. ``None`` means auto-detect via
            :func:`_auto_pre_post`.
        post_attrs: Override the auto-detected list of post-block
            children. ``None`` means auto-detect.

    Returns:
        A list of ``n_pp`` :class:`_StageWrapper` instances, ready to
        be passed to a pipeline runtime.

    Raises:
        ValueError: If ``model`` lacks ``blocks_attr``, the attribute
            is empty, or block-level ``pp_stage`` annotations are not
            non-decreasing.
        TypeError: If ``model.<blocks_attr>`` is not sized (no
            ``__len__``).
    """
    blocks = getattr(model, blocks_attr, None)
    if blocks is None:
        raise ValueError(
            f"model has no attribute {blocks_attr!r}; supply blocks_attr or define model.pipeline_split(n_pp)."
        )
    if not hasattr(blocks, "__len__"):
        raise TypeError(f"model.{blocks_attr} must be a ModuleList (or sized); got {type(blocks).__name__}.")
    n_blocks = len(blocks)
    if n_blocks == 0:
        raise ValueError(f"model.{blocks_attr} must contain at least one block.")

    if pre_attrs is None or post_attrs is None:
        pre_auto, post_auto = _auto_pre_post(model, blocks_attr)
        if pre_attrs is None:
            pre_attrs = pre_auto
        if post_attrs is None:
            post_attrs = post_auto

    per_stage_extras: list[list[tuple[str, Module, str]]] = [[] for _ in range(n_pp)]

    for name in (*pre_attrs, *post_attrs):
        child = getattr(model, name)
        pp_stage = getattr(child, "pp_stage", None)
        if pp_stage is not None:
            stage_idx = _resolve_stage(pp_stage, n_pp)
            position = "pre" if name in pre_attrs else "post"
            per_stage_extras[stage_idx].append((name, child, position))
        else:
            if name in pre_attrs:
                per_stage_extras[0].append((name, child, "pre"))
            else:
                per_stage_extras[n_pp - 1].append((name, child, "post"))

    first_block = blocks[0]
    carry_indices = getattr(first_block, "block_carry", None)
    if carry_indices is None:
        params = _forward_positional_params(first_block)
        if len(params) > 1:
            param_names = [p.name for p in params]
            warnings.warn(
                f"{type(first_block).__name__}.forward takes {len(params)} "
                f"args ({', '.join(param_names)}) but has no `block_carry` "
                f"attribute. The pipeline will only pass the first arg "
                f"between blocks and discard the rest. Set "
                f"`block_carry = (0,)` for single-arg chains, or "
                f"`block_carry = (0, {len(params) - 1})` to carry the "
                f"first and last args through the pipeline. Example:\n"
                f"  class {type(first_block).__name__}(spx.Module):\n"
                f"      block_carry = (0, {len(params) - 1})",
                UserWarning,
                stacklevel=3,
            )
        carry_indices = (0,)
    if isinstance(carry_indices, int):
        carry_indices = (carry_indices,)
    carry_indices = tuple(carry_indices)

    block_list = list(blocks)
    block_stages = _block_stage_indices(block_list, n_pp)
    stages: list[Module] = []
    for r in range(n_pp):
        slab = [block for block, stage_idx in zip(block_list, block_stages, strict=True) if stage_idx == r]
        pre_pairs = [(name, mod) for name, mod, pos in per_stage_extras[r] if pos == "pre"]
        post_pairs = [(name, mod) for name, mod, pos in per_stage_extras[r] if pos == "post"]
        stages.append(_StageWrapper(pre=pre_pairs, blocks=slab, post=post_pairs, carry_indices=carry_indices))
    return stages


def auto_split(model: Module, n_pp: int) -> list[Module]:
    """Slice ``model`` into ``n_pp`` per-stage modules for pipeline parallelism.

    The preferred user override is to define a
    ``pipeline_split(n_pp)`` method on the model class — that method
    can implement model-specific splitting (multi-tower architectures,
    cross-attention bridges, etc.) without touching the runtime. When
    no such method exists we fall back to :func:`split_block_stack`,
    which auto-detects the ``blocks: ModuleList`` + surrounding
    pre/post modules and respects any ``pp_stage`` annotations the
    user has attached.

    Args:
        model: The full single-device model.
        n_pp: Number of pipeline stages.

    Returns:
        A list of ``n_pp`` :class:`Module` instances suitable for
        wrapping with :class:`~spectrax.nn.PipelineSequential` or
        feeding directly into a pipeline runtime as
        :class:`PipelineStage` callables.

    Raises:
        ValueError: If ``model.pipeline_split`` returns the wrong
            shape, or if the fallback :func:`split_block_stack` cannot
            find a valid ``blocks`` attribute.
    """
    if hasattr(model, "pipeline_split"):
        out = model.pipeline_split(n_pp)
        if not isinstance(out, (list, tuple)) or len(out) != n_pp:
            raise ValueError(
                f"{type(model).__name__}.pipeline_split(n_pp={n_pp}) must "
                f"return a sequence of {n_pp} Modules; got {type(out).__name__} "
                f"of length {len(out) if hasattr(out, '__len__') else '?'}."
            )
        return list(out)
    return split_block_stack(model, n_pp)


def _auto_pre_post(model: Module, blocks_attr: str) -> tuple[list[str], list[str]]:
    """Classify ``model``'s :class:`Module` children by position relative to ``blocks``.

    Walks ``model.__dict__`` in insertion order — the order in which
    ``__init__`` set the attributes — and returns two lists of
    attribute names: one for modules declared before ``blocks_attr``,
    one for modules declared after. The ``blocks_attr`` itself is
    skipped.

    The insertion-order traversal mirrors the user's intent: a module
    declared earlier in ``__init__`` is assumed to run earlier in the
    forward pass, so it belongs on an earlier pipeline stage.

    Args:
        model: The model whose children to classify.
        blocks_attr: Name of the ``ModuleList`` attribute that
            partitions the model into pre / post halves.

    Returns:
        ``(pre_attrs, post_attrs)`` — names of :class:`Module`
        children before and after ``blocks_attr`` respectively.
    """
    pre: list[str] = []
    post: list[str] = []
    seen_blocks = False
    for name, val in vars(model).items():
        if name == blocks_attr:
            seen_blocks = True
            continue
        if isinstance(val, Module):
            (post if seen_blocks else pre).append(name)
    return pre, post
