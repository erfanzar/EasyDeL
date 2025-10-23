# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Dataset pipeline builder.

This module provides the main pipeline function for building unified datasets
from multiple sources with various processing and mixing strategies.
"""

from __future__ import annotations

import typing as tp

from .mixture import block_mixture_interleave
from .pack import pack_constant_length, pack_pre_tokenized
from .sources import load_for_inform
from .types import DatasetMixture, TextDatasetInform
from .utils import align_columns_intersection, is_streaming, wrap_format_callback

if tp.TYPE_CHECKING:
    from datasets import Dataset as DS


def build_dataset(mixture: DatasetMixture) -> DS:
    """Build a unified dataset from a DatasetMixture configuration.

    This is the main entry point for creating datasets. It handles loading
    multiple data sources, applying transformations, mixing datasets with
    various strategies, and optionally packing sequences for efficient training.

    The pipeline supports:
    - Loading from HuggingFace Hub and local files
    - Field renaming and custom format callbacks
    - Multiple mixing strategies (standard interleave or block-deterministic)
    - Optional token packing (pre-tokenized or on-the-fly)
    - Streaming and non-streaming modes

    Args:
        mixture: DatasetMixture configuration object containing all settings
            for dataset loading, processing, and mixing.

    Returns:
        A Dataset or IterableDataset ready for training, with all transformations
        and mixing strategies applied.

    Example:
        >>> from easydel.utils.data_managers import DatasetMixture, TextDatasetInform
        >>>
        >>> # Simple single dataset
        >>> mixture = DatasetMixture(
        ...     informs=[TextDatasetInform(type="json", data_files="data.json")],
        ...     batch_size=32
        ... )
        >>> dataset = build_dataset(mixture)
        >>>
        >>> # Complex multi-dataset mixture with packing
        >>> mixture = DatasetMixture(
        ...     informs=[
        ...         TextDatasetInform(type="parquet", data_files="dataset1/*.parquet"),
        ...         TextDatasetInform(type="json", data_files="dataset2.json"),
        ...     ],
        ...     block_mixture=True,
        ...     mixture_weights={"dataset1": 0.7, "dataset2": 0.3},
        ...     pack_tokens=True,
        ...     pack_seq_length=2048,
        ... )
        >>> dataset = build_dataset(mixture)
    """
    per_ds = []
    content_target = mixture.text_target_field

    for inform in mixture.informs:
        ds = load_for_inform(inform, mixture)

        if getattr(inform, "format_fields", None):
            mapping_local = dict(inform.format_fields)

            def rename_fields(ex, _mapping=mapping_local):
                for old_name, new_name in _mapping.items():
                    if old_name in ex:
                        ex[new_name] = ex.pop(old_name)
                for k in list(ex.keys()):
                    v = ex[k]
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        ex[k] = [{(_mapping.get(kk) or kk): vv for kk, vv in d.items()} for d in v]
                return ex

            ds = ds.map(rename_fields, batched=False)

        if getattr(inform, "format_callback", None):
            fmt = wrap_format_callback(inform.format_callback, getattr(inform, "content_field", "content"))

            ex0 = next(iter(ds.take(1))) if is_streaming(ds) else ds[0]
            after = fmt(dict(ex0))
            cols_to_remove = list(set(ex0.keys()) - set(after.keys()))
            ds = ds.map(fmt, batched=False, remove_columns=cols_to_remove or None)

        if isinstance(inform, TextDatasetInform):
            keep = {content_target}
            addl = getattr(inform, "additional_fields", None) or []
            keep.update(addl)

            content_field = inform.content_field
            addl_fields = tuple(addl or ())

            def to_target(ex, _content_field=content_field, _addl=addl_fields, _target=content_target):
                if _content_field is None:
                    return ex
                try:
                    out = {_target: ex[_content_field]}
                except KeyError as e:
                    raise KeyError(f"Missing content field '{_content_field}'. Available keys: {list(ex.keys())}") from e
                for f in _addl:
                    if f in ex:
                        out[f] = ex[f]
                return out

            ds = ds.map(to_target, batched=False)
            try:
                ds = ds.select_columns(list(keep))
            except Exception:
                pass

        per_ds.append(ds)

    if mixture.streaming:
        if getattr(mixture, "block_mixture", False):
            weights = None
            if mixture.mixture_weights and len(mixture.mixture_weights) == len(per_ds):
                weights = mixture.mixture_weights
            mixed = block_mixture_interleave(
                per_ds,
                weights=weights,
                block_size=getattr(mixture, "mixture_block_size", 2048),
                seed=mixture.seed or 0,
                stop=getattr(mixture, "stop_strategy", "restart"),
            )
        else:
            from datasets import interleave_datasets

            mixed = interleave_datasets(per_ds, seed=mixture.seed, stopping_strategy="first_exhausted")
            if mixture.shuffle_buffer_size:
                mixed = mixed.shuffle(buffer_size=mixture.shuffle_buffer_size, seed=mixture.seed)
    else:
        per_ds = align_columns_intersection(per_ds)
        from datasets import concatenate_datasets

        mixed = concatenate_datasets(per_ds)
        if mixture.shuffle_buffer_size:
            mixed = mixed.shuffle(seed=mixture.seed)

    if getattr(mixture, "pack_tokens", False):
        from datasets import IterableDataset

        gen = pack_pre_tokenized(
            iter(mixed),
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if getattr(mixture, "pack_on_the_fly", False):
        if mixture.tokenize_callback is None:
            raise ValueError("pack_on_the_fly=True requires mixture.tokenize_callback")
        from datasets import IterableDataset

        gen = pack_constant_length(
            iter(mixed),
            tokenize_fn=mixture.tokenize_callback,
            seq_length=mixture.pack_seq_length or 1024,
            eos_token_id=mixture.pack_eos_token_id,
            batch_size=mixture.batch_size,
            shuffle=mixture.pack_shuffle,
            buffer_factor=mixture.pack_shuffle_buffer_factor,
        )
        return IterableDataset.from_generator(gen)

    if mixture.batch_size and mixture.batch_size > 1 and is_streaming(mixed):
        mixed = mixed.batch(mixture.batch_size)

    return mixed
